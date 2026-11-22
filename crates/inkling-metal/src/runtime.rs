//! Thin wrapper over objc2-metal: one device, one queue, pipelines from the build-time metallib,
//! shared-storage buffers, and an encoder helper that keeps all of a token's work in one command
//! buffer.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString, NSURL};
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePassDescriptor, MTLComputePipelineState,
    MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor, MTLCounterSet,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType, MTLLibrary, MTLResourceOptions,
    MTLSize, MTLStorageMode,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no Metal device")]
    NoDevice,
    #[error("metal: {0}")]
    Metal(String),
    #[error("kernel {0} not found in the metallib")]
    MissingKernel(String),
    #[error(transparent)]
    Model(#[from] inkling::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

pub type Device = ProtocolObject<dyn MTLDevice>;
pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// A shared-storage (CPU-visible, unified memory) buffer.
pub struct Buffer {
    pub raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub len: usize,
}

impl Buffer {
    pub fn as_slice<T: Copy>(&self) -> &[T] {
        // SAFETY: shared storage is CPU-visible; callers only read after the GPU work that writes
        // this buffer has completed (Context::run waits).
        unsafe {
            std::slice::from_raw_parts(
                self.raw.contents().as_ptr() as *const T,
                self.len / size_of::<T>(),
            )
        }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn as_mut_slice<T: Copy>(&self) -> &mut [T] {
        // SAFETY: as above; writes happen only while no command buffer using it is in flight.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.raw.contents().as_ptr() as *mut T,
                self.len / size_of::<T>(),
            )
        }
    }

    pub fn write<T: Copy>(&self, data: &[T]) {
        self.as_mut_slice::<T>()[..data.len()].copy_from_slice(data);
    }
}

pub struct Context {
    pub device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: Mutex<HashMap<String, Pipeline>>,
}

impl Context {
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| Error::Metal("newCommandQueue".into()))?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(env!("INKLING_METALLIB")));
        let library = device
            .newLibraryWithURL_error(&url)
            .map_err(|e| Error::Metal(format!("load metallib: {e}")))?;
        Ok(Context {
            device,
            queue,
            library,
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    pub fn pipeline(&self, name: &str) -> Result<Pipeline> {
        let mut cache = self.pipelines.lock().expect("pipeline cache poisoned");
        if let Some(p) = cache.get(name) {
            return Ok(p.clone());
        }
        let f = self
            .library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| Error::MissingKernel(name.into()))?;
        let p = self
            .device
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| Error::Metal(format!("pipeline {name}: {e}")))?;
        cache.insert(name.to_string(), p.clone());
        Ok(p)
    }

    pub fn buffer(&self, len: usize) -> Result<Buffer> {
        let raw = self
            .device
            .newBufferWithLength_options(len.max(16), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| Error::Metal(format!("alloc {len} bytes")))?;
        Ok(Buffer { raw, len })
    }

    pub fn buffer_from<T: Copy>(&self, data: &[T]) -> Result<Buffer> {
        let b = self.buffer(std::mem::size_of_val(data))?;
        b.write(data);
        Ok(b)
    }

    /// Wraps existing page-aligned memory (the model mmap) without copying.
    ///
    /// # Safety
    /// `ptr` must be page-aligned and stay mapped and unmodified for the buffer's lifetime.
    pub unsafe fn buffer_no_copy(&self, ptr: *const u8, len: usize) -> Result<Buffer> {
        let page = 16384;
        assert_eq!(
            ptr as usize % page,
            0,
            "no-copy buffers need page-aligned memory"
        );
        let rounded = len.div_ceil(page) * page;
        let p =
            NonNull::new(ptr as *mut c_void).ok_or_else(|| Error::Metal("null pointer".into()))?;
        // SAFETY: caller guarantees the mapping outlives the buffer; no deallocator, the mmap owns it.
        let raw = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    p,
                    rounded,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or_else(|| Error::Metal(format!("no-copy buffer of {rounded} bytes")))?;
        Ok(Buffer { raw, len })
    }

    /// Encodes `f` into one compute encoder in one command buffer, commits it, and waits. The
    /// encoder is concurrent, but every dispatch is followed by a buffer barrier unless it sits
    /// inside `Encoder::parallel`, so ordering is serial by default.
    pub fn run(&self, f: impl FnOnce(&mut Encoder) -> Result<()>) -> Result<RunStats> {
        self.submit(f)?.wait()
    }

    /// Like `run`, but returns once the command buffer is committed. Command buffers on the queue
    /// execute in submission order, so later submissions see this one's writes.
    pub fn submit(&self, f: impl FnOnce(&mut Encoder) -> Result<()>) -> Result<Pending> {
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::Metal("commandBuffer".into()))?;
        let enc = cb
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or_else(|| Error::Metal("computeCommandEncoder".into()))?;
        let mut e = Encoder {
            ctx: self,
            cb: cb.clone(),
            enc: Some(enc),
            profile: None,
            n_dispatch: 0,
            in_parallel: false,
        };
        let t_enc = std::time::Instant::now();
        let res = f(&mut e);
        if let Some(enc) = &e.enc {
            enc.endEncoding();
        }
        let encode_s = t_enc.elapsed().as_secs_f64();
        let n_dispatch = e.n_dispatch;
        res?;
        cb.commit();
        Ok(Pending {
            cb,
            encode_s,
            n_dispatch,
        })
    }
}

/// A committed command buffer.
pub struct Pending {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encode_s: f64,
    n_dispatch: usize,
}

impl Pending {
    pub fn wait(self) -> Result<RunStats> {
        self.cb.waitUntilCompleted();
        if self.cb.status() == MTLCommandBufferStatus::Error {
            return Err(Error::Metal(format!(
                "command buffer failed: {:?}",
                self.cb.error()
            )));
        }
        Ok(RunStats {
            gpu_s: self.cb.GPUEndTime() - self.cb.GPUStartTime(),
            encode_s: self.encode_s,
            n_dispatch: self.n_dispatch,
        })
    }
}

impl Context {
    /// Runs `f` with every dispatch in its own encoder bracketed by GPU timestamps, and returns
    /// per-dispatch (kernel, seconds). Separate encoders add overhead, so use it to rank kernels,
    /// not for absolute numbers.
    pub fn profile(
        &self,
        max_dispatch: usize,
        f: impl FnOnce(&mut Encoder) -> Result<()>,
    ) -> Result<Vec<(String, f64)>> {
        let sets = self
            .device
            .counterSets()
            .ok_or_else(|| Error::Metal("no counter sets".into()))?;
        let ts = (0..sets.count())
            .map(|i| sets.objectAtIndex(i))
            .find(|s| s.name().to_string() == "timestamp")
            .ok_or_else(|| Error::Metal("no timestamp counter set".into()))?;
        let desc = MTLCounterSampleBufferDescriptor::new();
        desc.setCounterSet(Some(&ts));
        desc.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: a positive sample count within the device limit.
        unsafe { desc.setSampleCount(SAMPLES_PER_BUF) };
        let samples = (0..(2 * max_dispatch).div_ceil(SAMPLES_PER_BUF))
            .map(|_| {
                self.device
                    .newCounterSampleBufferWithDescriptor_error(&desc)
                    .map_err(|e| Error::Metal(format!("counter buffer: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;

        let (mut c0, mut g0, mut c1, mut g1) = (0u64, 0u64, 0u64, 0u64);
        // SAFETY: plain out-pointers.
        unsafe {
            self.device
                .sampleTimestamps_gpuTimestamp(NonNull::from(&mut c0), NonNull::from(&mut g0))
        };
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::Metal("commandBuffer".into()))?;
        let mut e = Encoder {
            ctx: self,
            cb: cb.clone(),
            enc: None,
            profile: Some(Profiler {
                samples,
                names: Vec::new(),
            }),
            n_dispatch: 0,
            in_parallel: false,
        };
        f(&mut e)?;
        cb.commit();
        cb.waitUntilCompleted();
        unsafe {
            self.device
                .sampleTimestamps_gpuTimestamp(NonNull::from(&mut c1), NonNull::from(&mut g1))
        };

        // GPU ticks -> seconds, calibrated against the CPU clock (mach ticks -> ns)
        let mut tb = MachTimebase { numer: 0, denom: 0 };
        // SAFETY: fills the struct.
        unsafe { mach_timebase_info(&mut tb) };
        let cpu_ns = (c1 - c0) as f64 * tb.numer as f64 / tb.denom as f64;
        let s_per_tick = cpu_ns / (g1 - g0).max(1) as f64 * 1e-9;

        let prof = e.profile.take().expect("profiling encoder");
        let n = prof.names.len();
        let mut ticks: Vec<u64> = Vec::with_capacity(2 * n);
        for (bi, sb) in prof.samples.iter().enumerate() {
            let len = (2 * n)
                .saturating_sub(bi * SAMPLES_PER_BUF)
                .min(SAMPLES_PER_BUF);
            if len == 0 {
                break;
            }
            // SAFETY: range within the samples written above.
            let data = unsafe { sb.resolveCounterRange(NSRange::new(0, len)) }
                .ok_or_else(|| Error::Metal("resolve counters".into()))?;
            ticks.extend(
                data.to_vec()
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| u64::from_le_bytes(*c)),
            );
        }
        Ok(prof
            .names
            .into_iter()
            .enumerate()
            .map(|(i, k)| {
                (
                    k,
                    ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * s_per_tick,
                )
            })
            .collect())
    }
}

#[repr(C)]
struct MachTimebase {
    numer: u32,
    denom: u32,
}

unsafe extern "C" {
    fn mach_timebase_info(info: *mut MachTimebase) -> i32;
}

fn double_prefix() -> Option<&'static str> {
    static P: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    P.get_or_init(|| std::env::var("INKLING_DOUBLE").ok())
        .as_deref()
}

/// Timing for one command buffer.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunStats {
    /// GPU execution time.
    pub gpu_s: f64,
    /// CPU time spent encoding.
    pub encode_s: f64,
    pub n_dispatch: usize,
}

/// A kernel argument: a buffer (with byte offset) or a small by-value struct.
pub enum Arg<'a> {
    Buf(&'a Buffer, usize),
    Bytes(&'a [u8]),
}

pub fn buf(b: &Buffer) -> Arg<'_> {
    Arg::Buf(b, 0)
}

pub fn buf_at(b: &Buffer, offset: usize) -> Arg<'_> {
    Arg::Buf(b, offset)
}

/// Passes a `#[repr(C)]` value as a constant argument.
pub fn bytes<T: Copy>(v: &T) -> Arg<'_> {
    // SAFETY: reading the bytes of a Copy value for the lifetime of the borrow.
    Arg::Bytes(unsafe { std::slice::from_raw_parts(v as *const T as *const u8, size_of::<T>()) })
}

pub struct Encoder<'c> {
    ctx: &'c Context,
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// The shared serial encoder; `None` in profiling mode, where each dispatch gets its own.
    enc: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    profile: Option<Profiler>,
    pub n_dispatch: usize,
    in_parallel: bool,
}

/// Timestamp sample buffers hold at most 4096 samples (32 KB), so long profiles chain several.
const SAMPLES_PER_BUF: usize = 4096;

struct Profiler {
    samples: Vec<Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>>,
    names: Vec<String>,
}

impl Encoder<'_> {
    /// Dispatches `kernel` over `groups` threadgroups of `threads` threads. A serial encoder orders
    /// dispatches, so each one sees the previous one's writes.
    pub fn dispatch(
        &mut self,
        kernel: &str,
        args: &[Arg],
        groups: (usize, usize, usize),
        threads: (usize, usize, usize),
    ) -> Result<()> {
        let p = self.ctx.pipeline(kernel)?;
        // a kernel's register use can lower its thread limit below what the caller dispatches;
        // Metal would then not run it correctly, without any error
        let shared = p.staticThreadgroupMemoryLength();
        let shared_max = self.ctx.device.maxThreadgroupMemoryLength();
        if shared > shared_max {
            return Err(Error::Metal(format!(
                "{kernel}: {shared} bytes of threadgroup memory, the device allows {shared_max}"
            )));
        }
        let max = p.maxTotalThreadsPerThreadgroup();
        if threads.0 * threads.1 * threads.2 > max {
            return Err(Error::Metal(format!(
                "{kernel}: {} threads per threadgroup, the compiled pipeline allows {max}",
                threads.0 * threads.1 * threads.2
            )));
        }
        let own;
        let enc = match (&self.enc, &mut self.profile) {
            (Some(e), _) => e,
            (None, Some(prof)) => {
                let i = prof.names.len();
                let (bi, si) = (2 * i / SAMPLES_PER_BUF, 2 * i % SAMPLES_PER_BUF);
                let Some(sb) = prof.samples.get(bi) else {
                    return Err(Error::Metal("profiler sample buffers full".into()));
                };
                let desc = MTLComputePassDescriptor::computePassDescriptor();
                desc.setDispatchType(MTLDispatchType::Serial);
                // SAFETY: index 0 always exists in the attachments array; sample indices are in range.
                unsafe {
                    let att = desc.sampleBufferAttachments().objectAtIndexedSubscript(0);
                    att.setSampleBuffer(Some(sb));
                    att.setStartOfEncoderSampleIndex(si);
                    att.setEndOfEncoderSampleIndex(si + 1);
                }
                prof.names.push(kernel.to_string());
                own = self
                    .cb
                    .computeCommandEncoderWithDescriptor(&desc)
                    .ok_or_else(|| Error::Metal("profiling encoder".into()))?;
                &own
            }
            (None, None) => unreachable!("encoder without a mode"),
        };
        enc.setComputePipelineState(&p);
        for (i, a) in args.iter().enumerate() {
            match a {
                // SAFETY: buffers are kept alive by the caller until `run` returns.
                Arg::Buf(b, off) => unsafe { enc.setBuffer_offset_atIndex(Some(&b.raw), *off, i) },
                Arg::Bytes(s) => unsafe {
                    enc.setBytes_length_atIndex(
                        NonNull::new(s.as_ptr() as *mut c_void).expect("non-null"),
                        s.len(),
                        i,
                    )
                },
            }
        }
        let g = MTLSize {
            width: groups.0,
            height: groups.1,
            depth: groups.2,
        };
        let t = MTLSize {
            width: threads.0,
            height: threads.1,
            depth: threads.2,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(g, t);
        // INKLING_DOUBLE=<kernel-name prefix>: run matching kernels twice, so a timing difference
        // shows their true cost inside a real batch (results are garbage; measurement only)
        if let Some(prefix) = double_prefix()
            && kernel.starts_with(prefix)
        {
            enc.dispatchThreadgroups_threadsPerThreadgroup(g, t);
        }
        if self.enc.is_none() {
            enc.endEncoding();
        } else if !self.in_parallel {
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        self.n_dispatch += 1;
        Ok(())
    }

    /// Encodes `f` with no barriers between its dispatches, then one barrier after them. Only for
    /// dispatches that neither read nor write anything another one in the group writes.
    pub fn parallel(&mut self, f: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        assert!(!self.in_parallel, "nested parallel groups");
        self.in_parallel = true;
        let r = f(self);
        self.in_parallel = false;
        if let Some(enc) = &self.enc {
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        r
    }
}
