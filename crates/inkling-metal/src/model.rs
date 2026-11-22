//! Metal decode path: one token per command buffer. Mirrors `inkling::cpu` op for op; the
//! fixture-free test in tests/vs_cpu.rs diffs every traced tensor against it.

use gguf::{DType, Gguf};
use inkling::{Hparams, Trace};

use crate::runtime::{Buffer, Context, Encoder, Error, Result, RunStats, buf, buf_at, bytes};

#[repr(C)]
#[derive(Clone, Copy)]
struct MvArgs {
    rows: u32,
    cols: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct MoeArgs {
    n_embd: u32,
    n_ff: u32,
    n_used: u32,
    situ_beta: f32,
    situ_lin: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct TopkArgs {
    n_expert: u32,
    k: u32,
    scale: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct NormArgs {
    n: u32,
    eps: f32,
    has_weight: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct ClipArgs {
    dim: u32,
    threshold: f32,
    has_scale: u32,
    log_scale: u32,
    heads_per_token: u32,
    pos0: u32,
    alpha: f32,
    floor_: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct MixArgs {
    d: u32,
    n_blocks: u32,
    n_q: u32,
    plane: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct ConvArgs {
    ch: u32,
    nc: u32,
    dyn_heads: u32,
    scale: f32,
    n_tokens: u32,
    second: u32,
    snap: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ConvHeadArgs {
    kv_dim: u32,
    head_dim: u32,
    nc: u32,
    dyn_: u32,
    scale: f32,
    n_tokens: u32,
    mode: u32,
    threshold: f32,
    eps: f32,
    snap: u32,
}

/// Optional second output of a conv dispatch.
#[derive(Clone, Copy)]
enum Second<'b> {
    /// `y2 = y`
    Copy(&'b Buffer),
    /// `y2 = res + y`
    AddRes(&'b Buffer, &'b Buffer),
}
#[repr(C)]
#[derive(Clone, Copy)]
struct SplitArgs {
    head_dim: u32,
    n_head: u32,
    n_kv_head: u32,
    n_rel: u32,
    pos: u32,
    first: u32,
    extent: u32,
    scale: f32,
    kv_dim: u32,
    n_split: u32,
    split: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct CombineAttnArgs {
    head_dim: u32,
    n_split: u32,
}
/// Largest keys-per-block in decode attention (kernels/layer.metal ATTN_SPLIT); blocks shrink to
/// 32 keys at short context so at least ~64 threadgroups run.
const ATTN_SPLIT: usize = 256;
const ATTN_SPLIT_MIN: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
struct RelTableArgs {
    n_rows: u32,
    n_rel: u32,
    extent: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct TiledArgs {
    n_tokens: u32,
    n_head: u32,
    n_kv_head: u32,
    pos0: u32,
    n_swa: u32,
    extent: u32,
    scale: f32,
    kv_dim: u32,
}
/// Queries per threadgroup in attn_tiled (kernels/layer.metal TQ).
const ATTN_TQ: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
struct NArgs {
    n: u32,
    f: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MultiSeg {
    woff: u64,
    start: u32,
    rows: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MultiArgs {
    cols: u32,
    n_seg: u32,
    total_rows: u32,
    n_tokens: u32,
    seg: [MultiSeg; 8],
}
#[repr(C)]
#[derive(Clone, Copy)]
struct ArgmaxArgs {
    n: u32,
    out_index: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct GateArgs {
    d: u32,
    rank: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct EmbedArgs {
    d: u32,
    n_tokens: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GroupArgs {
    n_pairs: u32,
    n_expert: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct MoeGemmArgs {
    n_embd: u32,
    n_ff: u32,
    n_used: u32,
    situ_beta: f32,
    situ_lin: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct StridedNormArgs {
    n: u32,
    eps: f32,
    out_stride: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct GemmF32Args {
    rows: u32,
    cols: u32,
    n_tokens: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct CombineArgs {
    n_embd: u32,
    n_used: u32,
    n_tokens: u32,
}

const MV_SG: usize = 8;
/// GEMM tile sizes (kernels/gemm.metal BM, BN).
const GEMM_BM: usize = 32;
const GEMM_BN: usize = 32;
/// Rows per tile in the expert GEMMs (kernels/gemm.metal BM2).
const GEMM_BM2: usize = 64;
/// Tokens per prefill command buffer.
pub const MAX_BATCH: usize = 512;
/// Rows per simdgroup in the MoE kernels (kernels/matvec.metal MOE_NR).
const MOE_NR: usize = 2;
/// Largest speculative verify batch (anchor + 7 drafts).
pub const MAX_VERIFY: usize = 8;
const MAX_BLOCKS: usize = 8;

/// Trace hook used while encoding: (encoder, name, slot, source buffer, float offset, float count).
type RecFn<'a> =
    dyn FnMut(&mut Encoder, &'static str, Option<usize>, &Buffer, usize, usize) -> Result<()> + 'a;

/// Byte offset of a tensor inside the model buffer.
#[derive(Clone, Copy)]
struct T(usize);

struct LayerT {
    is_swa: bool,
    wq: T,
    wk: T,
    wv: T,
    wo: T,
    w_gate: T,
    w_rel: T,
    w_sink: T,
    k_conv_proj: Option<T>,
    v_conv_proj: Option<T>,
    attn_norm: Option<T>,
    qkv_inp_norm: Option<[T; 3]>,
    q_clip: T,
    rel_clip: T,
    rel_bias: T,
    rel_extent: usize,
    k_conv: T,
    v_conv: T,
    attn_post_conv: T,
    ffn_post_conv: T,
    attn_res_query: Option<T>,
    ffn_res_query: T,
    ffn_norm: T,
    router: T,
    exp_probs_b: Option<T>,
    gate_up: T,
    down: T,
    /// Kernel suffix for the expert weight type ("iq4_nl" or "q8_0").
    exp_kind: &'static str,
}

struct GateT {
    down: T,
    forget: T,
    gain: T,
}

struct LoopT {
    msg_query: T,
    gates: [GateT; 4],
}

/// Per-slot decode state on the GPU.
struct SlotBufs {
    k: Buffer,
    v: Buffer,
    conv: [Buffer; 4],
}

/// Scratch activations, reused every layer.
struct Act {
    h: Buffer,
    prelude_h: Buffer,
    bank_out: Buffer,
    blocks: Buffer,
    bank: Buffer,
    inp: Buffer,   // 3 x d
    n_qkv: Buffer, // 3 x d
    q: Buffer,
    k_lin: Buffer,
    v_lin: Buffer,
    k_dyn: Buffer,
    v_dyn: Buffer,
    r: Buffer,
    sinks: Buffer,
    o: Buffer,
    gate: Buffer,
    wo_out: Buffer,
    attn: Buffer,
    cur: Buffer,
    m: Buffer,
    n: Buffer,
    router_logits: Buffer,
    ids: Buffer,
    wts: Buffer,
    act: Buffer,
    y: Buffer,
    ffn: Buffer,
    msgs: Buffer, // 4 x d
    z: Buffer,
    tmp: Buffer,
    logits: Buffer,
    tokens: Buffer,
    grp_count: Buffer,
    grp_offset: Buffer,
    grp_pairs: Buffer,
    yp: Buffer,
    att_m: Buffer,   // [n_head][n_split] decode-attention partial maxima
    att_l: Buffer,   // [n_head][n_split] partial sums
    att_acc: Buffer, // [n_head][n_split][head_dim] partial accumulators
    rel_tab: Buffer, // [nt][n_head][extent] relative-bias table for prefill attention
    gin: Buffer,     // [nt][2d] normalized gate inputs
    gf: Buffer,      // [nt][d] forget logits
    gg: Buffer,      // [nt][d] gain logits
}

pub struct MetalModel {
    pub hp: Hparams,
    pub ctx: Context,
    w: Buffer,
    tok_embd: T,
    tok_norm: T,
    output: T,
    output_norm: T,
    output_res_query: T,
    layers: Vec<LayerT>,
    loops: Vec<LoopT>,
    slots: Vec<SlotBufs>,
    a: Act,
    max_ctx: usize,
    max_batch: usize,
    pos: usize,
    trace_buf: Buffer,
    /// KV cache stored as f16 instead of f32 (fixed at construction; see `Options`).
    kv_f16: bool,
    /// Prefill GEMM and attention tiles in f32 (INKLING_EXACT_PREFILL=1: prefill matches decode to
    /// ~1e-6) instead of f16 (default, ~16% faster prefill). Over all 3810 positions of the swa
    /// fixture, f16 tiles leave KL vs llama.cpp unchanged (mean 4.8e-3 vs 4.7e-3) and move us from
    /// our exact run by a mean 2.1e-3 (tests/quality.rs).
    pub exact_prefill: bool,
    /// Stats of the most recent step.
    pub last: RunStats,
    /// DSpark drafter features, [4][max_batch][d]: the inputs of slots `drafter::FEATURE_SLOTS`
    /// and the final normed hidden. Off unless `enable_capture` was called.
    capture: Option<Buffer>,
    /// Per-token conv histories of a speculative verify batch (`verify` / `rewind`), laid out by
    /// `snap_off`. Allocated on first use; `verify_rows` > 0 only while a verify is encoded.
    conv_snap: Option<Buffer>,
    verify_rows: usize,
}

fn tensor(g: &Gguf, name: &str, want: DType) -> Result<T> {
    let t = g.tensor(name).map_err(inkling::Error::from)?;
    if t.dtype != want {
        return Err(Error::Model(inkling::Error::Unsupported(format!(
            "{name}: {:?}, Metal path expects {want:?}",
            t.dtype
        ))));
    }
    Ok(T(t.offset))
}

/// Expert tensors: IQ4_NL in most layers, Q8_0 in the last ones (the original's MXFP8 layers).
fn expert(g: &Gguf, name: &str) -> Result<(T, &'static str)> {
    let t = g.tensor(name).map_err(inkling::Error::from)?;
    let kind = match t.dtype {
        DType::Iq4Nl => "iq4_nl",
        DType::Q8_0 => "q8_0",
        other => {
            return Err(Error::Model(inkling::Error::Unsupported(format!(
                "{name}: {other:?} experts"
            ))));
        }
    };
    Ok((T(t.offset), kind))
}

fn opt(g: &Gguf, name: &str, want: DType) -> Result<Option<T>> {
    if g.has_tensor(name) {
        tensor(g, name, want).map(Some)
    } else {
        Ok(None)
    }
}

/// Construction options.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// Store the KV cache as f16 (half the memory; default) instead of f32 (INKLING_KV_F32=1).
    /// Same accuracy over all positions (tests/quality.rs); it does perturb near-tied routing
    /// decisions, so bit-faithful comparisons (tests/vs_cpu.rs) pin an f32 cache. Kcur/Vcur traces
    /// need an f32 cache.
    pub kv_f16: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            kv_f16: std::env::var_os("INKLING_KV_F32").is_none(),
        }
    }
}

/// Records a traced tensor: copies `len` floats from `src` into the trace buffer during the run.
struct TraceRec {
    name: &'static str,
    slot: Option<usize>,
    offset: usize,
    len: usize,
}

impl MetalModel {
    /// `g` must outlive the model: its mmap backs the weight buffer.
    pub fn new(g: &Gguf, hp: Hparams, max_ctx: usize) -> Result<Self> {
        Self::with_options(g, hp, max_ctx, Options::default())
    }

    pub fn with_options(g: &Gguf, hp: Hparams, max_ctx: usize, opts: Options) -> Result<Self> {
        // Validates every tensor shape against the hyperparameters (same rules as the CPU path).
        inkling::weights::Weights::load(g, &hp)?;
        let ctx = Context::new()?;
        let map = g.mmap();
        // SAFETY: the mmap is page-aligned, read-only, and the caller keeps `g` alive.
        let w = unsafe { ctx.buffer_no_copy(map.as_ptr(), map.len())? };

        use DType::*;
        let mut layers = Vec::new();
        for i in 0..hp.n_layer {
            let b = |s: &str| format!("blk.{i}.{s}");
            let is_swa = hp.is_swa[i];
            layers.push(LayerT {
                is_swa,
                wq: tensor(g, &b("attn_q.weight"), Q8_0)?,
                wk: tensor(g, &b("attn_k.weight"), Q8_0)?,
                wv: tensor(g, &b("attn_v.weight"), Q8_0)?,
                wo: tensor(g, &b("attn_output.weight"), Q8_0)?,
                w_gate: tensor(g, &b("attn_gate.weight"), Q8_0)?,
                w_rel: tensor(g, &b("attn_rel.weight"), Q8_0)?,
                w_sink: tensor(g, &b("attn_sink_proj.weight"), Q8_0)?,
                k_conv_proj: if is_swa {
                    None
                } else {
                    Some(tensor(g, &b("attn_k_conv_proj.weight"), Q8_0)?)
                },
                v_conv_proj: if is_swa {
                    None
                } else {
                    Some(tensor(g, &b("attn_v_conv_proj.weight"), Q8_0)?)
                },
                attn_norm: if is_swa {
                    Some(tensor(g, &b("attn_norm.weight"), F32)?)
                } else {
                    None
                },
                qkv_inp_norm: if is_swa {
                    None
                } else {
                    Some([
                        tensor(g, &b("attn_q_inp_norm.weight"), F32)?,
                        tensor(g, &b("attn_k_inp_norm.weight"), F32)?,
                        tensor(g, &b("attn_v_inp_norm.weight"), F32)?,
                    ])
                },
                q_clip: tensor(g, &b("attn_q_clip.weight"), F32)?,
                rel_clip: tensor(g, &b("attn_rel_clip.weight"), F32)?,
                rel_bias: tensor(g, &b("attn_rel_bias.weight"), F32)?,
                rel_extent: if is_swa { hp.n_swa } else { hp.rel_extent },
                k_conv: tensor(g, &b("attn_k_conv.weight"), F32)?,
                v_conv: tensor(g, &b("attn_v_conv.weight"), F32)?,
                attn_post_conv: tensor(g, &b("attn_post_conv.weight"), F32)?,
                ffn_post_conv: tensor(g, &b("ffn_post_conv.weight"), F32)?,
                attn_res_query: opt(g, &b("attn_res_query.weight"), F32)?,
                ffn_res_query: tensor(g, &b("ffn_res_query.weight"), F32)?,
                ffn_norm: tensor(g, &b("ffn_norm.weight"), F32)?,
                router: tensor(g, &b("ffn_gate_inp.weight"), F32)?,
                exp_probs_b: opt(g, &b("exp_probs_b.bias"), F32)?,
                gate_up: expert(g, &b("ffn_gate_up_exps.weight"))?.0,
                down: expert(g, &b("ffn_down_exps.weight"))?.0,
                exp_kind: expert(g, &b("ffn_gate_up_exps.weight"))?.1,
            });
        }
        let mut loops = Vec::new();
        for i in hp.loop_start..hp.loop_end {
            let gate = |c: &str| -> Result<GateT> {
                Ok(GateT {
                    down: tensor(g, &format!("blk.{i}.loop_{c}_ctrl_down.weight"), F32)?,
                    forget: tensor(g, &format!("blk.{i}.loop_{c}_ctrl_forget.weight"), F32)?,
                    gain: tensor(g, &format!("blk.{i}.loop_{c}_ctrl_gain.weight"), F32)?,
                })
            };
            loops.push(LoopT {
                msg_query: tensor(g, &format!("blk.{i}.loop_msg_query.weight"), F32)?,
                gates: [gate("q")?, gate("k")?, gate("v")?, gate("ffn")?],
            });
        }

        let f = 4usize;
        let (d, kvd, qd) = (hp.n_embd, hp.kv_dim(), hp.q_dim());
        let hist = hp.n_conv - 1;
        let kv_bytes = if opts.kv_f16 { 2 } else { 4 };
        let mut slots = Vec::with_capacity(hp.n_slots());
        for _ in 0..hp.n_slots() {
            slots.push(SlotBufs {
                k: ctx.buffer(max_ctx * kvd * kv_bytes)?,
                v: ctx.buffer(max_ctx * kvd * kv_bytes)?,
                conv: [
                    ctx.buffer(hist * kvd * f)?,
                    ctx.buffer(hist * kvd * f)?,
                    ctx.buffer(hist * d * f)?,
                    ctx.buffer(hist * d * f)?,
                ],
            });
        }
        let nb = |n: usize| ctx.buffer(n * f);
        let ub = MAX_BATCH;
        let a = Act {
            h: nb(ub * d)?,
            prelude_h: nb(ub * d)?,
            bank_out: nb(ub * d)?,
            blocks: nb(MAX_BLOCKS * ub * d)?,
            bank: nb(MAX_BLOCKS * ub * d)?,
            inp: nb(3 * ub * d)?,
            n_qkv: nb(3 * ub * d)?,
            q: nb(ub * qd)?,
            k_lin: nb(ub * kvd)?,
            v_lin: nb(ub * kvd)?,
            k_dyn: nb(ub * hp.n_head_kv * hp.n_conv)?,
            v_dyn: nb(ub * hp.n_head_kv * hp.n_conv)?,
            r: nb(ub * hp.n_head * hp.n_rel)?,
            sinks: nb(ub * hp.n_head)?,
            o: nb(ub * qd)?,
            gate: nb(ub * qd)?,
            wo_out: nb(ub * d)?,
            attn: nb(ub * d)?,
            cur: nb(ub * d)?,
            m: nb(ub * d)?,
            n: nb(ub * d)?,
            router_logits: nb(ub * hp.n_expert)?,
            ids: nb(ub * hp.n_expert_used)?,
            wts: nb(ub * hp.n_expert_used)?,
            act: nb(ub * hp.n_expert_used * hp.n_ff_exp)?,
            y: nb(ub * d)?,
            ffn: nb(ub * d)?,
            msgs: nb(4 * ub * d)?,
            z: nb(ub * hp.loop_gate_rank)?,
            tmp: nb(ub * d)?,
            logits: nb(hp.n_vocab)?,
            tokens: nb(ub)?,
            grp_count: nb(hp.n_expert)?,
            grp_offset: nb(hp.n_expert)?,
            grp_pairs: nb(ub * hp.n_expert_used)?,
            yp: nb(ub * hp.n_expert_used * d)?,
            att_m: nb(hp.n_head * max_ctx.div_ceil(ATTN_SPLIT_MIN))?,
            att_l: nb(hp.n_head * max_ctx.div_ceil(ATTN_SPLIT_MIN))?,
            att_acc: nb(hp.n_head * max_ctx.div_ceil(ATTN_SPLIT_MIN) * hp.head_dim)?,
            rel_tab: nb(ub * hp.n_head * hp.rel_extent.max(hp.n_swa))?,
            gin: nb(ub * 2 * d)?,
            gf: nb(ub * d)?,
            gg: nb(ub * d)?,
        };
        let trace_buf = ctx.buffer(16)?;
        Ok(MetalModel {
            tok_embd: tensor(g, "token_embd.weight", Q8_0)?,
            tok_norm: tensor(g, "token_embd_norm.weight", F32)?,
            output: tensor(g, "output.weight", Q8_0)?,
            output_norm: tensor(g, "output_norm.weight", F32)?,
            output_res_query: tensor(g, "output_res_query.weight", F32)?,
            hp,
            ctx,
            w,
            layers,
            loops,
            slots,
            a,
            max_ctx,
            max_batch: MAX_BATCH,
            exact_prefill: std::env::var_os("INKLING_EXACT_PREFILL").is_some(),
            capture: None,
            conv_snap: None,
            verify_rows: 0,
            kv_f16: opts.kv_f16,
            pos: 0,
            trace_buf,
            last: RunStats::default(),
        })
    }

    /// Kernel name for the selected prefill tile precision.
    fn tile(&self, base: &str) -> String {
        // INKLING_FAST_PREFILL=moe|proj limits f16 tiles to one GEMM family (any other value: all)
        let fam = std::env::var("INKLING_FAST_PREFILL").unwrap_or_default();
        let f16 = !self.exact_prefill
            && match fam.as_str() {
                "moe" => base.starts_with("moe_"),
                "proj" => base.starts_with("gemm2_q8_0"),
                _ => true,
            };
        format!("{base}_{}", if f16 { "f16" } else { "f32" })
    }

    /// Raw K-cache bytes of `slot` for the first `n` positions (diagnostics).
    #[doc(hidden)]
    pub fn debug_k_cache(&self, slot: usize, n: usize) -> Vec<u8> {
        self.slots[slot].k.as_slice::<u8>()[..n * self.hp.kv_dim() * self.kv_bytes()].to_vec()
    }

    fn kv_bytes(&self) -> usize {
        if self.kv_f16 { 2 } else { 4 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Starts a new sequence. KV rows past `pos` are never read, so only the conv histories need
    /// clearing.
    pub fn reset(&mut self) {
        self.pos = 0;
        for s in &self.slots {
            for c in &s.conv {
                c.as_mut_slice::<f32>().fill(0.0);
            }
        }
    }

    /// Runs one token and returns its logits (unpadded vocab).
    pub fn step(&mut self, token: u32, tr: &mut dyn Trace) -> Result<Vec<f32>> {
        if self.pos >= self.max_ctx {
            return Err(Error::Model(inkling::Error::ContextFull(self.pos)));
        }
        // trace layout is planned during encoding; size the buffer with a dry run of names
        let mut recs: Vec<TraceRec> = Vec::new();
        let mut trace_len = 0usize;
        {
            let mut plan = |name: &'static str, slot: Option<usize>, len: usize| {
                if tr.wants(name, slot) {
                    recs.push(TraceRec {
                        name,
                        slot,
                        offset: trace_len,
                        len,
                    });
                    trace_len += len;
                }
            };
            self.plan_trace(&mut plan);
        }
        if trace_len * 4 > self.trace_buf.len {
            self.trace_buf = self.ctx.buffer(trace_len * 4)?;
        }

        let this = &*self;
        let mut cursor = 0usize;
        let stats = this.ctx.run(|e| {
            let mut rec = |e: &mut Encoder,
                           name: &'static str,
                           slot: Option<usize>,
                           src: &Buffer,
                           src_off: usize,
                           len: usize|
             -> Result<()> {
                if cursor < recs.len() && recs[cursor].name == name && recs[cursor].slot == slot {
                    let off = recs[cursor].offset;
                    cursor += 1;
                    e.dispatch(
                        "copy_f32",
                        &[
                            buf_at(src, src_off * 4),
                            buf_at(&this.trace_buf, off * 4),
                            bytes(&NArgs {
                                n: len as u32,
                                f: 0.0,
                            }),
                        ],
                        (len.div_ceil(256), 1, 1),
                        (256, 1, 1),
                    )?;
                }
                Ok(())
            };
            this.a.tokens.write(&[token]);
            this.encode(e, 1, None, &mut rec)
        })?;

        self.last = stats;
        let tb = self.trace_buf.as_slice::<f32>();
        for r in &recs {
            tr.record(r.name, r.slot, &tb[r.offset..r.offset + r.len]);
        }
        self.pos += 1;
        Ok(self.a.logits.as_slice::<f32>()[..self.hp.n_vocab].to_vec())
    }

    /// Runs a prompt in batches of up to MAX_BATCH tokens and returns the last token's logits.
    /// Greedy generation of `n` tokens after `first`, entirely on the GPU: each step ends with an
    /// argmax that writes the next input token, so steps are queued back to back (two command
    /// buffers in flight) while the CPU encodes ahead. Returns the generated tokens.
    pub fn decode_greedy(&mut self, first: u32, n: usize) -> Result<Vec<u32>> {
        if self.pos + n > self.max_ctx {
            return Err(Error::Model(inkling::Error::ContextFull(self.pos + n)));
        }
        let out = self.ctx.buffer(n.max(1) * 4)?;
        self.a.tokens.write(&[first]);
        let mut noop = |_: &mut Encoder,
                        _: &'static str,
                        _: Option<usize>,
                        _: &Buffer,
                        _: usize,
                        _: usize|
         -> Result<()> { Ok(()) };
        let mut in_flight = std::collections::VecDeque::new();
        for i in 0..n {
            let this = &*self;
            let p = this.ctx.submit(|e| {
                this.encode(e, 1, None, &mut noop)?;
                e.dispatch(
                    "argmax_token",
                    &[
                        buf(&this.a.logits),
                        buf(&this.a.tokens),
                        buf(&out),
                        bytes(&ArgmaxArgs {
                            n: this.hp.n_vocab as u32,
                            out_index: i as u32,
                        }),
                    ],
                    (1, 1, 1),
                    (1024, 1, 1),
                )
            })?;
            self.pos += 1;
            in_flight.push_back(p);
            if in_flight.len() > 2 {
                self.last = in_flight.pop_front().expect("non-empty").wait()?;
            }
        }
        while let Some(p) = in_flight.pop_front() {
            self.last = p.wait()?;
        }
        Ok(out.as_slice::<u32>()[..n].to_vec())
    }

    /// Float offset of slot `sid`'s conv `k` (0 K, 1 V, 2 post-attention, 3 post-FFN) in
    /// `conv_snap`, or None outside a verify. Each holds MAX_VERIFY rows of [n_conv-1][channels].
    fn snap_off(&self, sid: usize, k: usize) -> Option<usize> {
        if self.verify_rows == 0 {
            return None;
        }
        let (kvd, d, nh) = (self.hp.kv_dim(), self.hp.n_embd, self.hp.n_conv - 1);
        let per_row = |k: usize| if k < 2 { kvd } else { d } * nh;
        let per_slot: usize = (0..4).map(|k| MAX_VERIFY * per_row(k)).sum();
        let within: usize = (0..k).map(|k| MAX_VERIFY * per_row(k)).sum();
        Some(sid * per_slot + within)
    }

    fn snap_buf(&self, off: Option<usize>) -> crate::runtime::Arg<'_> {
        match (off, &self.conv_snap) {
            (Some(o), Some(b)) => buf_at(b, o * 4),
            // unused by the kernel when snap = 0, but every binding needs a buffer
            _ => buf(&self.a.tmp),
        }
    }

    /// Speculative verify: runs `tokens` (at most MAX_VERIFY) as one causal batch from the current
    /// position, returning every row's logits, and records the conv histories after each row so
    /// `rewind` can keep any prefix. The position advances by `tokens.len()`.
    pub fn verify(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        assert!(
            !tokens.is_empty() && tokens.len() <= MAX_VERIFY,
            "verify batch of {}",
            tokens.len()
        );
        if self.conv_snap.is_none() {
            let (kvd, d, nh) = (self.hp.kv_dim(), self.hp.n_embd, self.hp.n_conv - 1);
            let floats = self.hp.n_slots() * MAX_VERIFY * nh * (2 * kvd + 2 * d);
            self.conv_snap = Some(self.ctx.buffer(floats * 4)?);
        }
        self.verify_rows = tokens.len();
        let mut rows = Vec::with_capacity(tokens.len());
        let r = self.prefill_each(tokens, |_, l| rows.push(l.to_vec()));
        self.verify_rows = 0;
        r?;
        Ok(rows)
    }

    /// After `verify` of n rows, keeps the first `keep` (1..=n): rolls the position back and
    /// restores every conv history to its state after row keep-1. KV rows past the position are
    /// dead and get overwritten later.
    pub fn rewind(&mut self, n: usize, keep: usize) -> Result<()> {
        assert!(keep >= 1 && keep <= n && n <= MAX_VERIFY && n <= self.pos);
        self.pos -= n - keep;
        if keep == n {
            return Ok(());
        }
        let (kvd, d, nh) = (self.hp.kv_dim(), self.hp.n_embd, self.hp.n_conv - 1);
        self.verify_rows = n;
        let this = &*self;
        let r = this.ctx.run(|e| {
            e.parallel(|e| {
                for sid in 0..this.hp.n_slots() {
                    for k in 0..4 {
                        let ch = if k < 2 { kvd } else { d };
                        let off =
                            this.snap_off(sid, k).expect("verify_rows set") + (keep - 1) * nh * ch;
                        let snap = this.conv_snap.as_ref().expect("allocated by verify");
                        this.copy(e, snap, off, &this.slots[sid].conv[k], 0, nh * ch)?;
                    }
                }
                Ok(())
            })
        });
        self.verify_rows = 0;
        self.last = r?;
        Ok(())
    }

    /// Starts recording the drafter's target features on every forward (see `captured`).
    pub fn enable_capture(&mut self) -> Result<()> {
        assert_eq!(
            self.hp.n_slots(),
            48,
            "drafter feature slots assume the 3-pass layout"
        );
        let n = inkling::drafter::N_FEAT * self.max_batch * self.hp.n_embd;
        self.capture = Some(self.ctx.buffer(n * 4)?);
        Ok(())
    }

    /// The drafter features (`N_FEAT * d`, in drafter order) of row `row` of the most recent
    /// forward. The final-hidden feature exists for every row with `prefill_each` and `step`,
    /// otherwise only for the last row.
    pub fn captured(&self, row: usize) -> Vec<f32> {
        let c = self
            .capture
            .as_ref()
            .expect("enable_capture first")
            .as_slice::<f32>();
        let (d, plane) = (self.hp.n_embd, self.max_batch * self.hp.n_embd);
        (0..inkling::drafter::N_FEAT)
            .flat_map(|f| {
                c[f * plane + row * d..f * plane + (row + 1) * d]
                    .iter()
                    .copied()
            })
            .collect()
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(Error::Model(inkling::Error::Unsupported(
                "empty prompt".into(),
            )));
        }
        if self.pos + tokens.len() > self.max_ctx {
            return Err(Error::Model(inkling::Error::ContextFull(
                self.pos + tokens.len(),
            )));
        }
        for chunk in tokens.chunks(self.max_batch) {
            let this = &*self;
            this.a.tokens.write(chunk);
            self.last = this
                .ctx
                .run(|e| this.encode(e, chunk.len(), None, &mut |_, _, _, _, _, _| Ok(())))?;
            self.pos += chunk.len();
        }
        Ok(self.a.logits.as_slice::<f32>()[..self.hp.n_vocab].to_vec())
    }

    /// Prefills `tokens` like `prefill`, but computes the logits of every position and calls
    /// `f(position, logits)` for each, in order (quality measurement; the full output head runs
    /// on every row).
    pub fn prefill_each(&mut self, tokens: &[u32], mut f: impl FnMut(usize, &[f32])) -> Result<()> {
        if self.pos + tokens.len() > self.max_ctx {
            return Err(Error::Model(inkling::Error::ContextFull(
                self.pos + tokens.len(),
            )));
        }
        let nv = self.hp.n_vocab;
        let out = self.ctx.buffer(self.max_batch * nv * 4)?;
        for chunk in tokens.chunks(self.max_batch) {
            let this = &*self;
            this.a.tokens.write(chunk);
            self.last = this
                .ctx
                .run(|e| this.encode(e, chunk.len(), Some(&out), &mut |_, _, _, _, _, _| Ok(())))?;
            let lg = out.as_slice::<f32>();
            for (i, _) in chunk.iter().enumerate() {
                f(self.pos + i, &lg[i * nv..(i + 1) * nv]);
            }
            self.pos += chunk.len();
        }
        Ok(())
    }

    /// Runs one token with per-dispatch GPU timestamps and returns (kernel, total seconds, count),
    /// sorted by time. The token advances the sequence like `step`.
    pub fn profile_step(&mut self, token: u32) -> Result<Vec<(String, f64, usize)>> {
        self.profile_batch(&[token])
    }

    /// Like `profile_step` for a batch of up to MAX_BATCH tokens (prefill).
    pub fn profile_batch(&mut self, tokens: &[u32]) -> Result<Vec<(String, f64, usize)>> {
        assert!(!tokens.is_empty() && tokens.len() <= self.max_batch);
        let this = &*self;
        this.a.tokens.write(tokens);
        let per = this.ctx.profile(4096, |e| {
            this.encode(e, tokens.len(), None, &mut |_, _, _, _, _, _| Ok(()))
        })?;
        self.pos += tokens.len();
        let mut agg: std::collections::HashMap<String, (f64, usize)> =
            std::collections::HashMap::new();
        for (k, t) in per {
            let a = agg.entry(k).or_default();
            a.0 += t;
            a.1 += 1;
        }
        let mut v: Vec<_> = agg.into_iter().map(|(k, (t, n))| (k, t, n)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(v)
    }

    /// Calls `f(name, slot, len)` for every trace point, in the order `encode_token` emits them.
    fn plan_trace(&self, f: &mut dyn FnMut(&'static str, Option<usize>, usize)) {
        let hp = &self.hp;
        let mut layer = |sid: usize| {
            let s = Some(sid);
            f("Qcur", s, hp.q_dim());
            f("Kcur", s, hp.kv_dim());
            f("Vcur", s, hp.kv_dim());
            f("Rcur", s, hp.n_head * hp.n_rel);
            f("kqv_out", s, hp.q_dim());
            f("attn_out", s, hp.n_embd);
            f("ffn_inp", s, hp.n_embd);
            f("ffn_moe_logits", s, hp.n_expert);
            f("ffn_moe_out", s, hp.n_embd);
            f("ffn_out", s, hp.n_embd);
            f("l_out", s, hp.n_embd);
        };
        for il in 0..hp.loop_start {
            layer(il);
        }
        for pass in 0..hp.loop_passes {
            for il in hp.loop_start..hp.loop_end {
                layer(hp.slot(il, pass));
            }
        }
        for il in hp.loop_end..hp.n_layer {
            layer(il);
        }
        f("result_norm", None, hp.n_embd);
        f("result_output", None, hp.n_vocab);
    }

    /// Row-major matvec / per-token batched matvec (`mv_f32` takes the token on grid y).
    #[allow(clippy::too_many_arguments)]
    fn mv(
        &self,
        e: &mut Encoder,
        kernel: &str,
        w: T,
        x: &Buffer,
        y: &Buffer,
        rows: usize,
        cols: usize,
        nt: usize,
    ) -> Result<()> {
        e.dispatch(
            kernel,
            &[
                buf_at(&self.w, w.0),
                buf(x),
                buf(y),
                bytes(&MvArgs {
                    rows: rows as u32,
                    cols: cols as u32,
                }),
            ],
            (rows.div_ceil(MV_SG), nt, 1),
            (32 * MV_SG, 1, 1),
        )
    }

    /// `out_s[t] = W_s x[t]` for several Q8_0 matrices sharing input `x` (plane at float offset
    /// `x_off`): matvec kernel for one token, simdgroup-matrix GEMM for a batch.
    fn proj(
        &self,
        e: &mut Encoder,
        x: &Buffer,
        x_off: usize,
        segs: &[(T, usize, &Buffer)],
        nt: usize,
    ) -> Result<()> {
        assert!(!segs.is_empty() && segs.len() <= 8);
        let mut args = MultiArgs {
            cols: self.hp.n_embd as u32,
            n_seg: segs.len() as u32,
            ..Default::default()
        };
        let mut start = 0u32;
        for (i, (w, rows, _)) in segs.iter().enumerate() {
            args.seg[i] = MultiSeg {
                woff: w.0 as u64,
                start,
                rows: *rows as u32,
            };
            start += *rows as u32;
        }
        args.total_rows = start;
        args.n_tokens = nt as u32;
        let mut binds = vec![buf(&self.w), buf_at(x, x_off * 4)];
        for i in 0..8 {
            binds.push(buf(segs.get(i).map(|s| s.2).unwrap_or(segs[0].2)));
        }
        binds.push(bytes(&args));
        if nt == 1 {
            e.dispatch(
                "mv_q8_0_multi",
                &binds,
                ((start as usize).div_ceil(MV_SG), 1, 1),
                (32 * MV_SG, 1, 1),
            )
        } else {
            e.dispatch(
                self.tile("gemm2_q8_0_multi").as_str(),
                &binds,
                ((start as usize).div_ceil(GEMM_BM2), nt.div_ceil(GEMM_BN), 1),
                (128, 1, 1),
            )
        }
    }

    /// RMS norm of `rows` rows of `n` floats starting at float offset `x_off`.
    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        e: &mut Encoder,
        x: &Buffer,
        x_off: usize,
        w: Option<T>,
        y: &Buffer,
        y_off: usize,
        n: usize,
        eps: f32,
        rows: usize,
    ) -> Result<()> {
        let wb = match w {
            Some(t) => buf_at(&self.w, t.0),
            None => buf(x),
        };
        e.dispatch(
            "rms_norm",
            &[
                buf_at(x, x_off * 4),
                wb,
                buf_at(y, y_off * 4),
                bytes(&NormArgs {
                    n: n as u32,
                    eps,
                    has_weight: w.is_some() as u32,
                }),
            ],
            (rows, 1, 1),
            (1024, 1, 1),
        )
    }

    fn copy(
        &self,
        e: &mut Encoder,
        src: &Buffer,
        src_off: usize,
        dst: &Buffer,
        dst_off: usize,
        n: usize,
    ) -> Result<()> {
        e.dispatch(
            "copy_f32",
            &[
                buf_at(src, src_off * 4),
                buf_at(dst, dst_off * 4),
                bytes(&NArgs {
                    n: n as u32,
                    f: 0.0,
                }),
            ],
            (n.div_ceil(256), 1, 1),
            (256, 1, 1),
        )
    }

    fn eltwise(
        &self,
        e: &mut Encoder,
        kernel: &str,
        bufs: &[&Buffer],
        n: usize,
        f: f32,
    ) -> Result<()> {
        let args = NArgs { n: n as u32, f };
        let mut binds: Vec<_> = bufs.iter().map(|b| buf(b)).collect();
        binds.push(bytes(&args));
        e.dispatch(kernel, &binds, (n.div_ceil(256), 1, 1), (256, 1, 1))
    }

    /// Depth attention for every token: blocks/out are `[plane][nt][d]`, cur is `[nt][d]`.
    #[allow(clippy::too_many_arguments)]
    fn mix(
        &self,
        e: &mut Encoder,
        blocks: &Buffer,
        n_blocks: usize,
        cur: &Buffer,
        q: T,
        n_q: usize,
        out: &Buffer,
        nt: usize,
    ) -> Result<()> {
        let d = self.hp.n_embd;
        let plane = (self.max_batch * d) as u32;
        e.dispatch(
            "depth_mix",
            &[
                buf(blocks),
                buf(cur),
                buf_at(&self.w, q.0),
                buf(out),
                bytes(&MixArgs {
                    d: d as u32,
                    n_blocks: n_blocks as u32,
                    n_q: n_q as u32,
                    plane,
                }),
            ],
            (nt, 1, 1),
            (1024, 1, 1),
        )
    }

    /// Causal short conv over `nt` tokens. `y` is written at float offset `y_off` (the K/V convs
    /// write straight into the cache row); `second` optionally also writes `y` or `res + y`.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        e: &mut Encoder,
        x: &Buffer,
        hist: &Buffer,
        taps: T,
        dyn_: Option<&Buffer>,
        y: &Buffer,
        y_off: usize,
        second: Second,
        ch: usize,
        nt: usize,
        snap: Option<usize>,
    ) -> Result<()> {
        let nc = self.hp.n_conv;
        let dyn_heads = if dyn_.is_some() { self.hp.n_head_kv } else { 0 };
        let scale = 1.0 / (((if dyn_.is_some() { 2 } else { 1 }) * nc + 1) as f32).sqrt();
        let (mode, res, y2) = match second {
            Second::Copy(o) => (1, x, o),
            Second::AddRes(r, o) => (2, r, o),
        };
        e.dispatch(
            "conv_step",
            &[
                buf(x),
                buf(hist),
                buf_at(&self.w, taps.0),
                buf(dyn_.unwrap_or(x)),
                buf_at(y, y_off * 4),
                bytes(&ConvArgs {
                    ch: ch as u32,
                    nc: nc as u32,
                    dyn_heads: dyn_heads as u32,
                    scale,
                    n_tokens: nt as u32,
                    second: mode,
                    snap: snap.is_some() as u32,
                }),
                buf(res),
                buf(y2),
                self.snap_buf(snap),
            ],
            (ch.div_ceil(256), 1, 1),
            (256, 1, 1),
        )
    }

    /// Per-head L2 clip (optionally scaled, optionally times tau(position)) of `[nt][heads][dim]`
    /// starting at float offset `x_off`.
    #[allow(clippy::too_many_arguments)]
    fn clip(
        &self,
        e: &mut Encoder,
        x: &Buffer,
        x_off: usize,
        scale: Option<T>,
        dim: usize,
        heads: usize,
        threshold: f32,
        log_scale: bool,
        nt: usize,
    ) -> Result<()> {
        let hp = &self.hp;
        e.dispatch(
            "clip_heads",
            &[
                buf_at(x, x_off * 4),
                match scale {
                    Some(t) => buf_at(&self.w, t.0),
                    None => buf(x),
                },
                bytes(&ClipArgs {
                    dim: dim as u32,
                    threshold,
                    has_scale: scale.is_some() as u32,
                    log_scale: log_scale as u32,
                    heads_per_token: heads as u32,
                    pos0: self.pos as u32,
                    alpha: hp.log_scale_alpha,
                    floor_: hp.log_scale_floor,
                }),
            ],
            (heads * nt, 1, 1),
            (32, 1, 1),
        )
    }

    /// `out = gate(x, m)` per token for one message gate; x, m, out are `[nt][d]` planes at float
    /// offsets. `out` may alias `x`.
    #[allow(clippy::too_many_arguments)]
    fn msg_gate(
        &self,
        e: &mut Encoder,
        g: &GateT,
        x: &Buffer,
        x_off: usize,
        m: &Buffer,
        m_off: usize,
        out: &Buffer,
        out_off: usize,
        nt: usize,
    ) -> Result<()> {
        let d = self.hp.n_embd;
        let rank = self.hp.loop_gate_rank;
        let a = &self.a;
        if nt > 1 {
            // batch: z = D [rms(x); rms(m)] as GEMMs (the per-token kernels re-read the gate
            // matrices once per token)
            for (src, off, dst_off) in [(x, x_off, 0usize), (m, m_off, d)] {
                e.dispatch(
                    "rms_norm_strided",
                    &[
                        buf_at(src, off * 4),
                        buf_at(&a.gin, dst_off * 4),
                        bytes(&StridedNormArgs {
                            n: d as u32,
                            eps: 1e-6,
                            out_stride: (2 * d) as u32,
                        }),
                    ],
                    (nt, 1, 1),
                    (1024, 1, 1),
                )?;
            }
            self.gemm_f32(e, g.down, &a.gin, &a.z, rank, 2 * d, nt)?;
            self.gemm_f32(e, g.forget, &a.z, &a.gf, d, rank, nt)?;
            self.gemm_f32(e, g.gain, &a.z, &a.gg, d, rank, nt)?;
            return e.dispatch(
                "msg_gate_combine",
                &[
                    buf_at(x, x_off * 4),
                    buf_at(m, m_off * 4),
                    buf(&a.gf),
                    buf(&a.gg),
                    buf_at(out, out_off * 4),
                    bytes(&NArgs {
                        n: (nt * d) as u32,
                        f: 0.0,
                    }),
                ],
                ((nt * d).div_ceil(256), 1, 1),
                (256, 1, 1),
            );
        }
        let ga = GateArgs {
            d: d as u32,
            rank: rank as u32,
        };
        self.msg_gate_down(e, g, x, x_off, m, m_off, 0, &ga)?;
        self.msg_gate_up(e, g, x, x_off, m, m_off, out, out_off, 0, &ga)
    }

    /// Decode gate, first half: z[z_off..z_off+rank] = D [rms(x); rms(m)].
    #[allow(clippy::too_many_arguments)]
    fn msg_gate_down(
        &self,
        e: &mut Encoder,
        g: &GateT,
        x: &Buffer,
        x_off: usize,
        m: &Buffer,
        m_off: usize,
        z_off: usize,
        ga: &GateArgs,
    ) -> Result<()> {
        e.dispatch(
            "msg_gate_down",
            &[
                buf_at(&self.w, g.down.0),
                buf_at(x, x_off * 4),
                buf_at(m, m_off * 4),
                buf_at(&self.a.z, z_off * 4),
                bytes(ga),
            ],
            (ga.rank as usize, 1, 1),
            (1024, 1, 1),
        )
    }

    /// Decode gate, second half: out = x and m mixed through the forget / gain gates of z.
    #[allow(clippy::too_many_arguments)]
    fn msg_gate_up(
        &self,
        e: &mut Encoder,
        g: &GateT,
        x: &Buffer,
        x_off: usize,
        m: &Buffer,
        m_off: usize,
        out: &Buffer,
        out_off: usize,
        z_off: usize,
        ga: &GateArgs,
    ) -> Result<()> {
        e.dispatch(
            "msg_gate_up",
            &[
                buf_at(&self.w, g.forget.0),
                buf_at(&self.w, g.gain.0),
                buf_at(&self.a.z, z_off * 4),
                buf_at(x, x_off * 4),
                buf_at(m, m_off * 4),
                buf_at(out, out_off * 4),
                bytes(ga),
            ],
            ((ga.d as usize).div_ceil(MV_SG), 1, 1),
            (32 * MV_SG, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_f32(
        &self,
        e: &mut Encoder,
        w: T,
        x: &Buffer,
        y: &Buffer,
        rows: usize,
        cols: usize,
        nt: usize,
    ) -> Result<()> {
        e.dispatch(
            "gemm_f32",
            &[
                buf_at(&self.w, w.0),
                buf(x),
                buf(y),
                bytes(&GemmF32Args {
                    rows: rows as u32,
                    cols: cols as u32,
                    n_tokens: nt as u32,
                }),
            ],
            (rows.div_ceil(GEMM_BM), nt.div_ceil(GEMM_BN), 1),
            (128, 1, 1),
        )
    }

    fn moe(&self, e: &mut Encoder, lw: &LayerT, nt: usize) -> Result<()> {
        let hp = &self.hp;
        let a = &self.a;
        let d = hp.n_embd;
        let moe = MoeArgs {
            n_embd: d as u32,
            n_ff: hp.n_ff_exp as u32,
            n_used: hp.n_expert_used as u32,
            situ_beta: hp.situ_beta,
            situ_lin: hp.situ_linear_beta,
        };
        // decode and short batches (speculative verify): one matvec per (token, expert) pair; a
        // grouped GEMM would fill its 32-token tiles with a token or two per expert
        if nt <= MAX_VERIFY {
            e.dispatch(
                &format!("moe_gate_up_{}", lw.exp_kind),
                &[
                    buf_at(&self.w, lw.gate_up.0),
                    buf(&a.n),
                    buf(&a.ids),
                    buf(&a.act),
                    bytes(&moe),
                ],
                (
                    hp.n_ff_exp.div_ceil(MV_SG * MOE_NR),
                    nt * hp.n_expert_used,
                    1,
                ),
                (32 * MV_SG, 1, 1),
            )?;
            return e.dispatch(
                &format!("moe_down_{}", lw.exp_kind),
                &[
                    buf_at(&self.w, lw.down.0),
                    buf(&a.act),
                    buf(&a.ids),
                    buf(&a.wts),
                    buf(&a.y),
                    bytes(&moe),
                ],
                (d.div_ceil(MV_SG * MOE_NR), nt, 1),
                (32 * MV_SG, 1, 1),
            );
        }
        // batch: group (token, k) pairs by expert, grouped GEMMs, then combine in k order
        let n_pairs = nt * hp.n_expert_used;
        e.dispatch(
            "moe_group",
            &[
                buf(&a.ids),
                buf(&a.grp_count),
                buf(&a.grp_offset),
                buf(&a.grp_pairs),
                bytes(&GroupArgs {
                    n_pairs: n_pairs as u32,
                    n_expert: hp.n_expert as u32,
                }),
            ],
            (1, 1, 1),
            (hp.n_expert, 1, 1),
        )?;
        // a single expert can receive at most one pair per token
        let max_pairs = nt;
        let gm = MoeGemmArgs {
            n_embd: d as u32,
            n_ff: hp.n_ff_exp as u32,
            n_used: hp.n_expert_used as u32,
            situ_beta: hp.situ_beta,
            situ_lin: hp.situ_linear_beta,
        };
        e.dispatch(
            &self.tile(&format!("moe_gemm2_gate_up_{}", lw.exp_kind)),
            &[
                buf_at(&self.w, lw.gate_up.0),
                buf(&a.n),
                buf(&a.grp_count),
                buf(&a.grp_offset),
                buf(&a.grp_pairs),
                buf(&a.act),
                bytes(&gm),
            ],
            (
                hp.n_ff_exp.div_ceil(GEMM_BM2),
                max_pairs.div_ceil(GEMM_BN),
                hp.n_expert,
            ),
            (128, 1, 1),
        )?;
        e.dispatch(
            &self.tile(&format!("moe_gemm2_down_{}", lw.exp_kind)),
            &[
                buf_at(&self.w, lw.down.0),
                buf(&a.act),
                buf(&a.grp_count),
                buf(&a.grp_offset),
                buf(&a.grp_pairs),
                buf(&a.yp),
                bytes(&gm),
            ],
            (
                d.div_ceil(GEMM_BM2),
                max_pairs.div_ceil(GEMM_BN),
                hp.n_expert,
            ),
            (128, 1, 1),
        )?;
        e.dispatch(
            "moe_combine",
            &[
                buf(&a.yp),
                buf(&a.wts),
                buf(&a.y),
                bytes(&CombineArgs {
                    n_embd: d as u32,
                    n_used: hp.n_expert_used as u32,
                    n_tokens: nt as u32,
                }),
            ],
            ((nt * d).div_ceil(256), 1, 1),
            (256, 1, 1),
        )
    }

    /// One decoder layer for `nt` tokens at positions pos..pos+nt.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        e: &mut Encoder,
        il: usize,
        sid: usize,
        n_blocks: &mut usize,
        msgs: bool,
        nt: usize,
        rec: &mut RecFn,
    ) -> Result<()> {
        let hp = &self.hp;
        let lw = &self.layers[il];
        let a = &self.a;
        let s = Some(sid);
        let (d, hd, nh, nkv, nr, kvd, qd) = (
            hp.n_embd,
            hp.head_dim,
            hp.n_head,
            hp.n_head_kv,
            hp.n_rel,
            hp.kv_dim(),
            hp.q_dim(),
        );
        let plane = self.max_batch * d;
        let slot = &self.slots[sid];
        if let Some(c) = &self.capture
            && let Some(fi) = inkling::drafter::FEATURE_SLOTS
                .iter()
                .position(|&x| x == sid)
        {
            self.copy(e, &a.h, 0, c, fi * plane, nt * d)?;
        }

        // attention inputs: planes of a.inp ([3][nt][d]). SWA layers outside loop passes 2-3 feed
        // q, k and v from one input, so only plane 0 is built and normalized.
        let shared_input = lw.is_swa && !msgs;
        let n_planes = if shared_input { 1 } else { 3 };
        match lw.attn_res_query {
            None => e.parallel(|e| {
                for j in 0..n_planes {
                    self.copy(e, &a.h, 0, &a.inp, j * plane, nt * d)?;
                }
                Ok(())
            })?,
            Some(q) => {
                if lw.is_swa {
                    self.mix(e, &a.blocks, *n_blocks, &a.h, q, 1, &a.inp, nt)?;
                    e.parallel(|e| {
                        for j in 1..n_planes {
                            self.copy(e, &a.inp, 0, &a.inp, j * plane, nt * d)?;
                        }
                        Ok(())
                    })?;
                } else {
                    self.mix(e, &a.blocks, *n_blocks, &a.h, q, 3, &a.inp, nt)?;
                }
            }
        }
        if msgs {
            let gates = &self.loops[il - hp.loop_start].gates;
            if nt == 1 {
                // the three planes' gates are independent; each gets its own slice of a.z
                let ga = GateArgs {
                    d: d as u32,
                    rank: hp.loop_gate_rank as u32,
                };
                let rank = hp.loop_gate_rank;
                e.parallel(|e| {
                    for (j, g) in gates.iter().take(3).enumerate() {
                        self.msg_gate_down(
                            e,
                            g,
                            &a.inp,
                            j * plane,
                            &a.msgs,
                            j * plane,
                            j * rank,
                            &ga,
                        )?;
                    }
                    Ok(())
                })?;
                e.parallel(|e| {
                    for (j, g) in gates.iter().take(3).enumerate() {
                        self.msg_gate_up(
                            e,
                            g,
                            &a.inp,
                            j * plane,
                            &a.msgs,
                            j * plane,
                            &a.inp,
                            j * plane,
                            j * rank,
                            &ga,
                        )?;
                    }
                    Ok(())
                })?;
            } else {
                for (j, gate) in gates.iter().take(3).enumerate() {
                    self.msg_gate(
                        e,
                        gate,
                        &a.inp,
                        j * plane,
                        &a.msgs,
                        j * plane,
                        &a.inp,
                        j * plane,
                        nt,
                    )?;
                }
            }
        }
        e.parallel(|e| {
            for j in 0..n_planes {
                let w = match (lw.attn_norm, lw.qkv_inp_norm) {
                    (Some(t), _) => t,
                    (None, Some(ts)) => ts[j],
                    _ => unreachable!("checked at load"),
                };
                self.norm(
                    e,
                    &a.inp,
                    j * plane,
                    Some(w),
                    &a.n_qkv,
                    j * plane,
                    d,
                    hp.rms_eps,
                    nt,
                )?;
            }
            Ok(())
        })?;

        // projections: one multi-matrix dispatch per distinct input
        let ncv = nkv * hp.n_conv;
        let mut q_side: Vec<(T, usize, &Buffer)> = vec![
            (lw.wq, qd, &a.q),
            (lw.w_rel, nh * nr, &a.r),
            (lw.w_sink, nh, &a.sinks),
            (lw.w_gate, qd, &a.gate),
        ];
        let mut k_side: Vec<(T, usize, &Buffer)> = vec![(lw.wk, kvd, &a.k_lin)];
        let mut v_side: Vec<(T, usize, &Buffer)> = vec![(lw.wv, kvd, &a.v_lin)];
        if let Some(p) = lw.k_conv_proj {
            k_side.push((p, ncv, &a.k_dyn));
        }
        if let Some(p) = lw.v_conv_proj {
            v_side.push((p, ncv, &a.v_dyn));
        }
        if shared_input {
            q_side.append(&mut k_side);
            q_side.append(&mut v_side);
            self.proj(e, &a.n_qkv, 0, &q_side, nt)?;
        } else {
            e.parallel(|e| {
                self.proj(e, &a.n_qkv, 0, &q_side, nt)?;
                self.proj(e, &a.n_qkv, plane, &k_side, nt)?;
                self.proj(e, &a.n_qkv, 2 * plane, &v_side, nt)
            })?;
        }
        let kv_off = self.pos * kvd;
        // q clip, the K and V convs into the cache, and the r clip touch disjoint buffers
        e.parallel(|e| {
            self.clip(
                e,
                &a.q,
                0,
                Some(lw.q_clip),
                hd,
                nh,
                hp.qk_clip,
                !lw.is_swa,
                nt,
            )?;

            // K and V convolve straight into the cache rows pos..pos+nt, with the per-head key clip /
            // value RMS norm fused in (one threadgroup per KV head)
            let nc = hp.n_conv;
            for (lin, hist, taps, dynw, cache, mode) in [
                (
                    &a.k_lin,
                    &slot.conv[0],
                    lw.k_conv,
                    lw.k_conv_proj.map(|_| &a.k_dyn),
                    &slot.k,
                    0u32,
                ),
                (
                    &a.v_lin,
                    &slot.conv[1],
                    lw.v_conv,
                    lw.v_conv_proj.map(|_| &a.v_dyn),
                    &slot.v,
                    1u32,
                ),
            ] {
                let dyn_ = dynw.is_some();
                e.dispatch(
                    if self.kv_f16 {
                        "conv_head_f16"
                    } else {
                        "conv_head_f32"
                    },
                    &[
                        buf(lin),
                        buf(hist),
                        buf_at(&self.w, taps.0),
                        buf(dynw.unwrap_or(lin)),
                        buf_at(cache, kv_off * self.kv_bytes()),
                        bytes(&ConvHeadArgs {
                            kv_dim: kvd as u32,
                            head_dim: hd as u32,
                            nc: nc as u32,
                            dyn_: dyn_ as u32,
                            scale: 1.0 / (((if dyn_ { 2 } else { 1 }) * nc + 1) as f32).sqrt(),
                            n_tokens: nt as u32,
                            mode,
                            threshold: hp.qk_clip,
                            eps: hp.rms_eps,
                            snap: self.snap_off(sid, mode as usize).is_some() as u32,
                        }),
                        self.snap_buf(self.snap_off(sid, mode as usize)),
                    ],
                    (nkv, 1, 1),
                    (hd, 1, 1),
                )?;
            }
            self.clip(
                e,
                &a.r,
                0,
                Some(lw.rel_clip),
                nr,
                nh,
                hp.rel_clip,
                !lw.is_swa,
                nt,
            )?;
            Ok(())
        })?;
        rec(e, "Qcur", s, &a.q, 0, qd)?;
        if !self.kv_f16 {
            rec(e, "Kcur", s, &slot.k, kv_off, kvd)?;
            rec(e, "Vcur", s, &slot.v, kv_off, kvd)?;
        }
        rec(e, "Rcur", s, &a.r, 0, nh * nr)?;

        assert_eq!(
            hd, 128,
            "attention kernels run one thread per head dimension"
        );
        // gated attention output ends up in a.gate (decode: fused into the combine step)
        if nt == 1 {
            // decode: split the visible keys into blocks across threadgroups, one per (kv head,
            // block), then merge the blocks per head
            let first = if lw.is_swa {
                (self.pos + 1).saturating_sub(hp.n_swa)
            } else {
                0
            };
            let n_keys = self.pos + 1 - first;
            let split = n_keys
                .div_ceil(64 / nkv)
                .next_multiple_of(32)
                .clamp(ATTN_SPLIT_MIN, ATTN_SPLIT);
            let n_split = n_keys.div_ceil(split);
            e.dispatch(
                if self.kv_f16 {
                    "attn_split_f16"
                } else {
                    "attn_split_f32"
                },
                &[
                    buf(&a.q),
                    buf(&slot.k),
                    buf(&slot.v),
                    buf(&a.r),
                    buf_at(&self.w, lw.rel_bias.0),
                    buf(&a.att_m),
                    buf(&a.att_l),
                    buf(&a.att_acc),
                    bytes(&SplitArgs {
                        head_dim: hd as u32,
                        n_head: nh as u32,
                        n_kv_head: nkv as u32,
                        n_rel: nr as u32,
                        pos: self.pos as u32,
                        first: first as u32,
                        extent: lw.rel_extent as u32,
                        scale: 1.0 / (hd as f32).sqrt(),
                        kv_dim: kvd as u32,
                        n_split: n_split as u32,
                        split: split as u32,
                    }),
                ],
                (nkv, n_split, 1),
                (hd, 1, 1),
            )?;
            e.dispatch(
                "attn_combine",
                &[
                    buf(&a.att_m),
                    buf(&a.att_l),
                    buf(&a.att_acc),
                    buf(&a.sinks),
                    buf(&a.o),
                    bytes(&CombineAttnArgs {
                        head_dim: hd as u32,
                        n_split: n_split as u32,
                    }),
                    buf(&a.gate),
                ],
                (nh, 1, 1),
                (hd, 1, 1),
            )?;
            rec(e, "kqv_out", s, &a.o, 0, qd)?;
        } else {
            // prefill: relative-bias table per (token, head), then query-tiled attention
            e.dispatch(
                "rel_table",
                &[
                    buf(&a.r),
                    buf_at(&self.w, lw.rel_bias.0),
                    buf(&a.rel_tab),
                    bytes(&RelTableArgs {
                        n_rows: (nt * nh) as u32,
                        n_rel: nr as u32,
                        extent: lw.rel_extent as u32,
                    }),
                ],
                (lw.rel_extent.div_ceil(256), nt * nh, 1),
                (256, 1, 1),
            )?;
            let kernel = format!(
                "attn_tiled_{}_{}",
                if self.kv_f16 { "kv16" } else { "kv32" },
                if self.exact_prefill { "f32" } else { "f16" }
            );
            e.dispatch(
                &kernel,
                &[
                    buf(&a.q),
                    buf(&slot.k),
                    buf(&slot.v),
                    buf(&a.rel_tab),
                    buf(&a.sinks),
                    buf(&a.o),
                    bytes(&TiledArgs {
                        n_tokens: nt as u32,
                        n_head: nh as u32,
                        n_kv_head: nkv as u32,
                        pos0: self.pos as u32,
                        n_swa: if lw.is_swa { hp.n_swa as u32 } else { 0 },
                        extent: lw.rel_extent as u32,
                        scale: 1.0 / (hd as f32).sqrt(),
                        kv_dim: kvd as u32,
                    }),
                ],
                (nt.div_ceil(ATTN_TQ), nh, 1),
                (128, 1, 1),
            )?;
            rec(e, "kqv_out", s, &a.o, 0, qd)?;
            // gate <- o * sigmoid(gate)
            e.dispatch(
                "gate_mul",
                &[
                    buf(&a.o),
                    buf(&a.gate),
                    bytes(&NArgs {
                        n: (nt * qd) as u32,
                        f: 0.0,
                    }),
                ],
                ((nt * qd).div_ceil(256), 1, 1),
                (256, 1, 1),
            )?;
        }
        if nt == 1 {
            self.mv(e, "mv_q8_0", lw.wo, &a.gate, &a.wo_out, d, qd, 1)?;
        } else {
            let mut args = MultiArgs {
                cols: qd as u32,
                n_seg: 1,
                total_rows: d as u32,
                n_tokens: nt as u32,
                ..Default::default()
            };
            args.seg[0] = MultiSeg {
                woff: lw.wo.0 as u64,
                start: 0,
                rows: d as u32,
            };
            let mut binds = vec![buf(&self.w), buf(&a.gate)];
            binds.extend((0..8).map(|_| buf(&a.wo_out)));
            binds.push(bytes(&args));
            e.dispatch(
                self.tile("gemm2_q8_0_multi").as_str(),
                &binds,
                (d.div_ceil(GEMM_BM2), nt.div_ceil(GEMM_BN), 1),
                (128, 1, 1),
            )?;
        }

        // post-attention conv, fused with the residual bookkeeping: the first layer of a block
        // pushes its input onto the block list and restarts the stream from the attention output
        let boundary = il.is_multiple_of(hp.attn_res_block);
        let second = if boundary {
            Second::Copy(&a.cur)
        } else {
            Second::AddRes(&a.h, &a.cur)
        };
        // on a boundary the conv does not read a.h, so the push can run alongside it
        let blk = *n_blocks;
        e.parallel(|e| {
            if boundary {
                assert!(blk < MAX_BLOCKS, "block list overflow");
                self.copy(e, &a.h, 0, &a.blocks, blk * plane, nt * d)?;
            }
            self.conv(
                e,
                &a.wo_out,
                &slot.conv[2],
                lw.attn_post_conv,
                None,
                &a.attn,
                0,
                second,
                d,
                nt,
                self.snap_off(sid, 2),
            )
        })?;
        if boundary {
            *n_blocks += 1;
        }
        rec(e, "attn_out", s, &a.attn, 0, d)?;
        rec(e, "ffn_inp", s, &a.cur, 0, d)?;

        // MoE
        self.mix(
            e,
            &a.blocks,
            *n_blocks,
            &a.cur,
            lw.ffn_res_query,
            1,
            &a.m,
            nt,
        )?;
        if msgs {
            let g = &self.loops[il - hp.loop_start].gates[3];
            self.msg_gate(e, g, &a.m, 0, &a.msgs, 3 * plane, &a.m, 0, nt)?;
        }
        self.norm(e, &a.m, 0, Some(lw.ffn_norm), &a.n, 0, d, hp.rms_eps, nt)?;
        e.dispatch(
            "mv_f32_rowtg",
            &[
                buf_at(&self.w, lw.router.0),
                buf(&a.n),
                buf(&a.router_logits),
                bytes(&MvArgs {
                    rows: hp.n_expert as u32,
                    cols: d as u32,
                }),
            ],
            (hp.n_expert, nt, 1),
            (256, 1, 1),
        )?;
        rec(e, "ffn_moe_logits", s, &a.router_logits, 0, hp.n_expert)?;
        let Some(bias) = lw.exp_probs_b else {
            return Err(Error::Model(inkling::Error::Unsupported(
                "Metal path needs exp_probs_b".into(),
            )));
        };
        e.dispatch(
            "moe_topk",
            &[
                buf(&a.router_logits),
                buf_at(&self.w, bias.0),
                buf(&a.ids),
                buf(&a.wts),
                bytes(&TopkArgs {
                    n_expert: hp.n_expert as u32,
                    k: hp.n_expert_used as u32,
                    scale: hp.expert_weights_scale,
                }),
            ],
            (nt, 1, 1),
            (hp.n_expert, 1, 1),
        )?;
        self.moe(e, lw, nt)?;
        rec(e, "ffn_moe_out", s, &a.y, 0, d)?;
        // post-FFN conv fused with the residual add: h = cur + conv(y)
        self.conv(
            e,
            &a.y,
            &slot.conv[3],
            lw.ffn_post_conv,
            None,
            &a.ffn,
            0,
            Second::AddRes(&a.cur, &a.h),
            d,
            nt,
            self.snap_off(sid, 3),
        )?;
        rec(e, "ffn_out", s, &a.ffn, 0, d)?;
        rec(e, "l_out", s, &a.h, 0, d)?;
        Ok(())
    }

    /// Encodes `nt` tokens (already in a.tokens) at positions pos..pos+nt; logits of the last one.
    fn encode(
        &self,
        e: &mut Encoder,
        nt: usize,
        all_logits: Option<&Buffer>,
        rec: &mut RecFn,
    ) -> Result<()> {
        let hp = &self.hp;
        let a = &self.a;
        let d = hp.n_embd;
        let plane = self.max_batch * d;

        e.dispatch(
            "embed_q8_0",
            &[
                buf_at(&self.w, self.tok_embd.0),
                buf(&a.tokens),
                buf(&a.tmp),
                bytes(&EmbedArgs {
                    d: d as u32,
                    n_tokens: nt as u32,
                }),
            ],
            (d.div_ceil(256), nt, 1),
            (256, 1, 1),
        )?;
        self.norm(
            e,
            &a.tmp,
            0,
            Some(self.tok_norm),
            &a.h,
            0,
            d,
            hp.rms_eps,
            nt,
        )?;

        let mut n_blocks = 0usize;
        for il in 0..hp.loop_start {
            self.layer(e, il, il, &mut n_blocks, false, nt, rec)?;
        }
        let n_pre = n_blocks;
        self.copy(e, &a.h, 0, &a.prelude_h, 0, nt * d)?;
        for pass in 0..hp.loop_passes {
            if pass > 0 {
                // bank = previous pass's blocks after the first core block, plus its output
                let n_bank = n_blocks - (n_pre + 1);
                self.copy(
                    e,
                    &a.blocks,
                    (n_pre + 1) * plane,
                    &a.bank,
                    0,
                    n_bank * plane,
                )?;
                self.copy(e, &a.h, 0, &a.bank_out, 0, nt * d)?;
                self.copy(e, &a.prelude_h, 0, &a.h, 0, nt * d)?;
                n_blocks = n_pre;
                for il in hp.loop_start..hp.loop_end {
                    let lt = &self.loops[il - hp.loop_start];
                    self.mix(
                        e,
                        &a.bank,
                        n_bank,
                        &a.bank_out,
                        lt.msg_query,
                        4,
                        &a.msgs,
                        nt,
                    )?;
                    self.layer(e, il, hp.slot(il, pass), &mut n_blocks, true, nt, rec)?;
                }
            } else {
                for il in hp.loop_start..hp.loop_end {
                    self.layer(e, il, il, &mut n_blocks, false, nt, rec)?;
                }
            }
        }
        for il in hp.loop_end..hp.n_layer {
            self.layer(e, il, il, &mut n_blocks, false, nt, rec)?;
        }

        if let Some(out) = all_logits {
            // every position's logits ([nt][n_vocab], for quality measurement)
            self.mix(
                e,
                &a.blocks,
                n_blocks,
                &a.h,
                self.output_res_query,
                1,
                &a.m,
                nt,
            )?;
            self.norm(
                e,
                &a.m,
                0,
                Some(self.output_norm),
                &a.n,
                0,
                d,
                hp.rms_eps,
                nt,
            )?;
            if let Some(c) = &self.capture {
                self.copy(e, &a.n, 0, c, 3 * plane, nt * d)?;
            }
            let mut args = MultiArgs {
                cols: d as u32,
                n_seg: 1,
                total_rows: hp.n_vocab as u32,
                n_tokens: nt as u32,
                ..Default::default()
            };
            args.seg[0] = MultiSeg {
                woff: self.output.0 as u64,
                start: 0,
                rows: hp.n_vocab as u32,
            };
            let mut binds = vec![buf(&self.w), buf(&a.n)];
            binds.extend((0..8).map(|_| buf(out)));
            binds.push(bytes(&args));
            e.dispatch(
                "gemm2_q8_0_multi_f32",
                &binds,
                (hp.n_vocab.div_ceil(GEMM_BM2), nt.div_ceil(GEMM_BN), 1),
                (128, 1, 1),
            )?;
            if hp.final_softcap > 0.0 {
                self.eltwise(
                    e,
                    "softcap_inplace",
                    &[out],
                    nt * hp.n_vocab,
                    hp.final_softcap,
                )?;
            }
            return Ok(());
        }
        // only the last token's logits are needed: run the output head on that row
        let last = (nt - 1) * d;
        for b in 0..n_blocks {
            self.copy(e, &a.blocks, b * plane + last, &a.bank, b * plane, d)?;
        }
        self.copy(e, &a.h, last, &a.tmp, 0, d)?;
        self.mix(
            e,
            &a.bank,
            n_blocks,
            &a.tmp,
            self.output_res_query,
            1,
            &a.m,
            1,
        )?;
        self.norm(
            e,
            &a.m,
            0,
            Some(self.output_norm),
            &a.n,
            0,
            d,
            hp.rms_eps,
            1,
        )?;
        if let Some(c) = &self.capture {
            self.copy(e, &a.n, 0, c, 3 * plane + last, d)?;
        }
        rec(e, "result_norm", None, &a.n, 0, d)?;
        self.mv(e, "mv_q8_0", self.output, &a.n, &a.logits, hp.n_vocab, d, 1)?;
        if hp.final_softcap > 0.0 {
            self.eltwise(
                e,
                "softcap_inplace",
                &[&a.logits],
                hp.n_vocab,
                hp.final_softcap,
            )?;
        }
        rec(e, "result_output", None, &a.logits, 0, hp.n_vocab)?;
        Ok(())
    }
}
