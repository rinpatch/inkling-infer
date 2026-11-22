//! Minimal GGUF v3 reader: metadata, tensor directory, and zero-copy tensor bytes over an mmap.

mod quant;
pub mod safetensors;

pub use quant::{BLOCK_IQ4_NL_BYTES, BLOCK_Q8_0_BYTES, IQ4_NL_VALUES, QK, dequantize_row, dot_row};

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a GGUF file")]
    BadMagic,
    #[error("unsupported GGUF version {0}")]
    Version(u32),
    #[error("truncated file at offset {0}")]
    Truncated(usize),
    #[error("unknown metadata value type {0}")]
    ValueType(u32),
    #[error("unsupported tensor type {0}")]
    TensorType(u32),
    #[error("missing metadata key {0}")]
    MissingKey(String),
    #[error("metadata key {0} has an unexpected type")]
    KeyType(String),
    #[error("missing tensor {0}")]
    MissingTensor(String),
    #[error("tensor {0} extends past the end of the file")]
    TensorBounds(String),
    #[error("safetensors header: {0}")]
    Header(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// ggml tensor types present in the models we load. Values are the ggml type ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Q8_0,
    Iq4Nl,
    Bf16,
}

impl DType {
    fn from_id(id: u32) -> Result<Self> {
        Ok(match id {
            0 => DType::F32,
            1 => DType::F16,
            8 => DType::Q8_0,
            20 => DType::Iq4Nl,
            30 => DType::Bf16,
            other => return Err(Error::TensorType(other)),
        })
    }

    /// Elements per storage block.
    pub fn block_elems(self) -> usize {
        match self {
            DType::F32 | DType::F16 | DType::Bf16 => 1,
            DType::Q8_0 | DType::Iq4Nl => QK,
        }
    }

    /// Bytes per storage block.
    pub fn block_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::Bf16 => 2,
            DType::Q8_0 => BLOCK_Q8_0_BYTES,
            DType::Iq4Nl => BLOCK_IQ4_NL_BYTES,
        }
    }

    /// Bytes for `n` consecutive elements; `n` must be a multiple of the block size.
    pub fn row_bytes(self, n: usize) -> usize {
        debug_assert_eq!(n % self.block_elems(), 0);
        n / self.block_elems() * self.block_bytes()
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::I64(v) if v >= 0 => v as u64,
            _ => return None,
        })
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => None,
        }
    }
}

/// Directory entry for one tensor. `dims[0]` is the fastest-varying (ggml `ne[0]`).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: DType,
    /// Absolute byte offset in the file.
    pub offset: usize,
    pub nbytes: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    mmap: Mmap,
    pub version: u32,
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    by_name: HashMap<String, usize>,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(Error::Truncated(self.pos))?;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or(Error::Truncated(self.pos))?;
        self.pos = end;
        Ok(s)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    fn len(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| Error::Truncated(self.pos))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.len()?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.array::<1>()?[0]),
            1 => Value::I8(self.array::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.array()?)),
            3 => Value::I16(i16::from_le_bytes(self.array()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.array()?)),
            6 => Value::F32(f32::from_le_bytes(self.array()?)),
            7 => Value::Bool(self.array::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let inner = self.u32()?;
                let n = self.len()?;
                let mut items = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    items.push(self.value(inner)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.array()?)),
            12 => Value::F64(f64::from_le_bytes(self.array()?)),
            other => return Err(Error::ValueType(other)),
        })
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the model file is treated as read-only for the life of the process; a concurrent
        // writer would be a user error we cannot defend against with mmap anyway.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self> {
        let mut r = Reader { buf: &mmap, pos: 0 };
        if r.take(4)? != b"GGUF" {
            return Err(Error::BadMagic);
        }
        let version = r.u32()?;
        if version != 3 {
            return Err(Error::Version(version));
        }
        let n_tensors = r.len()?;
        let n_kv = r.len()?;

        let mut metadata = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            metadata.insert(key, r.value(ty)?);
        }

        let mut raw = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.len()?);
            }
            let dtype = DType::from_id(r.u32()?)?;
            let rel = r.len()?;
            raw.push((name, dims, dtype, rel));
        }

        let align = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(32) as usize;
        let data_start = r.pos.div_ceil(align) * align;

        let mut tensors = Vec::with_capacity(n_tensors);
        let mut by_name = HashMap::with_capacity(n_tensors);
        for (name, dims, dtype, rel) in raw {
            let n: usize = dims.iter().product();
            let nbytes = dtype.row_bytes(n);
            let offset = data_start + rel;
            if offset + nbytes > mmap.len() {
                return Err(Error::TensorBounds(name));
            }
            by_name.insert(name.clone(), tensors.len());
            tensors.push(TensorInfo {
                name,
                dims,
                dtype,
                offset,
                nbytes,
            });
        }

        Ok(Gguf {
            mmap,
            version,
            metadata,
            tensors,
            by_name,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.by_name
            .get(name)
            .map(|&i| &self.tensors[i])
            .ok_or_else(|| Error::MissingTensor(name.to_string()))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        &self.mmap[t.offset..t.offset + t.nbytes]
    }

    /// The whole mapping, for zero-copy GPU buffers.
    pub fn mmap(&self) -> &[u8] {
        &self.mmap
    }

    pub fn get(&self, key: &str) -> Result<&Value> {
        self.metadata
            .get(key)
            .ok_or_else(|| Error::MissingKey(key.to_string()))
    }

    pub fn get_u64(&self, key: &str) -> Result<u64> {
        self.get(key)?
            .as_u64()
            .ok_or_else(|| Error::KeyType(key.to_string()))
    }

    pub fn get_f32(&self, key: &str) -> Result<f32> {
        self.get(key)?
            .as_f32()
            .ok_or_else(|| Error::KeyType(key.to_string()))
    }

    pub fn get_bool(&self, key: &str) -> Result<bool> {
        match self.get(key)? {
            Value::Bool(b) => Ok(*b),
            _ => Err(Error::KeyType(key.to_string())),
        }
    }

    pub fn get_str(&self, key: &str) -> Result<&str> {
        match self.get(key)? {
            Value::Str(s) => Ok(s),
            _ => Err(Error::KeyType(key.to_string())),
        }
    }

    pub fn get_array(&self, key: &str) -> Result<&[Value]> {
        match self.get(key)? {
            Value::Array(a) => Ok(a),
            _ => Err(Error::KeyType(key.to_string())),
        }
    }

    /// A scalar or a per-layer array, expanded to `n` entries.
    pub fn get_u64_per_layer(&self, key: &str, n: usize) -> Result<Vec<u64>> {
        match self.get(key)? {
            Value::Array(a) => a
                .iter()
                .map(|v| v.as_u64().ok_or_else(|| Error::KeyType(key.to_string())))
                .collect(),
            v => Ok(vec![
                v.as_u64()
                    .ok_or_else(|| Error::KeyType(key.to_string()))?;
                n
            ]),
        }
    }

    /// Dequantizes a whole tensor to f32 (for small tensors: norms, tables, router).
    pub fn to_f32(&self, t: &TensorInfo) -> Vec<f32> {
        let mut out = vec![0.0f32; t.n_elements()];
        dequantize_row(t.dtype, self.bytes(t), &mut out);
        out
    }
}
