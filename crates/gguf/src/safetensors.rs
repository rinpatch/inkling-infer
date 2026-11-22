//! Minimal safetensors reader for sharded Hugging Face bundles: mmaps every shard, parses the JSON
//! headers, and hands out zero-copy tensor byte slices by name.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::{Error, Result};

/// Element types that occur in the bundles we load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StDType {
    F32,
    F16,
    Bf16,
    U32,
    U8,
    I8,
    U16,
    I32,
    I64,
    /// NVFP4 block scales and MXFP8 values
    F8E4M3,
    /// MXFP8 shared exponents
    F8E8M0,
}

impl StDType {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "F32" => StDType::F32,
            "F16" => StDType::F16,
            "BF16" => StDType::Bf16,
            "U32" => StDType::U32,
            "U8" => StDType::U8,
            "I8" => StDType::I8,
            "U16" => StDType::U16,
            "I32" => StDType::I32,
            "I64" => StDType::I64,
            "F8_E4M3" => StDType::F8E4M3,
            "F8_E8M0" => StDType::F8E8M0,
            _ => return None,
        })
    }

    pub fn size(self) -> usize {
        match self {
            StDType::I64 => 8,
            StDType::F32 | StDType::U32 | StDType::I32 => 4,
            StDType::F16 | StDType::Bf16 | StDType::U16 => 2,
            StDType::U8 | StDType::I8 | StDType::F8E4M3 | StDType::F8E8M0 => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StTensor {
    pub name: String,
    pub dtype: StDType,
    /// Row-major shape (outermost first), as stored.
    pub shape: Vec<usize>,
    pub shard: usize,
    /// Absolute byte offset within the shard file.
    pub offset: usize,
    pub nbytes: usize,
}

pub struct SafeTensors {
    pub dir: PathBuf,
    shards: Vec<Mmap>,
    tensors: HashMap<String, StTensor>,
}

impl SafeTensors {
    /// Opens every `*.safetensors` file in `dir`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        files.sort();
        let mut shards = Vec::new();
        let mut tensors = HashMap::new();
        for (si, path) in files.iter().enumerate() {
            let file = File::open(path)?;
            // SAFETY: model files are treated as read-only for the life of the process.
            let map = unsafe { Mmap::map(&file)? };
            if map.len() < 8 {
                return Err(Error::Truncated(0));
            }
            let n = u64::from_le_bytes(map[..8].try_into().expect("8 bytes")) as usize;
            let header: serde_json::Value = serde_json::from_slice(map.get(8..8 + n).ok_or(Error::Truncated(8))?)
                .map_err(|e| Error::Header(format!("{}: {e}", path.display())))?;
            let base = 8 + n;
            for (name, info) in header.as_object().ok_or_else(|| Error::Header("header is not an object".into()))? {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = info["dtype"].as_str().and_then(StDType::parse).ok_or_else(|| Error::Header(format!("{name}: unsupported dtype {}", info["dtype"])))?;
                let shape: Vec<usize> = info["shape"].as_array().ok_or_else(|| Error::Header(format!("{name}: shape")))?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
                let off = info["data_offsets"].as_array().ok_or_else(|| Error::Header(format!("{name}: offsets")))?;
                let (a, b) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
                if base + b > map.len() || b < a || shape.iter().product::<usize>() * dtype.size() != b - a {
                    return Err(Error::TensorBounds(name.clone()));
                }
                tensors.insert(name.clone(), StTensor { name: name.clone(), dtype, shape, shard: si, offset: base + a, nbytes: b - a });
            }
            shards.push(map);
        }
        Ok(SafeTensors { dir, shards, tensors })
    }

    pub fn get(&self, name: &str) -> Result<&StTensor> {
        self.tensors.get(name).ok_or_else(|| Error::MissingTensor(name.to_string()))
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn bytes(&self, t: &StTensor) -> &[u8] {
        &self.shards[t.shard][t.offset..t.offset + t.nbytes]
    }

    /// The mapping of shard `i`, for zero-copy GPU buffers.
    pub fn shard(&self, i: usize) -> &[u8] {
        &self.shards[i]
    }

    pub fn n_shards(&self) -> usize {
        self.shards.len()
    }

    /// Converts an F32 / F16 / BF16 tensor to f32 (for small tensors: norms, convs, router).
    pub fn to_f32(&self, t: &StTensor) -> Result<Vec<f32>> {
        let b = self.bytes(t);
        Ok(match t.dtype {
            StDType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            StDType::F16 => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            StDType::Bf16 => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            other => return Err(Error::Header(format!("{}: {other:?} is not a float tensor", t.name))),
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(repo: &str) -> Option<PathBuf> {
        let home = PathBuf::from(std::env::var_os("HOME")?);
        let snaps = home.join(format!(".cache/huggingface/hub/models--{repo}/snapshots"));
        std::fs::read_dir(snaps).ok()?.flatten().map(|e| e.path()).next()
    }

    /// Every header in the native and JANGH2 bundles parses and every tensor fits its shard.
    #[test]
    fn opens_downloaded_bundles() {
        for repo in ["thinkingmachines--Inkling-2-Mini", "OsaurusAI--Inkling-2-Mini-JANGH2"] {
            let Some(dir) = snapshot(repo) else {
                eprintln!("SKIP: {repo} not downloaded");
                continue;
            };
            let st = SafeTensors::open(&dir).unwrap();
            let n = st.names().count();
            assert!(n > 1000, "{repo}: only {n} tensors");
            let mut by_dtype = std::collections::BTreeMap::new();
            for name in st.names() {
                *by_dtype.entry(format!("{:?}", st.get(name).unwrap().dtype)).or_insert(0usize) += 1;
            }
            println!("{repo}: {n} tensors in {} shards, {by_dtype:?}", st.n_shards());
        }
    }
}
