//! Inkling 2 Mini (`inkling2_loop`) inference. `cpu` is the f32 reference implementation that
//! every faster backend is tested against.

pub mod cpu;
pub mod drafter;
pub mod hparams;
pub mod ops;
pub mod tokenizer;
pub mod weights;

pub use hparams::Hparams;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Gguf(#[from] gguf::Error),
    #[error("unsupported model: {0}")]
    Unsupported(String),
    #[error("tensor {name} has shape {got:?}, expected {want:?}")]
    Shape {
        name: String,
        got: Vec<usize>,
        want: Vec<usize>,
    },
    #[error("context full: position {0} exceeds the cache capacity")]
    ContextFull(usize),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Receives named intermediate tensors during a forward step. Names and slot numbers follow
/// llama.cpp's debug names (`l_out-30` = slot 30) so they can be diffed against its fixtures.
pub trait Trace {
    fn wants(&self, _name: &str, _slot: Option<usize>) -> bool {
        false
    }
    fn record(&mut self, _name: &str, _slot: Option<usize>, _data: &[f32]) {}
}

/// A trace that records nothing.
pub struct NoTrace;
impl Trace for NoTrace {}
