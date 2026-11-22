//! Scalar f32 building blocks shared by the CPU reference. Written for clarity, not speed.

use gguf::DType;
use rayon::prelude::*;

/// A quantized matrix stored row-major in the mmap: `rows` rows of `cols` elements each.
/// ggml dims `[cols, rows]` (`ne0` = cols).
#[derive(Debug, Clone, Copy)]
pub struct QMat<'a> {
    pub data: &'a [u8],
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
}

impl<'a> QMat<'a> {
    pub fn row_bytes(&self) -> usize {
        self.dtype.row_bytes(self.cols)
    }

    pub fn row(&self, r: usize) -> &'a [u8] {
        let rb = self.row_bytes();
        &self.data[r * rb..(r + 1) * rb]
    }

    /// Rows `[start, start+n)` as their own matrix (one expert out of a stacked 3-D tensor).
    pub fn slice_rows(&self, start: usize, n: usize) -> QMat<'a> {
        let rb = self.row_bytes();
        QMat {
            data: &self.data[start * rb..(start + n) * rb],
            dtype: self.dtype,
            rows: n,
            cols: self.cols,
        }
    }

    /// `out = W x`
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        out.par_iter_mut()
            .with_min_len(16)
            .enumerate()
            .for_each(|(r, o)| {
                *o = gguf::dot_row(self.dtype, self.row(r), x);
            });
    }

    pub fn matvec_new(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; self.rows];
        self.matvec(x, &mut out);
        out
    }
}

/// `x / sqrt(mean(x²) + eps)`
pub fn rms_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    x.iter().map(|v| v * s).collect()
}

/// `w ⊙ rms_norm(x)`
pub fn rms_norm_w(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut y = rms_norm(x, eps);
    for (a, b) in y.iter_mut().zip(w) {
        *a *= b;
    }
    y
}

/// Scales each `chunk`-sized vector of `x` (after an optional per-element `scale`) to an L2 norm of
/// at most `threshold`.
pub fn clip_heads(x: &mut [f32], chunk: usize, scale: Option<&[f32]>, threshold: f32) {
    if let Some(s) = scale {
        for (a, b) in x.iter_mut().zip(s) {
            *a *= b;
        }
    }
    for h in x.chunks_mut(chunk) {
        let norm = h.iter().map(|v| v * v).sum::<f32>().sqrt();
        let f = (threshold / (norm + 1e-6)).min(1.0);
        for v in h {
            *v *= f;
        }
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn softcap(x: f32, cap: f32) -> f32 {
    cap * (x / cap).tanh()
}

/// In-place softmax.
pub fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Depth attention ("attention residuals"): each query row in `queries` (`n_q × d`) attends over
/// the RMS-normalized stack `[blocks..., cur]` and returns the softmax-weighted sum of the
/// *normalized* states. Returns `n_q` vectors of length `d`.
pub fn depth_mix(blocks: &[Vec<f32>], cur: &[f32], queries: &[f32], d: usize) -> Vec<Vec<f32>> {
    let states: Vec<Vec<f32>> = blocks
        .iter()
        .map(|b| rms_norm(b, 1e-6))
        .chain(std::iter::once(rms_norm(cur, 1e-6)))
        .collect();
    let scale = 1.0 / (d as f32).sqrt();
    queries
        .chunks_exact(d)
        .map(|q| {
            let mut p: Vec<f32> = states
                .iter()
                .map(|s| scale * s.iter().zip(q).map(|(a, b)| a * b).sum::<f32>())
                .collect();
            softmax(&mut p);
            let mut out = vec![0.0; d];
            for (s, w) in states.iter().zip(&p) {
                for (o, v) in out.iter_mut().zip(s) {
                    *o += w * v;
                }
            }
            out
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_leaves_short_vectors_and_caps_long_ones() {
        let mut x = vec![3.0, 4.0, 0.3, 0.4];
        clip_heads(&mut x, 2, None, 1.0);
        assert!((x[0] - 0.6).abs() < 1e-5 && (x[1] - 0.8).abs() < 1e-5);
        assert!((x[2] - 0.3).abs() < 1e-6 && (x[3] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn depth_mix_with_zero_query_averages_normalized_states() {
        let blocks = vec![vec![2.0, 0.0]];
        let cur = vec![0.0, 4.0];
        let out = depth_mix(&blocks, &cur, &[0.0, 0.0], 2);
        // rms([2,0]) = [√2, 0], rms([0,4]) = [0, √2]; equal weights
        let r = 2f32.sqrt() / 2.0;
        assert!((out[0][0] - r).abs() < 1e-4 && (out[0][1] - r).abs() < 1e-4);
    }
}
