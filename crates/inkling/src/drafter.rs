//! CPU reference for the DSpark drafter bundled with Inkling 2 Mini (`mtp.safetensors`), in plain
//! f32. Design and sources: docs/dspark.md. It is the oracle for the Metal drafter and lets us
//! measure acceptance on the GGUF model's features before any GPU work.
//!
//! The reference implementations round to BF16 at several points; this one stays in f32, so its
//! drafts can differ from theirs on near-ties. Acceptance, not bit-exactness, is what matters.

use gguf::safetensors::SafeTensors;
use rayon::prelude::*;

use crate::ops::{rms_norm_w, softcap, softmax};
use crate::{Error, Result};

pub const D: usize = 2304;
pub const N_LAYER: usize = 3;
pub const N_HEAD: usize = 24;
pub const N_KV: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const N_FF: usize = 9216;
pub const EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 1_000_000.0;
pub const WINDOW: usize = 1024;
pub const BLOCK: usize = 8;
pub const RANK: usize = 256;
pub const SOFTCAP: f32 = 30.0;
/// Target features read by the drafter, in layer-visit numbering: inputs of visits 5, 23, 41 and
/// the final normed hidden (48). In our cache-slot numbering: slots 5, 29, 47, then the output.
pub const FEATURE_SLOTS: [usize; 3] = [5, 29, 47];
pub const N_FEAT: usize = 4;
const LOCAL: [bool; N_LAYER] = [true, false, true];

/// Row-major dense f32 matrix.
pub struct Dense {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<f32>,
}

impl Dense {
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols);
        (0..self.rows)
            .into_par_iter()
            .with_min_len(16)
            .map(|r| {
                self.w[r * self.cols..(r + 1) * self.cols]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }
}

pub struct DLayer {
    pub q: Dense,
    pub k: Dense,
    pub v: Dense,
    pub o: Dense,
    pub k_ctx: Dense,
    pub v_ctx: Dense,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub k_norm_ctx: Vec<f32>,
    pub input_ln: Vec<f32>,
    pub post_ln: Vec<f32>,
    pub dflare_norm: Vec<f32>,
    pub gate: Dense,
    pub up: Dense,
    pub down: Dense,
}

pub struct Drafter {
    pub layers: Vec<DLayer>,
    pub norm: Vec<f32>,
    pub mask_embed: Vec<f32>,
    /// softmax(fuse_weight) per row: rows 0..3 the drafter layers, row 3 the Markov head.
    pub fuse_alpha: [[f32; N_FEAT]; N_LAYER + 1],
    /// Markov token embedding [vocab, RANK].
    pub markov_w1: Dense,
    pub rnn_proj: Dense,
    pub rnn_bias: Vec<f32>,
    pub hidden_out: Dense,
    pub hidden_out_bias: Vec<f32>,
    pub markov_norm: Vec<f32>,
}

/// Per-sequence draft state: context K/V per drafter layer for every committed position, and the
/// features of the last committed position (for the Markov head).
#[derive(Default)]
pub struct DraftState {
    /// [layer][position] -> N_KV * HEAD_DIM, keys already rotated
    pub ctx_k: [Vec<Vec<f32>>; N_LAYER],
    pub ctx_v: [Vec<Vec<f32>>; N_LAYER],
    pub last_feat: Vec<f32>,
}

impl DraftState {
    pub fn len(&self) -> usize {
        self.ctx_k[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops committed positions past `n` (after a rejected draft).
    pub fn truncate(&mut self, n: usize) {
        for l in 0..N_LAYER {
            self.ctx_k[l].truncate(n);
            self.ctx_v[l].truncate(n);
        }
    }
}

fn get_vec(st: &SafeTensors, name: &str, len: usize) -> Result<Vec<f32>> {
    let t = st.get(name)?;
    if t.shape.iter().product::<usize>() != len {
        return Err(Error::Shape {
            name: name.into(),
            got: t.shape.clone(),
            want: vec![len],
        });
    }
    Ok(st.to_f32(t)?)
}

fn get_mat(st: &SafeTensors, name: &str, rows: usize, cols: usize) -> Result<Dense> {
    let t = st.get(name)?;
    if t.shape != [rows, cols] {
        return Err(Error::Shape {
            name: name.into(),
            got: t.shape.clone(),
            want: vec![rows, cols],
        });
    }
    Ok(Dense {
        rows,
        cols,
        w: st.to_f32(t)?,
    })
}

/// Neox (half-split) rotary embedding over one head at absolute position `pos`.
pub fn rope(x: &mut [f32], pos: usize) {
    let half = x.len() / 2;
    for i in 0..half {
        let theta = pos as f64 * (ROPE_THETA as f64).powf(-2.0 * i as f64 / x.len() as f64);
        let (s, c) = theta.sin_cos();
        let (a, b) = (x[i] as f64, x[i + half] as f64);
        x[i] = (a * c - b * s) as f32;
        x[i + half] = (a * s + b * c) as f32;
    }
}

fn norm_heads(x: &mut [f32], w: &[f32]) {
    for h in x.chunks_mut(HEAD_DIM) {
        let n = rms_norm_w(h, w, EPS);
        h.copy_from_slice(&n);
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl Drafter {
    /// Loads `mtp.safetensors` from the native checkpoint directory.
    pub fn load(st: &SafeTensors, n_vocab_padded: usize) -> Result<Self> {
        let mut layers = Vec::new();
        for l in 0..N_LAYER {
            let p = |s: &str| format!("layers.{l}.{s}");
            let (qd, kvd) = (N_HEAD * HEAD_DIM, N_KV * HEAD_DIM);
            layers.push(DLayer {
                q: get_mat(st, &p("self_attn.q_proj.weight"), qd, D)?,
                k: get_mat(st, &p("self_attn.k_proj.weight"), kvd, D)?,
                v: get_mat(st, &p("self_attn.v_proj.weight"), kvd, D)?,
                o: get_mat(st, &p("self_attn.o_proj.weight"), D, qd)?,
                k_ctx: get_mat(st, &p("self_attn.k_proj_ctx.weight"), kvd, D)?,
                v_ctx: get_mat(st, &p("self_attn.v_proj_ctx.weight"), kvd, D)?,
                q_norm: get_vec(st, &p("self_attn.q_norm.weight"), HEAD_DIM)?,
                k_norm: get_vec(st, &p("self_attn.k_norm.weight"), HEAD_DIM)?,
                k_norm_ctx: get_vec(st, &p("self_attn.k_norm_ctx.weight"), HEAD_DIM)?,
                input_ln: get_vec(st, &p("input_layernorm.weight"), D)?,
                post_ln: get_vec(st, &p("post_attention_layernorm.weight"), D)?,
                dflare_norm: get_vec(st, &p("dflare_norm.weight"), D)?,
                gate: get_mat(st, &p("mlp.gate_proj.weight"), N_FF, D)?,
                up: get_mat(st, &p("mlp.up_proj.weight"), N_FF, D)?,
                down: get_mat(st, &p("mlp.down_proj.weight"), D, N_FF)?,
            });
        }
        let fw = get_vec(st, "fuse_weight", (N_LAYER + 1) * N_FEAT)?;
        let mut fuse_alpha = [[0.0; N_FEAT]; N_LAYER + 1];
        for (r, row) in fuse_alpha.iter_mut().enumerate() {
            row.copy_from_slice(&fw[r * N_FEAT..(r + 1) * N_FEAT]);
            softmax(row);
        }
        Ok(Drafter {
            layers,
            norm: get_vec(st, "norm.weight", D)?,
            mask_embed: get_vec(st, "mask_embed", D)?,
            fuse_alpha,
            markov_w1: get_mat(st, "markov_head.markov_w1.weight", n_vocab_padded, RANK)?,
            rnn_proj: get_mat(
                st,
                "markov_head.rnn_proj.weight",
                3 * RANK,
                2 * RANK + 2 * D,
            )?,
            rnn_bias: get_vec(st, "markov_head.rnn_proj.bias", 3 * RANK)?,
            hidden_out: get_mat(st, "markov_head.hidden_out.weight", D, RANK)?,
            hidden_out_bias: get_vec(st, "markov_head.hidden_out.bias", D)?,
            markov_norm: get_vec(st, "markov_head.dflare_norm.weight", D)?,
        })
    }

    /// `rmsnorm(Σ_t alpha[row][t] · feat_t) · w` over the 4 target features `[N_FEAT * D]`.
    fn mix(&self, feat: &[f32], row: usize, w: &[f32]) -> Vec<f32> {
        let mut m = vec![0.0f32; D];
        for (t, a) in self.fuse_alpha[row].iter().enumerate() {
            for (o, x) in m.iter_mut().zip(&feat[t * D..(t + 1) * D]) {
                *o += a * x;
            }
        }
        rms_norm_w(&m, w, EPS)
    }

    /// Commits one position: appends its context K/V to every drafter layer and remembers its
    /// features for the Markov head. `feat` = the target's 4 features for that position.
    pub fn commit(&self, st: &mut DraftState, feat: &[f32]) {
        assert_eq!(feat.len(), N_FEAT * D);
        let pos = st.len();
        for (l, layer) in self.layers.iter().enumerate() {
            let c = self.mix(feat, l, &layer.dflare_norm);
            let mut k = layer.k_ctx.matvec(&c);
            norm_heads(&mut k, &layer.k_norm_ctx);
            for h in k.chunks_mut(HEAD_DIM) {
                rope(h, pos);
            }
            st.ctx_k[l].push(k);
            st.ctx_v[l].push(layer.v_ctx.matvec(&c));
        }
        st.last_feat = feat.to_vec();
    }

    /// Post-norm drafter hidden for the block at positions P..P+BLOCK (P = committed length),
    /// rows 1..BLOCK. `anchor_emb` = the target's embed_norm(embed(anchor)).
    pub fn block_hidden(&self, st: &DraftState, anchor_emb: &[f32]) -> Vec<Vec<f32>> {
        let p0 = st.len();
        let mut res: Vec<Vec<f32>> = (0..BLOCK)
            .map(|i| {
                if i == 0 {
                    anchor_emb.to_vec()
                } else {
                    self.mask_embed.clone()
                }
            })
            .collect();
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let group = N_HEAD / N_KV;
        for (l, layer) in self.layers.iter().enumerate() {
            let a0: Vec<Vec<f32>> = res
                .iter()
                .map(|r| rms_norm_w(r, &layer.input_ln, EPS))
                .collect();
            let mut qs = Vec::new();
            let mut ks = Vec::new();
            let mut vs = Vec::new();
            for (i, x) in a0.iter().enumerate() {
                let mut q = layer.q.matvec(x);
                norm_heads(&mut q, &layer.q_norm);
                let mut k = layer.k.matvec(x);
                norm_heads(&mut k, &layer.k_norm);
                for h in q.chunks_mut(HEAD_DIM).chain(k.chunks_mut(HEAD_DIM)) {
                    rope(h, p0 + i);
                }
                qs.push(q);
                ks.push(k);
                vs.push(layer.v.matvec(x));
            }
            // keys: windowed context, then the block itself; no mask (bidirectional in the block)
            let lo = if LOCAL[l] {
                p0.saturating_sub(WINDOW)
            } else {
                0
            };
            let keys: Vec<&Vec<f32>> = st.ctx_k[l][lo..p0].iter().chain(ks.iter()).collect();
            let vals: Vec<&Vec<f32>> = st.ctx_v[l][lo..p0].iter().chain(vs.iter()).collect();
            for (i, q) in qs.iter().enumerate() {
                let mut o = vec![0.0f32; N_HEAD * HEAD_DIM];
                for h in 0..N_HEAD {
                    let kh = h / group;
                    let qh = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                    let mut s: Vec<f32> = keys
                        .iter()
                        .map(|k| {
                            qh.iter()
                                .zip(&k[kh * HEAD_DIM..(kh + 1) * HEAD_DIM])
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                * scale
                        })
                        .collect();
                    softmax(&mut s);
                    let oh = &mut o[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                    for (w, v) in s.iter().zip(&vals) {
                        for (a, b) in oh.iter_mut().zip(&v[kh * HEAD_DIM..(kh + 1) * HEAD_DIM]) {
                            *a += w * b;
                        }
                    }
                }
                let ao = layer.o.matvec(&o);
                for (r, a) in res[i].iter_mut().zip(ao) {
                    *r += a;
                }
            }
            for r in res.iter_mut() {
                let x = rms_norm_w(r, &layer.post_ln, EPS);
                let (g, u) = (layer.gate.matvec(&x), layer.up.matvec(&x));
                let act: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
                for (a, b) in r.iter_mut().zip(layer.down.matvec(&act)) {
                    *a += b;
                }
            }
        }
        res[1..]
            .iter()
            .map(|r| rms_norm_w(r, &self.norm, EPS))
            .collect()
    }

    /// Greedy Markov walk: up to BLOCK-1 draft tokens after `anchor`. `lm_head(x)` returns the
    /// target's raw (pre-softcap) logits over the unpadded vocabulary.
    pub fn walk(
        &self,
        st: &DraftState,
        h: &[Vec<f32>],
        anchor: u32,
        lm_head: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> Vec<u32> {
        self.walk_topk(st, h, anchor, lm_head, None)
    }

    /// `walk`, optionally choosing each draft only among the `k` best tokens of the base logits
    /// (the Markov correction is then needed for k tokens instead of the whole vocabulary).
    pub fn walk_topk(
        &self,
        st: &DraftState,
        h: &[Vec<f32>],
        anchor: u32,
        lm_head: &dyn Fn(&[f32]) -> Vec<f32>,
        k: Option<usize>,
    ) -> Vec<u32> {
        let f_rnn = self.mix(&st.last_feat, N_LAYER, &self.markov_norm);
        let mut state = vec![0.0f32; RANK];
        let mut prev = anchor as usize;
        let mut out = Vec::new();
        for hs in h {
            let mut xin = state.clone();
            xin.extend_from_slice(&self.markov_w1.w[prev * RANK..(prev + 1) * RANK]);
            xin.extend_from_slice(hs);
            xin.extend_from_slice(&f_rnn);
            let mut p = self.rnn_proj.matvec(&xin);
            for (a, b) in p.iter_mut().zip(&self.rnn_bias) {
                *a += b;
            }
            for i in 0..RANK {
                let g = 1.0 / (1.0 + (-p[i]).exp());
                state[i] = g * state[i] + (1.0 - g) * p[RANK + i].tanh();
            }
            let latent: Vec<f32> = p[2 * RANK..].iter().map(|x| x.tanh()).collect();
            let mut hid = self.hidden_out.matvec(&latent);
            for (a, b) in hid.iter_mut().zip(&self.hidden_out_bias) {
                *a += b;
            }
            let (base, bias) = (lm_head(hs), lm_head(&hid));
            let cand: Vec<usize> = match k {
                None => (0..base.len()).collect(),
                Some(k) => {
                    let mut idx: Vec<usize> = (0..base.len()).collect();
                    idx.select_nth_unstable_by(k, |a, b| base[*b].total_cmp(&base[*a]));
                    idx.truncate(k);
                    idx
                }
            };
            let mut best = cand[0];
            let mut best_v = f32::NEG_INFINITY;
            for &i in &cand {
                let v = softcap(base[i] + bias[i], SOFTCAP);
                if v > best_v || (v == best_v && i < best) {
                    best_v = v;
                    best = i;
                }
            }
            out.push(best as u32);
            prev = best;
        }
        out
    }
}
