//! Typed views of the GGUF tensors. Small tensors (norms, queries, conv taps, router, loop gates)
//! are copied out as f32; large matrices stay quantized in the mmap as `QMat`s.

use gguf::{DType, Gguf, TensorInfo};

use crate::ops::QMat;
use crate::{Error, Hparams, Result};

pub struct LayerWeights<'a> {
    pub is_swa: bool,
    // attention projections (Q8_0 in the IQ4_NL file)
    pub wq: QMat<'a>,
    pub wk: QMat<'a>,
    pub wv: QMat<'a>,
    pub wo: QMat<'a>,
    pub w_gate: QMat<'a>,
    pub w_rel: QMat<'a>,
    pub w_sink: QMat<'a>,
    /// Full layers: per-token dynamic conv taps, `[n_head_kv * n_conv]` rows.
    pub w_k_conv_proj: Option<QMat<'a>>,
    pub w_v_conv_proj: Option<QMat<'a>>,
    // norms: SWA layers share `attn_norm` for q/k/v; full layers have one per input
    pub attn_norm: Option<Vec<f32>>,
    pub q_inp_norm: Option<Vec<f32>>,
    pub k_inp_norm: Option<Vec<f32>>,
    pub v_inp_norm: Option<Vec<f32>>,
    pub q_clip: Vec<f32>,
    pub rel_clip: Vec<f32>,
    /// `[extent][n_rel]`: `rel_bias[δ*n_rel + i]`
    pub rel_bias: Vec<f32>,
    pub rel_extent: usize,
    /// Static conv taps `[channel][n_conv]` in ggml_ssm_conv order (last tap = current token).
    pub k_conv: Vec<f32>,
    pub v_conv: Vec<f32>,
    pub attn_post_conv: Vec<f32>,
    pub ffn_post_conv: Vec<f32>,
    /// Depth-attention queries; `None` on layer 0. 1 row on SWA layers, 3 (q, k, v) on full layers.
    pub attn_res_query: Option<Vec<f32>>,
    pub ffn_res_query: Vec<f32>,
    // MoE
    pub ffn_norm: Vec<f32>,
    /// Router `[n_expert][n_embd]` (F32).
    pub router: Vec<f32>,
    pub exp_probs_b: Vec<f32>,
    /// `[n_expert * 2*n_ff]` rows of `n_embd`; per expert, gate rows then up rows.
    pub gate_up_exps: QMat<'a>,
    /// `[n_expert * n_embd]` rows of `n_ff`.
    pub down_exps: QMat<'a>,
}

/// The cross-pass message gate: `(1 − tanh(F z))⊙x + tanh(G z)⊙m`, `z = D [rms(x); rms(m)]`.
pub struct MessageGate {
    /// `[rank][2*n_embd]`
    pub down: Vec<f32>,
    /// `[n_embd][rank]`
    pub forget: Vec<f32>,
    /// `[n_embd][rank]`
    pub gain: Vec<f32>,
}

pub struct LoopWeights {
    /// `[4][n_embd]`: queries for the q, k, v and ffn messages.
    pub msg_query: Vec<f32>,
    /// Gates for q, k, v, ffn, in that order.
    pub gates: [MessageGate; 4],
}

pub struct Weights<'a> {
    pub tok_embd: QMat<'a>,
    pub tok_norm: Vec<f32>,
    pub output: QMat<'a>,
    pub output_norm: Vec<f32>,
    pub output_res_query: Vec<f32>,
    pub layers: Vec<LayerWeights<'a>>,
    /// Indexed by `il - loop_start`.
    pub loops: Vec<LoopWeights>,
}

struct Loader<'a> {
    g: &'a Gguf,
}

impl<'a> Loader<'a> {
    fn info(&self, name: &str, dims: &[usize]) -> Result<&'a TensorInfo> {
        let t = self.g.tensor(name)?;
        if t.dims != dims {
            return Err(Error::Shape {
                name: name.into(),
                got: t.dims.clone(),
                want: dims.to_vec(),
            });
        }
        Ok(t)
    }

    /// Matrix with ggml dims `[cols, rows, (stack)]`; stacked dims are folded into rows.
    fn mat(&self, name: &str, dims: &[usize]) -> Result<QMat<'a>> {
        let t = self.info(name, dims)?;
        Ok(QMat {
            data: self.g.bytes(t),
            dtype: t.dtype,
            rows: dims[1..].iter().product(),
            cols: dims[0],
        })
    }

    fn f32(&self, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
        let t = self.info(name, dims)?;
        if !matches!(t.dtype, DType::F32 | DType::F16 | DType::Bf16) {
            return Err(Error::Unsupported(format!(
                "{name}: expected a float tensor, got {:?}",
                t.dtype
            )));
        }
        Ok(self.g.to_f32(t))
    }

    fn opt_f32(&self, name: &str, dims: &[usize]) -> Result<Option<Vec<f32>>> {
        if self.g.has_tensor(name) {
            self.f32(name, dims).map(Some)
        } else {
            Ok(None)
        }
    }
}

impl<'a> Weights<'a> {
    pub fn load(g: &'a Gguf, hp: &Hparams) -> Result<Self> {
        let l = Loader { g };
        let d = hp.n_embd;
        let (qd, kvd, nc, nr) = (hp.q_dim(), hp.kv_dim(), hp.n_conv, hp.n_rel);

        let mut layers = Vec::with_capacity(hp.n_layer);
        for i in 0..hp.n_layer {
            let b = |s: &str| format!("blk.{i}.{s}");
            let is_swa = hp.is_swa[i];
            let extent = if is_swa { hp.n_swa } else { hp.rel_extent };
            let ff = hp.n_ff_exp;
            let ne = hp.n_expert;
            let res_dims = if is_swa { vec![d] } else { vec![d, 3] };
            layers.push(LayerWeights {
                is_swa,
                wq: l.mat(&b("attn_q.weight"), &[d, qd])?,
                wk: l.mat(&b("attn_k.weight"), &[d, kvd])?,
                wv: l.mat(&b("attn_v.weight"), &[d, kvd])?,
                wo: l.mat(&b("attn_output.weight"), &[qd, d])?,
                w_gate: l.mat(&b("attn_gate.weight"), &[d, qd])?,
                w_rel: l.mat(&b("attn_rel.weight"), &[d, nr * hp.n_head])?,
                w_sink: l.mat(&b("attn_sink_proj.weight"), &[d, hp.n_head])?,
                w_k_conv_proj: if is_swa {
                    None
                } else {
                    Some(l.mat(&b("attn_k_conv_proj.weight"), &[d, nc * hp.n_head_kv])?)
                },
                w_v_conv_proj: if is_swa {
                    None
                } else {
                    Some(l.mat(&b("attn_v_conv_proj.weight"), &[d, nc * hp.n_head_kv])?)
                },
                attn_norm: if is_swa {
                    Some(l.f32(&b("attn_norm.weight"), &[d])?)
                } else {
                    None
                },
                q_inp_norm: l.opt_f32(&b("attn_q_inp_norm.weight"), &[d])?,
                k_inp_norm: l.opt_f32(&b("attn_k_inp_norm.weight"), &[d])?,
                v_inp_norm: l.opt_f32(&b("attn_v_inp_norm.weight"), &[d])?,
                q_clip: l.f32(&b("attn_q_clip.weight"), &[hp.head_dim, hp.n_head])?,
                rel_clip: l.f32(&b("attn_rel_clip.weight"), &[nr, hp.n_head])?,
                rel_bias: l.f32(&b("attn_rel_bias.weight"), &[nr, extent])?,
                rel_extent: extent,
                k_conv: l.f32(&b("attn_k_conv.weight"), &[nc, kvd])?,
                v_conv: l.f32(&b("attn_v_conv.weight"), &[nc, kvd])?,
                attn_post_conv: l.f32(&b("attn_post_conv.weight"), &[nc, d])?,
                ffn_post_conv: l.f32(&b("ffn_post_conv.weight"), &[nc, d])?,
                attn_res_query: if i == 0 {
                    None
                } else {
                    Some(l.f32(&b("attn_res_query.weight"), &res_dims)?)
                },
                ffn_res_query: l.f32(&b("ffn_res_query.weight"), &[d])?,
                ffn_norm: l.f32(&b("ffn_norm.weight"), &[d])?,
                router: l.f32(&b("ffn_gate_inp.weight"), &[d, ne])?,
                exp_probs_b: l
                    .opt_f32(&b("exp_probs_b.bias"), &[ne])?
                    .unwrap_or_else(|| vec![0.0; ne]),
                gate_up_exps: l.mat(&b("ffn_gate_up_exps.weight"), &[d, 2 * ff, ne])?,
                down_exps: l.mat(&b("ffn_down_exps.weight"), &[ff, d, ne])?,
            });
            if !is_swa
                && (layers[i].q_inp_norm.is_none()
                    || layers[i].k_inp_norm.is_none()
                    || layers[i].v_inp_norm.is_none())
            {
                return Err(Error::Unsupported(format!(
                    "full-attention layer {i} lacks q/k/v input norms"
                )));
            }
        }

        let rank = hp.loop_gate_rank;
        let mut loops = Vec::with_capacity(hp.n_loop());
        for i in hp.loop_start..hp.loop_end {
            let gate = |c: &str| -> Result<MessageGate> {
                Ok(MessageGate {
                    down: l.f32(
                        &format!("blk.{i}.loop_{c}_ctrl_down.weight"),
                        &[2 * d, rank],
                    )?,
                    forget: l.f32(&format!("blk.{i}.loop_{c}_ctrl_forget.weight"), &[rank, d])?,
                    gain: l.f32(&format!("blk.{i}.loop_{c}_ctrl_gain.weight"), &[rank, d])?,
                })
            };
            loops.push(LoopWeights {
                msg_query: l.f32(&format!("blk.{i}.loop_msg_query.weight"), &[d, 4])?,
                gates: [gate("q")?, gate("k")?, gate("v")?, gate("ffn")?],
            });
        }

        Ok(Weights {
            tok_embd: l.mat("token_embd.weight", &[d, hp.n_vocab_padded])?,
            tok_norm: l.f32("token_embd_norm.weight", &[d])?,
            output: l.mat("output.weight", &[d, hp.n_vocab_padded])?,
            output_norm: l.f32("output_norm.weight", &[d])?,
            output_res_query: l.f32("output_res_query.weight", &[d])?,
            layers,
            loops,
        })
    }
}
