//! f32 CPU reference forward pass, one token per step. It mirrors docs/reference/ARCHITECTURE.md
//! line by line and exists to be obviously correct: the Metal backend is tested against it.

use rayon::prelude::*;

use crate::ops::{clip_heads, depth_mix, rms_norm, rms_norm_w, sigmoid, softcap, softmax};
use crate::weights::{LayerWeights, MessageGate, Weights};
use crate::{Hparams, Result, Trace};

/// Per-slot state. KV rows are stored densely by position (row `p` = token at position `p`).
/// SWA slots keep every row too; attention just skips rows outside the window. That wastes memory
/// on long prompts but keeps the reference free of ring-buffer indexing.
struct Slot {
    k: Vec<f32>,
    v: Vec<f32>,
    /// Conv histories for k, v, post-attention, post-ffn: `[n_conv-1][channels]`, oldest first.
    conv: [Vec<f32>; 4],
}

pub struct State {
    slots: Vec<Slot>,
    pos: usize,
}

impl State {
    pub fn new(hp: &Hparams) -> Self {
        let h = hp.n_conv - 1;
        let slots = (0..hp.n_slots())
            .map(|_| Slot {
                k: Vec::new(),
                v: Vec::new(),
                conv: [
                    vec![0.0; h * hp.kv_dim()],
                    vec![0.0; h * hp.kv_dim()],
                    vec![0.0; h * hp.n_embd],
                    vec![0.0; h * hp.n_embd],
                ],
            })
            .collect();
        State { slots, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }
}

const CONV_K: usize = 0;
const CONV_V: usize = 1;
const CONV_ATTN: usize = 2;
const CONV_FFN: usize = 3;

pub struct CpuModel<'a> {
    pub hp: Hparams,
    pub w: Weights<'a>,
}

impl<'a> CpuModel<'a> {
    pub fn new(hp: Hparams, w: Weights<'a>) -> Self {
        CpuModel { hp, w }
    }

    /// Causal short convolution with a residual, for one token.
    /// `taps` is `[channel][n_conv]` in ggml_ssm_conv order (tap `n_conv-1` multiplies the current
    /// token). `dynamic` adds per-token taps `[head][n_conv]` indexed by backward lag, shared by
    /// the `channels/heads` channels of each head.
    fn convolve(
        &self,
        x: &[f32],
        hist: &mut [f32],
        taps: &[f32],
        dynamic: Option<&[f32]>,
    ) -> Vec<f32> {
        let nc = self.hp.n_conv;
        let ch = x.len();
        // value at backward lag l: l = 0 is x, l >= 1 comes from the history (oldest first)
        let at = |c: usize, l: usize| {
            if l == 0 {
                x[c]
            } else {
                hist[(nc - 1 - l) * ch + c]
            }
        };
        let n_heads = dynamic.map(|d| d.len() / nc).unwrap_or(1);
        let per_head = ch / n_heads;
        let scale = 1.0 / (((if dynamic.is_some() { 2 } else { 1 }) * nc + 1) as f32).sqrt();
        let y = (0..ch)
            .map(|c| {
                let mut acc = 0.0;
                for l in 0..nc {
                    acc += taps[c * nc + (nc - 1 - l)] * at(c, l);
                }
                if let Some(d) = dynamic {
                    let h = c / per_head;
                    for l in 0..nc {
                        acc += d[h * nc + l] * at(c, l);
                    }
                }
                (x[c] + acc) * scale
            })
            .collect();
        hist.copy_within(ch.., 0);
        hist[(nc - 2) * ch..].copy_from_slice(x);
        y
    }

    fn message_gate(&self, x: &[f32], m: &[f32], g: &MessageGate) -> Vec<f32> {
        let d = self.hp.n_embd;
        let rank = self.hp.loop_gate_rank;
        let mut inp = rms_norm(x, 1e-6);
        inp.extend(rms_norm(m, 1e-6));
        let z: Vec<f32> = (0..rank)
            .map(|r| {
                g.down[r * 2 * d..(r + 1) * 2 * d]
                    .iter()
                    .zip(&inp)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
        (0..d)
            .map(|c| {
                let f = g.forget[c * rank..(c + 1) * rank]
                    .iter()
                    .zip(&z)
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    .tanh();
                let gn = g.gain[c * rank..(c + 1) * rank]
                    .iter()
                    .zip(&z)
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    .tanh();
                (1.0 - f) * x[c] + gn * m[c]
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        lw: &LayerWeights,
        slot: &mut Slot,
        n_q: &[f32],
        n_k: &[f32],
        n_v: &[f32],
        pos: usize,
        sid: usize,
        tr: &mut dyn Trace,
    ) -> Vec<f32> {
        let hp = &self.hp;
        let (hd, nh, nkv, nr, nc) = (hp.head_dim, hp.n_head, hp.n_head_kv, hp.n_rel, hp.n_conv);
        let s = Some(sid);

        let mut q = lw.wq.matvec_new(n_q);
        clip_heads(&mut q, hd, Some(&lw.q_clip), hp.qk_clip);

        let k_dyn = lw.w_k_conv_proj.map(|w| w.matvec_new(n_k));
        let mut k = self.convolve(
            &lw.wk.matvec_new(n_k),
            &mut slot.conv[CONV_K],
            &lw.k_conv,
            k_dyn.as_deref(),
        );
        clip_heads(&mut k, hd, None, hp.qk_clip);

        let v_dyn = lw.w_v_conv_proj.map(|w| w.matvec_new(n_v));
        let v_lin = self.convolve(
            &lw.wv.matvec_new(n_v),
            &mut slot.conv[CONV_V],
            &lw.v_conv,
            v_dyn.as_deref(),
        );
        let v: Vec<f32> = v_lin
            .chunks(hd)
            .flat_map(|h| rms_norm(h, hp.rms_eps))
            .collect();

        let mut r = lw.w_rel.matvec_new(n_q);
        clip_heads(&mut r, nr, Some(&lw.rel_clip), hp.rel_clip);

        if !lw.is_swa {
            let tau = hp.tau(pos);
            q.iter_mut().for_each(|x| *x *= tau);
            r.iter_mut().for_each(|x| *x *= tau);
        }
        debug_assert_eq!(nc, hp.n_conv);
        for (name, t) in [("Qcur", &q), ("Kcur", &k), ("Vcur", &v), ("Rcur", &r)] {
            if tr.wants(name, s) {
                tr.record(name, s, t);
            }
        }

        assert_eq!(
            slot.k.len(),
            pos * hp.kv_dim(),
            "slot {sid} is out of sync with the position"
        );
        slot.k.extend_from_slice(&k);
        slot.v.extend_from_slice(&v);

        let sinks = lw.w_sink.matvec_new(n_q);
        let first = if lw.is_swa {
            (pos + 1).saturating_sub(hp.n_swa)
        } else {
            0
        };
        let scale = 1.0 / (hd as f32).sqrt();
        let grp = nh / nkv;
        let kvd = hp.kv_dim();

        let mut o = vec![0.0f32; nh * hd];
        o.par_chunks_mut(hd).enumerate().for_each(|(h, oh)| {
            let qh = &q[h * hd..(h + 1) * hd];
            let rh = &r[h * nr..(h + 1) * nr];
            let kvh = h / grp;
            // logits for keys first..=pos, then the sink as the last entry (it has no value)
            let mut p: Vec<f32> = (first..=pos)
                .map(|j| {
                    let kj = &slot.k[j * kvd + kvh * hd..j * kvd + (kvh + 1) * hd];
                    let dot: f32 = qh.iter().zip(kj).map(|(a, b)| a * b).sum();
                    let dist = pos - j;
                    let bias = if dist < lw.rel_extent {
                        rh.iter()
                            .zip(&lw.rel_bias[dist * nr..(dist + 1) * nr])
                            .map(|(a, b)| a * b)
                            .sum()
                    } else {
                        0.0
                    };
                    dot * scale + bias
                })
                .collect();
            p.push(sinks[h]);
            softmax(&mut p);
            for (j, &pj) in (first..=pos).zip(&p) {
                let vj = &slot.v[j * kvd + kvh * hd..j * kvd + (kvh + 1) * hd];
                for (a, b) in oh.iter_mut().zip(vj) {
                    *a += pj * b;
                }
            }
        });
        if tr.wants("kqv_out", s) {
            tr.record("kqv_out", s, &o);
        }

        let gate = lw.w_gate.matvec_new(n_q);
        for (a, g) in o.iter_mut().zip(&gate) {
            *a *= sigmoid(*g);
        }
        let out = lw.wo.matvec_new(&o);
        self.convolve(&out, &mut slot.conv[CONV_ATTN], &lw.attn_post_conv, None)
    }

    fn moe(&self, lw: &LayerWeights, n: &[f32], sid: usize, tr: &mut dyn Trace) -> Vec<f32> {
        let hp = &self.hp;
        let (d, ff) = (hp.n_embd, hp.n_ff_exp);
        let s = Some(sid);
        let logits: Vec<f32> = (0..hp.n_expert)
            .map(|e| {
                lw.router[e * d..(e + 1) * d]
                    .iter()
                    .zip(n)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
        let probs: Vec<f32> = logits.iter().map(|&x| sigmoid(x)).collect();
        if tr.wants("ffn_moe_logits", s) {
            tr.record("ffn_moe_logits", s, &logits);
        }
        if tr.wants("ffn_moe_probs", s) {
            tr.record("ffn_moe_probs", s, &probs);
        }

        // the bias steers which experts are picked, not how much they weigh
        let mut order: Vec<usize> = (0..hp.n_expert).collect();
        order.sort_by(|&a, &b| {
            (probs[b] + lw.exp_probs_b[b]).total_cmp(&(probs[a] + lw.exp_probs_b[a]))
        });
        let top = &order[..hp.n_expert_used];
        let sum: f32 = top.iter().map(|&e| probs[e]).sum();
        let weights: Vec<f32> = top
            .iter()
            .map(|&e| probs[e] / (sum + 1e-20) * hp.expert_weights_scale)
            .collect();
        if tr.wants("ffn_moe_topk", s) {
            tr.record(
                "ffn_moe_topk",
                s,
                &top.iter().map(|&e| e as f32).collect::<Vec<_>>(),
            );
        }

        let mut y = vec![0.0f32; d];
        for (&e, &we) in top.iter().zip(&weights) {
            let gu = lw.gate_up_exps.slice_rows(e * 2 * ff, 2 * ff).matvec_new(n);
            let act: Vec<f32> = (0..ff)
                .map(|i| {
                    let (g, u) = (gu[i], gu[ff + i]);
                    softcap(g, hp.situ_beta) * sigmoid(g) * softcap(u, hp.situ_linear_beta)
                })
                .collect();
            let out = lw.down_exps.slice_rows(e * d, d).matvec_new(&act);
            for (a, b) in y.iter_mut().zip(&out) {
                *a += we * b;
            }
        }
        if tr.wants("ffn_moe_out", s) {
            tr.record("ffn_moe_out", s, &y);
        }
        y
    }

    /// One decoder layer for one token. `blocks` is the depth-attention block list, updated in place.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        il: usize,
        sid: usize,
        h: Vec<f32>,
        blocks: &mut Vec<Vec<f32>>,
        msgs: Option<&[Vec<f32>]>,
        state: &mut State,
        pos: usize,
        tr: &mut dyn Trace,
    ) -> Vec<f32> {
        let hp = &self.hp;
        let lw = &self.w.layers[il];
        let d = hp.n_embd;
        let s = Some(sid);
        let gates = msgs.map(|_| &self.w.loops[il - hp.loop_start].gates);

        // attention inputs: one depth-mix on SWA layers, three (q, k, v) on full layers
        let mut inp: [Vec<f32>; 3] = match &lw.attn_res_query {
            None => [h.clone(), h.clone(), h.clone()],
            Some(qs) => {
                let mixed = depth_mix(blocks, &h, qs, d);
                if lw.is_swa {
                    [mixed[0].clone(), mixed[0].clone(), mixed[0].clone()]
                } else {
                    [mixed[0].clone(), mixed[1].clone(), mixed[2].clone()]
                }
            }
        };
        if let (Some(m), Some(g)) = (msgs, gates) {
            for j in 0..3 {
                inp[j] = self.message_gate(&inp[j], &m[j], &g[j]);
            }
        }
        let [n_q, n_k, n_v] = if lw.is_swa {
            let w = lw.attn_norm.as_ref().expect("checked at load");
            [
                rms_norm_w(&inp[0], w, hp.rms_eps),
                rms_norm_w(&inp[1], w, hp.rms_eps),
                rms_norm_w(&inp[2], w, hp.rms_eps),
            ]
        } else {
            [
                rms_norm_w(
                    &inp[0],
                    lw.q_inp_norm.as_ref().expect("checked at load"),
                    hp.rms_eps,
                ),
                rms_norm_w(
                    &inp[1],
                    lw.k_inp_norm.as_ref().expect("checked at load"),
                    hp.rms_eps,
                ),
                rms_norm_w(
                    &inp[2],
                    lw.v_inp_norm.as_ref().expect("checked at load"),
                    hp.rms_eps,
                ),
            ]
        };

        let attn = self.attention(lw, &mut state.slots[sid], &n_q, &n_k, &n_v, pos, sid, tr);
        if tr.wants("attn_out", s) {
            tr.record("attn_out", s, &attn);
        }

        // the first layer of each block starts a new residual stream; its input joins the block list
        // (`il % block` is the position within the 6-layer block, the same for every loop pass)
        let cur = if il.is_multiple_of(hp.attn_res_block) {
            blocks.push(h);
            attn
        } else {
            h.iter().zip(&attn).map(|(a, b)| a + b).collect()
        };
        if tr.wants("ffn_inp", s) {
            tr.record("ffn_inp", s, &cur);
        }

        let mut m = depth_mix(blocks, &cur, &lw.ffn_res_query, d).swap_remove(0);
        if let (Some(msg), Some(g)) = (msgs, gates) {
            m = self.message_gate(&m, &msg[3], &g[3]);
        }
        let n = rms_norm_w(&m, &lw.ffn_norm, hp.rms_eps);
        let y = self.moe(lw, &n, sid, tr);
        let ffn = self.convolve(
            &y,
            &mut state.slots[sid].conv[CONV_FFN],
            &lw.ffn_post_conv,
            None,
        );
        if tr.wants("ffn_out", s) {
            tr.record("ffn_out", s, &ffn);
        }
        let out: Vec<f32> = cur.iter().zip(&ffn).map(|(a, b)| a + b).collect();
        if tr.wants("l_out", s) {
            tr.record("l_out", s, &out);
        }
        out
    }

    /// Runs one token at the next position and returns its logits (`n_vocab`, unpadded).
    pub fn step(&self, token: u32, state: &mut State, tr: &mut dyn Trace) -> Result<Vec<f32>> {
        let hp = &self.hp;
        let d = hp.n_embd;
        let pos = state.pos;

        let mut emb = vec![0.0f32; d];
        gguf::dequantize_row(
            self.w.tok_embd.dtype,
            self.w.tok_embd.row(token as usize),
            &mut emb,
        );
        let mut h = rms_norm_w(&emb, &self.w.tok_norm, hp.rms_eps);

        let mut blocks: Vec<Vec<f32>> = Vec::new();
        for il in 0..hp.loop_start {
            h = self.layer(il, il, h, &mut blocks, None, state, pos, tr);
        }

        // every pass restarts from the prelude output and block list
        let (prelude_h, prelude_blocks) = (h.clone(), blocks.clone());
        for pass in 0..hp.loop_passes {
            let bank = if pass == 0 {
                None
            } else {
                // previous pass: its block outputs after the first core block, plus its final output
                let bank_blocks = blocks[prelude_blocks.len() + 1..].to_vec();
                let bank_out = std::mem::replace(&mut h, prelude_h.clone());
                blocks = prelude_blocks.clone();
                Some((bank_blocks, bank_out))
            };
            for il in hp.loop_start..hp.loop_end {
                let msgs = bank.as_ref().map(|(bb, bo)| {
                    depth_mix(bb, bo, &self.w.loops[il - hp.loop_start].msg_query, d)
                });
                h = self.layer(
                    il,
                    hp.slot(il, pass),
                    h,
                    &mut blocks,
                    msgs.as_deref(),
                    state,
                    pos,
                    tr,
                );
            }
        }

        for il in hp.loop_end..hp.n_layer {
            h = self.layer(il, il, h, &mut blocks, None, state, pos, tr);
        }

        let mixed = depth_mix(&blocks, &h, &self.w.output_res_query, d).swap_remove(0);
        let normed = rms_norm_w(&mixed, &self.w.output_norm, hp.rms_eps);
        if tr.wants("result_norm", None) {
            tr.record("result_norm", None, &normed);
        }
        let out = self.w.output.slice_rows(0, hp.n_vocab);
        let mut logits = out.matvec_new(&normed);
        if hp.final_softcap > 0.0 {
            logits
                .iter_mut()
                .for_each(|x| *x = softcap(*x, hp.final_softcap));
        }
        if tr.wants("result_output", None) {
            tr.record("result_output", None, &logits);
        }
        state.pos += 1;
        Ok(logits)
    }
}
