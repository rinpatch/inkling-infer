//! DSpark drafter on Metal and speculative greedy decoding (docs/dspark.md). The f32 CPU
//! reference is `inkling::drafter`; this mirrors it with Q8_0 weights, plus one deliberate
//! difference: each Markov step picks among the 8 best base-logit tokens only (acceptance is
//! unchanged down to top-4, see the doc).
//!
//! One speculative round is one command buffer: draft 7 tokens, verify [anchor, d1..d7] as a
//! batch, accept on the GPU, rewind the conv states and commit the drafter context. The CPU only
//! reads back the accepted count, which sets the next round's position.

use std::path::Path;

use inkling::drafter as dc;

use super::*;

/// Byte offsets of one drafter layer's tensors in `MetalDrafter::w`.
struct DLayerOff {
    q: usize,
    k: usize,
    v: usize,
    o: usize,
    k_ctx: usize,
    v_ctx: usize,
    gate: usize,
    up: usize,
    down: usize,
    q_norm: usize,
    k_norm: usize,
    k_norm_ctx: usize,
    input_ln: usize,
    post_ln: usize,
    dflare: usize,
}

struct Scratch {
    x: Buffer,
    a0: Buffer,
    q: Buffer,
    o: Buffer,
    ao: Buffer,
    g: Buffer,
    u: Buffer,
    dn: Buffer,
    h: Buffer,
    base: Buffer,
    cid: Buffer,
    cval: Buffer,
    xin: Buffer,
    p: Buffer,
    state: Buffer,
    zeros: Buffer,
    latent: Buffer,
    hid: Buffer,
    prev: Buffer,
    drafts: Buffer,
    fr: Buffer,
    c: Buffer,
    last_feat: Buffer,
    row_arg: Buffer,
    res: Buffer,
    logits: Buffer,
}

pub struct MetalDrafter {
    w: Buffer,
    layers: Vec<DLayerOff>,
    norm: usize,
    mask_tpl: Buffer,
    rnn: usize,
    rnn_b: usize,
    hid: usize,
    hid_b: usize,
    markov_norm: usize,
    w1: usize,
    alpha: [[f32; dc::N_FEAT]; dc::N_LAYER + 1],
    kc: Vec<Buffer>,
    vc: Vec<Buffer>,
    s: Scratch,
    /// Positions whose context K/V are committed (kept equal to the target's position).
    len: usize,
    /// Drafts per round (1..=7; env INKLING_DRAFTS, default 7). The block always has 8 rows; only
    /// the walk, the base logits and the verify batch shrink.
    pub n_drafts: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DrMixArgs {
    d: u32,
    plane: u32,
    row0: u32,
    eps: f32,
    alpha: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy)]
struct DrRopeArgs {
    row_stride: u32,
    pos0: u32,
    theta: f32,
    eps: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct DrAttnArgs {
    heads: u32,
    kv_heads: u32,
    kv_dim: u32,
    lo: u32,
    hi: u32,
    scale: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct DrN {
    n: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct DrWalkInArgs {
    s: u32,
    d: u32,
    rank: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct DrPickArgs {
    s: u32,
    d: u32,
    softcap: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct DrFeatArgs {
    d: u32,
    plane: u32,
    row: u32,
    indirect: u32,
}

const TOPK: usize = 8;
const DRAFTS: usize = dc::BLOCK - 1;

/// Weight blob under construction: Q8_0 matrices, f32 vectors and the f16 Markov embedding,
/// each 16-byte aligned.
#[derive(Default)]
struct Blob(Vec<u8>);

impl Blob {
    fn align(&mut self) -> usize {
        while !self.0.len().is_multiple_of(16) {
            self.0.push(0);
        }
        self.0.len()
    }
    fn q8(&mut self, m: &dc::Dense) -> usize {
        let off = self.align();
        for r in 0..m.rows {
            gguf::quantize_row_q8_0(&m.w[r * m.cols..(r + 1) * m.cols], &mut self.0);
        }
        off
    }
    fn f32(&mut self, v: &[f32]) -> usize {
        let off = self.align();
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
        off
    }
    fn f16(&mut self, v: &[f32]) -> usize {
        let off = self.align();
        for x in v {
            self.0
                .extend_from_slice(&half::f16::from_f32(*x).to_le_bytes());
        }
        off
    }
}

impl MetalModel {
    /// Loads the drafter from the native checkpoint directory (`mtp.safetensors`) and turns on
    /// feature capture. Call on a fresh sequence (position 0).
    pub fn load_drafter(&mut self, native_dir: impl AsRef<Path>) -> Result<()> {
        assert_eq!(self.pos, 0, "load the drafter before processing tokens");
        let st = gguf::safetensors::SafeTensors::open(native_dir.as_ref())
            .map_err(|e| Error::Model(inkling::Error::Gguf(e)))?;
        let vocab_rows = self.hp.n_vocab_padded;
        let cpu = dc::Drafter::load(&st, vocab_rows).map_err(Error::Model)?;
        let mut b = Blob::default();
        let layers = cpu
            .layers
            .iter()
            .map(|l| DLayerOff {
                q: b.q8(&l.q),
                k: b.q8(&l.k),
                v: b.q8(&l.v),
                o: b.q8(&l.o),
                k_ctx: b.q8(&l.k_ctx),
                v_ctx: b.q8(&l.v_ctx),
                gate: b.q8(&l.gate),
                up: b.q8(&l.up),
                down: b.q8(&l.down),
                q_norm: b.f32(&l.q_norm),
                k_norm: b.f32(&l.k_norm),
                k_norm_ctx: b.f32(&l.k_norm_ctx),
                input_ln: b.f32(&l.input_ln),
                post_ln: b.f32(&l.post_ln),
                dflare: b.f32(&l.dflare_norm),
            })
            .collect();
        let norm = b.f32(&cpu.norm);
        let rnn = b.q8(&cpu.rnn_proj);
        let rnn_b = b.f32(&cpu.rnn_bias);
        let hid = b.q8(&cpu.hidden_out);
        let hid_b = b.f32(&cpu.hidden_out_bias);
        let markov_norm = b.f32(&cpu.markov_norm);
        let w1 = b.f16(&cpu.markov_w1.w);
        let ctx = &self.ctx;
        let d = self.hp.n_embd;
        let mut tpl = vec![0.0f32; dc::BLOCK * d];
        for r in 1..dc::BLOCK {
            tpl[r * d..(r + 1) * d].copy_from_slice(&cpu.mask_embed);
        }
        let kvd = dc::N_KV * dc::HEAD_DIM;
        let qd = dc::N_HEAD * dc::HEAD_DIM;
        let nv = self.hp.n_vocab;
        let f = |n: usize| ctx.buffer(n * 4);
        let s = Scratch {
            x: f(dc::BLOCK * d)?,
            a0: f(dc::BLOCK * d)?,
            q: f(dc::BLOCK * qd)?,
            o: f(dc::BLOCK * qd)?,
            ao: f(dc::BLOCK * d)?,
            g: f(dc::BLOCK * dc::N_FF)?,
            u: f(dc::BLOCK * dc::N_FF)?,
            dn: f(dc::BLOCK * d)?,
            h: f(dc::BLOCK * d)?,
            base: f(DRAFTS * nv)?,
            cid: f(DRAFTS * TOPK)?,
            cval: f(DRAFTS * TOPK)?,
            xin: f(2 * dc::RANK + 2 * d)?,
            p: f(3 * dc::RANK)?,
            state: f(dc::RANK)?,
            zeros: f(dc::RANK)?,
            latent: f(dc::RANK)?,
            hid: f(d)?,
            prev: f(1)?,
            drafts: f(DRAFTS)?,
            fr: f(d)?,
            c: f(self.max_batch * d)?,
            last_feat: f(dc::N_FEAT * d)?,
            row_arg: f(MAX_VERIFY)?,
            res: f(2)?,
            logits: f(MAX_VERIFY * nv)?,
        };
        let cache = || -> Result<Vec<Buffer>> {
            (0..dc::N_LAYER)
                .map(|_| f((self.max_ctx + MAX_VERIFY) * kvd))
                .collect()
        };
        let (kc, vc) = (cache()?, cache()?);
        self.drafter = Some(Box::new(MetalDrafter {
            w: ctx.buffer_from(&b.0)?,
            layers,
            norm,
            mask_tpl: ctx.buffer_from(&tpl)?,
            rnn,
            rnn_b,
            hid,
            hid_b,
            markov_norm,
            w1,
            alpha: cpu.fuse_alpha,
            kc,
            vc,
            s,
            len: 0,
            n_drafts: std::env::var("INKLING_DRAFTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DRAFTS)
                .clamp(1, DRAFTS),
        }));
        self.enable_capture()
    }

    fn dr(&self) -> &MetalDrafter {
        self.drafter.as_deref().expect("load_drafter first")
    }

    /// y[nt][rows] = W x[nt][cols] for a Q8_0 drafter matrix at byte offset `off`.
    #[allow(clippy::too_many_arguments)]
    fn dr_proj(
        &self,
        e: &mut Encoder,
        wb: &Buffer,
        off: usize,
        x: &Buffer,
        x_off: usize,
        y: &Buffer,
        y_off: usize,
        rows: usize,
        cols: usize,
        nt: usize,
    ) -> Result<()> {
        if nt == 1 {
            return e.dispatch(
                "mv_q8_0",
                &[
                    buf_at(wb, off),
                    buf_at(x, x_off * 4),
                    buf_at(y, y_off * 4),
                    bytes(&MvArgs {
                        rows: rows as u32,
                        cols: cols as u32,
                    }),
                ],
                (rows.div_ceil(MV_SG), 1, 1),
                (32 * MV_SG, 1, 1),
            );
        }
        let mut args = MultiArgs {
            cols: cols as u32,
            n_seg: 1,
            total_rows: rows as u32,
            n_tokens: nt as u32,
            ..Default::default()
        };
        args.seg[0] = MultiSeg {
            woff: off as u64,
            start: 0,
            rows: rows as u32,
        };
        let mut binds = vec![buf(wb), buf_at(x, x_off * 4)];
        binds.extend((0..8).map(|_| buf_at(y, y_off * 4)));
        binds.push(bytes(&args));
        self.multi(e, &binds, rows, nt, false)
    }

    fn dr_norm(
        &self,
        e: &mut Encoder,
        x: &Buffer,
        x_off: usize,
        w_off: usize,
        y: &Buffer,
        rows: usize,
    ) -> Result<()> {
        let d = self.hp.n_embd;
        e.dispatch(
            "rms_norm",
            &[
                buf_at(x, x_off * 4),
                buf_at(&self.dr().w, w_off),
                buf(y),
                bytes(&NormArgs {
                    n: d as u32,
                    eps: dc::EPS,
                    has_weight: 1,
                }),
            ],
            (rows, 1, 1),
            (1024, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dr_rope(
        &self,
        e: &mut Encoder,
        x: &Buffer,
        x_off: usize,
        w_off: usize,
        heads: usize,
        rows: usize,
        pos0: usize,
    ) -> Result<()> {
        e.dispatch(
            "dr_head_norm_rope",
            &[
                buf_at(x, x_off * 4),
                buf_at(&self.dr().w, w_off),
                bytes(&DrRopeArgs {
                    row_stride: (heads * dc::HEAD_DIM) as u32,
                    pos0: pos0 as u32,
                    theta: dc::ROPE_THETA,
                    eps: dc::EPS,
                }),
            ],
            (heads, rows, 1),
            (dc::HEAD_DIM, 1, 1),
        )
    }

    fn dr_add(&self, e: &mut Encoder, x: &Buffer, y: &Buffer, n: usize) -> Result<()> {
        e.dispatch(
            "add",
            &[
                buf(x),
                buf(y),
                buf(x),
                bytes(&NArgs {
                    n: n as u32,
                    f: 0.0,
                }),
            ],
            (n.div_ceil(256), 1, 1),
            (256, 1, 1),
        )
    }

    /// Context K/V for `nt` positions starting at `pos0`, from capture rows 0..nt; then the
    /// Markov feature row (`res[0]` when `indirect`, else row nt-1).
    pub(super) fn dr_commit(
        &self,
        e: &mut Encoder,
        nt: usize,
        pos0: usize,
        indirect: bool,
    ) -> Result<()> {
        let dr = self.dr();
        let (d, kvd) = (self.hp.n_embd, dc::N_KV * dc::HEAD_DIM);
        let cap = self.capture.as_ref().expect("capture is on with a drafter");
        let plane = self.max_batch * d;
        for (l, lo) in dr.layers.iter().enumerate() {
            let a = &dr.alpha[l];
            e.dispatch(
                "dr_mix",
                &[
                    buf(cap),
                    buf_at(&dr.w, lo.dflare),
                    buf(&dr.s.c),
                    bytes(&DrMixArgs {
                        d: d as u32,
                        plane: plane as u32,
                        row0: 0,
                        eps: dc::EPS,
                        alpha: *a,
                    }),
                ],
                (nt, 1, 1),
                (1024, 1, 1),
            )?;
            e.parallel(|e| {
                self.dr_proj(
                    e,
                    &dr.w,
                    lo.k_ctx,
                    &dr.s.c,
                    0,
                    &dr.kc[l],
                    pos0 * kvd,
                    kvd,
                    d,
                    nt,
                )?;
                self.dr_proj(
                    e,
                    &dr.w,
                    lo.v_ctx,
                    &dr.s.c,
                    0,
                    &dr.vc[l],
                    pos0 * kvd,
                    kvd,
                    d,
                    nt,
                )
            })?;
            self.dr_rope(e, &dr.kc[l], pos0 * kvd, lo.k_norm_ctx, dc::N_KV, nt, pos0)?;
        }
        e.dispatch(
            "dr_copy_feat_row",
            &[
                buf(cap),
                buf(&dr.s.last_feat),
                buf(&dr.s.res),
                bytes(&DrFeatArgs {
                    d: d as u32,
                    plane: plane as u32,
                    row: (nt - 1) as u32,
                    indirect: indirect as u32,
                }),
            ],
            (d.div_ceil(256), dc::N_FEAT, 1),
            (256, 1, 1),
        )
    }

    /// Drafts 7 tokens after the anchor in a.tokens[0] at position `p0` (= committed length):
    /// writes them to the drafter's `drafts` buffer and to a.tokens[1..8].
    fn dr_draft(&self, e: &mut Encoder, p0: usize) -> Result<()> {
        let dr = self.dr();
        let s = &dr.s;
        let hp = &self.hp;
        let (d, kvd, qd, b) = (
            hp.n_embd,
            dc::N_KV * dc::HEAD_DIM,
            dc::N_HEAD * dc::HEAD_DIM,
            dc::BLOCK,
        );
        // block input: embed_norm(embed(anchor)), then mask_embed x 7
        e.dispatch(
            "embed_q8_0",
            &[
                buf_at(&self.w, self.tok_embd.0),
                buf(&self.a.tokens),
                buf(&s.a0),
                bytes(&EmbedArgs {
                    d: d as u32,
                    n_tokens: 1,
                }),
            ],
            (d.div_ceil(256), 1, 1),
            (256, 1, 1),
        )?;
        e.parallel(|e| {
            self.norm(e, &s.a0, 0, Some(self.tok_norm), &s.x, 0, d, hp.rms_eps, 1)?;
            self.copy(e, &dr.mask_tpl, d, &s.x, d, (b - 1) * d)
        })?;
        for (l, lo) in dr.layers.iter().enumerate() {
            self.dr_norm(e, &s.x, 0, lo.input_ln, &s.a0, b)?;
            e.parallel(|e| {
                self.dr_proj(e, &dr.w, lo.q, &s.a0, 0, &s.q, 0, qd, d, b)?;
                self.dr_proj(e, &dr.w, lo.k, &s.a0, 0, &dr.kc[l], p0 * kvd, kvd, d, b)?;
                self.dr_proj(e, &dr.w, lo.v, &s.a0, 0, &dr.vc[l], p0 * kvd, kvd, d, b)
            })?;
            e.parallel(|e| {
                self.dr_rope(e, &s.q, 0, lo.q_norm, dc::N_HEAD, b, p0)?;
                self.dr_rope(e, &dr.kc[l], p0 * kvd, lo.k_norm, dc::N_KV, b, p0)
            })?;
            let lo_key = if [true, false, true][l] {
                p0.saturating_sub(dc::WINDOW)
            } else {
                0
            };
            e.dispatch(
                "dr_attn",
                &[
                    buf(&s.q),
                    buf(&dr.kc[l]),
                    buf(&dr.vc[l]),
                    buf(&s.o),
                    bytes(&DrAttnArgs {
                        heads: dc::N_HEAD as u32,
                        kv_heads: dc::N_KV as u32,
                        kv_dim: kvd as u32,
                        lo: lo_key as u32,
                        hi: (p0 + b) as u32,
                        scale: 1.0 / (dc::HEAD_DIM as f32).sqrt(),
                    }),
                ],
                (dc::N_HEAD, b, 1),
                (dc::HEAD_DIM, 1, 1),
            )?;
            self.dr_proj(e, &dr.w, lo.o, &s.o, 0, &s.ao, 0, d, qd, b)?;
            self.dr_add(e, &s.x, &s.ao, b * d)?;
            self.dr_norm(e, &s.x, 0, lo.post_ln, &s.a0, b)?;
            e.parallel(|e| {
                self.dr_proj(e, &dr.w, lo.gate, &s.a0, 0, &s.g, 0, dc::N_FF, d, b)?;
                self.dr_proj(e, &dr.w, lo.up, &s.a0, 0, &s.u, 0, dc::N_FF, d, b)
            })?;
            let n = b * dc::N_FF;
            e.dispatch(
                "dr_silu_mul",
                &[buf(&s.g), buf(&s.u), bytes(&DrN { n: n as u32 })],
                (n.div_ceil(256), 1, 1),
                (256, 1, 1),
            )?;
            self.dr_proj(e, &dr.w, lo.down, &s.g, 0, &s.dn, 0, d, dc::N_FF, b)?;
            self.dr_add(e, &s.x, &s.dn, b * d)?;
        }
        // rows 1..8 -> final norm -> base logits through the target's head -> top-8 candidates
        let k = dr.n_drafts;
        self.dr_norm(e, &s.x, d, dr.norm, &s.h, k)?;
        let mut args = MultiArgs {
            cols: d as u32,
            n_seg: 1,
            total_rows: hp.n_vocab as u32,
            n_tokens: k as u32,
            ..Default::default()
        };
        args.seg[0] = MultiSeg {
            woff: self.output.0 as u64,
            start: 0,
            rows: hp.n_vocab as u32,
        };
        let mut binds = vec![buf(&self.w), buf(&s.h)];
        binds.extend((0..8).map(|_| buf(&s.base)));
        binds.push(bytes(&args));
        self.multi(e, &binds, hp.n_vocab, k, false)?;
        e.dispatch(
            "dr_topk8",
            &[
                buf(&s.base),
                buf(&s.cid),
                buf(&s.cval),
                bytes(&DrN {
                    n: hp.n_vocab as u32,
                }),
            ],
            (k, 1, 1),
            (256, 1, 1),
        )?;
        // Markov walk
        e.parallel(|e| {
            e.dispatch(
                "dr_mix",
                &[
                    buf(&s.last_feat),
                    buf_at(&dr.w, dr.markov_norm),
                    buf(&s.fr),
                    bytes(&DrMixArgs {
                        d: d as u32,
                        plane: d as u32,
                        row0: 0,
                        eps: dc::EPS,
                        alpha: dr.alpha[dc::N_LAYER],
                    }),
                ],
                (1, 1, 1),
                (1024, 1, 1),
            )?;
            self.copy(e, &s.zeros, 0, &s.state, 0, dc::RANK)
        })?;
        let nin = 2 * dc::RANK + 2 * d;
        for step in 0..k {
            let prev = if step == 0 { &self.a.tokens } else { &s.prev };
            e.dispatch(
                "dr_walk_in",
                &[
                    buf(&s.state),
                    buf_at(&dr.w, dr.w1),
                    buf(prev),
                    buf(&s.h),
                    buf(&s.fr),
                    buf(&s.xin),
                    bytes(&DrWalkInArgs {
                        s: step as u32,
                        d: d as u32,
                        rank: dc::RANK as u32,
                    }),
                ],
                (nin.div_ceil(256), 1, 1),
                (256, 1, 1),
            )?;
            self.dr_proj(e, &dr.w, dr.rnn, &s.xin, 0, &s.p, 0, 3 * dc::RANK, nin, 1)?;
            e.dispatch(
                "dr_walk_state",
                &[
                    buf(&s.p),
                    buf_at(&dr.w, dr.rnn_b),
                    buf(&s.state),
                    buf(&s.latent),
                    bytes(&DrN { n: dc::RANK as u32 }),
                ],
                (1, 1, 1),
                (dc::RANK, 1, 1),
            )?;
            self.dr_proj(e, &dr.w, dr.hid, &s.latent, 0, &s.hid, 0, d, dc::RANK, 1)?;
            e.dispatch(
                "dr_walk_pick",
                &[
                    buf_at(&self.w, self.output.0),
                    buf(&s.hid),
                    buf_at(&dr.w, dr.hid_b),
                    buf(&s.cid),
                    buf(&s.cval),
                    buf(&s.drafts),
                    buf(&s.prev),
                    buf(&self.a.tokens),
                    bytes(&DrPickArgs {
                        s: step as u32,
                        d: d as u32,
                        softcap: dc::SOFTCAP,
                    }),
                ],
                (1, 1, 1),
                (32 * TOPK, 1, 1),
            )?;
        }
        Ok(())
    }

    /// Greedy speculative decoding of `n` tokens after `first` with the drafter (load_drafter,
    /// then process the prompt with prefill / step so the draft context is in sync).
    pub fn decode_speculative(&mut self, first: u32, n: usize) -> Result<SpecOut> {
        let dr_len = self.dr().len;
        assert_eq!(
            dr_len, self.pos,
            "draft context out of sync with the target"
        );
        self.a.tokens.write(&[first]);
        let mut out = SpecOut::default();
        let (d, kvd, nh) = (self.hp.n_embd, self.hp.kv_dim(), self.hp.n_conv - 1);
        let nv = self.hp.n_vocab;
        while out.tokens.len() < n {
            if self.pos + MAX_VERIFY > self.max_ctx {
                return Err(Error::Model(inkling::Error::ContextFull(
                    self.pos + MAX_VERIFY,
                )));
            }
            if self.conv_snap.is_none() {
                let floats = self.hp.n_slots() * MAX_VERIFY * nh * (2 * kvd + 2 * d);
                self.conv_snap = Some(self.ctx.buffer(floats * 4)?);
            }
            let p0 = self.pos;
            let nv_rows = self.dr().n_drafts + 1;
            self.verify_rows = nv_rows;
            let this = &*self;
            let dr = this.dr();
            let stats = this.ctx.run(|e| {
                this.dr_draft(e, p0)?;
                this.encode(e, nv_rows, Some(&dr.s.logits), &mut |_, _, _, _, _, _| {
                    Ok(())
                })?;
                e.dispatch(
                    "dr_argmax_rows",
                    &[
                        buf(&dr.s.logits),
                        buf(&dr.s.row_arg),
                        bytes(&DrN { n: nv as u32 }),
                    ],
                    (nv_rows, 1, 1),
                    (1024, 1, 1),
                )?;
                e.dispatch(
                    "dr_accept",
                    &[
                        buf(&dr.s.row_arg),
                        buf(&this.a.tokens),
                        buf(&dr.s.res),
                        bytes(&DrN {
                            n: (nv_rows - 1) as u32,
                        }),
                    ],
                    (1, 1, 1),
                    (1, 1, 1),
                )?;
                e.parallel(|e| {
                    let snap = this.conv_snap.as_ref().expect("allocated above");
                    for sid in 0..this.hp.n_slots() {
                        for k in 0..4 {
                            let len = nh * if k < 2 { kvd } else { d };
                            let off = this.snap_off(sid, k).expect("verify_rows set");
                            e.dispatch(
                                "dr_conv_rewind",
                                &[
                                    buf_at(snap, off * 4),
                                    buf(&this.slots[sid].conv[k]),
                                    buf(&dr.s.res),
                                    bytes(&DrN { n: len as u32 }),
                                ],
                                (len.div_ceil(256), 1, 1),
                                (256, 1, 1),
                            )?;
                        }
                    }
                    Ok(())
                })?;
                this.dr_commit(e, nv_rows, p0, true)
            });
            self.verify_rows = 0;
            self.last = stats?;
            let dr = self.dr();
            let res = dr.s.res.as_slice::<u32>();
            let (acc, bonus) = (res[0] as usize, res[1]);
            out.tokens
                .extend_from_slice(&dr.s.drafts.as_slice::<u32>()[..acc]);
            out.tokens.push(bonus);
            out.rounds += 1;
            out.accepted += acc;
            self.pos = p0 + acc + 1;
            self.drafter.as_deref_mut().expect("loaded").len = self.pos;
        }
        out.tokens.truncate(n);
        Ok(out)
    }

    /// Keeps the draft context in sync after a forward of `nt` tokens at `pos0` (capture rows).
    pub(super) fn dr_after_forward(&self, e: &mut Encoder, nt: usize, pos0: usize) -> Result<()> {
        if self.drafter.is_some() {
            self.dr_commit(e, nt, pos0, false)?;
        }
        Ok(())
    }

    pub(super) fn dr_set_len(&mut self, len: usize) {
        if let Some(dr) = self.drafter.as_deref_mut() {
            dr.len = len;
        }
    }
}

/// Result of `decode_speculative`.
#[derive(Default, Debug)]
pub struct SpecOut {
    pub tokens: Vec<u32>,
    pub rounds: usize,
    pub accepted: usize,
}

/// Intermediate values of one Metal draft, for comparison with the CPU reference.
pub struct DraftDebug {
    /// layer-0 context key (rotated) of position 0 and of position len-1
    pub k0_first: Vec<f32>,
    pub k0_last: Vec<f32>,
    /// post-norm block hidden, rows 1..8
    pub h: Vec<f32>,
    pub cand: Vec<u32>,
    pub drafts: Vec<u32>,
    pub fr: Vec<f32>,
}

impl MetalModel {
    /// Runs one draft after `anchor` at the current position without verifying (debug only).
    pub fn debug_draft(&mut self, anchor: u32) -> Result<DraftDebug> {
        let kvd = dc::N_KV * dc::HEAD_DIM;
        let p0 = self.pos;
        let d = self.hp.n_embd;
        // context keys before the draft overwrites rows p0.. with block keys
        let kc0 = self.dr().kc[0].as_slice::<f32>();
        let (k0_first, k0_last) = (kc0[..kvd].to_vec(), kc0[(p0 - 1) * kvd..p0 * kvd].to_vec());
        self.a.tokens.write(&[anchor]);
        let this = &*self;
        self.last = this.ctx.run(|e| this.dr_draft(e, p0))?;
        let s = &self.dr().s;
        Ok(DraftDebug {
            k0_first,
            k0_last,
            h: s.h.as_slice::<f32>()[..DRAFTS * d].to_vec(),
            cand: s.cid.as_slice::<u32>()[..DRAFTS * TOPK].to_vec(),
            drafts: s.drafts.as_slice::<u32>()[..DRAFTS].to_vec(),
            fr: s.fr.as_slice::<f32>()[..d].to_vec(),
        })
    }
}

impl MetalModel {
    /// Expert ids of the last layer of the most recent forward, `nt * n_used` (debug only).
    pub fn debug_last_ids(&self, nt: usize) -> Vec<u32> {
        self.a.ids.as_slice::<u32>()[..nt * self.hp.n_expert_used].to_vec()
    }
}
