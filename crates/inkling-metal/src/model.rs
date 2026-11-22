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
    mul: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct MixArgs {
    d: u32,
    n_blocks: u32,
    n_q: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct ConvArgs {
    ch: u32,
    nc: u32,
    dyn_heads: u32,
    scale: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct AttnArgs {
    head_dim: u32,
    n_head: u32,
    n_kv_head: u32,
    n_rel: u32,
    pos: u32,
    first: u32,
    extent: u32,
    scale: f32,
    kv_dim: u32,
    ctx: u32,
}
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
    pad: u32,
    seg: [MultiSeg; 8],
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
    token: u32,
}

const MV_SG: usize = 8;
/// Rows per simdgroup in the MoE kernels (kernels/matvec.metal MOE_NR).
const MOE_NR: usize = 2;
const MAX_BLOCKS: usize = 8;

/// Trace hook used while encoding: (encoder, name, slot, source buffer, float count).
type RecFn<'a> =
    dyn FnMut(&mut Encoder, &'static str, Option<usize>, &Buffer, usize) -> Result<()> + 'a;

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
    k: Buffer,
    v: Buffer,
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
    attn_scratch: Buffer,
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
    pos: usize,
    trace_buf: Buffer,
    /// Stats of the most recent step.
    pub last: RunStats,
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
        let mut slots = Vec::with_capacity(hp.n_slots());
        for _ in 0..hp.n_slots() {
            slots.push(SlotBufs {
                k: ctx.buffer(max_ctx * kvd * f)?,
                v: ctx.buffer(max_ctx * kvd * f)?,
                conv: [
                    ctx.buffer(hist * kvd * f)?,
                    ctx.buffer(hist * kvd * f)?,
                    ctx.buffer(hist * d * f)?,
                    ctx.buffer(hist * d * f)?,
                ],
            });
        }
        let nb = |n: usize| ctx.buffer(n * f);
        let a = Act {
            h: nb(d)?,
            prelude_h: nb(d)?,
            bank_out: nb(d)?,
            blocks: nb(MAX_BLOCKS * d)?,
            bank: nb(MAX_BLOCKS * d)?,
            inp: nb(3 * d)?,
            n_qkv: nb(3 * d)?,
            q: nb(qd)?,
            k_lin: nb(kvd)?,
            v_lin: nb(kvd)?,
            k: nb(kvd)?,
            v: nb(kvd)?,
            k_dyn: nb(hp.n_head_kv * hp.n_conv)?,
            v_dyn: nb(hp.n_head_kv * hp.n_conv)?,
            r: nb(hp.n_head * hp.n_rel)?,
            sinks: nb(hp.n_head)?,
            o: nb(qd)?,
            gate: nb(qd)?,
            wo_out: nb(d)?,
            attn: nb(d)?,
            cur: nb(d)?,
            m: nb(d)?,
            n: nb(d)?,
            router_logits: nb(hp.n_expert)?,
            ids: nb(hp.n_expert_used)?,
            wts: nb(hp.n_expert_used)?,
            act: nb(hp.n_expert_used * hp.n_ff_exp)?,
            y: nb(d)?,
            ffn: nb(d)?,
            msgs: nb(4 * d)?,
            z: nb(hp.loop_gate_rank)?,
            tmp: nb(d)?,
            logits: nb(hp.n_vocab)?,
            attn_scratch: nb(hp.n_head * max_ctx)?,
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
            pos: 0,
            trace_buf,
            last: RunStats::default(),
        })
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
                           len: usize|
             -> Result<()> {
                if cursor < recs.len() && recs[cursor].name == name && recs[cursor].slot == slot {
                    let off = recs[cursor].offset;
                    cursor += 1;
                    e.dispatch(
                        "copy_f32",
                        &[
                            buf(src),
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
            this.encode_token(e, token, &mut rec)
        })?;

        self.last = stats;
        let tb = self.trace_buf.as_slice::<f32>();
        for r in &recs {
            tr.record(r.name, r.slot, &tb[r.offset..r.offset + r.len]);
        }
        self.pos += 1;
        Ok(self.a.logits.as_slice::<f32>()[..self.hp.n_vocab].to_vec())
    }

    /// Runs one token with per-dispatch GPU timestamps and returns (kernel, total seconds, count),
    /// sorted by time. The token advances the sequence like `step`.
    pub fn profile_step(&mut self, token: u32) -> Result<Vec<(String, f64, usize)>> {
        let this = &*self;
        let per = this.ctx.profile(4096, |e| {
            this.encode_token(e, token, &mut |_, _, _, _, _| Ok(()))
        })?;
        self.pos += 1;
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
            (rows.div_ceil(MV_SG), 1, 1),
            (32 * MV_SG, 1, 1),
        )
    }

    /// `out_s = W_s x` for several Q8_0 matrices sharing input `x` (at float offset `x_off`).
    fn mv_multi(
        &self,
        e: &mut Encoder,
        x: &Buffer,
        x_off: usize,
        segs: &[(T, usize, &Buffer)],
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
        let mut binds = vec![buf(&self.w), buf_at(x, x_off * 4)];
        for i in 0..8 {
            binds.push(buf(segs.get(i).map(|s| s.2).unwrap_or(segs[0].2)));
        }
        binds.push(bytes(&args));
        e.dispatch(
            "mv_q8_0_multi",
            &binds,
            ((start as usize).div_ceil(MV_SG), 1, 1),
            (32 * MV_SG, 1, 1),
        )
    }

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
            (1, 1, 1),
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
    ) -> Result<()> {
        let d = self.hp.n_embd;
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
                }),
            ],
            (1, 1, 1),
            (1024, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        e: &mut Encoder,
        x: &Buffer,
        hist: &Buffer,
        taps: T,
        dyn_: Option<&Buffer>,
        y: &Buffer,
        ch: usize,
    ) -> Result<()> {
        let nc = self.hp.n_conv;
        let dyn_heads = if dyn_.is_some() { self.hp.n_head_kv } else { 0 };
        let scale = 1.0 / (((if dyn_.is_some() { 2 } else { 1 }) * nc + 1) as f32).sqrt();
        e.dispatch(
            "conv_step",
            &[
                buf(x),
                buf(hist),
                buf_at(&self.w, taps.0),
                buf(dyn_.unwrap_or(x)),
                buf(y),
                bytes(&ConvArgs {
                    ch: ch as u32,
                    nc: nc as u32,
                    dyn_heads: dyn_heads as u32,
                    scale,
                }),
            ],
            (ch.div_ceil(256), 1, 1),
            (256, 1, 1),
        )
    }

    /// `out = gate(x, m)` for one message gate; x and m are d-vectors at the given float offsets.
    /// `out` may alias `x`.
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
    ) -> Result<()> {
        let d = self.hp.n_embd;
        let rank = self.hp.loop_gate_rank;
        let ga = GateArgs {
            d: d as u32,
            rank: rank as u32,
        };
        e.dispatch(
            "msg_gate_down",
            &[
                buf_at(&self.w, g.down.0),
                buf_at(x, x_off * 4),
                buf_at(m, m_off * 4),
                buf(&self.a.z),
                bytes(&ga),
            ],
            (rank, 1, 1),
            (1024, 1, 1),
        )?;
        e.dispatch(
            "msg_gate_up",
            &[
                buf_at(&self.w, g.forget.0),
                buf_at(&self.w, g.gain.0),
                buf(&self.a.z),
                buf_at(x, x_off * 4),
                buf_at(m, m_off * 4),
                buf_at(out, out_off * 4),
                bytes(&ga),
            ],
            (d.div_ceil(MV_SG), 1, 1),
            (32 * MV_SG, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        e: &mut Encoder,
        il: usize,
        sid: usize,
        n_blocks: &mut usize,
        msgs: bool,
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
        let slot = &self.slots[sid];

        // attention inputs into a.inp[0..3]
        match lw.attn_res_query {
            None => {
                for j in 0..3 {
                    self.copy(e, &a.h, 0, &a.inp, j * d, d)?;
                }
            }
            Some(q) => {
                if lw.is_swa {
                    self.mix(e, &a.blocks, *n_blocks, &a.h, q, 1, &a.inp)?;
                    self.copy(e, &a.inp, 0, &a.inp, d, d)?;
                    self.copy(e, &a.inp, 0, &a.inp, 2 * d, d)?;
                } else {
                    self.mix(e, &a.blocks, *n_blocks, &a.h, q, 3, &a.inp)?;
                }
            }
        }
        if msgs {
            let gates = &self.loops[il - hp.loop_start].gates;
            for (j, gate) in gates.iter().take(3).enumerate() {
                self.msg_gate(e, gate, &a.inp, j * d, &a.msgs, j * d, &a.inp, j * d)?;
            }
        }
        for j in 0..3 {
            let w = match (lw.attn_norm, lw.qkv_inp_norm) {
                (Some(t), _) => t,
                (None, Some(ts)) => ts[j],
                _ => unreachable!("checked at load"),
            };
            self.norm(e, &a.inp, j * d, Some(w), &a.n_qkv, j * d, d, hp.rms_eps)?;
        }

        // projections: one multi-matrix dispatch per distinct input (all of them share n_q on SWA
        // layers outside the later loop passes, where q/k/v inputs are identical)
        let tau = if lw.is_swa { 1.0 } else { hp.tau(self.pos) };
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
        if lw.is_swa && !msgs {
            q_side.append(&mut k_side);
            q_side.append(&mut v_side);
            self.mv_multi(e, &a.n_qkv, 0, &q_side)?;
        } else {
            self.mv_multi(e, &a.n_qkv, 0, &q_side)?;
            self.mv_multi(e, &a.n_qkv, d, &k_side)?;
            self.mv_multi(e, &a.n_qkv, 2 * d, &v_side)?;
        }
        e.dispatch(
            "clip_heads",
            &[
                buf(&a.q),
                buf_at(&self.w, lw.q_clip.0),
                bytes(&ClipArgs {
                    dim: hd as u32,
                    threshold: hp.qk_clip,
                    has_scale: 1,
                    mul: tau,
                }),
            ],
            (nh, 1, 1),
            (32, 1, 1),
        )?;
        self.conv(
            e,
            &a.k_lin,
            &slot.conv[0],
            lw.k_conv,
            lw.k_conv_proj.map(|_| &a.k_dyn),
            &a.k,
            kvd,
        )?;
        e.dispatch(
            "clip_heads",
            &[
                buf(&a.k),
                buf(&a.k),
                bytes(&ClipArgs {
                    dim: hd as u32,
                    threshold: hp.qk_clip,
                    has_scale: 0,
                    mul: 1.0,
                }),
            ],
            (nkv, 1, 1),
            (32, 1, 1),
        )?;
        self.conv(
            e,
            &a.v_lin,
            &slot.conv[1],
            lw.v_conv,
            lw.v_conv_proj.map(|_| &a.v_dyn),
            &a.v,
            kvd,
        )?;
        e.dispatch(
            "rms_heads",
            &[
                buf(&a.v),
                bytes(&NormArgs {
                    n: hd as u32,
                    eps: hp.rms_eps,
                    has_weight: 0,
                }),
            ],
            (nkv, 1, 1),
            (32, 1, 1),
        )?;
        e.dispatch(
            "clip_heads",
            &[
                buf(&a.r),
                buf_at(&self.w, lw.rel_clip.0),
                bytes(&ClipArgs {
                    dim: nr as u32,
                    threshold: hp.rel_clip,
                    has_scale: 1,
                    mul: tau,
                }),
            ],
            (nh, 1, 1),
            (32, 1, 1),
        )?;
        rec(e, "Qcur", s, &a.q, qd)?;
        rec(e, "Kcur", s, &a.k, kvd)?;
        rec(e, "Vcur", s, &a.v, kvd)?;
        rec(e, "Rcur", s, &a.r, nh * nr)?;

        // append to the cache at this position
        self.copy(e, &a.k, 0, &slot.k, self.pos * kvd, kvd)?;
        self.copy(e, &a.v, 0, &slot.v, self.pos * kvd, kvd)?;

        let first = if lw.is_swa {
            (self.pos + 1).saturating_sub(hp.n_swa)
        } else {
            0
        };
        e.dispatch(
            "attn_decode",
            &[
                buf(&a.q),
                buf(&slot.k),
                buf(&slot.v),
                buf(&a.r),
                buf_at(&self.w, lw.rel_bias.0),
                buf(&a.sinks),
                buf(&a.attn_scratch),
                buf(&a.o),
                bytes(&AttnArgs {
                    head_dim: hd as u32,
                    n_head: nh as u32,
                    n_kv_head: nkv as u32,
                    n_rel: nr as u32,
                    pos: self.pos as u32,
                    first: first as u32,
                    extent: lw.rel_extent as u32,
                    scale: 1.0 / (hd as f32).sqrt(),
                    kv_dim: kvd as u32,
                    ctx: self.max_ctx as u32,
                }),
            ],
            (nh, 1, 1),
            (256, 1, 1),
        )?;
        rec(e, "kqv_out", s, &a.o, qd)?;
        e.dispatch(
            "sigmoid_mul",
            &[
                buf(&a.o),
                buf(&a.gate),
                bytes(&NArgs {
                    n: qd as u32,
                    f: 0.0,
                }),
            ],
            (qd.div_ceil(256), 1, 1),
            (256, 1, 1),
        )?;
        self.mv(e, "mv_q8_0", lw.wo, &a.o, &a.wo_out, d, qd)?;
        self.conv(
            e,
            &a.wo_out,
            &slot.conv[2],
            lw.attn_post_conv,
            None,
            &a.attn,
            d,
        )?;
        rec(e, "attn_out", s, &a.attn, d)?;

        // residual bookkeeping: the first layer of a block pushes its input onto the block list
        if il.is_multiple_of(hp.attn_res_block) {
            assert!(*n_blocks < MAX_BLOCKS, "block list overflow");
            self.copy(e, &a.h, 0, &a.blocks, *n_blocks * d, d)?;
            *n_blocks += 1;
            self.copy(e, &a.attn, 0, &a.cur, 0, d)?;
        } else {
            e.dispatch(
                "add",
                &[
                    buf(&a.h),
                    buf(&a.attn),
                    buf(&a.cur),
                    bytes(&NArgs {
                        n: d as u32,
                        f: 0.0,
                    }),
                ],
                (d.div_ceil(256), 1, 1),
                (256, 1, 1),
            )?;
        }
        rec(e, "ffn_inp", s, &a.cur, d)?;

        // MoE
        self.mix(e, &a.blocks, *n_blocks, &a.cur, lw.ffn_res_query, 1, &a.m)?;
        if msgs {
            let g = &self.loops[il - hp.loop_start].gates[3];
            self.msg_gate(e, g, &a.m, 0, &a.msgs, 3 * d, &a.m, 0)?;
        }
        self.norm(e, &a.m, 0, Some(lw.ffn_norm), &a.n, 0, d, hp.rms_eps)?;
        self.mv(
            e,
            "mv_f32",
            lw.router,
            &a.n,
            &a.router_logits,
            hp.n_expert,
            d,
        )?;
        rec(e, "ffn_moe_logits", s, &a.router_logits, hp.n_expert)?;
        let bias = match lw.exp_probs_b {
            Some(t) => buf_at(&self.w, t.0),
            None => {
                return Err(Error::Model(inkling::Error::Unsupported(
                    "Metal path needs exp_probs_b".into(),
                )));
            }
        };
        e.dispatch(
            "moe_topk",
            &[
                buf(&a.router_logits),
                bias,
                buf(&a.ids),
                buf(&a.wts),
                bytes(&TopkArgs {
                    n_expert: hp.n_expert as u32,
                    k: hp.n_expert_used as u32,
                    scale: hp.expert_weights_scale,
                }),
            ],
            (1, 1, 1),
            (hp.n_expert, 1, 1),
        )?;
        let moe = MoeArgs {
            n_embd: d as u32,
            n_ff: hp.n_ff_exp as u32,
            n_used: hp.n_expert_used as u32,
            situ_beta: hp.situ_beta,
            situ_lin: hp.situ_linear_beta,
        };
        e.dispatch(
            &format!("moe_gate_up_{}", lw.exp_kind),
            &[
                buf_at(&self.w, lw.gate_up.0),
                buf(&a.n),
                buf(&a.ids),
                buf(&a.act),
                bytes(&moe),
            ],
            (hp.n_ff_exp.div_ceil(MV_SG * MOE_NR), hp.n_expert_used, 1),
            (32 * MV_SG, 1, 1),
        )?;
        e.dispatch(
            &format!("moe_down_{}", lw.exp_kind),
            &[
                buf_at(&self.w, lw.down.0),
                buf(&a.act),
                buf(&a.ids),
                buf(&a.wts),
                buf(&a.y),
                bytes(&moe),
            ],
            (d.div_ceil(MV_SG * MOE_NR), 1, 1),
            (32 * MV_SG, 1, 1),
        )?;
        rec(e, "ffn_moe_out", s, &a.y, d)?;
        self.conv(e, &a.y, &slot.conv[3], lw.ffn_post_conv, None, &a.ffn, d)?;
        rec(e, "ffn_out", s, &a.ffn, d)?;
        e.dispatch(
            "add",
            &[
                buf(&a.cur),
                buf(&a.ffn),
                buf(&a.h),
                bytes(&NArgs {
                    n: d as u32,
                    f: 0.0,
                }),
            ],
            (d.div_ceil(256), 1, 1),
            (256, 1, 1),
        )?;
        rec(e, "l_out", s, &a.h, d)?;
        Ok(())
    }

    fn encode_token(&self, e: &mut Encoder, token: u32, rec: &mut RecFn) -> Result<()> {
        let hp = &self.hp;
        let a = &self.a;
        let d = hp.n_embd;

        e.dispatch(
            "embed_q8_0",
            &[
                buf_at(&self.w, self.tok_embd.0),
                buf(&a.tmp),
                bytes(&EmbedArgs { d: d as u32, token }),
            ],
            (d.div_ceil(256), 1, 1),
            (256, 1, 1),
        )?;
        self.norm(e, &a.tmp, 0, Some(self.tok_norm), &a.h, 0, d, hp.rms_eps)?;

        let mut n_blocks = 0usize;
        for il in 0..hp.loop_start {
            self.layer(e, il, il, &mut n_blocks, false, rec)?;
        }
        let n_pre = n_blocks;
        self.copy(e, &a.h, 0, &a.prelude_h, 0, d)?;
        for pass in 0..hp.loop_passes {
            if pass > 0 {
                // bank = previous pass's blocks after the first core block, plus its output
                let n_bank = n_blocks - (n_pre + 1);
                self.copy(e, &a.blocks, (n_pre + 1) * d, &a.bank, 0, n_bank * d)?;
                self.copy(e, &a.h, 0, &a.bank_out, 0, d)?;
                self.copy(e, &a.prelude_h, 0, &a.h, 0, d)?;
                n_blocks = n_pre;
                for il in hp.loop_start..hp.loop_end {
                    let lt = &self.loops[il - hp.loop_start];
                    self.mix(e, &a.bank, n_bank, &a.bank_out, lt.msg_query, 4, &a.msgs)?;
                    self.layer(e, il, hp.slot(il, pass), &mut n_blocks, true, rec)?;
                }
            } else {
                for il in hp.loop_start..hp.loop_end {
                    self.layer(e, il, il, &mut n_blocks, false, rec)?;
                }
            }
        }
        for il in hp.loop_end..hp.n_layer {
            self.layer(e, il, il, &mut n_blocks, false, rec)?;
        }

        self.mix(e, &a.blocks, n_blocks, &a.h, self.output_res_query, 1, &a.m)?;
        self.norm(e, &a.m, 0, Some(self.output_norm), &a.n, 0, d, hp.rms_eps)?;
        rec(e, "result_norm", None, &a.n, d)?;
        self.mv(e, "mv_q8_0", self.output, &a.n, &a.logits, hp.n_vocab, d)?;
        if hp.final_softcap > 0.0 {
            e.dispatch(
                "softcap_inplace",
                &[
                    buf(&a.logits),
                    bytes(&NArgs {
                        n: hp.n_vocab as u32,
                        f: hp.final_softcap,
                    }),
                ],
                (hp.n_vocab.div_ceil(256), 1, 1),
                (256, 1, 1),
            )?;
        }
        rec(e, "result_output", None, &a.logits, hp.n_vocab)?;
        Ok(())
    }
}
