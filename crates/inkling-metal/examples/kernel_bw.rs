//! Where decode time goes: achieved bandwidth of the decode kernels on cold weights vs the
//! practical peak, and the alignment cost of 18-byte IQ4_NL blocks.
//! `cargo run --release -p inkling-metal --example kernel_bw -- <model.gguf>`
//!
//! Each measurement is one command buffer of many dispatches over data much larger than the GPU
//! caches, so per-buffer overhead is amortized and nothing is served from cache.

use inkling::Hparams;
use inkling_metal::model::MetalModel;
use inkling_metal::runtime::{Context, buf, buf_at, bytes};

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
struct NormArgs {
    n: u32,
    eps: f32,
    has_weight: u32,
}

const MV_SG: usize = 8;
const MOE_NR: usize = 2;
const REPS: usize = 5;

fn best(f: &mut dyn FnMut() -> anyhow::Result<f64>) -> anyhow::Result<f64> {
    let mut b = f64::MAX;
    for _ in 0..REPS {
        b = b.min(f()?);
    }
    Ok(b)
}

fn report(name: &str, nbytes: usize, s: f64) {
    println!(
        "{name:34} {:>8.1} MB  {:8.3} ms  {:6.1} GB/s",
        nbytes as f64 / 1e6,
        s * 1e3,
        nbytes as f64 / s / 1e9
    );
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("model path");
    let g = gguf::Gguf::open(&path)?;
    let hp = Hparams::from_gguf(&g)?;
    let ctx = Context::new()?;
    // SAFETY: the mmap outlives the buffer (both live until the end of main).
    let w = unsafe { ctx.buffer_no_copy(g.mmap().as_ptr(), g.mmap().len())? };

    // 1. practical peak: 2 GB read with aligned uint4 loads, vs the same bytes as 18-byte blocks
    let n = 2usize << 30;
    let src = ctx.buffer(n)?;
    let threads = 20 * 1024 * 16;
    let out = ctx.buffer(threads * 4)?;
    let n16 = (n / 16) as u32;
    let t = best(&mut || {
        Ok(ctx
            .run(|e| {
                e.dispatch(
                    "bw_aligned",
                    &[buf(&src), buf(&out), bytes(&n16)],
                    (threads / 256, 1, 1),
                    (256, 1, 1),
                )
            })?
            .gpu_s)
    })?;
    report("read, aligned uint4", n, t);
    let nhalf = (n / 18 * 2) as u32;
    let t = best(&mut || {
        Ok(ctx
            .run(|e| {
                e.dispatch(
                    "bw_blocks18",
                    &[buf(&src), buf(&out), bytes(&nhalf)],
                    (threads / 256, 1, 1),
                    (256, 1, 1),
                )
            })?
            .gpu_s)
    })?;
    report("read, 18-byte blocks (packed)", n / 18 * 18, t);

    // 2. the decode kernels on cold weights: every layer once, different experts per layer
    let k = hp.n_expert_used;
    let (d, ff) = (hp.n_embd, hp.n_ff_exp);
    let x = ctx.buffer_from(&vec![0.01f32; d.max(ff) * 4])?;
    let act = ctx.buffer_from(&vec![0.01f32; k * ff])?;
    let yo = ctx.buffer(d * 4 * 4)?;
    let wts = ctx.buffer_from(&vec![1.0f32; k])?;
    let a = MoeArgs {
        n_embd: d as u32,
        n_ff: ff as u32,
        n_used: k as u32,
        situ_beta: 4.0,
        situ_lin: 25.0,
    };
    let id_bufs: Vec<_> = (0..hp.n_layer)
        .map(|l| {
            let ids: Vec<u32> = (0..k as u32)
                .map(|i| (l as u32 * 37 + i * 13) % hp.n_expert as u32)
                .collect();
            ctx.buffer_from(&ids)
        })
        .collect::<Result<_, _>>()?;
    for kind in ["iq4_nl", "q8_0"] {
        let layers: Vec<usize> = (0..hp.n_layer)
            .filter(|&l| {
                let t = g
                    .tensor(&format!("blk.{l}.ffn_gate_up_exps.weight"))
                    .unwrap();
                t.dtype
                    == if kind == "q8_0" {
                        gguf::DType::Q8_0
                    } else {
                        gguf::DType::Iq4Nl
                    }
            })
            .collect();
        if layers.is_empty() {
            continue;
        }
        let (mut gu_b, mut dn_b) = (0, 0);
        for &l in &layers {
            let gu = g.tensor(&format!("blk.{l}.ffn_gate_up_exps.weight"))?;
            let dn = g.tensor(&format!("blk.{l}.ffn_down_exps.weight"))?;
            gu_b += gu.nbytes / hp.n_expert * k;
            dn_b += dn.nbytes / hp.n_expert * k;
        }
        let tg = best(&mut || {
            Ok(ctx
                .run(|e| {
                    for &l in &layers {
                        let gu = g
                            .tensor(&format!("blk.{l}.ffn_gate_up_exps.weight"))
                            .unwrap();
                        e.dispatch(
                            &format!("moe_gate_up_{kind}"),
                            &[
                                buf_at(&w, gu.offset),
                                buf(&x),
                                buf(&id_bufs[l]),
                                buf(&act),
                                bytes(&a),
                            ],
                            (ff.div_ceil(MV_SG * MOE_NR), k, 1),
                            (32 * MV_SG, 1, 1),
                        )?;
                    }
                    Ok(())
                })?
                .gpu_s)
        })?;
        let td = best(&mut || {
            Ok(ctx
                .run(|e| {
                    for &l in &layers {
                        let dn = g.tensor(&format!("blk.{l}.ffn_down_exps.weight")).unwrap();
                        e.dispatch(
                            &format!("moe_down_{kind}"),
                            &[
                                buf_at(&w, dn.offset),
                                buf(&act),
                                buf(&id_bufs[l]),
                                buf(&wts),
                                buf(&yo),
                                bytes(&a),
                            ],
                            (d.div_ceil(MV_SG * MOE_NR), 1, 1),
                            (32 * MV_SG, 1, 1),
                        )?;
                    }
                    Ok(())
                })?
                .gpu_s)
        })?;
        report(
            &format!("moe_gate_up_{kind} x{} layers", layers.len()),
            gu_b,
            tg,
        );
        report(
            &format!("moe_down_{kind} x{} layers", layers.len()),
            dn_b,
            td,
        );
    }

    // Q8_0 dense projections (attention and friends), every layer
    let dense: Vec<_> = g
        .tensors
        .iter()
        .filter(|t| {
            t.name.starts_with("blk.")
                && !t.name.contains("_exps")
                && t.dtype == gguf::DType::Q8_0
                && t.dims.len() == 2
        })
        .collect();
    let dense_b: usize = dense.iter().map(|t| t.nbytes).sum();
    let ys = ctx.buffer(dense.iter().map(|t| t.dims[1]).max().unwrap() * 4)?;
    let tq = best(&mut || {
        Ok(ctx
            .run(|e| {
                for t in &dense {
                    let (rows, cols) = (t.dims[1], t.dims[0]);
                    e.dispatch(
                        "mv_q8_0",
                        &[
                            buf_at(&w, t.offset),
                            buf(&x),
                            buf(&ys),
                            bytes(&MvArgs {
                                rows: rows as u32,
                                cols: cols as u32,
                            }),
                        ],
                        (rows.div_ceil(MV_SG), 1, 1),
                        (32 * MV_SG, 1, 1),
                    )?;
                }
                Ok(())
            })?
            .gpu_s)
    })?;
    report(
        &format!("mv_q8_0, {} dense tensors", dense.len()),
        dense_b,
        tq,
    );

    // the same dispatches with no barriers between them (they are independent; the shared output
    // buffer races, which does not matter for timing)
    let tp = best(&mut || {
        Ok(ctx
            .run(|e| {
                e.parallel(|e| {
                    for t in &dense {
                        let (rows, cols) = (t.dims[1], t.dims[0]);
                        e.dispatch(
                            "mv_q8_0",
                            &[
                                buf_at(&w, t.offset),
                                buf(&x),
                                buf(&ys),
                                bytes(&MvArgs {
                                    rows: rows as u32,
                                    cols: cols as u32,
                                }),
                            ],
                            (rows.div_ceil(MV_SG), 1, 1),
                            (32 * MV_SG, 1, 1),
                        )?;
                    }
                    Ok(())
                })
            })?
            .gpu_s)
    })?;
    report(
        &format!("mv_q8_0, {} dense, no barriers", dense.len()),
        dense_b,
        tp,
    );

    // the same kernel on one large tensor: dispatch ramp-up and drain amortized away
    let lm = g.tensor("output.weight")?;
    let (rows, cols) = (lm.dims[1], lm.dims[0]);
    let ylm = ctx.buffer(rows * 4)?;
    let t = best(&mut || {
        Ok(ctx
            .run(|e| {
                e.dispatch(
                    "mv_q8_0",
                    &[
                        buf_at(&w, lm.offset),
                        buf(&x),
                        buf(&ylm),
                        bytes(&MvArgs {
                            rows: rows as u32,
                            cols: cols as u32,
                        }),
                    ],
                    (rows.div_ceil(MV_SG), 1, 1),
                    (32 * MV_SG, 1, 1),
                )
            })?
            .gpu_s)
    })?;
    report("mv_q8_0, lm_head (one dispatch)", lm.nbytes, t);

    // cost of a tiny dependent dispatch: rms_norm of one 2304-float row, ping-ponging buffers
    let (na, nb2) = (ctx.buffer_from(&vec![1.0f32; d])?, ctx.buffer(d * 4)?);
    let args = NormArgs {
        n: d as u32,
        eps: 1e-6,
        has_weight: 0,
    };
    let nd = 1000;
    let t = best(&mut || {
        Ok(ctx
            .run(|e| {
                for i in 0..nd {
                    let (src, dst) = if i % 2 == 0 { (&na, &nb2) } else { (&nb2, &na) };
                    e.dispatch(
                        "rms_norm",
                        &[buf(src), buf(src), buf(dst), bytes(&args)],
                        (1, 1, 1),
                        (1024, 1, 1),
                    )?;
                }
                Ok(())
            })?
            .gpu_s)
    })?;
    println!(
        "tiny dependent dispatch (rms_norm, 1 row): {:.2} us each",
        t / nd as f64 * 1e6
    );

    // 4. repack test: the 8 used experts of every IQ4_NL layer copied into compact buffers, once in
    // the GGUF block layout and once split into aligned values + scales; identical bytes and work.
    let iq_layers: Vec<usize> = (0..hp.n_layer)
        .filter(|&l| {
            g.tensor(&format!("blk.{l}.ffn_gate_up_exps.weight"))
                .unwrap()
                .dtype
                == gguf::DType::Iq4Nl
        })
        .collect();
    let compact_ids = ctx.buffer_from(&(0..k as u32).collect::<Vec<_>>())?;
    let repack = |name: &str, l: usize| -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let t = g.tensor(&format!("blk.{l}.{name}.weight"))?;
        let per = t.nbytes / hp.n_expert;
        let raw = g.bytes(t);
        let mut aos = Vec::with_capacity(per * k);
        for i in 0..k as u32 {
            let e = ((l as u32 * 37 + i * 13) % hp.n_expert as u32) as usize;
            aos.extend_from_slice(&raw[e * per..(e + 1) * per]);
        }
        let (mut q, mut dsc) = (
            Vec::with_capacity(aos.len() / 18 * 16),
            Vec::with_capacity(aos.len() / 18 * 2),
        );
        for blk in aos.chunks_exact(18) {
            dsc.extend_from_slice(&blk[..2]);
            q.extend_from_slice(&blk[2..]);
        }
        Ok((aos, q, dsc))
    };
    let (mut gu_aos, mut gu_q, mut gu_d, mut dn_aos, mut dn_q, mut dn_d) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut gu_bytes, mut dn_bytes) = (0, 0);
    for &l in &iq_layers {
        let (a1, q1, d1) = repack("ffn_gate_up_exps", l)?;
        let (a2, q2, d2) = repack("ffn_down_exps", l)?;
        gu_bytes += a1.len();
        dn_bytes += a2.len();
        gu_aos.push(ctx.buffer_from(&a1)?);
        gu_q.push(ctx.buffer_from(&q1)?);
        gu_d.push(ctx.buffer_from(&d1)?);
        dn_aos.push(ctx.buffer_from(&a2)?);
        dn_q.push(ctx.buffer_from(&q2)?);
        dn_d.push(ctx.buffer_from(&d2)?);
    }
    let n = iq_layers.len();
    let t_aos = best(&mut || {
        Ok(ctx
            .run(|e| {
                for i in 0..n {
                    e.dispatch(
                        "moe_gate_up_iq4_nl",
                        &[
                            buf(&gu_aos[i]),
                            buf(&x),
                            buf(&compact_ids),
                            buf(&act),
                            bytes(&a),
                        ],
                        (ff.div_ceil(MV_SG * MOE_NR), k, 1),
                        (32 * MV_SG, 1, 1),
                    )?;
                    e.dispatch(
                        "moe_down_iq4_nl",
                        &[
                            buf(&dn_aos[i]),
                            buf(&act),
                            buf(&compact_ids),
                            buf(&wts),
                            buf(&yo),
                            bytes(&a),
                        ],
                        (d.div_ceil(MV_SG * MOE_NR), 1, 1),
                        (32 * MV_SG, 1, 1),
                    )?;
                }
                Ok(())
            })?
            .gpu_s)
    })?;
    let t_soa = best(&mut || {
        Ok(ctx
            .run(|e| {
                for i in 0..n {
                    e.dispatch(
                        "moe_gate_up_iq4_soa",
                        &[
                            buf(&gu_q[i]),
                            buf(&gu_d[i]),
                            buf(&x),
                            buf(&compact_ids),
                            buf(&act),
                            bytes(&a),
                        ],
                        (ff.div_ceil(MV_SG * MOE_NR), k, 1),
                        (32 * MV_SG, 1, 1),
                    )?;
                    e.dispatch(
                        "moe_down_iq4_soa",
                        &[
                            buf(&dn_q[i]),
                            buf(&dn_d[i]),
                            buf(&act),
                            buf(&compact_ids),
                            buf(&wts),
                            buf(&yo),
                            bytes(&a),
                        ],
                        (d.div_ceil(MV_SG * MOE_NR), 1, 1),
                        (32 * MV_SG, 1, 1),
                    )?;
                }
                Ok(())
            })?
            .gpu_s)
    })?;
    report(
        &format!("IQ4_NL experts, GGUF layout x{n}"),
        gu_bytes + dn_bytes,
        t_aos,
    );
    report(
        &format!("IQ4_NL experts, repacked x{n}"),
        gu_bytes + dn_bytes,
        t_soa,
    );

    // 5. matrix-unit peak: 8x8x8 multiply-accumulates per simdgroup, registers only
    let out_peak = ctx.buffer(4 << 20)?;
    let iters: u32 = 4096;
    for kname in ["mma_peak_f16", "mma_peak_f32"] {
        let groups = 20 * 64;
        let t = best(&mut || {
            Ok(ctx
                .run(|e| {
                    e.dispatch(
                        kname,
                        &[buf(&out_peak), bytes(&iters)],
                        (groups, 1, 1),
                        (256, 1, 1),
                    )
                })?
                .gpu_s)
        })?;
        let simdgroups = groups * 256 / 32;
        let flops = simdgroups as f64 * iters as f64 * 8.0 * 512.0 * 2.0;
        println!("{kname:34} {:6.2} TFLOP/s", flops / t / 1e12);
    }

    // 3. a real decode step for comparison
    let mut m = MetalModel::new(&g, hp.clone(), 256)?;
    for i in 0..64u32 {
        m.step(1000 + i, &mut inkling::NoTrace)?;
    }
    let mut step = f64::MAX;
    for i in 0..16u32 {
        m.step(2000 + i, &mut inkling::NoTrace)?;
        step = step.min(m.last.gpu_s);
    }
    let t0 = std::time::Instant::now();
    let mut enc = 0.0;
    for i in 0..32u32 {
        m.step(3000 + i, &mut inkling::NoTrace)?;
        enc += m.last.encode_s;
    }
    let wall = t0.elapsed().as_secs_f64() / 32.0;
    println!(
        "decode step: {:.3} ms GPU (best), {:.3} ms wall, {:.3} ms encoding, {} dispatches",
        step * 1e3,
        wall * 1e3,
        enc / 32.0 * 1e3,
        m.last.n_dispatch
    );
    Ok(())
}
