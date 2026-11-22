//! Effective bandwidth of the decode matvec kernels on real weights:
//! `cargo run --release -p inkling-metal --example kernel_bw -- <model.gguf>`

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

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("model path");
    let g = gguf::Gguf::open(&path)?;
    let ctx = Context::new()?;
    // SAFETY: the mmap outlives the buffer (both live until the end of main).
    let w = unsafe { ctx.buffer_no_copy(g.mmap().as_ptr(), g.mmap().len())? };
    let reps = 20;

    let out = g.tensor("output.weight")?;
    let (rows, cols) = (out.dims[1], out.dims[0]);
    let x = ctx.buffer_from(&vec![0.01f32; cols])?;
    let y = ctx.buffer(rows * 4)?;
    for kernel in ["mv_q8_0"] {
        let mut best = f64::MAX;
        for _ in 0..reps {
            let s = ctx.run(|e| {
                e.dispatch(
                    kernel,
                    &[
                        buf_at(&w, out.offset),
                        buf(&x),
                        buf(&y),
                        bytes(&MvArgs {
                            rows: rows as u32,
                            cols: cols as u32,
                        }),
                    ],
                    (rows.div_ceil(8), 1, 1),
                    (256, 1, 1),
                )
            })?;
            best = best.min(s.gpu_s);
        }
        println!(
            "{kernel:22} lm_head {:>6.1} MB  {:6.3} ms  {:6.1} GB/s",
            out.nbytes as f64 / 1e6,
            best * 1e3,
            out.nbytes as f64 / best / 1e9
        );
    }

    let gu = g.tensor("blk.6.ffn_gate_up_exps.weight")?;
    let dn = g.tensor("blk.6.ffn_down_exps.weight")?;
    let (d, ff, ne) = (gu.dims[0], gu.dims[1] / 2, gu.dims[2]);
    let k = 8usize;
    let ids = ctx.buffer_from(&[3u32, 17, 29, 44, 58, 71, 90, 120])?;
    let wts = ctx.buffer_from(&[1.0f32; 8])?;
    let xin = ctx.buffer_from(&vec![0.01f32; d])?;
    let act = ctx.buffer(k * ff * 4)?;
    let yo = ctx.buffer(d * 4)?;
    let a = MoeArgs {
        n_embd: d as u32,
        n_ff: ff as u32,
        n_used: k as u32,
        situ_beta: 4.0,
        situ_lin: 25.0,
    };
    let gu_bytes = gu.nbytes / ne * k;
    let dn_bytes = dn.nbytes / ne * k;
    let mut best = (f64::MAX, f64::MAX);
    for _ in 0..reps {
        let s1 = ctx.run(|e| {
            e.dispatch(
                "moe_gate_up_iq4_nl",
                &[
                    buf_at(&w, gu.offset),
                    buf(&xin),
                    buf(&ids),
                    buf(&act),
                    bytes(&a),
                ],
                (ff.div_ceil(16), k, 1),
                (256, 1, 1),
            )
        })?;
        let s2 = ctx.run(|e| {
            e.dispatch(
                "moe_down_iq4_nl",
                &[
                    buf_at(&w, dn.offset),
                    buf(&act),
                    buf(&ids),
                    buf(&wts),
                    buf(&yo),
                    bytes(&a),
                ],
                (d.div_ceil(16), 1, 1),
                (256, 1, 1),
            )
        })?;
        best = (best.0.min(s1.gpu_s), best.1.min(s2.gpu_s));
    }
    println!(
        "{:22} 8 experts {:>5.1} MB  {:6.3} ms  {:6.1} GB/s",
        "moe_gate_up_iq4_nl",
        gu_bytes as f64 / 1e6,
        best.0 * 1e3,
        gu_bytes as f64 / best.0 / 1e9
    );
    println!(
        "{:22} 8 experts {:>5.1} MB  {:6.3} ms  {:6.1} GB/s",
        "moe_down_iq4_nl",
        dn_bytes as f64 / 1e6,
        best.1 * 1e3,
        dn_bytes as f64 / best.1 / 1e9
    );

    let q = g.tensor("blk.6.attn_q.weight")?;
    let (rows, cols) = (q.dims[1], q.dims[0]);
    let yq = ctx.buffer(rows * 4)?;
    let mut bq = f64::MAX;
    for _ in 0..reps {
        let s = ctx.run(|e| {
            e.dispatch(
                "mv_q8_0",
                &[
                    buf_at(&w, q.offset),
                    buf(&xin),
                    buf(&yq),
                    bytes(&MvArgs {
                        rows: rows as u32,
                        cols: cols as u32,
                    }),
                ],
                (rows.div_ceil(8), 1, 1),
                (256, 1, 1),
            )
        })?;
        bq = bq.min(s.gpu_s);
    }
    println!(
        "{:22} attn_q {:>6.1} MB  {:6.3} ms  {:6.1} GB/s  (includes command-buffer overhead)",
        "mv_q8_0",
        q.nbytes as f64 / 1e6,
        bq * 1e3,
        q.nbytes as f64 / bq / 1e9
    );
    Ok(())
}
