//! argmax_token with a temperature draws by Gumbel-max: over many seeds, pick frequencies must
//! match softmax(logits / T).

use inkling_metal::runtime::{Context, buf, bytes};

#[repr(C)]
#[derive(Clone, Copy)]
struct ArgmaxArgs {
    n: u32,
    out_index: u32,
    inv_temp: f32,
    seed: u32,
}

#[test]
fn gumbel_max_matches_softmax() {
    let ctx = Context::new().unwrap();
    let logits: Vec<f32> = vec![1.0, 2.5, 0.0, 2.0, -1.0, 3.0, 2.9, 0.5];
    let temp = 0.8f32;
    let draws = 40_000usize;
    let lb = ctx.buffer_from(&logits).unwrap();
    let tokens = ctx.buffer(4).unwrap();
    let out = ctx.buffer(draws * 4).unwrap();
    // one dispatch per draw, many per command buffer
    for chunk in (0..draws).collect::<Vec<_>>().chunks(4000) {
        ctx.run(|e| {
            for &i in chunk {
                e.dispatch(
                    "argmax_token",
                    &[
                        buf(&lb),
                        buf(&tokens),
                        buf(&out),
                        bytes(&ArgmaxArgs {
                            n: logits.len() as u32,
                            out_index: i as u32,
                            inv_temp: 1.0 / temp,
                            seed: (i as u32).wrapping_mul(2654435761).wrapping_add(12345),
                        }),
                    ],
                    (1, 1, 1),
                    (1024, 1, 1),
                )?;
            }
            Ok(())
        })
        .unwrap();
    }
    let picks = out.as_slice::<u32>();
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f64> = logits
        .iter()
        .map(|&l| (((l - m) / temp) as f64).exp())
        .collect();
    let z: f64 = w.iter().sum();
    for (k, wk) in w.iter().enumerate() {
        let p = wk / z;
        let got = picks.iter().filter(|&&t| t as usize == k).count() as f64 / draws as f64;
        // 5 standard errors of a binomial proportion
        let tol = 5.0 * (p * (1.0 - p) / draws as f64).sqrt() + 1e-4;
        assert!(
            (got - p).abs() < tol,
            "token {k}: frequency {got:.4} vs softmax {p:.4}"
        );
    }
    // temperature 0 stays an exact argmax
    ctx.run(|e| {
        e.dispatch(
            "argmax_token",
            &[
                buf(&lb),
                buf(&tokens),
                buf(&out),
                bytes(&ArgmaxArgs {
                    n: logits.len() as u32,
                    out_index: 0,
                    inv_temp: 0.0,
                    seed: 7,
                }),
            ],
            (1, 1, 1),
            (1024, 1, 1),
        )
    })
    .unwrap();
    assert_eq!(out.as_slice::<u32>()[0], 5);
}
