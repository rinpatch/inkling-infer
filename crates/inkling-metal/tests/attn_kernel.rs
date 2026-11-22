//! Kernel-level check of attn_tiled: the same K/V values (pre-rounded to f16) given as an f32 cache
//! and as an f16 cache must give the same attention output, since the kernel computes in f32.

use inkling_metal::runtime::{Context, buf, bytes};

#[repr(C)]
#[derive(Clone, Copy)]
struct TiledArgs {
    n_tokens: u32,
    n_head: u32,
    n_kv_head: u32,
    pos0: u32,
    n_swa: u32,
    extent: u32,
    scale: f32,
    kv_dim: u32,
}

fn f16_bits(x: f32) -> u16 {
    half_from_f32(x)
}

// round-to-nearest-even f32 -> f16 (normal range only; test values are small and well inside it)
fn half_from_f32(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    let mut h = ((exp as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

fn f32_from_half(h: u16) -> f32 {
    let s = if h >> 15 == 1 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    if e == 0 {
        return s * m * 2f32.powi(-24);
    }
    s * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
}

#[test]
fn tiled_attention_f16_cache_matches_f32_cache_on_identical_values() {
    let ctx = Context::new().unwrap();
    let (nt, nh, nkv, hd) = (7usize, 24usize, 8usize, 128usize);
    let kvd = nkv * hd;
    let extent = 1024usize;
    // deterministic pseudo-random values in a realistic range
    let mut seed = 12345u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    };
    let q: Vec<f32> = (0..nt * nh * hd).map(|_| rnd() * 2.0).collect();
    let kv_vals: Vec<f32> = (0..nt * kvd)
        .map(|_| f32_from_half(f16_bits(rnd() * 2.0)))
        .collect();
    let v_vals: Vec<f32> = (0..nt * kvd)
        .map(|_| f32_from_half(f16_bits(rnd())))
        .collect();
    let bias: Vec<f32> = (0..nt * nh * extent).map(|_| rnd() * 0.5).collect();
    let sinks: Vec<f32> = (0..nt * nh).map(|_| rnd()).collect();

    let qb = ctx.buffer_from(&q).unwrap();
    let k32 = ctx.buffer_from(&kv_vals).unwrap();
    let v32 = ctx.buffer_from(&v_vals).unwrap();
    let k16 = ctx
        .buffer_from(&kv_vals.iter().map(|&x| f16_bits(x)).collect::<Vec<u16>>())
        .unwrap();
    let v16 = ctx
        .buffer_from(&v_vals.iter().map(|&x| f16_bits(x)).collect::<Vec<u16>>())
        .unwrap();
    let bb = ctx.buffer_from(&bias).unwrap();
    let sb = ctx.buffer_from(&sinks).unwrap();
    let o32 = ctx.buffer(nt * nh * hd * 4).unwrap();
    let o16 = ctx.buffer(nt * nh * hd * 4).unwrap();
    let args = TiledArgs {
        n_tokens: nt as u32,
        n_head: nh as u32,
        n_kv_head: nkv as u32,
        pos0: 0,
        n_swa: 0,
        extent: extent as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        kv_dim: kvd as u32,
    };
    ctx.run(|e| {
        e.dispatch(
            "attn_tiled_kv32_f32",
            &[
                buf(&qb),
                buf(&k32),
                buf(&v32),
                buf(&bb),
                buf(&sb),
                buf(&o32),
                bytes(&args),
            ],
            (1, nh, 1),
            (128, 1, 1),
        )?;
        e.dispatch(
            "attn_tiled_kv16_f32",
            &[
                buf(&qb),
                buf(&k16),
                buf(&v16),
                buf(&bb),
                buf(&sb),
                buf(&o16),
                bytes(&args),
            ],
            (1, nh, 1),
            (128, 1, 1),
        )
    })
    .unwrap();
    let (a, b) = (o32.as_slice::<f32>(), o16.as_slice::<f32>());
    let mut worst = (0usize, 0.0f32);
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > worst.1 {
            worst = (i, d);
        }
    }
    let (t, h, c) = (worst.0 / (nh * hd), worst.0 / hd % nh, worst.0 % hd);
    println!(
        "worst |f32-cache - f16-cache| = {:.3e} at token {t} head {h} dim {c} ({} vs {})",
        worst.1, a[worst.0], b[worst.0]
    );
    assert!(
        worst.1 < 1e-5,
        "attn_tiled differs between f32 and f16 caches holding identical values"
    );
}
