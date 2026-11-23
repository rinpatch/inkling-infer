//! Metal decode vs the f32 CPU reference on the same tokens. Both compute in f32, so every traced
//! tensor must agree to rounding (reduction order differs), far tighter than the llama.cpp fixtures.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use inkling::cpu::{CpuModel, State};
use inkling::weights::Weights;
use inkling::{Hparams, Trace};
use inkling_metal::model::{MetalModel, Options};

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INKLING_MODEL") {
        return Some(p.into());
    }
    let home = PathBuf::from(std::env::var_os("HOME")?);
    let snaps = home.join(".cache/huggingface/hub/models--ggml-org--Inkling-2-Mini-GGUF/snapshots");
    fs::read_dir(snaps)
        .ok()?
        .flatten()
        .map(|e| e.path().join("Inkling-2-Mini-IQ4_NL.gguf"))
        .find(|p| p.exists())
}

#[derive(Default)]
struct All(BTreeMap<String, Vec<f32>>);

impl Trace for All {
    fn wants(&self, _: &str, _: Option<usize>) -> bool {
        true
    }
    fn record(&mut self, name: &str, slot: Option<usize>, data: &[f32]) {
        let k = match slot {
            Some(s) => format!("{name}-{s:02}"),
            None => name.to_string(),
        };
        self.0.insert(k, data.to_vec());
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

#[test]
fn metal_matches_cpu_reference_on_short_prompt() {
    let Some(mp) = model_path() else {
        eprintln!("SKIP: model not found");
        return;
    };
    let tokens: Vec<u32> = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/out/short/tokens.txt"),
    )
    .map(|s| s.lines().map(|l| l.trim().parse().unwrap()).collect())
    .unwrap_or_else(|_| vec![200002, 200005, 39250, 25, 220, 15, 200009]);
    let g = gguf::Gguf::open(mp).unwrap();
    let hp = Hparams::from_gguf(&g).unwrap();
    let cpu = CpuModel::new(hp.clone(), Weights::load(&g, &hp).unwrap());
    let mut cs = State::new(&hp);
    let mut gpu = MetalModel::with_options(
        &g,
        hp.clone(),
        256,
        Options {
            kv_f16: false,
            gate_f16: false,
        },
    )
    .unwrap();

    let mut worst: BTreeMap<String, f32> = BTreeMap::new();
    for (i, &t) in tokens.iter().enumerate() {
        let (mut a, mut b) = (All::default(), All::default());
        cpu.step(t, &mut cs, &mut a).unwrap();
        gpu.step(t, &mut b).unwrap();
        // the CPU traces a few router internals the Metal path doesn't; compare what both have
        let common: Vec<_> = a.0.keys().filter(|k| b.0.contains_key(*k)).collect();
        assert!(
            common.len() >= hp.n_slots() * 11,
            "token {i}: only {} shared trace points",
            common.len()
        );
        for k in common {
            let (va, vb) = (&a.0[k], &b.0[k]);
            assert_eq!(va.len(), vb.len(), "{k}");
            let e = if k.starts_with("ffn_moe_topk") {
                0.0
            } else {
                rel_l2(vb, va)
            };
            let w = worst
                .entry(k.split('-').next().unwrap().to_string())
                .or_insert(0.0);
            *w = w.max(e);
            assert!(e < 1e-3, "token {i} {k}: rel_l2 {e:.2e}");
        }
    }
    for (k, e) in &worst {
        println!("{k:16} worst rel_l2 {e:.2e}");
    }
}

/// Records only the logits and the outputs of a few slots (SWA and full layers, prelude, loop
/// passes, coda) to keep memory flat over a long prompt.
struct Sparse(BTreeMap<String, Vec<f32>>);

impl Trace for Sparse {
    fn wants(&self, name: &str, slot: Option<usize>) -> bool {
        name == "result_output"
            || (name == "l_out" && matches!(slot, Some(0 | 5 | 11 | 17 | 23 | 30 | 47)))
    }
    fn record(&mut self, name: &str, slot: Option<usize>, data: &[f32]) {
        self.0.insert(format!("{name}-{slot:?}"), data.to_vec());
    }
}

#[test]
#[ignore = "~6 min: 1100 tokens through the CPU reference; crosses attention chunks and the 1024 window"]
fn metal_matches_cpu_reference_past_the_sliding_window() {
    let Some(mp) = model_path() else {
        eprintln!("SKIP: model not found");
        return;
    };
    let Ok(text) = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/out/swa/tokens.txt"),
    ) else {
        eprintln!("SKIP: fixtures/out/swa missing");
        return;
    };
    let tokens: Vec<u32> = text
        .lines()
        .take(1100)
        .map(|l| l.trim().parse().unwrap())
        .collect();
    let g = gguf::Gguf::open(mp).unwrap();
    let hp = Hparams::from_gguf(&g).unwrap();
    let cpu = CpuModel::new(hp.clone(), Weights::load(&g, &hp).unwrap());
    let mut cs = State::new(&hp);
    let mut gpu = MetalModel::with_options(
        &g,
        hp.clone(),
        tokens.len() + 1,
        Options {
            kv_f16: false,
            gate_f16: false,
        },
    )
    .unwrap();
    let mut worst = 0.0f32;
    for (i, &t) in tokens.iter().enumerate() {
        let (mut a, mut b) = (Sparse(BTreeMap::new()), Sparse(BTreeMap::new()));
        cpu.step(t, &mut cs, &mut a).unwrap();
        gpu.step(t, &mut b).unwrap();
        for (k, va) in &a.0 {
            let e = rel_l2(&b.0[k], va);
            worst = worst.max(e);
            assert!(e < 1e-3, "token {i} {k}: rel_l2 {e:.2e}");
        }
    }
    println!("worst rel_l2 over {} tokens: {worst:.2e}", tokens.len());
}

fn swa_tokens(n: usize) -> Option<Vec<u32>> {
    let text = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/out/swa/tokens.txt"),
    )
    .ok()?;
    Some(
        text.lines()
            .take(n)
            .map(|l| l.trim().parse().unwrap())
            .collect(),
    )
}

/// Batched prefill with f32 tiles must leave the model in the same state as token-by-token decode:
/// same last-token logits, and the same logits for the tokens decoded afterwards.
#[test]
fn prefill_matches_token_by_token_decode() {
    prefill_vs_decode(false);
}

/// Same check with an f16 KV cache on both sides. Not a valid exactness test: f16 rounding of the
/// cache perturbs near-tied router scores (e.g. layer 12, pass 2: 8th vs 9th expert 3.4e-5 apart,
/// router_margin_at_drift_point), and one flipped expert moves a token's output by ~4%. The
/// kernels themselves agree exactly on identical values (tests/attn_kernel.rs).
#[test]
#[ignore = "fails on legitimate near-tie routing flips; kept to show the effect"]
fn prefill_matches_decode_with_f16_kv() {
    prefill_vs_decode(true);
}

fn prefill_vs_decode(kv_f16: bool) {
    let Some(mp) = model_path() else {
        eprintln!("SKIP: model not found");
        return;
    };
    let Some(all) = swa_tokens(605) else {
        eprintln!("SKIP: fixtures/out/swa missing");
        return;
    };
    let g = gguf::Gguf::open(mp).unwrap();
    let hp = Hparams::from_gguf(&g).unwrap();
    for n_prompt in [7usize, 300, 600] {
        let (prompt, rest) = (&all[..n_prompt], &all[n_prompt..n_prompt + 5]);
        let mut batched = MetalModel::with_options(
            &g,
            hp.clone(),
            700,
            Options {
                kv_f16,
                gate_f16: false,
            },
        )
        .unwrap();
        batched.exact_prefill = true;
        let mut serial = MetalModel::with_options(
            &g,
            hp.clone(),
            700,
            Options {
                kv_f16,
                gate_f16: false,
            },
        )
        .unwrap();
        let mut lb = batched.prefill(prompt).unwrap();
        let mut ls = Vec::new();
        for &t in prompt {
            ls = serial.step(t, &mut inkling::NoTrace).unwrap();
        }
        let e = rel_l2(&lb, &ls);
        println!("kv_f16={kv_f16}, {n_prompt} prompt tokens: last-token logits rel_l2 {e:.2e}");
        assert!(
            e < 1e-3,
            "prefill of {n_prompt} tokens diverges from decode: {e:.2e}"
        );
        for (i, &t) in rest.iter().enumerate() {
            lb = batched.step(t, &mut inkling::NoTrace).unwrap();
            ls = serial.step(t, &mut inkling::NoTrace).unwrap();
            let e = rel_l2(&lb, &ls);
            assert!(
                e < 1e-3,
                "decode step {i} after a {n_prompt}-token prefill diverges: {e:.2e}"
            );
        }
    }
}

fn kl_div(p_logits: &[f32], q_logits: &[f32]) -> f32 {
    let lse = |x: &[f32]| {
        let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        m + x.iter().map(|v| ((v - m) as f64).exp()).sum::<f64>().ln() as f32
    };
    let (lp, lq) = (lse(p_logits), lse(q_logits));
    p_logits
        .iter()
        .zip(q_logits)
        .map(|(&a, &b)| ((a - lp) as f64).exp() * ((a - lp) - (b - lq)) as f64)
        .sum::<f64>() as f32
}

fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

/// The opt-in f16 prefill tiles must keep the next-token choice on short prompts: same top-1 and
/// KL below 1e-2 nats against decode around a 300-token prefill. Measured: KL 4e-5..1.1e-3 at 300
/// tokens, up to 6.5e-3 at 600, and 0.23 against llama.cpp at 13.5k tokens (vs 0.04 with f32
/// tiles), which is why f16 is not the default.
#[test]
fn fast_prefill_keeps_next_token_distribution() {
    let Some(mp) = model_path() else {
        eprintln!("SKIP: model not found");
        return;
    };
    let Some(all) = swa_tokens(605) else {
        eprintln!("SKIP: fixtures/out/swa missing");
        return;
    };
    let g = gguf::Gguf::open(mp).unwrap();
    let hp = Hparams::from_gguf(&g).unwrap();
    for n_prompt in [300usize] {
        let (prompt, rest) = (&all[..n_prompt], &all[n_prompt..n_prompt + 5]);
        let mut fast = MetalModel::with_options(
            &g,
            hp.clone(),
            700,
            Options {
                kv_f16: false,
                gate_f16: false,
            },
        )
        .unwrap();
        fast.exact_prefill = false;
        let mut serial = MetalModel::with_options(
            &g,
            hp.clone(),
            700,
            Options {
                kv_f16: false,
                gate_f16: false,
            },
        )
        .unwrap();
        let mut lf = fast.prefill(prompt).unwrap();
        let mut ls = Vec::new();
        for &t in prompt {
            ls = serial.step(t, &mut inkling::NoTrace).unwrap();
        }
        for step in 0..=rest.len() {
            let kl = kl_div(&ls, &lf);
            println!(
                "{n_prompt} prompt tokens, step {step}: KL {kl:.2e}, rel_l2 {:.2e}",
                rel_l2(&lf, &ls)
            );
            assert_eq!(
                argmax(&lf),
                argmax(&ls),
                "top-1 differs at step {step} after a {n_prompt}-token fast prefill"
            );
            assert!(
                kl < 1e-2,
                "KL {kl:.2e} at step {step} after a {n_prompt}-token fast prefill"
            );
            if step < rest.len() {
                lf = fast.step(rest[step], &mut inkling::NoTrace).unwrap();
                ls = serial.step(rest[step], &mut inkling::NoTrace).unwrap();
            }
        }
    }
}

/// Diagnostic: how far do f16-cache prefill and f16-cache decode each move from their f32-cache
/// versions? Plain f16 rounding of keys and values should cost ~1e-3 relative.
#[test]
#[ignore = "diagnostic"]
fn f16_kv_drift_per_path() {
    let Some(mp) = model_path() else { return };
    let Some(all) = swa_tokens(64) else { return };
    let g = gguf::Gguf::open(mp).unwrap();
    let hp = Hparams::from_gguf(&g).unwrap();
    let prompt = &all[..7];
    let mk = |kv_f16| {
        MetalModel::with_options(
            &g,
            hp.clone(),
            128,
            Options {
                kv_f16,
                gate_f16: false,
            },
        )
        .unwrap()
    };
    let (mut p16, mut p32) = (mk(true), mk(false));
    let e_prefill = rel_l2(&p16.prefill(prompt).unwrap(), &p32.prefill(prompt).unwrap());
    let (mut d16, mut d32) = (mk(true), mk(false));
    let (mut l16, mut l32) = (Vec::new(), Vec::new());
    for &t in prompt {
        l16 = d16.step(t, &mut inkling::NoTrace).unwrap();
        l32 = d32.step(t, &mut inkling::NoTrace).unwrap();
    }
    println!(
        "prefill f16 vs f32 cache: {e_prefill:.2e}; decode f16 vs f32 cache: {:.2e}",
        rel_l2(&l16, &l32)
    );
    // per-slot K cache drift between the f16- and f32-cache prefills (pure rounding: ~3e-4)
    for slot in (0usize..48).collect::<Vec<_>>() {
        let a = p16.debug_k_cache(slot, prompt.len());
        let b = p32.debug_k_cache(slot, prompt.len());
        let fa: Vec<f32> = a
            .chunks(2)
            .map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let fb: Vec<f32> = b
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        println!(
            "slot {slot:2}: K rel diff f16-cache vs f32-cache prefill {:.2e}",
            rel_l2(&fa, &fb)
        );
    }
    // per-position breakdown at the slot where the drift jumps (a routing flip would hit one token)
    for slot in [30usize, 31] {
        let a = p16.debug_k_cache(slot, prompt.len());
        let b = p32.debug_k_cache(slot, prompt.len());
        let fa: Vec<f32> = a
            .chunks(2)
            .map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let fb: Vec<f32> = b
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let kvd = hp.kv_dim();
        let per: Vec<String> = (0..prompt.len())
            .map(|t| {
                format!(
                    "{:.1e}",
                    rel_l2(&fa[t * kvd..(t + 1) * kvd], &fb[t * kvd..(t + 1) * kvd])
                )
            })
            .collect();
        println!("slot {slot} per position: {}", per.join(" "));
    }
    // compare the f16 K cache written by prefill with the one written by token-by-token decode
    for slot in [0usize, 5, 30, 47] {
        let a = p16.debug_k_cache(slot, prompt.len());
        let b = d16.debug_k_cache(slot, prompt.len());
        let diff = a.chunks(2).zip(b.chunks(2)).filter(|(x, y)| x != y).count();
        println!(
            "slot {slot}: {diff} of {} f16 K values differ between prefill and decode",
            a.len() / 2
        );
        let h = |c: &[u8]| half_to_f32(u16::from_le_bytes([c[0], c[1]]));
        let mut worst = (0usize, 0.0f32, 0.0f32);
        for (i, (x, y)) in a.chunks(2).zip(b.chunks(2)).enumerate() {
            let (fx, fy) = (h(x), h(y));
            if (fx - fy).abs() > (worst.1 - worst.2).abs() {
                worst = (i, fx, fy);
            }
        }
        let kvd = hp.kv_dim();
        println!(
            "   worst: position {} channel {}: prefill {} vs decode {}",
            worst.0 / kvd,
            worst.0 % kvd,
            worst.1,
            worst.2
        );
    }
}

fn half_to_f32(h: u16) -> f32 {
    let s = (h >> 15) as u32;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    let v = if e == 0 {
        m * 2f32.powi(-24)
    } else if e == 31 {
        f32::INFINITY
    } else {
        (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    };
    if s == 1 { -v } else { v }
}

struct RouterAt(usize, Vec<f32>);
impl Trace for RouterAt {
    fn wants(&self, name: &str, slot: Option<usize>) -> bool {
        name == "ffn_moe_logits" && slot == Some(self.0)
    }
    fn record(&mut self, _: &str, _: Option<usize>, data: &[f32]) {
        self.1 = data.to_vec();
    }
}

/// Diagnostic: router selection margin (8th vs 9th expert score) at slot 30 (layer 12, pass 2) for
/// the token where the f16-cache drift jumps, with an f16 and an f32 cache.
#[test]
#[ignore = "diagnostic"]
fn router_margin_at_drift_point() {
    let Some(mp) = model_path() else { return };
    let Some(all) = swa_tokens(64) else { return };
    let g = gguf::Gguf::open(mp).unwrap();
    let hp = Hparams::from_gguf(&g).unwrap();
    let bias = g.to_f32(g.tensor("blk.12.exp_probs_b.bias").unwrap());
    for kv_f16 in [false, true] {
        let mut m = MetalModel::with_options(
            &g,
            hp.clone(),
            128,
            Options {
                kv_f16,
                gate_f16: false,
            },
        )
        .unwrap();
        let mut r = RouterAt(30, Vec::new());
        for &t in &all[..7] {
            r.1.clear();
            m.step(t, &mut r).unwrap();
        }
        let mut sc: Vec<(usize, f32)> =
            r.1.iter()
                .enumerate()
                .map(|(e, &l)| (e, 1.0 / (1.0 + (-l).exp()) + bias[e]))
                .collect();
        sc.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!(
            "kv_f16={kv_f16}: 8th {:?} 9th {:?} margin {:.2e}; top8 {:?}",
            sc[7],
            sc[8],
            sc[7].1 - sc[8].1,
            sc[..8].iter().map(|x| x.0).collect::<Vec<_>>()
        );
    }
}
