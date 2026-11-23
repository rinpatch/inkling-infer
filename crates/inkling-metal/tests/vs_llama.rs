//! Long prompts against the llama.cpp fixtures, through the Metal path (which matches the f32 CPU
//! reference to ~1e-5, but prefills thousands of tokens in seconds instead of an hour).
//! Checks the last-token logits and the greedy continuation. The swa prompt (3810 tokens) crosses
//! the 1024 window many times; the long one (13571) runs past position 8192, where the
//! log-position scale tau on the full-attention layers rises above 1.
//!
//! What is compared: llama.cpp's CPU backend quantizes activations to Q8_0 before every quantized
//! matmul. That adds ~1% noise per layer, and the 48 layer applications (24 of them in the 3-pass
//! loop) amplify it: deep layer outputs differ by 5-15% at every position (examples/drift.rs
//! shows it is flat in position, not a windowing or tau bug), and raw logits by 5-10% at the end
//! of long prompts. So the checks are on the next-token distribution: KL divergence and top-1 at
//! the last prompt token, and greedy agreement afterwards except at llama.cpp's own near-ties.

use std::fs;
use std::path::{Path, PathBuf};

use inkling::Hparams;
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

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

/// KL(p || q) in nats between the softmax distributions of two logit vectors.
fn kl_div(p_logits: &[f32], q_logits: &[f32]) -> f32 {
    let lse = |x: &[f32]| {
        let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        m + x.iter().map(|v| ((v - m) as f64).exp()).sum::<f64>().ln() as f32
    };
    let (lp, lq) = (lse(p_logits), lse(q_logits));
    p_logits
        .iter()
        .zip(q_logits)
        .map(|(&a, &b)| {
            let pa = ((a - lp) as f64).exp();
            pa * ((a - lp) - (b - lq)) as f64
        })
        .sum::<f64>() as f32
}

fn argmax(x: &[f32]) -> u32 {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

fn check(name: &str, greedy_steps: usize) {
    let Some(mp) = model_path() else {
        eprintln!("SKIP: model not found");
        return;
    };
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/out")
        .join(name);
    let (Ok(toks), Ok(lb), Ok(greedy)) = (
        fs::read_to_string(dir.join("tokens.txt")),
        fs::read(dir.join("logits_last.f32")),
        fs::read_to_string(dir.join("greedy.tsv")),
    ) else {
        eprintln!("SKIP: fixtures/out/{name} missing");
        return;
    };
    let tokens: Vec<u32> = toks.lines().map(|l| l.trim().parse().unwrap()).collect();
    let want: Vec<f32> = lb
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let g = gguf::Gguf::open(mp).unwrap();
    let hp = Hparams::from_gguf(&g).unwrap();
    // single positions swing with near-tied routing flips, so these checks pin the exact settings
    // (f32 cache, f32 tiles); tests/quality.rs measures the default precision over all positions
    let mut m = MetalModel::with_options(
        &g,
        hp.clone(),
        tokens.len() + greedy_steps + 1,
        Options {
            kv_f16: false,
            gate_f16: false,
        },
    )
    .unwrap();
    m.exact_prefill = true;
    let t0 = std::time::Instant::now();
    let mut logits = m.prefill(&tokens).unwrap();
    println!(
        "{name}: prefilled {} tokens in {:.1}s",
        tokens.len(),
        t0.elapsed().as_secs_f32()
    );

    let e = rel_l2(&logits, &want[..hp.n_vocab]);
    let kl = kl_div(&want[..hp.n_vocab], &logits);
    println!(
        "{name}: last-token logits rel_l2 {e:.2e}, KL(llama.cpp || ours) {kl:.2e} nats, top-1 {} vs {}",
        argmax(&logits),
        argmax(&want)
    );
    assert_eq!(
        argmax(&logits),
        argmax(&want),
        "{name}: different top-1 at the last prompt token"
    );
    assert!(
        kl < 0.05,
        "{name}: next-token distributions differ (KL {kl:.3})"
    );

    let mut near_ties = 0;
    for (step, line) in greedy.lines().take(greedy_steps).enumerate() {
        let f: Vec<&str> = line.split('\t').collect();
        let want_tok: u32 = f[1].parse().unwrap();
        let got = argmax(&logits);
        if got != want_tok {
            let top: Vec<(u32, f32)> = f[3]
                .split_whitespace()
                .map(|p| {
                    let (a, b) = p.split_once(':').unwrap();
                    (a.parse().unwrap(), b.parse().unwrap())
                })
                .collect();
            let margin = top[0].1
                - top
                    .iter()
                    .find(|t| t.0 == got)
                    .map(|t| t.1)
                    .unwrap_or(f32::NEG_INFINITY);
            // a swap is acceptable only where llama.cpp itself had the two tokens within 0.5 logits
            assert!(
                margin < 0.5,
                "{name} greedy step {step}: got {got}, llama.cpp picked {want_tok} (llama.cpp margin {margin:.4})"
            );
            near_ties += 1;
        }
        logits = m.step(want_tok, &mut inkling::NoTrace).unwrap();
    }
    println!(
        "{name}: greedy matched llama.cpp for {greedy_steps} tokens ({near_ties} near-tie swaps)"
    );
}

#[test]
fn swa_prompt_matches_llama_via_metal() {
    check("swa", 32);
}

#[test]
fn long_prompt_matches_llama_via_metal() {
    check("long", 32);
}
