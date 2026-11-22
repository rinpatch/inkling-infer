//! Multi-position quality: next-token KL at every prompt position (truncated to the reference's
//! top-128 tokens with full-vocabulary normalization, the method of the JANGH2 model card),
//! reported as median / mean / p99 / max plus top-1 agreement. One position alone is too noisy:
//! near-tied MoE routing decisions flip under tiny perturbations (tests/vs_cpu.rs
//! router_margin_at_drift_point), so single-position numbers swing.
//!
//! Reference fixture: fixtures/out/swa_topk/topk.bin from
//! `tools/llama-dump/llama-dump M fixtures/prompts/swa.txt fixtures/out/swa_topk 1 '^$' 512 0 10 1`.

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

/// Per position: log-normalizer and top-128 (id, logit), highest first.
struct Top {
    lse: f32,
    ids: Vec<u32>,
    logits: Vec<f32>,
}

const K: usize = 128;

fn read_topk(path: &Path) -> Vec<Top> {
    let b = fs::read(path).unwrap();
    let rec = 4 + K * 8;
    b.chunks_exact(rec)
        .map(|r| {
            let f = |o: usize| f32::from_le_bytes(r[o..o + 4].try_into().unwrap());
            let u = |o: usize| u32::from_le_bytes(r[o..o + 4].try_into().unwrap());
            Top {
                lse: f(0),
                ids: (0..K).map(|i| u(4 + i * 8)).collect(),
                logits: (0..K).map(|i| f(8 + i * 8)).collect(),
            }
        })
        .collect()
}

fn lse(x: &[f32]) -> f32 {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    m + x.iter().map(|v| ((v - m) as f64).exp()).sum::<f64>().ln() as f32
}

fn top_of(x: &[f32]) -> Top {
    let mut idx: Vec<u32> = (0..x.len() as u32).collect();
    idx.select_nth_unstable_by(K, |a, b| x[*b as usize].total_cmp(&x[*a as usize]));
    idx.truncate(K);
    idx.sort_by(|a, b| x[*b as usize].total_cmp(&x[*a as usize]));
    Top {
        lse: lse(x),
        logits: idx.iter().map(|&i| x[i as usize]).collect(),
        ids: idx,
    }
}

/// KL(ref || ours) over the reference's top-128 tokens.
fn kl(r: &Top, ours: &[f32], ours_lse: f32) -> f32 {
    r.ids
        .iter()
        .zip(&r.logits)
        .map(|(&id, &l)| {
            let lp = (l - r.lse) as f64;
            let lq = (ours[id as usize] - ours_lse) as f64;
            lp.exp() * (lp - lq)
        })
        .sum::<f64>() as f32
}

fn summary(name: &str, kls: &mut [f32], top1: usize) -> f32 {
    kls.sort_by(|a, b| a.total_cmp(b));
    let n = kls.len();
    let mean = kls.iter().sum::<f32>() / n as f32;
    println!(
        "{name:44} median {:.2e}  mean {:.2e}  p99 {:.2e}  max {:.2e}  top-1 {:.2}%  ({n} positions)",
        kls[n / 2],
        mean,
        kls[n * 99 / 100],
        kls[n - 1],
        100.0 * top1 as f32 / n as f32
    );
    mean
}

#[test]
#[ignore = "~2 min: three full prefills with all-position logits"]
fn next_token_kl_over_all_positions() {
    let Some(mp) = model_path() else {
        eprintln!("SKIP: model not found");
        return;
    };
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/out/swa_topk");
    let Ok(toks) = fs::read_to_string(dir.join("tokens.txt")) else {
        eprintln!("SKIP: fixtures/out/swa_topk missing");
        return;
    };
    let tokens: Vec<u32> = toks.lines().map(|l| l.trim().parse().unwrap()).collect();
    let llama = read_topk(&dir.join("topk.bin"));
    assert_eq!(llama.len(), tokens.len());
    let g = gguf::Gguf::open(mp).unwrap();
    let hp = Hparams::from_gguf(&g).unwrap();

    let run = |exact: bool, kv_f16: bool, reference: Option<&[Top]>| -> (Vec<Top>, f32, f32) {
        let mut m =
            MetalModel::with_options(&g, hp.clone(), tokens.len() + 1, Options { kv_f16 }).unwrap();
        m.exact_prefill = exact;
        let (mut kl_llama, mut kl_ref, mut top1_llama, mut top1_ref) =
            (Vec::new(), Vec::new(), 0usize, 0usize);
        let mut tops = Vec::new();
        m.prefill_each(&tokens, |p, lg| {
            let t = top_of(lg);
            kl_llama.push(kl(&llama[p], lg, t.lse));
            top1_llama += (t.ids[0] == llama[p].ids[0]) as usize;
            if let Some(r) = reference {
                kl_ref.push(kl(&r[p], lg, t.lse));
                top1_ref += (t.ids[0] == r[p].ids[0]) as usize;
            }
            tops.push(t);
        })
        .unwrap();
        let label = format!(
            "{} tiles, {} KV",
            if exact { "f32" } else { "f16" },
            if kv_f16 { "f16" } else { "f32" }
        );
        let ml = summary(&format!("{label}: vs llama.cpp"), &mut kl_llama, top1_llama);
        let mr = if reference.is_some() {
            summary(&format!("{label}: vs our exact run"), &mut kl_ref, top1_ref)
        } else {
            0.0
        };
        (tops, ml, mr)
    };

    let (exact, ml_exact, _) = run(true, false, None);
    let (_, ml_tiles, mr_tiles) = run(false, false, Some(&exact));
    let (_, ml_kv, mr_kv) = run(true, true, Some(&exact));
    // a precision mode is acceptable when it moves us from our exact run by much less than two correct
    // engines (ours and llama.cpp) differ from each other
    println!(
        "mean KL vs llama.cpp: exact {ml_exact:.2e}, f16 tiles {ml_tiles:.2e}, f16 KV {ml_kv:.2e}"
    );
    println!("mean KL vs our exact run: f16 tiles {mr_tiles:.2e}, f16 KV {mr_kv:.2e}");
}
