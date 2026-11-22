//! Metal decode vs the f32 CPU reference on the same tokens. Both compute in f32, so every traced
//! tensor must agree to rounding (reduction order differs), far tighter than the llama.cpp fixtures.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use inkling::cpu::{CpuModel, State};
use inkling::weights::Weights;
use inkling::{Hparams, Trace};
use inkling_metal::model::MetalModel;

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
    let mut gpu = MetalModel::new(&g, hp.clone(), 256).unwrap();

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
    let mut gpu = MetalModel::new(&g, hp.clone(), tokens.len() + 1).unwrap();
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

/// Batched prefill must leave the model in the same state as token-by-token decode: same
/// last-token logits, and the same logits for the tokens decoded afterwards.
#[test]
fn prefill_matches_token_by_token_decode() {
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
        let mut batched = MetalModel::new(&g, hp.clone(), 700).unwrap();
        let mut serial = MetalModel::new(&g, hp.clone(), 700).unwrap();
        let mut lb = batched.prefill(prompt).unwrap();
        let mut ls = Vec::new();
        for &t in prompt {
            ls = serial.step(t, &mut inkling::NoTrace).unwrap();
        }
        let e = rel_l2(&lb, &ls);
        println!("{n_prompt} prompt tokens: last-token logits rel_l2 {e:.2e}");
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
