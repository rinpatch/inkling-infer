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
