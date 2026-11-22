//! Compares the CPU reference against llama.cpp fixtures (tools/llama-dump, fixtures/out/*).
//! Needs the IQ4_NL GGUF and generated fixtures; tests skip with a message when either is missing.
//! The swa/long tests take ~15 min / ~1 h and are `#[ignore]`d: `cargo nextest run --run-ignored all`.
//!
//! Tolerances: llama.cpp's CPU backend quantizes activations to Q8_0 before every Q8_0/IQ4_NL
//! matmul, our reference keeps them in f32. That gives a few percent relative error per layer
//! output that stays bounded (it does not compound), and occasional expert swaps where two router
//! scores are nearly tied.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use inkling::cpu::{CpuModel, State};
use inkling::weights::Weights;
use inkling::{Hparams, Trace};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

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

struct Fixture {
    dir: PathBuf,
    tokens: Vec<u32>,
    /// name -> files of each ubatch in order; per-token tensors concatenate along the token axis
    tensors: HashMap<String, Vec<PathBuf>>,
}

impl Fixture {
    fn load(name: &str) -> Option<Fixture> {
        let dir = repo().join("fixtures/out").join(name);
        let tokens = fs::read_to_string(dir.join("tokens.txt"))
            .ok()?
            .lines()
            .map(|l| l.trim().parse().unwrap())
            .collect();
        let mut tensors: HashMap<String, Vec<(usize, PathBuf)>> = HashMap::new();
        for line in fs::read_to_string(dir.join("manifest.tsv")).ok()?.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f[0].contains('#') {
                continue;
            }
            tensors
                .entry(f[0].to_string())
                .or_default()
                .push((f[1].parse().unwrap(), dir.join(f[6])));
        }
        let tensors = tensors
            .into_iter()
            .map(|(k, mut v)| {
                v.sort_by_key(|x| x.0);
                (k, v.into_iter().map(|x| x.1).collect())
            })
            .collect();
        Some(Fixture {
            dir,
            tokens,
            tensors,
        })
    }

    fn read(&self, name: &str) -> Vec<f32> {
        let mut out = Vec::new();
        for p in &self.tensors[name] {
            let bytes = fs::read(p).unwrap();
            out.extend(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
        }
        out
    }

    /// greedy.tsv: (token, top-10 (id, logit))
    fn greedy(&self) -> Vec<(u32, Vec<(u32, f32)>)> {
        fs::read_to_string(self.dir.join("greedy.tsv"))
            .unwrap()
            .lines()
            .map(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                let top = f[3].split_whitespace().map(|p| {
                    let (a, b) = p.split_once(':').unwrap();
                    (a.parse().unwrap(), b.parse().unwrap())
                });
                (f[1].parse().unwrap(), top.collect())
            })
            .collect()
    }
}

/// Records every traced tensor whose fixture exists, one row per step.
struct Recorder<'f> {
    fixture: &'f Fixture,
    rows: HashMap<String, Vec<Vec<f32>>>,
}

fn key(name: &str, slot: Option<usize>) -> String {
    match slot {
        Some(s) => format!("{name}-{s}"),
        None => name.to_string(),
    }
}

impl Trace for Recorder<'_> {
    fn wants(&self, name: &str, slot: Option<usize>) -> bool {
        self.fixture.tensors.contains_key(&key(name, slot))
    }
    fn record(&mut self, name: &str, slot: Option<usize>, data: &[f32]) {
        self.rows
            .entry(key(name, slot))
            .or_default()
            .push(data.to_vec());
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

/// Fraction of (token, expert) choices in `got` that are absent from the same token's set in `want`.
fn expert_set_mismatch(got: &[f32], want: &[f32], k: usize) -> f32 {
    let bad: usize = got
        .chunks(k)
        .zip(want.chunks(k))
        .map(|(g, w)| g.iter().filter(|e| !w.contains(e)).count())
        .sum();
    bad as f32 / got.len() as f32
}

fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

/// Sort key that orders `l_out-9` before `l_out-10`.
fn natural(s: &str) -> (String, usize) {
    match s.rsplit_once('-') {
        Some((a, b)) if b.parse::<usize>().is_ok() => (a.to_string(), b.parse().unwrap()),
        _ => (s.to_string(), 0),
    }
}

struct Tolerance {
    l_out: f32,
    result_norm: f32,
    expert_mismatch: f32,
    greedy_steps: usize,
}

fn check_fixture(name: &str, tol: Tolerance) {
    let Some(mp) = model_path() else {
        eprintln!("SKIP: model not found (set INKLING_MODEL)");
        return;
    };
    let Some(fx) = Fixture::load(name) else {
        eprintln!("SKIP: fixtures/out/{name} missing (see fixtures/README.md)");
        return;
    };
    let g = gguf::Gguf::open(mp).unwrap();
    let hp = Hparams::from_gguf(&g).unwrap();
    let model = CpuModel::new(hp.clone(), Weights::load(&g, &hp).unwrap());
    let mut state = State::new(&hp);
    let mut rec = Recorder {
        fixture: &fx,
        rows: HashMap::new(),
    };
    let t0 = std::time::Instant::now();
    let mut logits = Vec::new();
    for &t in &fx.tokens {
        logits = model.step(t, &mut state, &mut rec).unwrap();
    }
    println!(
        "{name}: {} prompt tokens in {:.1}s",
        fx.tokens.len(),
        t0.elapsed().as_secs_f32()
    );

    let mut report: Vec<(String, f32)> = Vec::new();
    let mut topk: Vec<f32> = Vec::new();
    for (key, rows) in &rec.rows {
        let want = fx.read(key);
        let per = rows[0].len();
        // `result_*` only exists for the last prompt token in llama.cpp
        let got: Vec<f32> = if want.len() == per {
            rows.last().unwrap().clone()
        } else {
            rows.concat()
        };
        assert_eq!(
            got.len(),
            want.len(),
            "{key}: {} values vs {} in the fixture",
            got.len(),
            want.len()
        );
        if key.starts_with("ffn_moe_topk") {
            topk.push(expert_set_mismatch(&got, &want, per));
        } else {
            report.push((key.clone(), rel_l2(&got, &want)));
        }
    }
    assert!(!report.is_empty(), "no tensors compared");
    report.sort_by_key(|a| natural(&a.0));
    for (n, e) in &report {
        println!("{n:28} rel_l2 = {e:.2e}");
    }

    let worst = |prefix: &str| {
        report
            .iter()
            .filter(|(n, _)| n.starts_with(prefix))
            .map(|r| r.1)
            .fold(0.0, f32::max)
    };
    let (l_out, result) = (worst("l_out"), worst("result_norm"));
    println!("worst l_out {l_out:.2e}, result_norm {result:.2e}");
    assert!(
        l_out < tol.l_out,
        "a layer output diverged from llama.cpp ({l_out:.2e})"
    );
    assert!(
        result < tol.result_norm,
        "final hidden state diverged ({result:.2e})"
    );
    if !topk.is_empty() {
        let avg = topk.iter().sum::<f32>() / topk.len() as f32;
        println!(
            "expert choices differing from llama.cpp: mean {:.2}%",
            100.0 * avg
        );
        assert!(
            avg < tol.expert_mismatch,
            "expert selection disagrees with llama.cpp too often"
        );
    }

    // greedy continuation from the same state, teacher-forced with llama.cpp's tokens
    for (step, (want, top)) in fx.greedy().into_iter().take(tol.greedy_steps).enumerate() {
        let got = argmax(&logits) as u32;
        if got != want {
            let margin = top[0].1
                - top
                    .iter()
                    .find(|t| t.0 == got)
                    .map(|t| t.1)
                    .unwrap_or(f32::NEG_INFINITY);
            panic!(
                "{name} step {step}: got {got}, llama.cpp picked {want} (llama margin {margin:.4})"
            );
        }
        logits = model.step(want, &mut state, &mut inkling::NoTrace).unwrap();
    }
    println!(
        "{name}: greedy matched llama.cpp for {} tokens",
        tol.greedy_steps
    );
}

#[test]
fn short_prompt_matches_llama() {
    check_fixture(
        "short",
        Tolerance {
            l_out: 0.1,
            result_norm: 0.05,
            expert_mismatch: 0.03,
            greedy_steps: 64,
        },
    );
}

#[test]
#[ignore = "~15 min: 3810 tokens through the f32 reference"]
fn swa_prompt_matches_llama() {
    check_fixture(
        "swa",
        Tolerance {
            l_out: 0.1,
            result_norm: 0.05,
            expert_mismatch: 0.03,
            greedy_steps: 32,
        },
    );
}

#[test]
#[ignore = "~1 h: 13571 tokens through the f32 reference"]
fn long_prompt_matches_llama() {
    check_fixture(
        "long",
        Tolerance {
            l_out: 0.1,
            result_norm: 0.05,
            expert_mismatch: 0.03,
            greedy_steps: 32,
        },
    );
}
