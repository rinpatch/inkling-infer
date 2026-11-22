//! Pipelined greedy decode (GPU argmax, steps queued back to back) must pick exactly the tokens
//! that step-by-step decode with a CPU argmax picks.

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

/// Lowest index on ties, like the argmax_token kernel.
fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as u32
}

#[test]
fn pipelined_greedy_matches_step_by_step() {
    let Some(mp) = model_path() else {
        eprintln!("SKIP: model not found");
        return;
    };
    let toks = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/out/short/tokens.txt"),
    );
    let Ok(toks) = toks else {
        eprintln!("SKIP: fixtures/out/short missing");
        return;
    };
    let prompt: Vec<u32> = toks.lines().map(|l| l.trim().parse().unwrap()).collect();
    let g = gguf::Gguf::open(mp).unwrap();
    let hp = Hparams::from_gguf(&g).unwrap();
    let n = 48;
    let mut m = MetalModel::new(&g, hp.clone(), prompt.len() + n + 1).unwrap();

    let logits = m.prefill(&prompt[..prompt.len() - 1]).unwrap();
    drop(logits);
    let last = *prompt.last().unwrap();
    let mut want = Vec::new();
    let mut next = last;
    for _ in 0..n {
        let lg = m.step(next, &mut inkling::NoTrace).unwrap();
        next = argmax(&lg);
        want.push(next);
    }

    m.reset();
    m.prefill(&prompt[..prompt.len() - 1]).unwrap();
    let got = m.decode_greedy(last, n).unwrap();
    assert_eq!(got, want);
    assert_eq!(m.pos(), prompt.len() - 1 + n);
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

/// verify(8 tokens) + rewind(keep) + one more step must match a model that only ever saw the kept
/// prefix: same logits for every verify row, and the same next-step logits (KV, conv state, pos).
#[test]
fn verify_then_rewind_matches_the_kept_prefix() {
    let Some(mp) = model_path() else {
        eprintln!("SKIP: model not found");
        return;
    };
    let Ok(toks) = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/out/short/tokens.txt"),
    ) else {
        eprintln!("SKIP: fixtures/out/short missing");
        return;
    };
    let prompt: Vec<u32> = toks.lines().map(|l| l.trim().parse().unwrap()).collect();
    let g = gguf::Gguf::open(mp).unwrap();
    let hp = Hparams::from_gguf(&g).unwrap();
    let (head, batch) = prompt.split_at(prompt.len() - inkling_metal::model::MAX_VERIFY);
    let next = 4242u32;
    // f32 cache and tiles: with f16 storage, batch and decode can flip near-tied expert choices
    let mut m =
        MetalModel::with_options(&g, hp.clone(), prompt.len() + 4, Options { kv_f16: false })
            .unwrap();
    m.exact_prefill = true;
    for keep in [1, 3, batch.len() - 1, batch.len()] {
        // reference: the kept prefix token by token, then `next`
        m.reset();
        m.prefill(head).unwrap();
        let mut want_rows = Vec::new();
        for &t in &batch[..keep] {
            want_rows.push(m.step(t, &mut inkling::NoTrace).unwrap());
        }
        let want_next = m.step(next, &mut inkling::NoTrace).unwrap();

        m.reset();
        m.prefill(head).unwrap();
        let rows = m.verify(batch).unwrap();
        for (i, w) in want_rows.iter().enumerate() {
            let e = rel_l2(&rows[i], w);
            assert!(
                e < 1e-4,
                "keep {keep}: verify row {i} differs from decode (rel_l2 {e:.2e})"
            );
        }
        m.rewind(batch.len(), keep).unwrap();
        assert_eq!(m.pos(), head.len() + keep);
        let got_next = m.step(next, &mut inkling::NoTrace).unwrap();
        let e = rel_l2(&got_next, &want_next);
        println!("keep {keep}: next-step rel_l2 {e:.2e}");
        assert!(
            e < 1e-4,
            "keep {keep}: state after rewind differs (rel_l2 {e:.2e})"
        );
    }
}
