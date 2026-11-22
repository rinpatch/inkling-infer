//! Tokenizer must reproduce the official HF tokenizer (tokenizer.json) on every
//! fixture prompt, and decode must round-trip.

use std::fs;
use std::path::{Path, PathBuf};

use inkling::tokenizer::Tokenizer;

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

#[test]
fn tokenization_matches_hf_on_fixture_prompts() {
    let Some(mp) = model_path() else {
        eprintln!("SKIP: model not found");
        return;
    };
    let g = gguf::Gguf::open(mp).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let mut checked = 0;
    for name in ["short", "swa", "long"] {
        let Ok(want) = fs::read_to_string(fixtures.join("out").join(name).join("tokens_hf.txt"))
        else {
            eprintln!("SKIP {name}: no fixture");
            continue;
        };
        let want: Vec<u32> = want.lines().map(|l| l.trim().parse().unwrap()).collect();
        let text =
            fs::read_to_string(fixtures.join("prompts").join(format!("{name}.txt"))).unwrap();
        let got = tok.encode(&text);
        if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
            panic!(
                "{name}: first mismatch at token {i}: got {} {:?}, HF {} {:?}",
                got[i],
                tok.decode(&[got[i]]),
                want[i],
                tok.decode(&[want[i]])
            );
        }
        assert_eq!(got.len(), want.len(), "{name}: token count");
        assert_eq!(tok.decode(&got), text, "{name}: decode round-trip");
        checked += got.len();
    }
    println!("matched {checked} tokens");
}
