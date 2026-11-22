//! How often would the bundled DSpark drafter's guesses be accepted by *our* GGUF model?
//! The target (Metal, IQ4_NL) decodes greedily token by token and supplies the drafter's features;
//! the drafter is the f32 CPU reference (inkling::drafter). Each round drafts up to 7 tokens from
//! the committed state and counts how many leading drafts match the target's own greedy tokens.
//!
//! `cargo run --release -p inkling-metal --example dspark_accept -- <model.gguf> <native snapshot dir> <prompt.txt> [n_gen] [n_prompt]`

use inkling::drafter::{self, DraftState, Drafter};
use inkling::ops::rms_norm_w;
use inkling::{Hparams, NoTrace};
use inkling_metal::model::MetalModel;

fn argmax(x: &[f32]) -> u32 {
    let mut b = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[b] {
            b = i;
        }
    }
    b as u32
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (gguf_path, native, prompt_path) = (&a[1], &a[2], &a[3]);
    let n_gen: usize = a.get(4).map_or(256, |s| s.parse().unwrap());
    let n_prompt: usize = a.get(5).map_or(512, |s| s.parse().unwrap());

    let g = gguf::Gguf::open(gguf_path)?;
    let hp = Hparams::from_gguf(&g)?;
    let w = inkling::weights::Weights::load(&g, &hp)?;
    let tok = inkling::tokenizer::Tokenizer::from_gguf(&g)?;
    let mut prompt = tok.encode(&std::fs::read_to_string(prompt_path)?);
    prompt.truncate(n_prompt);

    let st = gguf::safetensors::SafeTensors::open(native)?;
    let t0 = std::time::Instant::now();
    let dr = Drafter::load(&st, w.output.rows)?;
    eprintln!("drafter loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let lm_head = |x: &[f32]| w.output.slice_rows(0, hp.n_vocab).matvec_new(x);
    let embed = |t: u32| {
        let mut e = vec![0.0f32; hp.n_embd];
        gguf::dequantize_row(w.tok_embd.dtype, w.tok_embd.row(t as usize), &mut e);
        rms_norm_w(&e, &w.tok_norm, hp.rms_eps)
    };

    let mut m = MetalModel::new(&g, hp.clone(), prompt.len() + n_gen + 16)?;
    m.enable_capture()?;
    let mut ds = DraftState::default();
    // prompt: one ubatch at a time, committing every position's features
    let mut last_logits = Vec::new();
    for chunk in prompt.chunks(256) {
        let mut lg = Vec::new();
        m.prefill_each(chunk, |_, l| lg = l.to_vec())?;
        for r in 0..chunk.len() {
            dr.commit(&mut ds, &m.captured(r));
        }
        last_logits = lg;
    }
    let mut anchor = argmax(&last_logits);

    let (mut generated, mut rounds, mut accepted_total) = (0usize, 0usize, 0usize);
    let mut hist = [0usize; drafter::BLOCK];
    let t0 = std::time::Instant::now();
    while generated < n_gen {
        let h = dr.block_hidden(&ds, &embed(anchor));
        let drafts = dr.walk(&ds, &h, anchor, &lm_head);
        // the target's greedy tokens after the anchor, as far as the drafts keep matching
        let mut cur = anchor;
        let mut n_acc = 0;
        loop {
            let lg = m.step(cur, &mut NoTrace)?;
            dr.commit(&mut ds, &m.captured(0));
            let next = argmax(&lg);
            generated += 1;
            cur = next;
            if n_acc < drafts.len() && drafts[n_acc] == next {
                n_acc += 1;
            } else {
                break;
            }
        }
        hist[n_acc] += 1;
        accepted_total += n_acc;
        rounds += 1;
        anchor = cur;
    }
    let per_round = generated as f64 / rounds as f64;
    println!(
        "{} prompt tokens, {generated} generated in {rounds} rounds ({:.0}s)",
        prompt.len(),
        t0.elapsed().as_secs_f32()
    );
    println!(
        "tokens per round {per_round:.2}  accepted drafts per round {:.2}  per-draft acceptance {:.2}",
        accepted_total as f64 / rounds as f64,
        accepted_total as f64 / (rounds * (drafter::BLOCK - 1)) as f64
    );
    println!("accepted-count histogram (0..7): {hist:?}");
    Ok(())
}
