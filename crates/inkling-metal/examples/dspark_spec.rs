//! End-to-end speculative greedy decoding with the CPU reference drafter, checked token for token
//! against plain greedy decode. Validates verify / rewind / feature commit; the CPU drafter makes
//! it slow, so it says nothing about speed.
//!
//! `cargo run --release -p inkling-metal --example dspark_spec -- <model.gguf> <native snapshot dir> <prompt.txt> [n_gen] [n_prompt]`

use inkling::Hparams;
use inkling::drafter::{DraftState, Drafter};
use inkling::ops::rms_norm_w;
use inkling_metal::model::{MAX_VERIFY, MetalModel};

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
    let n_gen: usize = a.get(4).map_or(96, |s| s.parse().unwrap());
    let n_prompt: usize = a.get(5).map_or(256, |s| s.parse().unwrap());

    let g = gguf::Gguf::open(gguf_path)?;
    let hp = Hparams::from_gguf(&g)?;
    let w = inkling::weights::Weights::load(&g, &hp)?;
    let tok = inkling::tokenizer::Tokenizer::from_gguf(&g)?;
    let mut prompt = tok.encode(&std::fs::read_to_string(prompt_path)?);
    prompt.truncate(n_prompt);
    let st = gguf::safetensors::SafeTensors::open(native)?;
    let dr = Drafter::load(&st, w.output.rows)?;
    let lm_head = |x: &[f32]| w.output.slice_rows(0, hp.n_vocab).matvec_new(x);
    let embed = |t: u32| {
        let mut e = vec![0.0f32; hp.n_embd];
        gguf::dequantize_row(w.tok_embd.dtype, w.tok_embd.row(t as usize), &mut e);
        rms_norm_w(&e, &w.tok_norm, hp.rms_eps)
    };
    let cap = prompt.len() + n_gen + MAX_VERIFY + 1;
    let (head, last) = prompt.split_at(prompt.len() - 1);

    // reference: plain greedy decode
    let mut m = MetalModel::new(&g, hp.clone(), cap)?;
    m.prefill(head)?;
    let want = m.decode_greedy(last[0], n_gen)?;

    // speculative: draft 7, verify [anchor, d1..d7] in one batch, keep the agreeing prefix
    m.reset();
    m.enable_capture()?;
    let mut ds = DraftState::default();
    for chunk in head.chunks(256) {
        m.prefill_each(chunk, |_, _| {})?;
        for r in 0..chunk.len() {
            dr.commit(&mut ds, &m.captured(r));
        }
    }
    let mut anchor = last[0];
    let mut got: Vec<u32> = Vec::new();
    let mut rounds = 0;
    while got.len() < n_gen {
        assert_eq!(
            ds.len(),
            m.pos(),
            "draft context out of sync with the target"
        );
        let h = dr.block_hidden(&ds, &embed(anchor));
        let drafts = dr.walk(&ds, &h, anchor, &lm_head);
        let mut batch = vec![anchor];
        batch.extend_from_slice(&drafts);
        let rows = m.verify(&batch)?;
        let mut n = 0;
        while n < drafts.len() && argmax(&rows[n]) == drafts[n] {
            n += 1;
        }
        let bonus = argmax(&rows[n]);
        let keep = n + 1;
        // features of the kept rows, read before rewind reuses nothing but the conv snapshots
        for r in 0..keep {
            dr.commit(&mut ds, &m.captured(r));
        }
        m.rewind(batch.len(), keep)?;
        got.extend_from_slice(&drafts[..n]);
        got.push(bonus);
        anchor = bonus;
        rounds += 1;
    }
    got.truncate(n_gen);
    let first_diff = got.iter().zip(&want).position(|(a, b)| a != b);
    println!(
        "{n_gen} tokens in {rounds} rounds ({:.2} per round); first difference vs greedy: {first_diff:?}",
        n_gen as f64 / rounds as f64
    );
    assert_eq!(got, want, "speculative output differs from greedy decode");
    println!("identical to greedy decode");
    Ok(())
}
