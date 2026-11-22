//! Metal drafter vs the CPU reference on the same committed features: context keys, block hidden,
//! Markov feature and drafts.
//! `cargo run --release -p inkling-metal --example dspark_debug -- <model.gguf> <native dir> <prompt.txt>`

use inkling::Hparams;
use inkling::drafter::{self as dc, DraftState, Drafter};
use inkling::ops::rms_norm_w;
use inkling_metal::model::MetalModel;

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let n: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let d: f32 = b.iter().map(|y| y * y).sum();
    (n / d.max(1e-30)).sqrt()
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = gguf::Gguf::open(&a[1])?;
    let hp = Hparams::from_gguf(&g)?;
    let w = inkling::weights::Weights::load(&g, &hp)?;
    let tok = inkling::tokenizer::Tokenizer::from_gguf(&g)?;
    let mut prompt = tok.encode(&std::fs::read_to_string(&a[3])?);
    prompt.truncate(300);
    let st = gguf::safetensors::SafeTensors::open(&a[2])?;
    let dr = Drafter::load(&st, hp.n_vocab_padded)?;

    let mut m = MetalModel::new(&g, hp.clone(), 400)?;
    m.load_drafter(&a[2])?;
    let (head, last) = prompt.split_at(prompt.len() - 1);
    // CPU state from the same captured features, one 256-row chunk at a time like prefill
    let mut ds = DraftState::default();
    for chunk in head.chunks(256) {
        m.prefill(chunk)?;
        for r in 0..chunk.len() {
            dr.commit(&mut ds, &m.captured(r));
        }
    }
    let anchor = last[0];
    let dbg = m.debug_draft(anchor)?;
    let kvd = dc::N_KV * dc::HEAD_DIM;
    println!(
        "ctx key pos 0:      rel {:.2e}",
        rel(&dbg.k0_first, &ds.ctx_k[0][0][..kvd])
    );
    println!(
        "ctx key pos last:   rel {:.2e}",
        rel(&dbg.k0_last, &ds.ctx_k[0][head.len() - 1][..kvd])
    );

    let mut e = vec![0.0f32; hp.n_embd];
    gguf::dequantize_row(w.tok_embd.dtype, w.tok_embd.row(anchor as usize), &mut e);
    let emb = rms_norm_w(&e, &w.tok_norm, hp.rms_eps);
    let h = dr.block_hidden(&ds, &emb);
    for (r, hr) in h.iter().enumerate() {
        println!(
            "block hidden row {}: rel {:.2e}",
            r + 1,
            rel(&dbg.h[r * hp.n_embd..(r + 1) * hp.n_embd], hr)
        );
    }
    let lm_head = |x: &[f32]| w.output.slice_rows(0, hp.n_vocab).matvec_new(x);
    let cpu = dr.walk_topk(&ds, &h, anchor, &lm_head, Some(8));
    println!("drafts metal {:?}\ndrafts cpu   {:?}", dbg.drafts, cpu);
    println!("metal top-8 of row 0: {:?}", &dbg.cand[..8]);
    let base0 = lm_head(&h[0]);
    let mut idx: Vec<usize> = (0..base0.len()).collect();
    idx.sort_by(|a, b| base0[*b].total_cmp(&base0[*a]));
    println!("cpu   top-8 of row 0: {:?}", &idx[..8]);
    Ok(())
}
