//! How full are the grouped MoE GEMM's 32-pair tiles in a prefill batch? Routes a 512-token batch
//! and reports, for the last layer, the pairs per expert and the tile utilization.
use inkling::Hparams;
use inkling_metal::model::MetalModel;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = gguf::Gguf::open(&a[1])?;
    let hp = Hparams::from_gguf(&g)?;
    let tok = inkling::tokenizer::Tokenizer::from_gguf(&g)?;
    let mut prompt = tok.encode(&std::fs::read_to_string(&a[2])?);
    prompt.truncate(512);
    let mut m = MetalModel::new(&g, hp.clone(), 600)?;
    m.prefill(&prompt)?;
    let ids = m.debug_last_ids(prompt.len());
    let mut count = vec![0usize; hp.n_expert];
    for &e in &ids {
        count[e as usize] += 1;
    }
    for bn in [16usize, 32, 64] {
        let tiles: usize = count.iter().map(|c| c.div_ceil(bn)).sum();
        println!(
            "tile {bn:2} pairs: {tiles} tiles for {} pairs, {:.0}% full",
            ids.len(),
            100.0 * ids.len() as f64 / (tiles * bn) as f64
        );
    }
    let mut c = count.clone();
    c.sort();
    println!(
        "pairs per expert: min {} median {} max {}, {} experts unused",
        c[0],
        c[c.len() / 2],
        c[c.len() - 1],
        c.iter().filter(|&&x| x == 0).count()
    );
    Ok(())
}
