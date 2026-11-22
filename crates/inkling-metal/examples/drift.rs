//! Per-position error of Metal layer outputs against the llama.cpp swa fixture, bucketed by
//! position, to tell accumulated drift from a position-dependent bug.
//! `cargo run --release -p inkling-metal --example drift -- <model.gguf>`

use std::collections::BTreeMap;
use std::fs;

use inkling::{Hparams, Trace};
use inkling_metal::model::MetalModel;

struct Rec(BTreeMap<usize, Vec<f32>>);
const SLOTS: [usize; 9] = [0, 4, 5, 6, 11, 17, 23, 30, 47];
impl Trace for Rec {
    fn wants(&self, name: &str, slot: Option<usize>) -> bool {
        name == "l_out" && slot.is_some_and(|s| SLOTS.contains(&s))
    }
    fn record(&mut self, _: &str, slot: Option<usize>, data: &[f32]) {
        self.0.insert(slot.unwrap(), data.to_vec());
    }
}

fn main() -> anyhow::Result<()> {
    let g = gguf::Gguf::open(std::env::args().nth(1).expect("model"))?;
    let hp = Hparams::from_gguf(&g)?;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/out/swa");
    let tokens: Vec<u32> = fs::read_to_string(dir.join("tokens.txt"))?
        .lines()
        .map(|l| l.trim().parse().unwrap())
        .collect();
    let d = hp.n_embd;
    let fx: BTreeMap<usize, Vec<f32>> = SLOTS
        .iter()
        .map(|&s| {
            let b = fs::read(dir.join(format!("t/l_out-{s}.u0.f32"))).unwrap();
            (
                s,
                b.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect(),
            )
        })
        .collect();
    let mut m = MetalModel::new(&g, hp, tokens.len() + 1)?;
    let buckets = [
        0usize,
        64,
        512,
        1000,
        1024,
        1100,
        2048,
        3000,
        3700,
        usize::MAX,
    ];
    let mut err: BTreeMap<(usize, usize), (f32, usize)> = BTreeMap::new();
    for (p, &t) in tokens.iter().enumerate() {
        let mut r = Rec(BTreeMap::new());
        m.step(t, &mut r)?;
        let bi = buckets.iter().rposition(|&b| p >= b).unwrap();
        for (&s, got) in &r.0 {
            let want = &fx[&s][p * d..(p + 1) * d];
            let num: f32 = got.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f32 = want.iter().map(|b| b * b).sum();
            let e = err.entry((s, bi)).or_insert((0.0, 0));
            e.0 += (num / den).sqrt();
            e.1 += 1;
        }
    }
    print!("{:>12}", "pos >=");
    for b in &buckets[..buckets.len() - 1] {
        print!("{b:>9}");
    }
    println!();
    for s in SLOTS {
        print!("l_out-{s:<6}");
        for bi in 0..buckets.len() - 1 {
            let (e, n) = err.get(&(s, bi)).copied().unwrap_or((0.0, 1));
            print!("{:>9.2e}", e / n as f32);
        }
        println!();
    }
    Ok(())
}
