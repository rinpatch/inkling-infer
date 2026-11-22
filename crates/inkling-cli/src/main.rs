//! inkling run   -m model.gguf -p "prompt" [-n 64] [--chat] [--temp T --seed S] [--backend metal|cpu]
//! inkling bench -m model.gguf [--pp 512] [--tg 128] [-r 5] [--backend metal|cpu] [--json out.json]

use std::io::Write;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use inkling::cpu::{CpuModel, State};
use inkling::tokenizer::Tokenizer;
use inkling::weights::Weights;
use inkling::{Hparams, NoTrace};
use inkling_metal::model::MetalModel;

/// Decode roofline (tok/s) from ARCHITECTURE.md: bytes read per token over 273 GB/s.
const ROOFLINE_TOK_S: f64 = 68.0;

#[derive(Clone, Copy, PartialEq)]
enum BackendKind {
    Cpu,
    Metal,
}

struct Args {
    cmd: String,
    model: String,
    prompt: String,
    n_gen: usize,
    chat: bool,
    temp: f32,
    seed: u64,
    backend: BackendKind,
    pp: usize,
    tg: usize,
    reps: usize,
    depth: usize,
    json: Option<String>,
}

const USAGE: &str = "usage:
  inkling run   -m <model.gguf> -p <prompt> [-f file] [-n N] [--chat] [--temp T] [--seed S] [--backend metal|cpu]
  inkling bench -m <model.gguf> [--pp 512] [--tg 128] [-r 5] [-d depth] [--backend metal|cpu] [--json out.json]";

fn parse() -> Result<Args> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().unwrap_or_default();
    if cmd != "run" && cmd != "bench" {
        bail!("{USAGE}");
    }
    let mut a = Args {
        cmd,
        model: String::new(),
        prompt: String::new(),
        n_gen: 64,
        chat: false,
        temp: 0.0,
        seed: 0,
        backend: BackendKind::Metal,
        pp: 512,
        tg: 128,
        reps: 5,
        depth: 0,
        json: None,
    };
    while let Some(arg) = it.next() {
        let mut val = || it.next().with_context(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "-m" | "--model" => a.model = val()?,
            "-p" | "--prompt" => a.prompt = val()?,
            "-f" | "--file" => a.prompt = std::fs::read_to_string(val()?)?,
            "-n" => a.n_gen = val()?.parse()?,
            "--temp" => a.temp = val()?.parse()?,
            "--seed" => a.seed = val()?.parse()?,
            "--chat" => a.chat = true,
            "--pp" => a.pp = val()?.parse()?,
            "--tg" => a.tg = val()?.parse()?,
            "-r" | "--reps" => a.reps = val()?.parse()?,
            "--json" => a.json = Some(val()?),
            "-d" | "--depth" => a.depth = val()?.parse()?,
            "--backend" => {
                a.backend = match val()?.as_str() {
                    "cpu" => BackendKind::Cpu,
                    "metal" => BackendKind::Metal,
                    other => bail!("unknown backend {other}"),
                }
            }
            other => bail!("unknown argument {other}\n{USAGE}"),
        }
    }
    if a.model.is_empty() {
        bail!("-m <model.gguf> is required\n{USAGE}");
    }
    Ok(a)
}

/// One sequence on either backend.
enum Engine<'g> {
    Cpu(Box<(CpuModel<'g>, State)>),
    Metal(Box<MetalModel>),
}

impl<'g> Engine<'g> {
    fn new(kind: BackendKind, g: &'g gguf::Gguf, hp: &Hparams, max_ctx: usize) -> Result<Self> {
        Ok(match kind {
            BackendKind::Cpu => Engine::Cpu(Box::new((
                CpuModel::new(hp.clone(), Weights::load(g, hp)?),
                State::new(hp),
            ))),
            BackendKind::Metal => Engine::Metal(Box::new(MetalModel::new(g, hp.clone(), max_ctx)?)),
        })
    }

    fn reset(&mut self, hp: &Hparams) {
        match self {
            Engine::Cpu(b) => b.1 = State::new(hp),
            Engine::Metal(m) => m.reset(),
        }
    }

    /// Processes a prompt: batched on Metal, token by token on the CPU reference.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        match self {
            Engine::Metal(m) => Ok(m.prefill(tokens)?),
            Engine::Cpu(_) => {
                let mut logits = Vec::new();
                for &t in tokens {
                    logits = self.step(t)?;
                }
                Ok(logits)
            }
        }
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        Ok(match self {
            Engine::Cpu(b) => {
                let (m, s) = &mut **b;
                m.step(token, s, &mut NoTrace)?
            }
            Engine::Metal(m) => m.step(token, &mut NoTrace)?,
        })
    }
}

/// Single-turn chat prompt in the model's template (chat_template.jinja), thinking effort 0.
fn chat_prompt(user: &str) -> String {
    format!(
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>{user}<|end_message|><|message_model|>"
    )
}

/// xorshift64*; deterministic per seed so sampled runs are reproducible.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

fn sample(logits: &[f32], temp: f32, rng: &mut Rng) -> u32 {
    if temp <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|x| x.0 as u32)
            .unwrap_or(0);
    }
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f32> = logits.iter().map(|&l| ((l - m) / temp).exp()).collect();
    let mut r = rng.next_f32() * p.iter().sum::<f32>();
    for (i, &pi) in p.iter().enumerate() {
        r -= pi;
        if r <= 0.0 {
            return i as u32;
        }
    }
    (p.len() - 1) as u32
}

fn run(args: &Args, g: &gguf::Gguf, hp: &Hparams) -> Result<()> {
    let tok = Tokenizer::from_gguf(g)?;
    let text = if args.chat {
        chat_prompt(&args.prompt)
    } else {
        args.prompt.clone()
    };
    let ids = tok.encode(&text);
    if ids.is_empty() {
        bail!("empty prompt");
    }
    let mut eng = Engine::new(args.backend, g, hp, ids.len() + args.n_gen + 1)?;

    let t0 = Instant::now();
    let mut logits = eng.prefill(&ids)?;
    let t_prompt = t0.elapsed().as_secs_f64();

    let mut rng = Rng(args.seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
    let mut out = std::io::stdout().lock();
    let t1 = Instant::now();
    let mut n = 0;
    let mut pending: Vec<u8> = Vec::new();
    for _ in 0..args.n_gen {
        let next = sample(&logits, args.temp, &mut rng);
        n += 1;
        if next == tok.eos {
            break;
        }
        // tokens can end mid UTF-8 sequence; print only complete characters
        pending.extend(tok.token_bytes(next));
        let valid = match std::str::from_utf8(&pending) {
            Ok(_) => pending.len(),
            Err(e) => e.valid_up_to(),
        };
        out.write_all(&pending[..valid])?;
        out.flush()?;
        pending.drain(..valid);
        logits = eng.step(next)?;
    }
    out.write_all(&pending)?;
    writeln!(out)?;
    let t_gen = t1.elapsed().as_secs_f64();
    eprintln!(
        "prompt: {} tokens in {:.2}s ({:.2} tok/s) | gen: {} tokens in {:.2}s ({:.2} tok/s)",
        ids.len(),
        t_prompt,
        ids.len() as f64 / t_prompt,
        n,
        t_gen,
        n as f64 / t_gen
    );
    Ok(())
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len().max(2) - 1) as f64;
    (m, v.sqrt())
}

/// llama-bench style: pp = prompt tokens/s from an empty cache, tg = generated tokens/s after the
/// prompt-free start. Token ids are fixed (not random) so runs are comparable.
fn bench(args: &Args, g: &gguf::Gguf, hp: &Hparams) -> Result<()> {
    let backend = if args.backend == BackendKind::Metal {
        "metal"
    } else {
        "cpu"
    };
    let tokens: Vec<u32> = (0..(args.pp.max(args.tg) + args.depth + 300) as u32)
        .map(|i| 1000 + (i * 7919) % 150000)
        .collect();
    let mut pp = Vec::new();
    let mut tg = Vec::new();
    let mut eng = Engine::new(args.backend, g, hp, args.pp.max(args.tg) + args.depth + 1)?;
    for rep in 0..=args.reps {
        // rep 0 is warmup (pipeline compilation, page faults on the weights)
        if args.pp > 0 {
            eng.reset(hp);
            let t = Instant::now();
            eng.prefill(&tokens[..args.pp])?;
            if rep > 0 {
                pp.push(args.pp as f64 / t.elapsed().as_secs_f64());
            }
        }
        if args.tg > 0 {
            eng.reset(hp);
            // decode after `depth` tokens of context, like llama-bench -d
            if args.depth > 0 {
                eng.prefill(&tokens[..args.depth])?;
            }
            let mut next = tokens[0];
            let t = Instant::now();
            for _ in 0..args.tg {
                let logits = eng.step(next)?;
                next = sample(&logits, 0.0, &mut Rng(1));
            }
            if rep > 0 {
                tg.push(args.tg as f64 / t.elapsed().as_secs_f64());
                if let Engine::Metal(m) = &eng {
                    let l = m.last;
                    eprintln!(
                        "  last token: {} dispatches, encode {:.2} ms, GPU {:.2} ms",
                        l.n_dispatch,
                        l.encode_s * 1e3,
                        l.gpu_s * 1e3
                    );
                }
            }
        }
    }
    let mut json = Vec::new();
    if !pp.is_empty() {
        let (m, s) = mean_std(&pp);
        println!("{backend:6} pp{:<5} {m:8.2} ± {s:.2} tok/s", args.pp);
        json.push(format!(r#"{{"backend":"{backend}","test":"pp{}","avg_ts":{m:.4},"stddev_ts":{s:.4},"samples":{pp:?}}}"#, args.pp));
    }
    if !tg.is_empty() {
        let (m, s) = mean_std(&tg);
        println!(
            "{backend:6} tg{} @ depth {:<5} {m:8.2} ± {s:.2} tok/s  ({:.0}% of the ~{ROOFLINE_TOK_S:.0} tok/s roofline)",
            args.tg,
            args.depth,
            100.0 * m / ROOFLINE_TOK_S
        );
        json.push(format!(r#"{{"backend":"{backend}","test":"tg{}","depth":{},"avg_ts":{m:.4},"stddev_ts":{s:.4},"samples":{tg:?}}}"#, args.tg, args.depth));
    }
    if let (Engine::Metal(m), true) = (&mut eng, std::env::var_os("INKLING_PROFILE").is_some()) {
        m.reset();
        // profile after `depth` tokens of context (at least 64)
        let depth = args.depth.max(64);
        m.prefill(&tokens[..depth])?;
        // INKLING_PROFILE=N profiles a batch of N tokens (1 = decode)
        let batch = std::env::var("INKLING_PROFILE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 256);
        let prof = m.profile_batch(&tokens[depth..depth + batch])?;
        let total: f64 = prof.iter().map(|p| p.1).sum();
        // timestamps come from separate per-dispatch encoders, so only the shares are meaningful
        println!("per-kernel GPU share for {batch} token(s) at position {depth}:");
        for (k, t, n) in &prof {
            println!("  {k:28} {:5.1}%  x{n}", 100.0 * t / total);
        }
    }
    if let Some(path) = &args.json {
        std::fs::write(path, format!("[{}]\n", json.join(",")))?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = parse()?;
    let g = gguf::Gguf::open(&args.model)?;
    let hp = Hparams::from_gguf(&g)?;
    match args.cmd.as_str() {
        "run" => run(&args, &g, &hp),
        _ => bench(&args, &g, &hp),
    }
}
