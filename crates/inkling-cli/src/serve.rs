//! `inkling serve`: an OpenAI-compatible chat completions server (Metal, one sequence at a time).
//!
//! Requests queue for the single engine thread. Between requests the engine keeps its KV cache and
//! the tokens behind it; a new prompt reuses the longest common prefix. Full-attention KV rows stay
//! in place, so rewinding needs the short-conv histories and the sliding-window rows at the target
//! position: the server snapshots them (`SeqState`) at every prompt end and every CHECKPOINT_EVERY
//! prompt tokens, and resumes from the latest snapshot at or below the common prefix. Agent loops
//! either send the answer back verbatim or drop its thinking, so the reusable prefix usually ends
//! at the previous prompt's end or past it.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use inkling::tokenizer::Tokenizer;
use inkling::{Hparams, NoTrace};
use inkling_metal::model::{MetalModel, SeqState};
use serde_json::{Value, json};

const CHECKPOINT_EVERY: usize = 4096;
/// A snapshot holds ~160 MiB of sliding-window rows; the oldest is evicted first.
const MAX_CHECKPOINTS: usize = 4;
/// Pipelined decoding (greedy or plain temperature sampling, picked on the GPU) runs this many
/// steps per call; a stop is noticed at the end of a chunk, and the tokens after it are discarded.
const PIPELINE_CHUNK: usize = 16;

pub struct ServeArgs {
    model: String,
    host: String,
    port: u16,
    ctx: usize,
    effort: f64,
    name: String,
    log: Option<String>,
}

const SERVE_USAGE: &str =
    "usage: inkling serve -m <model.gguf> [--host 127.0.0.1] [--port 8080] [--ctx 65536]
                     [--effort 0.9] [--name inkling-2-mini] [--log requests.jsonl]

  --effort: default thinking effort (0..0.99) when a request has no reasoning_effort";

pub fn parse(mut it: impl Iterator<Item = String>) -> Result<ServeArgs> {
    let mut a = ServeArgs {
        model: String::new(),
        host: "127.0.0.1".into(),
        port: 8080,
        ctx: 65536,
        effort: 0.9,
        name: "inkling-2-mini".into(),
        log: None,
    };
    while let Some(arg) = it.next() {
        let mut val = || it.next().with_context(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "-m" | "--model" => a.model = val()?,
            "--host" => a.host = val()?,
            "--port" => a.port = val()?.parse()?,
            "--ctx" => a.ctx = val()?.parse()?,
            "--effort" => a.effort = val()?.parse()?,
            "--name" => a.name = val()?,
            "--log" => a.log = Some(val()?),
            _ => bail!("unknown argument {arg}\n{SERVE_USAGE}"),
        }
    }
    if a.model.is_empty() {
        bail!("{SERVE_USAGE}");
    }
    if !(0.0..=0.99).contains(&a.effort) {
        bail!("--effort must be in [0, 0.99]");
    }
    Ok(a)
}

/// Special token ids the server renders and parses.
#[derive(Clone, Copy)]
struct Special {
    model: u32,
    user: u32,
    system: u32,
    tool: u32,
    text: u32,
    thinking: u32,
    tool_json: u32,
    end: u32,
    eos: u32,
    endoftext: u32,
}

impl Special {
    fn new(tok: &Tokenizer) -> Result<Self> {
        let id = |s: &str| -> Result<u32> {
            match tok.encode(s).as_slice() {
                [t] => Ok(*t),
                other => bail!("{s} is not a single token ({other:?})"),
            }
        };
        Ok(Special {
            model: id("<|message_model|>")?,
            user: id("<|message_user|>")?,
            system: id("<|message_system|>")?,
            tool: id("<|message_tool|>")?,
            text: id("<|content_text|>")?,
            thinking: id("<|content_thinking|>")?,
            tool_json: id("<|content_invoke_tool_json|>")?,
            end: id("<|end_message|>")?,
            eos: tok.eos,
            endoftext: id("<|endoftext|>")?,
        })
    }
}

// ---- prompt rendering (docs/reference/hf/chat_template.jinja) ----------------------------------

fn effort_value(v: &Value) -> Result<Option<f64>> {
    Ok(match v {
        Value::Null => None,
        Value::String(s) => Some(match s.trim() {
            "none" => 0.0,
            "minimal" => 0.1,
            "low" => 0.2,
            "medium" => 0.7,
            "high" => 0.9,
            "xhigh" | "max" => 0.99,
            other => bail!("unknown reasoning_effort {other:?}"),
        }),
        Value::Number(n) => Some(n.as_f64().unwrap_or(0.9)),
        _ => bail!("reasoning_effort must be a string or a number"),
    })
}

fn render_effort(out: &mut String, effort: f64) {
    out.push_str("<|message_system|><|content_text|>Thinking effort level: ");
    if effort == 0.0 {
        out.push('0');
    } else {
        out.push_str(&format!("{effort}"));
    }
    out.push_str("<|end_message|>");
}

/// Tool call arguments as the template writes them: top-level keys sorted, compact values.
fn render_args(args: &Value) -> Result<String> {
    let obj = match args {
        Value::Object(m) => m.clone(),
        Value::String(s) if s.trim().is_empty() => Default::default(),
        Value::String(s) => match serde_json::from_str::<Value>(s)? {
            Value::Object(m) => m,
            _ => bail!("tool call arguments must be an object"),
        },
        Value::Null => Default::default(),
        _ => bail!("tool call arguments must be an object"),
    };
    let mut keys: Vec<&String> = obj.keys().collect();
    keys.sort();
    let parts: Vec<String> = keys
        .iter()
        .map(|k| format!("{}:{}", Value::String((*k).clone()), obj[*k]))
        .collect();
    Ok(format!("{{{}}}", parts.join(",")))
}

/// Keys sorted at every level, like tojson(sort_keys=true) (maps otherwise keep insertion order).
fn sorted(v: Value) -> Value {
    match v {
        Value::Object(m) => {
            let mut kv: Vec<(String, Value)> = m.into_iter().collect();
            kv.sort_by(|a, b| a.0.cmp(&b.0));
            Value::Object(kv.into_iter().map(|(k, v)| (k, sorted(v))).collect())
        }
        Value::Array(a) => Value::Array(a.into_iter().map(sorted).collect()),
        v => v,
    }
}

fn content_parts(content: &Value) -> Result<Vec<String>> {
    Ok(match content {
        Value::Null => vec![],
        Value::String(s) => vec![s.clone()],
        Value::Array(parts) => {
            let mut v = Vec::new();
            for p in parts {
                match p {
                    Value::String(s) => v.push(s.clone()),
                    Value::Object(o) => match o.get("type").and_then(Value::as_str) {
                        None | Some("text") | Some("input_text") => v.push(
                            o.get("text")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                        ),
                        Some(t) => bail!("unsupported content part type {t}"),
                    },
                    _ => bail!("bad content part"),
                }
            }
            v
        }
        _ => bail!("content must be a string or a list of parts"),
    })
}

fn render(messages: &[Value], tools: Option<&Vec<Value>>, effort: f64) -> Result<String> {
    let mut out = String::new();
    if let Some(tools) = tools.filter(|t| !t.is_empty()) {
        let specs: Vec<Value> = tools
            .iter()
            .map(|t| {
                let f = t.get("function").unwrap_or(t);
                json!({
                    "description": f.get("description").and_then(Value::as_str).unwrap_or(""),
                    "name": f.get("name").cloned().unwrap_or(Value::Null),
                    "parameters": f.get("parameters").filter(|p| !p.is_null()).cloned().unwrap_or(json!({})),
                    "type": t.get("type").and_then(Value::as_str).unwrap_or("function"),
                })
            })
            .collect();
        out.push_str("<|message_system|>tool_declare<|content_xml|>");
        out.push_str(&serde_json::to_string(&sorted(Value::Array(specs)))?);
        out.push_str("<|end_message|>");
    }
    let mut effort_done = false;
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("");
        let role = if role == "developer" { "system" } else { role };
        let mut header = match role {
            "user" => "<|message_user|>",
            "assistant" => "<|message_model|>",
            "system" => "<|message_system|>",
            "tool" => "<|message_tool|>",
            _ => bail!("unknown message role {role:?}"),
        }
        .to_string();
        if !effort_done && role != "system" {
            render_effort(&mut out, effort);
            effort_done = true;
        }
        let name = m
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        if role == "tool" {
            if let Some(n) = name {
                header.push_str(n);
            } else if let Some(id) = m.get("tool_call_id").and_then(Value::as_str) {
                for prev in messages {
                    for tc in prev
                        .get("tool_calls")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                    {
                        if tc.get("id").and_then(Value::as_str) == Some(id)
                            && let Some(n) = tc.pointer("/function/name").and_then(Value::as_str)
                        {
                            header.push_str(n);
                        }
                    }
                }
            }
        } else if role == "assistant"
            && let Some(n) = name
        {
            header.push_str(n);
        }
        let reasoning = m
            .get("reasoning_content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        let tool_calls = m
            .get("tool_calls")
            .and_then(Value::as_array)
            .filter(|t| !t.is_empty());
        if role == "assistant"
            && let Some(r) = reasoning
        {
            out.push_str(&format!("{header}<|content_thinking|>{r}<|end_message|>"));
        }
        let content = m.get("content").unwrap_or(&Value::Null);
        match content {
            Value::String(s) => {
                if !s.is_empty()
                    || !(role == "assistant" && (tool_calls.is_some() || reasoning.is_some()))
                {
                    out.push_str(&format!("{header}<|content_text|>{s}<|end_message|>"));
                }
            }
            _ => {
                for p in content_parts(content)? {
                    out.push_str(&format!("{header}<|content_text|>{p}<|end_message|>"));
                }
            }
        }
        if role == "assistant" {
            for tc in tool_calls.into_iter().flatten() {
                let f = tc.get("function").context("tool call without function")?;
                let fname = f
                    .get("name")
                    .and_then(Value::as_str)
                    .context("tool call name")?;
                let args = render_args(f.get("arguments").unwrap_or(&Value::Null))?;
                out.push_str(&format!(
                    "<|message_model|>{fname}<|content_invoke_tool_json|>{{\"name\":{},\"args\":{args}}}<|end_message|>",
                    Value::String(fname.to_string())
                ));
            }
            out.push_str("<|content_model_end_sampling|>");
        }
    }
    if !effort_done {
        render_effort(&mut out, effort);
    }
    Ok(out)
}

// ---- output parsing ----------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Header,
    Text,
    Thinking,
    Tool,
    Other,
}

/// Splits generated tokens into the model's segments: `<|message_model|>[name]<|content_*|>...
/// <|end_message|>`, ending with `<|content_model_end_sampling|>`.
struct Parser {
    sp: Special,
    kind: Kind,
    header: Vec<u8>,
    pending: Vec<u8>,
    content: String,
    reasoning: String,
    tool_buf: String,
    tool_calls: Vec<(String, String)>,
}

enum Piece {
    Content(String),
    Reasoning(String),
}

impl Parser {
    fn new(sp: Special) -> Self {
        Parser {
            sp,
            kind: Kind::Other,
            header: Vec::new(),
            pending: Vec::new(),
            content: String::new(),
            reasoning: String::new(),
            tool_buf: String::new(),
            tool_calls: Vec::new(),
        }
    }

    /// Feeds one token; returns the text it completes, or None at the end of sampling.
    fn push(&mut self, tok: &Tokenizer, t: u32) -> Option<Option<Piece>> {
        let sp = self.sp;
        if t == sp.eos || t == sp.endoftext {
            return None;
        }
        if t == sp.model || t == sp.user || t == sp.system || t == sp.tool {
            self.finish_segment();
            self.kind = Kind::Header;
            self.header.clear();
            return Some(None);
        }
        if t == sp.text || t == sp.thinking || t == sp.tool_json {
            self.kind = if t == sp.text {
                Kind::Text
            } else if t == sp.thinking {
                Kind::Thinking
            } else {
                Kind::Tool
            };
            return Some(None);
        }
        if t == sp.end {
            self.finish_segment();
            self.kind = Kind::Other;
            return Some(None);
        }
        let bytes = tok.token_bytes(t);
        match self.kind {
            Kind::Header => {
                self.header.extend(bytes);
                Some(None)
            }
            Kind::Other => Some(None),
            k => {
                self.pending.extend(bytes);
                let valid = match std::str::from_utf8(&self.pending) {
                    Ok(_) => self.pending.len(),
                    Err(e) => e.valid_up_to(),
                };
                let s = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                if s.is_empty() {
                    return Some(None);
                }
                Some(match k {
                    Kind::Text => {
                        self.content.push_str(&s);
                        Some(Piece::Content(s))
                    }
                    Kind::Thinking => {
                        self.reasoning.push_str(&s);
                        Some(Piece::Reasoning(s))
                    }
                    _ => {
                        self.tool_buf.push_str(&s);
                        None
                    }
                })
            }
        }
    }

    fn finish_segment(&mut self) {
        if self.kind == Kind::Tool {
            let name = String::from_utf8_lossy(&self.header).into_owned();
            // the model writes {"name": ..., "args": {...}}; OpenAI wants the args as a string
            let args = serde_json::from_str::<Value>(&self.tool_buf)
                .ok()
                .and_then(|v| v.get("args").cloned())
                .map(|a| a.to_string())
                .unwrap_or_else(|| self.tool_buf.clone());
            self.tool_calls.push((name, args));
            self.tool_buf.clear();
        }
        self.pending.clear();
    }
}

// ---- engine thread -----------------------------------------------------------------------------

struct Params {
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: u64,
    stop: Vec<String>,
}

enum Event {
    Piece(Piece),
    Done(Box<Done>),
    Error(String),
}

struct Done {
    finish: &'static str,
    content: String,
    reasoning: String,
    tool_calls: Vec<(String, String)>,
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
    prefill_s: f64,
    gen_s: f64,
}

struct Job {
    ids: Vec<u32>,
    params: Params,
    events: mpsc::Sender<Event>,
    cancel: Arc<AtomicBool>,
}

/// Tokens behind the engine's current state (len == pos) and conv snapshots along them.
struct Cache {
    tokens: Vec<u32>,
    snaps: Vec<SeqState>,
}

impl Cache {
    /// Moves the engine to the longest reusable prefix of `ids`; returns its length. At least one
    /// prompt token is always left to run, for the logits.
    fn prepare(&mut self, m: &mut MetalModel, ids: &[u32]) -> usize {
        let lcp = self
            .tokens
            .iter()
            .zip(ids)
            .take_while(|(a, b)| a == b)
            .count();
        let target = lcp.min(ids.len() - 1);
        if target < self.tokens.len() {
            match self
                .snaps
                .iter()
                .filter(|s| s.pos <= target)
                .max_by_key(|s| s.pos)
            {
                Some(s) => m.load_state(s),
                None => m.reset(),
            }
            self.tokens.truncate(m.pos());
            // snapshots past the new position see KV rows that are about to be rewritten
            let p = m.pos();
            self.snaps.retain(|s| s.pos <= p);
        }
        self.tokens.len()
    }

    fn snapshot(&mut self, m: &MetalModel) {
        if self.snaps.iter().any(|s| s.pos == m.pos()) {
            return;
        }
        if self.snaps.len() >= MAX_CHECKPOINTS {
            self.snaps.remove(0);
        }
        self.snaps.push(m.save_state());
    }
}

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|x| x.0 as u32)
        .unwrap_or(0)
}

fn sample(logits: &[f32], temp: f32, top_p: f32, rng: &mut Rng) -> u32 {
    if temp <= 0.0 {
        return argmax(logits);
    }
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, ((l - m) / temp).exp()))
        .filter(|x| x.1 > 0.0)
        .collect();
    if top_p < 1.0 {
        p.sort_by(|a, b| b.1.total_cmp(&a.1));
        let total: f32 = p.iter().map(|x| x.1).sum();
        let mut acc = 0.0;
        let mut keep = p.len();
        for (i, x) in p.iter().enumerate() {
            acc += x.1;
            if acc >= top_p * total {
                keep = i + 1;
                break;
            }
        }
        p.truncate(keep);
    }
    let mut r = rng.next_f32() * p.iter().map(|x| x.1).sum::<f32>();
    for &(i, pi) in &p {
        r -= pi;
        if r <= 0.0 {
            return i;
        }
    }
    p.last().map(|x| x.0).unwrap_or(0)
}

fn run_job(
    m: &mut MetalModel,
    cache: &mut Cache,
    tok: &Tokenizer,
    sp: Special,
    job: &Job,
) -> Result<Done> {
    let ids = &job.ids;
    let p = &job.params;
    let t0 = Instant::now();
    let reused = cache.prepare(m, ids);
    let mut logits = Vec::new();
    let mut at = reused;
    while at < ids.len() {
        let end = ((at / CHECKPOINT_EVERY + 1) * CHECKPOINT_EVERY).min(ids.len());
        logits = m.prefill(&ids[at..end])?;
        cache.tokens.extend_from_slice(&ids[at..end]);
        cache.snapshot(m);
        at = end;
    }
    let prefill_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut parser = Parser::new(sp);
    let mut n = 0usize;
    let mut finish = "length";
    let max_ctx = m.max_ctx();
    let max_tokens = p.max_tokens.min(max_ctx.saturating_sub(ids.len()));
    // content already sent; with stop strings the tail that could start one is held back
    let hold = p
        .stop
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(1)
        .saturating_sub(1);
    let mut sent = 0usize;
    let mut rng = Rng(p.seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);

    // handles one generated token; Some(reason) ends generation
    let handle = |t: u32, parser: &mut Parser, sent: &mut usize| -> Option<&'static str> {
        match parser.push(tok, t) {
            None => {
                return Some(if parser.tool_calls.is_empty() {
                    "stop"
                } else {
                    "tool_calls"
                });
            }
            Some(Some(Piece::Reasoning(s))) => {
                let _ = job.events.send(Event::Piece(Piece::Reasoning(s)));
            }
            Some(Some(Piece::Content(_))) => {
                if let Some(cut) = p
                    .stop
                    .iter()
                    .filter_map(|s| parser.content.find(s.as_str()))
                    .min()
                {
                    parser.content.truncate(cut);
                    if cut > *sent {
                        let _ = job.events.send(Event::Piece(Piece::Content(
                            parser.content[*sent..].to_string(),
                        )));
                        *sent = cut;
                    }
                    return Some("stop");
                }
                let mut upto = parser.content.len().saturating_sub(hold);
                while !parser.content.is_char_boundary(upto) {
                    upto -= 1;
                }
                if upto > *sent {
                    let _ = job.events.send(Event::Piece(Piece::Content(
                        parser.content[*sent..upto].to_string(),
                    )));
                    *sent = upto;
                }
            }
            Some(None) => {}
        }
        if job.cancel.load(Ordering::Relaxed) {
            return Some("cancelled");
        }
        None
    };

    if max_tokens > 0 {
        let mut next = sample(&logits, p.temperature, p.top_p, &mut rng);
        'generate: loop {
            if let Some(f) = handle(next, &mut parser, &mut sent) {
                finish = f;
                break;
            }
            n += 1;
            let room = (max_tokens - n).min(max_ctx - m.pos());
            if room == 0 {
                break;
            }
            if p.top_p >= 1.0 {
                // the pick stays on the GPU for a chunk of steps (top-p needs the CPU sampler)
                let chunk = PIPELINE_CHUNK.min(room);
                let seed = p.seed ^ (n as u64).wrapping_mul(0xD1B54A32D192ED03);
                let got = m.decode_pipelined(next, chunk, p.temperature, seed)?;
                cache.tokens.push(next);
                cache.tokens.extend_from_slice(&got[..chunk - 1]);
                for &t in &got[..chunk - 1] {
                    if let Some(f) = handle(t, &mut parser, &mut sent) {
                        finish = f;
                        break 'generate;
                    }
                    n += 1;
                }
                next = got[chunk - 1];
            } else {
                logits = m.step(next, &mut NoTrace)?;
                cache.tokens.push(next);
                next = sample(&logits, p.temperature, p.top_p, &mut rng);
            }
        }
    }
    if parser.content.len() > sent {
        let _ = job.events.send(Event::Piece(Piece::Content(
            parser.content[sent..].to_string(),
        )));
    }
    debug_assert_eq!(cache.tokens.len(), m.pos());
    Ok(Done {
        finish,
        content: parser.content,
        reasoning: parser.reasoning,
        tool_calls: parser.tool_calls,
        prompt_tokens: ids.len(),
        cached_tokens: reused,
        completion_tokens: n,
        prefill_s,
        gen_s: t1.elapsed().as_secs_f64(),
    })
}

fn engine_thread(
    args: Arc<ServeArgs>,
    tok: Arc<Tokenizer>,
    sp: Special,
    jobs: mpsc::Receiver<Job>,
    ready: mpsc::Sender<Result<()>>,
) {
    let setup = (|| -> Result<MetalModel> {
        let g = gguf::Gguf::open(&args.model)?;
        let hp = Hparams::from_gguf(&g)?;
        Ok(MetalModel::new(&g, hp, args.ctx)?)
    })();
    let mut m = match setup {
        Ok(m) => {
            let _ = ready.send(Ok(()));
            m
        }
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let mut cache = Cache {
        tokens: Vec::new(),
        snaps: Vec::new(),
    };
    let mut log = args.log.as_ref().and_then(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .ok()
    });
    for job in jobs {
        match run_job(&mut m, &mut cache, &tok, sp, &job) {
            Ok(d) => {
                eprintln!(
                    "request: prompt {} (cached {}) in {:.2}s ({:.0} tok/s) | gen {} in {:.2}s ({:.1} tok/s) | {}",
                    d.prompt_tokens,
                    d.cached_tokens,
                    d.prefill_s,
                    (d.prompt_tokens - d.cached_tokens) as f64 / d.prefill_s.max(1e-9),
                    d.completion_tokens,
                    d.gen_s,
                    d.completion_tokens as f64 / d.gen_s.max(1e-9),
                    d.finish
                );
                if let Some(f) = &mut log {
                    let _ = writeln!(
                        f,
                        "{}",
                        json!({
                            "time": now(),
                            "prompt_tokens": d.prompt_tokens,
                            "cached_tokens": d.cached_tokens,
                            "prefill_s": d.prefill_s,
                            "completion_tokens": d.completion_tokens,
                            "gen_s": d.gen_s,
                            "finish": d.finish,
                        })
                    );
                }
                let _ = job.events.send(Event::Done(Box::new(d)));
            }
            Err(e) => {
                // the engine state may be partial; start the next request from scratch
                m.reset();
                cache.tokens.clear();
                cache.snaps.clear();
                let _ = job.events.send(Event::Error(format!("{e:#}")));
            }
        }
    }
}

// ---- HTTP --------------------------------------------------------------------------------------

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(s: &mut BufReader<TcpStream>) -> Result<Request> {
    let mut line = String::new();
    s.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().context("empty request")?.to_string();
    let path = parts.next().context("no path")?.to_string();
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if s.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':')
            && k.trim().eq_ignore_ascii_case("content-length")
        {
            len = v.trim().parse()?;
        }
    }
    let mut body = vec![0; len];
    s.read_exact(&mut body)?;
    Ok(Request { method, path, body })
}

fn respond(s: &mut TcpStream, status: &str, body: &Value) -> Result<()> {
    let b = body.to_string();
    write!(
        s,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}",
        b.len()
    )?;
    Ok(())
}

fn error(s: &mut TcpStream, status: &str, kind: &str, msg: &str) -> Result<()> {
    respond(
        s,
        status,
        &json!({"error": {"message": msg, "type": kind, "code": kind}}),
    )
}

struct Server {
    args: Arc<ServeArgs>,
    tok: Arc<Tokenizer>,
    jobs: mpsc::Sender<Job>,
}

impl Server {
    fn handle(&self, stream: TcpStream) -> Result<()> {
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut s = stream;
        let req = read_request(&mut reader)?;
        let path = req.path.split('?').next().unwrap_or("");
        match (req.method.as_str(), path) {
            ("GET", "/v1/models") | ("GET", "/models") => respond(
                &mut s,
                "200 OK",
                &json!({"object": "list", "data": [{"id": self.args.name, "object": "model", "created": 0, "owned_by": "inkling"}]}),
            ),
            ("GET", "/health") => respond(&mut s, "200 OK", &json!({"status": "ok"})),
            ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
                self.chat(&mut s, &req.body)
            }
            _ => error(
                &mut s,
                "404 Not Found",
                "not_found",
                &format!("no route {} {}", req.method, path),
            ),
        }
    }

    fn chat(&self, s: &mut TcpStream, body: &[u8]) -> Result<()> {
        let req: Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(e) => {
                return error(
                    s,
                    "400 Bad Request",
                    "invalid_request_error",
                    &format!("bad JSON: {e}"),
                );
            }
        };
        let prepared = (|| -> Result<(Vec<u32>, Params, bool, bool)> {
            let messages = req
                .get("messages")
                .and_then(Value::as_array)
                .context("messages must be a list")?;
            if req.get("n").and_then(Value::as_u64).unwrap_or(1) != 1 {
                bail!("only n = 1 is supported");
            }
            let effort = match effort_value(req.get("reasoning_effort").unwrap_or(&Value::Null))? {
                Some(e) => e,
                None => effort_value(
                    req.pointer("/chat_template_kwargs/reasoning_effort")
                        .unwrap_or(&Value::Null),
                )?
                .unwrap_or(self.args.effort),
            };
            if !(0.0..=0.99).contains(&effort) {
                bail!("reasoning_effort must be in [0.0, 0.99]");
            }
            let text = render(messages, req.get("tools").and_then(Value::as_array), effort)?;
            let ids = self.tok.encode(&text);
            if ids.len() >= self.args.ctx {
                bail!(
                    "context_length_exceeded: this model's maximum context length is {} tokens, the prompt has {}",
                    self.args.ctx,
                    ids.len()
                );
            }
            let max_tokens = req
                .get("max_completion_tokens")
                .or_else(|| req.get("max_tokens"))
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(usize::MAX);
            let stop = match req.get("stop") {
                Some(Value::String(s)) => vec![s.clone()],
                Some(Value::Array(a)) => a
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect(),
                _ => vec![],
            };
            let params = Params {
                max_tokens,
                temperature: req
                    .get("temperature")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0) as f32,
                top_p: req.get("top_p").and_then(Value::as_f64).unwrap_or(1.0) as f32,
                seed: req.get("seed").and_then(Value::as_u64).unwrap_or_else(now),
                stop,
            };
            let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
            let usage = req
                .pointer("/stream_options/include_usage")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            Ok((ids, params, stream, usage))
        })();
        let (ids, params, stream, include_usage) = match prepared {
            Ok(x) => x,
            Err(e) => {
                let msg = format!("{e:#}");
                let kind = if msg.starts_with("context_length_exceeded") {
                    "context_length_exceeded"
                } else {
                    "invalid_request_error"
                };
                return error(s, "400 Bad Request", kind, &msg);
            }
        };
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        self.jobs
            .send(Job {
                ids,
                params,
                events: tx,
                cancel: cancel.clone(),
            })
            .context("engine thread gone")?;
        let id = format!(
            "chatcmpl-{:x}",
            now() ^ (s.peer_addr().map(|a| a.port() as u64).unwrap_or(0) << 32)
        );
        let model = self.args.name.clone();
        let created = now();
        if stream {
            write!(
                s,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
            )?;
            let chunk = |delta: Value, finish: Value| {
                json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
            };
            let send = |s: &mut TcpStream, v: &Value| -> bool {
                if write!(s, "data: {v}\n\n").and_then(|_| s.flush()).is_err() {
                    // client went away: stop generating for it
                    cancel.store(true, Ordering::Relaxed);
                    false
                } else {
                    true
                }
            };
            send(
                s,
                &chunk(json!({"role": "assistant", "content": ""}), Value::Null),
            );
            for ev in rx {
                match ev {
                    Event::Piece(Piece::Content(c)) => {
                        send(s, &chunk(json!({"content": c}), Value::Null));
                    }
                    Event::Piece(Piece::Reasoning(r)) => {
                        send(s, &chunk(json!({"reasoning_content": r}), Value::Null));
                    }
                    Event::Done(d) => {
                        if !d.tool_calls.is_empty() {
                            send(
                                s,
                                &chunk(
                                    json!({"tool_calls": tool_calls_json(&d.tool_calls, true)}),
                                    Value::Null,
                                ),
                            );
                        }
                        send(s, &chunk(json!({}), json!(d.finish)));
                        if include_usage {
                            let mut u = json!({"id": id, "object": "chat.completion.chunk", "created": created,
                                               "model": model, "choices": []});
                            u["usage"] = usage_json(&d);
                            send(s, &u);
                        }
                        break;
                    }
                    Event::Error(e) => {
                        send(s, &json!({"error": {"message": e, "type": "server_error"}}));
                        break;
                    }
                }
            }
            let _ = write!(s, "data: [DONE]\n\n");
            return Ok(());
        }
        for ev in rx {
            match ev {
                Event::Piece(_) => {}
                Event::Done(d) => {
                    let mut msg = json!({"role": "assistant", "content": d.content});
                    if !d.reasoning.is_empty() {
                        msg["reasoning_content"] = json!(d.reasoning);
                    }
                    if !d.tool_calls.is_empty() {
                        msg["tool_calls"] = tool_calls_json(&d.tool_calls, false);
                    }
                    return respond(
                        s,
                        "200 OK",
                        &json!({"id": id, "object": "chat.completion", "created": created, "model": model,
                                "choices": [{"index": 0, "message": msg, "finish_reason": d.finish}],
                                "usage": usage_json(&d),
                                "timings": {"prompt_n": d.prompt_tokens - d.cached_tokens, "cache_n": d.cached_tokens,
                                            "prompt_ms": d.prefill_s * 1e3, "predicted_n": d.completion_tokens,
                                            "predicted_ms": d.gen_s * 1e3}}),
                    );
                }
                Event::Error(e) => {
                    return error(s, "500 Internal Server Error", "server_error", &e);
                }
            }
        }
        error(
            s,
            "500 Internal Server Error",
            "server_error",
            "engine stopped",
        )
    }
}

fn tool_calls_json(calls: &[(String, String)], indexed: bool) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, (name, args))| {
                let mut v = json!({"id": format!("call_{i}_{:x}", now()), "type": "function",
                                   "function": {"name": name, "arguments": args}});
                if indexed {
                    v["index"] = json!(i);
                }
                v
            })
            .collect(),
    )
}

fn usage_json(d: &Done) -> Value {
    json!({"prompt_tokens": d.prompt_tokens, "completion_tokens": d.completion_tokens,
           "total_tokens": d.prompt_tokens + d.completion_tokens,
           "prompt_tokens_details": {"cached_tokens": d.cached_tokens}})
}

pub fn serve(args: ServeArgs) -> Result<()> {
    let args = Arc::new(args);
    let g = gguf::Gguf::open(&args.model)?;
    let tok = Arc::new(Tokenizer::from_gguf(&g)?);
    drop(g);
    let sp = Special::new(&tok)?;
    let (jobs_tx, jobs_rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel();
    {
        let (args, tok) = (args.clone(), tok.clone());
        std::thread::spawn(move || engine_thread(args, tok, sp, jobs_rx, ready_tx));
    }
    ready_rx.recv().context("engine thread died")??;
    let listener = TcpListener::bind((args.host.as_str(), args.port))?;
    eprintln!(
        "serving {} on http://{}:{}/v1 (context {} tokens, default effort {})",
        args.name, args.host, args.port, args.ctx, args.effort
    );
    let server = Arc::new(Server {
        args: args.clone(),
        tok,
        jobs: jobs_tx,
    });
    for conn in listener.incoming() {
        let Ok(conn) = conn else { continue };
        let server = server.clone();
        std::thread::spawn(move || {
            if let Err(e) = server.handle(conn) {
                eprintln!("connection: {e:#}");
            }
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// fixtures/chat/template_cases.json: chat_template.jinja rendered by jinja2 (with HF's tojson).
    #[test]
    fn matches_jinja_template() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/chat/template_cases.json"
        );
        let cases: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for (i, c) in cases.as_array().unwrap().iter().enumerate() {
            let out = render(
                c["messages"].as_array().unwrap(),
                c.get("tools").and_then(Value::as_array),
                c["effort"].as_f64().unwrap(),
            )
            .unwrap();
            assert_eq!(out, c["expected"].as_str().unwrap(), "case {i}");
        }
    }

    #[test]
    fn renders_like_the_template() {
        let msgs = json!([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning_content": "think"},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        ]);
        let out = render(msgs.as_array().unwrap(), None, 0.2).unwrap();
        assert_eq!(
            out,
            "<|message_system|><|content_text|>sys<|end_message|>\
             <|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>\
             <|message_user|><|content_text|>hi<|end_message|>\
             <|message_model|><|content_thinking|>think<|end_message|>\
             <|message_model|><|content_text|>hello<|end_message|><|content_model_end_sampling|>\
             <|message_user|><|content_text|>a<|end_message|><|message_user|><|content_text|>b<|end_message|>"
        );
        let out = render(
            json!([{"role": "user", "content": "x"}])
                .as_array()
                .unwrap(),
            None,
            0.0,
        )
        .unwrap();
        assert!(out.starts_with(
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
        ));
    }

    #[test]
    fn renders_tool_calls_with_sorted_args() {
        let msgs = json!([
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"b\":1,\"a\":\"s\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]);
        let out = render(msgs.as_array().unwrap(), None, 0.9).unwrap();
        assert!(out.contains(
            "<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":\"s\",\"b\":1}}<|end_message|><|content_model_end_sampling|><|message_tool|>f<|content_text|>ok<|end_message|>"
        ));
    }
}
