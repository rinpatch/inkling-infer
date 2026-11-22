//! Byte-level BPE tokenizer built from the vocabulary embedded in the GGUF (o200k-style, as in the
//! model's tokenizer.json). Control tokens such as `<|message_user|>` are matched literally before
//! pre-tokenization, like llama.cpp's `parse_special = true`.

use std::collections::HashMap;

use fancy_regex::Regex;
use gguf::{Gguf, Value};

use crate::{Error, Result};

/// The model's pre-tokenizer split (tokenizer.json `pre_tokenizer`, behavior "Isolated").
const PRE_TOKENIZER: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

// GGUF token types (llama_token_type)
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

pub struct Tokenizer {
    tokens: Vec<String>,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    /// Control / user-defined tokens, longest first so longer matches win.
    specials: Vec<(String, u32)>,
    special_ids: Vec<bool>,
    re: Regex,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pub eos: u32,
}

/// GPT-2's reversible byte -> printable unicode mapping used by byte-level BPE.
fn byte_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (b'!' as u32..=b'~' as u32).contains(&b)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    table
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        if g.get_str("tokenizer.ggml.model")? != "gpt2" {
            return Err(Error::Unsupported(
                "tokenizer model is not byte-level BPE".into(),
            ));
        }
        let strs = |key: &str| -> Result<Vec<String>> {
            g.get_array(key)?
                .iter()
                .map(|v| match v {
                    Value::Str(s) => Ok(s.clone()),
                    _ => Err(Error::Unsupported(format!("{key}: non-string entry"))),
                })
                .collect()
        };
        let tokens = strs("tokenizer.ggml.tokens")?;
        let types: Vec<i32> = g
            .get_array("tokenizer.ggml.token_type")?
            .iter()
            .map(|v| match v {
                Value::I32(t) => *t,
                _ => 1,
            })
            .collect();
        let merges = strs("tokenizer.ggml.merges")?;

        let ids: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        let ranks = merges
            .iter()
            .enumerate()
            .filter_map(|(r, m)| {
                m.split_once(' ')
                    .map(|(a, b)| ((a.to_string(), b.to_string()), r))
            })
            .collect();
        let mut specials: Vec<(String, u32)> = types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == TOKEN_TYPE_CONTROL || **t == TOKEN_TYPE_USER_DEFINED)
            .map(|(i, _)| (tokens[i].clone(), i as u32))
            .filter(|(s, _)| !s.is_empty())
            .collect();
        specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        let special_ids = (0..tokens.len())
            .map(|i| types.get(i) == Some(&TOKEN_TYPE_CONTROL))
            .collect();

        let byte_to_char = byte_table();
        let char_to_byte = byte_to_char
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();
        let eos = g.get_u64("tokenizer.ggml.eos_token_id")? as u32;
        let re = Regex::new(PRE_TOKENIZER)
            .map_err(|e| Error::Unsupported(format!("pre-tokenizer regex: {e}")))?;
        Ok(Tokenizer {
            tokens,
            ids,
            ranks,
            specials,
            special_ids,
            re,
            byte_to_char,
            char_to_byte,
            eos,
        })
    }

    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // the earliest special token occurrence splits the text; plain text before it is BPE'd
            let next = self
                .specials
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|p| (p, s.len(), *id)))
                .min_by_key(|&(p, len, _)| (p, std::cmp::Reverse(len)));
            match next {
                Some((p, len, id)) => {
                    self.encode_plain(&rest[..p], &mut out);
                    out.push(id);
                    rest = &rest[p + len..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for m in self.re.find_iter(text) {
            let Ok(m) = m else { continue };
            let word: String = m
                .as_str()
                .bytes()
                .map(|b| self.byte_to_char[b as usize])
                .collect();
            // tokenizer.json sets ignore_merges: a pre-token that is already a vocab entry is emitted whole
            if let Some(&id) = self.ids.get(&word) {
                out.push(id);
                continue;
            }
            self.bpe(&word, out);
        }
    }

    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        let mut parts: Vec<String> = word.chars().map(String::from).collect();
        while parts.len() > 1 {
            let best = (0..parts.len() - 1)
                .filter_map(|i| {
                    self.ranks
                        .get(&(parts[i].clone(), parts[i + 1].clone()))
                        .map(|&r| (r, i))
                })
                .min();
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts.splice(i..i + 2, [merged]);
        }
        for p in parts {
            match self.ids.get(&p) {
                Some(&id) => out.push(id),
                // every single byte is in the vocab, so this only happens for a corrupt vocab
                None => out.extend(
                    p.chars()
                        .filter_map(|c| self.ids.get(&c.to_string()).copied()),
                ),
            }
        }
    }

    /// Raw bytes of one token (control tokens are rendered as their literal text).
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(t) = self.tokens.get(id as usize) else {
            return Vec::new();
        };
        if self.special_ids[id as usize] {
            return t.as_bytes().to_vec();
        }
        t.chars()
            .map(|c| self.char_to_byte.get(&c).copied().unwrap_or(b'?'))
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| self.token_bytes(i)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
