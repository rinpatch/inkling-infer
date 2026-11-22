# llama.cpp: Inkling 2 tokenizer splits words that are single tokens

Found 2026-11-22 against llama.cpp `c66103334`. Still present on upstream master `e300919cc` (2026-11-22 08:08 UTC). No matching issue, and none of the open Inkling PRs (#32187, #32218, #32268, #32751, #32773) touch `ignore_merges`.

## Symptom
On code-heavy text, llama.cpp produces token sequences the model never saw in training. Over our 13.5k-token `fixtures/prompts/long.txt` it emits 13571 tokens where the official tokenizer emits 13516.

```
input: def f(*args, **kwargs):
HF tokenizer.json : [1719, 149983, 7, 9, 70747, 11, 197033, 87550, 8, 25]           ... 'Ġ**', 'kwargs', ')', ':'
llama.cpp         : [1719, 149983, 7, 9, 70747, 11, 197033, 86226, 70, 82, 8, 25]   ... 'Ġ**', 'kwar', 'g', 's', ')', ':'
```

## Cause
The model's `tokenizer.json` sets `"ignore_merges": true` (tiktoken semantics: a pre-token that is already a vocabulary entry is emitted whole, without running BPE merges). In `src/llama-vocab.cpp`, the `tokenizer_pre == "inkling2"` branch sets `pre_type = LLAMA_VOCAB_PRE_TYPE_GPT4O`, but unlike the llama3 branch and several others it doesn't set `ignore_merges = true`. Running BPE merges on `kwargs` ends in `kwar` + `g` + `s`, because no merge sequence reaches the `kwargs` entry.

## Proposed fix (one line)
```cpp
} else if (
        tokenizer_pre == "inkling2") {
    pre_type = LLAMA_VOCAB_PRE_TYPE_GPT4O;
    special_bos_id = LLAMA_TOKEN_NULL;
    clean_spaces = false;
    ignore_merges = true;   // tokenizer.json: "ignore_merges": true
}
```
Verify: `llama-tokenize -m <inkling2 gguf> -p 'def f(*args, **kwargs):' --ids` should print 87550 for `kwargs`, and tokenization of a large code sample should match `tokenizers` on the official `tokenizer.json` exactly. Our engine's tokenizer, which implements `ignore_merges`, matches HF on all 17,351 fixture tokens (`crates/inkling/tests/tokenizer.rs`).

## Impact
Every llama.cpp, Ollama and LM Studio user of Inkling 2 GGUFs: worse quality on code and identifiers, plus slightly longer prompts. Other GGUFs whose `tokenizer.json` sets `ignore_merges` but whose pre-type branch doesn't set it may have the same problem.
