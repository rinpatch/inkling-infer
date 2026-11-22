# Writes fixtures/out/<name>/tokens_hf.txt with the official HF tokenizer (the tokenizer reference;
# llama.cpp's inkling2 pre-tokenizer ignores tokenizer.json's ignore_merges and splits some words).
# usage: uv run --with tokenizers python -I tools/hf_tokens.py <tokenizer.json> <repo>
import sys
from pathlib import Path
from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
repo = Path(sys.argv[2])
for name in ["short", "swa", "long"]:
    text = (repo / "fixtures/prompts" / f"{name}.txt").read_text()
    ids = tok.encode(text, add_special_tokens=False).ids
    out = repo / "fixtures/out" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokens_hf.txt").write_text("".join(f"{i}\n" for i in ids))
    print(name, len(ids))
