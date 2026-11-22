# Reference material

There is no Inkling 2 technical report. The exact model definition is the reference code below. The prose sources only explain intent.

| file | source |
|---|---|
| `ARCHITECTURE.md` | our op-by-op spec with dimensions and the decode roofline |
| `NOTES.md` | where transformers and llama.cpp differ in form |
| `transformers/inkling2_loop/*.py`, `transformers/inkling2/*.py` | huggingface/transformers `main`, `src/transformers/models/inkling2{,_loop}` (fetched 2026-11-22) |
| `hf/*` | `thinkingmachines/Inkling-2-Mini` config, tokenizer config, chat template, quant config |
| `gguf-IQ4_NL-dump.txt` | `gguf_dump.py` output for the ggml-org IQ4_NL GGUF (metadata + all 704 tensors) |
| `web/inkling-2-mini-blog.{html,md}` | https://thinkingmachines.ai/news/inkling-2-mini/ |
| `web/inkling-2-mini-model-card.{html,md}` | https://thinkingmachines.ai/model-card/inkling-2-mini/ |
| `web/varloop-research-model-card.{html,md}` | https://thinkingmachines.ai/model-card/tml-d24-varloop-research/ |
| `web/end-to-end-4-bit-training.{html,md}` | https://thinkingmachines.ai/blog/end-to-end-4-bit-training/ |
| `web/goodfire-interpreting-inkling-2-mini.{html,md}` | https://www.goodfire.com/research/interpreting-inkling-2-mini/ (its appendix has the loop equations) |
| `attention-residuals-2603.15031.pdf` | Kimi Team, "Attention Residuals", arXiv 2603.15031 |
| `universal-transformers-1807.03819.pdf` | Dehghani et al., arXiv 1807.03819 |

llama.cpp's port lives in `~/sw/llama.cpp/src/models/inkling2.cpp` and `inkling2-loop.cpp` (not copied; pinned by commit in the bench results).
