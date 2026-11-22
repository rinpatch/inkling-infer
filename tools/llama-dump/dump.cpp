// Fixture generator: runs a prompt through llama.cpp (CPU by default) and writes every named
// intermediate tensor matching a regex as raw little-endian f32, plus the final logits and a
// greedy continuation. Our engine's tests compare against these files.
//
// Output layout in <out_dir>:
//   tokens.txt                one prompt token id per line
//   manifest.tsv              name \t ubatch \t ne0 \t ne1 \t ne2 \t ne3 \t file
//   t/<name>.u<k>.f32         tensor data in logical (ne0-fastest) order
//   logits_last.f32           unpadded logits of the last prompt token
//   topk.bin                  (all_logits=1) per prompt position: f32 logsumexp over the full vocab,
//                             then 128 x (u32 id, f32 logit), highest first
//   greedy.tsv                step \t token \t piece \t top10 "id:logit" pairs (logits before choosing)

#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

struct dump_state {
    std::string out_dir;
    std::regex  filter;
    bool        active = true; // only dump during the prompt pass
    int         ubatch = 0;
    std::map<std::string, int> seen; // name -> times seen in the current ubatch
    FILE *      manifest = nullptr;
    std::vector<uint8_t> buf;
};

static float read_f32(const uint8_t * p, ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return *(const float *) p;
        case GGML_TYPE_F16: return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
        case GGML_TYPE_I32: return (float) *(const int32_t *) p;
        default: return 0.0f;
    }
}

static bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    const bool want = st->active && t->name[0] != '\0' &&
                      (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_I32) &&
                      std::regex_search(t->name, st->filter);
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }

    const size_t nbytes = ggml_nbytes(t);
    const uint8_t * data = (const uint8_t *) t->data;
    if (!ggml_backend_buffer_is_host(t->buffer)) {
        st->buf.resize(nbytes);
        ggml_backend_tensor_get(t, st->buf.data(), 0, nbytes);
        data = st->buf.data();
    }

    // the same name can appear several times in one graph (e.g. views); keep each occurrence
    int & occ = st->seen[t->name];
    std::string name = t->name;
    if (occ > 0) {
        name += "#" + std::to_string(occ);
    }
    occ++;

    std::string fname = name;
    std::replace(fname.begin(), fname.end(), '/', '_');
    fname = "t/" + fname + ".u" + std::to_string(st->ubatch) + ".f32";

    std::vector<float> out;
    out.reserve(ggml_nelements(t));
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        const size_t off = i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
        out.push_back(read_f32(data + off, t->type));
    }

    FILE * f = fopen((st->out_dir + "/" + fname).c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", fname.c_str());
        return false;
    }
    fwrite(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    fprintf(st->manifest, "%s\t%d\t%lld\t%lld\t%lld\t%lld\t%s\n", name.c_str(), st->ubatch,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], fname.c_str());
    return true;
}

static std::vector<std::pair<int, float>> top_k(const float * logits, int n, int k) {
    std::vector<std::pair<int, float>> v(n);
    for (int i = 0; i < n; ++i) v[i] = {i, logits[i]};
    std::partial_sort(v.begin(), v.begin() + k, v.end(), [](auto & a, auto & b) { return a.second > b.second; });
    v.resize(k);
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt.txt> <out_dir> [n_gen=64] [filter=.] [ubatch=4096] [ngl=0] [threads=10] [all_logits=0]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string prompt_path = argv[2];
    dump_state st;
    st.out_dir = argv[3];
    const int n_gen   = argc > 4 ? atoi(argv[4]) : 64;
    st.filter         = std::regex(argc > 5 ? argv[5] : ".");
    const int n_ub    = argc > 6 ? atoi(argv[6]) : 4096;
    const int ngl     = argc > 7 ? atoi(argv[7]) : 0;
    const int threads = argc > 8 ? atoi(argv[8]) : 10;
    const bool all_logits = argc > 9 && atoi(argv[9]) != 0;

    std::ifstream pf(prompt_path);
    std::stringstream ss;
    ss << pf.rdbuf();
    const std::string prompt = ss.str();

    if (system(("mkdir -p '" + st.out_dir + "/t'").c_str()) != 0) {
        return 1;
    }
    st.manifest = fopen((st.out_dir + "/manifest.tsv").c_str(), "w");

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // parse_special so chat-template control tokens become single ids; this model adds no BOS
    std::vector<llama_token> tokens(prompt.size() + 16);
    int n_tok = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true);
    if (n_tok < 0) {
        fprintf(stderr, "tokenize failed\n");
        return 1;
    }
    tokens.resize(n_tok);
    {
        FILE * f = fopen((st.out_dir + "/tokens.txt").c_str(), "w");
        for (auto t : tokens) fprintf(f, "%d\n", t);
        fclose(f);
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx           = n_tok + n_gen + 16;
    cparams.n_batch         = std::max(n_tok, 1);
    cparams.n_ubatch        = std::min(std::max(n_tok, 1), n_ub);
    cparams.n_threads       = threads;
    cparams.n_threads_batch = threads;
    cparams.cb_eval           = cb_eval;
    cparams.cb_eval_user_data = &st;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int n_vocab_all = llama_vocab_n_tokens(vocab);

    // feed the prompt one ubatch at a time so each dumped tensor is tagged with its ubatch index
    for (int start = 0; start < n_tok; start += (int) cparams.n_ubatch) {
        const int n = std::min((int) cparams.n_ubatch, n_tok - start);
        st.seen.clear();
        common_batch batch = all_logits ? common_batch(ctx) : common_batch_get_one(ctx, tokens.data() + start, n);
        if (all_logits) {
            for (int i = 0; i < n; ++i) batch.add(tokens[start + i], start + i, 0, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "prompt decode failed at %d\n", start);
            return 1;
        }
        if (all_logits) {
            FILE * tf = fopen((st.out_dir + "/topk.bin").c_str(), start == 0 ? "wb" : "ab");
            for (int i = 0; i < n; ++i) {
                const float * lg = llama_get_logits_ith(ctx, i);
                double m = -INFINITY;
                for (int v = 0; v < n_vocab_all; ++v) m = std::max(m, (double) lg[v]);
                double z = 0.0;
                for (int v = 0; v < n_vocab_all; ++v) z += std::exp((double) lg[v] - m);
                float lse = (float) (m + std::log(z));
                fwrite(&lse, sizeof(float), 1, tf);
                auto top = top_k(lg, n_vocab_all, 128);
                for (auto & [id, val] : top) {
                    uint32_t u = (uint32_t) id;
                    fwrite(&u, sizeof(uint32_t), 1, tf);
                    fwrite(&val, sizeof(float), 1, tf);
                }
            }
            fclose(tf);
        }
        st.ubatch++;
    }
    st.active = false;
    fclose(st.manifest);

    const float * logits = llama_get_logits_ith(ctx, -1);
    {
        FILE * f = fopen((st.out_dir + "/logits_last.f32").c_str(), "wb");
        fwrite(logits, sizeof(float), n_vocab, f);
        fclose(f);
    }

    FILE * g = fopen((st.out_dir + "/greedy.tsv").c_str(), "w");
    for (int step = 0; step < n_gen; ++step) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        auto top = top_k(lg, n_vocab, 10);
        llama_token next = top[0].first;
        char piece[256];
        int np = llama_token_to_piece(vocab, next, piece, sizeof(piece) - 1, 0, true);
        std::string pc(piece, np > 0 ? np : 0);
        for (auto & c : pc) if (c == '\n' || c == '\t') c = ' ';
        fprintf(g, "%d\t%d\t%s\t", step, next, pc.c_str());
        for (auto & [id, v] : top) fprintf(g, "%d:%.6f ", id, v);
        fprintf(g, "\n");
        common_batch batch = common_batch_get_one(ctx, &next, 1);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "decode failed at step %d\n", step);
            return 1;
        }
    }
    fclose(g);

    llama_free(ctx);
    llama_model_free(model);
    fprintf(stderr, "done: %d prompt tokens, %d generated, %d ubatches\n", n_tok, n_gen, st.ubatch);
    return 0;
}
