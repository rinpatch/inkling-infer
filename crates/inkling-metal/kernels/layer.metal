// Per-token layer building blocks: norms, depth attention, clipping, short convolutions,
// attention decode and small elementwise ops. Semantics: docs/reference/ARCHITECTURE.md.
#include "common.h"

// ---- norms -------------------------------------------------------------------------------------

struct NormArgs {
    uint n;
    float eps;
    uint has_weight;
};

// y = x / rms(x) (* w), one row of n per threadgroup (grid = rows).
kernel void rms_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *y [[buffer(2)]],
                     constant NormArgs &a [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    x += (ulong)row * a.n;
    y += (ulong)row * a.n;
    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) ss += x[i] * x[i];
    float inv = rsqrt(tg_sum(ss, scratch, tid, ntg) / a.n + a.eps);
    for (uint i = tid; i < a.n; i += ntg) y[i] = x[i] * inv * (a.has_weight ? w[i] : 1.0f);
}

// Per-head unweighted RMS norm in place (values). Grid: n_heads threadgroups of 32.
kernel void rms_heads(device float *x [[buffer(0)]], constant NormArgs &a [[buffer(1)]], uint h [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    device float *v = x + h * a.n;
    float ss = 0.0f;
    for (uint i = lane; i < a.n; i += 32) ss += v[i] * v[i];
    float inv = rsqrt(simd_sum(ss) / a.n + a.eps);
    for (uint i = lane; i < a.n; i += 32) v[i] *= inv;
}

struct ClipArgs {
    uint dim;             // per-head vector length
    float threshold;
    uint has_scale;
    uint log_scale;       // multiply by tau(pos) after clipping (full-attention layers)
    uint heads_per_token;
    uint pos0;
    float alpha;
    float floor_;
};

// x_h = min(1, c / (|s*x_h| + 1e-6)) * s*x_h [* tau(pos0 + token)], in place.
// Grid: n_tokens * heads_per_token threadgroups of 32 (rows [token][head][dim]).
kernel void clip_heads(device float *x [[buffer(0)]], device const float *scale [[buffer(1)]], constant ClipArgs &a [[buffer(2)]],
                       uint h [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    device float *v = x + (ulong)h * a.dim;
    device const float *s = scale + (h % a.heads_per_token) * a.dim;
    float ss = 0.0f;
    for (uint i = lane; i < a.dim; i += 32) {
        float t = v[i] * (a.has_scale ? s[i] : 1.0f);
        v[i] = t;
        ss += t * t;
    }
    float f = min(1.0f, a.threshold / (sqrt(simd_sum(ss)) + 1e-6f));
    if (a.log_scale) {
        float pos = float(a.pos0 + h / a.heads_per_token);
        f *= 1.0f + a.alpha * log(max((pos + 1.0f) / a.floor_, 1.0f));
    }
    for (uint i = lane; i < a.dim; i += 32) v[i] *= f;
}

// ---- depth attention ("attention residuals") ---------------------------------------------------

struct MixArgs {
    uint d;
    uint n_blocks;  // states = blocks..., cur
    uint n_q;
    uint plane;     // floats between block planes / output planes ([plane][token][d])
};

#define MIX_MAX_STATES 8
#define MIX_MAX_Q 4

// out[q] = sum_s softmax_s(q . rms(S_s) / sqrt(d)) * rms(S_s), S = [blocks..., cur]. One threadgroup.
// Grid: one threadgroup per token.
kernel void depth_mix(device const float *blocks [[buffer(0)]], device const float *cur [[buffer(1)]],
                      device const float *queries [[buffer(2)]], device float *out [[buffer(3)]], constant MixArgs &a [[buffer(4)]],
                      uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                      uint ntg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    blocks += (ulong)t * a.d;
    cur += (ulong)t * a.d;
    out += (ulong)t * a.d;
    threadgroup float inv_rms[MIX_MAX_STATES];
    threadgroup float p[MIX_MAX_Q][MIX_MAX_STATES];
    uint ns = a.n_blocks + 1;
#define state(s) ((s) < a.n_blocks ? blocks + (ulong)(s) * a.plane : cur)

    for (uint s = 0; s < ns; ++s) {
        device const float *x = state(s);
        float ss = 0.0f;
        for (uint i = tid; i < a.d; i += ntg) ss += x[i] * x[i];
        float t = tg_sum(ss, scratch, tid, ntg);
        if (tid == 0) inv_rms[s] = rsqrt(t / a.d + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = 0; q < a.n_q; ++q) {
        for (uint s = 0; s < ns; ++s) {
            device const float *x = state(s);
            device const float *w = queries + q * a.d;
            float acc = 0.0f;
            for (uint i = tid; i < a.d; i += ntg) acc += x[i] * w[i];
            float t = tg_sum(acc, scratch, tid, ntg);
            if (tid == 0) p[q][s] = t * inv_rms[s] * rsqrt(float(a.d));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < a.n_q) {
        float m = -INFINITY;
        for (uint s = 0; s < ns; ++s) m = max(m, p[tid][s]);
        float sum = 0.0f;
        for (uint s = 0; s < ns; ++s) { p[tid][s] = exp(p[tid][s] - m); sum += p[tid][s]; }
        for (uint s = 0; s < ns; ++s) p[tid][s] /= sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < a.d; i += ntg) {
        for (uint q = 0; q < a.n_q; ++q) {
            float acc = 0.0f;
            for (uint s = 0; s < ns; ++s) acc += p[q][s] * inv_rms[s] * state(s)[i];
            out[(ulong)q * a.plane + i] = acc;
        }
    }
#undef state
}

// ---- short convolution -------------------------------------------------------------------------

struct ConvArgs {
    uint ch;
    uint nc;          // taps
    uint dyn_heads;   // 0 = static only
    float scale;
    uint n_tokens;
};

// For t = 0..n_tokens in order: y[t] = (x[t] + sum_l taps[c][nc-1-l]*x_{t-l}
// [+ sum_l dyn[t][h][l]*x_{t-l}]) * scale, shifting the history after each token.
// hist is [nc-1][ch], oldest first. One thread per channel; tokens are sequential per channel.
kernel void conv_step(device const float *x [[buffer(0)]], device float *hist [[buffer(1)]], device const float *taps [[buffer(2)]],
                      device const float *dyn [[buffer(3)]], device float *y [[buffer(4)]], constant ConvArgs &a [[buffer(5)]],
                      uint c [[thread_position_in_grid]]) {
    if (c >= a.ch) return;
    uint nh = a.nc - 1;
    float xs[8];  // xs[l] = value at backward lag l (nc <= 8)
    for (uint l = 1; l < a.nc; ++l) xs[l] = hist[(nh - l) * a.ch + c];
    float w[8];
    for (uint l = 0; l < a.nc; ++l) w[l] = taps[c * a.nc + (a.nc - 1 - l)];
    uint h = a.dyn_heads ? c / (a.ch / a.dyn_heads) : 0;
    for (uint t = 0; t < a.n_tokens; ++t) {
        xs[0] = x[(ulong)t * a.ch + c];
        float acc = 0.0f;
        for (uint l = 0; l < a.nc; ++l) acc += w[l] * xs[l];
        if (a.dyn_heads) {
            device const float *dt = dyn + ((ulong)t * a.dyn_heads + h) * a.nc;
            for (uint l = 0; l < a.nc; ++l) acc += dt[l] * xs[l];
        }
        y[(ulong)t * a.ch + c] = (xs[0] + acc) * a.scale;
        for (uint l = a.nc - 1; l > 0; --l) xs[l] = xs[l - 1];
    }
    for (uint l = 1; l < a.nc; ++l) hist[(nh - l) * a.ch + c] = xs[l];
}

// ---- attention decode --------------------------------------------------------------------------

struct AttnArgs {
    uint head_dim;    // == threads per threadgroup (128)
    uint n_head;
    uint n_kv_head;
    uint n_rel;
    uint pos0;        // position of query token 0; token t sits at pos0 + t
    uint n_swa;       // sliding window, 0 for full attention
    uint extent;      // relative bias is zero for distance >= extent
    float scale;
    uint kv_dim;      // floats per cached position
};

#define ATTN_CHUNK 128

// Causal attention for T query tokens, one threadgroup per (head, token). Keys for all T tokens
// must already be in the cache. Thread c owns output dimension c. Keys are walked in chunks of
// ATTN_CHUNK with an online softmax (running max m, sum l, accumulator rescaled when m grows);
// the per-query sink logit joins the normalizer at the end and contributes no value.
// q: [T][n_head][head_dim], r: [T][n_head][n_rel], sinks: [T][n_head], out: [T][n_head][head_dim].
kernel void attn_flash(device const float *q [[buffer(0)]], device const float *kc [[buffer(1)]], device const float *vc [[buffer(2)]],
                       device const float *r [[buffer(3)]], device const float *rel_bias [[buffer(4)]],
                       device const float *sinks [[buffer(5)]], device float *out [[buffer(6)]], constant AttnArgs &a [[buffer(7)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    // launched with head_dim threads (the grid is 2-D, so threads_per_threadgroup would have to be too)
    const uint ntg = a.head_dim;
    threadgroup float scratch[32];
    threadgroup float qs[256];
    threadgroup float rs[32];
    threadgroup float pl[ATTN_CHUNK];
    uint h = tg.x, t = tg.y;
    uint kvh = h / (a.n_head / a.n_kv_head);
    uint pos = a.pos0 + t;
    uint first = (a.n_swa && pos + 1 > a.n_swa) ? pos + 1 - a.n_swa : 0;
    device const float *qh = q + ((ulong)t * a.n_head + h) * a.head_dim;
    device const float *rh = r + ((ulong)t * a.n_head + h) * a.n_rel;
    for (uint i = tid; i < a.head_dim; i += ntg) qs[i] = qh[i];
    for (uint i = tid; i < a.n_rel; i += ntg) rs[i] = rh[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (uint j0 = first; j0 <= pos; j0 += ATTN_CHUNK) {
        uint n = min((uint)ATTN_CHUNK, pos + 1 - j0);
        float lg = -INFINITY;
        if (tid < n) {
            uint j = j0 + tid;
            device const float4 *kj = (device const float4 *)(kc + (ulong)j * a.kv_dim + kvh * a.head_dim);
            threadgroup const float4 *q4 = (threadgroup const float4 *)qs;
            float d = 0.0f;
            for (uint i = 0; i < a.head_dim / 4; ++i) d += dot(q4[i], kj[i]);
            uint dist = pos - j;
            float bias = 0.0f;
            if (dist < a.extent) {
                device const float *bt = rel_bias + dist * a.n_rel;
                for (uint i = 0; i < a.n_rel; ++i) bias += rs[i] * bt[i];
            }
            lg = d * a.scale + bias;
        }
        float m_new = max(m, tg_max(lg, scratch, tid, ntg));
        float p = tid < n ? exp(lg - m_new) : 0.0f;
        pl[tid] = p;
        float corr = exp(m - m_new);
        l = l * corr + tg_sum(p, scratch, tid, ntg);
        acc *= corr;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < a.head_dim) {
            device const float *vp = vc + (ulong)j0 * a.kv_dim + kvh * a.head_dim + tid;
            for (uint jj = 0; jj < n; ++jj) acc += pl[jj] * vp[(ulong)jj * a.kv_dim];
        }
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sink = sinks[(ulong)t * a.n_head + h];
    float mf = max(m, sink);
    float total = l * exp(m - mf) + exp(sink - mf);
    if (tid < a.head_dim) out[((ulong)t * a.n_head + h) * a.head_dim + tid] = acc * exp(m - mf) / total;
}

// ---- elementwise -------------------------------------------------------------------------------

struct NArgs {
    uint n;
    float f;
};

kernel void sigmoid_mul(device float *o [[buffer(0)]], device const float *g [[buffer(1)]], constant NArgs &a [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < a.n) o[i] *= sigmoidf(g[i]);
}

kernel void add(device const float *x [[buffer(0)]], device const float *y [[buffer(1)]], device float *o [[buffer(2)]],
                constant NArgs &a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) o[i] = x[i] + y[i];
}

kernel void copy_f32(device const float *x [[buffer(0)]], device float *o [[buffer(1)]], constant NArgs &a [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < a.n) o[i] = x[i];
}

kernel void softcap_inplace(device float *x [[buffer(0)]], constant NArgs &a [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) x[i] = softcapf(x[i], a.f);
}

// (1 - tanh(f)) * x + tanh(g) * m
kernel void msg_gate_combine(device const float *x [[buffer(0)]], device const float *m [[buffer(1)]], device const float *f [[buffer(2)]],
                             device const float *g [[buffer(3)]], device float *o [[buffer(4)]], constant NArgs &a [[buffer(5)]],
                             uint i [[thread_position_in_grid]]) {
    if (i < a.n) o[i] = (1.0f - precise::tanh(f[i])) * x[i] + precise::tanh(g[i]) * m[i];
}

struct EmbedArgs {
    uint d;
    uint n_tokens;
};

// o[t] = dequant(E[tokens[t]]). Grid: (ceil(d/256), n_tokens).
kernel void embed_q8_0(device const block_q8_0 *E [[buffer(0)]], device const uint *tokens [[buffer(1)]], device float *o [[buffer(2)]],
                       constant EmbedArgs &a [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, t = gid.y;
    if (i >= a.d || t >= a.n_tokens) return;
    device const block_q8_0 *row = E + (ulong)tokens[t] * (a.d / QK);
    o[(ulong)t * a.d + i] = float(row[i / QK].d) * float(row[i / QK].qs[i % QK]);
}

// Greedy pick: index of the max logit. One threadgroup.
kernel void argmax(device const float *x [[buffer(0)]], device uint *out [[buffer(1)]], constant NArgs &a [[buffer(2)]],
                   uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    threadgroup float bv[1024];
    threadgroup uint bi[1024];
    float best = -INFINITY;
    uint idx = 0;
    for (uint i = tid; i < a.n; i += ntg) {
        if (x[i] > best) { best = x[i]; idx = i; }
    }
    bv[tid] = best;
    bi[tid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ntg / 2; s > 0; s >>= 1) {
        if (tid < s && (bv[tid + s] > bv[tid] || (bv[tid + s] == bv[tid] && bi[tid + s] < bi[tid]))) {
            bv[tid] = bv[tid + s];
            bi[tid] = bi[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[0] = bi[0];
}

struct StridedNormArgs {
    uint n;
    float eps;
    uint out_stride;
};

// y[row*out_stride + i] = x[row*n + i] / rms(x[row]) (unweighted). Writes one half of the
// [rows][2d] message-gate input. Grid: rows.
kernel void rms_norm_strided(device const float *x [[buffer(0)]], device float *y [[buffer(1)]], constant StridedNormArgs &a [[buffer(2)]],
                             uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    x += (ulong)row * a.n;
    y += (ulong)row * a.out_stride;
    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) ss += x[i] * x[i];
    float inv = rsqrt(tg_sum(ss, scratch, tid, ntg) / a.n + a.eps);
    for (uint i = tid; i < a.n; i += ntg) y[i] = x[i] * inv;
}

// ---- split-K decode attention -------------------------------------------------------------------
// For one query token, a single threadgroup per head leaves the GPU idle and re-reads each KV head
// once per query head. Here a threadgroup takes one KV head and one block of ATTN_SPLIT keys for
// all the query heads that share it (GQA), writes partial (max, sum, accumulator) results, and
// attn_combine merges the blocks per head, adding the sink logit.

#define ATTN_SPLIT 256
#define ATTN_MAX_GRP 4

struct SplitArgs {
    uint head_dim;    // == threads per threadgroup (128)
    uint n_head;
    uint n_kv_head;
    uint n_rel;
    uint pos;         // query position
    uint first;       // first visible key
    uint extent;
    float scale;
    uint kv_dim;
    uint n_split;     // blocks of ATTN_SPLIT keys covering [first, pos]
};

// Grid: (n_kv_head, n_split), head_dim threads. Partials: pm/pl [n_head][n_split], pacc [n_head][n_split][head_dim].
kernel void attn_split(device const float *q [[buffer(0)]], device const float *kc [[buffer(1)]], device const float *vc [[buffer(2)]],
                       device const float *r [[buffer(3)]], device const float *rel_bias [[buffer(4)]], device float *pm [[buffer(5)]],
                       device float *pl [[buffer(6)]], device float *pacc [[buffer(7)]], constant SplitArgs &a [[buffer(8)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint ntg = a.head_dim;
    const uint grp = a.n_head / a.n_kv_head;
    threadgroup float scratch[32];
    threadgroup float qs[ATTN_MAX_GRP * 128];
    threadgroup float rs[ATTN_MAX_GRP * 32];
    threadgroup float pl_t[ATTN_MAX_GRP][ATTN_CHUNK];
    uint kvh = tg.x, s = tg.y;
    for (uint g = 0; g < grp; ++g) {
        uint h = kvh * grp + g;
        for (uint i = tid; i < a.head_dim; i += ntg) qs[g * a.head_dim + i] = q[h * a.head_dim + i];
        for (uint i = tid; i < a.n_rel; i += ntg) rs[g * a.n_rel + i] = r[h * a.n_rel + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint lo = a.first + s * ATTN_SPLIT;
    uint hi = min(lo + ATTN_SPLIT, a.pos + 1);
    float m[ATTN_MAX_GRP], l[ATTN_MAX_GRP], acc[ATTN_MAX_GRP];
    for (uint g = 0; g < grp; ++g) { m[g] = -INFINITY; l[g] = 0.0f; acc[g] = 0.0f; }
    threadgroup float lg_t[ATTN_MAX_GRP][ATTN_CHUNK];
    uint sg = tid / 32, lane = tid % 32, nsg = ntg / 32;
    for (uint j0 = lo; j0 < hi; j0 += ATTN_CHUNK) {
        uint n = min((uint)ATTN_CHUNK, hi - j0);
        // logits: one simdgroup per key, each lane 4 of the 128 dims, so a key row is one
        // contiguous 512-byte read (rows of the cache are kv_dim floats apart)
        for (uint t = sg; t < n; t += nsg) {
            uint j = j0 + t;
            float4 kv = ((device const float4 *)(kc + (ulong)j * a.kv_dim + kvh * a.head_dim))[lane];
            uint dist = a.pos - j;
            for (uint g = 0; g < grp; ++g) {
                float d = simd_sum(dot(((threadgroup const float4 *)(qs + g * a.head_dim))[lane], kv));
                float bias = 0.0f;
                if (dist < a.extent && lane < a.n_rel) bias = rs[g * a.n_rel + lane] * rel_bias[dist * a.n_rel + lane];
                bias = simd_sum(bias);
                if (lane == 0) lg_t[g][t] = d * a.scale + bias;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float lg[ATTN_MAX_GRP];
        for (uint g = 0; g < grp; ++g) lg[g] = tid < n ? lg_t[g][tid] : -INFINITY;
        for (uint g = 0; g < grp; ++g) {
            float m_new = max(m[g], tg_max(lg[g], scratch, tid, ntg));
            float p = tid < n ? exp(lg[g] - m_new) : 0.0f;
            pl_t[g][tid] = p;
            float corr = exp(m[g] - m_new);
            l[g] = l[g] * corr + tg_sum(p, scratch, tid, ntg);
            acc[g] *= corr;
            m[g] = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const float *vp = vc + (ulong)j0 * a.kv_dim + kvh * a.head_dim + tid;
        for (uint jj = 0; jj < n; ++jj) {
            float v = vp[(ulong)jj * a.kv_dim];
            for (uint g = 0; g < grp; ++g) acc[g] += pl_t[g][jj] * v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint g = 0; g < grp; ++g) {
        uint h = kvh * grp + g;
        uint idx = h * a.n_split + s;
        if (tid == 0) { pm[idx] = m[g]; pl[idx] = l[g]; }
        pacc[(ulong)idx * a.head_dim + tid] = acc[g];
    }
}

struct CombineAttnArgs {
    uint head_dim;
    uint n_split;
};

// out[h] = sum_s acc_s exp(m_s - M) / (sum_s l_s exp(m_s - M) + exp(sink - M)). Grid: n_head, head_dim threads.
kernel void attn_combine(device const float *pm [[buffer(0)]], device const float *pl [[buffer(1)]], device const float *pacc [[buffer(2)]],
                         device const float *sinks [[buffer(3)]], device float *out [[buffer(4)]], constant CombineAttnArgs &a [[buffer(5)]],
                         uint h [[threadgroup_position_in_grid]], uint c [[thread_position_in_threadgroup]]) {
    float sink = sinks[h];
    float M = sink;
    for (uint s = 0; s < a.n_split; ++s) M = max(M, pm[h * a.n_split + s]);
    float total = exp(sink - M), acc = 0.0f;
    for (uint s = 0; s < a.n_split; ++s) {
        uint idx = h * a.n_split + s;
        float w = exp(pm[idx] - M);  // exp(-inf) = 0 for empty blocks
        total += pl[idx] * w;
        acc += pacc[(ulong)idx * a.head_dim + c] * w;
    }
    out[h * a.head_dim + c] = acc / total;
}
