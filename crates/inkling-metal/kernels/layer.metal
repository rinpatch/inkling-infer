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
// Grid: one threadgroup per token. All partial sums (|S_s|^2 and q.S_s for every query) are
// accumulated in one pass and reduced together, instead of one barrier-heavy reduction each.
kernel void depth_mix(device const float *blocks [[buffer(0)]], device const float *cur [[buffer(1)]],
                      device const float *queries [[buffer(2)]], device float *out [[buffer(3)]], constant MixArgs &a [[buffer(4)]],
                      uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                      uint ntg [[threads_per_threadgroup]]) {
    // value v: ss[s] = v (s < ns), dot[q][s] = ns + q*ns + s
    threadgroup float part[32][MIX_MAX_STATES * (MIX_MAX_Q + 1)];
    threadgroup float inv_rms[MIX_MAX_STATES];
    threadgroup float p[MIX_MAX_Q][MIX_MAX_STATES];
    blocks += (ulong)t * a.d;
    cur += (ulong)t * a.d;
    out += (ulong)t * a.d;
    uint ns = a.n_blocks + 1;
    uint nv = ns * (a.n_q + 1);
#define state(s) ((s) < a.n_blocks ? blocks + (ulong)(s) * a.plane : cur)

    float acc[MIX_MAX_STATES * (MIX_MAX_Q + 1)];
    for (uint v = 0; v < nv; ++v) acc[v] = 0.0f;
    for (uint i = tid; i < a.d; i += ntg) {
        float qv[MIX_MAX_Q];
        for (uint q = 0; q < a.n_q; ++q) qv[q] = queries[q * a.d + i];
        for (uint s = 0; s < ns; ++s) {
            float x = state(s)[i];
            acc[s] += x * x;
            for (uint q = 0; q < a.n_q; ++q) acc[ns + q * ns + s] += x * qv[q];
        }
    }
    uint sg = tid / 32, lane = tid % 32, nsg = (ntg + 31) / 32;
    for (uint v = 0; v < nv; ++v) {
        float r = simd_sum(acc[v]);
        if (lane == 0) part[sg][v] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < nv) {
        float r = 0.0f;
        for (uint k = 0; k < nsg; ++k) r += part[k][tid];
        part[0][tid] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < ns) inv_rms[tid] = rsqrt(part[0][tid] / a.d + 1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < a.n_q) {
        float sc = rsqrt(float(a.d));
        float m = -INFINITY;
        for (uint s = 0; s < ns; ++s) {
            p[tid][s] = part[0][ns + tid * ns + s] * inv_rms[s] * sc;
            m = max(m, p[tid][s]);
        }
        float sum = 0.0f;
        for (uint s = 0; s < ns; ++s) { p[tid][s] = exp(p[tid][s] - m); sum += p[tid][s]; }
        for (uint s = 0; s < ns; ++s) p[tid][s] /= sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < a.d; i += ntg) {
        for (uint q = 0; q < a.n_q; ++q) {
            float o = 0.0f;
            for (uint s = 0; s < ns; ++s) o += p[q][s] * inv_rms[s] * state(s)[i];
            out[(ulong)q * a.plane + i] = o;
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
    uint second;      // 0: none, 1: y2 = y, 2: y2 = res + y (fused residual add)
    uint snap;        // 1: also store the history after every token in snap [n_tokens][nc-1][ch]
};

// For t = 0..n_tokens in order: y[t] = (x[t] + sum_l taps[c][nc-1-l]*x_{t-l}
// [+ sum_l dyn[t][h][l]*x_{t-l}]) * scale, shifting the history after each token.
// hist is [nc-1][ch], oldest first. One thread per channel; tokens are sequential per channel.
kernel void conv_step(device const float *x [[buffer(0)]], device float *hist [[buffer(1)]], device const float *taps [[buffer(2)]],
                      device const float *dyn [[buffer(3)]], device float *y [[buffer(4)]], constant ConvArgs &a [[buffer(5)]],
                      device const float *res [[buffer(6)]], device float *y2 [[buffer(7)]], device float *snap [[buffer(8)]],
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
        float out = (xs[0] + acc) * a.scale;
        ulong i = (ulong)t * a.ch + c;
        y[i] = out;
        if (a.second == 1) y2[i] = out;
        else if (a.second == 2) y2[i] = res[i] + out;
        for (uint l = a.nc - 1; l > 0; --l) xs[l] = xs[l - 1];
        if (a.snap)
            for (uint l = 1; l < a.nc; ++l) snap[((ulong)t * nh + nh - l) * a.ch + c] = xs[l];
    }
    for (uint l = 1; l < a.nc; ++l) hist[(nh - l) * a.ch + c] = xs[l];
}

// ---- attention decode --------------------------------------------------------------------------

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

#define ATTN_CHUNK 128  // keys per online-softmax step
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
    uint n_split;     // blocks of `split` keys covering [first, pos]
    uint split;       // keys per block (multiple of 32, at most ATTN_SPLIT)
};

// Grid: (n_kv_head, n_split), head_dim threads. Partials: pm/pl [n_head][n_split], pacc [n_head][n_split][head_dim].
template <typename KV>
kernel void attn_split(device const float *q [[buffer(0)]], device const KV *kc [[buffer(1)]], device const KV *vc [[buffer(2)]],
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

    uint lo = a.first + s * a.split;
    uint hi = min(lo + a.split, a.pos + 1);
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
            float4 kv = float4(((device const vec<KV, 4> *)(kc + (ulong)j * a.kv_dim + kvh * a.head_dim))[lane]);
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
        device const KV *vp = vc + (ulong)j0 * a.kv_dim + kvh * a.head_dim + tid;
        for (uint jj = 0; jj < n; ++jj) {
            float v = float(vp[(ulong)jj * a.kv_dim]);
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

// attn_split for long key ranges: one thread per key computes the whole dot for every query head
// of the group, so no cross-lane reductions per key. Faster once there are many keys per block
// (4K context: +3% decode); with few keys most threads idle on a long serial dot.
template <typename KV>
kernel void attn_split_tpk(device const float *q [[buffer(0)]], device const KV *kc [[buffer(1)]], device const KV *vc [[buffer(2)]],
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

    uint lo = a.first + s * a.split;
    uint hi = min(lo + a.split, a.pos + 1);
    float m[ATTN_MAX_GRP], l[ATTN_MAX_GRP], acc[ATTN_MAX_GRP];
    for (uint g = 0; g < grp; ++g) { m[g] = -INFINITY; l[g] = 0.0f; acc[g] = 0.0f; }
    threadgroup float lg_t[ATTN_MAX_GRP][ATTN_CHUNK];
    for (uint j0 = lo; j0 < hi; j0 += ATTN_CHUNK) {
        uint n = min((uint)ATTN_CHUNK, hi - j0);
        // logits: one thread per key, the whole 128-dim dot for every query head of the group (q is
        // in threadgroup memory), so no cross-lane reductions; the thread also dots its key's
        // relative-bias row with each head's r (0 past the extent)
        if (tid < n) {
            uint j = j0 + tid;
            device const vec<KV, 4> *kr = (device const vec<KV, 4> *)(kc + (ulong)j * a.kv_dim + kvh * a.head_dim);
            float d[ATTN_MAX_GRP] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint i = 0; i < 32; ++i) {
                float4 kv = float4(kr[i]);
                for (uint g = 0; g < grp; ++g) d[g] += dot(((threadgroup const float4 *)(qs + g * a.head_dim))[i], kv);
            }
            uint dist = a.pos - j;
            float bias[ATTN_MAX_GRP] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (dist < a.extent) {
                device const float *rb = rel_bias + (ulong)dist * a.n_rel;
                for (uint i = 0; i < a.n_rel; ++i) {
                    float w = rb[i];
                    for (uint g = 0; g < grp; ++g) bias[g] += rs[g * a.n_rel + i] * w;
                }
            }
            for (uint g = 0; g < grp; ++g) lg_t[g][tid] = d[g] * a.scale + bias[g];
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
        device const KV *vp = vc + (ulong)j0 * a.kv_dim + kvh * a.head_dim + tid;
        for (uint jj = 0; jj < n; ++jj) {
            float v = float(vp[(ulong)jj * a.kv_dim]);
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
// Also applies the output gate: gate[i] <- out[i] * sigmoid(gate[i]) (in place).
kernel void attn_combine(device const float *pm [[buffer(0)]], device const float *pl [[buffer(1)]], device const float *pacc [[buffer(2)]],
                         device const float *sinks [[buffer(3)]], device float *out [[buffer(4)]], constant CombineAttnArgs &a [[buffer(5)]],
                         device float *gate [[buffer(6)]], uint h [[threadgroup_position_in_grid]],
                         uint c [[thread_position_in_threadgroup]]) {
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
    uint i = h * a.head_dim + c;
    float o = acc / total;
    out[i] = o;
    gate[i] = o * sigmoidf(gate[i]);
}

// g[i] = o[i] * sigmoid(g[i]): the attention output gate, written over the gate logits.
kernel void gate_mul(device const float *o [[buffer(0)]], device float *g [[buffer(1)]], constant NArgs &a [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < a.n) g[i] = o[i] * sigmoidf(g[i]);
}

// ---- K / V short conv fused with the per-head normalization ------------------------------------
// One threadgroup per KV head (head_dim threads, one per channel): the conv is per channel, and
// the head's L2 clip (keys) or unweighted RMS norm (values) is a reduction over the threadgroup.
// Writes [n_tokens][kv_dim] at y (the cache rows).

struct ConvHeadArgs {
    uint kv_dim;
    uint head_dim;
    uint nc;
    uint dyn;         // dynamic taps present (full-attention layers)
    float scale;
    uint n_tokens;
    uint mode;        // 0: L2 clip to `threshold` (keys), 1: RMS norm with `eps` (values)
    float threshold;
    float eps;
    uint snap;        // as in ConvArgs
};

template <typename KV>
kernel void conv_head(device const float *x [[buffer(0)]], device float *hist [[buffer(1)]], device const float *taps [[buffer(2)]],
                      device const float *dynw [[buffer(3)]], device KV *y [[buffer(4)]], constant ConvHeadArgs &a [[buffer(5)]],
                      device float *snap [[buffer(6)]], uint h [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float scratch[32];
    uint c = h * a.head_dim + tid;
    uint nh = a.nc - 1;
    float xs[8];
    for (uint l = 1; l < a.nc; ++l) xs[l] = hist[(nh - l) * a.kv_dim + c];
    float w[8];
    for (uint l = 0; l < a.nc; ++l) w[l] = taps[c * a.nc + (a.nc - 1 - l)];
    uint n_heads = a.kv_dim / a.head_dim;
    for (uint t = 0; t < a.n_tokens; ++t) {
        xs[0] = x[(ulong)t * a.kv_dim + c];
        float acc = 0.0f;
        for (uint l = 0; l < a.nc; ++l) acc += w[l] * xs[l];
        if (a.dyn) {
            device const float *dt = dynw + ((ulong)t * n_heads + h) * a.nc;
            for (uint l = 0; l < a.nc; ++l) acc += dt[l] * xs[l];
        }
        float out = (xs[0] + acc) * a.scale;
        float ss = tg_sum(out * out, scratch, tid, a.head_dim);
        float f = a.mode == 0 ? min(1.0f, a.threshold / (sqrt(ss) + 1e-6f)) : rsqrt(ss / a.head_dim + a.eps);
        y[(ulong)t * a.kv_dim + c] = KV(out * f);
        for (uint l = a.nc - 1; l > 0; --l) xs[l] = xs[l - 1];
        if (a.snap)
            for (uint l = 1; l < a.nc; ++l) snap[((ulong)t * nh + nh - l) * a.kv_dim + c] = xs[l];
    }
    for (uint l = 1; l < a.nc; ++l) hist[(nh - l) * a.kv_dim + c] = xs[l];
}

template [[host_name("conv_head_f32")]] kernel void conv_head<float>(device const float *, device float *, device const float *,
                                                                      device const float *, device float *, constant ConvHeadArgs &,
                                                                      device float *, uint, uint);
template [[host_name("conv_head_f16")]] kernel void conv_head<half>(device const float *, device float *, device const float *,
                                                                     device const float *, device half *, constant ConvHeadArgs &,
                                                                     device float *, uint, uint);

#define ATTN_INSTANCES(KV, name)                                                                                                        \
    template [[host_name("attn_split_" name)]] kernel void attn_split<KV>(device const float *, device const KV *, device const KV *,     \
                                                                         device const float *, device const float *, device float *,       \
                                                                         device float *, device float *, constant SplitArgs &, uint2, uint);
#define ATTN_TPK_INSTANCES(KV, name)                                                                                                    \
    template [[host_name("attn_split_tpk_" name)]] kernel void attn_split_tpk<KV>(device const float *, device const KV *, device const KV *, \
                                                                                 device const float *, device const float *, device float *, \
                                                                                 device float *, device float *, constant SplitArgs &, uint2, uint);
ATTN_TPK_INSTANCES(float, "f32")
ATTN_TPK_INSTANCES(half, "f16")
ATTN_INSTANCES(float, "f32")
ATTN_INSTANCES(half, "f16")

// ---- prefill attention, query-tiled ------------------------------------------------------------
// Design follows mlx-lm's fused Inkling kernel (mlx_lm/models/inkling2/kernels.py,
// relative_attention): 32 queries per threadgroup (8 per simdgroup), keys in blocks of 16, scores
// and P.V on simdgroup 8x8 matrices with f32 accumulation, the relative bias looked up from a
// per-(token, head) table by distance, and the sink logit as the start of each row's online
// softmax. Each K/V block load serves 32 queries instead of one.

struct RelTableArgs {
    uint n_rows;   // n_tokens * n_head
    uint n_rel;
    uint extent;
};

// table[row][dist] = r[row] . rel_bias[dist] for dist < extent. Grid: (ceil(extent/256), n_rows).
kernel void rel_table(device const float *r [[buffer(0)]], device const float *rel_bias [[buffer(1)]], device float *table [[buffer(2)]],
                      constant RelTableArgs &a [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    uint dist = gid.x, row = gid.y;
    if (dist >= a.extent || row >= a.n_rows) return;
    device const float *rr = r + (ulong)row * a.n_rel;
    device const float *bt = rel_bias + (ulong)dist * a.n_rel;
    float s = 0.0f;
    for (uint i = 0; i < a.n_rel; ++i) s += rr[i] * bt[i];
    table[(ulong)row * a.extent + dist] = s;
}

struct TiledArgs {
    uint n_tokens;
    uint n_head;
    uint n_kv_head;
    uint pos0;      // position of query token 0 (keys are cached at their positions)
    uint n_swa;     // 0 for full attention
    uint extent;    // bias table length; bias is 0 for distance >= extent
    float scale;
    uint kv_dim;
};

#define TQ 32
#define TK 16
#define THD 128

// Grid: (ceil(n_tokens/TQ), n_head), 128 threads. C: tile/fragment compute type (float or half).
template <typename KV, typename C>
kernel void attn_tiled(device const float *q [[buffer(0)]], device const KV *kc [[buffer(1)]], device const KV *vc [[buffer(2)]],
                       device const float *bias_tab [[buffer(3)]], device const float *sinks [[buffer(4)]], device float *out [[buffer(5)]],
                       constant TiledArgs &a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    typedef typename metal::conditional<metal::is_same<C, half>::value, simdgroup_half8x8, simdgroup_float8x8>::type Mat;
    constexpr uint LDK = TK + 8;
    constexpr uint LDV = THD + 8;
    threadgroup C Ks[THD * LDK];  // K block transposed: Ks[dim][key]
    threadgroup C Vs[TK * LDV];   // V block: Vs[key][dim]

    const uint h = tg.y;
    const uint kvh = h / (a.n_head / a.n_kv_head);
    const uint q0 = tg.x * TQ;
    // lane holds row fm and columns fn, fn+1 of each 8x8 fragment (Apple simdgroup layout)
    const ushort fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const ushort fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const uint row = q0 + sg * 8 + fm;
    const bool row_ok = row < a.n_tokens;
    const uint diag = a.pos0 + row;  // this query's position

    Mat q_frag[THD / 8];
    for (uint db = 0; db < THD / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            device const float *qp = q + ((ulong)row * a.n_head + h) * THD + db * 8 + fn;
            qv = float2(qp[0], qp[1]) * a.scale;
        }
        q_frag[db].thread_elements()[0] = C(qv.x);
        q_frag[db].thread_elements()[1] = C(qv.y);
    }
    simdgroup_float8x8 acc[THD / 8];
    for (uint db = 0; db < THD / 8; ++db) acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float row_max = row_ok ? sinks[(ulong)row * a.n_head + h] : 0.0f;
    float row_sum = 1.0f;  // the sink's exp(sink - row_max) with row_max = sink
    device const float *bias = bias_tab + ((ulong)(row_ok ? row : 0) * a.n_head + h) * a.extent;

    const uint last = a.pos0 + min(q0 + TQ, a.n_tokens) - 1;
    const uint first_pos = a.pos0 + q0;
    uint first_key = (a.n_swa && first_pos + 1 > a.n_swa) ? first_pos + 1 - a.n_swa : 0;
    device const KV *k_head = kc + kvh * THD;
    device const KV *v_head = vc + kvh * THD;

    for (uint k0 = first_key / TK * TK; k0 <= last; k0 += TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = tid; c < TK * THD / 4; c += 128) {
            uint r = c / (THD / 4), col = (c % (THD / 4)) * 4;
            uint key = k0 + r;
            float4 kv = float4(0), vv = float4(0);
            if (key <= last) {
                kv = float4(*(device const vec<KV, 4> *)(k_head + (ulong)key * a.kv_dim + col));
                vv = float4(*(device const vec<KV, 4> *)(v_head + (ulong)key * a.kv_dim + col));
            }
            for (uint j = 0; j < 4; ++j) {
                Ks[(col + j) * LDK + r] = C(kv[j]);
                Vs[r * LDV + col + j] = C(vv[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + sg * 8 >= a.n_tokens) continue;

        simdgroup_float8x8 scores[TK / 8];
        for (uint kb = 0; kb < TK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint db = 0; db < THD / 8; ++db) {
                Mat k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }
        float tile_max = -INFINITY;
        for (uint kb = 0; kb < TK / 8; ++kb) {
            for (uint e = 0; e < 2; ++e) {
                uint key = k0 + kb * 8 + fn + e;
                bool visible = row_ok && key <= diag && (a.n_swa == 0 || diag - key < a.n_swa);
                float s = -INFINITY;
                if (visible) {
                    uint d = diag - key;
                    s = scores[kb].thread_elements()[e] + (d < a.extent ? bias[d] : 0.0f);
                }
                scores[kb].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        // the 4 lanes of a row are lane^1 and lane^8 apart
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        float new_max = max(row_max, tile_max);
        float factor = exp(row_max - new_max);
        float tile_sum = 0.0f;
        Mat probs[TK / 8];
        for (uint kb = 0; kb < TK / 8; ++kb) {
            for (uint e = 0; e < 2; ++e) {
                float p = exp(scores[kb].thread_elements()[e] - new_max);
                probs[kb].thread_elements()[e] = C(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        for (uint db = 0; db < THD / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (uint kb = 0; kb < TK / 8; ++kb) {
                Mat v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }
    if (row_ok) {
        device float *op = out + ((ulong)row * a.n_head + h) * THD;
        for (uint db = 0; db < THD / 8; ++db) {
            op[db * 8 + fn] = acc[db].thread_elements()[0] / row_sum;
            op[db * 8 + fn + 1] = acc[db].thread_elements()[1] / row_sum;
        }
    }
}

#define TILED_INSTANCE(KV, C, name)                                                                                                     \
    template [[host_name("attn_tiled_" name)]] kernel void attn_tiled<KV, C>(device const float *, device const KV *, device const KV *, \
                                                                            device const float *, device const float *, device float *, \
                                                                            constant TiledArgs &, uint2, uint, uint, uint);
TILED_INSTANCE(float, float, "kv32_f32")
TILED_INSTANCE(half, float, "kv16_f32")
TILED_INSTANCE(float, half, "kv32_f16")
TILED_INSTANCE(half, half, "kv16_f16")

struct ArgmaxArgs {
    uint n;
    uint out_index;
};

// Greedy pick on the GPU: tokens[0] <- argmax(logits) (lowest index on ties), also stored at
// out[out_index]. tokens[0] is what the next step's embedding lookup reads, so decode steps can be
// queued back to back without a CPU round trip. One threadgroup of 1024 threads.
kernel void argmax_token(device const float *logits [[buffer(0)]], device uint *tokens [[buffer(1)]], device uint *out [[buffer(2)]],
                         constant ArgmaxArgs &a [[buffer(3)]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup uint bi[32];
    float v = -INFINITY;
    uint idx = 0xffffffffu;
    for (uint i = tid; i < a.n; i += 1024) {
        float x = logits[i];
        if (x > v) { v = x; idx = i; }
    }
    // reduce (value desc, index asc) within the simdgroup, then across simdgroups
    for (uint o = 16; o > 0; o /= 2) {
        float v2 = simd_shuffle_down(v, o);
        uint i2 = simd_shuffle_down(idx, o);
        if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
    }
    if (lane == 0) { bv[sg] = v; bi[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        v = bv[lane];
        idx = bi[lane];
        for (uint o = 16; o > 0; o /= 2) {
            float v2 = simd_shuffle_down(v, o);
            uint i2 = simd_shuffle_down(idx, o);
            if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
        }
        if (lane == 0) { tokens[0] = idx; out[a.out_index] = idx; }
    }
}

// ---- short convolutions over a batch, token-parallel ---------------------------------------------
// A short conv has no recurrence: token t's output reads only the inputs at t-nc+1..t, which are in
// x (or, before the batch, in hist). So every (channel, token) runs at once, where conv_step /
// conv_head walk the tokens of a channel in order. hist is left untouched; conv_hist_update then
// shifts the batch's last nc-1 inputs in (a separate dispatch, after every reader of the old hist).

// value of channel c at batch position p (p < 0: from hist, which is [nc-1][ch], oldest first)
inline float conv_at(device const float *x, device const float *hist, int p, uint nh, uint ch, uint c) {
    return p >= 0 ? x[(ulong)p * ch + c] : hist[(ulong)(int(nh) + p) * ch + c];
}

// Grid: (ceil(ch / 256), n_tokens), 256 threads. Same arguments as conv_step.
kernel void conv_step_par(device const float *x [[buffer(0)]], device const float *hist [[buffer(1)]], device const float *taps [[buffer(2)]],
                          device const float *dyn [[buffer(3)]], device float *y [[buffer(4)]], constant ConvArgs &a [[buffer(5)]],
                          device const float *res [[buffer(6)]], device float *y2 [[buffer(7)]], device float *snap [[buffer(8)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x, t = gid.y;
    if (c >= a.ch || t >= a.n_tokens) return;
    uint nh = a.nc - 1;
    uint h = a.dyn_heads ? c / (a.ch / a.dyn_heads) : 0;
    float acc = 0.0f, x0 = 0.0f;
    for (uint l = 0; l < a.nc; ++l) {
        float v = conv_at(x, hist, int(t) - int(l), nh, a.ch, c);
        if (l == 0) x0 = v;
        acc += taps[c * a.nc + (a.nc - 1 - l)] * v;
        if (a.dyn_heads) acc += dyn[((ulong)t * a.dyn_heads + h) * a.nc + l] * v;
    }
    float out = (x0 + acc) * a.scale;
    ulong i = (ulong)t * a.ch + c;
    y[i] = out;
    if (a.second == 1) y2[i] = out;
    else if (a.second == 2) y2[i] = res[i] + out;
    // history after token t: row nh-l holds the input at t+1-l
    if (a.snap)
        for (uint l = 1; l < a.nc; ++l) snap[((ulong)t * nh + nh - l) * a.ch + c] = conv_at(x, hist, int(t + 1) - int(l), nh, a.ch, c);
}

// Grid: (n_kv_heads, n_tokens), head_dim threads. Same arguments as conv_head.
template <typename KV>
kernel void conv_head_par(device const float *x [[buffer(0)]], device const float *hist [[buffer(1)]], device const float *taps [[buffer(2)]],
                          device const float *dynw [[buffer(3)]], device KV *y [[buffer(4)]], constant ConvHeadArgs &a [[buffer(5)]],
                          device float *snap [[buffer(6)]], uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    uint h = g.x, t = g.y;
    uint c = h * a.head_dim + tid;
    uint nh = a.nc - 1;
    uint n_heads = a.kv_dim / a.head_dim;
    float acc = 0.0f, x0 = 0.0f;
    for (uint l = 0; l < a.nc; ++l) {
        float v = conv_at(x, hist, int(t) - int(l), nh, a.kv_dim, c);
        if (l == 0) x0 = v;
        acc += taps[c * a.nc + (a.nc - 1 - l)] * v;
        if (a.dyn) acc += dynw[((ulong)t * n_heads + h) * a.nc + l] * v;
    }
    float out = (x0 + acc) * a.scale;
    float ss = tg_sum(out * out, scratch, tid, a.head_dim);
    float f = a.mode == 0 ? min(1.0f, a.threshold / (sqrt(ss) + 1e-6f)) : rsqrt(ss / a.head_dim + a.eps);
    y[(ulong)t * a.kv_dim + c] = KV(out * f);
    if (a.snap)
        for (uint l = 1; l < a.nc; ++l) snap[((ulong)t * nh + nh - l) * a.kv_dim + c] = conv_at(x, hist, int(t + 1) - int(l), nh, a.kv_dim, c);
}

template [[host_name("conv_head_par_f32")]] kernel void conv_head_par<float>(device const float *, device const float *, device const float *,
                                                                              device const float *, device float *, constant ConvHeadArgs &,
                                                                              device float *, uint2, uint);
template [[host_name("conv_head_par_f16")]] kernel void conv_head_par<half>(device const float *, device const float *, device const float *,
                                                                             device const float *, device half *, constant ConvHeadArgs &,
                                                                             device float *, uint2, uint);

// hist <- the last nc-1 inputs once the batch of n tokens is in: row r holds position n-nh+r
// (from x, or from an older hist row when the batch is shorter than nh). Rows ascend and read only
// higher old rows, so one thread per channel can shift in place. Grid: ceil(ch / 256), 256 threads.
struct ConvHistArgs {
    uint ch;
    uint nc;
    uint n_tokens;
};

kernel void conv_hist_update(device const float *x [[buffer(0)]], device float *hist [[buffer(1)]], constant ConvHistArgs &a [[buffer(2)]],
                             uint c [[thread_position_in_grid]]) {
    if (c >= a.ch) return;
    uint nh = a.nc - 1;
    for (uint r = 0; r < nh; ++r) {
        int p = int(a.n_tokens) - int(nh) + int(r);
        hist[(ulong)r * a.ch + c] = p >= 0 ? x[(ulong)p * a.ch + c] : hist[(ulong)(int(nh) + p) * a.ch + c];
    }
}

// ---- depth mix for decode and short batches: split over the hidden dimension -------------------
// depth_mix runs one threadgroup per token, so a decode step had one core read every block state
// (~15 us per call, 120 calls per token). Here MIX_SLICES threadgroups per token each take a slice
// of d: depth_mix_part writes the slice's partial |S_s|^2 and q.S_s sums, depth_mix_apply reduces
// the partials (a few dozen floats), takes the softmax over states and writes its slice of out.
#define MIX_SLICES 9
#define MIX_NV (MIX_MAX_STATES * (MIX_MAX_Q + 1))
// Gram terms S_s . S_s' (s < s') follow the nv sums when the mix output gets RMS-normed: its squared
// norm is sum_ss' w_s w_s' S_s.S_s', so the norm that follows the mix needs no pass of its own.
#define MIX_NG (MIX_MAX_STATES * (MIX_MAX_STATES - 1) / 2)
#define MIX_PART (MIX_NV + MIX_NG)

template <bool GRAM>
inline void mix_acc(thread float *acc, thread const float *v, thread const float *qv, uint ns, uint nq) {
    for (uint s = 0; s < ns; ++s) {
        acc[s] += v[s] * v[s];
        for (uint q = 0; q < nq; ++q) acc[ns + q * ns + s] += v[s] * qv[q];
    }
    if (GRAM) {
        uint k = ns * (nq + 1);
        for (uint s = 0; s < ns; ++s)
            for (uint s2 = s + 1; s2 < ns; ++s2) acc[k++] += v[s] * v[s2];
    }
}

// Reduces each thread's acc[0..n) over the threadgroup into part[0..n).
inline void mix_reduce(thread float *acc, uint n, threadgroup float (*red)[MIX_PART], device float *part, uint tid) {
    uint sg = tid / 32, lane = tid % 32;
    for (uint v = 0; v < n; ++v) {
        float r = simd_sum(acc[v]);
        if (lane == 0) red[sg][v] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < n) {
        float r = 0.0f;
        for (uint k = 0; k < 8; ++k) r += red[k][tid];
        part[tid] = r;
    }
}

inline uint mix_nsums(uint ns, uint nq, bool gram) { return ns * (nq + 1) + (gram ? ns * (ns - 1) / 2 : 0); }

// Grid: (MIX_SLICES, n_tokens), 256 threads. part: [token][slice][MIX_PART].
template <bool GRAM>
kernel void depth_mix_part(device const float *blocks [[buffer(0)]], device const float *cur [[buffer(1)]],
                           device const float *queries [[buffer(2)]], device float *part [[buffer(3)]], constant MixArgs &a [[buffer(4)]],
                           uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[8][MIX_PART];
    uint t = g.y, sl = g.x;
    blocks += (ulong)t * a.d;
    cur += (ulong)t * a.d;
    uint ns = a.n_blocks + 1, n = mix_nsums(ns, a.n_q, GRAM);
    uint chunk = (a.d + MIX_SLICES - 1) / MIX_SLICES;
    uint lo = sl * chunk, hi = min(lo + chunk, a.d);
    float acc[GRAM ? MIX_PART : MIX_NV];
    for (uint v = 0; v < n; ++v) acc[v] = 0.0f;
    for (uint i = lo + tid; i < hi; i += 256) {
        float qv[MIX_MAX_Q], v[MIX_MAX_STATES];
        for (uint q = 0; q < a.n_q; ++q) qv[q] = queries[q * a.d + i];
        for (uint s = 0; s < ns; ++s) v[s] = s < a.n_blocks ? blocks[(ulong)s * a.plane + i] : cur[i];
        mix_acc<GRAM>(acc, v, qv, ns, a.n_q);
    }
    mix_reduce(acc, n, red, part + ((ulong)t * MIX_SLICES + sl) * MIX_PART, tid);
}

template [[host_name("depth_mix_part")]] kernel void depth_mix_part<false>(device const float *, device const float *, device const float *,
                                                                           device float *, constant MixArgs &, uint2, uint);
template [[host_name("depth_mix_part_gram")]] kernel void depth_mix_part<true>(device const float *, device const float *, device const float *,
                                                                               device float *, constant MixArgs &, uint2, uint);

struct MixConvArgs {
    uint nc;
    float scale;
    uint push;  // 1: block boundary, blocks[n_blocks-1] = res and cur = y; 0: cur = res + y
};

// One decode token: the post-attention conv (static taps, one thread per channel as in conv_step)
// and its residual bookkeeping, then depth_mix_part's sums. Each thread reads back only the
// channels it wrote, so the conv needs no barrier of its own; this saves a dependent dispatch.
// Grid: (MIX_SLICES, 1), 256 threads.
template <bool GRAM>
kernel void depth_mix_part_conv(device float *blocks [[buffer(0)]], device float *cur [[buffer(1)]],
                                device const float *queries [[buffer(2)]], device float *part [[buffer(3)]],
                                constant MixArgs &a [[buffer(4)]], device const float *x [[buffer(5)]],
                                device float *hist [[buffer(6)]], device const float *taps [[buffer(7)]],
                                device float *y [[buffer(8)]], device const float *res [[buffer(9)]],
                                constant MixConvArgs &c [[buffer(10)]],
                                uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[8][MIX_PART];
    uint sl = g.x;
    uint ns = a.n_blocks + 1, n = mix_nsums(ns, a.n_q, GRAM);
    uint chunk = (a.d + MIX_SLICES - 1) / MIX_SLICES;
    uint lo = sl * chunk, hi = min(lo + chunk, a.d);
    uint nh = c.nc - 1;
    float acc[GRAM ? MIX_PART : MIX_NV];
    for (uint v = 0; v < n; ++v) acc[v] = 0.0f;
    for (uint i = lo + tid; i < hi; i += 256) {
        float xs[8];
        xs[0] = x[i];
        for (uint l = 1; l < c.nc; ++l) xs[l] = hist[(nh - l) * a.d + i];
        float conv = 0.0f;
        for (uint l = 0; l < c.nc; ++l) conv += taps[i * c.nc + (c.nc - 1 - l)] * xs[l];
        float out = (xs[0] + conv) * c.scale;
        for (uint l = 1; l < c.nc; ++l) hist[(nh - l) * a.d + i] = xs[l - 1];
        y[i] = out;
        float r = res[i];
        float cv = c.push ? out : r + out;
        cur[i] = cv;
        if (c.push) blocks[(ulong)(a.n_blocks - 1) * a.plane + i] = r;
        float qv[MIX_MAX_Q], v[MIX_MAX_STATES];
        for (uint q = 0; q < a.n_q; ++q) qv[q] = queries[q * a.d + i];
        for (uint s = 0; s < ns; ++s) v[s] = s < a.n_blocks ? blocks[(ulong)s * a.plane + i] : cv;
        mix_acc<GRAM>(acc, v, qv, ns, a.n_q);
    }
    mix_reduce(acc, n, red, part + (ulong)sl * MIX_PART, tid);
}

#define MIX_PART_CONV_ARGS                                                                                                              \
    device float *, device float *, device const float *, device float *, constant MixArgs &, device const float *, device float *, \
        device const float *, device float *, device const float *, constant MixConvArgs &, uint2, uint
template [[host_name("depth_mix_part_conv")]] kernel void depth_mix_part_conv<false>(MIX_PART_CONV_ARGS);
template [[host_name("depth_mix_part_conv_gram")]] kernel void depth_mix_part_conv<true>(MIX_PART_CONV_ARGS);

struct MixNormArgs {
    ulong w[MIX_MAX_Q];  // float offsets of each output's norm weights in wbuf
    float eps;
    uint on;             // 0: no norm (part has no Gram terms)
};

// Grid: (MIX_SLICES, n_tokens), 256 threads. With norm on, also writes
// nout[q][t] = rms_norm(out[q][t]) * w_q, using the Gram terms from depth_mix_part_gram.
kernel void depth_mix_apply(device const float *blocks [[buffer(0)]], device const float *cur [[buffer(1)]],
                            device const float *part [[buffer(2)]], device float *out [[buffer(3)]], constant MixArgs &a [[buffer(4)]],
                            device const float *wbuf [[buffer(5)]], device float *nout [[buffer(6)]], constant MixNormArgs &nm [[buffer(7)]],
                            uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tot[MIX_PART];
    threadgroup float inv_rms[MIX_MAX_STATES];
    threadgroup float p[MIX_MAX_Q][MIX_MAX_STATES];
    threadgroup float inv_out[MIX_MAX_Q];
    uint t = g.y, sl = g.x;
    blocks += (ulong)t * a.d;
    cur += (ulong)t * a.d;
    out += (ulong)t * a.d;
    nout += (ulong)t * a.d;
    uint ns = a.n_blocks + 1, nv = ns * (a.n_q + 1), n = mix_nsums(ns, a.n_q, nm.on);
    if (tid < n) {
        float r = 0.0f;
        for (uint k = 0; k < MIX_SLICES; ++k) r += part[((ulong)t * MIX_SLICES + k) * MIX_PART + tid];
        tot[tid] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < ns) inv_rms[tid] = rsqrt(tot[tid] / a.d + 1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < a.n_q) {
        float sc = rsqrt(float(a.d));
        float m = -INFINITY;
        for (uint s = 0; s < ns; ++s) {
            p[tid][s] = tot[ns + tid * ns + s] * inv_rms[s] * sc;
            m = max(m, p[tid][s]);
        }
        float sum = 0.0f;
        for (uint s = 0; s < ns; ++s) { p[tid][s] = exp(p[tid][s] - m); sum += p[tid][s]; }
        for (uint s = 0; s < ns; ++s) p[tid][s] /= sum;
        if (nm.on) {
            // out = sum_s c_s S_s with c_s = p_s / rms_s
            float ss = 0.0f;
            uint k = nv;
            for (uint s = 0; s < ns; ++s) {
                float cs = p[tid][s] * inv_rms[s];
                ss += cs * cs * tot[s];
                for (uint s2 = s + 1; s2 < ns; ++s2) ss += 2.0f * cs * p[tid][s2] * inv_rms[s2] * tot[k++];
            }
            inv_out[tid] = rsqrt(max(ss, 0.0f) / a.d + nm.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint chunk = (a.d + MIX_SLICES - 1) / MIX_SLICES;
    uint lo = sl * chunk, hi = min(lo + chunk, a.d);
#define state(s) ((s) < a.n_blocks ? blocks + (ulong)(s) * a.plane : cur)
    for (uint i = lo + tid; i < hi; i += 256)
        for (uint q = 0; q < a.n_q; ++q) {
            float o = 0.0f;
            for (uint s = 0; s < ns; ++s) o += p[q][s] * inv_rms[s] * state(s)[i];
            out[(ulong)q * a.plane + i] = o;
            if (nm.on) nout[(ulong)q * a.plane + i] = o * inv_out[q] * wbuf[nm.w[q] + i];
        }
#undef state
}

// ---- decode attention for short contexts ---------------------------------------------------------
// Up to ATTN_SHORT keys: one threadgroup per query head does the whole softmax in one pass (logits
// in threadgroup memory, sink in the denominator) and applies the output gate, where attn_split +
// attn_combine cost two dispatches and ~12 barriers per chunk for blocks of 32 keys.
// Grid: n_head threadgroups of head_dim (128) threads. Same buffers as attn_split, plus sinks,
// out and gate as in attn_combine.
#define ATTN_SHORT 256

template <typename KV>
kernel void attn_short(device const float *q [[buffer(0)]], device const KV *kc [[buffer(1)]], device const KV *vc [[buffer(2)]],
                       device const float *r [[buffer(3)]], device const float *rel_bias [[buffer(4)]], device const float *sinks [[buffer(5)]],
                       device float *out [[buffer(6)]], device float *gate [[buffer(7)]], constant SplitArgs &a [[buffer(8)]],
                       uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float qs[128];
    threadgroup float rs[32];
    threadgroup float sc[ATTN_SHORT];
    threadgroup float scratch[32];
    uint kvh = h / (a.n_head / a.n_kv_head);
    qs[tid] = q[h * a.head_dim + tid];
    if (tid < a.n_rel) rs[tid] = r[h * a.n_rel + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n = a.pos + 1 - a.first;
    // logits: thread t handles keys t and t + 128 (whole 128-dim dot and its bias row)
    float lmax = -INFINITY;
    for (uint t = tid; t < n; t += 128) {
        uint j = a.first + t;
        device const vec<KV, 4> *kr = (device const vec<KV, 4> *)(kc + (ulong)j * a.kv_dim + kvh * a.head_dim);
        float d = 0.0f;
        for (uint i = 0; i < 32; ++i) d += dot(((threadgroup const float4 *)qs)[i], float4(kr[i]));
        uint dist = a.pos - j;
        float b = 0.0f;
        if (dist < a.extent) {
            device const float *rb = rel_bias + (ulong)dist * a.n_rel;
            for (uint i = 0; i < a.n_rel; ++i) b += rs[i] * rb[i];
        }
        float s = d * a.scale + b;
        sc[t] = s;
        lmax = max(lmax, s);
    }
    float sink = sinks[h];
    float m = max(tg_max(lmax, scratch, tid, 128), sink);
    float lsum = 0.0f;
    for (uint t = tid; t < n; t += 128) {
        float p = exp(sc[t] - m);
        sc[t] = p;
        lsum += p;
    }
    float total = tg_sum(lsum, scratch, tid, 128) + exp(sink - m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const KV *vp = vc + (ulong)a.first * a.kv_dim + kvh * a.head_dim + tid;
    float acc = 0.0f;
    for (uint t = 0; t < n; ++t) acc += sc[t] * float(vp[(ulong)t * a.kv_dim]);
    uint i = h * a.head_dim + tid;
    float o = acc / total;
    out[i] = o;
    gate[i] = o * sigmoidf(gate[i]);
}

template [[host_name("attn_short_f32")]] kernel void attn_short<float>(device const float *, device const float *, device const float *,
                                                                        device const float *, device const float *, device const float *,
                                                                        device float *, device float *, constant SplitArgs &, uint, uint);
template [[host_name("attn_short_f16")]] kernel void attn_short<half>(device const float *, device const half *, device const half *,
                                                                       device const float *, device const float *, device const float *,
                                                                       device float *, device float *, constant SplitArgs &, uint, uint);

// Timing probe (runtime INKLING_EMPTY): stands in for a kernel with the same grid and barriers.
kernel void empty_probe(uint gid [[thread_position_in_grid]]) {}
