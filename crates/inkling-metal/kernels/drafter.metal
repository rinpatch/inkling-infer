// DSpark drafter (docs/dspark.md) and the speculative round around it. The CPU oracle is
// crates/inkling/src/drafter.rs; dense projections reuse the Q8_0 matvec / GEMM kernels.
#include "common.h"

// out[r] = rmsnorm(sum_t alpha[t] * F[t][row0 + r]) * w, F = the [4][rows][d] feature capture.
// Grid: rows threadgroups of 1024.
struct DrMixArgs {
    uint d;
    uint plane;  // floats between feature planes
    uint row0;
    float eps;
    float alpha[4];
};

kernel void dr_mix(device const float *F [[buffer(0)]], device const float *w [[buffer(1)]], device float *out [[buffer(2)]],
                   constant DrMixArgs &a [[buffer(3)]], uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                   uint ntg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    ulong base = (ulong)(a.row0 + r) * a.d;
    float ss = 0.0f;
    for (uint i = tid; i < a.d; i += ntg) {
        float m = 0.0f;
        for (uint t = 0; t < 4; ++t) m += a.alpha[t] * F[t * a.plane + base + i];
        out[(ulong)r * a.d + i] = m;
        ss += m * m;
    }
    float inv = rsqrt(tg_sum(ss, scratch, tid, ntg) / a.d + a.eps);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = tid; i < a.d; i += ntg) out[(ulong)r * a.d + i] *= inv * w[i];
}

// Per-head RMS norm (weight w[128]) then neox rotary at position pos0 + row, in place on
// x[row][head][128] with rows `row_stride` floats apart. Grid: (heads, rows), 128 threads.
struct DrRopeArgs {
    uint row_stride;
    uint pos0;
    float theta;
    float eps;
};

kernel void dr_head_norm_rope(device float *x [[buffer(0)]], device const float *w [[buffer(1)]], constant DrRopeArgs &a [[buffer(2)]],
                              uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    threadgroup float y[128];
    device float *p = x + (ulong)g.y * a.row_stride + g.x * 128;
    float v = p[tid];
    float inv = rsqrt(tg_sum(v * v, scratch, tid, 128) / 128.0f + a.eps);
    y[tid] = v * inv * w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) {
        float ang = float(a.pos0 + g.y) * precise::pow(a.theta, -float(2 * tid) / 128.0f);
        float s = precise::sin(ang), c = precise::cos(ang);
        float u0 = y[tid], u1 = y[tid + 64];
        p[tid] = u0 * c - u1 * s;
        p[tid + 64] = u0 * s + u1 * c;
    }
}

// Bidirectional attention of block queries over cache keys [lo, hi) (context + the block's own
// keys, no mask): o[r][h] = softmax(q[r][h] . K / sqrt(128)) V, GQA by h / (heads / kv_heads).
// K, V are [pos][kv_dim] f32. Keys go in chunks of 128 with an online softmax.
// Grid: (heads, rows), 128 threads (one per head dimension).
struct DrAttnArgs {
    uint heads;
    uint kv_heads;
    uint kv_dim;
    uint lo;
    uint hi;
    float scale;
};

kernel void dr_attn(device const float *Q [[buffer(0)]], device const float *K [[buffer(1)]], device const float *V [[buffer(2)]],
                    device float *O [[buffer(3)]], constant DrAttnArgs &a [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float q[128];
    threadgroup float sc[128];
    threadgroup float scratch[32];
    uint h = g.x, r = g.y, qd = a.heads * 128;
    uint kh = h / (a.heads / a.kv_heads);
    q[tid] = Q[(ulong)r * qd + h * 128 + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (uint c0 = a.lo; c0 < a.hi; c0 += 128) {
        uint key = c0 + tid;
        float s = -INFINITY;
        if (key < a.hi) {
            device const float4 *kr = (device const float4 *)(K + (ulong)key * a.kv_dim + kh * 128);
            threadgroup const float4 *q4 = (threadgroup const float4 *)q;
            float d = 0.0f;
            for (uint j = 0; j < 32; ++j) d += dot(q4[j], kr[j]);
            s = d * a.scale;
        }
        float cm = tg_max(s, scratch, tid, 128);
        float mn = max(m, cm);
        float e = key < a.hi ? exp(s - mn) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sc[tid] = e;
        float cs = tg_sum(e, scratch, tid, 128);
        float f = exp(m - mn);
        acc *= f;
        l = l * f + cs;
        m = mn;
        uint n = min(128u, a.hi - c0);
        for (uint j = 0; j < n; ++j) acc += sc[j] * V[(ulong)(c0 + j) * a.kv_dim + kh * 128 + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    O[(ulong)r * qd + h * 128 + tid] = acc / l;
}

struct DrNArgs {
    uint n;
};

// g <- silu(g) * u
kernel void dr_silu_mul(device float *g [[buffer(0)]], device const float *u [[buffer(1)]], constant DrNArgs &a [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < a.n) {
        float x = g[i];
        g[i] = x / (1.0f + exp(-x)) * u[i];
    }
}

// Top-8 of each row of base [rows][n] (largest first; lowest index on ties). Each of 256 threads
// keeps a sorted local top-8, then simdgroup 0 picks the 8 best of the 2048 candidates.
// Grid: rows threadgroups of 256.
#define DR_TOPK 8
struct DrTopkArgs {
    uint n;
};

inline bool dr_better(float v, uint i, float bv, uint bi) { return v > bv || (v == bv && i < bi); }

kernel void dr_topk8(device const float *base [[buffer(0)]], device uint *ids [[buffer(1)]], device float *vals [[buffer(2)]],
                     constant DrTopkArgs &a [[buffer(3)]], uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tv[256 * DR_TOPK];
    threadgroup uint ti[256 * DR_TOPK];
    float v[DR_TOPK];
    uint ix[DR_TOPK];
    for (uint k = 0; k < DR_TOPK; ++k) { v[k] = -INFINITY; ix[k] = 0xffffffffu; }
    device const float *row = base + (ulong)r * a.n;
    for (uint i = tid; i < a.n; i += 256) {
        float x = row[i];
        if (!dr_better(x, i, v[DR_TOPK - 1], ix[DR_TOPK - 1])) continue;
        uint k = DR_TOPK - 1;
        while (k > 0 && dr_better(x, i, v[k - 1], ix[k - 1])) { v[k] = v[k - 1]; ix[k] = ix[k - 1]; --k; }
        v[k] = x;
        ix[k] = i;
    }
    for (uint k = 0; k < DR_TOPK; ++k) { tv[tid * DR_TOPK + k] = v[k]; ti[tid * DR_TOPK + k] = ix[k]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    for (uint k = 0; k < DR_TOPK; ++k) {
        float bv = -INFINITY;
        uint bi = 0xffffffffu, bj = 0;
        for (uint j = lane; j < 256 * DR_TOPK; j += 32)
            if (dr_better(tv[j], ti[j], bv, bi)) { bv = tv[j]; bi = ti[j]; bj = j; }
        for (uint o = 16; o > 0; o /= 2) {
            float v2 = simd_shuffle_down(bv, o);
            uint i2 = simd_shuffle_down(bi, o), j2 = simd_shuffle_down(bj, o);
            if (dr_better(v2, i2, bv, bi)) { bv = v2; bi = i2; bj = j2; }
        }
        bj = simd_broadcast_first(bj);
        if (lane == 0) {
            ids[r * DR_TOPK + k] = bi;
            vals[r * DR_TOPK + k] = bv;
            tv[bj] = -INFINITY;
            ti[bj] = 0xffffffffu;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Markov step input: xin = [state (rank), w1[prev] (rank), h[s] (d), f_rnn (d)].
struct DrWalkInArgs {
    uint s;
    uint d;
    uint rank;
};

kernel void dr_walk_in(device const float *state [[buffer(0)]], device const half *w1 [[buffer(1)]], device const uint *prev [[buffer(2)]],
                       device const float *h [[buffer(3)]], device const float *fr [[buffer(4)]], device float *xin [[buffer(5)]],
                       constant DrWalkInArgs &a [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    uint R = a.rank, d = a.d;
    if (i < R) xin[i] = state[i];
    else if (i < 2 * R) xin[i] = float(w1[(ulong)prev[0] * R + (i - R)]);
    else if (i < 2 * R + d) xin[i] = h[(ulong)a.s * d + (i - 2 * R)];
    else if (i < 2 * R + 2 * d) xin[i] = fr[i - 2 * R - d];
}

// p = rnn_proj xin (without bias): g = sigmoid(p0+b0); state = g*state + (1-g)*tanh(p1+b1);
// latent = tanh(p2+b2). One threadgroup of `rank` threads.
kernel void dr_walk_state(device const float *p [[buffer(0)]], device const float *b [[buffer(1)]], device float *state [[buffer(2)]],
                          device float *latent [[buffer(3)]], constant DrNArgs &a [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint R = a.n;
    if (i >= R) return;
    float g = 1.0f / (1.0f + exp(-(p[i] + b[i])));
    state[i] = g * state[i] + (1.0f - g) * precise::tanh(p[R + i] + b[R + i]);
    latent[i] = precise::tanh(p[2 * R + i] + b[2 * R + i]);
}

// Draft s: among the 8 base candidates of row s, argmax of softcap(base + W_U[c] . (hid + hb)).
// Writes drafts[s], prev[0], and tokens[1 + s] (the verify batch). One threadgroup of 8 simdgroups.
struct DrPickArgs {
    uint s;
    uint d;
    float softcap;
};

kernel void dr_walk_pick(device const block_q8_0 *WU [[buffer(0)]], device const float *hid [[buffer(1)]], device const float *hb [[buffer(2)]],
                         device const uint *cid [[buffer(3)]], device const float *cval [[buffer(4)]], device uint *drafts [[buffer(5)]],
                         device uint *prev [[buffer(6)]], device uint *tokens [[buffer(7)]], constant DrPickArgs &a [[buffer(8)]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float score[DR_TOPK];
    uint c = cid[a.s * DR_TOPK + sg];
    uint nb = a.d / QK;
    device const block_q8_0 *row = WU + (ulong)c * nb;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32) {
        float s = 0.0f;
        for (uint j = 0; j < QK; ++j) s += float(row[b].qs[j]) * (hid[b * QK + j] + hb[b * QK + j]);
        acc += s * float(row[b].d);
    }
    acc = simd_sum(acc);
    if (lane == 0) score[sg] = softcapf(cval[a.s * DR_TOPK + sg] + acc, a.softcap);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        uint best = 0;
        for (uint k = 1; k < DR_TOPK; ++k)
            if (score[k] > score[best] || (score[k] == score[best] && cid[a.s * DR_TOPK + k] < cid[a.s * DR_TOPK + best])) best = k;
        uint t = cid[a.s * DR_TOPK + best];
        drafts[a.s] = t;
        prev[0] = t;
        tokens[1 + a.s] = t;
    }
}

// Per-row argmax of logits [rows][n] (lowest index on ties). Grid: rows threadgroups of 1024.
kernel void dr_argmax_rows(device const float *x [[buffer(0)]], device uint *out [[buffer(1)]], constant DrNArgs &a [[buffer(2)]],
                           uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup uint bi[32];
    device const float *row = x + (ulong)r * a.n;
    float v = -INFINITY;
    uint idx = 0xffffffffu;
    for (uint i = tid; i < a.n; i += 1024)
        if (row[i] > v) { v = row[i]; idx = i; }
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
        if (lane == 0) out[r] = idx;
    }
}

// Greedy acceptance of a verify batch [anchor, d1..dm]: n = leading drafts equal to the target's
// argmax of the previous row; bonus = the target's token after the last kept row.
// res = {n, bonus}; tokens[0] = bonus (the next anchor). One thread.
kernel void dr_accept(device const uint *row_arg [[buffer(0)]], device uint *tokens [[buffer(1)]], device uint *res [[buffer(2)]],
                      constant DrNArgs &a [[buffer(3)]]) {
    uint n = 0;
    while (n < a.n && tokens[1 + n] == row_arg[n]) ++n;
    res[0] = n;
    res[1] = row_arg[n];
    tokens[0] = row_arg[n];
}

// hist[i] = snap[(keep-1) * len + i] with keep = res[0] + 1 (conv rewind after a verify).
kernel void dr_conv_rewind(device const float *snap [[buffer(0)]], device float *hist [[buffer(1)]], device const uint *res [[buffer(2)]],
                           constant DrNArgs &a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) hist[i] = snap[(ulong)res[0] * a.n + i];
}

// out[t][i] = F[t][row][i] for the last kept row (row = res[0]; res = null means row = a.n2).
struct DrFeatArgs {
    uint d;
    uint plane;
    uint row;
    uint indirect;
};

kernel void dr_copy_feat_row(device const float *F [[buffer(0)]], device float *out [[buffer(1)]], device const uint *res [[buffer(2)]],
                             constant DrFeatArgs &a [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
    uint i = g.x, t = g.y;
    if (i >= a.d) return;
    uint row = a.indirect ? res[0] : a.row;
    out[t * a.d + i] = F[t * a.plane + (ulong)row * a.d + i];
}
