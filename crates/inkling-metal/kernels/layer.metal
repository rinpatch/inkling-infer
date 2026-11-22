// Per-token layer building blocks: norms, depth attention, clipping, short convolutions,
// attention decode and small elementwise ops. Semantics: docs/reference/ARCHITECTURE.md.
#include "common.h"

// ---- norms -------------------------------------------------------------------------------------

struct NormArgs {
    uint n;
    float eps;
    uint has_weight;
};

// y = x / rms(x) (* w). One threadgroup.
kernel void rms_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *y [[buffer(2)]],
                     constant NormArgs &a [[buffer(3)]], uint tid [[thread_position_in_threadgroup]],
                     uint ntg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
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
    uint dim;        // per-head vector length
    float threshold;
    uint has_scale;
    float mul;       // applied after clipping (log-position scale on full layers, else 1)
};

// x_h = min(1, c / (|s*x_h| + 1e-6)) * s*x_h * mul, in place. Grid: n_heads threadgroups of 32.
kernel void clip_heads(device float *x [[buffer(0)]], device const float *scale [[buffer(1)]], constant ClipArgs &a [[buffer(2)]],
                       uint h [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    device float *v = x + h * a.dim;
    device const float *s = scale + h * a.dim;
    float ss = 0.0f;
    for (uint i = lane; i < a.dim; i += 32) {
        float t = v[i] * (a.has_scale ? s[i] : 1.0f);
        v[i] = t;
        ss += t * t;
    }
    float f = min(1.0f, a.threshold / (sqrt(simd_sum(ss)) + 1e-6f)) * a.mul;
    for (uint i = lane; i < a.dim; i += 32) v[i] *= f;
}

// ---- depth attention ("attention residuals") ---------------------------------------------------

struct MixArgs {
    uint d;
    uint n_blocks;  // states = blocks..., cur
    uint n_q;
};

#define MIX_MAX_STATES 8
#define MIX_MAX_Q 4

// out[q] = sum_s softmax_s(q . rms(S_s) / sqrt(d)) * rms(S_s), S = [blocks..., cur]. One threadgroup.
kernel void depth_mix(device const float *blocks [[buffer(0)]], device const float *cur [[buffer(1)]],
                      device const float *queries [[buffer(2)]], device float *out [[buffer(3)]], constant MixArgs &a [[buffer(4)]],
                      uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    threadgroup float inv_rms[MIX_MAX_STATES];
    threadgroup float p[MIX_MAX_Q][MIX_MAX_STATES];
    uint ns = a.n_blocks + 1;
#define state(s) ((s) < a.n_blocks ? blocks + (s) * a.d : cur)

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
            out[q * a.d + i] = acc;
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
};

// y = (x + sum_l taps[c][nc-1-l]*x_{t-l} [+ sum_l dyn[h][l]*x_{t-l}]) * scale; history shifts by one.
// hist is [nc-1][ch], oldest first. One thread per channel.
kernel void conv_step(device const float *x [[buffer(0)]], device float *hist [[buffer(1)]], device const float *taps [[buffer(2)]],
                      device const float *dyn [[buffer(3)]], device float *y [[buffer(4)]], constant ConvArgs &a [[buffer(5)]],
                      uint c [[thread_position_in_grid]]) {
    if (c >= a.ch) return;
    uint nh = a.nc - 1;
    float xs[8];  // xs[l] = value at backward lag l (nc <= 8)
    xs[0] = x[c];
    for (uint l = 1; l < a.nc; ++l) xs[l] = hist[(nh - l) * a.ch + c];
    float acc = 0.0f;
    for (uint l = 0; l < a.nc; ++l) acc += taps[c * a.nc + (a.nc - 1 - l)] * xs[l];
    if (a.dyn_heads) {
        uint h = c / (a.ch / a.dyn_heads);
        for (uint l = 0; l < a.nc; ++l) acc += dyn[h * a.nc + l] * xs[l];
    }
    y[c] = (xs[0] + acc) * a.scale;
    for (uint r = 0; r + 1 < nh; ++r) hist[r * a.ch + c] = hist[(r + 1) * a.ch + c];
    hist[(nh - 1) * a.ch + c] = xs[0];
}

// ---- attention decode --------------------------------------------------------------------------

struct AttnArgs {
    uint head_dim;
    uint n_head;
    uint n_kv_head;
    uint n_rel;
    uint pos;         // query position; keys first..=pos are cached
    uint first;
    uint extent;      // relative bias is zero for distance >= extent
    float scale;
    uint kv_dim;      // floats per cached position
    uint ctx;         // scratch row length (max positions)
};

// One threadgroup per head: logits (q.k*scale + rel bias) into scratch, softmax including the sink
// logit, then out = sum p_j v_j. Two passes over the cache; flash-style splitting comes later.
kernel void attn_decode(device const float *q [[buffer(0)]], device const float *kc [[buffer(1)]], device const float *vc [[buffer(2)]],
                        device const float *r [[buffer(3)]], device const float *rel_bias [[buffer(4)]],
                        device const float *sinks [[buffer(5)]], device float *scratch_logits [[buffer(6)]], device float *out [[buffer(7)]],
                        constant AttnArgs &a [[buffer(8)]], uint h [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    uint kvh = h / (a.n_head / a.n_kv_head);
    device const float *qh = q + h * a.head_dim;
    device const float *rh = r + h * a.n_rel;
    device float *lg = scratch_logits + (ulong)h * a.ctx;
    uint n = a.pos - a.first + 1;

    float m = -INFINITY;
    for (uint t = tid; t < n; t += ntg) {
        uint j = a.first + t;
        device const float *kj = kc + (ulong)j * a.kv_dim + kvh * a.head_dim;
        float dotv = 0.0f;
        for (uint i = 0; i < a.head_dim; i += 4) dotv += dot(*(device const float4 *)(qh + i), *(device const float4 *)(kj + i));
        uint dist = a.pos - j;
        float bias = 0.0f;
        if (dist < a.extent) {
            device const float *bt = rel_bias + dist * a.n_rel;
            for (uint i = 0; i < a.n_rel; ++i) bias += rh[i] * bt[i];
        }
        float l = dotv * a.scale + bias;
        lg[t] = l;
        m = max(m, l);
    }
    float sink = sinks[h];
    m = max(tg_max(m, scratch, tid, ntg), sink);
    float sum = 0.0f;
    for (uint t = tid; t < n; t += ntg) {
        float e = exp(lg[t] - m);
        lg[t] = e;
        sum += e;
    }
    sum = tg_sum(sum, scratch, tid, ntg) + exp(sink - m);
    threadgroup_barrier(mem_flags::mem_device);
    float inv = 1.0f / sum;
    for (uint c = tid; c < a.head_dim; c += ntg) {
        float acc = 0.0f;
        for (uint t = 0; t < n; ++t) acc += lg[t] * vc[(ulong)(a.first + t) * a.kv_dim + kvh * a.head_dim + c];
        out[h * a.head_dim + c] = acc * inv;
    }
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
    uint token;
};

kernel void embed_q8_0(device const block_q8_0 *E [[buffer(0)]], device float *o [[buffer(1)]], constant EmbedArgs &a [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= a.d) return;
    device const block_q8_0 *row = E + (ulong)a.token * (a.d / QK);
    o[i] = float(row[i / QK].d) * float(row[i / QK].qs[i % QK]);
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
