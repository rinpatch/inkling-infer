// Matrix-vector products for single-token decode. One simdgroup per output row, 8 simdgroups per
// threadgroup. Rows are independent, so the grid is just ceil(rows / 8) threadgroups.
#include "common.h"

struct MvArgs {
    uint rows;
    uint cols;
};

#define MV_SG 8

kernel void mv_q8_0(device const block_q8_0 *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
                    constant MvArgs &a [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint r = tg * MV_SG + sg;
    if (r >= a.rows) return;
    uint nb = a.cols / QK;
    float s = simd_sum(dot_q8_0_lane(W + r * nb, x, nb, lane));
    if (lane == 0) y[r] = s;
}

// Batched over tokens on grid y: y[t] = W x[t]. Weights are small (router) and stay in cache.
kernel void mv_f32(device const float *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
                   constant MvArgs &a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint r = tg.x * MV_SG + sg;
    if (r >= a.rows) return;
    x += (ulong)tg.y * a.cols;
    y += (ulong)tg.y * a.rows;
    device const float *w = W + (ulong)r * a.cols;
    float acc = 0.0f;
    for (uint i = lane; i < a.cols; i += 32) acc += w[i] * x[i];
    float s = simd_sum(acc);
    if (lane == 0) y[r] = s;
}

// --- MoE ---------------------------------------------------------------------------------------

struct MoeArgs {
    uint n_embd;      // d
    uint n_ff;        // expert hidden size
    uint n_used;      // experts per token (k)
    float situ_beta;  // gate softcap
    float situ_lin;   // up softcap
};

// Block payloads start at byte 2 of 34- or 18-byte blocks, so they are only 2-byte aligned; vector
// loads must use the packed (byte-aligned) types.

// One block's partial dot product. IQ4_NL decodes through a 16-entry table that the kernel stages
// in threadgroup memory: lanes index it with different nibbles, which serializes in `constant`.
inline float block_dot(device const block_q8_0 *b, device const float *x, threadgroup const float *) {
    device const packed_char4 *q = (device const packed_char4 *)b->qs;
    device const float4 *xv = (device const float4 *)x;
    float s = 0.0f;
    for (uint j = 0; j < QK / 4; ++j) s += dot(float4(char4(q[j])), xv[j]);
    return s * float(b->d);
}

inline float block_dot(device const block_iq4_nl *b, device const float *x, threadgroup const float *tab) {
    device const packed_uchar4 *q = (device const packed_uchar4 *)b->qs;
    device const float4 *lo = (device const float4 *)x;
    device const float4 *hi = (device const float4 *)(x + QK / 2);
    float s = 0.0f;
    for (uint j = 0; j < QK / 8; ++j) {
        uchar4 v = uchar4(q[j]);
        s += dot(float4(tab[v.x & 15], tab[v.y & 15], tab[v.z & 15], tab[v.w & 15]), lo[j]);
        s += dot(float4(tab[v.x >> 4], tab[v.y >> 4], tab[v.z >> 4], tab[v.w >> 4]), hi[j]);
    }
    return s * float(b->d);
}

inline void load_table(threadgroup float *tab, uint tid) {
    if (tid < 16) tab[tid] = IQ4_NL_VALUES[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Half-block partial dot: lane half h in {0,1} covers half of block b's elements, so a block is
// split across two lanes and every lane loads only its slice of x.
//   Q8_0:   elements [16h, 16h+16)
//   IQ4_NL: bytes [8h, 8h+8) -> elements [8h, 8h+8) (low nibbles) and [16+8h, 16+8h+8) (high)
struct XHalf {
    float4 v[4];
};

inline XHalf load_x_half(block_q8_0, device const float *x, uint h) {
    device const float4 *p = (device const float4 *)(x + 16 * h);
    return XHalf{{p[0], p[1], p[2], p[3]}};
}

inline XHalf load_x_half(block_iq4_nl, device const float *x, uint h) {
    device const float4 *lo = (device const float4 *)(x + 8 * h);
    device const float4 *hi = (device const float4 *)(x + 16 + 8 * h);
    return XHalf{{lo[0], lo[1], hi[0], hi[1]}};
}

inline float half_dot(device const block_q8_0 *b, thread const XHalf &x, uint h, threadgroup const float *) {
    device const packed_char4 *q = (device const packed_char4 *)(b->qs + 16 * h);
    return float(b->d) * (dot(float4(char4(q[0])), x.v[0]) + dot(float4(char4(q[1])), x.v[1]) + dot(float4(char4(q[2])), x.v[2]) +
                          dot(float4(char4(q[3])), x.v[3]));
}

inline float half_dot(device const block_iq4_nl *b, thread const XHalf &x, uint h, threadgroup const float *tab) {
    device const packed_uchar4 *q = (device const packed_uchar4 *)(b->qs + 8 * h);
    uchar4 a = uchar4(q[0]), c = uchar4(q[1]);
    float s = dot(float4(tab[a.x & 15], tab[a.y & 15], tab[a.z & 15], tab[a.w & 15]), x.v[0]) +
              dot(float4(tab[c.x & 15], tab[c.y & 15], tab[c.z & 15], tab[c.w & 15]), x.v[1]) +
              dot(float4(tab[a.x >> 4], tab[a.y >> 4], tab[a.z >> 4], tab[a.w >> 4]), x.v[2]) +
              dot(float4(tab[c.x >> 4], tab[c.y >> 4], tab[c.z >> 4], tab[c.w >> 4]), x.v[3]);
    return float(b->d) * s;
}

#define MOE_NR 2  // ff rows (gate+up pairs) per simdgroup in gate_up; d rows per simdgroup in down

// act[k][i] = softcap(g)*sigmoid(g)*softcap(u), [g; u] = GateUp_{ids[k]} x, rows i and n_ff+i.
// A simdgroup computes MOE_NR consecutive i (2*MOE_NR rows), reusing each loaded x slice for all
// of them. Grid: (ceil(n_ff / (MV_SG*MOE_NR)), n_used).
template <typename B>
kernel void moe_gate_up(device const B *W [[buffer(0)]], device const float *x [[buffer(1)]], device const uint *ids [[buffer(2)]],
                        device float *act [[buffer(3)]], constant MoeArgs &a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tab[16];
    load_table(tab, tid);
    uint i0 = (tg.x * MV_SG + sg) * MOE_NR;
    uint k = tg.y;
    if (i0 >= a.n_ff) return;
    uint nb = a.n_embd / QK;
    device const B *E = W + (ulong)ids[k] * 2 * a.n_ff * nb;
    float g[MOE_NR] = {0}, u[MOE_NR] = {0};
    for (uint t = lane; t < 2 * nb; t += 32) {
        uint b = t / 2, h = t % 2;
        XHalf xs = load_x_half(B{}, x + b * QK, h);
        for (uint r = 0; r < MOE_NR; ++r) {
            g[r] += half_dot(E + (ulong)(i0 + r) * nb + b, xs, h, tab);
            u[r] += half_dot(E + (ulong)(a.n_ff + i0 + r) * nb + b, xs, h, tab);
        }
    }
    for (uint r = 0; r < MOE_NR; ++r) {
        float gs = simd_sum(g[r]), us = simd_sum(u[r]);
        if (lane == 0 && i0 + r < a.n_ff) act[k * a.n_ff + i0 + r] = softcapf(gs, a.situ_beta) * sigmoidf(gs) * softcapf(us, a.situ_lin);
    }
}

// y[r] = sum_k w[k] * Down_{ids[k]}[r] . act[k]. A simdgroup computes MOE_NR rows, walking all k
// experts' half-blocks as one range (nb is only 36 for n_ff = 1152). Grid: ceil(n_embd / (MV_SG*MOE_NR)).
template <typename B>
kernel void moe_down(device const B *W [[buffer(0)]], device const float *act [[buffer(1)]], device const uint *ids [[buffer(2)]],
                     device const float *wts [[buffer(3)]], device float *y [[buffer(4)]], constant MoeArgs &a [[buffer(5)]],
                     uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tab[16];
    load_table(tab, tid);
    uint r0 = (tg * MV_SG + sg) * MOE_NR;
    if (r0 >= a.n_embd) return;
    uint nb = a.n_ff / QK;
    float acc[MOE_NR] = {0};
    for (uint t = lane; t < a.n_used * nb * 2; t += 32) {
        uint k = t / (2 * nb), b = (t / 2) % nb, h = t % 2;
        XHalf xs = load_x_half(B{}, act + k * a.n_ff + b * QK, h);
        device const B *base = W + ((ulong)ids[k] * a.n_embd + r0) * nb + b;
        float wk = wts[k];
        for (uint r = 0; r < MOE_NR; ++r) acc[r] += wk * half_dot(base + (ulong)r * nb, xs, h, tab);
    }
    for (uint r = 0; r < MOE_NR; ++r) {
        float v = simd_sum(acc[r]);
        if (lane == 0 && r0 + r < a.n_embd) y[r0 + r] = v;
    }
}

#define MOE_INSTANCES(T, name)                                                                                                     \
    template [[host_name("moe_gate_up_" name)]] kernel void moe_gate_up<T>(device const T *, device const float *,                \
                                                                           device const uint *, device float *, constant MoeArgs &, \
                                                                           uint2, uint, uint, uint);                                 \
    template [[host_name("moe_down_" name)]] kernel void moe_down<T>(device const T *, device const float *, device const uint *,   \
                                                                     device const float *, device float *, constant MoeArgs &, uint, \
                                                                     uint, uint, uint);
MOE_INSTANCES(block_iq4_nl, "iq4_nl")
MOE_INSTANCES(block_q8_0, "q8_0")

struct TopkArgs {
    uint n_expert;
    uint k;
    float scale;
};

// Router selection: probs = sigmoid(logits); pick top-k of probs + bias; weights = scale * p / sum(p).
// One threadgroup of n_expert threads (<= 1024): k rounds of a threadgroup argmax (ties -> lowest id).
kernel void moe_topk(device const float *logits [[buffer(0)]], device const float *bias [[buffer(1)]], device uint *ids [[buffer(2)]],
                     device float *wts [[buffer(3)]], constant TopkArgs &a [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    logits += (ulong)t * a.n_expert;
    ids += t * a.k;
    wts += t * a.k;
    threadgroup uint chosen[16];
    float p = tid < a.n_expert ? sigmoidf(logits[tid]) : 0.0f;
    float score = tid < a.n_expert ? p + bias[tid] : -INFINITY;
    float sum = 0.0f;
    for (uint j = 0; j < a.k; ++j) {
        float m = tg_max(score, scratch, tid, ntg);
        // lowest id holding the max: max of -id over the candidates
        float cand = score == m ? -float(tid) : -INFINITY;
        uint best = uint(-tg_max(cand, scratch, tid, ntg));
        if (tid == best) {
            chosen[j] = tid;
            score = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // every thread accumulates the chosen probability from the owner's value via threadgroup memory
        if (tid == best) scratch[0] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum += scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < a.k) {
        uint e = chosen[tid];
        ids[tid] = e;
        wts[tid] = sigmoidf(logits[e]) / (sum + 1e-20f) * a.scale;
    }
}

// ---- loop message gate: out = (1 - tanh(F z)) * x + tanh(G z) * m, z = D [rms(x); rms(m)] --------

struct GateArgs {
    uint d;
    uint rank;
};

// z[r] = D[r] . [rms(x); rms(m)]. One threadgroup per rank row; each recomputes the two norms
// (2*d floats, served from cache) instead of a separate dispatch.
// Grid: (rank, n_tokens), 1024 threads.
kernel void msg_gate_down(device const float *D [[buffer(0)]], device const float *x [[buffer(1)]], device const float *m [[buffer(2)]],
                          device float *z [[buffer(3)]], constant GateArgs &a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    const uint ntg = 1024;
    uint r = tg.x;
    x += (ulong)tg.y * a.d;
    m += (ulong)tg.y * a.d;
    z += (ulong)tg.y * a.rank;
    float sx = 0.0f, sm = 0.0f, dx = 0.0f, dm = 0.0f;
    device const float *row = D + (ulong)r * 2 * a.d;
    for (uint i = tid; i < a.d; i += ntg) {
        float xv = x[i], mv = m[i];
        sx += xv * xv;
        sm += mv * mv;
        dx += row[i] * xv;
        dm += row[a.d + i] * mv;
    }
    float ix = rsqrt(tg_sum(sx, scratch, tid, ntg) / a.d + 1e-6f);
    float im = rsqrt(tg_sum(sm, scratch, tid, ntg) / a.d + 1e-6f);
    float tx = tg_sum(dx, scratch, tid, ntg);
    float tm = tg_sum(dm, scratch, tid, ntg);
    if (tid == 0) z[r] = tx * ix + tm * im;
}

// One simdgroup per channel c; out may alias x (each channel reads and writes only its own c).
// Grid: (ceil(d / MV_SG), n_tokens).
kernel void msg_gate_up(device const float *F [[buffer(0)]], device const float *G [[buffer(1)]], device const float *z [[buffer(2)]],
                        device const float *x [[buffer(3)]], device const float *m [[buffer(4)]], device float *out [[buffer(5)]],
                        constant GateArgs &a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint c = tg.x * MV_SG + sg;
    if (c >= a.d) return;
    z += (ulong)tg.y * a.rank;
    x += (ulong)tg.y * a.d;
    m += (ulong)tg.y * a.d;
    out += (ulong)tg.y * a.d;
    device const float *fr = F + (ulong)c * a.rank;
    device const float *gr = G + (ulong)c * a.rank;
    float f = 0.0f, g = 0.0f;
    for (uint j = lane; j < a.rank; j += 32) {
        f += fr[j] * z[j];
        g += gr[j] * z[j];
    }
    f = simd_sum(f);
    g = simd_sum(g);
    if (lane == 0) out[c] = (1.0f - precise::tanh(f)) * x[c] + precise::tanh(g) * m[c];
}

// ---- several Q8_0 matrices that share one input, in one dispatch -------------------------------
// Small projections (sinks: 24 rows, conv taps: 32, rel: 384) can't fill the GPU on their own;
// concatenating their rows gives one large grid. Segment s covers global rows
// [start, start+rows) of the matrix at byte offset woff and writes output buffer o_s.

#define MULTI_MAX 8

struct MultiSeg {
    ulong woff;
    uint start;
    uint rows;
};

struct MultiArgs {
    uint cols;
    uint n_seg;
    uint total_rows;
    uint pad;
    MultiSeg seg[MULTI_MAX];
};

kernel void mv_q8_0_multi(device const uchar *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *o0 [[buffer(2)]],
                          device float *o1 [[buffer(3)]], device float *o2 [[buffer(4)]], device float *o3 [[buffer(5)]],
                          device float *o4 [[buffer(6)]], device float *o5 [[buffer(7)]], device float *o6 [[buffer(8)]],
                          device float *o7 [[buffer(9)]], constant MultiArgs &a [[buffer(10)]], uint tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint r = tg * MV_SG + sg;
    if (r >= a.total_rows) return;
    uint s = 0;
    while (s + 1 < a.n_seg && r >= a.seg[s + 1].start) ++s;
    uint lr = r - a.seg[s].start;
    uint nb = a.cols / QK;
    device const block_q8_0 *row = (device const block_q8_0 *)(W + a.seg[s].woff) + (ulong)lr * nb;
    float v = simd_sum(dot_q8_0_lane(row, x, nb, lane));
    if (lane != 0) return;
    device float *outs[MULTI_MAX] = {o0, o1, o2, o3, o4, o5, o6, o7};
    outs[s][lr] = v;
}

// y[t][r] = W[r] . x[t] with a whole threadgroup per row (split over its simdgroups), for short
// fat f32 matrices like the 128-row router that would otherwise launch only a few threadgroups.
// Grid: (rows, n_tokens), 256 threads.
kernel void mv_f32_rowtg(device const float *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
                         constant MvArgs &a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    device const float4 *w = (device const float4 *)(W + (ulong)tg.x * a.cols);
    device const float4 *xv = (device const float4 *)(x + (ulong)tg.y * a.cols);
    float acc = 0.0f;
    for (uint i = tid; i < a.cols / 4; i += 256) acc += dot(w[i], xv[i]);
    float s = tg_sum(acc, scratch, tid, 256);
    if (tid == 0) y[(ulong)tg.y * a.rows + tg.x] = s;
}
