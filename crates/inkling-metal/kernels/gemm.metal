// Batched (prefill) matrix products with simdgroup 8x8 matrices. Tiles: 32 weight rows x 32
// tokens x 32 columns (one quantized block) per step, 4 simdgroups each owning a 16x16 output
// sub-tile. Weights are dequantized into threadgroup memory as f32; accumulation is f32.
#include "common.h"

#define BM 32  // weight rows per tile
#define BN 32  // tokens per tile
#define BK 32  // columns per step (= QK)
// Tiles are instantiated with f32 elements (default: prefill matches decode to ~1e-6) and f16
// elements (opt-in: ~2x MMA rate and half the threadgroup traffic, but the rounding builds up
// through the KV cache over long prompts; see MetalModel::exact_prefill). Accumulation is f32.

// Dequantize one 32-column block of weight row `row` into dst[0..32).
template <typename E>
inline void deq_block(device const block_q8_0 *row, uint b, threadgroup E *dst, uint part, threadgroup const float *) {
    // part in 0..3: 8 values each
    device const block_q8_0 *blk = row + b;
    float d = float(blk->d);
    for (uint j = 0; j < 8; ++j) dst[part * 8 + j] = E(d * float(blk->qs[part * 8 + j]));
}

template <typename E>
inline void deq_block(device const block_iq4_nl *row, uint b, threadgroup E *dst, uint part, threadgroup const float *tab) {
    device const block_iq4_nl *blk = row + b;
    float d = float(blk->d);
    // part 0,1 -> low nibbles of bytes [8p, 8p+8) = elements [8p, 8p+8); part 2,3 -> high nibbles
    uint p = part & 1;
    bool hi = part >= 2;
    for (uint j = 0; j < 8; ++j) {
        uchar q = blk->qs[p * 8 + j];
        dst[(hi ? 16 : 0) + p * 8 + j] = E(d * tab[hi ? (q >> 4) : (q & 15)]);
    }
}

// Accumulates C[32 rows][32 tokens] += W[rows][k-block] * X[tokens][k-block]^T for one k step.
// Wt: [BM][BK], Xt: [BN][BK] (row-major, padded stride BK+... not needed with 8x8 loads).
template <typename E, typename Mat>
inline void mma_step_t(threadgroup const E *Wt, threadgroup const E *Xt, thread simdgroup_float8x8 acc[2][2], uint sg) {
    uint rb = (sg / 2) * 16, tb = (sg % 2) * 16;
    for (uint kk = 0; kk < BK; kk += 8) {
        Mat a[2], b[2];
        for (uint i = 0; i < 2; ++i) simdgroup_load(a[i], Wt + (rb + 8 * i) * BK + kk, BK);
        // B = X^T: load X rows (tokens) as columns with transpose
        for (uint j = 0; j < 2; ++j) simdgroup_load(b[j], Xt + (tb + 8 * j) * BK + kk, BK, ulong2(0, 0), true);
        for (uint i = 0; i < 2; ++i)
            for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
}

inline void mma_step(threadgroup const half *Wt, threadgroup const half *Xt, thread simdgroup_float8x8 acc[2][2], uint sg) {
    mma_step_t<half, simdgroup_half8x8>(Wt, Xt, acc, sg);
}
inline void mma_step(threadgroup const float *Wt, threadgroup const float *Xt, thread simdgroup_float8x8 acc[2][2], uint sg) {
    mma_step_t<float, simdgroup_float8x8>(Wt, Xt, acc, sg);
}

// Stores acc (rows x tokens) into Y[token][row] for valid rows/tokens via a threadgroup bounce.
inline void store_tile(thread simdgroup_float8x8 acc[2][2], threadgroup float *Ct, uint sg) {
    uint rb = (sg / 2) * 16, tb = (sg % 2) * 16;
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) simdgroup_store(acc[i][j], Ct + (rb + 8 * i) * BN + tb + 8 * j, BN);
}

// ---- several Q8_0 matrices sharing one input, batched -------------------------------------------

#define MULTI_MAX 8

struct GMultiSeg {
    ulong woff;
    uint start;
    uint rows;
};

struct GMultiArgs {
    uint cols;
    uint n_seg;
    uint total_rows;
    uint n_tokens;
    GMultiSeg seg[MULTI_MAX];
};

// ---- MoE prefill ---------------------------------------------------------------------------------

struct GroupArgs {
    uint n_pairs;   // n_tokens * n_used
    uint n_expert;
};

// Groups (token, k) pairs by expert. Thread e scans all pairs in order, so the per-expert lists are
// deterministic. Writes count[e], offset[e] (exclusive prefix) and pairs[offset[e] + i] = pair index.
// One threadgroup of n_expert threads.
kernel void moe_group(device const uint *ids [[buffer(0)]], device uint *count [[buffer(1)]], device uint *offset [[buffer(2)]],
                      device uint *pairs [[buffer(3)]], constant GroupArgs &a [[buffer(4)]], uint e [[thread_position_in_threadgroup]]) {
    threadgroup uint cnt[1024];
    uint c = 0;
    if (e < a.n_expert)
        for (uint p = 0; p < a.n_pairs; ++p) c += ids[p] == e;
    cnt[e] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e >= a.n_expert) return;
    uint off = 0;
    for (uint i = 0; i < e; ++i) off += cnt[i];
    count[e] = c;
    offset[e] = off;
    uint w = off;
    for (uint p = 0; p < a.n_pairs; ++p)
        if (ids[p] == e) pairs[w++] = p;
}

struct MoeGemmArgs {
    uint n_embd;
    uint n_ff;
    uint n_used;
    float situ_beta;
    float situ_lin;
};

struct CombineArgs {
    uint n_embd;
    uint n_used;
    uint n_tokens;
};

// y[t][c] = sum_k w[t][k] * yp[t*k_used + k][c]; summed in k order, so results don't depend on
// how pairs were grouped.
kernel void moe_combine(device const float *yp [[buffer(0)]], device const float *wts [[buffer(1)]], device float *y [[buffer(2)]],
                        constant CombineArgs &a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.n_tokens * a.n_embd) return;
    uint t = i / a.n_embd, c = i % a.n_embd;
    float acc = 0.0f;
    for (uint k = 0; k < a.n_used; ++k) acc += wts[t * a.n_used + k] * yp[((ulong)t * a.n_used + k) * a.n_embd + c];
    y[i] = acc;
}

// ---- f32 weights (loop message gates), batched ----------------------------------------------------

struct GemmF32Args {
    uint rows;
    uint cols;      // multiple of BK
    uint n_tokens;
};

// y[t][r] = W[r] . x[t] for row-major f32 W. Grid: (ceil(rows/BM), ceil(n_tokens/BN)), 128 threads.
kernel void gemm_f32(device const float *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
                     constant GemmF32Args &a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Wt[BM * BK];
    threadgroup float Xt[BN * BK];
    threadgroup float Ct[BM * BN];
    uint r0 = tg.x * BM, t0 = tg.y * BN;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0);
    uint lr = tid / 4, part = tid % 4;
    uint row = r0 + lr, tok = t0 + lr;
    for (uint k0 = 0; k0 < a.cols; k0 += BK) {
        for (uint j = 0; j < 8; ++j) {
            uint c = k0 + part * 8 + j;
            Wt[lr * BK + part * 8 + j] = row < a.rows ? W[(ulong)row * a.cols + c] : 0.0f;
            Xt[lr * BK + part * 8 + j] = tok < a.n_tokens ? x[(ulong)tok * a.cols + c] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_step(Wt, Xt, acc, sg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    store_tile(acc, Ct, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BM * BN; i += 128) {
        uint rr = i / BN, tk = i % BN;
        if (r0 + rr < a.rows && t0 + tk < a.n_tokens) y[(ulong)(t0 + tk) * a.rows + r0 + rr] = Ct[i];
    }
}

// ---- MoE prefill GEMMs ------------------------------------------------------------------------
// gate_up: act[pair] = situ(GateUp_e x[token(pair)]); down: yp[pair] = Down_e act[pair], for the
// pairs moe_group routed to expert e. Each simdgroup owns a 32-row x 16-token output (4x2
// accumulators), so a k step does 6 threadgroup-matrix loads per 8 multiplies (a 16x16 layout
// does 4 per 4). Threadgroup memory is one raw array: the weight/activation
// tiles during the k loop, then the f32 output tiles in the epilogue (32 KB limit).

#define BM2 64

template <typename E>
inline void mma_4x2(threadgroup const E *Wt, threadgroup const E *Xt, thread simdgroup_float8x8 acc[4][2], uint sg) {
    typedef typename metal::conditional<metal::is_same<E, half>::value, simdgroup_half8x8, simdgroup_float8x8>::type Mat;
    uint rb = (sg / 2) * 32, tb = (sg % 2) * 16;
    for (uint kk = 0; kk < BK; kk += 8) {
        Mat a[4], b[2];
        for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], Wt + (rb + 8 * i) * BK + kk, BK);
        for (uint j = 0; j < 2; ++j) simdgroup_load(b[j], Xt + (tb + 8 * j) * BK + kk, BK, ulong2(0, 0), true);
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
}

inline void store_4x2(thread simdgroup_float8x8 acc[4][2], threadgroup float *Ct, uint sg) {
    uint rb = (sg / 2) * 32, tb = (sg % 2) * 16;
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 2; ++j) simdgroup_store(acc[i][j], Ct + (rb + 8 * i) * BN + tb + 8 * j, BN);
}

// Grid: (ceil(n_ff/BM2), ceil(max_pairs/BN), n_expert), 128 threads.
template <typename B, typename E>
kernel void moe_gemm2_gate_up(device const B *W [[buffer(0)]], device const float *x [[buffer(1)]], device const uint *count [[buffer(2)]],
                              device const uint *offset [[buffer(3)]], device const uint *pairs [[buffer(4)]], device float *act [[buffer(5)]],
                              constant MoeGemmArgs &a [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint e = tg.z;
    uint n = count[e];
    uint p0 = tg.y * BN;
    if (p0 >= n) return;
    threadgroup float tab[16];
    // k loop: Gt [BM2][BK], Ut [BM2][BK], Xt [BN][BK] (E); epilogue: Cg, Cu [BM2][BN] (f32) over Gt/Ut
    threadgroup uchar smem[2 * BM2 * BN * sizeof(float) + BN * BK * sizeof(float)];
    threadgroup E *Gt = (threadgroup E *)smem;
    threadgroup E *Ut = Gt + BM2 * BK;
    threadgroup E *Xt = (threadgroup E *)(smem + 2 * BM2 * BN * sizeof(float));
    if (tid < 16) tab[tid] = IQ4_NL_VALUES[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint i0 = tg.x * BM2;
    uint nb = a.n_embd / QK;
    device const B *Ex = W + (ulong)e * 2 * a.n_ff * nb;
    simdgroup_float8x8 ag[4][2], au[4][2];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 2; ++j) { ag[i][j] = simdgroup_float8x8(0); au[i][j] = simdgroup_float8x8(0); }

    uint wr = tid / 2, wh = tid % 2;       // weight loads: 64 rows x 2 halves (2 parts of 8 each)
    uint xr = tid / 4, xp = tid % 4;       // activation loads: 32 tokens x 4 parts of 8
    uint fi = i0 + wr;
    uint pi = p0 + xr;
    uint tok = pi < n ? pairs[offset[e] + pi] / a.n_used : 0;
    for (uint b = 0; b < nb; ++b) {
        for (uint q = 0; q < 2; ++q) {
            uint part = wh * 2 + q;
            if (fi < a.n_ff) {
                deq_block(Ex + (ulong)fi * nb, b, Gt + wr * BK, part, tab);
                deq_block(Ex + (ulong)(a.n_ff + fi) * nb, b, Ut + wr * BK, part, tab);
            } else {
                for (uint j = 0; j < 8; ++j) { Gt[wr * BK + part * 8 + j] = E(0); Ut[wr * BK + part * 8 + j] = E(0); }
            }
        }
        for (uint j = 0; j < 8; ++j) Xt[xr * BK + xp * 8 + j] = E(pi < n ? x[(ulong)tok * a.n_embd + b * QK + xp * 8 + j] : 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_4x2(Gt, Xt, ag, sg);
        mma_4x2(Ut, Xt, au, sg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float *Cg = (threadgroup float *)smem;
    threadgroup float *Cu = Cg + BM2 * BN;
    store_4x2(ag, Cg, sg);
    store_4x2(au, Cu, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = tid; k < BM2 * BN; k += 128) {
        uint rr = k / BN, cc = k % BN;
        uint f = i0 + rr, p = p0 + cc;
        if (f >= a.n_ff || p >= n) continue;
        float g = Cg[k], u = Cu[k];
        act[(ulong)pairs[offset[e] + p] * a.n_ff + f] = softcapf(g, a.situ_beta) * sigmoidf(g) * softcapf(u, a.situ_lin);
    }
}

// Grid: (ceil(n_embd/BM2), ceil(max_pairs/BN), n_expert), 128 threads.
template <typename B, typename E>
kernel void moe_gemm2_down(device const B *W [[buffer(0)]], device const float *act [[buffer(1)]], device const uint *count [[buffer(2)]],
                           device const uint *offset [[buffer(3)]], device const uint *pairs [[buffer(4)]], device float *yp [[buffer(5)]],
                           constant MoeGemmArgs &a [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint e = tg.z;
    uint n = count[e];
    uint p0 = tg.y * BN;
    if (p0 >= n) return;
    threadgroup float tab[16];
    // k loop: Wt [BM2][BK], Xt [BN][BK] (E); epilogue: Ct [BM2][BN] (f32) over Wt
    threadgroup uchar smem[BM2 * BN * sizeof(float) + BN * BK * sizeof(float)];
    threadgroup E *Wt = (threadgroup E *)smem;
    threadgroup E *Xt = (threadgroup E *)(smem + BM2 * BN * sizeof(float));
    if (tid < 16) tab[tid] = IQ4_NL_VALUES[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r0 = tg.x * BM2;
    uint nb = a.n_ff / QK;
    device const B *Ex = W + (ulong)e * a.n_embd * nb;
    simdgroup_float8x8 acc[4][2];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0);
    uint wr = tid / 2, wh = tid % 2;
    uint xr = tid / 4, xp = tid % 4;
    uint row = r0 + wr;
    uint pi = p0 + xr;
    uint pair = pi < n ? pairs[offset[e] + pi] : 0;
    for (uint b = 0; b < nb; ++b) {
        for (uint q = 0; q < 2; ++q) {
            uint part = wh * 2 + q;
            if (row < a.n_embd) deq_block(Ex + (ulong)row * nb, b, Wt + wr * BK, part, tab);
            else for (uint j = 0; j < 8; ++j) Wt[wr * BK + part * 8 + j] = E(0);
        }
        for (uint j = 0; j < 8; ++j) Xt[xr * BK + xp * 8 + j] = E(pi < n ? act[(ulong)pair * a.n_ff + b * QK + xp * 8 + j] : 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_4x2(Wt, Xt, acc, sg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float *Ct = (threadgroup float *)smem;
    store_4x2(acc, Ct, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = tid; k < BM2 * BN; k += 128) {
        uint rr = k / BN, cc = k % BN;
        uint r = r0 + rr, p = p0 + cc;
        if (r >= a.n_embd || p >= n) continue;
        yp[(ulong)pairs[offset[e] + p] * a.n_embd + r] = Ct[k];
    }
}

#define MOE_GEMM2_INSTANCES(T, E, name)                                                                                                  \
    template [[host_name("moe_gemm2_gate_up_" name)]] kernel void moe_gemm2_gate_up<T, E>(                                               \
        device const T *, device const float *, device const uint *, device const uint *, device const uint *, device float *,           \
        constant MoeGemmArgs &, uint3, uint, uint);                                                                                       \
    template [[host_name("moe_gemm2_down_" name)]] kernel void moe_gemm2_down<T, E>(                                                     \
        device const T *, device const float *, device const uint *, device const uint *, device const uint *, device float *,           \
        constant MoeGemmArgs &, uint3, uint, uint);
MOE_GEMM2_INSTANCES(block_iq4_nl, half, "iq4_nl_f16")
MOE_GEMM2_INSTANCES(block_q8_0, half, "q8_0_f16")
MOE_GEMM2_INSTANCES(block_iq4_nl, float, "iq4_nl_f32")
MOE_GEMM2_INSTANCES(block_q8_0, float, "q8_0_f32")

// ---- several Q8_0 matrices sharing one input, 64-row tiles ---------------------------------------
// Same contract as gemm_q8_0_multi, with the 4x2-accumulator layout of the expert GEMMs.
// Grid: (ceil(total_rows/BM2), ceil(n_tokens/BN)), 128 threads.
template <typename E>
kernel void gemm2_q8_0_multi(device const uchar *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *o0 [[buffer(2)]],
                             device float *o1 [[buffer(3)]], device float *o2 [[buffer(4)]], device float *o3 [[buffer(5)]],
                             device float *o4 [[buffer(6)]], device float *o5 [[buffer(7)]], device float *o6 [[buffer(8)]],
                             device float *o7 [[buffer(9)]], constant GMultiArgs &a [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uchar smem[BM2 * BN * sizeof(float) + BN * BK * sizeof(float)];
    threadgroup E *Wt = (threadgroup E *)smem;
    threadgroup E *Xt = (threadgroup E *)(smem + BM2 * BN * sizeof(float));
    uint r0 = tg.x * BM2, t0 = tg.y * BN;
    uint nb = a.cols / QK;
    simdgroup_float8x8 acc[4][2];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0);
    uint wr = tid / 2, wh = tid % 2;
    uint xr = tid / 4, xp = tid % 4;
    uint gr = r0 + wr;
    device const block_q8_0 *wrow = nullptr;
    if (gr < a.total_rows) {
        uint s = 0;
        while (s + 1 < a.n_seg && gr >= a.seg[s + 1].start) ++s;
        wrow = (device const block_q8_0 *)(W + a.seg[s].woff) + (ulong)(gr - a.seg[s].start) * nb;
    }
    uint tt = t0 + xr;
    for (uint b = 0; b < nb; ++b) {
        for (uint q = 0; q < 2; ++q) {
            uint part = wh * 2 + q;
            if (wrow) deq_block(wrow, b, Wt + wr * BK, part, (threadgroup const float *)nullptr);
            else for (uint j = 0; j < 8; ++j) Wt[wr * BK + part * 8 + j] = E(0);
        }
        for (uint j = 0; j < 8; ++j) Xt[xr * BK + xp * 8 + j] = E(tt < a.n_tokens ? x[(ulong)tt * a.cols + b * QK + xp * 8 + j] : 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_4x2(Wt, Xt, acc, sg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float *Ct = (threadgroup float *)smem;
    store_4x2(acc, Ct, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *outs[MULTI_MAX] = {o0, o1, o2, o3, o4, o5, o6, o7};
    for (uint i = tid; i < BM2 * BN; i += 128) {
        uint rr = i / BN, tk = i % BN;
        uint row = r0 + rr, tok = t0 + tk;
        if (row >= a.total_rows || tok >= a.n_tokens) continue;
        uint s = 0;
        while (s + 1 < a.n_seg && row >= a.seg[s + 1].start) ++s;
        outs[s][(ulong)tok * a.seg[s].rows + row - a.seg[s].start] = Ct[i];
    }
}

#define MULTI2_INSTANCE(E, name)                                                                                                        \
    template [[host_name("gemm2_q8_0_multi_" name)]] kernel void gemm2_q8_0_multi<E>(                                                    \
        device const uchar *, device const float *, device float *, device float *, device float *, device float *, device float *,      \
        device float *, device float *, device float *, constant GMultiArgs &, uint2, uint, uint);
MULTI2_INSTANCE(half, "f16")
MULTI2_INSTANCE(float, "f32")
