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
// 64-row tiles: each simdgroup owns a 32-row x 16-token output (4x2 accumulators), so a k step
// does 6 threadgroup-matrix loads per 8 multiplies (a 16x16 layout does 4 per 4). Threadgroup
// memory is one raw array: the weight/activation tiles during the k loop, then the f32 output
// tiles in the epilogue (32 KB limit).

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

// ---- several Q8_0 matrices sharing one input, 64-row tiles ---------------------------------------
// out_s[t][r] = W_s[r] . x[t] for several Q8_0 matrices sharing one input (segment concatenation
// as in mv_q8_0_multi), with the 4x2-accumulator layout of the expert GEMMs.
// Grid: (ceil(total_rows/BM2), ceil(n_tokens/BN)), 128 threads.
template <typename E>
kernel void gemm2_q8_0_multi(device const uchar *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *o0 [[buffer(2)]],
                             device float *o1 [[buffer(3)]], device float *o2 [[buffer(4)]], device float *o3 [[buffer(5)]],
                             device float *o4 [[buffer(6)]], device float *o5 [[buffer(7)]], device float *o6 [[buffer(8)]],
                             device float *o7 [[buffer(9)]], constant GMultiArgs &a [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    // only the k loop's tiles in threadgroup memory (results go straight from the accumulators),
    // so several threadgroups fit per core
    threadgroup E Wt[BM2 * BK], Xt[BN * BK];
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
    device float *outs[MULTI_MAX] = {o0, o1, o2, o3, o4, o5, o6, o7};
    // lane holds row fm and columns fn, fn+1 of each 8x8 accumulator (Apple simdgroup layout)
    const uint fm = ((lane / 4) & 4) + (lane / 2) % 4, fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const uint rb = (sg / 2) * 32, tb = (sg % 2) * 16;
    for (uint i = 0; i < 4; ++i) {
        uint row = r0 + rb + 8 * i + fm;
        if (row >= a.total_rows) continue;
        uint s = 0;
        while (s + 1 < a.n_seg && row >= a.seg[s + 1].start) ++s;
        for (uint j = 0; j < 2; ++j)
            for (uint q = 0; q < 2; ++q) {
                uint tok = t0 + tb + 8 * j + fn + q;
                if (tok < a.n_tokens) outs[s][(ulong)tok * a.seg[s].rows + row - a.seg[s].start] = acc[i][j].thread_elements()[q];
            }
    }
}

#define MULTI2_INSTANCE(E, name)                                                                                                        \
    template [[host_name("gemm2_q8_0_multi_" name)]] kernel void gemm2_q8_0_multi<E>(                                                    \
        device const uchar *, device const float *, device float *, device float *, device float *, device float *, device float *,      \
        device float *, device float *, device float *, constant GMultiArgs &, uint2, uint, uint, uint);
MULTI2_INSTANCE(half, "f16")
MULTI2_INSTANCE(float, "f32")

// ---- MoE GEMMs over a tile list, 32- or 16-pair tiles ---------------------------------------------
// Routing is uneven (512-token batch: median 18 pairs per expert, max 180), so one-size 32-pair
// tiles run 63% full. moe_tiles lists full 32-pair tiles per expert plus one tile for the
// remainder: 32-wide if more than 16 pairs are left, 16-wide otherwise. Each kernel instance reads
// (expert, first pair) from its list; threadgroups past the list's length exit at once.

struct TileArgs {
    uint n_expert;
    uint max32;  // list capacities
    uint max16;
};

// One threadgroup of n_expert threads. counts[0] / counts[1] = tiles in list32 / list16.
kernel void moe_tiles(device const uint *count [[buffer(0)]], device uint2 *list32 [[buffer(1)]], device uint2 *list16 [[buffer(2)]],
                      device uint *counts [[buffer(3)]], constant TileArgs &a [[buffer(4)]], uint e [[thread_position_in_threadgroup]]) {
    threadgroup uint n32[1024], n16[1024];
    uint n = e < a.n_expert ? count[e] : 0;
    uint rem = n % 32;
    uint t32 = n / 32 + (rem > 16 ? 1 : 0), t16 = (rem > 0 && rem <= 16) ? 1 : 0;
    n32[e] = t32;
    n16[e] = t16;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e >= a.n_expert) return;
    uint o32 = 0, o16 = 0;
    for (uint i = 0; i < e; ++i) { o32 += n32[i]; o16 += n16[i]; }
    for (uint t = 0; t < t32; ++t) list32[o32 + t] = uint2(e, t * 32);
    if (t16) list16[o16] = uint2(e, n - rem);
    if (e == a.n_expert - 1) { counts[0] = o32 + t32; counts[1] = o16 + t16; }
}

// C[64 rows][TN pairs] += W x^T for one 32-column step; simdgroup sg owns rows (sg/2)*32.. and pairs
// (sg%2)*(TN/2).., i.e. a 4 x (TN/16) block of 8x8 accumulators.
template <typename E, uint TN>
inline void mma_4xj(threadgroup const E *Wt, threadgroup const E *Xt, thread simdgroup_float8x8 acc[4][TN / 16], uint sg) {
    typedef typename metal::conditional<metal::is_same<E, half>::value, simdgroup_half8x8, simdgroup_float8x8>::type Mat;
    uint rb = (sg / 2) * 32, tb = (sg % 2) * (TN / 2);
    for (uint kk = 0; kk < BK; kk += 8) {
        Mat a[4], b[TN / 16];
        for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], Wt + (rb + 8 * i) * BK + kk, BK);
        for (uint j = 0; j < TN / 16; ++j) simdgroup_load(b[j], Xt + (tb + 8 * j) * BK + kk, BK, ulong2(0, 0), true);
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < TN / 16; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
}

// Grid: (n_ff / 64, list capacity), 128 threads.
template <typename B, typename E, uint TN>
kernel void moe_tgemm_gate_up(device const B *W [[buffer(0)]], device const float *x [[buffer(1)]], device const uint *count [[buffer(2)]],
                              device const uint *offset [[buffer(3)]], device const uint *pairs [[buffer(4)]], device float *act [[buffer(5)]],
                              constant MoeGemmArgs &a [[buffer(6)]], device const uint2 *list [[buffer(7)]], device const uint *n_tiles [[buffer(8)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    if (tg.y >= n_tiles[0]) return;
    uint e = list[tg.y].x, p0 = list[tg.y].y;
    uint n = count[e];
    threadgroup float tab[16];
    // only the k loop's tiles live in threadgroup memory (results go straight from the
    // accumulators to act), so it stays small enough for several threadgroups per core
    threadgroup E Gt[BM2 * BK], Ut[BM2 * BK], Xt[TN * BK];
    if (tid < 16) tab[tid] = IQ4_NL_VALUES[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint i0 = tg.x * BM2;
    uint nb = a.n_embd / QK;
    device const B *Ex = W + (ulong)e * 2 * a.n_ff * nb;
    simdgroup_float8x8 ag[4][TN / 16], au[4][TN / 16];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < TN / 16; ++j) { ag[i][j] = simdgroup_float8x8(0); au[i][j] = simdgroup_float8x8(0); }
    uint wr = tid / 2, wh = tid % 2;            // weights: 64 rows x 2 halves (2 parts of 8 each)
    constexpr uint XPT = TN * BK / 128;         // input elements per thread
    uint xr = tid / (BK / XPT), xc = (tid % (BK / XPT)) * XPT;
    uint pi = p0 + xr;
    uint tok = pi < n ? pairs[offset[e] + pi] / a.n_used : 0;
    uint fi = i0 + wr;
    for (uint b = 0; b < nb; ++b) {
        for (uint q = 0; q < 2; ++q) {
            uint part = wh * 2 + q;
            deq_block(Ex + (ulong)fi * nb, b, Gt + wr * BK, part, tab);
            deq_block(Ex + (ulong)(a.n_ff + fi) * nb, b, Ut + wr * BK, part, tab);
        }
        for (uint j = 0; j < XPT; ++j) Xt[xr * BK + xc + j] = E(pi < n ? x[(ulong)tok * a.n_embd + b * QK + xc + j] : 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_4xj<E, TN>(Gt, Xt, ag, sg);
        mma_4xj<E, TN>(Ut, Xt, au, sg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // lane holds row fm and columns fn, fn+1 of each 8x8 accumulator (Apple simdgroup layout)
    const uint fm = ((lane / 4) & 4) + (lane / 2) % 4, fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const uint rb = (sg / 2) * 32, tb = (sg % 2) * (TN / 2);
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < TN / 16; ++j)
            for (uint q = 0; q < 2; ++q) {
                uint p = p0 + tb + 8 * j + fn + q;
                if (p >= n) continue;
                float g = ag[i][j].thread_elements()[q], u = au[i][j].thread_elements()[q];
                act[(ulong)pairs[offset[e] + p] * a.n_ff + i0 + rb + 8 * i + fm] =
                    softcapf(g, a.situ_beta) * sigmoidf(g) * softcapf(u, a.situ_lin);
            }
}

// Grid: (n_embd / 64, list capacity), 128 threads.
template <typename B, typename E, uint TN>
kernel void moe_tgemm_down(device const B *W [[buffer(0)]], device const float *act [[buffer(1)]], device const uint *count [[buffer(2)]],
                           device const uint *offset [[buffer(3)]], device const uint *pairs [[buffer(4)]], device float *yp [[buffer(5)]],
                           constant MoeGemmArgs &a [[buffer(6)]], device const uint2 *list [[buffer(7)]], device const uint *n_tiles [[buffer(8)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    if (tg.y >= n_tiles[0]) return;
    uint e = list[tg.y].x, p0 = list[tg.y].y;
    uint n = count[e];
    threadgroup float tab[16];
    threadgroup E Wt[BM2 * BK], Xt[TN * BK];
    if (tid < 16) tab[tid] = IQ4_NL_VALUES[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r0 = tg.x * BM2;
    uint nb = a.n_ff / QK;
    device const B *Ex = W + (ulong)e * a.n_embd * nb;
    simdgroup_float8x8 acc[4][TN / 16];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < TN / 16; ++j) acc[i][j] = simdgroup_float8x8(0);
    uint wr = tid / 2, wh = tid % 2;
    constexpr uint XPT = TN * BK / 128;
    uint xr = tid / (BK / XPT), xc = (tid % (BK / XPT)) * XPT;
    uint pi = p0 + xr;
    uint pair = pi < n ? pairs[offset[e] + pi] : 0;
    uint row = r0 + wr;
    for (uint b = 0; b < nb; ++b) {
        for (uint q = 0; q < 2; ++q) deq_block(Ex + (ulong)row * nb, b, Wt + wr * BK, wh * 2 + q, tab);
        for (uint j = 0; j < XPT; ++j) Xt[xr * BK + xc + j] = E(pi < n ? act[(ulong)pair * a.n_ff + b * QK + xc + j] : 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_4xj<E, TN>(Wt, Xt, acc, sg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint fm = ((lane / 4) & 4) + (lane / 2) % 4, fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const uint rb = (sg / 2) * 32, tb = (sg % 2) * (TN / 2);
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < TN / 16; ++j)
            for (uint q = 0; q < 2; ++q) {
                uint p = p0 + tb + 8 * j + fn + q;
                if (p < n) yp[(ulong)pairs[offset[e] + p] * a.n_embd + r0 + rb + 8 * i + fm] = acc[i][j].thread_elements()[q];
            }
}

#define MOE_TGEMM_INSTANCES(T, E, name, TN)                                                                                              \
    template [[host_name("moe_tgemm_gate_up_" name "_t" #TN)]] kernel void moe_tgemm_gate_up<T, E, TN>(                                \
        device const T *, device const float *, device const uint *, device const uint *, device const uint *, device float *,           \
        constant MoeGemmArgs &, device const uint2 *, device const uint *, uint2, uint, uint, uint);                                      \
    template [[host_name("moe_tgemm_down_" name "_t" #TN)]] kernel void moe_tgemm_down<T, E, TN>(                                      \
        device const T *, device const float *, device const uint *, device const uint *, device const uint *, device float *,           \
        constant MoeGemmArgs &, device const uint2 *, device const uint *, uint2, uint, uint, uint);
MOE_TGEMM_INSTANCES(block_iq4_nl, half, "iq4_nl_f16", 32)
MOE_TGEMM_INSTANCES(block_iq4_nl, half, "iq4_nl_f16", 16)
MOE_TGEMM_INSTANCES(block_q8_0, half, "q8_0_f16", 32)
MOE_TGEMM_INSTANCES(block_q8_0, half, "q8_0_f16", 16)
MOE_TGEMM_INSTANCES(block_iq4_nl, float, "iq4_nl_f32", 32)
MOE_TGEMM_INSTANCES(block_iq4_nl, float, "iq4_nl_f32", 16)
MOE_TGEMM_INSTANCES(block_q8_0, float, "q8_0_f32", 32)
MOE_TGEMM_INSTANCES(block_q8_0, float, "q8_0_f32", 16)
