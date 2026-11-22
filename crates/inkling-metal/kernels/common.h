#pragma once
#include <metal_stdlib>
using namespace metal;

#define QK 32

struct block_q8_0 {
    half d;
    char qs[QK];
};

struct block_iq4_nl {
    half d;
    uchar qs[QK / 2];
};

constant float IQ4_NL_VALUES[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// Threadgroup-wide sum for up to 1024 threads (32 simdgroups). `scratch` holds 32 floats.
// All threads get the result.
inline float tg_sum(float v, threadgroup float *scratch, uint tid, uint ntg) {
    float s = simd_sum(v);
    uint sg = tid / 32, lane = tid % 32;
    uint nsg = (ntg + 31) / 32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) scratch[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = lane < nsg ? scratch[lane] : 0.0f;
    return simd_sum(t);
}

inline float tg_max(float v, threadgroup float *scratch, uint tid, uint ntg) {
    float s = simd_max(v);
    uint sg = tid / 32, lane = tid % 32;
    uint nsg = (ntg + 31) / 32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) scratch[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = lane < nsg ? scratch[lane] : -INFINITY;
    return simd_max(t);
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float softcapf(float x, float c) { return c * precise::tanh(x / c); }

// dot of one quantized row (n elements, n % 32 == 0) with x, split across a simdgroup:
// each lane takes whole blocks lane, lane+32, ...; the caller reduces with simd_sum.
inline float dot_q8_0_lane(device const block_q8_0 *row, device const float *x, uint nb, uint lane) {
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32) {
        device const char *q = row[b].qs;
        device const float4 *xv = (device const float4 *)(x + b * QK);
        float s = 0.0f;
        for (uint j = 0; j < QK / 4; ++j) {
            float4 qv = float4(q[4 * j], q[4 * j + 1], q[4 * j + 2], q[4 * j + 3]);
            s += dot(qv, xv[j]);
        }
        acc += s * float(row[b].d);
    }
    return acc;
}

inline float dot_iq4_nl_lane(device const block_iq4_nl *row, device const float *x, uint nb, uint lane) {
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32) {
        device const uchar *q = row[b].qs;
        device const float *xb = x + b * QK;
        float s = 0.0f;
        for (uint j = 0; j < QK / 2; ++j) {
            s += xb[j] * IQ4_NL_VALUES[q[j] & 0x0f] + xb[j + QK / 2] * IQ4_NL_VALUES[q[j] >> 4];
        }
        acc += s * float(row[b].d);
    }
    return acc;
}
