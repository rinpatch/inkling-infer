//! Reference dequantization and dot products for the storage types in Inkling GGUFs.
//! Layouts follow ggml-common.h: Q8_0 = { f16 d; i8 qs[32] }, IQ4_NL = { f16 d; u8 qs[16] } where
//! byte j holds element j in its low nibble and element j+16 in its high nibble.

use half::{bf16, f16};

use crate::DType;

pub const QK: usize = 32;
pub const BLOCK_Q8_0_BYTES: usize = 2 + QK;
pub const BLOCK_IQ4_NL_BYTES: usize = 2 + QK / 2;

/// Non-linear 4-bit codebook (ggml `kvalues_iq4nl`).
pub const IQ4_NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[inline]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// Dequantizes `out.len()` elements from the start of `src`.
pub fn dequantize_row(dtype: DType, src: &[u8], out: &mut [f32]) {
    match dtype {
        DType::F32 => {
            for (o, c) in out.iter_mut().zip(src.as_chunks::<4>().0.iter()) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        DType::F16 => {
            for (o, c) in out.iter_mut().zip(src.as_chunks::<2>().0.iter()) {
                *o = f16::from_le_bytes([c[0], c[1]]).to_f32();
            }
        }
        DType::Bf16 => {
            for (o, c) in out.iter_mut().zip(src.as_chunks::<2>().0.iter()) {
                *o = bf16::from_le_bytes([c[0], c[1]]).to_f32();
            }
        }
        DType::Q8_0 => {
            for (o, b) in out
                .as_chunks_mut::<QK>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<BLOCK_Q8_0_BYTES>().0.iter())
            {
                let d = f16_at(b, 0);
                for (x, &q) in o.iter_mut().zip(&b[2..]) {
                    *x = d * (q as i8) as f32;
                }
            }
        }
        DType::Iq4Nl => {
            for (o, b) in out
                .as_chunks_mut::<QK>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<BLOCK_IQ4_NL_BYTES>().0.iter())
            {
                let d = f16_at(b, 0);
                for j in 0..QK / 2 {
                    let q = b[2 + j];
                    o[j] = d * IQ4_NL_VALUES[(q & 0x0f) as usize] as f32;
                    o[j + QK / 2] = d * IQ4_NL_VALUES[(q >> 4) as usize] as f32;
                }
            }
        }
    }
}

/// Dot product of one stored row (`x.len()` elements starting at `src`) with an f32 vector.
/// Block scales are applied per block so quantized rows never get materialized.
pub fn dot_row(dtype: DType, src: &[u8], x: &[f32]) -> f32 {
    match dtype {
        DType::Q8_0 => {
            let mut acc = 0.0f32;
            for (xb, b) in x
                .as_chunks::<QK>()
                .0
                .iter()
                .zip(src.as_chunks::<BLOCK_Q8_0_BYTES>().0.iter())
            {
                let mut s = 0.0f32;
                for (&xi, &q) in xb.iter().zip(&b[2..]) {
                    s += xi * (q as i8) as f32;
                }
                acc += s * f16_at(b, 0);
            }
            acc
        }
        DType::Iq4Nl => {
            let mut acc = 0.0f32;
            for (xb, b) in x
                .as_chunks::<QK>()
                .0
                .iter()
                .zip(src.as_chunks::<BLOCK_IQ4_NL_BYTES>().0.iter())
            {
                let mut s = 0.0f32;
                for j in 0..QK / 2 {
                    let q = b[2 + j];
                    s += xb[j] * IQ4_NL_VALUES[(q & 0x0f) as usize] as f32;
                    s += xb[j + QK / 2] * IQ4_NL_VALUES[(q >> 4) as usize] as f32;
                }
                acc += s * f16_at(b, 0);
            }
            acc
        }
        DType::F32 => {
            let mut acc = 0.0f32;
            for (&xi, c) in x.iter().zip(src.as_chunks::<4>().0.iter()) {
                acc += xi * f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
            acc
        }
        DType::F16 | DType::Bf16 => {
            let mut row = vec![0.0f32; x.len()];
            dequantize_row(dtype, src, &mut row);
            row.iter().zip(x).map(|(a, b)| a * b).sum()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block_q8(d: f32, qs: [i8; QK]) -> Vec<u8> {
        let mut b = f16::from_f32(d).to_le_bytes().to_vec();
        b.extend(qs.iter().map(|&q| q as u8));
        b
    }

    #[test]
    fn q8_0_dequantizes_scale_times_int() {
        let mut qs = [0i8; QK];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = i as i8 - 16;
        }
        let b = block_q8(0.5, qs);
        let mut out = [0.0f32; QK];
        dequantize_row(DType::Q8_0, &b, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.5 * (i as f32 - 16.0));
        }
    }

    #[test]
    fn iq4_nl_low_nibble_is_first_half_high_nibble_second_half() {
        let mut b = f16::from_f32(2.0).to_le_bytes().to_vec();
        // byte j: low nibble j % 16, high nibble 15 - j % 16
        b.extend((0..16u8).map(|j| j | ((15 - j) << 4)));
        let mut out = [0.0f32; QK];
        dequantize_row(DType::Iq4Nl, &b, &mut out);
        for j in 0..16 {
            assert_eq!(out[j], 2.0 * IQ4_NL_VALUES[j] as f32);
            assert_eq!(out[j + 16], 2.0 * IQ4_NL_VALUES[15 - j] as f32);
        }
    }

    #[test]
    fn dot_row_matches_dequantize_then_dot() {
        let mut src = Vec::new();
        for k in 0..4u8 {
            src.extend(f16::from_f32(0.25 + k as f32).to_le_bytes());
            src.extend((0..16u8).map(|j| j.wrapping_mul(7).wrapping_add(k)));
        }
        let x: Vec<f32> = (0..4 * QK).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut row = vec![0.0f32; x.len()];
        dequantize_row(DType::Iq4Nl, &src, &mut row);
        let want: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
        let got = dot_row(DType::Iq4Nl, &src, &x);
        assert!(
            (want - got).abs() < 1e-3 * want.abs().max(1.0),
            "{want} vs {got}"
        );
    }
}

/// Quantizes `x` (length % 32 == 0) to Q8_0 blocks, as ggml's reference does: per block
/// d = max|x| / 127 stored as f16, q = round(x / d).
pub fn quantize_row_q8_0(x: &[f32], out: &mut Vec<u8>) {
    assert!(x.len().is_multiple_of(QK));
    for blk in x.chunks_exact(QK) {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        out.extend(blk.iter().map(|v| ((v * id).round() as i8) as u8));
    }
}

#[cfg(test)]
mod q8_tests {
    use super::*;

    #[test]
    fn q8_0_round_trip() {
        let x: Vec<f32> = (0..256)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
            .collect();
        let mut q = Vec::new();
        quantize_row_q8_0(&x, &mut q);
        assert_eq!(q.len(), 256 / QK * BLOCK_Q8_0_BYTES);
        let mut y = vec![0.0; 256];
        dequantize_row(crate::DType::Q8_0, &q, &mut y);
        let max_err = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 0.66 / 127.0 * 0.7, "max err {max_err}");
    }
}
