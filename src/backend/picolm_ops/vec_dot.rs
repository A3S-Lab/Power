//! Fused dequant + dot-product kernels.
//!
//! Computes `dot(dequant(quantized_row), x)` without materializing the full
//! dequantized row. This eliminates the f32 intermediate buffer and halves
//! memory traffic compared to separate dequant-then-dot.
//!
//! Each format has a dedicated kernel. Unsupported formats fall back to
//! dequant-then-dot via the generic path.

use half::f16;

use super::dequant::{block_bytes, block_size, dequantize_block};

/// Fused dot product: `dot(dequant(row), x)` for a single row.
/// `row_raw` is the raw quantized bytes for one row of `n` elements.
#[inline]
pub fn vec_dot(row_raw: &[u8], x: &[f32], n: usize, ggml_type: u32) -> f32 {
    match ggml_type {
        0 => vec_dot_f32(row_raw, x, n),
        1 => vec_dot_f16(row_raw, x, n),
        8 => vec_dot_q8_0(row_raw, x, n),
        12 => vec_dot_q4_k(row_raw, x, n),
        14 => vec_dot_q6_k(row_raw, x, n),
        _ => vec_dot_generic(row_raw, x, n, ggml_type),
    }
}

// ── F32 (type 0) ─────────────────────────────────────────────────────────────

fn vec_dot_f32(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    // Process 4 elements at a time for better ILP
    while i + 3 < n {
        let w0 = f32::from_le_bytes([
            row_raw[i * 4],
            row_raw[i * 4 + 1],
            row_raw[i * 4 + 2],
            row_raw[i * 4 + 3],
        ]);
        let w1 = f32::from_le_bytes([
            row_raw[i * 4 + 4],
            row_raw[i * 4 + 5],
            row_raw[i * 4 + 6],
            row_raw[i * 4 + 7],
        ]);
        let w2 = f32::from_le_bytes([
            row_raw[i * 4 + 8],
            row_raw[i * 4 + 9],
            row_raw[i * 4 + 10],
            row_raw[i * 4 + 11],
        ]);
        let w3 = f32::from_le_bytes([
            row_raw[i * 4 + 12],
            row_raw[i * 4 + 13],
            row_raw[i * 4 + 14],
            row_raw[i * 4 + 15],
        ]);
        sum += w0 * x[i] + w1 * x[i + 1] + w2 * x[i + 2] + w3 * x[i + 3];
        i += 4;
    }
    while i < n {
        let w = f32::from_le_bytes([
            row_raw[i * 4],
            row_raw[i * 4 + 1],
            row_raw[i * 4 + 2],
            row_raw[i * 4 + 3],
        ]);
        sum += w * x[i];
        i += 1;
    }
    sum
}

// ── F16 (type 1) ─────────────────────────────────────────────────────────────

fn vec_dot_f16(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..n {
        let w = f16::from_le_bytes([row_raw[i * 2], row_raw[i * 2 + 1]]).to_f32();
        sum += w * x[i];
    }
    sum
}

// ── Q8_0 (type 8): 32 elements per block, 34 bytes ──────────────────────────
// Fused: accumulate i32 dot, multiply by scale at end of block.

fn vec_dot_q8_0(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
    let nb = n / 32;
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let blk = i * 34;
        let scale = f16::from_le_bytes([row_raw[blk], row_raw[blk + 1]]).to_f32();
        let xp = &x[i * 32..];

        let mut sum = 0.0f32;
        for j in 0..32 {
            sum += (row_raw[blk + 2 + j] as i8) as f32 * xp[j];
        }
        sumf += scale * sum;
    }
    sumf
}

// ── Q4_K (type 12): 256 elements per block, 144 bytes ───────────────────────
// Fused: accumulate sum_qx and sum_x per chunk, apply scale/min at end.

#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

fn vec_dot_q4_k(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
    let nb = n / 256;
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let blk = i * 144;
        let d = f16::from_le_bytes([row_raw[blk], row_raw[blk + 1]]).to_f32();
        let dmin = f16::from_le_bytes([row_raw[blk + 2], row_raw[blk + 3]]).to_f32();
        let scales_raw = &row_raw[blk + 4..blk + 16];
        let qs = &row_raw[blk + 16..blk + 144];
        let xp = &x[i * 256..];

        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut x_off = 0usize;

        for _ in 0..4 {
            let (sc1, mn1) = get_scale_min_k4(is, scales_raw);
            let (sc2, mn2) = get_scale_min_k4(is + 1, scales_raw);
            let d1 = d * sc1 as f32;
            let m1 = dmin * mn1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * mn2 as f32;

            // Accumulate: sum_qx = sum(nibble * x), sum_x = sum(x)
            let mut sum_qx1 = 0.0f32;
            let mut sum_x1 = 0.0f32;
            let mut sum_qx2 = 0.0f32;
            let mut sum_x2 = 0.0f32;

            for l in 0..32 {
                let xlo = xp[x_off + l];
                let xhi = xp[x_off + l + 32];
                sum_qx1 += (qs[q_off + l] & 0x0F) as f32 * xlo;
                sum_x1 += xlo;
                sum_qx2 += (qs[q_off + l] >> 4) as f32 * xhi;
                sum_x2 += xhi;
            }

            sumf += d1 * sum_qx1 - m1 * sum_x1 + d2 * sum_qx2 - m2 * sum_x2;

            q_off += 32;
            x_off += 64;
            is += 2;
        }
    }
    sumf
}

// ── Q6_K (type 14): 256 elements per block, 210 bytes ───────────────────────
// Fused: accumulate per-scale-group sums, multiply by scale*d at end.

fn vec_dot_q6_k(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
    let nb = n / 256;
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let blk = i * 210;
        let ql = &row_raw[blk..blk + 128];
        let qh = &row_raw[blk + 128..blk + 192];
        let sc = &row_raw[blk + 192..blk + 208];
        let d = f16::from_le_bytes([row_raw[blk + 208], row_raw[blk + 209]]).to_f32();
        let xp = &x[i * 256..];

        // 16 scale groups, accumulate sum per group
        let mut sums = [0.0f32; 16];

        for chunk in 0..2 {
            let is = chunk * 8;
            let ql_c = &ql[chunk * 64..];
            let qh_c = &qh[chunk * 32..];
            let xp_c = &xp[chunk * 128..];

            for l in 0..16 {
                let q1 = ((ql_c[l] & 0xF) | (((qh_c[l] >> 0) & 3) << 4)) as i8 - 32;
                let q2 = ((ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql_c[l] >> 4) | (((qh_c[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql_c[l + 32] >> 4) | (((qh_c[l] >> 6) & 3) << 4)) as i8 - 32;
                sums[is] += q1 as f32 * xp_c[l];
                sums[is + 2] += q2 as f32 * xp_c[l + 32];
                sums[is + 4] += q3 as f32 * xp_c[l + 64];
                sums[is + 6] += q4 as f32 * xp_c[l + 96];
            }
            for l in 16..32 {
                let q1 = ((ql_c[l] & 0xF) | (((qh_c[l] >> 0) & 3) << 4)) as i8 - 32;
                let q2 = ((ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql_c[l] >> 4) | (((qh_c[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql_c[l + 32] >> 4) | (((qh_c[l] >> 6) & 3) << 4)) as i8 - 32;
                sums[is + 1] += q1 as f32 * xp_c[l];
                sums[is + 3] += q2 as f32 * xp_c[l + 32];
                sums[is + 5] += q3 as f32 * xp_c[l + 64];
                sums[is + 7] += q4 as f32 * xp_c[l + 96];
            }
        }

        for j in 0..16 {
            sumf += d * (sc[j] as i8) as f32 * sums[j];
        }
    }
    sumf
}

// ── Generic fallback: dequant-then-dot ───────────────────────────────────────

fn vec_dot_generic(row_raw: &[u8], x: &[f32], n: usize, ggml_type: u32) -> f32 {
    let bs = block_size(ggml_type);
    let bb = block_bytes(ggml_type);
    let blocks_per_row = if bs == 1 { n } else { n / bs };
    let mut buf = [0.0f32; 256];
    let mut sum = 0.0f32;

    for blk in 0..blocks_per_row {
        let blk_data = &row_raw[blk * bb..(blk + 1) * bb];
        let col_offset = blk * bs;
        dequantize_block(blk_data, ggml_type, &mut buf[..bs]);
        for j in 0..bs {
            sum += buf[j] * x[col_offset + j];
        }
    }
    sum
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_dot_f32() {
        let mut row = Vec::new();
        for v in [1.0f32, 2.0, 3.0] {
            row.extend_from_slice(&v.to_le_bytes());
        }
        let x = [4.0f32, 5.0, 6.0];
        let result = vec_dot(&row, &x, 3, 0);
        assert!((result - 32.0).abs() < 1e-5); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_vec_dot_q8_0() {
        // scale=2.0, quants=[1,1,...,1] (32 elements)
        // dot with x=[1,1,...,1] = 2.0 * 32 = 64.0
        let scale = f16::from_f32(2.0);
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&scale.to_le_bytes());
        for j in 0..32 {
            block[2 + j] = 1u8;
        }
        let x = [1.0f32; 32];
        let result = vec_dot(&block, &x, 32, 8);
        assert!((result - 64.0).abs() < 0.1);
    }

    #[test]
    fn test_vec_dot_q4_k_matches_dequant() {
        // Create a Q4_K block with known values, verify fused matches dequant-then-dot
        let mut block = [0u8; 144];
        let d = f16::from_f32(1.0);
        block[0..2].copy_from_slice(&d.to_le_bytes());
        // dmin = 0
        // scales[0]=2, scales[1]=3
        block[4] = 2;
        block[5] = 3;
        // qs: fill with 0x31 (lo=1, hi=3)
        for i in 16..144 {
            block[i] = 0x31;
        }

        let x = [1.0f32; 256];

        // Fused
        let fused = vec_dot(&block, &x, 256, 12);

        // Dequant-then-dot
        let mut buf = [0.0f32; 256];
        super::super::dequant::dequantize_block(&block, 12, &mut buf);
        let reference: f32 = buf.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (fused - reference).abs() < 0.01,
            "fused={fused}, reference={reference}"
        );
    }

    #[test]
    fn test_vec_dot_q6_k_matches_dequant() {
        let mut block = [0u8; 210];
        let d = f16::from_f32(0.5);
        block[208..210].copy_from_slice(&d.to_le_bytes());
        // scales: all 2
        for i in 192..208 {
            block[i] = 2u8;
        }
        // ql: fill with 0x21
        for i in 0..128 {
            block[i] = 0x21;
        }

        let x = [1.0f32; 256];

        let fused = vec_dot(&block, &x, 256, 14);

        let mut buf = [0.0f32; 256];
        super::super::dequant::dequantize_block(&block, 14, &mut buf);
        let reference: f32 = buf.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (fused - reference).abs() < 0.01,
            "fused={fused}, reference={reference}"
        );
    }
}
