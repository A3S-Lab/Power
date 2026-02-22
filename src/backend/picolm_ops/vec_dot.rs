//! Fused dequant + dot-product kernels.
//!
//! Computes `dot(dequant(quantized_row), x)` without materializing the full
//! dequantized row. This eliminates the f32 intermediate buffer and halves
//! memory traffic compared to separate dequant-then-dot.
//!
//! Each format has a dedicated kernel. Unsupported formats fall back to
//! dequant-then-dot via the generic path.
//!
//! # SIMD Acceleration
//!
//! On x86-64, AVX2 kernels are used when available (runtime detection via
//! `is_x86_feature_detected!`). This gives 4-8x speedup on the dot product
//! inner loop — the single biggest performance win for TEE inference on
//! AMD SEV-SNP / Intel TDX (both x86-64 only).

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
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::vec_dot_f32_avx2(row_raw, x, n) };
        }
    }
    vec_dot_f32_scalar(row_raw, x, n)
}

fn vec_dot_f32_scalar(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
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

fn vec_dot_q8_0(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::vec_dot_q8_0_avx2(row_raw, x, n) };
        }
    }
    vec_dot_q8_0_scalar(row_raw, x, n)
}

fn vec_dot_q8_0_scalar(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
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
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::vec_dot_q4_k_avx2(row_raw, x, n) };
        }
    }
    vec_dot_q4_k_scalar(row_raw, x, n)
}

fn vec_dot_q4_k_scalar(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
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

fn vec_dot_q6_k(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::vec_dot_q6_k_avx2(row_raw, x, n) };
        }
    }
    vec_dot_q6_k_scalar(row_raw, x, n)
}

fn vec_dot_q6_k_scalar(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
    let nb = n / 256;
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let blk = i * 210;
        let ql = &row_raw[blk..blk + 128];
        let qh = &row_raw[blk + 128..blk + 192];
        let sc = &row_raw[blk + 192..blk + 208];
        let d = f16::from_le_bytes([row_raw[blk + 208], row_raw[blk + 209]]).to_f32();
        let xp = &x[i * 256..];

        let mut sums = [0.0f32; 16];

        for chunk in 0..2 {
            let is = chunk * 8;
            let ql_c = &ql[chunk * 64..];
            let qh_c = &qh[chunk * 32..];
            let xp_c = &xp[chunk * 128..];

            for l in 0..16 {
                let q1 = ((ql_c[l] & 0xF) | ((qh_c[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql_c[l] >> 4) | (((qh_c[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql_c[l + 32] >> 4) | (((qh_c[l] >> 6) & 3) << 4)) as i8 - 32;
                sums[is] += q1 as f32 * xp_c[l];
                sums[is + 2] += q2 as f32 * xp_c[l + 32];
                sums[is + 4] += q3 as f32 * xp_c[l + 64];
                sums[is + 6] += q4 as f32 * xp_c[l + 96];
            }
            for l in 16..32 {
                let q1 = ((ql_c[l] & 0xF) | ((qh_c[l] & 3) << 4)) as i8 - 32;
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

// ── AVX2 SIMD kernels (x86-64 only) ─────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use half::f16;
    use std::arch::x86_64::*;

    /// AVX2 F32 dot product: loads 8 floats at a time via _mm256_loadu_ps.
    ///
    /// # Safety
    /// Caller must verify AVX2+FMA are available.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn vec_dot_f32_avx2(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;

        // Process 16 elements per iteration (2 × 8-wide AVX2)
        while i + 15 < n {
            let w0 = _mm256_loadu_ps(row_raw.as_ptr().add(i * 4) as *const f32);
            let x0 = _mm256_loadu_ps(x.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(w0, x0, acc0);

            let w1 = _mm256_loadu_ps(row_raw.as_ptr().add((i + 8) * 4) as *const f32);
            let x1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);

            i += 16;
        }

        while i + 7 < n {
            let w = _mm256_loadu_ps(row_raw.as_ptr().add(i * 4) as *const f32);
            let xv = _mm256_loadu_ps(x.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(w, xv, acc0);
            i += 8;
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        let mut sum = hsum_avx2(acc0);

        // Scalar tail
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

    /// AVX2 Q8_0 dot product: converts i8 quants to f32 in groups of 8, FMA with x.
    ///
    /// # Safety
    /// Caller must verify AVX2+FMA are available.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn vec_dot_q8_0_avx2(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
        let nb = n / 32;
        let mut total = _mm256_setzero_ps();

        for i in 0..nb {
            let blk = i * 34;
            let scale = f16::from_le_bytes([row_raw[blk], row_raw[blk + 1]]).to_f32();
            let quants = row_raw.as_ptr().add(blk + 2);
            let xp = x.as_ptr().add(i * 32);

            let mut block_acc = _mm256_setzero_ps();

            // 4 groups of 8 quants
            for g in 0..4 {
                let base = g * 8;
                // Sign-extend 8 × i8 → i32 → f32
                // Load 8 bytes, sign-extend via _mm256_cvtepi8_epi32
                let q8 = _mm_loadl_epi64(quants.add(base) as *const __m128i);
                let q32 = _mm256_cvtepi8_epi32(q8);
                let qf = _mm256_cvtepi32_ps(q32);
                let xv = _mm256_loadu_ps(xp.add(base));
                block_acc = _mm256_fmadd_ps(qf, xv, block_acc);
            }

            let scale_v = _mm256_set1_ps(scale);
            total = _mm256_fmadd_ps(scale_v, block_acc, total);
        }

        hsum_avx2(total)
    }

    /// Horizontal sum of 8 f32 lanes in a __m256.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hsum_avx2(v: __m256) -> f32 {
        let hi128 = _mm256_extractf128_ps(v, 1);
        let lo128 = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }

    /// AVX2 Q4_K dot product: dequant nibbles to f32 in groups of 8, FMA with x.
    ///
    /// Q4_K block layout (144 bytes, 256 elements):
    ///   [0..2]   d (f16 scale)
    ///   [2..4]   dmin (f16 min)
    ///   [4..16]  scales (12 bytes, packed 6-bit scale + 6-bit min)
    ///   [16..144] qs (128 bytes, 256 nibbles packed as lo/hi 4-bit)
    ///
    /// Each block has 4 sub-blocks of 64 elements (32 lo-nibble + 32 hi-nibble).
    ///
    /// # Safety
    /// Caller must verify AVX2+FMA are available.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn vec_dot_q4_k_avx2(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
        let nb = n / 256;
        let mask_lo = _mm256_set1_epi32(0x0F);
        let mut total = _mm256_setzero_ps();

        for i in 0..nb {
            let blk = i * 144;
            let d = f16::from_le_bytes([row_raw[blk], row_raw[blk + 1]]).to_f32();
            let dmin = f16::from_le_bytes([row_raw[blk + 2], row_raw[blk + 3]]).to_f32();
            let scales_raw = &row_raw[blk + 4..blk + 16];
            let qs = row_raw.as_ptr().add(blk + 16);
            let xp = x.as_ptr().add(i * 256);

            let mut q_off = 0usize;
            let mut x_off = 0usize;
            let mut is = 0usize;

            for _ in 0..4 {
                let (sc1, mn1) = super::get_scale_min_k4(is, scales_raw);
                let (sc2, mn2) = super::get_scale_min_k4(is + 1, scales_raw);
                let d1 = d * sc1 as f32;
                let m1 = dmin * mn1 as f32;
                let d2 = d * sc2 as f32;
                let m2 = dmin * mn2 as f32;

                let d1_v = _mm256_set1_ps(d1);
                let m1_v = _mm256_set1_ps(m1);
                let d2_v = _mm256_set1_ps(d2);
                let m2_v = _mm256_set1_ps(m2);

                // Process 32 lo-nibble elements (sub-block 1) in 4 groups of 8
                let mut sum_qx1 = _mm256_setzero_ps();
                let mut sum_x1 = _mm256_setzero_ps();
                let mut sum_qx2 = _mm256_setzero_ps();
                let mut sum_x2 = _mm256_setzero_ps();

                for g in 0..4 {
                    let base = g * 8;
                    // Load 8 bytes of qs, extract lo nibbles
                    let q8 = _mm_loadl_epi64(qs.add(q_off + base) as *const __m128i);
                    let q32 = _mm256_cvtepu8_epi32(q8);
                    let lo = _mm256_and_si256(q32, mask_lo);
                    let hi = _mm256_srli_epi32(q32, 4);

                    let lo_f = _mm256_cvtepi32_ps(lo);
                    let hi_f = _mm256_cvtepi32_ps(hi);

                    let xlo = _mm256_loadu_ps(xp.add(x_off + base));
                    let xhi = _mm256_loadu_ps(xp.add(x_off + base + 32));

                    sum_qx1 = _mm256_fmadd_ps(lo_f, xlo, sum_qx1);
                    sum_x1 = _mm256_add_ps(sum_x1, xlo);
                    sum_qx2 = _mm256_fmadd_ps(hi_f, xhi, sum_qx2);
                    sum_x2 = _mm256_add_ps(sum_x2, xhi);
                }

                // total += d1*sum_qx1 - m1*sum_x1 + d2*sum_qx2 - m2*sum_x2
                let t1 = _mm256_fmsub_ps(d1_v, sum_qx1, _mm256_mul_ps(m1_v, sum_x1));
                let t2 = _mm256_fmsub_ps(d2_v, sum_qx2, _mm256_mul_ps(m2_v, sum_x2));
                total = _mm256_add_ps(total, _mm256_add_ps(t1, t2));

                q_off += 32;
                x_off += 64;
                is += 2;
            }
        }

        hsum_avx2(total)
    }

    /// AVX2 Q6_K dot product: reconstruct 6-bit quants from ql+qh, convert to f32, FMA.
    ///
    /// Q6_K block layout (210 bytes, 256 elements):
    ///   [0..128]   ql (128 bytes, low 4 bits of each 6-bit quant)
    ///   [128..192]  qh (64 bytes, high 2 bits packed)
    ///   [192..208]  scales (16 × i8 scale factors)
    ///   [208..210]  d (f16 block scale)
    ///
    /// # Safety
    /// Caller must verify AVX2+FMA are available.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn vec_dot_q6_k_avx2(row_raw: &[u8], x: &[f32], n: usize) -> f32 {
        let nb = n / 256;
        let mut total = _mm256_setzero_ps();

        for i in 0..nb {
            let blk = i * 210;
            let ql = row_raw.as_ptr().add(blk);
            let qh = row_raw.as_ptr().add(blk + 128);
            let sc = &row_raw[blk + 192..blk + 208];
            let d = f16::from_le_bytes([row_raw[blk + 208], row_raw[blk + 209]]).to_f32();
            let xp = x.as_ptr().add(i * 256);

            let mut sums = [_mm256_setzero_ps(); 16];

            for chunk in 0..2 {
                let is = chunk * 8;
                let ql_c = ql.add(chunk * 64);
                let qh_c = qh.add(chunk * 32);
                let xp_c = xp.add(chunk * 128);

                // Process 16 elements at a time using scalar reconstruction
                // (Q6_K bit layout is complex — scalar dequant + AVX2 FMA)
                for l in 0..16 {
                    let q1 = ((*ql_c.add(l) & 0xF) | ((*qh_c.add(l) & 3) << 4)) as i8 - 32;
                    let q2 =
                        ((*ql_c.add(l + 32) & 0xF) | (((*qh_c.add(l) >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((*ql_c.add(l) >> 4) | (((*qh_c.add(l) >> 4) & 3) << 4)) as i8 - 32;
                    let q4 =
                        ((*ql_c.add(l + 32) >> 4) | (((*qh_c.add(l) >> 6) & 3) << 4)) as i8 - 32;

                    let x1 = *xp_c.add(l);
                    let x2 = *xp_c.add(l + 32);
                    let x3 = *xp_c.add(l + 64);
                    let x4 = *xp_c.add(l + 96);

                    // Accumulate into scalar sums array (indexed by scale group)
                    let s = &mut sums;
                    s[is] = _mm256_add_ps(s[is], _mm256_set1_ps(q1 as f32 * x1));
                    s[is + 2] = _mm256_add_ps(s[is + 2], _mm256_set1_ps(q2 as f32 * x2));
                    s[is + 4] = _mm256_add_ps(s[is + 4], _mm256_set1_ps(q3 as f32 * x3));
                    s[is + 6] = _mm256_add_ps(s[is + 6], _mm256_set1_ps(q4 as f32 * x4));
                }
                for l in 16..32 {
                    let q1 = ((*ql_c.add(l) & 0xF) | ((*qh_c.add(l) & 3) << 4)) as i8 - 32;
                    let q2 =
                        ((*ql_c.add(l + 32) & 0xF) | (((*qh_c.add(l) >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((*ql_c.add(l) >> 4) | (((*qh_c.add(l) >> 4) & 3) << 4)) as i8 - 32;
                    let q4 =
                        ((*ql_c.add(l + 32) >> 4) | (((*qh_c.add(l) >> 6) & 3) << 4)) as i8 - 32;

                    let x1 = *xp_c.add(l);
                    let x2 = *xp_c.add(l + 32);
                    let x3 = *xp_c.add(l + 64);
                    let x4 = *xp_c.add(l + 96);

                    let s = &mut sums;
                    s[is + 1] = _mm256_add_ps(s[is + 1], _mm256_set1_ps(q1 as f32 * x1));
                    s[is + 3] = _mm256_add_ps(s[is + 3], _mm256_set1_ps(q2 as f32 * x2));
                    s[is + 5] = _mm256_add_ps(s[is + 5], _mm256_set1_ps(q3 as f32 * x3));
                    s[is + 7] = _mm256_add_ps(s[is + 7], _mm256_set1_ps(q4 as f32 * x4));
                }
            }

            for j in 0..16 {
                let scale = d * (sc[j] as i8) as f32;
                total = _mm256_add_ps(total, _mm256_mul_ps(_mm256_set1_ps(scale), sums[j]));
            }
        }

        hsum_avx2(total)
    }
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
    fn test_vec_dot_f32_large() {
        // Test with 64 elements to exercise AVX2 path (16-element loop + 8-element loop)
        let n = 64;
        let mut row = Vec::new();
        let mut x = Vec::new();
        let mut expected = 0.0f32;
        for i in 0..n {
            let w = (i + 1) as f32 * 0.1;
            let xv = (n - i) as f32 * 0.1;
            row.extend_from_slice(&w.to_le_bytes());
            x.push(xv);
            expected += w * xv;
        }
        let result = vec_dot(&row, &x, n, 0);
        assert!(
            (result - expected).abs() < 0.1,
            "f32 large: got={result}, expected={expected}"
        );
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
    fn test_vec_dot_q8_0_multi_block() {
        // 2 blocks = 64 elements, verify AVX2 accumulation across blocks
        let scale = f16::from_f32(1.0);
        let mut blocks = vec![0u8; 68]; // 2 × 34
        for b in 0..2 {
            let off = b * 34;
            blocks[off..off + 2].copy_from_slice(&scale.to_le_bytes());
            for j in 0..32 {
                blocks[off + 2 + j] = 2u8; // quant = 2
            }
        }
        let x = [1.0f32; 64];
        let result = vec_dot(&blocks, &x, 64, 8);
        // 2 blocks × 32 elements × (scale=1.0 × quant=2 × x=1.0) = 128.0
        assert!(
            (result - 128.0).abs() < 0.5,
            "q8_0 multi: got={result}, expected=128.0"
        );
    }

    #[test]
    fn test_vec_dot_q4_k_matches_dequant() {
        let mut block = [0u8; 144];
        let d = f16::from_f32(1.0);
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[4] = 2;
        block[5] = 3;
        for i in 16..144 {
            block[i] = 0x31;
        }

        let x = [1.0f32; 256];

        let fused = vec_dot(&block, &x, 256, 12);

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
        for i in 192..208 {
            block[i] = 2u8;
        }
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_f32_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let n = 64;
        let mut row = Vec::new();
        let mut x = Vec::new();
        for i in 0..n {
            let w = (i as f32 + 1.0) * 0.37;
            row.extend_from_slice(&w.to_le_bytes());
            x.push((n as f32 - i as f32) * 0.13);
        }
        let scalar = vec_dot_f32_scalar(&row, &x, n);
        let simd = unsafe { avx2::vec_dot_f32_avx2(&row, &x, n) };
        assert!((scalar - simd).abs() < 0.01, "scalar={scalar}, simd={simd}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_q8_0_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // 4 blocks = 128 elements
        let nb = 4;
        let n = nb * 32;
        let mut blocks = vec![0u8; nb * 34];
        let mut x = vec![0.0f32; n];
        for b in 0..nb {
            let off = b * 34;
            let scale = f16::from_f32((b as f32 + 1.0) * 0.5);
            blocks[off..off + 2].copy_from_slice(&scale.to_le_bytes());
            for j in 0..32 {
                blocks[off + 2 + j] = ((j as i8) - 16) as u8;
                x[b * 32 + j] = (j as f32 + 1.0) * 0.1;
            }
        }
        let scalar = vec_dot_q8_0_scalar(&blocks, &x, n);
        let simd = unsafe { avx2::vec_dot_q8_0_avx2(&blocks, &x, n) };
        assert!((scalar - simd).abs() < 0.5, "scalar={scalar}, simd={simd}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_q4_k_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut block = [0u8; 144];
        let d = f16::from_f32(1.0);
        block[0..2].copy_from_slice(&d.to_le_bytes());
        let dmin = f16::from_f32(0.5);
        block[2..4].copy_from_slice(&dmin.to_le_bytes());
        block[4] = 2;
        block[5] = 3;
        block[6] = 1;
        block[7] = 4;
        for i in 16..144 {
            block[i] = 0x53; // lo=3, hi=5
        }
        let mut x = [0.0f32; 256];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.01;
        }
        let scalar = vec_dot_q4_k_scalar(&block, &x, 256);
        let simd = unsafe { avx2::vec_dot_q4_k_avx2(&block, &x, 256) };
        assert!(
            (scalar - simd).abs() < 0.5,
            "q4_k: scalar={scalar}, simd={simd}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_q6_k_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut block = [0u8; 210];
        let d = f16::from_f32(0.5);
        block[208..210].copy_from_slice(&d.to_le_bytes());
        for i in 192..208 {
            block[i] = 2u8;
        }
        for i in 0..128 {
            block[i] = 0x21;
        }
        let mut x = [0.0f32; 256];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.01;
        }
        let scalar = vec_dot_q6_k_scalar(&block, &x, 256);
        let simd = unsafe { avx2::vec_dot_q6_k_avx2(&block, &x, 256) };
        assert!(
            (scalar - simd).abs() < 0.5,
            "q6_k: scalar={scalar}, simd={simd}"
        );
    }
}
