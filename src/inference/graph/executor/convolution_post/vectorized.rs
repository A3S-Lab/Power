use super::{BatchNormActivation, ChannelPostOperation};

pub(super) fn apply(
    operation: ChannelPostOperation,
    values: &mut [f32],
    convolution_bias: Option<f32>,
) {
    let vectorized = if supported() && values.len() >= 8 {
        // SAFETY: runtime feature admission guarantees AVX2 support, the
        // helper processes complete eight-value blocks, and its return value
        // identifies the untouched scalar tail.
        unsafe { apply_avx2(operation, values, convolution_bias) }
    } else {
        0
    };
    for value in &mut values[vectorized..] {
        let biased = convolution_bias.map_or(*value, |bias| *value + bias);
        *value = operation.apply(biased);
    }
}

pub(super) fn add_bias_and_residual(
    values: &mut [f32],
    residual: &[f32],
    convolution_bias: Option<f32>,
) {
    debug_assert_eq!(values.len(), residual.len());
    let vectorized = if supported() && values.len() >= 8 {
        // SAFETY: runtime feature admission guarantees AVX2 support, both
        // slices have the same length, and the helper touches complete blocks.
        unsafe { add_bias_and_residual_avx2(values, residual, convolution_bias) }
    } else {
        0
    };
    for (value, residual) in values[vectorized..].iter_mut().zip(&residual[vectorized..]) {
        let biased = convolution_bias.map_or(*value, |bias| *value + bias);
        *value = biased + residual;
    }
}

fn supported() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn apply_avx2(
    operation: ChannelPostOperation,
    values: &mut [f32],
    convolution_bias: Option<f32>,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm256_add_ps, _mm256_div_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps,
        _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_div_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps,
        _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };

    #[cfg(test)]
    if let ChannelPostOperation::GeluErf {
        divisor,
        offset,
        scale,
    } = operation
    {
        return unsafe { apply_gelu_erf_avx2(values, convolution_bias, divisor, offset, scale) };
    }
    const LANES: usize = 8;
    let vectorized = values.len() / LANES * LANES;
    let zero = _mm256_set1_ps(0.0);
    let one = _mm256_set1_ps(1.0);
    let convolution_bias = convolution_bias.map(|bias| _mm256_set1_ps(bias));
    for offset in (0..vectorized).step_by(LANES) {
        let mut value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        if let Some(bias) = convolution_bias {
            value = _mm256_add_ps(value, bias);
        }
        value = match operation {
            ChannelPostOperation::Identity => value,
            #[cfg(test)]
            ChannelPostOperation::Relu => _mm256_max_ps(value, zero),
            // GELU is rejected before the vector loop and remains on the
            // scalar implementation. Keeping this arm total avoids a panic
            // if the admission code is changed independently later.
            #[cfg(test)]
            ChannelPostOperation::GeluErf { .. } => value,
            ChannelPostOperation::BatchNormalization {
                scale,
                bias,
                mean,
                stddev,
                activation,
            } => {
                let normalized = _mm256_add_ps(
                    _mm256_mul_ps(
                        _mm256_div_ps(
                            _mm256_sub_ps(value, _mm256_set1_ps(mean)),
                            _mm256_set1_ps(stddev),
                        ),
                        _mm256_set1_ps(scale),
                    ),
                    _mm256_set1_ps(bias),
                );
                match activation {
                    BatchNormActivation::Identity => normalized,
                    BatchNormActivation::Relu => _mm256_max_ps(normalized, zero),
                    BatchNormActivation::HardSwish { alpha, beta } => {
                        let gate = _mm256_add_ps(
                            _mm256_mul_ps(normalized, _mm256_set1_ps(alpha)),
                            _mm256_set1_ps(beta),
                        );
                        let gate = _mm256_min_ps(_mm256_max_ps(gate, zero), one);
                        _mm256_mul_ps(normalized, gate)
                    }
                }
            }
        };
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), value) };
    }
    vectorized
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_bias_and_residual_avx2(
    values: &mut [f32],
    residual: &[f32],
    convolution_bias: Option<f32>,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm256_add_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

    const LANES: usize = 8;
    let vectorized = values.len() / LANES * LANES;
    let convolution_bias = convolution_bias.map(|bias| _mm256_set1_ps(bias));
    for offset in (0..vectorized).step_by(LANES) {
        let mut value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        if let Some(bias) = convolution_bias {
            value = _mm256_add_ps(value, bias);
        }
        let residual = unsafe { _mm256_loadu_ps(residual.as_ptr().add(offset)) };
        value = _mm256_add_ps(value, residual);
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), value) };
    }
    vectorized
}

/// Applies the exact polynomial-only `erff` domains eight lanes at a time and
/// delegates the exponential and non-finite domains to the scalar implementation.
///
/// The polynomials and operation order match the `libm::erff` branches used by
/// Candle. This preserves output bits while avoiding scalar math-library calls
/// for tiny values, `2^-28 <= |x| < 1.25`, and finite `|x| >= 6`.
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), test))]
#[target_feature(enable = "avx2")]
unsafe fn apply_gelu_erf_avx2(
    values: &mut [f32],
    convolution_bias: Option<f32>,
    divisor: f32,
    offset: f32,
    scale: f32,
) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm256_add_ps, _mm256_and_ps, _mm256_div_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps, _mm256_xor_ps,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_and_ps, _mm256_div_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps, _mm256_xor_ps,
    };

    const LANES: usize = 8;
    const CENTRAL_MIN_BITS: u32 = 0x3180_0000;
    const CENTRAL_MAX_BITS: u32 = 0x3f58_0000;
    const MODERATE_MAX_BITS: u32 = 0x3fa0_0000;
    const FAR_MIN_BITS: u32 = 0x40c0_0000;
    const INFINITY_BITS: u32 = 0x7f80_0000;
    const EFX8: f32 = 1.027_033_3;
    const PP0: f32 = 1.283_791_7e-1;
    const PP1: f32 = -3.250_421e-1;
    const PP2: f32 = -2.848_175e-2;
    const PP3: f32 = -5.770_270_2e-3;
    const PP4: f32 = -2.376_301_7e-5;
    const QQ1: f32 = 3.979_172e-1;
    const QQ2: f32 = 6.502_225e-2;
    const QQ3: f32 = 5.081_306e-3;
    const QQ4: f32 = 1.324_947_4e-4;
    const QQ5: f32 = -3.960_228_2e-6;
    const ERX: f32 = 8.450_629e-1;
    const PA0: f32 = -2.362_118_6e-3;
    const PA1: f32 = 4.148_561e-1;
    const PA2: f32 = -3.722_078_8e-1;
    const PA3: f32 = 3.183_466_2e-1;
    const PA4: f32 = -1.108_947e-1;
    const PA5: f32 = 3.547_830_5e-2;
    const PA6: f32 = -2.166_375_5e-3;
    const QA1: f32 = 1.064_208_8e-1;
    const QA2: f32 = 5.403_979_4e-1;
    const QA3: f32 = 7.182_865_6e-2;
    const QA4: f32 = 1.261_712_2e-1;
    const QA5: f32 = 1.363_708_4e-2;
    const QA6: f32 = 1.198_45e-2;

    let vectorized = values.len() / LANES * LANES;
    let divisor_vector = _mm256_set1_ps(divisor);
    let offset_vector = _mm256_set1_ps(offset);
    let scale_vector = _mm256_set1_ps(scale);
    let convolution_bias_vector = convolution_bias.map(|bias| _mm256_set1_ps(bias));
    for base in (0..vectorized).step_by(LANES) {
        let mut value = unsafe { _mm256_loadu_ps(values.as_ptr().add(base)) };
        if let Some(bias) = convolution_bias_vector {
            value = _mm256_add_ps(value, bias);
        }
        let divided = _mm256_div_ps(value, divisor_vector);
        let squared = _mm256_mul_ps(divided, divided);

        let numerator = _mm256_add_ps(
            _mm256_set1_ps(PP0),
            _mm256_mul_ps(
                squared,
                _mm256_add_ps(
                    _mm256_set1_ps(PP1),
                    _mm256_mul_ps(
                        squared,
                        _mm256_add_ps(
                            _mm256_set1_ps(PP2),
                            _mm256_mul_ps(
                                squared,
                                _mm256_add_ps(
                                    _mm256_set1_ps(PP3),
                                    _mm256_mul_ps(squared, _mm256_set1_ps(PP4)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let denominator = _mm256_add_ps(
            _mm256_set1_ps(1.0),
            _mm256_mul_ps(
                squared,
                _mm256_add_ps(
                    _mm256_set1_ps(QQ1),
                    _mm256_mul_ps(
                        squared,
                        _mm256_add_ps(
                            _mm256_set1_ps(QQ2),
                            _mm256_mul_ps(
                                squared,
                                _mm256_add_ps(
                                    _mm256_set1_ps(QQ3),
                                    _mm256_mul_ps(
                                        squared,
                                        _mm256_add_ps(
                                            _mm256_set1_ps(QQ4),
                                            _mm256_mul_ps(squared, _mm256_set1_ps(QQ5)),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let activated = _mm256_add_ps(
            divided,
            _mm256_mul_ps(divided, _mm256_div_ps(numerator, denominator)),
        );
        let result = _mm256_mul_ps(
            _mm256_mul_ps(value, _mm256_add_ps(activated, offset_vector)),
            scale_vector,
        );
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(base), result) };

        let mut divided_lanes = [0.0_f32; LANES];
        let mut biased_lanes = [0.0_f32; LANES];
        unsafe {
            _mm256_storeu_ps(divided_lanes.as_mut_ptr(), divided);
            _mm256_storeu_ps(biased_lanes.as_mut_ptr(), value);
        }

        let has_moderate_lane = divided_lanes.iter().any(|lane| {
            let absolute_bits = lane.to_bits() & 0x7fff_ffff;
            (CENTRAL_MAX_BITS..MODERATE_MAX_BITS).contains(&absolute_bits)
        });
        let mut moderate_lanes = [0.0_f32; LANES];
        if has_moderate_lane {
            let absolute = _mm256_and_ps(divided, _mm256_set1_ps(f32::from_bits(0x7fff_ffff)));
            let sign = _mm256_and_ps(divided, _mm256_set1_ps(-0.0));
            let shifted = _mm256_sub_ps(absolute, _mm256_set1_ps(1.0));
            let numerator = _mm256_add_ps(
                _mm256_set1_ps(PA0),
                _mm256_mul_ps(
                    shifted,
                    _mm256_add_ps(
                        _mm256_set1_ps(PA1),
                        _mm256_mul_ps(
                            shifted,
                            _mm256_add_ps(
                                _mm256_set1_ps(PA2),
                                _mm256_mul_ps(
                                    shifted,
                                    _mm256_add_ps(
                                        _mm256_set1_ps(PA3),
                                        _mm256_mul_ps(
                                            shifted,
                                            _mm256_add_ps(
                                                _mm256_set1_ps(PA4),
                                                _mm256_mul_ps(
                                                    shifted,
                                                    _mm256_add_ps(
                                                        _mm256_set1_ps(PA5),
                                                        _mm256_mul_ps(shifted, _mm256_set1_ps(PA6)),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            );
            let denominator = _mm256_add_ps(
                _mm256_set1_ps(1.0),
                _mm256_mul_ps(
                    shifted,
                    _mm256_add_ps(
                        _mm256_set1_ps(QA1),
                        _mm256_mul_ps(
                            shifted,
                            _mm256_add_ps(
                                _mm256_set1_ps(QA2),
                                _mm256_mul_ps(
                                    shifted,
                                    _mm256_add_ps(
                                        _mm256_set1_ps(QA3),
                                        _mm256_mul_ps(
                                            shifted,
                                            _mm256_add_ps(
                                                _mm256_set1_ps(QA4),
                                                _mm256_mul_ps(
                                                    shifted,
                                                    _mm256_add_ps(
                                                        _mm256_set1_ps(QA5),
                                                        _mm256_mul_ps(shifted, _mm256_set1_ps(QA6)),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            );
            let complement = _mm256_sub_ps(
                _mm256_sub_ps(_mm256_set1_ps(1.0), _mm256_set1_ps(ERX)),
                _mm256_div_ps(numerator, denominator),
            );
            let activated = _mm256_xor_ps(_mm256_sub_ps(_mm256_set1_ps(1.0), complement), sign);
            unsafe { _mm256_storeu_ps(moderate_lanes.as_mut_ptr(), activated) };
        }
        for lane in 0..LANES {
            let absolute_bits = divided_lanes[lane].to_bits() & 0x7fff_ffff;
            if (CENTRAL_MIN_BITS..CENTRAL_MAX_BITS).contains(&absolute_bits) {
                continue;
            }
            let activated = if absolute_bits < CENTRAL_MIN_BITS {
                0.125 * ((8.0 * divided_lanes[lane]) + (EFX8 * divided_lanes[lane]))
            } else if absolute_bits < MODERATE_MAX_BITS {
                moderate_lanes[lane]
            } else if (FAR_MIN_BITS..INFINITY_BITS).contains(&absolute_bits) {
                let magnitude = 1.0 - f32::from_bits(0x0380_0000);
                if divided_lanes[lane].is_sign_negative() {
                    -magnitude
                } else {
                    magnitude
                }
            } else {
                candle_core::cpu::erf::erf_f32(divided_lanes[lane])
            };
            values[base + lane] = (biased_lanes[lane] * (activated + offset)) * scale;
        }
    }
    vectorized
}
