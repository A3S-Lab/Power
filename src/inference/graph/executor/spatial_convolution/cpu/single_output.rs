use candle_core::{CpuStorage, Result, Shape};
use rayon::prelude::*;

use super::SpatialConv2d;

pub(super) fn execute(
    convolution: &SpatialConv2d,
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
) -> Result<(CpuStorage, Shape)> {
    debug_assert_eq!(convolution.output_channels, 1);
    let output_spatial = convolution.output_height * convolution.output_width;
    let input_batch_elements =
        convolution.input_channels * convolution.input_height * convolution.input_width;
    let mut output = vec![0.0_f32; convolution.batch * output_spatial];
    let vectorized = vectorized_supported() && convolution.stride == 1;

    output
        .par_chunks_mut(output_spatial)
        .enumerate()
        .try_for_each(|(batch, output)| -> Result<()> {
            let input_base = batch * input_batch_elements;
            let interior_x = interior_output_x_range(convolution);
            for output_y in 0..convolution.output_height {
                let padded_y = output_y * convolution.stride;
                let interior_y = padded_y.checked_sub(convolution.pads.0).filter(|input_y| {
                    input_y
                        .checked_add((convolution.kernel_height - 1) * convolution.dilation)
                        .is_some_and(|last| last < convolution.input_height)
                });
                let output_row =
                    &mut output[output_y * convolution.output_width..][..convolution.output_width];
                let Some(input_y) = interior_y.filter(|_| !interior_x.is_empty()) else {
                    for (output_x, value) in output_row.iter_mut().enumerate() {
                        *value = accumulate_boundary(
                            convolution,
                            input,
                            kernel,
                            input_base,
                            padded_y,
                            output_x * convolution.stride,
                        );
                    }
                    continue;
                };

                for (output_x, value) in output_row[..interior_x.start].iter_mut().enumerate() {
                    *value = accumulate_boundary(
                        convolution,
                        input,
                        kernel,
                        input_base,
                        padded_y,
                        output_x * convolution.stride,
                    );
                }
                accumulate_interior_row(
                    convolution,
                    input,
                    kernel,
                    input_base,
                    input_y,
                    interior_x.clone(),
                    output_row,
                    vectorized,
                );
                for (offset, value) in output_row[interior_x.end..].iter_mut().enumerate() {
                    let output_x = interior_x.end + offset;
                    *value = accumulate_boundary(
                        convolution,
                        input,
                        kernel,
                        input_base,
                        padded_y,
                        output_x * convolution.stride,
                    );
                }
            }
            if bias.is_some() || !convolution.post_operation.is_identity() {
                convolution
                    .post_operation
                    .apply_channel(0, output, bias.map(|bias| bias[0]))?;
            }
            Ok(())
        })?;

    Ok((CpuStorage::F32(output), convolution.output_shape.clone()))
}

fn interior_output_x_range(convolution: &SpatialConv2d) -> std::ops::Range<usize> {
    let kernel_span = (convolution.kernel_width - 1) * convolution.dilation;
    if kernel_span >= convolution.input_width {
        return 0..0;
    }
    let start = convolution.pads.1.div_ceil(convolution.stride);
    let last_padded_x = convolution.pads.1 + convolution.input_width - kernel_span - 1;
    let end = (last_padded_x / convolution.stride + 1).min(convolution.output_width);
    start.min(end)..end
}

#[allow(clippy::too_many_arguments)]
fn accumulate_interior_row(
    convolution: &SpatialConv2d,
    input: &[f32],
    kernel: &[f32],
    input_base: usize,
    input_y: usize,
    output_x: std::ops::Range<usize>,
    output: &mut [f32],
    vectorized: bool,
) {
    let output_values = &mut output[output_x.clone()];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if vectorized && output_values.len() >= 8 {
        // SAFETY: the exact interior proof above keeps every eight-lane load
        // inside one contiguous NCHW row, and the helper owns its scalar tail.
        unsafe {
            accumulate_interior_row_avx2(
                convolution,
                input,
                kernel,
                input_base,
                input_y,
                output_x.start,
                output_values,
            );
        }
        return;
    }
    for (offset, value) in output_values.iter_mut().enumerate() {
        *value = accumulate_interior(
            convolution,
            input,
            kernel,
            input_base,
            input_y,
            output_x.start + offset,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_interior(
    convolution: &SpatialConv2d,
    input: &[f32],
    kernel: &[f32],
    input_base: usize,
    input_y: usize,
    output_x: usize,
) -> f32 {
    let input_spatial = convolution.input_height * convolution.input_width;
    let kernel_spatial = convolution.kernel_height * convolution.kernel_width;
    let input_x = output_x * convolution.stride - convolution.pads.1;
    let mut sum = 0.0_f32;
    for kernel_y in 0..convolution.kernel_height {
        for kernel_x in 0..convolution.kernel_width {
            for input_channel in 0..convolution.input_channels {
                let input_index = input_base
                    + input_channel * input_spatial
                    + (input_y + kernel_y * convolution.dilation) * convolution.input_width
                    + input_x
                    + kernel_x * convolution.dilation;
                let kernel_index =
                    input_channel * kernel_spatial + kernel_y * convolution.kernel_width + kernel_x;
                sum = input[input_index].mul_add(kernel[kernel_index], sum);
            }
        }
    }
    sum
}

#[allow(clippy::too_many_arguments)]
fn accumulate_boundary(
    convolution: &SpatialConv2d,
    input: &[f32],
    kernel: &[f32],
    input_base: usize,
    padded_y: usize,
    padded_x: usize,
) -> f32 {
    let input_spatial = convolution.input_height * convolution.input_width;
    let kernel_spatial = convolution.kernel_height * convolution.kernel_width;
    let mut sum = 0.0_f32;
    for kernel_y in 0..convolution.kernel_height {
        let input_y = (padded_y + kernel_y * convolution.dilation)
            .checked_sub(convolution.pads.0)
            .filter(|input_y| *input_y < convolution.input_height);
        for kernel_x in 0..convolution.kernel_width {
            let input_x = (padded_x + kernel_x * convolution.dilation)
                .checked_sub(convolution.pads.1)
                .filter(|input_x| *input_x < convolution.input_width);
            for input_channel in 0..convolution.input_channels {
                let value = input_y.zip(input_x).map_or(0.0_f32, |(input_y, input_x)| {
                    input[input_base
                        + input_channel * input_spatial
                        + input_y * convolution.input_width
                        + input_x]
                });
                let kernel_index =
                    input_channel * kernel_spatial + kernel_y * convolution.kernel_width + kernel_x;
                sum = value.mul_add(kernel[kernel_index], sum);
            }
        }
    }
    sum
}

fn vectorized_supported() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn accumulate_interior_row_avx2(
    convolution: &SpatialConv2d,
    input: &[f32],
    kernel: &[f32],
    input_base: usize,
    input_y: usize,
    output_x_start: usize,
    output: &mut [f32],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    const LANES: usize = 8;
    let input_spatial = convolution.input_height * convolution.input_width;
    let kernel_spatial = convolution.kernel_height * convolution.kernel_width;
    let input_x = output_x_start - convolution.pads.1;
    let vectorized = output.len() / LANES * LANES;
    for offset in (0..vectorized).step_by(LANES) {
        let mut sum = _mm256_setzero_ps();
        for kernel_y in 0..convolution.kernel_height {
            for kernel_x in 0..convolution.kernel_width {
                for input_channel in 0..convolution.input_channels {
                    let input_index = input_base
                        + input_channel * input_spatial
                        + (input_y + kernel_y * convolution.dilation) * convolution.input_width
                        + input_x
                        + kernel_x * convolution.dilation
                        + offset;
                    let kernel_index = input_channel * kernel_spatial
                        + kernel_y * convolution.kernel_width
                        + kernel_x;
                    let values = unsafe { _mm256_loadu_ps(input.as_ptr().add(input_index)) };
                    let weight = _mm256_set1_ps(kernel[kernel_index]);
                    sum = _mm256_fmadd_ps(values, weight, sum);
                }
            }
        }
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(offset), sum) };
    }
    for (offset, value) in output[vectorized..].iter_mut().enumerate() {
        *value = accumulate_interior(
            convolution,
            input,
            kernel,
            input_base,
            input_y,
            output_x_start + vectorized + offset,
        );
    }
}
