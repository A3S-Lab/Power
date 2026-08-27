use super::DepthwiseConv2d;

pub(super) fn supported() -> bool {
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
pub(super) unsafe fn accumulate_interior_row(
    convolution: &DepthwiseConv2d,
    input: &[f32],
    kernel: &[f32],
    input_base: usize,
    kernel_base: usize,
    input_y: usize,
    output_x_start: usize,
    output: &mut [f32],
    bias: Option<f32>,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    const LANES: usize = 8;
    let vectorized = output.len() / LANES * LANES;
    for offset in (0..vectorized).step_by(LANES) {
        let mut sum = _mm256_setzero_ps();
        for kernel_y in 0..convolution.kernel_height {
            let input_row = input_base
                + (input_y + kernel_y * convolution.dilation) * convolution.input_width
                + output_x_start
                - convolution.pads.1;
            let kernel_row = kernel_base + kernel_y * convolution.kernel_width;
            for kernel_x in 0..convolution.kernel_width {
                let input_start = input_row + kernel_x * convolution.dilation + offset;
                let values = unsafe { _mm256_loadu_ps(input.as_ptr().add(input_start)) };
                let weight = _mm256_set1_ps(kernel[kernel_row + kernel_x]);
                sum = _mm256_fmadd_ps(values, weight, sum);
            }
        }
        if let Some(bias) = bias {
            sum = _mm256_add_ps(sum, _mm256_set1_ps(bias));
        }
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(offset), sum) };
    }
    for (offset, output) in output[vectorized..].iter_mut().enumerate() {
        let output_x = output_x_start + vectorized + offset;
        let mut sum = 0.0_f32;
        for kernel_y in 0..convolution.kernel_height {
            let input_row = input_base
                + (input_y + kernel_y * convolution.dilation) * convolution.input_width
                + output_x
                - convolution.pads.1;
            let kernel_row = kernel_base + kernel_y * convolution.kernel_width;
            for kernel_x in 0..convolution.kernel_width {
                let input_index = input_row + kernel_x * convolution.dilation;
                sum = input[input_index].mul_add(kernel[kernel_row + kernel_x], sum);
            }
        }
        *output = bias.map_or(sum, |bias| sum + bias);
    }
}
