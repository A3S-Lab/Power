use candle_core::{Result, Tensor};

use super::convolution_post::ConvolutionPostOperation;

mod cpu;
#[cfg(feature = "embedded-cuda")]
mod cuda;

/// Executes a contiguous CPU spatial convolution without materializing a
/// complete NHWC copy of the NCHW input.
pub(super) fn conv2d(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    pads: (usize, usize, usize, usize),
    stride: usize,
    dilation: usize,
) -> Result<Tensor> {
    #[cfg(feature = "embedded-cuda")]
    if input.device().is_cuda() {
        return cuda::conv2d(input, kernel, bias, pads, stride, dilation);
    }

    cpu::conv2d(input, kernel, bias, pads, stride, dilation)
}

pub(super) fn conv2d_with_post_operation(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    pads: (usize, usize, usize, usize),
    stride: usize,
    dilation: usize,
    post_operation: ConvolutionPostOperation,
) -> Result<Tensor> {
    #[cfg(feature = "embedded-cuda")]
    if input.device().is_cuda() {
        return cuda::conv2d_with_post_operation(
            input,
            kernel,
            bias,
            pads,
            stride,
            dilation,
            post_operation,
        );
    }

    cpu::conv2d_with_post_operation(input, kernel, bias, pads, stride, dilation, post_operation)
}

#[cfg(test)]
mod tests;
