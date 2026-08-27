use std::sync::Arc;

use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};
use gemm::{gemm, Parallelism};
use rayon::prelude::*;

use super::ConvolutionPostOperation;

pub(super) fn conv2d(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    post_operation: ConvolutionPostOperation,
) -> Result<Tensor> {
    let operation = PointwiseConv2d::new(input.layout(), kernel.layout(), post_operation)?;
    match bias {
        Some(bias) => {
            operation.validate_bias(bias.layout())?;
            input.apply_op3_no_bwd(kernel, bias, &operation)
        }
        None => input.apply_op2_no_bwd(kernel, &operation),
    }
}

pub(super) fn conv2d_with_residual(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    let bias = bias
        .map(|bias| bias.flatten_all()?.to_vec1::<f32>().map(Arc::<[f32]>::from))
        .transpose()?;
    let operation = PointwiseConv2d::new(
        input.layout(),
        kernel.layout(),
        ConvolutionPostOperation::Identity,
    )?
    .with_residual(residual.layout(), bias)?;
    input.apply_op3_no_bwd(kernel, residual, &operation)
}

#[derive(Clone)]
enum ThirdInput {
    Bias,
    Residual { bias: Option<Arc<[f32]>> },
}

#[derive(Clone)]
struct PointwiseConv2d {
    output_shape: Shape,
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    spatial: usize,
    post_operation: ConvolutionPostOperation,
    third_input: ThirdInput,
}

impl PointwiseConv2d {
    fn new(
        input: &Layout,
        kernel: &Layout,
        post_operation: ConvolutionPostOperation,
    ) -> Result<Self> {
        if !input.is_contiguous() || !kernel.is_contiguous() {
            candle_core::bail!("direct CPU pointwise convolution requires contiguous inputs")
        }
        let (batch, input_channels, height, width) = input.shape().dims4()?;
        let (output_channels, kernel_channels, kernel_height, kernel_width) =
            kernel.shape().dims4()?;
        if batch == 0
            || input_channels == 0
            || output_channels == 0
            || kernel_channels != input_channels
            || kernel_height != 1
            || kernel_width != 1
        {
            candle_core::bail!(
                "direct CPU pointwise convolution requires non-empty matching 1x1 kernels"
            )
        }
        if !post_operation.supports_channels(output_channels) {
            candle_core::bail!(
                "pointwise convolution post-operation channel count does not match the output"
            )
        }
        let spatial = height
            .checked_mul(width)
            .ok_or_else(|| candle_core::Error::Msg("pointwise spatial size overflowed".into()))?;
        Ok(Self {
            output_shape: Shape::from_dims(&[batch, output_channels, height, width]),
            batch,
            input_channels,
            output_channels,
            spatial,
            post_operation,
            third_input: ThirdInput::Bias,
        })
    }

    fn validate_bias(&self, bias: &Layout) -> Result<()> {
        if !bias.is_contiguous() || bias.shape().dims1()? != self.output_channels {
            candle_core::bail!("pointwise bias channel count does not match the output")
        }
        Ok(())
    }

    fn with_residual(mut self, residual: &Layout, bias: Option<Arc<[f32]>>) -> Result<Self> {
        if !residual.is_contiguous() || residual.shape() != &self.output_shape {
            candle_core::bail!(
                "pointwise residual must be contiguous and match the convolution output"
            )
        }
        if bias
            .as_ref()
            .is_some_and(|bias| bias.len() != self.output_channels)
        {
            candle_core::bail!("pointwise bias channel count does not match the output")
        }
        self.third_input = ThirdInput::Residual { bias };
        Ok(self)
    }

    fn execute(
        &self,
        input: &[f32],
        kernel: &[f32],
        bias: Option<&[f32]>,
        residual: Option<&[f32]>,
    ) -> Result<(CpuStorage, Shape)> {
        let input_batch_elements = self.input_channels * self.spatial;
        let output_batch_elements = self.output_channels * self.spatial;
        let nested = rayon::current_thread_index().is_some();
        let mut output = vec![0.0_f32; self.batch * output_batch_elements];
        let workers = rayon::current_num_threads();
        if !nested
            && should_parallelize_batches(self.batch, self.input_channels, self.spatial, workers)
        {
            output
                .par_chunks_mut(output_batch_elements)
                .zip(input.par_chunks(input_batch_elements))
                .for_each(|(output, input)| unsafe {
                    multiply(
                        output,
                        input,
                        kernel,
                        self.output_channels,
                        self.spatial,
                        self.input_channels,
                        Parallelism::None,
                    );
                });
        } else {
            let parallelism = if nested || workers <= 1 {
                Parallelism::None
            } else {
                Parallelism::Rayon(workers)
            };
            for batch in 0..self.batch {
                let input = &input[batch * input_batch_elements..][..input_batch_elements];
                let output = &mut output[batch * output_batch_elements..][..output_batch_elements];
                unsafe {
                    multiply(
                        output,
                        input,
                        kernel,
                        self.output_channels,
                        self.spatial,
                        self.input_channels,
                        parallelism,
                    );
                }
            }
        }
        self.apply_post_operations(&mut output, bias, residual, nested)?;
        Ok((CpuStorage::F32(output), self.output_shape.clone()))
    }

    fn apply_post_operations(
        &self,
        output: &mut [f32],
        bias: Option<&[f32]>,
        residual: Option<&[f32]>,
        nested: bool,
    ) -> Result<()> {
        if bias.is_some() || !self.post_operation.is_identity() || residual.is_some() {
            let apply = |batch_channel: usize,
                         output: &mut [f32],
                         residual: Option<&[f32]>|
             -> Result<()> {
                let channel = batch_channel % self.output_channels;
                let convolution_bias = bias.map(|bias| bias[channel]);
                if let Some(residual) = residual {
                    self.post_operation.add_identity_residual(
                        output,
                        residual,
                        convolution_bias,
                    )?;
                } else {
                    self.post_operation
                        .apply_channel(channel, output, convolution_bias)?;
                }
                Ok(())
            };
            if nested {
                match residual {
                    Some(residual) => output
                        .chunks_mut(self.spatial)
                        .zip(residual.chunks(self.spatial))
                        .enumerate()
                        .try_for_each(|(batch_channel, (output, residual))| {
                            apply(batch_channel, output, Some(residual))
                        })?,
                    None => output.chunks_mut(self.spatial).enumerate().try_for_each(
                        |(batch_channel, output)| apply(batch_channel, output, None),
                    )?,
                }
            } else {
                match residual {
                    Some(residual) => output
                        .par_chunks_mut(self.spatial)
                        .zip(residual.par_chunks(self.spatial))
                        .enumerate()
                        .try_for_each(|(batch_channel, (output, residual))| {
                            apply(batch_channel, output, Some(residual))
                        })?,
                    None => output
                        .par_chunks_mut(self.spatial)
                        .enumerate()
                        .try_for_each(|(batch_channel, output)| {
                            apply(batch_channel, output, None)
                        })?,
                }
            }
        }
        Ok(())
    }
}

/// Selects independent batch work when it can occupy the complete worker
/// pool, or when each matrix has too few spatial columns to expose useful
/// inner-GEMM parallelism. The decision depends only on tensor geometry and
/// the live executor capacity; document and model identities are irrelevant.
fn should_parallelize_batches(
    batch: usize,
    input_channels: usize,
    spatial: usize,
    workers: usize,
) -> bool {
    batch > 1 && workers > 1 && (batch >= workers || spatial < input_channels)
}

#[allow(clippy::too_many_arguments)]
unsafe fn multiply(
    output: &mut [f32],
    input: &[f32],
    kernel: &[f32],
    output_channels: usize,
    spatial: usize,
    input_channels: usize,
    parallelism: Parallelism,
) {
    unsafe {
        gemm(
            output_channels,
            spatial,
            input_channels,
            output.as_mut_ptr(),
            1,
            spatial as isize,
            false,
            kernel.as_ptr(),
            1,
            input_channels as isize,
            input.as_ptr(),
            1,
            spatial as isize,
            0.0_f32,
            1.0_f32,
            false,
            false,
            false,
            parallelism,
        );
    }
}

impl CustomOp2 for PointwiseConv2d {
    fn name(&self) -> &'static str {
        "a3s-direct-cpu-pointwise-convolution"
    }

    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        input_layout: &Layout,
        kernel: &CpuStorage,
        kernel_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let input = contiguous_values(
            input.as_slice::<f32>()?,
            input_layout,
            self.batch * self.input_channels * self.spatial,
            "pointwise input",
        )?;
        let kernel = contiguous_values(
            kernel.as_slice::<f32>()?,
            kernel_layout,
            self.output_channels * self.input_channels,
            "pointwise kernel",
        )?;
        self.execute(input, kernel, None, None)
    }
}

impl CustomOp3 for PointwiseConv2d {
    fn name(&self) -> &'static str {
        "a3s-direct-cpu-biased-pointwise-convolution"
    }

    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        input_layout: &Layout,
        kernel: &CpuStorage,
        kernel_layout: &Layout,
        bias: &CpuStorage,
        bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let input = contiguous_values(
            input.as_slice::<f32>()?,
            input_layout,
            self.batch * self.input_channels * self.spatial,
            "pointwise input",
        )?;
        let kernel = contiguous_values(
            kernel.as_slice::<f32>()?,
            kernel_layout,
            self.output_channels * self.input_channels,
            "pointwise kernel",
        )?;
        match &self.third_input {
            ThirdInput::Bias => {
                self.validate_bias(bias_layout)?;
                let bias = contiguous_values(
                    bias.as_slice::<f32>()?,
                    bias_layout,
                    self.output_channels,
                    "pointwise bias",
                )?;
                self.execute(input, kernel, Some(bias), None)
            }
            ThirdInput::Residual {
                bias: convolution_bias,
            } => {
                if !bias_layout.is_contiguous() || bias_layout.shape() != &self.output_shape {
                    candle_core::bail!(
                        "pointwise residual must be contiguous and match the convolution output"
                    )
                }
                let residual = contiguous_values(
                    bias.as_slice::<f32>()?,
                    bias_layout,
                    self.batch * self.output_channels * self.spatial,
                    "pointwise residual",
                )?;
                self.execute(input, kernel, convolution_bias.as_deref(), Some(residual))
            }
        }
    }
}

fn contiguous_values<'a>(
    storage: &'a [f32],
    layout: &Layout,
    elements: usize,
    label: &str,
) -> Result<&'a [f32]> {
    let start = layout.start_offset();
    storage
        .get(start..start + elements)
        .ok_or_else(|| candle_core::Error::Msg(format!("{label} layout is out of bounds")))
}

#[cfg(test)]
mod tests;
