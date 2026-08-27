use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

use super::super::convolution_post::ConvolutionPostOperation;

mod vectorized;

/// Executes multiplier-one depthwise convolution directly over contiguous
/// NCHW storage. Each output retains Candle GEMM's fused multiply-add order,
/// while avoiding one convolution graph and one concatenation per channel.
pub(super) fn conv2d(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    pads: (usize, usize, usize, usize),
    strides: (usize, usize),
    dilation: usize,
) -> Result<Tensor> {
    conv2d_with_post_operation(
        input,
        kernel,
        bias,
        pads,
        strides,
        dilation,
        ConvolutionPostOperation::Identity,
    )
}

pub(super) fn conv2d_with_post_operation(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    pads: (usize, usize, usize, usize),
    strides: (usize, usize),
    dilation: usize,
    post_operation: ConvolutionPostOperation,
) -> Result<Tensor> {
    if !supports_fused_multiply_add() {
        if !post_operation.is_identity() {
            candle_core::bail!(
                "direct CPU depthwise post-operations require fused multiply-add support"
            )
        }
        return candle_channel_conv2d(input, kernel, bias, pads, strides, dilation);
    }
    let operation = DepthwiseConv2d::new(
        input.layout(),
        kernel.layout(),
        pads,
        strides,
        dilation,
        post_operation,
    )?;
    match bias {
        Some(bias) => {
            operation.validate_bias(bias.layout())?;
            input.apply_op3_no_bwd(kernel, bias, &operation)
        }
        None => input.apply_op2_no_bwd(kernel, &operation),
    }
}

pub(super) fn supports_fused_multiply_add() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("fma")
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

#[derive(Clone)]
struct DepthwiseConv2d {
    output_shape: Shape,
    batch: usize,
    channels: usize,
    input_height: usize,
    input_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    output_height: usize,
    output_width: usize,
    pads: (usize, usize, usize, usize),
    strides: (usize, usize),
    dilation: usize,
    kernel_height_span: usize,
    kernel_width_span: usize,
    post_operation: ConvolutionPostOperation,
}

impl DepthwiseConv2d {
    fn new(
        input: &Layout,
        kernel: &Layout,
        pads: (usize, usize, usize, usize),
        strides: (usize, usize),
        dilation: usize,
        post_operation: ConvolutionPostOperation,
    ) -> Result<Self> {
        if !input.is_contiguous() || !kernel.is_contiguous() {
            candle_core::bail!("direct CPU depthwise convolution requires contiguous inputs")
        }
        let (batch, channels, input_height, input_width) = input.shape().dims4()?;
        let (output_channels, kernel_channels, kernel_height, kernel_width) =
            kernel.shape().dims4()?;
        if batch == 0
            || channels == 0
            || output_channels != channels
            || kernel_channels != 1
            || kernel_height == 0
            || kernel_width == 0
            || strides.0 == 0
            || strides.1 == 0
            || dilation == 0
        {
            candle_core::bail!(
                "direct CPU depthwise convolution requires one non-empty kernel per channel and positive spatial strides"
            )
        }
        if !post_operation.supports_channels(channels) {
            candle_core::bail!(
                "depthwise convolution post-operation channel count does not match the output"
            )
        }
        let effective_height = dilation
            .checked_mul(kernel_height - 1)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| candle_core::Error::Msg("depthwise kernel height overflowed".into()))?;
        let effective_width = dilation
            .checked_mul(kernel_width - 1)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| candle_core::Error::Msg("depthwise kernel width overflowed".into()))?;
        let padded_height = input_height
            .checked_add(pads.0)
            .and_then(|value| value.checked_add(pads.2))
            .ok_or_else(|| candle_core::Error::Msg("depthwise input height overflowed".into()))?;
        let padded_width = input_width
            .checked_add(pads.1)
            .and_then(|value| value.checked_add(pads.3))
            .ok_or_else(|| candle_core::Error::Msg("depthwise input width overflowed".into()))?;
        let output_height = padded_height
            .checked_sub(effective_height)
            .map(|remaining| remaining / strides.0 + 1)
            .ok_or_else(|| {
                candle_core::Error::Msg("depthwise kernel exceeds input height".into())
            })?;
        let output_width = padded_width
            .checked_sub(effective_width)
            .map(|remaining| remaining / strides.1 + 1)
            .ok_or_else(|| {
                candle_core::Error::Msg("depthwise kernel exceeds input width".into())
            })?;
        Ok(Self {
            output_shape: Shape::from_dims(&[batch, channels, output_height, output_width]),
            batch,
            channels,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            output_height,
            output_width,
            pads,
            strides,
            dilation,
            kernel_height_span: effective_height - 1,
            kernel_width_span: effective_width - 1,
            post_operation,
        })
    }

    fn validate_bias(&self, bias: &Layout) -> Result<()> {
        if !bias.is_contiguous() || bias.shape().dims1()? != self.channels {
            candle_core::bail!("depthwise bias channel count does not match the output")
        }
        Ok(())
    }

    fn execute(
        &self,
        input: &[f32],
        kernel: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<(CpuStorage, Shape)> {
        let output_spatial = self.output_height * self.output_width;
        let mut output = vec![0.0_f32; self.batch * self.channels * output_spatial];
        let vectorized_interior = vectorized::supported() && self.strides.1 == 1;
        let interior_x = self.interior_output_x_range();
        let execute_channel = |batch: usize, channel: usize, output: &mut [f32]| -> Result<()> {
            let input_base =
                (batch * self.channels + channel) * self.input_height * self.input_width;
            let kernel_base = channel * self.kernel_height * self.kernel_width;
            for output_y in 0..self.output_height {
                let padded_y = output_y * self.strides.0;
                let interior_y = padded_y.checked_sub(self.pads.0).filter(|input_y| {
                    input_y
                        .checked_add(self.kernel_height_span)
                        .is_some_and(|last| last < self.input_height)
                });
                let output_row = &mut output[output_y * self.output_width..][..self.output_width];
                let Some(input_y) = interior_y.filter(|_| !interior_x.is_empty()) else {
                    for (output_x, output) in output_row.iter_mut().enumerate() {
                        let padded_x = output_x * self.strides.1;
                        let sum = self.accumulate_boundary(
                            input,
                            kernel,
                            input_base,
                            kernel_base,
                            padded_y,
                            padded_x,
                        );
                        *output = bias.map_or(sum, |bias| sum + bias[channel]);
                    }
                    continue;
                };
                for (output_x, output) in output_row[..interior_x.start].iter_mut().enumerate() {
                    let padded_x = output_x * self.strides.1;
                    let sum = self.accumulate_boundary(
                        input,
                        kernel,
                        input_base,
                        kernel_base,
                        padded_y,
                        padded_x,
                    );
                    *output = bias.map_or(sum, |bias| sum + bias[channel]);
                }
                let interior_bias = bias.map(|bias| bias[channel]);
                let bias_applied = self.accumulate_interior_row(
                    input,
                    kernel,
                    input_base,
                    kernel_base,
                    input_y,
                    interior_x.clone(),
                    output_row,
                    vectorized_interior,
                    interior_bias,
                );
                if !bias_applied {
                    if let Some(bias) = bias {
                        output_row[interior_x.clone()]
                            .iter_mut()
                            .for_each(|output| *output += bias[channel]);
                    }
                }
                for (offset, output) in output_row[interior_x.end..].iter_mut().enumerate() {
                    let output_x = interior_x.end + offset;
                    let padded_x = output_x * self.strides.1;
                    let sum = self.accumulate_boundary(
                        input,
                        kernel,
                        input_base,
                        kernel_base,
                        padded_y,
                        padded_x,
                    );
                    *output = bias.map_or(sum, |bias| sum + bias[channel]);
                }
            }
            if !self.post_operation.is_identity() {
                self.post_operation.apply_channel(channel, output, None)?;
            }
            Ok(())
        };

        output
            .par_chunks_mut(output_spatial)
            .enumerate()
            .try_for_each(|(batch_channel, output)| {
                execute_channel(
                    batch_channel / self.channels,
                    batch_channel % self.channels,
                    output,
                )
            })?;
        Ok((CpuStorage::F32(output), self.output_shape.clone()))
    }

    fn interior_output_x_range(&self) -> std::ops::Range<usize> {
        if self.kernel_width_span >= self.input_width {
            return 0..0;
        }
        let start = (self.pads.1 / self.strides.1)
            .saturating_add(usize::from(!self.pads.1.is_multiple_of(self.strides.1)));
        let last_padded_x = self.pads.1 + self.input_width - self.kernel_width_span - 1;
        let end = (last_padded_x / self.strides.1)
            .saturating_add(1)
            .min(self.output_width);
        start.min(end)..end
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_interior_row(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_base: usize,
        kernel_base: usize,
        input_y: usize,
        output_x: std::ops::Range<usize>,
        output: &mut [f32],
        vectorized: bool,
        bias: Option<f32>,
    ) -> bool {
        let output = &mut output[output_x.clone()];
        // `execute` allocates a fresh zeroed output and visits every row and
        // horizontal segment exactly once. The interior therefore already
        // has the required additive identity; clearing it again only adds a
        // redundant full-width memory pass.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if vectorized && output.len() >= 8 {
            // SAFETY: the feature check is performed once for this execution,
            // every input span has already passed the interior-range proof,
            // and the helper handles its scalar tail without reading beyond
            // either slice. Each lane retains the same kernel traversal and
            // fused multiply-add order as the scalar reference.
            unsafe {
                vectorized::accumulate_interior_row(
                    self,
                    input,
                    kernel,
                    input_base,
                    kernel_base,
                    input_y,
                    output_x.start,
                    output,
                    bias,
                );
            }
            return true;
        }
        for kernel_y in 0..self.kernel_height {
            let input_row = input_base
                + (input_y + kernel_y * self.dilation) * self.input_width
                + output_x.start * self.strides.1
                - self.pads.1;
            let kernel_row = kernel_base + kernel_y * self.kernel_width;
            for kernel_x in 0..self.kernel_width {
                let input_start = input_row + kernel_x * self.dilation;
                let weight = kernel[kernel_row + kernel_x];
                if self.strides.1 == 1 {
                    let length = output.len();
                    output
                        .iter_mut()
                        .zip(&input[input_start..input_start + length])
                        .for_each(|(output, input)| *output = input.mul_add(weight, *output));
                } else {
                    output.iter_mut().enumerate().for_each(|(index, output)| {
                        *output =
                            input[input_start + index * self.strides.1].mul_add(weight, *output);
                    });
                }
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_boundary(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_base: usize,
        kernel_base: usize,
        padded_y: usize,
        padded_x: usize,
    ) -> f32 {
        let mut sum = 0.0_f32;
        for kernel_y in 0..self.kernel_height {
            let input_y = (padded_y + kernel_y * self.dilation)
                .checked_sub(self.pads.0)
                .filter(|input_y| *input_y < self.input_height);
            let kernel_row = kernel_base + kernel_y * self.kernel_width;
            for kernel_x in 0..self.kernel_width {
                let input_x = (padded_x + kernel_x * self.dilation)
                    .checked_sub(self.pads.1)
                    .filter(|input_x| *input_x < self.input_width);
                let value = input_y.zip(input_x).map_or(0.0_f32, |(input_y, input_x)| {
                    input[input_base + input_y * self.input_width + input_x]
                });
                sum = value.mul_add(kernel[kernel_row + kernel_x], sum);
            }
        }
        sum
    }
}

impl CustomOp2 for DepthwiseConv2d {
    fn name(&self) -> &'static str {
        "a3s-direct-cpu-depthwise-convolution"
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
            self.batch * self.channels * self.input_height * self.input_width,
            "depthwise input",
        )?;
        let kernel = contiguous_values(
            kernel.as_slice::<f32>()?,
            kernel_layout,
            self.channels * self.kernel_height * self.kernel_width,
            "depthwise kernel",
        )?;
        self.execute(input, kernel, None)
    }
}

impl CustomOp3 for DepthwiseConv2d {
    fn name(&self) -> &'static str {
        "a3s-direct-cpu-biased-depthwise-convolution"
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
        self.validate_bias(bias_layout)?;
        let input = contiguous_values(
            input.as_slice::<f32>()?,
            input_layout,
            self.batch * self.channels * self.input_height * self.input_width,
            "depthwise input",
        )?;
        let kernel = contiguous_values(
            kernel.as_slice::<f32>()?,
            kernel_layout,
            self.channels * self.kernel_height * self.kernel_width,
            "depthwise kernel",
        )?;
        let bias = contiguous_values(
            bias.as_slice::<f32>()?,
            bias_layout,
            self.channels,
            "depthwise bias",
        )?;
        self.execute(input, kernel, Some(bias))
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

fn candle_channel_conv2d(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    pads: (usize, usize, usize, usize),
    strides: (usize, usize),
    dilation: usize,
) -> Result<Tensor> {
    let input = input
        .pad_with_zeros(2, pads.0, pads.2)?
        .pad_with_zeros(3, pads.1, pads.3)?;
    let (_, channels, _, _) = input.dims4()?;
    let inputs = input.chunk(channels, 1)?;
    let kernels = kernel.chunk(channels, 0)?;
    let outputs = inputs
        .into_par_iter()
        .zip(kernels)
        .map(|(input, kernel)| {
            let output = input.conv2d(&kernel, 0, 1, dilation, 1)?;
            let output = subsample_axis(output, 2, strides.0)?;
            subsample_axis(output, 3, strides.1)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut output = Tensor::cat(&outputs, 1)?;
    if let Some(bias) = bias {
        output = output.broadcast_add(&bias.reshape((1, channels, 1, 1))?)?;
    }
    Ok(output)
}

fn subsample_axis(input: Tensor, axis: usize, stride: usize) -> Result<Tensor> {
    if stride == 1 {
        return Ok(input);
    }
    let appended_axis = input.rank();
    input.unfold(axis, 1, stride)?.squeeze(appended_axis)
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::Instant;

    use candle_core::Device;

    use super::*;

    #[test]
    fn direct_depthwise_accumulation_matches_candle_channel_bits() {
        let device = Device::Cpu;
        for (
            batch,
            channels,
            height,
            width,
            kernel_height,
            kernel_width,
            pads,
            strides,
            dilation,
        ) in [
            (1, 3, 9, 13, 3, 3, (0, 0, 0, 0), (1, 1), 1),
            (2, 7, 11, 17, 3, 3, (1, 1, 1, 1), (1, 1), 1),
            (4, 5, 12, 18, 3, 3, (1, 1, 0, 0), (2, 2), 1),
            (2, 4, 8, 16, 1, 1, (0, 0, 0, 0), (1, 1), 1),
            (2, 5, 13, 19, 2, 3, (2, 1, 1, 2), (1, 1), 2),
            (3, 6, 15, 23, 1, 7, (0, 3, 0, 3), (1, 1), 1),
            (2, 5, 13, 19, 3, 3, (1, 2, 0, 1), (2, 1), 1),
            (2, 5, 13, 19, 3, 3, (0, 1, 2, 0), (1, 2), 1),
        ] {
            let input = Tensor::from_iter(
                (0..(batch + 1) * channels * height * width)
                    .map(|index| ((index * 37 % 509) as f32 - 254.0) / 113.0),
                &device,
            )
            .unwrap()
            .reshape((batch + 1, channels, height, width))
            .unwrap()
            .narrow(0, 1, batch)
            .unwrap();
            let kernel = Tensor::from_iter(
                (0..(channels + 1) * kernel_height * kernel_width)
                    .map(|index| ((index * 53 % 257) as f32 - 128.0) / 97.0),
                &device,
            )
            .unwrap()
            .reshape((channels + 1, 1, kernel_height, kernel_width))
            .unwrap()
            .narrow(0, 1, channels)
            .unwrap();
            let bias = Tensor::from_iter(
                (0..channels + 2).map(|index| (index as f32 - 3.0) / 17.0),
                &device,
            )
            .unwrap()
            .narrow(0, 1, channels)
            .unwrap();
            assert!(input.is_contiguous());
            assert!(kernel.is_contiguous());
            assert!(bias.is_contiguous());
            assert_ne!(input.layout().start_offset(), 0);
            assert_ne!(kernel.layout().start_offset(), 0);
            assert_ne!(bias.layout().start_offset(), 0);

            for bias in [None, Some(&bias)] {
                let expected =
                    candle_channel_conv2d(&input, &kernel, bias, pads, strides, dilation).unwrap();
                let actual = conv2d(&input, &kernel, bias, pads, strides, dilation).unwrap();

                assert_eq!(actual.dims(), expected.dims());
                assert_eq!(
                    actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    "batch={batch} channels={channels} input={height}x{width} kernel={kernel_height}x{kernel_width} pads={pads:?} strides={strides:?} dilation={dilation} bias={}",
                    bias.is_some()
                );
            }
        }
    }

    #[test]
    fn fused_depthwise_batch_norm_matches_explicit_graph_bits() {
        let device = Device::Cpu;
        for (batch, channels, height, width, pads, strides) in [
            (1, 3, 9, 13, (0, 0, 0, 0), (1, 1)),
            (2, 7, 11, 17, (1, 1, 1, 1), (1, 1)),
            (4, 5, 12, 18, (1, 1, 0, 0), (2, 2)),
        ] {
            let input = Tensor::from_iter(
                (0..batch * channels * height * width)
                    .map(|index| ((index * 37 % 509) as f32 - 254.0) / 113.0),
                &device,
            )
            .unwrap()
            .reshape((batch, channels, height, width))
            .unwrap();
            let kernel = Tensor::from_iter(
                (0..channels * 3 * 3).map(|index| ((index * 53 % 257) as f32 - 128.0) / 97.0),
                &device,
            )
            .unwrap()
            .reshape((channels, 1, 3, 3))
            .unwrap();
            let convolution_bias = Tensor::from_iter(
                (0..channels).map(|index| (index as f32 - 3.0) / 17.0),
                &device,
            )
            .unwrap();
            let scale = (0..channels)
                .map(|channel| (channel as f32 + 3.0) / 11.0)
                .collect::<Vec<_>>();
            let bias = (0..channels)
                .map(|channel| (channel as f32 - 5.0) / 17.0)
                .collect::<Vec<_>>();
            let mean = (0..channels)
                .map(|channel| (channel as f32 - 2.0) / 13.0)
                .collect::<Vec<_>>();
            let variance = (0..channels)
                .map(|channel| (channel as f32 + 7.0) / 19.0)
                .collect::<Vec<_>>();
            let epsilon = 0.000_01_f32;
            let alpha = 1.0_f32 / 6.0;
            let beta = 0.5_f32;
            let explicit =
                conv2d(&input, &kernel, Some(&convolution_bias), pads, strides, 1).unwrap();
            let (_, _, output_height, output_width) = explicit.dims4().unwrap();
            let output_spatial = output_height * output_width;
            let expected = explicit
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .chunks(output_spatial)
                .enumerate()
                .flat_map(|(batch_channel, values)| {
                    let channel = batch_channel % channels;
                    let stddev = (variance[channel] + epsilon).sqrt();
                    let scale = scale[channel];
                    let bias = bias[channel];
                    let mean = mean[channel];
                    values.iter().map(move |value| {
                        let normalized = (((*value - mean) / stddev) * scale) + bias;
                        normalized * ((normalized * alpha) + beta).clamp(0.0, 1.0)
                    })
                })
                .collect::<Vec<_>>();
            let post_operation = ConvolutionPostOperation::batch_normalization(
                &scale,
                &bias,
                &mean,
                &variance,
                epsilon,
                Some((alpha, beta)),
            )
            .unwrap();
            let actual = conv2d_with_post_operation(
                &input,
                &kernel,
                Some(&convolution_bias),
                pads,
                strides,
                1,
                post_operation,
            )
            .unwrap();

            assert_eq!(
                actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected,
                "batch={batch} channels={channels} input={height}x{width} pads={pads:?} strides={strides:?}",
            );
        }
    }

    #[test]
    #[ignore = "diagnostic CPU depthwise kernel comparison"]
    fn compare_direct_and_channel_graph_depthwise() {
        let device = Device::Cpu;
        for (batch, channels, height, width, stride) in [
            (2, 96, 14, 130, 1),
            (8, 96, 14, 258, 1),
            (8, 192, 8, 130, 1),
            (24, 192, 5, 67, 1),
            (8, 192, 8, 130, 2),
            (8, 384, 5, 67, 1),
        ] {
            let input = Tensor::zeros(
                (batch, channels, height, width),
                candle_core::DType::F32,
                &device,
            )
            .unwrap();
            let kernel =
                Tensor::zeros((channels, 1, 3, 3), candle_core::DType::F32, &device).unwrap();
            let bias = Tensor::zeros(channels, candle_core::DType::F32, &device).unwrap();
            black_box(
                candle_channel_conv2d(
                    &input,
                    &kernel,
                    Some(&bias),
                    (0, 0, 0, 0),
                    (stride, stride),
                    1,
                )
                .unwrap(),
            );
            black_box(
                conv2d(
                    &input,
                    &kernel,
                    Some(&bias),
                    (0, 0, 0, 0),
                    (stride, stride),
                    1,
                )
                .unwrap(),
            );
            let iterations = 3_u32;
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(
                    candle_channel_conv2d(
                        &input,
                        &kernel,
                        Some(&bias),
                        (0, 0, 0, 0),
                        (stride, stride),
                        1,
                    )
                    .unwrap(),
                );
            }
            let channel_graph = started.elapsed();
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(
                    conv2d(
                        &input,
                        &kernel,
                        Some(&bias),
                        (0, 0, 0, 0),
                        (stride, stride),
                        1,
                    )
                    .unwrap(),
                );
            }
            let direct = started.elapsed();
            eprintln!(
                "CPU depthwise profile: batch={batch} channels={channels} input={height}x{width} stride={stride} channel_graph_ms={:.3} direct_ms={:.3}",
                channel_graph.as_secs_f64() * 1_000.0 / f64::from(iterations),
                direct.as_secs_f64() * 1_000.0 / f64::from(iterations),
            );
        }
    }

    #[test]
    #[ignore = "diagnostic CPU depthwise padding comparison"]
    fn compare_native_and_materialized_padding() {
        let device = Device::Cpu;
        for (batch, channels, height, width, pads, strides) in [
            (2, 32, 48, 320, (1, 1, 1, 1), (1, 1)),
            (8, 96, 12, 130, (1, 1, 1, 1), (1, 1)),
            (8, 96, 12, 130, (0, 1, 1, 1), (2, 1)),
            (24, 192, 6, 67, (1, 1, 1, 1), (1, 1)),
        ] {
            let input = Tensor::zeros(
                (batch, channels, height, width),
                candle_core::DType::F32,
                &device,
            )
            .unwrap();
            let kernel =
                Tensor::zeros((channels, 1, 3, 3), candle_core::DType::F32, &device).unwrap();
            let bias = Tensor::zeros(channels, candle_core::DType::F32, &device).unwrap();
            let materialized = || {
                let input = input
                    .pad_with_zeros(2, pads.0, pads.2)?
                    .pad_with_zeros(3, pads.1, pads.3)?;
                conv2d(&input, &kernel, Some(&bias), (0, 0, 0, 0), strides, 1)
            };
            black_box(materialized().unwrap());
            black_box(conv2d(&input, &kernel, Some(&bias), pads, strides, 1).unwrap());
            let iterations = 3_u32;
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(materialized().unwrap());
            }
            let padded = started.elapsed();
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(conv2d(&input, &kernel, Some(&bias), pads, strides, 1).unwrap());
            }
            let native = started.elapsed();
            eprintln!(
                "CPU depthwise padding profile: batch={batch} channels={channels} input={height}x{width} pads={pads:?} strides={strides:?} materialized_ms={:.3} native_ms={:.3}",
                padded.as_secs_f64() * 1_000.0 / f64::from(iterations),
                native.as_secs_f64() * 1_000.0 / f64::from(iterations),
            );
        }
    }
}
