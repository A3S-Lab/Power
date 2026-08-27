use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};
use gemm::{gemm, Parallelism};
use rayon::prelude::*;

use super::super::convolution_post::ConvolutionPostOperation;

mod single_output;

// Bound each task's temporary patch matrix. The output-position ceiling also
// matches Candle's released tiled convolution geometry, which keeps the GEMM
// reduction order stable while avoiding a document- or model-specific policy.
const MAX_TILE_OUTPUTS: usize = 512;

pub(super) fn conv2d(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    pads: (usize, usize, usize, usize),
    stride: usize,
    dilation: usize,
) -> Result<Tensor> {
    conv2d_with_post_operation(
        input,
        kernel,
        bias,
        pads,
        stride,
        dilation,
        ConvolutionPostOperation::Identity,
    )
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
    let operation = SpatialConv2d::new(
        input.layout(),
        kernel.layout(),
        pads,
        stride,
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

#[derive(Clone)]
struct SpatialConv2d {
    output_shape: Shape,
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    input_height: usize,
    input_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    output_height: usize,
    output_width: usize,
    pads: (usize, usize, usize, usize),
    stride: usize,
    dilation: usize,
    post_operation: ConvolutionPostOperation,
}

impl SpatialConv2d {
    fn new(
        input: &Layout,
        kernel: &Layout,
        pads: (usize, usize, usize, usize),
        stride: usize,
        dilation: usize,
        post_operation: ConvolutionPostOperation,
    ) -> Result<Self> {
        if !input.is_contiguous() || !kernel.is_contiguous() {
            candle_core::bail!("direct CPU spatial convolution requires contiguous inputs")
        }
        let (batch, input_channels, input_height, input_width) = input.shape().dims4()?;
        let (output_channels, kernel_channels, kernel_height, kernel_width) =
            kernel.shape().dims4()?;
        if batch == 0
            || input_channels == 0
            || output_channels == 0
            || kernel_channels != input_channels
            || kernel_height == 0
            || kernel_width == 0
            || stride == 0
            || dilation == 0
        {
            candle_core::bail!(
                "direct CPU spatial convolution requires non-empty matching kernels, batches, and positive geometry"
            )
        }
        if !post_operation.supports_channels(output_channels) {
            candle_core::bail!(
                "spatial convolution post-operation channel count does not match the output"
            )
        }
        let effective_height = effective_kernel(kernel_height, dilation, "height")?;
        let effective_width = effective_kernel(kernel_width, dilation, "width")?;
        let padded_height = input_height
            .checked_add(pads.0)
            .and_then(|value| value.checked_add(pads.2))
            .ok_or_else(|| spatial_error("spatial convolution input height overflowed"))?;
        let padded_width = input_width
            .checked_add(pads.1)
            .and_then(|value| value.checked_add(pads.3))
            .ok_or_else(|| spatial_error("spatial convolution input width overflowed"))?;
        let output_height = output_dimension(padded_height, effective_height, stride, "height")?;
        let output_width = output_dimension(padded_width, effective_width, stride, "width")?;
        Ok(Self {
            output_shape: Shape::from_dims(&[batch, output_channels, output_height, output_width]),
            batch,
            input_channels,
            output_channels,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            output_height,
            output_width,
            pads,
            stride,
            dilation,
            post_operation,
        })
    }

    fn validate_bias(&self, bias: &Layout) -> Result<()> {
        if !bias.is_contiguous() || bias.shape().dims1()? != self.output_channels {
            candle_core::bail!("spatial convolution bias channel count does not match the output")
        }
        Ok(())
    }

    fn execute(
        &self,
        input: &[f32],
        kernel: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<(CpuStorage, Shape)> {
        // An im2col matrix can amortize its copy only when multiple output
        // channels reuse it. With one output channel there is no reuse, so a
        // direct traversal removes one full patch write/read round trip.
        if self.output_channels == 1 {
            return single_output::execute(self, input, kernel, bias);
        }
        let input_spatial = self.input_height * self.input_width;
        let output_spatial = self.output_height * self.output_width;
        let input_batch_elements = self.input_channels * input_spatial;
        let output_batch_elements = self.output_channels * output_spatial;
        let kernel_size = self.input_channels * self.kernel_height * self.kernel_width;
        let packed_kernel = self.pack_kernel(kernel, kernel_size);
        let tile_outputs = output_spatial.min(MAX_TILE_OUTPUTS);
        let tiles_per_batch = output_spatial.div_ceil(tile_outputs);
        let task_count = self.batch * tiles_per_batch;
        let mut output = vec![0.0_f32; self.batch * output_batch_elements];
        let output_address = output.as_mut_ptr() as usize;
        let nested = rayon::current_thread_index().is_some();
        let workers = rayon::current_num_threads();

        let execute_task = |task: usize, parallelism: Parallelism, scratch: &mut SpatialScratch| {
            let batch = task / tiles_per_batch;
            let tile = task % tiles_per_batch;
            let tile_start = tile * tile_outputs;
            let tile_len = (output_spatial - tile_start).min(tile_outputs);
            let input = &input[batch * input_batch_elements..][..input_batch_elements];
            scratch.prepare(kernel_size * tile_len, tile_len);
            self.extract_columns(
                input,
                tile_start,
                tile_len,
                &mut scratch.columns,
                &mut scratch.source_offsets,
            );
            let output_offset = batch * output_batch_elements + tile_start;
            // SAFETY: every task owns a disjoint batch/tile range. GEMM writes
            // exactly `output_channels * tile_len` values using the declared
            // NCHW channel stride and never crosses another tile.
            unsafe {
                gemm(
                    self.output_channels,
                    tile_len,
                    kernel_size,
                    (output_address as *mut f32).add(output_offset),
                    1,
                    output_spatial as isize,
                    false,
                    packed_kernel.as_ptr(),
                    1,
                    kernel_size as isize,
                    scratch.columns.as_ptr(),
                    1,
                    tile_len as isize,
                    0.0_f32,
                    1.0_f32,
                    false,
                    false,
                    false,
                    parallelism,
                );
            }
        };

        if task_count == 1 {
            let mut scratch = SpatialScratch::default();
            let parallelism = if nested || workers <= 1 {
                Parallelism::None
            } else {
                Parallelism::Rayon(workers)
            };
            execute_task(0, parallelism, &mut scratch);
        } else if nested || task_count >= workers {
            let parallel_tasks = task_count.min(workers);
            (0..parallel_tasks).into_par_iter().for_each(|worker| {
                let mut scratch = SpatialScratch::default();
                for task in (worker..task_count).step_by(parallel_tasks) {
                    execute_task(task, Parallelism::None, &mut scratch);
                }
            });
        } else {
            let mut scratch = SpatialScratch::default();
            for task in 0..task_count {
                execute_task(task, Parallelism::Rayon(workers), &mut scratch);
            }
        }

        if bias.is_some() || !self.post_operation.is_identity() {
            output
                .par_chunks_mut(output_spatial)
                .enumerate()
                .try_for_each(|(batch_channel, values)| -> Result<()> {
                    let channel = batch_channel % self.output_channels;
                    self.post_operation.apply_channel(
                        channel,
                        values,
                        bias.map(|bias| bias[channel]),
                    )
                })?;
        }
        Ok((CpuStorage::F32(output), self.output_shape.clone()))
    }

    fn pack_kernel(&self, kernel: &[f32], kernel_size: usize) -> Vec<f32> {
        let mut packed = vec![0.0_f32; self.output_channels * kernel_size];
        for output_channel in 0..self.output_channels {
            let output = &mut packed[output_channel * kernel_size..][..kernel_size];
            for kernel_y in 0..self.kernel_height {
                for kernel_x in 0..self.kernel_width {
                    let packed_base =
                        (kernel_y * self.kernel_width + kernel_x) * self.input_channels;
                    for input_channel in 0..self.input_channels {
                        let source = ((output_channel * self.input_channels + input_channel)
                            * self.kernel_height
                            + kernel_y)
                            * self.kernel_width
                            + kernel_x;
                        output[packed_base + input_channel] = kernel[source];
                    }
                }
            }
        }
        packed
    }
    fn extract_columns(
        &self,
        input: &[f32],
        tile_start: usize,
        tile_len: usize,
        columns: &mut [f32],
        source_offsets: &mut [usize],
    ) {
        let input_spatial = self.input_height * self.input_width;
        for kernel_y in 0..self.kernel_height {
            for kernel_x in 0..self.kernel_width {
                for (local, source_offset) in source_offsets.iter_mut().enumerate() {
                    let output = tile_start + local;
                    let output_y = output / self.output_width;
                    let output_x = output % self.output_width;
                    let padded_y = output_y * self.stride + kernel_y * self.dilation;
                    let padded_x = output_x * self.stride + kernel_x * self.dilation;
                    *source_offset = padded_y
                        .checked_sub(self.pads.0)
                        .filter(|input_y| *input_y < self.input_height)
                        .and_then(|input_y| {
                            padded_x
                                .checked_sub(self.pads.1)
                                .filter(|input_x| *input_x < self.input_width)
                                .map(|input_x| input_y * self.input_width + input_x)
                        })
                        .unwrap_or(usize::MAX);
                }
                let kernel_base = (kernel_y * self.kernel_width + kernel_x) * self.input_channels;
                for input_channel in 0..self.input_channels {
                    let output =
                        &mut columns[(kernel_base + input_channel) * tile_len..][..tile_len];
                    let input = &input[input_channel * input_spatial..][..input_spatial];
                    for (value, source_offset) in output.iter_mut().zip(source_offsets.iter()) {
                        if *source_offset != usize::MAX {
                            *value = input[*source_offset];
                        }
                    }
                }
            }
        }
    }
}

#[derive(Default)]
struct SpatialScratch {
    columns: Vec<f32>,
    source_offsets: Vec<usize>,
}

impl SpatialScratch {
    fn prepare(&mut self, column_elements: usize, tile_len: usize) {
        self.columns.resize(column_elements, 0.0);
        self.columns.fill(0.0);
        self.source_offsets.resize(tile_len, usize::MAX);
    }
}

fn effective_kernel(size: usize, dilation: usize, axis: &str) -> Result<usize> {
    dilation
        .checked_mul(size - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| spatial_error(format!("spatial convolution kernel {axis} overflowed")))
}

fn output_dimension(
    padded: usize,
    effective_kernel: usize,
    stride: usize,
    axis: &str,
) -> Result<usize> {
    padded
        .checked_sub(effective_kernel)
        .map(|remaining| remaining / stride + 1)
        .ok_or_else(|| spatial_error(format!("spatial convolution kernel exceeds input {axis}")))
}

fn spatial_error(message: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(message.into())
}

impl CustomOp2 for SpatialConv2d {
    fn name(&self) -> &'static str {
        "a3s-direct-cpu-spatial-convolution"
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
            self.batch * self.input_channels * self.input_height * self.input_width,
            "spatial convolution input",
        )?;
        let kernel = contiguous_values(
            kernel.as_slice::<f32>()?,
            kernel_layout,
            self.output_channels * self.input_channels * self.kernel_height * self.kernel_width,
            "spatial convolution kernel",
        )?;
        self.execute(input, kernel, None)
    }
}

impl CustomOp3 for SpatialConv2d {
    fn name(&self) -> &'static str {
        "a3s-direct-cpu-biased-spatial-convolution"
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
            self.batch * self.input_channels * self.input_height * self.input_width,
            "spatial convolution input",
        )?;
        let kernel = contiguous_values(
            kernel.as_slice::<f32>()?,
            kernel_layout,
            self.output_channels * self.input_channels * self.kernel_height * self.kernel_width,
            "spatial convolution kernel",
        )?;
        let bias = contiguous_values(
            bias.as_slice::<f32>()?,
            bias_layout,
            self.output_channels,
            "spatial convolution bias",
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
        .ok_or_else(|| spatial_error(format!("{label} layout is out of bounds")))
}
