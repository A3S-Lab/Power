use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::CudaSlice;
#[cfg(test)]
use candle_core::InplaceOp3;
use candle_core::{CpuStorage, CudaStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};

use super::super::convolution_post::ConvolutionPostOperation;

mod batch_norm;
use super::super::cuda_fast_divisor::FastDivisorU32;

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/depthwise_ptx.rs"));
    include!(concat!(env!("OUT_DIR"), "/depthwise_batch_norm_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_depthwise_f32_v5";
const WITHOUT_BIAS: &str = "depthwise_conv2d_f32";
const WITH_BIAS: &str = "depthwise_conv2d_bias_f32";
const CONTIGUOUS_U32_WITHOUT_BIAS: &str = "depthwise_conv2d_contiguous_u32_f32";
const CONTIGUOUS_U32_WITH_BIAS: &str = "depthwise_conv2d_bias_contiguous_u32_f32";

pub(super) fn conv2d(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    pads: (usize, usize, usize, usize),
    strides: (usize, usize),
    dilation: usize,
) -> Result<Tensor> {
    let operation = DepthwiseConv2d::new(input.layout(), kernel.layout(), pads, strides, dilation)?;
    match bias {
        Some(bias) => input.apply_op3_no_bwd(kernel, bias, &operation),
        None => input.apply_op2_no_bwd(kernel, &operation),
    }
}

#[cfg(test)]
pub(super) fn conv2d_into(
    output: &Tensor,
    input: &Tensor,
    kernel: &Tensor,
    pads: (usize, usize, usize, usize),
    strides: (usize, usize),
    dilation: usize,
) -> Result<()> {
    let operation = DepthwiseConv2d::new(input.layout(), kernel.layout(), pads, strides, dilation)?;
    output.inplace_op3(input, kernel, &operation)
}

pub(super) fn try_conv2d_with_post_operation(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    pads: (usize, usize, usize, usize),
    strides: (usize, usize),
    dilation: usize,
    post_operation: ConvolutionPostOperation,
) -> Result<Option<Tensor>> {
    batch_norm::try_conv2d(input, kernel, bias, pads, strides, dilation, post_operation)
}

#[derive(Clone, Copy)]
struct DepthwiseConv2d {
    batch: u64,
    channels: u64,
    input_height: u64,
    input_width: u64,
    output_height: u64,
    output_width: u64,
    kernel_height: u64,
    kernel_width: u64,
    stride_height: u64,
    stride_width: u64,
    dilation: u64,
    pad_top: u64,
    pad_left: u64,
    output_elements: usize,
}

impl DepthwiseConv2d {
    fn new(
        input: &Layout,
        kernel: &Layout,
        pads: (usize, usize, usize, usize),
        strides: (usize, usize),
        dilation: usize,
    ) -> Result<Self> {
        let (batch, channels, input_height, input_width) = input.shape().dims4()?;
        let (output_channels, kernel_channels, kernel_height, kernel_width) =
            kernel.shape().dims4()?;
        if channels == 0
            || output_channels != channels
            || kernel_channels != 1
            || kernel_height == 0
            || kernel_width == 0
            || strides.0 == 0
            || strides.1 == 0
            || dilation == 0
        {
            candle_core::bail!(
                "fused CUDA depthwise convolution requires one non-empty kernel per input channel"
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
            .ok_or_else(|| candle_core::Error::Msg("depthwise padded height overflowed".into()))?;
        let padded_width = input_width
            .checked_add(pads.1)
            .and_then(|value| value.checked_add(pads.3))
            .ok_or_else(|| candle_core::Error::Msg("depthwise padded width overflowed".into()))?;
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
        let output_elements = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(output_height))
            .and_then(|value| value.checked_mul(output_width))
            .ok_or_else(|| candle_core::Error::Msg("depthwise output size overflowed".into()))?;
        if output_elements == 0 || u32::try_from(output_elements).is_err() {
            candle_core::bail!("fused CUDA depthwise output exceeds the reviewed launch bound")
        }

        Ok(Self {
            batch: as_u64(batch)?,
            channels: as_u64(channels)?,
            input_height: as_u64(input_height)?,
            input_width: as_u64(input_width)?,
            output_height: as_u64(output_height)?,
            output_width: as_u64(output_width)?,
            kernel_height: as_u64(kernel_height)?,
            kernel_width: as_u64(kernel_width)?,
            stride_height: as_u64(strides.0)?,
            stride_width: as_u64(strides.1)?,
            dilation: as_u64(dilation)?,
            pad_top: as_u64(pads.0)?,
            pad_left: as_u64(pads.1)?,
            output_elements,
        })
    }

    fn output_shape(self) -> Shape {
        Shape::from_dims(&[
            self.batch as usize,
            self.channels as usize,
            self.output_height as usize,
            self.output_width as usize,
        ])
    }

    fn validate_bias(self, layout: &Layout) -> Result<()> {
        if layout.shape().dims1()? != self.channels as usize {
            candle_core::bail!("fused CUDA depthwise bias channel count does not match the output")
        }
        Ok(())
    }

    fn contiguous_u32_parameters(
        self,
        input_layout: &Layout,
        kernel_layout: &Layout,
        bias_layout: Option<&Layout>,
    ) -> Option<[u32; 13]> {
        if !input_layout.is_contiguous()
            || !kernel_layout.is_contiguous()
            || input_layout.start_offset() != 0
            || kernel_layout.start_offset() != 0
            || bias_layout
                .is_some_and(|layout| !layout.is_contiguous() || layout.start_offset() != 0)
        {
            return None;
        }
        let input_elements = self
            .batch
            .checked_mul(self.channels)?
            .checked_mul(self.input_height)?
            .checked_mul(self.input_width)?;
        let kernel_elements = self
            .channels
            .checked_mul(self.kernel_height)?
            .checked_mul(self.kernel_width)?;
        let maximum_padded_y = self
            .output_height
            .checked_sub(1)?
            .checked_mul(self.stride_height)?
            .checked_add(
                self.kernel_height
                    .checked_sub(1)?
                    .checked_mul(self.dilation)?,
            )?;
        let maximum_padded_x = self
            .output_width
            .checked_sub(1)?
            .checked_mul(self.stride_width)?
            .checked_add(
                self.kernel_width
                    .checked_sub(1)?
                    .checked_mul(self.dilation)?,
            )?;
        if input_elements > u64::from(u32::MAX)
            || kernel_elements > u64::from(u32::MAX)
            || maximum_padded_y > u64::from(u32::MAX)
            || maximum_padded_x > u64::from(u32::MAX)
        {
            return None;
        }
        let source = [
            self.batch,
            self.channels,
            self.input_height,
            self.input_width,
            self.output_height,
            self.output_width,
            self.kernel_height,
            self.kernel_width,
            self.stride_height,
            self.stride_width,
            self.dilation,
            self.pad_top,
            self.pad_left,
        ];
        let mut parameters = [0_u32; 13];
        for (target, value) in parameters.iter_mut().zip(source) {
            *target = u32::try_from(value).ok()?;
        }
        Some(parameters)
    }

    fn fast_divisor_parameters(self) -> Option<[u32; 9]> {
        let spatial = self.output_height.checked_mul(self.output_width)?;
        let divisors = [self.output_width, spatial, self.channels];
        let mut parameters = [0_u32; 9];
        for (target, divisor) in parameters.chunks_exact_mut(3).zip(divisors) {
            let divisor = u32::try_from(divisor).ok()?;
            target.copy_from_slice(&FastDivisorU32::new(divisor)?.launch_parameters());
        }
        Some(parameters)
    }

    fn launch_contiguous_u32_into(
        self,
        output: &mut CudaSlice<f32>,
        input: &CudaStorage,
        kernel: &CudaStorage,
        bias: Option<&CudaStorage>,
        parameters: [u32; 13],
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        if output.len() != self.output_elements {
            candle_core::bail!("depthwise destination length does not match the output geometry")
        }
        let device = input.device();
        let function_name = if bias.is_some() {
            CONTIGUOUS_U32_WITH_BIAS
        } else {
            CONTIGUOUS_U32_WITHOUT_BIAS
        };
        let function =
            device.get_or_load_custom_func(function_name, MODULE_NAME, kernels::DEPTHWISE)?;
        let mut builder = function.builder();
        builder.arg(input.as_cuda_slice::<f32>()?);
        builder.arg(kernel.as_cuda_slice::<f32>()?);
        if let Some(bias) = bias {
            builder.arg(bias.as_cuda_slice::<f32>()?);
        }
        builder.arg(&*output);
        for value in &parameters {
            builder.arg(value);
        }
        let fast_divisors = self
            .fast_divisor_parameters()
            .ok_or_else(|| candle_core::Error::Msg("depthwise index divisor exceeds u32".into()))?;
        for value in &fast_divisors {
            builder.arg(value);
        }
        const THREADS_PER_BLOCK: u32 = 512;
        let output_elements = u32::try_from(self.output_elements)
            .map_err(|_| candle_core::Error::Msg("depthwise launch size exceeds u32".into()))?;
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (output_elements.div_ceil(THREADS_PER_BLOCK), 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok(())
    }

    fn launch_into(
        self,
        output: &mut CudaSlice<f32>,
        input: &CudaStorage,
        input_layout: &Layout,
        kernel: &CudaStorage,
        kernel_layout: &Layout,
        bias: Option<(&CudaStorage, &Layout)>,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        if let Some((_, layout)) = bias {
            self.validate_bias(layout)?;
        }
        if output.len() != self.output_elements {
            candle_core::bail!("depthwise destination length does not match the output geometry")
        }
        if let Some(parameters) = self.contiguous_u32_parameters(
            input_layout,
            kernel_layout,
            bias.map(|(_, layout)| layout),
        ) {
            return self.launch_contiguous_u32_into(
                output,
                input,
                kernel,
                bias.map(|(storage, _)| storage),
                parameters,
            );
        }

        let input_values = input.as_cuda_slice::<f32>()?;
        let kernel_values = kernel.as_cuda_slice::<f32>()?;
        let input_strides = four_strides(input_layout)?;
        let kernel_strides = four_strides(kernel_layout)?;
        let output_elements = u32::try_from(self.output_elements)
            .map_err(|_| candle_core::Error::Msg("depthwise launch size exceeds u32".into()))?;
        let device = input.device();
        let function_name = if bias.is_some() {
            WITH_BIAS
        } else {
            WITHOUT_BIAS
        };
        let function =
            device.get_or_load_custom_func(function_name, MODULE_NAME, kernels::DEPTHWISE)?;
        let mut builder = function.builder();
        builder.arg(input_values);
        builder.arg(kernel_values);
        let bias_layout;
        if let Some((bias, layout)) = bias {
            builder.arg(bias.as_cuda_slice::<f32>()?);
            bias_layout = [as_u64(layout.start_offset())?, as_u64(layout.stride()[0])?];
        } else {
            bias_layout = [0, 0];
        }
        builder.arg(&*output);
        let parameters = [
            self.batch,
            self.channels,
            self.input_height,
            self.input_width,
            self.output_height,
            self.output_width,
            self.kernel_height,
            self.kernel_width,
            self.stride_height,
            self.stride_width,
            self.dilation,
            self.pad_top,
            self.pad_left,
            as_u64(input_layout.start_offset())?,
            input_strides[0],
            input_strides[1],
            input_strides[2],
            input_strides[3],
            as_u64(kernel_layout.start_offset())?,
            kernel_strides[0],
            kernel_strides[2],
            kernel_strides[3],
            bias_layout[0],
            bias_layout[1],
        ];
        for value in &parameters {
            builder.arg(value);
        }
        // This kernel retains roughly forty registers per thread on current
        // CUDA toolchains. A 1,024-thread block therefore admits only one
        // resident block per SM. Three 512-thread blocks expose the same
        // element-wise work with enough independent warps to hide the address
        // and input-load latency; tensor geometry and arithmetic stay intact.
        const THREADS_PER_BLOCK: u32 = 512;
        let grid = output_elements.div_ceil(THREADS_PER_BLOCK);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok(())
    }

    fn launch(
        self,
        input: &CudaStorage,
        input_layout: &Layout,
        kernel: &CudaStorage,
        kernel_layout: &Layout,
        bias: Option<(&CudaStorage, &Layout)>,
    ) -> Result<(CudaStorage, Shape)> {
        let device = input.device();
        let mut output = unsafe { device.alloc::<f32>(self.output_elements)? };
        self.launch_into(
            &mut output,
            input,
            input_layout,
            kernel,
            kernel_layout,
            bias,
        )?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            self.output_shape(),
        ))
    }
}

impl CustomOp2 for DepthwiseConv2d {
    fn name(&self) -> &'static str {
        "a3s-fused-depthwise-conv2d"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _kernel: &CpuStorage,
        _kernel_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused depthwise operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        kernel: &CudaStorage,
        kernel_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch(input, input_layout, kernel, kernel_layout, None)
    }
}

impl CustomOp3 for DepthwiseConv2d {
    fn name(&self) -> &'static str {
        "a3s-fused-depthwise-conv2d-bias"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _kernel: &CpuStorage,
        _kernel_layout: &Layout,
        _bias: &CpuStorage,
        _bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused depthwise bias operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        kernel: &CudaStorage,
        kernel_layout: &Layout,
        bias: &CudaStorage,
        bias_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch(
            input,
            input_layout,
            kernel,
            kernel_layout,
            Some((bias, bias_layout)),
        )
    }
}

#[cfg(test)]
impl InplaceOp3 for DepthwiseConv2d {
    fn name(&self) -> &'static str {
        "a3s-fused-depthwise-conv2d-into"
    }

    fn cpu_fwd(
        &self,
        _output: &mut CpuStorage,
        _output_layout: &Layout,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _kernel: &CpuStorage,
        _kernel_layout: &Layout,
    ) -> Result<()> {
        candle_core::bail!("the preallocated depthwise operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        output: &mut CudaStorage,
        output_layout: &Layout,
        input: &CudaStorage,
        input_layout: &Layout,
        kernel: &CudaStorage,
        kernel_layout: &Layout,
    ) -> Result<()> {
        if !output_layout.is_contiguous()
            || output_layout.start_offset() != 0
            || output_layout.shape() != &self.output_shape()
        {
            candle_core::bail!("depthwise destination must be one exact contiguous output tensor")
        }
        self.launch_into(
            output.as_cuda_slice_mut::<f32>()?,
            input,
            input_layout,
            kernel,
            kernel_layout,
            None,
        )
    }
}

fn four_strides(layout: &Layout) -> Result<[u64; 4]> {
    let [first, second, third, fourth] = layout.stride() else {
        candle_core::bail!("fused CUDA depthwise tensors must have rank four")
    };
    Ok([
        as_u64(*first)?,
        as_u64(*second)?,
        as_u64(*third)?,
        as_u64(*fourth)?,
    ])
}

fn as_u64(value: usize) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| candle_core::Error::Msg("depthwise dimension exceeds u64".into()))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Layout, Tensor};

    use super::*;

    #[test]
    fn contiguous_u32_route_requires_exactly_addressable_dense_layouts() {
        let input = Layout::contiguous((2, 3, 5, 7));
        let kernel = Layout::contiguous((3, 1, 3, 3));
        let bias = Layout::contiguous(3);
        let operation = DepthwiseConv2d::new(&input, &kernel, (1, 1, 1, 1), (1, 1), 1).unwrap();

        let parameters = operation
            .contiguous_u32_parameters(&input, &kernel, Some(&bias))
            .unwrap();
        assert_eq!(parameters, [2, 3, 5, 7, 5, 7, 3, 3, 1, 1, 1, 1, 1]);

        let offset_input = Layout::contiguous_with_offset((2, 3, 5, 7), 1);
        assert!(operation
            .contiguous_u32_parameters(&offset_input, &kernel, Some(&bias))
            .is_none());

        let oversized = DepthwiseConv2d {
            input_width: u64::from(u32::MAX),
            ..operation
        };
        assert!(oversized
            .contiguous_u32_parameters(&input, &kernel, Some(&bias))
            .is_none());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn contiguous_u32_cuda_depthwise_matches_the_scalar_f32_formula() {
        let device = Device::new_cuda(0).unwrap();
        let input_values = (0..2 * 3 * 5 * 7)
            .map(|value| ((value * 17 % 101) as f32 - 50.0) / 53.0)
            .collect::<Vec<_>>();
        let kernel_values = (0..3 * 3 * 3)
            .map(|value| ((value * 13 % 29) as f32 - 14.0) / 31.0)
            .collect::<Vec<_>>();
        let bias_values = [-0.25_f32, 0.125, 0.375];
        let input = Tensor::from_vec(input_values.clone(), (2, 3, 5, 7), &device).unwrap();
        let kernel = Tensor::from_vec(kernel_values.clone(), (3, 1, 3, 3), &device).unwrap();
        let bias = Tensor::from_slice(&bias_values, 3, &device).unwrap();

        let actual = conv2d(&input, &kernel, Some(&bias), (1, 1, 1, 1), (1, 1), 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut expected = Vec::with_capacity(actual.len());
        for batch in 0..2 {
            for channel in 0..3 {
                for output_y in 0..5 {
                    for output_x in 0..7 {
                        let mut accumulator = 0.0_f32;
                        let mut first = true;
                        for kernel_y in 0..3 {
                            for kernel_x in 0..3 {
                                let padded_y = output_y + kernel_y;
                                let padded_x = output_x + kernel_x;
                                let input_value = if padded_y >= 1
                                    && padded_x >= 1
                                    && padded_y - 1 < 5
                                    && padded_x - 1 < 7
                                {
                                    input_values[((batch * 3 + channel) * 5 + padded_y - 1) * 7
                                        + padded_x
                                        - 1]
                                } else {
                                    0.0
                                };
                                let product = input_value
                                    * kernel_values[(channel * 3 + kernel_y) * 3 + kernel_x];
                                if first {
                                    accumulator = product;
                                    first = false;
                                } else {
                                    accumulator += product;
                                }
                            }
                        }
                        expected.push(accumulator + bias_values[channel]);
                    }
                }
            }
        }
        let maximum_absolute_difference = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(maximum_absolute_difference <= 1.0e-6);
    }
}
