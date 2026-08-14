use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/depthwise_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_depthwise_f32_v1";
const WITHOUT_BIAS: &str = "depthwise_conv2d_f32";
const WITH_BIAS: &str = "depthwise_conv2d_bias_f32";

pub(super) fn conv2d(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    strides: (usize, usize),
    dilation: usize,
) -> Result<Tensor> {
    let operation = DepthwiseConv2d::new(input.layout(), kernel.layout(), strides, dilation)?;
    match bias {
        Some(bias) => input.apply_op3_no_bwd(kernel, bias, &operation),
        None => input.apply_op2_no_bwd(kernel, &operation),
    }
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
    output_elements: usize,
}

impl DepthwiseConv2d {
    fn new(
        input: &Layout,
        kernel: &Layout,
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
        let output_height = input_height
            .checked_sub(effective_height)
            .map(|remaining| remaining / strides.0 + 1)
            .ok_or_else(|| {
                candle_core::Error::Msg("depthwise kernel exceeds input height".into())
            })?;
        let output_width = input_width
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

    fn launch(
        self,
        input: &CudaStorage,
        input_layout: &Layout,
        kernel: &CudaStorage,
        kernel_layout: &Layout,
        bias: Option<(&CudaStorage, &Layout)>,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

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
        let output = unsafe { device.alloc::<f32>(self.output_elements)? };
        let mut builder = function.builder();
        builder.arg(input_values);
        builder.arg(kernel_values);
        let bias_layout;
        if let Some((bias, layout)) = bias {
            self.validate_bias(layout)?;
            builder.arg(bias.as_cuda_slice::<f32>()?);
            bias_layout = [as_u64(layout.start_offset())?, as_u64(layout.stride()[0])?];
        } else {
            bias_layout = [0, 0];
        }
        builder.arg(&output);
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
        unsafe { builder.launch(LaunchConfig::for_num_elems(output_elements)) }.w()?;
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
