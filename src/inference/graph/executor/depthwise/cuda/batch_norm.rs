use candle_core::backend::BackendStorage;
use candle_core::{
    CpuStorage, CudaStorage, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Storage, Tensor,
};

use super::super::super::convolution_post::{
    ConvolutionPostOperation, CudaBatchNormLaunchParameters,
};
use super::{as_u64, kernels, DepthwiseConv2d};

const MODULE_NAME: &str = "a3s_power_depthwise_batch_norm_f32_v4";
const WITHOUT_BIAS: &str = "depthwise_conv2d_batch_norm_contiguous_u32_f32";
const WITH_BIAS: &str = "depthwise_conv2d_bias_batch_norm_contiguous_u32_f32";
const THREADS_PER_BLOCK: u32 = 512;

pub(super) fn try_conv2d(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    pads: (usize, usize, usize, usize),
    strides: (usize, usize),
    dilation: usize,
    post_operation: ConvolutionPostOperation,
) -> Result<Option<Tensor>> {
    let Some(batch_norm) = post_operation.cuda_batch_normalization_parameters() else {
        return Ok(None);
    };
    if input.dtype() != DType::F32
        || kernel.dtype() != DType::F32
        || bias.is_some_and(|bias| bias.dtype() != DType::F32)
        || !input.device().is_cuda()
        || !kernel.device().same_device(input.device())
        || bias.is_some_and(|bias| !bias.device().same_device(input.device()))
        || !batch_norm
            .scale_and_bias
            .device()
            .same_device(input.device())
        || !batch_norm
            .mean_and_stddev
            .device()
            .same_device(input.device())
    {
        return Ok(None);
    }

    let convolution =
        DepthwiseConv2d::new(input.layout(), kernel.layout(), pads, strides, dilation)?;
    let Some(launch_parameters) = convolution.contiguous_u32_parameters(
        input.layout(),
        kernel.layout(),
        bias.map(Tensor::layout),
    ) else {
        return Ok(None);
    };
    if batch_norm.scale_and_bias.dims2()? != (2, convolution.channels as usize)
        || batch_norm.mean_and_stddev.dims2()? != (2, convolution.channels as usize)
        || !batch_norm.scale_and_bias.is_contiguous()
        || !batch_norm.mean_and_stddev.is_contiguous()
    {
        return Ok(None);
    }
    if let Some(bias) = bias {
        convolution.validate_bias(bias.layout())?;
    }

    let operation = DepthwiseBatchNorm {
        convolution,
        launch_parameters,
        scale_and_bias: batch_norm.scale_and_bias.clone(),
        mean_and_stddev: batch_norm.mean_and_stddev.clone(),
        activation: batch_norm.activation.launch_parameters(),
    };
    match bias {
        Some(bias) => input.apply_op3_no_bwd(kernel, bias, &operation).map(Some),
        None => input.apply_op2_no_bwd(kernel, &operation).map(Some),
    }
}

#[derive(Clone)]
struct DepthwiseBatchNorm {
    convolution: DepthwiseConv2d,
    launch_parameters: [u32; 13],
    scale_and_bias: Tensor,
    mean_and_stddev: Tensor,
    activation: CudaBatchNormLaunchParameters,
}

impl DepthwiseBatchNorm {
    fn launch(
        &self,
        input: &CudaStorage,
        kernel: &CudaStorage,
        bias: Option<(&CudaStorage, &Layout)>,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        if let Some((_, layout)) = bias {
            self.convolution.validate_bias(layout)?;
        }
        let (scale_and_bias_storage, scale_and_bias_layout) =
            self.scale_and_bias.storage_and_layout();
        let Storage::Cuda(scale_and_bias) = &*scale_and_bias_storage else {
            candle_core::bail!(
                "fused depthwise BatchNormalization parameters are not CUDA-resident"
            )
        };
        let (mean_and_stddev_storage, mean_and_stddev_layout) =
            self.mean_and_stddev.storage_and_layout();
        let Storage::Cuda(mean_and_stddev) = &*mean_and_stddev_storage else {
            candle_core::bail!(
                "fused depthwise BatchNormalization statistics are not CUDA-resident"
            )
        };

        let device = input.device();
        let function_name = if bias.is_some() {
            WITH_BIAS
        } else {
            WITHOUT_BIAS
        };
        let function = device.get_or_load_custom_func(
            function_name,
            MODULE_NAME,
            kernels::DEPTHWISE_BATCH_NORM,
        )?;
        let output = unsafe { device.alloc::<f32>(self.convolution.output_elements)? };
        let mut builder = function.builder();
        builder.arg(input.as_cuda_slice::<f32>()?);
        builder.arg(kernel.as_cuda_slice::<f32>()?);
        if let Some((bias, _)) = bias {
            builder.arg(bias.as_cuda_slice::<f32>()?);
        }
        builder.arg(scale_and_bias.as_cuda_slice::<f32>()?);
        builder.arg(mean_and_stddev.as_cuda_slice::<f32>()?);
        builder.arg(&output);
        for value in &self.launch_parameters {
            builder.arg(value);
        }
        let fast_divisors = self
            .convolution
            .fast_divisor_parameters()
            .ok_or_else(|| candle_core::Error::Msg("depthwise index divisor exceeds u32".into()))?;
        for value in &fast_divisors {
            builder.arg(value);
        }
        let scale_and_bias_offset = as_u64(scale_and_bias_layout.start_offset())?;
        let mean_and_stddev_offset = as_u64(mean_and_stddev_layout.start_offset())?;
        builder.arg(&scale_and_bias_offset);
        builder.arg(&mean_and_stddev_offset);
        builder.arg(&self.activation.kind);
        builder.arg(&self.activation.first);
        builder.arg(&self.activation.second);
        builder.arg(&self.activation.third);

        let output_elements = u32::try_from(self.convolution.output_elements)
            .map_err(|_| candle_core::Error::Msg("depthwise launch size exceeds u32".into()))?;
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (output_elements.div_ceil(THREADS_PER_BLOCK), 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            self.convolution.output_shape(),
        ))
    }
}

impl CustomOp2 for DepthwiseBatchNorm {
    fn name(&self) -> &'static str {
        "a3s-fused-depthwise-conv2d-batch-normalization"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _kernel: &CpuStorage,
        _kernel_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused depthwise BatchNormalization operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        _input_layout: &Layout,
        kernel: &CudaStorage,
        _kernel_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch(input, kernel, None)
    }
}

impl CustomOp3 for DepthwiseBatchNorm {
    fn name(&self) -> &'static str {
        "a3s-fused-depthwise-conv2d-bias-batch-normalization"
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
        candle_core::bail!("the fused depthwise bias BatchNormalization operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        _input_layout: &Layout,
        kernel: &CudaStorage,
        _kernel_layout: &Layout,
        bias: &CudaStorage,
        bias_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch(input, kernel, Some((bias, bias_layout)))
    }
}
