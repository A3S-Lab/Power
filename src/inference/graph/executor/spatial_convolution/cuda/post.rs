use candle_core::backend::BackendStorage;
#[cfg(test)]
use candle_core::cuda_backend::cudarc::driver::CudaViewMut;
#[cfg(test)]
use candle_core::InplaceOp2;
use candle_core::{
    CpuStorage, CudaStorage, CustomOp1, CustomOp2, Layout, Result, Shape, Storage, Tensor,
};

use super::super::super::convolution_post::{
    ConvolutionPostOperation, CudaBatchNormLaunchParameters,
};
use super::{kernels, Im2Col, MODULE_NAME};

const WITHOUT_BIAS: &str = "nhwc_to_nchw_batch_norm_f32";
const WITH_BIAS: &str = "nhwc_to_nchw_bias_batch_norm_f32";
#[cfg(test)]
const ACTIVATION_WITHOUT_BIAS: &str = "nhwc_to_nchw_activation_f32";
#[cfg(test)]
const ACTIVATION_WITH_BIAS: &str = "nhwc_to_nchw_bias_activation_f32";

pub(super) fn execute(
    product: &Tensor,
    convolution_bias: Option<&Tensor>,
    lowering: Im2Col,
    post_operation: ConvolutionPostOperation,
) -> Result<Tensor> {
    match post_operation {
        #[cfg(test)]
        ConvolutionPostOperation::Relu => execute_activation(
            product,
            convolution_bias,
            lowering,
            SpatialActivationKind::Relu,
        ),
        #[cfg(test)]
        ConvolutionPostOperation::GeluErf {
            divisor,
            offset,
            scale,
        } => execute_activation(
            product,
            convolution_bias,
            lowering,
            SpatialActivationKind::GeluErf {
                divisor,
                offset,
                scale,
            },
        ),
        operation @ ConvolutionPostOperation::CudaBatchNormalization(_) => {
            execute_batch_norm(product, convolution_bias, lowering, operation)
        }
        _ => candle_core::bail!("unsupported CUDA spatial convolution post-operation"),
    }
}

fn execute_batch_norm(
    product: &Tensor,
    convolution_bias: Option<&Tensor>,
    lowering: Im2Col,
    post_operation: ConvolutionPostOperation,
) -> Result<Tensor> {
    let Some(parameters) = post_operation.cuda_batch_normalization_parameters() else {
        candle_core::bail!(
            "CUDA spatial convolution only fuses a validated BatchNormalization post-operation"
        )
    };
    if !parameters
        .scale_and_bias
        .device()
        .same_device(product.device())
        || !parameters
            .mean_and_stddev
            .device()
            .same_device(product.device())
        || convolution_bias.is_some_and(|bias| !bias.device().same_device(product.device()))
    {
        candle_core::bail!("CUDA spatial post-operation requires co-located tensors")
    }
    let operation = SpatialBatchNorm::new(
        product.layout(),
        convolution_bias.map(Tensor::layout),
        &parameters.scale_and_bias,
        &parameters.mean_and_stddev,
        lowering,
        parameters.activation.launch_parameters(),
    )?;
    match convolution_bias {
        Some(bias) => product.apply_op2_no_bwd(bias, &operation),
        None => product.apply_op1_no_bwd(&operation),
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum SpatialActivationKind {
    Relu,
    GeluErf {
        divisor: f32,
        offset: f32,
        scale: f32,
    },
}

#[cfg(test)]
fn execute_activation(
    product: &Tensor,
    convolution_bias: Option<&Tensor>,
    lowering: Im2Col,
    activation: SpatialActivationKind,
) -> Result<Tensor> {
    if convolution_bias.is_some_and(|bias| !bias.device().same_device(product.device())) {
        candle_core::bail!("CUDA spatial activation requires co-located tensors")
    }
    let operation = SpatialActivation::new(
        product.layout(),
        convolution_bias.map(Tensor::layout),
        lowering,
        activation,
    )?;
    match convolution_bias {
        Some(bias) => product.apply_op2_no_bwd(bias, &operation),
        None => product.apply_op1_no_bwd(&operation),
    }
}

#[cfg(test)]
pub(super) fn relu_into(output: &Tensor, product: &Tensor, lowering: Im2Col) -> Result<()> {
    let operation = SpatialActivation::new(
        product.layout(),
        None,
        lowering,
        SpatialActivationKind::Relu,
    )?;
    output.inplace_op2(product, &operation)
}

#[cfg(test)]
#[derive(Clone)]
struct SpatialActivation {
    output_shape: Shape,
    elements: usize,
    channels: u64,
    spatial: u64,
    activation: SpatialActivationKind,
}

#[cfg(test)]
impl SpatialActivation {
    fn new(
        product: &Layout,
        convolution_bias: Option<&Layout>,
        lowering: Im2Col,
        activation: SpatialActivationKind,
    ) -> Result<Self> {
        let spatial = lowering
            .output_height
            .checked_mul(lowering.output_width)
            .ok_or_else(|| dimension_error("activation spatial size overflowed"))?;
        let rows = lowering
            .batch
            .checked_mul(spatial)
            .ok_or_else(|| dimension_error("activation row count overflowed"))?;
        let elements = rows
            .checked_mul(lowering.output_channels)
            .ok_or_else(|| dimension_error("activation element count overflowed"))?;
        let finite_activation = match activation {
            SpatialActivationKind::Relu => true,
            SpatialActivationKind::GeluErf {
                divisor,
                offset,
                scale,
            } => divisor.is_finite() && divisor != 0.0 && offset.is_finite() && scale.is_finite(),
        };
        if !product.is_contiguous()
            || product.shape().dims2()? != (rows, lowering.output_channels)
            || convolution_bias.is_some_and(|layout| {
                !layout.is_contiguous()
                    || layout.shape().dims1().ok() != Some(lowering.output_channels)
            })
            || !finite_activation
            || elements == 0
            || u32::try_from(elements).is_err()
        {
            candle_core::bail!("CUDA spatial activation requires exact contiguous finite geometry")
        }
        Ok(Self {
            output_shape: Shape::from_dims(&[
                lowering.batch,
                lowering.output_channels,
                lowering.output_height,
                lowering.output_width,
            ]),
            elements,
            channels: as_u64(lowering.output_channels)?,
            spatial: as_u64(spatial)?,
            activation,
        })
    }

    fn launch_into(
        &self,
        output: &mut CudaViewMut<'_, f32>,
        product: &CudaStorage,
        product_layout: &Layout,
        convolution_bias: Option<(&CudaStorage, &Layout)>,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        if output.len() != self.elements {
            candle_core::bail!("CUDA spatial activation destination has the wrong length")
        }
        let device = product.device();
        let function_name = if convolution_bias.is_some() {
            ACTIVATION_WITH_BIAS
        } else {
            ACTIVATION_WITHOUT_BIAS
        };
        let function =
            device.get_or_load_custom_func(function_name, MODULE_NAME, kernels::SPATIAL_IM2COL)?;
        let product_offset = as_u64(product_layout.start_offset())?;
        let bias_offset = convolution_bias
            .map(|(_, layout)| as_u64(layout.start_offset()))
            .transpose()?;
        let (activation, divisor, offset, scale) = match self.activation {
            SpatialActivationKind::Relu => (0_u32, 1.0_f32, 0.0_f32, 1.0_f32),
            SpatialActivationKind::GeluErf {
                divisor,
                offset,
                scale,
            } => (1_u32, divisor, offset, scale),
        };
        let element_count = as_u64(self.elements)?;
        let mut builder = function.builder();
        builder.arg(product.as_cuda_slice::<f32>()?);
        if let Some((bias, _)) = convolution_bias {
            builder.arg(bias.as_cuda_slice::<f32>()?);
        }
        builder.arg(output);
        builder.arg(&product_offset);
        if let Some(bias_offset) = bias_offset.as_ref() {
            builder.arg(bias_offset);
        }
        builder.arg(&element_count);
        builder.arg(&self.channels);
        builder.arg(&self.spatial);
        builder.arg(&activation);
        builder.arg(&divisor);
        builder.arg(&offset);
        builder.arg(&scale);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(self.elements as u32))
                .w()?
        };
        Ok(())
    }

    fn launch(
        &self,
        product: &CudaStorage,
        product_layout: &Layout,
        convolution_bias: Option<(&CudaStorage, &Layout)>,
    ) -> Result<(CudaStorage, Shape)> {
        let device = product.device();
        let mut output = unsafe { device.alloc::<f32>(self.elements)? };
        {
            let mut destination = output.slice_mut(..);
            self.launch_into(&mut destination, product, product_layout, convolution_bias)?;
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            self.output_shape.clone(),
        ))
    }
}

#[cfg(test)]
impl CustomOp1 for SpatialActivation {
    fn name(&self) -> &'static str {
        "a3s-nhwc-to-nchw-activation"
    }

    fn cpu_fwd(&self, _input: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the spatial activation operation is CUDA-only")
    }

    fn cuda_fwd(&self, input: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        self.launch(input, layout, None)
    }
}

#[cfg(test)]
impl CustomOp2 for SpatialActivation {
    fn name(&self) -> &'static str {
        "a3s-nhwc-to-nchw-bias-activation"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _bias: &CpuStorage,
        _bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the spatial bias activation operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        bias: &CudaStorage,
        bias_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch(input, input_layout, Some((bias, bias_layout)))
    }
}

#[cfg(test)]
impl InplaceOp2 for SpatialActivation {
    fn name(&self) -> &'static str {
        "a3s-nhwc-to-nchw-activation-into"
    }

    fn cpu_fwd(
        &self,
        _output: &mut CpuStorage,
        _output_layout: &Layout,
        _input: &CpuStorage,
        _input_layout: &Layout,
    ) -> Result<()> {
        candle_core::bail!("the preallocated spatial activation operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        output: &mut CudaStorage,
        output_layout: &Layout,
        input: &CudaStorage,
        input_layout: &Layout,
    ) -> Result<()> {
        if !output_layout.is_contiguous()
            || output_layout.start_offset() != 0
            || output_layout.shape() != &self.output_shape
        {
            candle_core::bail!(
                "spatial activation destination must be one exact contiguous output tensor"
            )
        }
        let mut destination = output
            .as_cuda_slice_mut::<f32>()?
            .slice_mut(..self.elements);
        self.launch_into(&mut destination, input, input_layout, None)
    }
}

#[derive(Clone)]
struct SpatialBatchNorm {
    scale_and_bias: Tensor,
    mean_and_stddev: Tensor,
    batch: u64,
    channels: u64,
    spatial: u64,
    output_height: usize,
    output_width: usize,
    elements: usize,
    activation: CudaBatchNormLaunchParameters,
}

impl SpatialBatchNorm {
    #[allow(clippy::too_many_arguments)]
    fn new(
        product: &Layout,
        convolution_bias: Option<&Layout>,
        scale_and_bias: &Tensor,
        mean_and_stddev: &Tensor,
        lowering: Im2Col,
        activation: CudaBatchNormLaunchParameters,
    ) -> Result<Self> {
        let spatial = lowering
            .output_height
            .checked_mul(lowering.output_width)
            .ok_or_else(|| dimension_error("spatial size overflowed"))?;
        let rows = lowering
            .batch
            .checked_mul(spatial)
            .ok_or_else(|| dimension_error("row count overflowed"))?;
        let elements = rows
            .checked_mul(lowering.output_channels)
            .ok_or_else(|| dimension_error("element count overflowed"))?;
        if !product.is_contiguous()
            || product.shape().dims2()? != (rows, lowering.output_channels)
            || !scale_and_bias.is_contiguous()
            || !mean_and_stddev.is_contiguous()
            || scale_and_bias.dims2()? != (2, lowering.output_channels)
            || mean_and_stddev.dims2()? != (2, lowering.output_channels)
            || convolution_bias.is_some_and(|layout| {
                !layout.is_contiguous()
                    || layout.shape().dims1().ok() != Some(lowering.output_channels)
            })
            || elements == 0
            || u32::try_from(elements).is_err()
        {
            candle_core::bail!(
                "CUDA spatial BatchNormalization requires exact contiguous finite geometry"
            )
        }
        Ok(Self {
            scale_and_bias: scale_and_bias.clone(),
            mean_and_stddev: mean_and_stddev.clone(),
            batch: as_u64(lowering.batch)?,
            channels: as_u64(lowering.output_channels)?,
            spatial: as_u64(spatial)?,
            output_height: lowering.output_height,
            output_width: lowering.output_width,
            elements,
            activation,
        })
    }

    fn launch(
        &self,
        product: &CudaStorage,
        product_layout: &Layout,
        convolution_bias: Option<(&CudaStorage, &Layout)>,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let (scale_and_bias_storage, scale_and_bias_layout) =
            self.scale_and_bias.storage_and_layout();
        let Storage::Cuda(scale_and_bias) = &*scale_and_bias_storage else {
            candle_core::bail!("spatial BatchNormalization parameters are not CUDA-resident")
        };
        let (mean_and_stddev_storage, mean_and_stddev_layout) =
            self.mean_and_stddev.storage_and_layout();
        let Storage::Cuda(mean_and_stddev) = &*mean_and_stddev_storage else {
            candle_core::bail!("spatial BatchNormalization statistics are not CUDA-resident")
        };

        let device = product.device();
        let function_name = if convolution_bias.is_some() {
            WITH_BIAS
        } else {
            WITHOUT_BIAS
        };
        let function =
            device.get_or_load_custom_func(function_name, MODULE_NAME, kernels::SPATIAL_IM2COL)?;
        let output = unsafe { device.alloc::<f32>(self.elements)? };
        let mut builder = function.builder();
        builder.arg(product.as_cuda_slice::<f32>()?);
        if let Some((bias, _)) = convolution_bias {
            builder.arg(bias.as_cuda_slice::<f32>()?);
        }
        builder.arg(scale_and_bias.as_cuda_slice::<f32>()?);
        builder.arg(mean_and_stddev.as_cuda_slice::<f32>()?);
        builder.arg(&output);
        let product_offset = as_u64(product_layout.start_offset())?;
        let bias_offset = convolution_bias
            .map(|(_, layout)| as_u64(layout.start_offset()))
            .transpose()?;
        builder.arg(&product_offset);
        if let Some(bias_offset) = bias_offset.as_ref() {
            builder.arg(bias_offset);
        }
        let scale_and_bias_offset = as_u64(scale_and_bias_layout.start_offset())?;
        let mean_and_stddev_offset = as_u64(mean_and_stddev_layout.start_offset())?;
        builder.arg(&scale_and_bias_offset);
        builder.arg(&mean_and_stddev_offset);
        builder.arg(&self.batch);
        builder.arg(&self.channels);
        builder.arg(&self.spatial);
        builder.arg(&self.activation.kind);
        builder.arg(&self.activation.first);
        builder.arg(&self.activation.second);
        builder.arg(&self.activation.third);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(self.elements as u32))
                .w()?
        };
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from_dims(&[
                self.batch as usize,
                self.channels as usize,
                self.output_height,
                self.output_width,
            ]),
        ))
    }
}

impl CustomOp1 for SpatialBatchNorm {
    fn name(&self) -> &'static str {
        "a3s-nhwc-to-nchw-batch-normalization"
    }

    fn cpu_fwd(&self, _input: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the spatial BatchNormalization operation is CUDA-only")
    }

    fn cuda_fwd(&self, input: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        self.launch(input, layout, None)
    }
}

impl CustomOp2 for SpatialBatchNorm {
    fn name(&self) -> &'static str {
        "a3s-nhwc-to-nchw-bias-batch-normalization"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _bias: &CpuStorage,
        _bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the spatial bias BatchNormalization operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        bias: &CudaStorage,
        bias_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch(input, input_layout, Some((bias, bias_layout)))
    }
}

fn as_u64(value: usize) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| candle_core::Error::Msg("spatial dimension exceeds u64".into()))
}

fn dimension_error(message: &str) -> candle_core::Error {
    candle_core::Error::Msg(format!("CUDA spatial BatchNormalization {message}"))
}

#[cfg(test)]
mod tests;
