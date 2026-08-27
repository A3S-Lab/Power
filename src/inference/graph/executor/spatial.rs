use candle_core::{DType, Device, Tensor};

use crate::error::Result;

use super::super::plan::GraphNode;
use super::super::value::GraphValue;
use super::convolution_post::ConvolutionPostOperation;
use super::tensor_geometry::{
    convolution_pads, execution_error, pad_spatial, pair, pool_pads, positive_usize, quad,
    subsample_spatial,
};
use super::{depthwise, max_pool, pointwise_convolution, required_tensor, spatial_convolution};

pub(super) fn conv(
    node: &GraphNode,
    inputs: &[&GraphValue],
    device: &Device,
) -> Result<GraphValue> {
    conv_internal(node, inputs, device, None, None)?.ok_or_else(|| {
        execution_error(
            node,
            "ordinary convolution execution unexpectedly declined its graph geometry",
        )
    })
}

pub(super) fn try_conv_with_post_operation(
    node: &GraphNode,
    inputs: &[&GraphValue],
    device: &Device,
    post_operation: ConvolutionPostOperation,
) -> Result<Option<GraphValue>> {
    conv_internal(node, inputs, device, Some(post_operation), None)
}

pub(super) fn try_conv_with_residual(
    node: &GraphNode,
    inputs: &[&GraphValue],
    device: &Device,
    residual: &Tensor,
) -> Result<Option<GraphValue>> {
    conv_internal(node, inputs, device, None, Some(residual))
}

fn conv_internal(
    node: &GraphNode,
    inputs: &[&GraphValue],
    device: &Device,
    convolution_post_operation: Option<ConvolutionPostOperation>,
    residual: Option<&Tensor>,
) -> Result<Option<GraphValue>> {
    if convolution_post_operation.is_some() && residual.is_some() {
        return Err(execution_error(
            node,
            "convolution cannot apply two fused post-operations",
        ));
    }
    let mut input = required_tensor(node, inputs, 0)?.clone();
    let kernel = required_tensor(node, inputs, 1)?;
    let kernel_shape = pair(&node.ints("kernel_shape", &[])?, "kernel_shape", node)?;
    let strides = pair(&node.ints("strides", &[1, 1])?, "strides", node)?;
    let dilations = pair(&node.ints("dilations", &[1, 1])?, "dilations", node)?;
    if dilations.0 != dilations.1 {
        return Err(execution_error(
            node,
            "mixed convolution dilation is unsupported",
        ));
    }
    let groups = positive_usize(node.int("group", 1)?, "group", node)?;
    let dimensions = input
        .dims4()
        .map_err(|error| execution_error(node, error))?;
    let kernel_dimensions = kernel
        .dims4()
        .map_err(|error| execution_error(node, error))?;
    let pads = convolution_pads(node, dimensions, kernel_shape, strides, dilations)?;
    let common_stride = if strides.0 == strides.1 { strides.0 } else { 1 };
    let bias = inputs
        .get(2)
        .map(|value| value.tensor(&node.name))
        .transpose()?
        .map(|bias| normalized_convolution_bias(node, bias))
        .transpose()?;
    let native_depthwise = (device.is_cpu() || device.is_cuda())
        && groups == dimensions.1
        && kernel_dimensions.0 == dimensions.1
        && kernel_dimensions.1 == 1
        && input.dtype() == DType::F32
        && kernel.dtype() == DType::F32
        && bias
            .as_ref()
            .is_none_or(|value| value.dtype() == DType::F32);
    let native_cpu_depthwise = native_depthwise && device.is_cpu();
    #[cfg(feature = "embedded-cuda")]
    let native_cuda_depthwise = native_depthwise && device.is_cuda();
    #[cfg(not(feature = "embedded-cuda"))]
    let native_cuda_depthwise = false;
    let native_cpu_pointwise = device.is_cpu()
        && groups == 1
        && kernel_dimensions.2 == 1
        && kernel_dimensions.3 == 1
        && strides == (1, 1)
        && dilations == (1, 1)
        && input.dtype() == DType::F32
        && kernel.dtype() == DType::F32
        && bias
            .as_ref()
            .is_none_or(|value| value.dtype() == DType::F32);
    let native_cuda_pointwise = device.is_cuda()
        && groups == 1
        && kernel_dimensions.2 == 1
        && kernel_dimensions.3 == 1
        && strides == (1, 1)
        && dilations == (1, 1)
        && pads == (0, 0, 0, 0)
        && input.dtype() == DType::F32
        && kernel.dtype() == DType::F32
        && input.is_contiguous()
        && kernel.is_contiguous()
        && bias.as_ref().is_none_or(|value| {
            value.dtype() == DType::F32
                && value.is_contiguous()
                && value.device().same_device(device)
        });
    let native_cpu_spatial = device.is_cpu()
        && groups == 1
        && kernel_dimensions.2 > 1
        && kernel_dimensions.3 > 1
        && strides.0 == strides.1
        && input.dtype() == DType::F32
        && kernel.dtype() == DType::F32
        && bias
            .as_ref()
            .is_none_or(|value| value.dtype() == DType::F32);
    #[cfg(feature = "embedded-cuda")]
    let native_cuda_spatial = device.is_cuda()
        && groups == 1
        && kernel_dimensions.2 > 1
        && kernel_dimensions.3 > 1
        && strides.0 == strides.1
        && input.dtype() == DType::F32
        && kernel.dtype() == DType::F32
        && input.is_contiguous()
        && kernel.is_contiguous()
        && bias.as_ref().is_none_or(|value| {
            value.dtype() == DType::F32
                && value.is_contiguous()
                && value.device().same_device(device)
        });
    #[cfg(not(feature = "embedded-cuda"))]
    let native_cuda_spatial = false;
    let native_cuda_spatial_post = native_cuda_spatial
        && convolution_post_operation
            .as_ref()
            .is_some_and(ConvolutionPostOperation::supports_cuda_spatial);
    // The low-level CUDA pointwise post-operation remains available to verify
    // exact arithmetic and cross-stream write ordering, but it is not selected
    // by graph execution: a guarded end-to-end A/B did not satisfy the
    // predeclared promotion gates. CUDA pointwise Conv-BatchNormalization thus
    // retains the ordinary two-output path for every graph and geometry.
    if convolution_post_operation.is_some()
        && !(native_cpu_depthwise
            || native_cuda_depthwise
            || native_cpu_pointwise
            || native_cpu_spatial
            || native_cuda_spatial_post)
    {
        return Ok(None);
    }
    if residual.is_some_and(|residual| {
        !native_cpu_pointwise
            || residual.dtype() != DType::F32
            || !residual.device().is_cpu()
            || !residual.device().same_device(device)
            || !residual.is_contiguous()
    }) {
        return Ok(None);
    }
    let native_pointwise = native_cpu_pointwise || native_cuda_pointwise;
    let native_convolution =
        native_depthwise || native_pointwise || native_cpu_spatial || native_cuda_spatial;
    let candle_padding = (!native_convolution)
        .then(|| symmetric_padding(pads))
        .flatten();
    if !native_depthwise && !native_cpu_spatial && !native_cuda_spatial && candle_padding.is_none()
    {
        input = pad_spatial(&input, pads, node)?;
    }
    let output = if native_depthwise {
        match convolution_post_operation {
            Some(post_operation) => {
                let Some(output) = depthwise::try_conv2d_with_post_operation(
                    &input,
                    kernel,
                    bias.as_ref(),
                    pads,
                    strides,
                    dilations.0,
                    post_operation,
                )
                .map_err(|error| execution_error(node, error))?
                else {
                    return Ok(None);
                };
                Ok(output)
            }
            None => depthwise::conv2d(&input, kernel, bias.as_ref(), pads, strides, dilations.0),
        }
    } else if native_pointwise {
        match (convolution_post_operation, residual) {
            (Some(post_operation), None) => pointwise_convolution::conv2d_with_post_operation(
                &input,
                kernel,
                bias.as_ref(),
                post_operation,
            ),
            (None, Some(residual)) => {
                pointwise_convolution::conv2d_with_residual(&input, kernel, bias.as_ref(), residual)
            }
            (None, None) => pointwise_convolution::conv2d(&input, kernel, bias.as_ref()),
            (Some(_), Some(_)) => Err(candle_core::Error::Msg(
                "convolution received conflicting fused post-operations".into(),
            )),
        }
    } else if native_cpu_spatial || native_cuda_spatial {
        match convolution_post_operation {
            Some(post_operation) => spatial_convolution::conv2d_with_post_operation(
                &input,
                kernel,
                bias.as_ref(),
                pads,
                strides.0,
                dilations.0,
                post_operation,
            ),
            None => spatial_convolution::conv2d(
                &input,
                kernel,
                bias.as_ref(),
                pads,
                strides.0,
                dilations.0,
            ),
        }
    } else {
        input.conv2d(
            kernel,
            candle_padding.unwrap_or_default(),
            common_stride,
            dilations.0,
            groups,
        )
    }
    .map_err(|error| execution_error(node, error))?;
    let mut output = if !native_convolution && strides.0 != strides.1 {
        subsample_spatial(&output, strides, device, node)?
    } else {
        output
    };
    if !native_convolution {
        if let Some(bias) = bias.as_ref() {
            let channels = bias.dims1().map_err(|error| execution_error(node, error))?;
            output = output
                .broadcast_add(
                    &bias
                        .reshape((1, channels, 1, 1))
                        .map_err(|error| execution_error(node, error))?,
                )
                .map_err(|error| execution_error(node, error))?;
        }
    }
    Ok(Some(GraphValue::Tensor(output)))
}

fn symmetric_padding(pads: (usize, usize, usize, usize)) -> Option<usize> {
    (pads.0 == pads.1 && pads.0 == pads.2 && pads.0 == pads.3).then_some(pads.0)
}

fn normalized_convolution_bias(node: &GraphNode, bias: &Tensor) -> Result<Tensor> {
    if bias.rank() == 1 {
        return Ok(bias.clone());
    }
    let dimensions = bias.dims4().map_err(|error| execution_error(node, error))?;
    if dimensions.0 != 1 || dimensions.2 != 1 || dimensions.3 != 1 {
        return Err(execution_error(
            node,
            "convolution bias must be one-dimensional or exact NCHW channel bias",
        ));
    }
    bias.reshape(dimensions.1)
        .map_err(|error| execution_error(node, error))
}

pub(super) fn conv_transpose(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    let kernel = required_tensor(node, inputs, 1)?;
    let strides = pair(&node.ints("strides", &[1, 1])?, "strides", node)?;
    let dilations = pair(&node.ints("dilations", &[1, 1])?, "dilations", node)?;
    let pads = quad(&node.ints("pads", &[0, 0, 0, 0])?, "pads", node)?;
    if strides.0 != strides.1
        || dilations.0 != dilations.1
        || pads.0 != pads.1
        || pads.0 != pads.2
        || pads.0 != pads.3
        || node.int("group", 1)? != 1
    {
        return Err(execution_error(
            node,
            "asymmetric or grouped transposed convolution is unsupported",
        ));
    }
    let mut output = input
        .conv_transpose2d(kernel, pads.0, 0, strides.0, dilations.0)
        .map_err(|error| execution_error(node, error))?;
    if let Some(bias) = inputs.get(2) {
        let bias = bias.tensor(&node.name)?;
        let channels = bias.dims1().map_err(|error| execution_error(node, error))?;
        output = output
            .broadcast_add(
                &bias
                    .reshape((1, channels, 1, 1))
                    .map_err(|error| execution_error(node, error))?,
            )
            .map_err(|error| execution_error(node, error))?;
    }
    Ok(GraphValue::Tensor(output))
}

pub(super) fn pool(node: &GraphNode, inputs: &[&GraphValue], maximum: bool) -> Result<GraphValue> {
    let mut input = required_tensor(node, inputs, 0)?.clone();
    let kernel = pair(&node.ints("kernel_shape", &[])?, "kernel_shape", node)?;
    let strides = pair(&node.ints("strides", &[1, 1])?, "strides", node)?;
    let dimensions = input
        .dims4()
        .map_err(|error| execution_error(node, error))?;
    let pads = pool_pads(node, dimensions, kernel, strides)?;
    if maximum && input.device().is_cpu() && input.dtype() == DType::F32 && input.is_contiguous() {
        return max_pool::execute(&input, kernel, strides, pads)
            .map(GraphValue::Tensor)
            .map_err(|error| execution_error(node, error));
    }
    input = pad_spatial(&input, pads, node)?;
    let output = if maximum {
        input.max_pool2d_with_stride(kernel, strides)
    } else {
        if node.int("count_include_pad", 0)? != 0 {
            return Err(execution_error(node, "count_include_pad is unsupported"));
        }
        input.avg_pool2d_with_stride(kernel, strides)
    }
    .map_err(|error| execution_error(node, error))?;
    Ok(GraphValue::Tensor(output))
}

pub(super) fn resize(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    let (_, _, height, width) = input
        .dims4()
        .map_err(|error| execution_error(node, error))?;
    let mode = node.string("mode", "nearest")?;
    if mode != "nearest"
        || node.string("coordinate_transformation_mode", "half_pixel")? != "asymmetric"
        || node.string("nearest_mode", "round_prefer_floor")? != "floor"
    {
        return Err(execution_error(node, "unsupported Resize policy"));
    }
    let scales = inputs
        .get(2)
        .ok_or_else(|| execution_error(node, "Resize requires scale factors"))?
        .tensor(&node.name)?
        .flatten_all()
        .and_then(|value| value.to_vec1::<f32>())
        .map_err(|error| execution_error(node, error))?;
    if scales.len() != 4 || scales[0] != 1.0 || scales[1] != 1.0 {
        return Err(execution_error(
            node,
            "Resize scales must be NCHW spatial scales",
        ));
    }
    let target_height = ((height as f64) * f64::from(scales[2])).floor() as usize;
    let target_width = ((width as f64) * f64::from(scales[3])).floor() as usize;
    input
        .upsample_nearest2d(target_height, target_width)
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

#[cfg(test)]
mod tests;
