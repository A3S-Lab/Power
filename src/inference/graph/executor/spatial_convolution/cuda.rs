use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::CudaSlice;
#[cfg(test)]
use candle_core::InplaceOp2;
use candle_core::{CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape, Tensor};

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/im2col_ptx.rs"));
}
mod post;
mod product;

use super::super::convolution_post::ConvolutionPostOperation;
use super::super::cuda_reproducibility::REPRODUCIBLE_BATCH_ITEMS;

const MODULE_NAME: &str = "a3s_power_spatial_im2col_f32_v4";
const FUNCTION_NAME: &str = "im2col_contiguous_u32_f32";

pub(super) fn conv2d(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    pads: (usize, usize, usize, usize),
    stride: usize,
    dilation: usize,
) -> Result<Tensor> {
    conv2d_unpartitioned(input, kernel, bias, pads, stride, dilation)
}

fn conv2d_unpartitioned(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    pads: (usize, usize, usize, usize),
    stride: usize,
    dilation: usize,
) -> Result<Tensor> {
    let (product, lowering) = lowered_product(input, kernel, bias, pads, stride, dilation)?;
    let output = product
        .reshape((
            lowering.batch,
            lowering.output_height,
            lowering.output_width,
            lowering.output_channels,
        ))?
        .permute((0, 3, 1, 2))
        .and_then(|output| output.contiguous())?;
    let Some(bias) = bias else {
        return Ok(output);
    };
    output.broadcast_add(&bias.reshape((1, lowering.output_channels, 1, 1))?)
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
    conv2d_with_post_operation_unpartitioned(
        input,
        kernel,
        bias,
        pads,
        stride,
        dilation,
        post_operation,
    )
}

fn conv2d_with_post_operation_unpartitioned(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    pads: (usize, usize, usize, usize),
    stride: usize,
    dilation: usize,
    post_operation: ConvolutionPostOperation,
) -> Result<Tensor> {
    let (product, lowering) = lowered_product(input, kernel, bias, pads, stride, dilation)?;
    post::execute(&product, bias, lowering, post_operation)
}

fn lowered_product(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    pads: (usize, usize, usize, usize),
    stride: usize,
    dilation: usize,
) -> Result<(Tensor, Im2Col)> {
    if input.dtype() != DType::F32
        || kernel.dtype() != DType::F32
        || bias.is_some_and(|bias| bias.dtype() != DType::F32)
        || !input.device().is_cuda()
        || !kernel.device().same_device(input.device())
        || bias.is_some_and(|bias| !bias.device().same_device(input.device()))
    {
        candle_core::bail!("CUDA spatial convolution requires co-located F32 tensors")
    }
    let lowering = Im2Col::new(input.layout(), kernel.layout(), pads, stride, dilation)?;
    let columns = input.apply_op1_no_bwd(&lowering)?;
    let kernel_matrix = kernel.reshape((lowering.output_channels, lowering.patch_elements))?;
    let product = if lowering.batch > REPRODUCIBLE_BATCH_ITEMS {
        product::execute(&columns, &kernel_matrix, lowering.batch)
    } else {
        kernel_matrix.t().and_then(|kernel| columns.matmul(&kernel))
    }?;
    Ok((product, lowering))
}

#[derive(Clone, Copy)]
struct Im2Col {
    batch: usize,
    output_channels: usize,
    output_height: usize,
    output_width: usize,
    patch_elements: usize,
    output_elements: usize,
    parameters: [u32; 11],
}

impl Im2Col {
    fn new(
        input: &Layout,
        kernel: &Layout,
        pads: (usize, usize, usize, usize),
        stride: usize,
        dilation: usize,
    ) -> Result<Self> {
        if !input.is_contiguous() || !kernel.is_contiguous() {
            candle_core::bail!("CUDA spatial im2col requires contiguous tensors")
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
                "CUDA spatial im2col requires non-empty matching convolution geometry"
            )
        }
        let effective_height = effective_kernel(kernel_height, dilation, "height")?;
        let effective_width = effective_kernel(kernel_width, dilation, "width")?;
        let padded_height = input_height
            .checked_add(pads.0)
            .and_then(|value| value.checked_add(pads.2))
            .ok_or_else(|| dimension_error("padded height overflowed"))?;
        let padded_width = input_width
            .checked_add(pads.1)
            .and_then(|value| value.checked_add(pads.3))
            .ok_or_else(|| dimension_error("padded width overflowed"))?;
        let output_height = output_dimension(padded_height, effective_height, stride, "height")?;
        let output_width = output_dimension(padded_width, effective_width, stride, "width")?;
        let patch_elements = input_channels
            .checked_mul(kernel_height)
            .and_then(|value| value.checked_mul(kernel_width))
            .ok_or_else(|| dimension_error("patch size overflowed"))?;
        let output_rows = batch
            .checked_mul(output_height)
            .and_then(|value| value.checked_mul(output_width))
            .ok_or_else(|| dimension_error("output row count overflowed"))?;
        let output_elements = output_rows
            .checked_mul(patch_elements)
            .ok_or_else(|| dimension_error("lowered output size overflowed"))?;
        let input_elements = batch
            .checked_mul(input_channels)
            .and_then(|value| value.checked_mul(input_height))
            .and_then(|value| value.checked_mul(input_width))
            .ok_or_else(|| dimension_error("input size overflowed"))?;
        if output_elements == 0
            || u32::try_from(output_elements).is_err()
            || u32::try_from(input_elements).is_err()
        {
            candle_core::bail!("CUDA spatial im2col exceeds the reviewed u32 launch bound")
        }
        let source = [
            input_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            stride,
            dilation,
            pads.0,
            pads.1,
        ];
        let mut parameters = [0_u32; 11];
        for (target, value) in parameters.iter_mut().zip(source) {
            *target = u32::try_from(value)
                .map_err(|_| dimension_error("convolution dimension exceeds u32"))?;
        }
        Ok(Self {
            batch,
            output_channels,
            output_height,
            output_width,
            patch_elements,
            output_elements,
            parameters,
        })
    }

    fn column_shape(self) -> Shape {
        Shape::from_dims(&[
            self.batch * self.output_height * self.output_width,
            self.patch_elements,
        ])
    }

    fn launch_into(
        self,
        output: &mut CudaSlice<f32>,
        input: &CudaStorage,
        input_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        if output.len() != self.output_elements {
            candle_core::bail!("CUDA spatial im2col destination has the wrong length")
        }
        let device = input.device();
        let function =
            device.get_or_load_custom_func(FUNCTION_NAME, MODULE_NAME, kernels::SPATIAL_IM2COL)?;
        let mut builder = function.builder();
        builder.arg(input.as_cuda_slice::<f32>()?);
        builder.arg(&*output);
        let output_elements = u32::try_from(self.output_elements)
            .map_err(|_| dimension_error("launch size exceeds u32"))?;
        builder.arg(&output_elements);
        let input_offset = u64::try_from(input_layout.start_offset())
            .map_err(|_| dimension_error("input offset exceeds u64"))?;
        builder.arg(&input_offset);
        for value in &self.parameters {
            builder.arg(value);
        }
        const THREADS_PER_BLOCK: u32 = 512;
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

    fn launch(self, input: &CudaStorage, input_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let device = input.device();
        let mut output = unsafe { device.alloc::<f32>(self.output_elements)? };
        self.launch_into(&mut output, input, input_layout)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            self.column_shape(),
        ))
    }
}

impl CustomOp1 for Im2Col {
    fn name(&self) -> &'static str {
        "a3s-cuda-contiguous-u32-im2col"
    }

    fn cpu_fwd(&self, _input: &CpuStorage, _input_layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the contiguous u32 im2col operation is CUDA-only")
    }

    fn cuda_fwd(&self, input: &CudaStorage, input_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        self.launch(input, input_layout)
    }
}

#[cfg(test)]
impl InplaceOp2 for Im2Col {
    fn name(&self) -> &'static str {
        "a3s-cuda-contiguous-u32-im2col-into"
    }

    fn cpu_fwd(
        &self,
        _output: &mut CpuStorage,
        _output_layout: &Layout,
        _input: &CpuStorage,
        _input_layout: &Layout,
    ) -> Result<()> {
        candle_core::bail!("the preallocated spatial im2col operation is CUDA-only")
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
            || output_layout.shape() != &self.column_shape()
        {
            candle_core::bail!(
                "spatial im2col destination must be one exact contiguous output tensor"
            )
        }
        self.launch_into(output.as_cuda_slice_mut::<f32>()?, input, input_layout)
    }
}

fn effective_kernel(kernel: usize, dilation: usize, axis: &str) -> Result<usize> {
    dilation
        .checked_mul(kernel - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| dimension_error(&format!("effective kernel {axis} overflowed")))
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
        .filter(|value| *value > 0)
        .ok_or_else(|| dimension_error(&format!("kernel exceeds input {axis}")))
}

fn dimension_error(message: &str) -> candle_core::Error {
    candle_core::Error::Msg(format!("CUDA spatial im2col {message}"))
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use candle_core::cuda_backend::cudarc::driver::sys::{
        CUgraphInstantiate_flags_enum, CUstreamCaptureMode_enum,
    };
    use candle_core::cuda_backend::cudarc::driver::CudaGraph;
    use candle_core::{DType, Device, Var};

    use crate::inference::graph::executor::batch_norm::prepare_cuda_statistics;
    use crate::inference::graph::executor::convolution_post::CudaBatchNormActivation;
    use crate::inference::{DevicePreference, RuntimeDevice};

    use super::*;

    fn bits(tensor: &Tensor) -> Vec<u32> {
        tensor
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn contiguous_u32_im2col_preserves_spatial_convolution_bits() {
        let device = Device::new_cuda_with_stream(0).unwrap();
        let input = Tensor::from_iter(
            (0..2 * 3 * 7 * 9).map(|value| ((value * 17 % 101) as f32 - 50.0) / 53.0),
            &device,
        )
        .unwrap()
        .reshape((2, 3, 7, 9))
        .unwrap();
        let kernel = Tensor::from_iter(
            (0..4 * 3 * 3 * 3).map(|value| ((value * 13 % 67) as f32 - 33.0) / 31.0),
            &device,
        )
        .unwrap()
        .reshape((4, 3, 3, 3))
        .unwrap();
        let bias = Tensor::new(&[-0.25_f32, 0.125, 0.375, -0.5], &device).unwrap();

        for (pads, stride, dilation) in [
            ((1, 1, 1, 1), 1, 1),
            ((0, 1, 2, 0), 2, 1),
            ((2, 2, 2, 2), 1, 2),
        ] {
            let padded = input
                .pad_with_zeros(2, pads.0, pads.2)
                .unwrap()
                .pad_with_zeros(3, pads.1, pads.3)
                .unwrap();
            for convolution_bias in [None, Some(&bias)] {
                let mut expected = padded.conv2d(&kernel, 0, stride, dilation, 1).unwrap();
                if let Some(convolution_bias) = convolution_bias {
                    expected = expected
                        .broadcast_add(&convolution_bias.reshape((1, 4, 1, 1)).unwrap())
                        .unwrap();
                }
                let actual =
                    conv2d(&input, &kernel, convolution_bias, pads, stride, dilation).unwrap();
                actual.device().synchronize().unwrap();
                assert_eq!(
                    bits(&actual),
                    bits(&expected),
                    "pads={pads:?} stride={stride} dilation={dilation} bias={}",
                    convolution_bias.is_some(),
                );
            }
        }
    }

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn preallocated_spatial_workspace_capture_replays_changed_input_exactly() {
        let runtime =
            RuntimeDevice::resolve_model_session(DevicePreference::Cuda { ordinal: 0 }).unwrap();
        let device = runtime.tensor_device();
        let Device::Cuda(cuda) = device else {
            panic!("explicit CUDA device resolved another backend")
        };
        assert!(!cuda.is_event_tracking());

        let shape = (2, 3, 7, 9);
        let output_shape = (2, 4, 7, 9);
        let elements = shape.0 * shape.1 * shape.2 * shape.3;
        let first = Tensor::from_iter(
            (0..elements).map(|value| ((value * 17 % 101) as f32 - 50.0) / 53.0),
            device,
        )
        .unwrap()
        .reshape(shape)
        .unwrap();
        let second = Tensor::from_iter(
            (0..elements).map(|value| ((value * 29 % 103) as f32 - 51.0) / 59.0),
            device,
        )
        .unwrap()
        .reshape(shape)
        .unwrap();
        let kernel = Tensor::from_iter(
            (0..4 * 3 * 3 * 3).map(|value| ((value * 13 % 67) as f32 - 33.0) / 31.0),
            device,
        )
        .unwrap()
        .reshape((4, 3, 3, 3))
        .unwrap();
        let pads = (1, 1, 1, 1);
        let lowering = Im2Col::new(first.layout(), kernel.layout(), pads, 1, 1).unwrap();
        let kernel_matrix = kernel.reshape((4, 27)).unwrap();

        let expected_first = bits(
            &conv2d_with_post_operation(
                &first,
                &kernel,
                None,
                pads,
                1,
                1,
                ConvolutionPostOperation::Relu,
            )
            .unwrap(),
        );
        let expected_second = bits(
            &conv2d_with_post_operation(
                &second,
                &kernel,
                None,
                pads,
                1,
                1,
                ConvolutionPostOperation::Relu,
            )
            .unwrap(),
        );

        let input = Var::zeros(shape, DType::F32, device).unwrap();
        input.set(&first).unwrap();
        let columns = Var::zeros(lowering.column_shape(), DType::F32, device).unwrap();
        let product = Var::zeros(
            (
                lowering.batch * lowering.output_height * lowering.output_width,
                lowering.output_channels,
            ),
            DType::F32,
            device,
        )
        .unwrap();
        // Im2col is dead after the product reads it. Reuse the leading region
        // of that larger slot for the final NCHW output on the following step.
        let output = columns
            .as_tensor()
            .flatten_all()
            .unwrap()
            .narrow(
                0,
                0,
                output_shape.0 * output_shape.1 * output_shape.2 * output_shape.3,
            )
            .unwrap()
            .reshape(output_shape)
            .unwrap();

        let stream = cuda.cuda_stream();
        columns
            .as_tensor()
            .inplace_op2(input.as_tensor(), &lowering)
            .unwrap();
        product::execute_into(
            product.as_tensor(),
            columns.as_tensor(),
            &kernel_matrix,
            lowering.batch,
        )
        .unwrap();
        post::relu_into(&output, product.as_tensor(), lowering).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(bits(&output), expected_first);
        input.set(&first).unwrap();
        stream
            .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        columns
            .as_tensor()
            .inplace_op2(input.as_tensor(), &lowering)
            .unwrap();
        product::execute_into(
            product.as_tensor(),
            columns.as_tensor(),
            &kernel_matrix,
            lowering.batch,
        )
        .unwrap();
        post::relu_into(&output, product.as_tensor(), lowering).unwrap();
        let graph = stream
            .end_capture(
                CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
            )
            .unwrap()
            .expect("spatial execution must capture CUDA work");
        assert_eq!(captured_memory_node_count(&graph), 0);
        graph.upload().unwrap();
        graph.launch().unwrap();
        assert_eq!(bits(&output), expected_first);

        input.set(&second).unwrap();
        graph.launch().unwrap();
        assert_eq!(bits(&output), expected_second);

        stream.synchronize().unwrap();
        drop(graph);
        stream.synchronize().unwrap();
    }

    fn captured_memory_node_count(graph: &CudaGraph) -> usize {
        use candle_core::cuda_backend::cudarc::driver::sys;

        let raw = graph.cu_graph();
        let mut count = 0_usize;
        unsafe { sys::cuGraphGetNodes(raw, std::ptr::null_mut(), &mut count) }
            .result()
            .unwrap();
        let mut nodes = vec![std::ptr::null_mut(); count];
        unsafe { sys::cuGraphGetNodes(raw, nodes.as_mut_ptr(), &mut count) }
            .result()
            .unwrap();
        nodes
            .into_iter()
            .take(count)
            .filter(|node| {
                let mut kind = MaybeUninit::<sys::CUgraphNodeType>::uninit();
                unsafe { sys::cuGraphNodeGetType(*node, kind.as_mut_ptr()) }
                    .result()
                    .unwrap();
                matches!(
                    unsafe { kind.assume_init() },
                    sys::CUgraphNodeType_enum::CU_GRAPH_NODE_TYPE_MEM_ALLOC
                        | sys::CUgraphNodeType_enum::CU_GRAPH_NODE_TYPE_MEM_FREE
                )
            })
            .count()
    }

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn large_batch_preserves_explicit_reproducible_chunk_bits() {
        let device = Device::new_cuda_with_stream(0).unwrap();
        let batch = 2 * REPRODUCIBLE_BATCH_ITEMS + 1;
        let input = Tensor::from_iter(
            (0..batch * 3 * 7 * 9).map(|value| ((value * 17 % 101) as f32 - 50.0) / 53.0),
            &device,
        )
        .unwrap()
        .reshape((batch, 3, 7, 9))
        .unwrap();
        let kernel = Tensor::from_iter(
            (0..4 * 3 * 3 * 3).map(|value| ((value * 13 % 67) as f32 - 33.0) / 31.0),
            &device,
        )
        .unwrap()
        .reshape((4, 3, 3, 3))
        .unwrap();
        let bias = Tensor::new(&[-0.25_f32, 0.125, 0.375, -0.5], &device).unwrap();
        let pads = (1, 2, 0, 1);
        let stride = 2;
        let dilation = 1;

        let mut expected_plain = Vec::new();
        let mut expected_relu = Vec::new();
        for offset in (0..batch).step_by(REPRODUCIBLE_BATCH_ITEMS) {
            let chunk_items = (batch - offset).min(REPRODUCIBLE_BATCH_ITEMS);
            let chunk = input.narrow(0, offset, chunk_items).unwrap();
            expected_plain.push(
                conv2d_unpartitioned(&chunk, &kernel, Some(&bias), pads, stride, dilation).unwrap(),
            );
            expected_relu.push(
                conv2d_with_post_operation_unpartitioned(
                    &chunk,
                    &kernel,
                    Some(&bias),
                    pads,
                    stride,
                    dilation,
                    ConvolutionPostOperation::Relu,
                )
                .unwrap(),
            );
        }
        let expected_plain = Tensor::cat(&expected_plain, 0).unwrap();
        let expected_relu = Tensor::cat(&expected_relu, 0).unwrap();

        let actual_plain = conv2d(&input, &kernel, Some(&bias), pads, stride, dilation).unwrap();
        let actual_relu = conv2d_with_post_operation(
            &input,
            &kernel,
            Some(&bias),
            pads,
            stride,
            dilation,
            ConvolutionPostOperation::Relu,
        )
        .unwrap();
        actual_relu.device().synchronize().unwrap();

        assert_eq!(actual_plain.dims(), expected_plain.dims());
        assert_eq!(bits(&actual_plain), bits(&expected_plain));
        assert_eq!(actual_relu.dims(), expected_relu.dims());
        assert_eq!(bits(&actual_relu), bits(&expected_relu));
    }

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn fused_layout_batch_norm_preserves_explicit_graph_bits_and_offsets() {
        let device = Device::new_cuda_with_stream(0).unwrap();
        let input = Tensor::from_iter(
            (0..2 * 3 * 7 * 9).map(|value| ((value * 17 % 101) as f32 - 50.0) / 53.0),
            &device,
        )
        .unwrap()
        .reshape((2, 3, 7, 9))
        .unwrap();
        let kernel = Tensor::from_iter(
            (0..4 * 3 * 3 * 3).map(|value| ((value * 13 % 67) as f32 - 33.0) / 31.0),
            &device,
        )
        .unwrap()
        .reshape((4, 3, 3, 3))
        .unwrap();
        let convolution_bias_storage =
            Tensor::new(&[91.0_f32, -0.25, 0.125, 0.375, -0.5, -92.0], &device).unwrap();
        let convolution_bias = convolution_bias_storage.narrow(0, 1, 4).unwrap();
        let scale_and_bias_storage = Tensor::new(
            &[
                81.0_f32, 0.75, -1.25, 2.0, 0.5, -0.5, 0.125, 1.5, -0.75, -82.0,
            ],
            &device,
        )
        .unwrap();
        let scale_and_bias = scale_and_bias_storage
            .narrow(0, 1, 8)
            .unwrap()
            .reshape((2, 4))
            .unwrap();
        let mean_and_variance_storage = Tensor::new(
            &[71.0_f32, 0.25, -0.75, 1.25, -1.5, 0.5, 1.5, 2.5, 3.5, -72.0],
            &device,
        )
        .unwrap();
        let mean_and_variance = mean_and_variance_storage
            .narrow(0, 1, 8)
            .unwrap()
            .reshape((2, 4))
            .unwrap();
        let epsilon = 0.000_01_f32;
        let mean_and_stddev = prepare_cuda_statistics(&mean_and_variance, epsilon).unwrap();
        let activations = [
            CudaBatchNormActivation::Identity,
            CudaBatchNormActivation::Relu,
            CudaBatchNormActivation::HardSwish {
                alpha: 0.2,
                beta: 0.5,
            },
            CudaBatchNormActivation::Swish,
            CudaBatchNormActivation::GeluErf {
                divisor: std::f32::consts::SQRT_2,
                offset: 1.0,
                scale: 0.5,
            },
        ];

        for convolution_bias in [None, Some(&convolution_bias)] {
            let convolution =
                conv2d(&input, &kernel, convolution_bias, (1, 2, 0, 1), 2, 1).unwrap();
            for activation in activations {
                let expected = explicit_batch_norm(
                    &convolution,
                    &scale_and_bias,
                    &mean_and_variance,
                    epsilon,
                    activation,
                );
                let post_operation = ConvolutionPostOperation::cuda_batch_normalization(
                    &scale_and_bias,
                    &mean_and_stddev,
                    activation,
                )
                .unwrap();
                let actual = conv2d_with_post_operation(
                    &input,
                    &kernel,
                    convolution_bias,
                    (1, 2, 0, 1),
                    2,
                    1,
                    post_operation,
                )
                .unwrap();
                actual.device().synchronize().unwrap();
                assert_eq!(actual.dims(), expected.dims());
                assert_eq!(
                    bits(&actual),
                    bits(&expected),
                    "activation={activation:?} convolution_bias={}",
                    convolution_bias.is_some(),
                );
            }
        }
    }

    fn explicit_batch_norm(
        input: &Tensor,
        scale_and_bias: &Tensor,
        mean_and_variance: &Tensor,
        epsilon: f32,
        activation: CudaBatchNormActivation,
    ) -> Tensor {
        let channel_shape = (1, input.dim(1).unwrap(), 1, 1);
        let scale = scale_and_bias
            .get(0)
            .unwrap()
            .reshape(channel_shape)
            .unwrap();
        let bias = scale_and_bias
            .get(1)
            .unwrap()
            .reshape(channel_shape)
            .unwrap();
        let mean = mean_and_variance
            .get(0)
            .unwrap()
            .reshape(channel_shape)
            .unwrap();
        let variance = mean_and_variance
            .get(1)
            .unwrap()
            .reshape(channel_shape)
            .unwrap();
        let normalized = input
            .broadcast_sub(&mean)
            .unwrap()
            .broadcast_div(
                &variance
                    .affine(1.0, epsilon as f64)
                    .unwrap()
                    .sqrt()
                    .unwrap(),
            )
            .unwrap()
            .broadcast_mul(&scale)
            .unwrap()
            .broadcast_add(&bias)
            .unwrap();
        match activation {
            CudaBatchNormActivation::Identity => normalized,
            CudaBatchNormActivation::Relu => normalized.relu().unwrap(),
            CudaBatchNormActivation::HardSwish { alpha, beta } => {
                let gate = (&normalized * alpha as f64)
                    .unwrap()
                    .affine(1.0, beta as f64)
                    .unwrap()
                    .clamp(0.0, 1.0)
                    .unwrap();
                normalized.broadcast_mul(&gate).unwrap()
            }
            CudaBatchNormActivation::Swish => normalized
                .broadcast_mul(&candle_nn::ops::sigmoid(&normalized).unwrap())
                .unwrap(),
            CudaBatchNormActivation::GeluErf {
                divisor,
                offset,
                scale,
            } => {
                let divisor = Tensor::new(&[divisor], normalized.device()).unwrap();
                let offset = Tensor::new(&[offset], normalized.device()).unwrap();
                let scale = Tensor::new(&[scale], normalized.device()).unwrap();
                normalized
                    .broadcast_div(&divisor)
                    .and_then(|value| value.erf())
                    .and_then(|value| value.broadcast_add(&offset))
                    .and_then(|value| normalized.broadcast_mul(&value))
                    .and_then(|value| value.broadcast_mul(&scale))
                    .unwrap()
            }
        }
    }
}
