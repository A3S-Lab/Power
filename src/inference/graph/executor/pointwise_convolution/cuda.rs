use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::CudaSlice;
#[cfg(test)]
use candle_core::InplaceOp3;
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Result, Shape, Tensor};

use super::super::batch_norm;
use super::super::convolution_post::{ConvolutionPostOperation, CudaBatchNormPostOperation};
use super::super::cuda_reproducibility::REPRODUCIBLE_BATCH_ITEMS;

/// Executes one shared F32 pointwise kernel over contiguous NCHW batches.
///
/// A 1x1 convolution does not need spatial lowering. Each batch is exactly the
/// matrix product `[outputs, channels] * [channels, spatial]`; cuBLAS accepts a
/// zero stride for the shared kernel and writes `[outputs, spatial]` directly
/// into the final NCHW layout.
pub(super) fn conv2d(input: &Tensor, kernel: &Tensor) -> Result<Tensor> {
    let operation = CudaPointwiseConv2d::new(input.layout(), kernel.layout())?;
    input.apply_op2_no_bwd(kernel, &operation)
}

#[cfg(test)]
fn conv2d_into(output: &Tensor, input: &Tensor, kernel: &Tensor) -> Result<()> {
    let operation = CudaPointwiseConv2d::new(input.layout(), kernel.layout())?;
    output.inplace_op3(input, kernel, &operation)
}

pub(super) fn conv2d_with_post_operation(
    input: &Tensor,
    kernel: &Tensor,
    post_operation: ConvolutionPostOperation,
) -> Result<Tensor> {
    let Some(batch_norm) = post_operation.cuda_batch_normalization_parameters() else {
        candle_core::bail!(
            "CUDA pointwise post-operation requires prepared BatchNormalization parameters"
        )
    };
    if !batch_norm
        .scale_and_bias
        .device()
        .same_device(input.device())
        || !batch_norm
            .mean_and_stddev
            .device()
            .same_device(input.device())
    {
        candle_core::bail!(
            "CUDA pointwise BatchNormalization parameters must share the convolution device"
        )
    }
    let operation =
        CudaPointwiseConv2d::new(input.layout(), kernel.layout())?.with_batch_norm(batch_norm)?;
    input.apply_op2_no_bwd(kernel, &operation)
}

#[derive(Clone)]
struct CudaPointwiseConv2d {
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    spatial: usize,
    output_elements: usize,
    batch_norm: Option<CudaBatchNormPostOperation>,
}

impl CudaPointwiseConv2d {
    fn new(input: &Layout, kernel: &Layout) -> Result<Self> {
        if !input.is_contiguous() || !kernel.is_contiguous() {
            candle_core::bail!("CUDA pointwise convolution requires contiguous tensors")
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
                "CUDA pointwise convolution requires non-empty matching 1x1 geometry"
            )
        }
        let spatial = height
            .checked_mul(width)
            .ok_or_else(|| dimension_error("spatial size overflowed"))?;
        let output_elements = batch
            .checked_mul(output_channels)
            .and_then(|value| value.checked_mul(spatial))
            .ok_or_else(|| dimension_error("output size overflowed"))?;
        if spatial == 0 || output_elements == 0 {
            candle_core::bail!("CUDA pointwise convolution requires a non-empty spatial canvas")
        }
        for (name, value) in [
            ("batch", batch),
            ("input channels", input_channels),
            ("output channels", output_channels),
            ("spatial size", spatial),
        ] {
            i32::try_from(value)
                .map_err(|_| dimension_error(&format!("{name} exceeds the cuBLAS i32 bound")))?;
        }
        for (name, value) in [
            (
                "input batch stride",
                input_channels
                    .checked_mul(spatial)
                    .ok_or_else(|| dimension_error("input batch stride overflowed"))?,
            ),
            (
                "output batch stride",
                output_channels
                    .checked_mul(spatial)
                    .ok_or_else(|| dimension_error("output batch stride overflowed"))?,
            ),
        ] {
            i64::try_from(value)
                .map_err(|_| dimension_error(&format!("{name} exceeds the cuBLAS i64 bound")))?;
        }
        Ok(Self {
            batch,
            input_channels,
            output_channels,
            spatial,
            output_elements,
            batch_norm: None,
        })
    }

    fn with_batch_norm(mut self, batch_norm: &CudaBatchNormPostOperation) -> Result<Self> {
        if batch_norm.scale_and_bias.dims2()? != (2, self.output_channels)
            || batch_norm.mean_and_stddev.dims2()? != (2, self.output_channels)
            || !batch_norm.scale_and_bias.is_contiguous()
            || !batch_norm.mean_and_stddev.is_contiguous()
        {
            candle_core::bail!(
                "CUDA pointwise BatchNormalization parameters must be contiguous [2, output_channels] tensors"
            )
        }
        self.batch_norm = Some(batch_norm.clone());
        Ok(self)
    }

    fn launch_values(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        kernel: &CudaStorage,
        kernel_layout: &Layout,
    ) -> Result<(CudaSlice<f32>, Shape)> {
        let device = input.device();
        let mut output = unsafe { device.alloc::<f32>(self.output_elements)? };
        self.launch_into(&mut output, input, input_layout, kernel, kernel_layout)?;
        let (_, _, height, width) = input_layout.shape().dims4()?;
        Ok((
            output,
            Shape::from_dims(&[self.batch, self.output_channels, height, width]),
        ))
    }

    fn launch_into(
        &self,
        output: &mut CudaSlice<f32>,
        input: &CudaStorage,
        input_layout: &Layout,
        kernel: &CudaStorage,
        kernel_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::cublas::{result, sys};
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut, DeviceSlice};
        use candle_core::cuda_backend::WrapErr;

        if output.len() != self.output_elements {
            candle_core::bail!(
                "CUDA pointwise destination length does not match the convolution geometry"
            )
        }
        let input_elements = self.batch * self.input_channels * self.spatial;
        let kernel_elements = self.output_channels * self.input_channels;
        let input_start = input_layout.start_offset();
        let kernel_start = kernel_layout.start_offset();
        let input_values = input.as_cuda_slice::<f32>()?;
        let kernel_values = kernel.as_cuda_slice::<f32>()?;
        let input_values = input_values.slice(input_start..input_start + input_elements);
        let kernel_values = kernel_values.slice(kernel_start..kernel_start + kernel_elements);
        let device = input.device();
        let blas = device.cublas_handle();
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let compute_type = if candle_core::cuda_backend::gemm_reduced_precision_f32() {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
        } else {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        };
        let input_batch_elements = self.input_channels * self.spatial;
        let output_batch_elements = self.output_channels * self.spatial;

        // NCHW `[O, S]` is column-major `[S, O]`. Therefore cuBLAS can
        // multiply the same storage as `[S, C] * [C, O]` without either an
        // im2col input or a transposed output allocation.
        for batch_offset in (0..self.batch).step_by(REPRODUCIBLE_BATCH_ITEMS) {
            let batch_items = (self.batch - batch_offset).min(REPRODUCIBLE_BATCH_ITEMS);
            let input_offset = batch_offset * input_batch_elements;
            let output_offset = batch_offset * output_batch_elements;
            let input_chunk =
                input_values.slice(input_offset..input_offset + batch_items * input_batch_elements);
            let mut output_chunk = output
                .slice_mut(output_offset..output_offset + batch_items * output_batch_elements);
            let stream = output_chunk.stream().clone();
            let (input_pointer, _input_guard) = input_chunk.device_ptr(&stream);
            let (kernel_pointer, _kernel_guard) = kernel_values.device_ptr(&stream);
            let (output_pointer, _output_guard) = output_chunk.device_ptr_mut(&stream);
            unsafe {
                result::gemm_strided_batched_ex(
                    *blas.handle(),
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    as_i32(self.spatial)?,
                    as_i32(self.output_channels)?,
                    as_i32(self.input_channels)?,
                    &alpha as *const f32 as *const _,
                    input_pointer as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    as_i32(self.spatial)?,
                    as_i64(input_batch_elements)?,
                    kernel_pointer as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    as_i32(self.input_channels)?,
                    0,
                    &beta as *const f32 as *const _,
                    output_pointer as *mut _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    as_i32(self.spatial)?,
                    as_i64(output_batch_elements)?,
                    as_i32(batch_items)?,
                    compute_type,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
            }
            .w()?;
            drop(_output_guard);
            drop(_kernel_guard);
            drop(_input_guard);
        }

        if let Some(post_operation) = &self.batch_norm {
            batch_norm::execute_cuda_post_in_place(
                output,
                device,
                self.batch,
                self.output_channels,
                self.spatial,
                post_operation,
            )?;
        }

        Ok(())
    }

    fn launch(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        kernel: &CudaStorage,
        kernel_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (output, shape) = self.launch_values(input, input_layout, kernel, kernel_layout)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, input.device().clone()),
            shape,
        ))
    }
}

impl CustomOp2 for CudaPointwiseConv2d {
    fn name(&self) -> &'static str {
        "a3s-cuda-pointwise-conv2d"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _kernel: &CpuStorage,
        _kernel_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the direct pointwise operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        kernel: &CudaStorage,
        kernel_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch(input, input_layout, kernel, kernel_layout)
    }
}

#[cfg(test)]
impl InplaceOp3 for CudaPointwiseConv2d {
    fn name(&self) -> &'static str {
        "a3s-cuda-pointwise-conv2d-into"
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
        candle_core::bail!("the preallocated pointwise operation is CUDA-only")
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
        let (_, _, height, width) = input_layout.shape().dims4()?;
        let expected = [self.batch, self.output_channels, height, width];
        if !output_layout.is_contiguous()
            || output_layout.start_offset() != 0
            || output_layout.shape().dims() != expected
        {
            candle_core::bail!(
                "CUDA pointwise destination must be one exact contiguous output tensor"
            )
        }
        self.launch_into(
            output.as_cuda_slice_mut::<f32>()?,
            input,
            input_layout,
            kernel,
            kernel_layout,
        )
    }
}

fn as_i32(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| dimension_error("dimension exceeds the cuBLAS i32 bound"))
}

fn as_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| dimension_error("stride exceeds the cuBLAS i64 bound"))
}

fn dimension_error(message: &str) -> candle_core::Error {
    candle_core::Error::Msg(format!("CUDA pointwise convolution {message}"))
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use candle_core::cuda_backend::cudarc::driver::sys::{
        CUgraphInstantiate_flags_enum, CUstreamCaptureMode_enum,
    };
    use candle_core::cuda_backend::cudarc::driver::CudaGraph;
    use candle_core::{DType, Device, Tensor, Var};

    use super::{conv2d, conv2d_into};
    use crate::inference::graph::executor::biased_activation::{
        cuda_channel_bias_relu, cuda_channel_bias_relu_into,
    };
    use crate::inference::graph::executor::depthwise::{
        conv2d as depthwise_conv2d, cuda_conv2d_into,
    };

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn preallocated_pointwise_capture_replays_changed_input_exactly() {
        let device = Device::new_cuda_with_stream(0).unwrap();
        let Device::Cuda(cuda) = &device else {
            panic!("explicit CUDA device resolved another backend")
        };
        // SAFETY: this test owns one device and stream, and all tensors remain
        // on that stream until the final synchronization.
        unsafe { cuda.disable_event_tracking() };

        let shape = (2, 7, 3, 5);
        let output_shape = (2, 11, 3, 5);
        let elements = shape.0 * shape.1 * shape.2 * shape.3;
        let first = Tensor::from_iter(
            (0..elements).map(|index| ((index * 17 % 251) as f32 - 125.0) / 127.0),
            &device,
        )
        .unwrap()
        .reshape(shape)
        .unwrap();
        let second = Tensor::from_iter(
            (0..elements).map(|index| ((index * 29 % 257) as f32 - 128.0) / 131.0),
            &device,
        )
        .unwrap()
        .reshape(shape)
        .unwrap();
        let kernel = Tensor::from_iter(
            (0..output_shape.1 * shape.1).map(|index| ((index * 31 % 263) as f32 - 131.0) / 137.0),
            &device,
        )
        .unwrap()
        .reshape((output_shape.1, shape.1, 1, 1))
        .unwrap();
        let bias = Tensor::from_iter(
            (0..output_shape.1).map(|index| (index as f32 - 5.0) / 17.0),
            &device,
        )
        .unwrap()
        .reshape((1, output_shape.1, 1, 1))
        .unwrap();
        let depthwise_kernel = Tensor::from_iter(
            (0..output_shape.1 * 3 * 3).map(|index| ((index * 37 % 269) as f32 - 134.0) / 139.0),
            &device,
        )
        .unwrap()
        .reshape((output_shape.1, 1, 3, 3))
        .unwrap();
        let first_activation =
            cuda_channel_bias_relu(&conv2d(&first, &kernel).unwrap(), &bias).unwrap();
        let expected_first = depthwise_conv2d(
            &first_activation,
            &depthwise_kernel,
            None,
            (1, 1, 1, 1),
            (1, 1),
            1,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let second_activation =
            cuda_channel_bias_relu(&conv2d(&second, &kernel).unwrap(), &bias).unwrap();
        let expected_second = depthwise_conv2d(
            &second_activation,
            &depthwise_kernel,
            None,
            (1, 1, 1, 1),
            (1, 1),
            1,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

        let input = Var::zeros(shape, DType::F32, &device).unwrap();
        input.set(&first).unwrap();
        let convolution = Var::zeros(output_shape, DType::F32, &device).unwrap();
        let activation = Var::zeros(output_shape, DType::F32, &device).unwrap();
        // The pointwise destination is dead after the activation reads it, so
        // the final depthwise write reuses that exact storage on the next step.
        let output = convolution.as_tensor().clone();
        let stream = cuda.cuda_stream();
        stream.synchronize().unwrap();
        stream
            .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        conv2d_into(convolution.as_tensor(), input.as_tensor(), &kernel).unwrap();
        cuda_channel_bias_relu_into(activation.as_tensor(), convolution.as_tensor(), &bias)
            .unwrap();
        cuda_conv2d_into(
            &output,
            activation.as_tensor(),
            &depthwise_kernel,
            (1, 1, 1, 1),
            (1, 1),
            1,
        )
        .unwrap();
        let graph = stream
            .end_capture(
                CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
            )
            .unwrap()
            .expect("pointwise execution must capture CUDA work");
        assert_eq!(captured_memory_node_count(&graph), 0);
        graph.upload().unwrap();
        graph.launch().unwrap();
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected_first,
        );

        input.set(&second).unwrap();
        graph.launch().unwrap();
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected_second,
        );

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
}
