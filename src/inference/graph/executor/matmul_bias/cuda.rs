use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp3, Layout, Result, Shape, Tensor};

use super::super::cuda_fast_divisor::FastDivisorU32;
use super::PostOperation;

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/matmul_bias_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_matmul_bias_f32_v2";
const BIAS_FUNCTION: &str = "last_axis_bias_in_place_f32";
const SWISH_FUNCTION: &str = "last_axis_bias_swish_in_place_f32";
const THREADS_PER_BLOCK: u32 = 512;

pub(super) fn execute(
    left: &Tensor,
    right: &Tensor,
    bias: &Tensor,
    post_operation: PostOperation,
) -> Result<Tensor> {
    let operation =
        LastAxisMatMulBias::new(left.layout(), right.layout(), bias.layout(), post_operation)?;
    left.apply_op3_no_bwd(right, bias, &operation)
}

#[derive(Clone)]
struct LastAxisMatMulBias {
    output_shape: Shape,
    batch: usize,
    rows: usize,
    inner: usize,
    columns: usize,
    output_elements: usize,
    feature_divisor: [u32; 3],
    post_operation: PostOperation,
}

impl LastAxisMatMulBias {
    fn new(
        left: &Layout,
        right: &Layout,
        bias: &Layout,
        post_operation: PostOperation,
    ) -> Result<Self> {
        if !left.is_contiguous() || !right.is_contiguous() || !bias.is_contiguous() {
            candle_core::bail!("MatMul bias requires contiguous inputs")
        }
        if left.shape().rank() < 2 || right.shape().rank() != 2 || bias.shape().rank() != 1 {
            candle_core::bail!(
                "MatMul bias requires rank-two-or-higher left, rank-two right, and rank-one bias"
            )
        }
        let left_dimensions = left.shape().dims();
        let right_dimensions = right.shape().dims();
        let rows = left_dimensions[left_dimensions.len() - 2];
        let inner = left_dimensions[left_dimensions.len() - 1];
        let columns = right_dimensions[1];
        if rows == 0
            || inner == 0
            || columns == 0
            || right_dimensions[0] != inner
            || bias.shape().dims() != [columns]
        {
            candle_core::bail!("MatMul bias received incompatible or empty geometry")
        }
        let batch = left_dimensions[..left_dimensions.len() - 2]
            .iter()
            .try_fold(1_usize, |batch, dimension| batch.checked_mul(*dimension))
            .ok_or_else(|| dimension_error("batch size overflowed"))?;
        let output_elements = batch
            .checked_mul(rows)
            .and_then(|elements| elements.checked_mul(columns))
            .ok_or_else(|| dimension_error("output size overflowed"))?;
        if batch == 0 || output_elements == 0 || u32::try_from(output_elements).is_err() {
            candle_core::bail!("MatMul bias exceeds the reviewed launch bound")
        }
        for (name, value) in [
            ("batch", batch),
            ("row count", rows),
            ("inner dimension", inner),
            ("column count", columns),
        ] {
            i32::try_from(value)
                .map_err(|_| dimension_error(&format!("{name} exceeds the cuBLAS i32 bound")))?;
        }
        for (name, value) in [
            (
                "left batch stride",
                rows.checked_mul(inner)
                    .ok_or_else(|| dimension_error("left batch stride overflowed"))?,
            ),
            (
                "output batch stride",
                rows.checked_mul(columns)
                    .ok_or_else(|| dimension_error("output batch stride overflowed"))?,
            ),
        ] {
            i64::try_from(value)
                .map_err(|_| dimension_error(&format!("{name} exceeds the cuBLAS i64 bound")))?;
        }
        let columns_u32 = u32::try_from(columns)
            .map_err(|_| dimension_error("column count exceeds the u32 launch bound"))?;
        let feature_divisor = FastDivisorU32::new(columns_u32)
            .ok_or_else(|| dimension_error("column count is zero"))?
            .launch_parameters();
        let mut output_dimensions = left_dimensions[..left_dimensions.len() - 1].to_vec();
        output_dimensions.push(columns);
        Ok(Self {
            output_shape: Shape::from_dims(&output_dimensions),
            batch,
            rows,
            inner,
            columns,
            output_elements,
            feature_divisor,
            post_operation,
        })
    }

    fn launch(
        &self,
        left: &CudaStorage,
        left_layout: &Layout,
        right: &CudaStorage,
        right_layout: &Layout,
        bias: &CudaStorage,
        bias_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::cublas::{result, sys};
        use candle_core::cuda_backend::cudarc::driver::{
            DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::WrapErr;

        let left_elements = self.batch * self.rows * self.inner;
        let right_elements = self.inner * self.columns;
        let left_start = left_layout.start_offset();
        let right_start = right_layout.start_offset();
        let left_values = left.as_cuda_slice::<f32>()?;
        let right_values = right.as_cuda_slice::<f32>()?;
        let bias_values = bias.as_cuda_slice::<f32>()?;
        let left_values = left_values.slice(left_start..left_start + left_elements);
        let right_values = right_values.slice(right_start..right_start + right_elements);
        let device = left.device();
        let blas = device.cublas_handle();
        let mut output = unsafe { device.alloc::<f32>(self.output_elements)? };
        let stream = output.stream().clone();
        let (left_pointer, left_guard) = left_values.device_ptr(&stream);
        let (right_pointer, right_guard) = right_values.device_ptr(&stream);
        let (output_pointer, output_guard) = output.device_ptr_mut(&stream);
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let compute_type = if candle_core::cuda_backend::gemm_reduced_precision_f32() {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
        } else {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        };
        unsafe {
            result::gemm_strided_batched_ex(
                *blas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_N,
                sys::cublasOperation_t::CUBLAS_OP_N,
                as_i32(self.columns)?,
                as_i32(self.rows)?,
                as_i32(self.inner)?,
                &alpha as *const f32 as *const _,
                right_pointer as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                as_i32(self.columns)?,
                0,
                left_pointer as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                as_i32(self.inner)?,
                as_i64(self.rows * self.inner)?,
                &beta as *const f32 as *const _,
                output_pointer as *mut _,
                sys::cudaDataType_t::CUDA_R_32F,
                as_i32(self.columns)?,
                as_i64(self.rows * self.columns)?,
                as_i32(self.batch)?,
                compute_type,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
        .w()?;
        drop(output_guard);
        drop(right_guard);
        drop(left_guard);

        let function_name = match self.post_operation {
            PostOperation::Bias => BIAS_FUNCTION,
            PostOperation::Swish => SWISH_FUNCTION,
        };
        let function =
            device.get_or_load_custom_func(function_name, MODULE_NAME, kernels::MATMUL_BIAS)?;
        let bias_offset = u64::try_from(bias_layout.start_offset())
            .map_err(|_| dimension_error("bias offset exceeds u64"))?;
        let element_count = u32::try_from(self.output_elements)
            .map_err(|_| dimension_error("output size exceeds u32"))?;
        let mut builder = function.builder();
        builder.arg(&mut output);
        builder.arg(bias_values);
        builder.arg(&bias_offset);
        builder.arg(&element_count);
        for value in &self.feature_divisor {
            builder.arg(value);
        }
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: (element_count.div_ceil(THREADS_PER_BLOCK), 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
                .w()?;
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            self.output_shape.clone(),
        ))
    }
}

impl CustomOp3 for LastAxisMatMulBias {
    fn name(&self) -> &'static str {
        "a3s-fused-last-axis-matmul-bias"
    }

    fn cpu_fwd(
        &self,
        _left: &CpuStorage,
        _left_layout: &Layout,
        _right: &CpuStorage,
        _right_layout: &Layout,
        _bias: &CpuStorage,
        _bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused MatMul bias operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        left: &CudaStorage,
        left_layout: &Layout,
        right: &CudaStorage,
        right_layout: &Layout,
        bias: &CudaStorage,
        bias_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch(left, left_layout, right, right_layout, bias, bias_layout)
    }
}

fn as_i32(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| dimension_error("dimension exceeds the cuBLAS i32 bound"))
}

fn as_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| dimension_error("stride exceeds the cuBLAS i64 bound"))
}

fn dimension_error(message: &str) -> candle_core::Error {
    candle_core::Error::Msg(format!("CUDA MatMul bias {message}"))
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

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
    fn matmul_bias_is_byte_exact_across_unrelated_geometries_and_offsets() {
        let cuda = Device::new_cuda_with_stream(0).unwrap();
        for (prefix, rows, inner, columns) in [
            (vec![], 3, 5, 7),
            (vec![2], 17, 11, 29),
            (vec![2, 3], 13, 7, 19),
        ] {
            let batch = prefix.iter().product::<usize>().max(1);
            let left_storage = Tensor::from_iter(
                (0..(batch + 2) * rows * inner)
                    .map(|index| ((index * 37 % 509) as f32 - 254.0) / 29.0),
                &cuda,
            )
            .unwrap()
            .reshape((batch + 2, rows, inner))
            .unwrap();
            let mut left_dimensions = prefix.clone();
            left_dimensions.extend([rows, inner]);
            let left = left_storage
                .narrow(0, 1, batch)
                .unwrap()
                .reshape(left_dimensions.as_slice())
                .unwrap();
            let right_storage = Tensor::from_iter(
                (0..(inner + 2) * columns).map(|index| ((index * 19 % 257) as f32 - 128.0) / 31.0),
                &cuda,
            )
            .unwrap()
            .reshape((inner + 2, columns))
            .unwrap();
            let right = right_storage.narrow(0, 1, inner).unwrap();
            let bias_storage = Tensor::from_iter(
                (0..columns + 2).map(|index| ((index * 23 % 127) as f32 - 63.0) / 17.0),
                &cuda,
            )
            .unwrap();
            let bias = bias_storage.narrow(0, 1, columns).unwrap();
            assert!(left.is_contiguous() && right.is_contiguous() && bias.is_contiguous());

            for post_operation in [PostOperation::Bias, PostOperation::Swish] {
                let product =
                    crate::inference::graph::matrix_multiplication::broadcast(&left, &right)
                        .unwrap();
                let biased = product.broadcast_add(&bias).unwrap();
                let expected = match post_operation {
                    PostOperation::Bias => biased,
                    PostOperation::Swish => biased
                        .broadcast_mul(&candle_nn::ops::sigmoid(&biased).unwrap())
                        .unwrap(),
                };
                let actual = execute(&left, &right, &bias, post_operation).unwrap();
                cuda.synchronize().unwrap();

                let mut output_dimensions = left_dimensions[..left_dimensions.len() - 1].to_vec();
                output_dimensions.push(columns);
                assert_eq!(actual.dims(), output_dimensions);
                assert_eq!(
                    bits(&actual),
                    bits(&expected),
                    "geometry={prefix:?}x{rows}x{inner}x{columns} post={post_operation:?}",
                );
            }
        }
    }
}
