use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::CudaSlice;
#[cfg(test)]
use candle_core::InplaceOp3;
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Result, Shape, Tensor};

use super::super::super::cuda_reproducibility::REPRODUCIBLE_BATCH_ITEMS;

/// Multiplies one complete im2col buffer while retaining a fixed reduction
/// geometry for each contiguous group of leading-axis items.
pub(super) fn execute(columns: &Tensor, kernel: &Tensor, batch: usize) -> Result<Tensor> {
    let operation = ReproducibleSpatialProduct::new(columns.layout(), kernel.layout(), batch)?;
    columns.apply_op2_no_bwd(kernel, &operation)
}

#[cfg(test)]
pub(super) fn execute_into(
    output: &Tensor,
    columns: &Tensor,
    kernel: &Tensor,
    batch: usize,
) -> Result<()> {
    let operation = ReproducibleSpatialProduct::new(columns.layout(), kernel.layout(), batch)?;
    output.inplace_op3(columns, kernel, &operation)
}

#[derive(Clone, Copy)]
struct ReproducibleSpatialProduct {
    batch: usize,
    rows: usize,
    rows_per_item: usize,
    patch_elements: usize,
    output_channels: usize,
    column_elements: usize,
    kernel_elements: usize,
    output_elements: usize,
}

impl ReproducibleSpatialProduct {
    fn new(columns: &Layout, kernel: &Layout, batch: usize) -> Result<Self> {
        if !columns.is_contiguous() || !kernel.is_contiguous() {
            candle_core::bail!("CUDA spatial product requires contiguous tensors")
        }
        let (rows, patch_elements) = columns.shape().dims2()?;
        let (output_channels, kernel_patch_elements) = kernel.shape().dims2()?;
        if batch == 0
            || rows == 0
            || patch_elements == 0
            || output_channels == 0
            || kernel_patch_elements != patch_elements
            || rows % batch != 0
        {
            candle_core::bail!(
                "CUDA spatial product requires a large non-empty batch with matching geometry"
            )
        }
        let rows_per_item = rows / batch;
        let column_elements = rows
            .checked_mul(patch_elements)
            .ok_or_else(|| dimension_error("column size overflowed"))?;
        let kernel_elements = output_channels
            .checked_mul(patch_elements)
            .ok_or_else(|| dimension_error("kernel size overflowed"))?;
        let output_elements = rows
            .checked_mul(output_channels)
            .ok_or_else(|| dimension_error("output size overflowed"))?;
        let maximum_rows = rows_per_item
            .checked_mul(REPRODUCIBLE_BATCH_ITEMS)
            .ok_or_else(|| dimension_error("partition row count overflowed"))?;
        let maximum_input_elements = maximum_rows
            .checked_mul(patch_elements)
            .ok_or_else(|| dimension_error("partition input size overflowed"))?;
        let maximum_output_elements = maximum_rows
            .checked_mul(output_channels)
            .ok_or_else(|| dimension_error("partition output size overflowed"))?;
        for (name, value) in [
            ("partition rows", maximum_rows),
            ("patch elements", patch_elements),
            ("output channels", output_channels),
        ] {
            i32::try_from(value)
                .map_err(|_| dimension_error(&format!("{name} exceeds the cuBLAS i32 bound")))?;
        }
        for (name, value) in [
            ("partition input elements", maximum_input_elements),
            ("partition output elements", maximum_output_elements),
        ] {
            i64::try_from(value)
                .map_err(|_| dimension_error(&format!("{name} exceeds the cuBLAS i64 bound")))?;
        }
        Ok(Self {
            batch,
            rows,
            rows_per_item,
            patch_elements,
            output_channels,
            column_elements,
            kernel_elements,
            output_elements,
        })
    }

    fn launch_into(
        self,
        output: &mut CudaSlice<f32>,
        columns: &CudaStorage,
        columns_layout: &Layout,
        kernel: &CudaStorage,
        kernel_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::cublas::{result, sys};
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut, DeviceSlice};
        use candle_core::cuda_backend::WrapErr;

        let column_start = columns_layout.start_offset();
        let kernel_start = kernel_layout.start_offset();
        let column_values = columns.as_cuda_slice::<f32>()?;
        let kernel_values = kernel.as_cuda_slice::<f32>()?;
        let column_values = column_values.slice(column_start..column_start + self.column_elements);
        let kernel_values = kernel_values.slice(kernel_start..kernel_start + self.kernel_elements);
        let device = columns.device();
        let blas = device.cublas_handle();
        if output.len() != self.output_elements {
            candle_core::bail!("CUDA spatial product destination has the wrong length")
        }
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let compute_type = if candle_core::cuda_backend::gemm_reduced_precision_f32() {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
        } else {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        };

        for batch_offset in (0..self.batch).step_by(REPRODUCIBLE_BATCH_ITEMS) {
            let batch_items = (self.batch - batch_offset).min(REPRODUCIBLE_BATCH_ITEMS);
            let rows = batch_items * self.rows_per_item;
            let column_offset = batch_offset * self.rows_per_item * self.patch_elements;
            let output_offset = batch_offset * self.rows_per_item * self.output_channels;
            let column_chunk =
                column_values.slice(column_offset..column_offset + rows * self.patch_elements);
            let mut output_chunk =
                output.slice_mut(output_offset..output_offset + rows * self.output_channels);
            let stream = output_chunk.stream().clone();
            let (kernel_pointer, _kernel_guard) = kernel_values.device_ptr(&stream);
            let (column_pointer, _column_guard) = column_chunk.device_ptr(&stream);
            let (output_pointer, _output_guard) = output_chunk.device_ptr_mut(&stream);
            unsafe {
                result::gemm_strided_batched_ex(
                    *blas.handle(),
                    sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    as_i32(self.output_channels)?,
                    as_i32(rows)?,
                    as_i32(self.patch_elements)?,
                    &alpha as *const f32 as *const _,
                    kernel_pointer as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    as_i32(self.patch_elements)?,
                    as_i64(self.kernel_elements)?,
                    column_pointer as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    as_i32(self.patch_elements)?,
                    as_i64(rows * self.patch_elements)?,
                    &beta as *const f32 as *const _,
                    output_pointer as *mut _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    as_i32(self.output_channels)?,
                    as_i64(rows * self.output_channels)?,
                    1,
                    compute_type,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
            }
            .w()?;
            drop(_output_guard);
            drop(_column_guard);
            drop(_kernel_guard);
        }

        Ok(())
    }

    fn launch(
        self,
        columns: &CudaStorage,
        columns_layout: &Layout,
        kernel: &CudaStorage,
        kernel_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let device = columns.device();
        let mut output = unsafe { device.alloc::<f32>(self.output_elements)? };
        self.launch_into(&mut output, columns, columns_layout, kernel, kernel_layout)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from_dims(&[self.rows, self.output_channels]),
        ))
    }
}

impl CustomOp2 for ReproducibleSpatialProduct {
    fn name(&self) -> &'static str {
        "a3s-cuda-reproducible-spatial-product"
    }

    fn cpu_fwd(
        &self,
        _columns: &CpuStorage,
        _columns_layout: &Layout,
        _kernel: &CpuStorage,
        _kernel_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the reproducible spatial product is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        columns: &CudaStorage,
        columns_layout: &Layout,
        kernel: &CudaStorage,
        kernel_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch(columns, columns_layout, kernel, kernel_layout)
    }
}

#[cfg(test)]
impl InplaceOp3 for ReproducibleSpatialProduct {
    fn name(&self) -> &'static str {
        "a3s-cuda-reproducible-spatial-product-into"
    }

    fn cpu_fwd(
        &self,
        _output: &mut CpuStorage,
        _output_layout: &Layout,
        _columns: &CpuStorage,
        _columns_layout: &Layout,
        _kernel: &CpuStorage,
        _kernel_layout: &Layout,
    ) -> Result<()> {
        candle_core::bail!("the preallocated spatial product is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        output: &mut CudaStorage,
        output_layout: &Layout,
        columns: &CudaStorage,
        columns_layout: &Layout,
        kernel: &CudaStorage,
        kernel_layout: &Layout,
    ) -> Result<()> {
        let expected = Shape::from_dims(&[self.rows, self.output_channels]);
        if !output_layout.is_contiguous()
            || output_layout.start_offset() != 0
            || output_layout.shape() != &expected
        {
            candle_core::bail!(
                "spatial product destination must be one exact contiguous output tensor"
            )
        }
        self.launch_into(
            output.as_cuda_slice_mut::<f32>()?,
            columns,
            columns_layout,
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
    candle_core::Error::Msg(format!("CUDA spatial product {message}"))
}
