use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape, Tensor};

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/contiguous_transpose_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_contiguous_transpose_f32_v1";
const FUNCTION_NAME: &str = "contiguous_last_two_transpose_u32_f32";
const TILE: usize = 32;
const MAXIMUM_GRID_Y_OR_Z: usize = 65_535;

pub(super) fn try_materialize(input: &Tensor) -> Result<Option<Tensor>> {
    if input.dtype() != DType::F32 || !input.device().is_cuda() || input.is_contiguous() {
        return Ok(None);
    }
    let Some(operation) = LastTwoTranspose::from_layout(input.layout()) else {
        return Ok(None);
    };
    input.apply_op1_no_bwd(&operation).map(Some)
}

#[derive(Clone)]
struct LastTwoTranspose {
    shape: Vec<usize>,
    prefix: u32,
    rows: u32,
    columns: u32,
    elements: usize,
}

impl LastTwoTranspose {
    fn from_layout(layout: &Layout) -> Option<Self> {
        let dimensions = layout.dims();
        let strides = layout.stride();
        if dimensions.len() < 2 || dimensions.len() != strides.len() || layout.is_contiguous() {
            return None;
        }
        let rank = dimensions.len();
        let rows = dimensions[rank - 2];
        let columns = dimensions[rank - 1];
        if rows == 0 || columns == 0 || strides[rank - 2] != 1 || strides[rank - 1] != rows {
            return None;
        }

        let matrix_elements = rows.checked_mul(columns)?;
        let mut expected_stride = matrix_elements;
        for axis in (0..rank - 2).rev() {
            if strides[axis] != expected_stride {
                return None;
            }
            expected_stride = expected_stride.checked_mul(dimensions[axis])?;
        }
        let prefix = dimensions[..rank - 2]
            .iter()
            .try_fold(1_usize, |total, dimension| total.checked_mul(*dimension))?;
        let elements = prefix.checked_mul(matrix_elements)?;
        if elements > u32::MAX as usize
            || prefix > MAXIMUM_GRID_Y_OR_Z
            || rows.div_ceil(TILE) > u32::MAX as usize
            || columns.div_ceil(TILE) > MAXIMUM_GRID_Y_OR_Z
        {
            return None;
        }
        Some(Self {
            shape: dimensions.to_vec(),
            prefix: u32::try_from(prefix).ok()?,
            rows: u32::try_from(rows).ok()?,
            columns: u32::try_from(columns).ok()?,
            elements,
        })
    }

    fn launch(&self, input: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let device = input.device();
        let function = device.get_or_load_custom_func(
            FUNCTION_NAME,
            MODULE_NAME,
            kernels::CONTIGUOUS_TRANSPOSE,
        )?;
        let output = unsafe { device.alloc::<f32>(self.elements)? };
        let input_offset = u64::try_from(layout.start_offset()).map_err(|_| {
            candle_core::Error::Msg("CUDA transpose input offset exceeds u64".into())
        })?;
        let mut builder = function.builder();
        builder.arg(input.as_cuda_slice::<f32>()?);
        builder.arg(&output);
        builder.arg(&input_offset);
        builder.arg(&self.rows);
        builder.arg(&self.columns);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    self.rows.div_ceil(TILE as u32),
                    self.columns.div_ceil(TILE as u32),
                    self.prefix,
                ),
                block_dim: (TILE as u32, 8, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from_dims(&self.shape),
        ))
    }
}

impl CustomOp1 for LastTwoTranspose {
    fn name(&self) -> &'static str {
        "a3s-cuda-contiguous-last-two-transpose"
    }

    fn cpu_fwd(&self, _input: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the tiled contiguous transpose is CUDA-only")
    }

    fn cuda_fwd(&self, input: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        self.launch(input, layout)
    }
}
