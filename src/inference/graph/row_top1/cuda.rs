use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Tensor};

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/row_top1_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_row_top1_last_finite_f32_v1";
const FUNCTION_NAME: &str = "row_top1_last_finite_f32";
const THREADS: u32 = 1024;

pub(super) fn execute(input: &Tensor) -> Result<Tensor> {
    execute_with_threads(input, THREADS)
}

pub(super) fn execute_with_threads(input: &Tensor, threads: u32) -> Result<Tensor> {
    let operation = RowTop1::new(input.layout(), threads)?;
    input.apply_op1_no_bwd(&operation)
}

#[derive(Clone)]
struct RowTop1 {
    output_shape: Shape,
    rows: u32,
    classes: u32,
    threads: u32,
}

impl RowTop1 {
    fn new(input: &Layout, threads: u32) -> Result<Self> {
        if !input.is_contiguous() {
            candle_core::bail!("row top-1 CUDA projection requires a contiguous input")
        }
        let mut dimensions = input.shape().dims().to_vec();
        let classes = dimensions.last().copied().unwrap_or_default();
        if dimensions.len() < 2 || classes == 0 || classes > (1 << 24) {
            candle_core::bail!("row top-1 CUDA projection received an invalid bounded shape")
        }
        if !matches!(threads, 128 | 256 | 512 | 1024) {
            candle_core::bail!("row top-1 CUDA projection received an invalid thread count")
        }
        let rows = input.shape().elem_count() / classes;
        let rows = u32::try_from(rows)
            .map_err(|_| candle_core::Error::Msg("row count exceeds u32".into()))?;
        let classes = u32::try_from(classes)
            .map_err(|_| candle_core::Error::Msg("class count exceeds u32".into()))?;
        let last_dimension = dimensions.len() - 1;
        dimensions[last_dimension] = 3;
        Ok(Self {
            output_shape: Shape::from_dims(&dimensions),
            rows,
            classes,
            threads,
        })
    }
}

impl CustomOp1 for RowTop1 {
    fn name(&self) -> &'static str {
        "a3s-row-top1-last-finite"
    }

    fn cpu_fwd(&self, _input: &CpuStorage, _input_layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused row top-1 projection is CUDA-only")
    }

    fn cuda_fwd(&self, input: &CudaStorage, input_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let input_values = input.as_cuda_slice::<f32>()?;
        let device = input.device();
        let function =
            device.get_or_load_custom_func(FUNCTION_NAME, MODULE_NAME, kernels::ROW_TOP1)?;
        let output_elements = usize::try_from(self.rows)
            .ok()
            .and_then(|rows| rows.checked_mul(3))
            .ok_or_else(|| candle_core::Error::Msg("row top-1 output size overflowed".into()))?;
        let output = unsafe { device.alloc::<f32>(output_elements)? };
        let input_offset = u64::try_from(input_layout.start_offset())
            .map_err(|_| candle_core::Error::Msg("row top-1 input offset exceeds u64".into()))?;
        let mut builder = function.builder();
        builder.arg(input_values);
        builder.arg(&output);
        builder.arg(&input_offset);
        builder.arg(&self.rows);
        builder.arg(&self.classes);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: (self.rows, 1, 1),
                    block_dim: (self.threads, 1, 1),
                    shared_mem_bytes: self.threads * 3 * 4,
                })
                .w()?
        };
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            self.output_shape.clone(),
        ))
    }
}
