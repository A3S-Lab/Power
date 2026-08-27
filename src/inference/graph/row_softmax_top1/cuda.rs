use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp1, CustomOp2, Layout, Result, Shape, Tensor};

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/row_softmax_top1_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_row_softmax_top1_last_finite_f32_v2";
const FUNCTION_NAME: &str = "row_softmax_top1_last_finite_f32";
const BIAS_FUNCTION_NAME: &str = "row_bias_softmax_top1_last_finite_f32";
const THREADS: u32 = 1024;

pub(super) fn execute(input: &Tensor) -> Result<Tensor> {
    let operation = RowSoftmaxTop1::new(input.layout())?;
    input.apply_op1_no_bwd(&operation)
}

pub(super) fn execute_with_bias(input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let operation = RowSoftmaxTop1::new(input.layout())?;
    operation.validate_bias(bias.layout())?;
    input.apply_op2_no_bwd(bias, &operation)
}

#[derive(Clone)]
struct RowSoftmaxTop1 {
    output_shape: Shape,
    rows: u32,
    classes: u32,
}

impl RowSoftmaxTop1 {
    fn new(input: &Layout) -> Result<Self> {
        if !input.is_contiguous() {
            candle_core::bail!("row Softmax top-1 CUDA projection requires contiguous input")
        }
        let mut dimensions = input.shape().dims().to_vec();
        let classes = dimensions.last().copied().unwrap_or_default();
        if dimensions.len() < 2 || classes == 0 || classes > (1 << 24) {
            candle_core::bail!("row Softmax top-1 CUDA projection received an invalid shape")
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
        })
    }

    fn validate_bias(&self, bias: &Layout) -> Result<()> {
        if !bias.is_contiguous() || bias.shape().dims1()? != self.classes as usize {
            candle_core::bail!(
                "row Softmax top-1 CUDA bias requires exact contiguous [classes] shape"
            )
        }
        Ok(())
    }

    fn allocate_output(
        &self,
        device: &candle_core::CudaDevice,
    ) -> Result<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>> {
        let output_elements = usize::try_from(self.rows)
            .ok()
            .and_then(|rows| rows.checked_mul(3))
            .ok_or_else(|| candle_core::Error::Msg("row output size overflowed".into()))?;
        unsafe { device.alloc::<f32>(output_elements) }
    }
}

impl CustomOp1 for RowSoftmaxTop1 {
    fn name(&self) -> &'static str {
        "a3s-row-softmax-top1-last-finite"
    }

    fn cpu_fwd(&self, _input: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused row Softmax top-1 projection is CUDA-only")
    }

    fn cuda_fwd(&self, input: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let input_values = input.as_cuda_slice::<f32>()?;
        let device = input.device();
        let function = device.get_or_load_custom_func(
            FUNCTION_NAME,
            MODULE_NAME,
            kernels::ROW_SOFTMAX_TOP1,
        )?;
        let output = self.allocate_output(device)?;
        let input_offset = u64::try_from(layout.start_offset())
            .map_err(|_| candle_core::Error::Msg("row input offset exceeds u64".into()))?;
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
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: THREADS * 4 * 4,
                })
                .w()?
        };
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            self.output_shape.clone(),
        ))
    }
}

impl CustomOp2 for RowSoftmaxTop1 {
    fn name(&self) -> &'static str {
        "a3s-row-bias-softmax-top1-last-finite"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _bias: &CpuStorage,
        _bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused row bias Softmax top-1 projection is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        bias: &CudaStorage,
        bias_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        self.validate_bias(bias_layout)?;
        let input_values = input.as_cuda_slice::<f32>()?;
        let bias_values = bias.as_cuda_slice::<f32>()?;
        let device = input.device();
        let function = device.get_or_load_custom_func(
            BIAS_FUNCTION_NAME,
            MODULE_NAME,
            kernels::ROW_SOFTMAX_TOP1,
        )?;
        let output = self.allocate_output(device)?;
        let input_offset = u64::try_from(input_layout.start_offset())
            .map_err(|_| candle_core::Error::Msg("row input offset exceeds u64".into()))?;
        let bias_offset = u64::try_from(bias_layout.start_offset())
            .map_err(|_| candle_core::Error::Msg("row bias offset exceeds u64".into()))?;
        let mut builder = function.builder();
        builder.arg(input_values);
        builder.arg(bias_values);
        builder.arg(&output);
        builder.arg(&input_offset);
        builder.arg(&bias_offset);
        builder.arg(&self.rows);
        builder.arg(&self.classes);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: (self.rows, 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: THREADS * 4 * 4,
                })
                .w()?
        };
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            self.output_shape.clone(),
        ))
    }
}
