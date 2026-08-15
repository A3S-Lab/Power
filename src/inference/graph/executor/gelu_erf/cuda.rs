use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Tensor};

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/gelu_erf_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_gelu_erf_f32_v2";
const FUNCTION_NAME: &str = "gelu_erf_f32";

pub(super) fn execute(input: &Tensor, divisor: f32, offset: f32, scale: f32) -> Result<Tensor> {
    let operation = GeluErf::new(input.layout(), divisor, offset, scale)?;
    input.apply_op1_no_bwd(&operation)
}

#[derive(Clone)]
struct GeluErf {
    shape: Shape,
    elements: usize,
    divisor: f32,
    offset: f32,
    scale: f32,
}

impl GeluErf {
    fn new(input: &Layout, divisor: f32, offset: f32, scale: f32) -> Result<Self> {
        if !input.is_contiguous() {
            candle_core::bail!("fused error-function activation requires contiguous inputs")
        }
        let shape = input.shape().clone();
        let elements = shape.elem_count();
        if elements == 0 || u32::try_from(elements).is_err() {
            candle_core::bail!("fused error-function activation exceeds the reviewed launch bound")
        }
        Ok(Self {
            shape,
            elements,
            divisor,
            offset,
            scale,
        })
    }
}

impl CustomOp1 for GeluErf {
    fn name(&self) -> &'static str {
        "a3s-fused-gelu-erf"
    }

    fn cpu_fwd(&self, _input: &CpuStorage, _input_layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused error-function activation is CUDA-only")
    }

    fn cuda_fwd(&self, input: &CudaStorage, input_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let input_values = input.as_cuda_slice::<f32>()?;
        let device = input.device();
        let function =
            device.get_or_load_custom_func(FUNCTION_NAME, MODULE_NAME, kernels::GELU_ERF)?;
        let output = unsafe { device.alloc::<f32>(self.elements)? };
        let input_offset = u64::try_from(input_layout.start_offset())
            .map_err(|_| candle_core::Error::Msg("activation input offset exceeds u64".into()))?;
        let element_count = u64::try_from(self.elements)
            .map_err(|_| candle_core::Error::Msg("activation size exceeds u64".into()))?;
        let mut builder = function.builder();
        builder.arg(input_values);
        builder.arg(&output);
        builder.arg(&input_offset);
        builder.arg(&element_count);
        builder.arg(&self.divisor);
        builder.arg(&self.offset);
        builder.arg(&self.scale);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(self.elements as u32))
                .w()?
        };
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            self.shape.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn fused_kernel_is_byte_exact_with_the_existing_cuda_sequence() {
        let cuda = Device::new_cuda(0).unwrap();
        let cpu = Device::Cpu;
        let input = Tensor::new(
            &[-17.0_f32, -3.0, -1.0, -0.0, 0.0, 0.25, 1.0, 3.0, 17.0],
            &cuda,
        )
        .unwrap();
        let divisor = Tensor::new(&[std::f32::consts::SQRT_2], &cuda).unwrap();
        let offset = Tensor::new(&[1.0_f32], &cuda).unwrap();
        let scale = Tensor::new(&[0.5_f32], &cuda).unwrap();

        let expected = input
            .broadcast_div(&divisor)
            .and_then(|value| value.erf())
            .and_then(|value| value.broadcast_add(&offset))
            .and_then(|value| input.broadcast_mul(&value))
            .and_then(|value| value.broadcast_mul(&scale))
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual = execute(&input, std::f32::consts::SQRT_2, 1.0_f32, 0.5_f32)
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}
