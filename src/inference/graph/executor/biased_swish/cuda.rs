use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Result, Shape, Tensor};

use super::super::cuda_fast_divisor::FastDivisorU32;

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/biased_swish_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_biased_swish_f32_v1";
const FUNCTION_NAME: &str = "last_axis_bias_swish_f32";
const THREADS_PER_BLOCK: u32 = 512;

pub(super) fn execute(input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let operation = LastAxisBiasedSwish::new(input.layout(), bias.layout())?;
    input.apply_op2_no_bwd(bias, &operation)
}

#[derive(Clone)]
struct LastAxisBiasedSwish {
    shape: Shape,
    elements: usize,
    feature_divisor: [u32; 3],
}

impl LastAxisBiasedSwish {
    fn new(input: &Layout, bias: &Layout) -> Result<Self> {
        if !input.is_contiguous() || !bias.is_contiguous() {
            candle_core::bail!("biased Swish requires contiguous inputs")
        }
        let Some(&features) = input.shape().dims().last() else {
            candle_core::bail!("biased Swish requires a non-scalar input")
        };
        if features == 0 || bias.shape().dims() != [features] {
            candle_core::bail!("biased Swish requires an exact last-axis bias")
        }
        let elements = input.shape().elem_count();
        if elements == 0 || u32::try_from(elements).is_err() {
            candle_core::bail!("biased Swish exceeds the reviewed launch bound")
        }
        let features = u32::try_from(features).map_err(|_| {
            candle_core::Error::Msg("biased Swish feature count exceeds u32".into())
        })?;
        let feature_divisor = FastDivisorU32::new(features)
            .ok_or_else(|| candle_core::Error::Msg("biased Swish feature count is zero".into()))?
            .launch_parameters();
        Ok(Self {
            shape: input.shape().clone(),
            elements,
            feature_divisor,
        })
    }
}

impl CustomOp2 for LastAxisBiasedSwish {
    fn name(&self) -> &'static str {
        "a3s-fused-last-axis-biased-swish"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _bias: &CpuStorage,
        _bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused biased Swish operation is CUDA-only")
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

        let device = input.device();
        let function =
            device.get_or_load_custom_func(FUNCTION_NAME, MODULE_NAME, kernels::BIASED_SWISH)?;
        let output = unsafe { device.alloc::<f32>(self.elements)? };
        let input_offset = layout_offset(input_layout, "biased Swish input")?;
        let bias_offset = layout_offset(bias_layout, "biased Swish bias")?;
        let element_count = u32::try_from(self.elements)
            .map_err(|_| candle_core::Error::Msg("biased Swish size exceeds u32".into()))?;
        let mut builder = function.builder();
        builder.arg(input.as_cuda_slice::<f32>()?);
        builder.arg(bias.as_cuda_slice::<f32>()?);
        builder.arg(&output);
        builder.arg(&input_offset);
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
                .w()?
        };
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            self.shape.clone(),
        ))
    }
}

fn layout_offset(layout: &Layout, label: &str) -> Result<u64> {
    u64::try_from(layout.start_offset())
        .map_err(|_| candle_core::Error::Msg(format!("{label} offset exceeds u64")))
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
    fn biased_swish_is_byte_exact_across_unrelated_geometries() {
        let cuda = Device::new_cuda_with_stream(0).unwrap();
        for shape in [vec![2, 3, 5], vec![7, 11], vec![3, 17, 29]] {
            let features = *shape.last().unwrap();
            let elements = shape.iter().product::<usize>();
            let input = Tensor::from_iter(
                (0..elements).map(|index| ((index * 37 % 509) as f32 - 254.0) / 29.0),
                &cuda,
            )
            .unwrap()
            .reshape(shape.as_slice())
            .unwrap();
            let bias = Tensor::from_iter(
                (0..features).map(|index| ((index * 19 % 97) as f32 - 48.0) / 17.0),
                &cuda,
            )
            .unwrap();
            let biased = input.broadcast_add(&bias).unwrap();
            let expected = biased
                .broadcast_mul(&candle_nn::ops::sigmoid(&biased).unwrap())
                .unwrap();
            let actual = execute(&input, &bias).unwrap();
            cuda.synchronize().unwrap();

            assert_eq!(actual.dims(), shape);
            assert_eq!(bits(&actual), bits(&expected), "shape={shape:?}");
        }
    }
}
