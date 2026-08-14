use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Result, Shape, Tensor};

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/gated_hard_sigmoid_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_gated_hard_sigmoid_f32_v1";
const SAME_SHAPE_FUNCTION: &str = "gated_hard_sigmoid_mul_f32";
const CHANNEL_GATE_FUNCTION: &str = "gated_hard_sigmoid_channel_mul_f32";

pub(super) fn mul(multiplicand: &Tensor, gate: &Tensor, alpha: f32, beta: f32) -> Result<Tensor> {
    let operation = GatedHardSigmoid::new(multiplicand.layout(), gate.layout(), alpha, beta)?;
    multiplicand.apply_op2_no_bwd(gate, &operation)
}

#[derive(Clone)]
struct GatedHardSigmoid {
    shape: Shape,
    elements: usize,
    layout: GatedLayout,
    alpha: f32,
    beta: f32,
}

#[derive(Clone, Copy)]
enum GatedLayout {
    SameShape,
    ChannelGate { spatial_elements: u64 },
}

impl GatedHardSigmoid {
    fn new(multiplicand: &Layout, gate: &Layout, alpha: f32, beta: f32) -> Result<Self> {
        if !multiplicand.is_contiguous() || !gate.is_contiguous() {
            candle_core::bail!("fused gated HardSigmoid requires contiguous inputs")
        }
        let multiplicand_dimensions = four_dimensions(multiplicand)?;
        let gate_dimensions = four_dimensions(gate)?;
        let layout = if multiplicand_dimensions == gate_dimensions {
            GatedLayout::SameShape
        } else if multiplicand_dimensions[0] == gate_dimensions[0]
            && multiplicand_dimensions[1] == gate_dimensions[1]
            && gate_dimensions[2] == 1
            && gate_dimensions[3] == 1
        {
            let spatial_elements = multiplicand_dimensions[2]
                .checked_mul(multiplicand_dimensions[3])
                .ok_or_else(|| {
                    candle_core::Error::Msg("gated activation spatial size overflowed".into())
                })?;
            GatedLayout::ChannelGate {
                spatial_elements: u64::try_from(spatial_elements).map_err(|_| {
                    candle_core::Error::Msg("gated activation spatial size exceeds u64".into())
                })?,
            }
        } else {
            candle_core::bail!(
                "fused gated HardSigmoid requires equal shapes or an exact NCHW channel gate"
            )
        };
        let shape = multiplicand.shape().clone();
        let elements = shape.elem_count();
        if elements == 0 || u32::try_from(elements).is_err() {
            candle_core::bail!("fused gated HardSigmoid exceeds the reviewed launch bound")
        }
        Ok(Self {
            shape,
            elements,
            layout,
            alpha,
            beta,
        })
    }
}

impl CustomOp2 for GatedHardSigmoid {
    fn name(&self) -> &'static str {
        "a3s-fused-gated-hard-sigmoid"
    }

    fn cpu_fwd(
        &self,
        _multiplicand: &CpuStorage,
        _multiplicand_layout: &Layout,
        _gate: &CpuStorage,
        _gate_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused gated HardSigmoid operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        multiplicand: &CudaStorage,
        multiplicand_layout: &Layout,
        gate: &CudaStorage,
        gate_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let multiplicand_values = multiplicand.as_cuda_slice::<f32>()?;
        let gate_values = gate.as_cuda_slice::<f32>()?;
        let device = multiplicand.device();
        let (function_name, spatial_elements) = match self.layout {
            GatedLayout::SameShape => (SAME_SHAPE_FUNCTION, 1),
            GatedLayout::ChannelGate { spatial_elements } => {
                (CHANNEL_GATE_FUNCTION, spatial_elements)
            }
        };
        let function = device.get_or_load_custom_func(
            function_name,
            MODULE_NAME,
            kernels::GATED_HARD_SIGMOID,
        )?;
        let output = unsafe { device.alloc::<f32>(self.elements)? };
        let element_count = u64::try_from(self.elements)
            .map_err(|_| candle_core::Error::Msg("gated activation size exceeds u64".into()))?;
        let multiplicand_offset = u64::try_from(multiplicand_layout.start_offset())
            .map_err(|_| candle_core::Error::Msg("multiplicand offset exceeds u64".into()))?;
        let gate_offset = u64::try_from(gate_layout.start_offset())
            .map_err(|_| candle_core::Error::Msg("gate offset exceeds u64".into()))?;
        let mut builder = function.builder();
        builder.arg(multiplicand_values);
        builder.arg(gate_values);
        builder.arg(&output);
        builder.arg(&multiplicand_offset);
        builder.arg(&gate_offset);
        builder.arg(&element_count);
        builder.arg(&spatial_elements);
        builder.arg(&self.alpha);
        builder.arg(&self.beta);
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

fn four_dimensions(layout: &Layout) -> Result<[usize; 4]> {
    let [batch, channels, height, width] = layout.shape().dims() else {
        candle_core::bail!("fused gated HardSigmoid requires rank-four NCHW inputs")
    };
    Ok([*batch, *channels, *height, *width])
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
        let multiplicand = Tensor::new(&[-7.0_f32, -1.0, -0.0, 0.0, 0.25, 1.0, 3.0, 17.0], &cuda)
            .unwrap()
            .reshape((1, 2, 2, 2))
            .unwrap();
        let same_shape_gate = Tensor::new(&[-9.0_f32, -3.0, -2.5, -0.0, 0.0, 2.5, 3.0, 9.0], &cuda)
            .unwrap()
            .reshape((1, 2, 2, 2))
            .unwrap();
        let channel_gate = Tensor::new(&[-2.5_f32, 2.5], &cuda)
            .unwrap()
            .reshape((1, 2, 1, 1))
            .unwrap();

        for gate in [&same_shape_gate, &channel_gate] {
            for (alpha, beta) in [(0.166_666_7_f32, 0.5_f32), (0.2_f32, 0.5_f32)] {
                let expected = (gate * f64::from(alpha))
                    .and_then(|value| value.affine(1.0, f64::from(beta)))
                    .and_then(|value| value.clamp(0.0, 1.0))
                    .and_then(|value| multiplicand.broadcast_mul(&value))
                    .unwrap()
                    .to_device(&cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let actual = mul(&multiplicand, gate, alpha, beta)
                    .unwrap()
                    .to_device(&cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                assert_eq!(actual, expected);
            }
        }
    }
}
