use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::CudaSlice;
#[cfg(test)]
use candle_core::InplaceOp3;
use candle_core::{CpuStorage, CudaStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};

use super::super::cuda_fast_divisor::FastDivisorU32;

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/biased_activation_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_biased_activation_f32_v4";
const BIAS_FUNCTION: &str = "channel_bias_f32";
const RELU_FUNCTION: &str = "channel_bias_relu_f32";
const GELU_FUNCTION: &str = "channel_bias_gelu_erf_f32";
const RESIDUAL_FUNCTION: &str = "channel_bias_residual_f32";
const SAME_SHAPE_GATE_FUNCTION: &str = "channel_bias_gated_hard_sigmoid_mul_f32";
const CHANNEL_GATE_FUNCTION: &str = "channel_bias_gated_hard_sigmoid_channel_mul_f32";
const THREADS_PER_BLOCK: u32 = 512;

pub(super) fn bias(input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let operation = ChannelBiasActivation::new(input.layout(), bias.layout(), Activation::Bias)?;
    input.apply_op2_no_bwd(bias, &operation)
}

pub(super) fn relu(input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let operation = ChannelBiasActivation::new(input.layout(), bias.layout(), Activation::Relu)?;
    input.apply_op2_no_bwd(bias, &operation)
}

#[cfg(test)]
pub(super) fn relu_into(output: &Tensor, input: &Tensor, bias: &Tensor) -> Result<()> {
    let operation = ChannelBiasActivation::new(input.layout(), bias.layout(), Activation::Relu)?;
    output.inplace_op3(input, bias, &operation)
}

pub(super) fn gelu_erf(
    input: &Tensor,
    bias: &Tensor,
    divisor: f32,
    offset: f32,
    scale: f32,
) -> Result<Tensor> {
    let operation = ChannelBiasActivation::new(
        input.layout(),
        bias.layout(),
        Activation::GeluErf {
            divisor,
            offset,
            scale,
        },
    )?;
    input.apply_op2_no_bwd(bias, &operation)
}

pub(super) fn bias_residual(input: &Tensor, residual: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let operation = ChannelBiasResidual::new(input.layout(), residual.layout(), bias.layout())?;
    input.apply_op3_no_bwd(residual, bias, &operation)
}

pub(super) fn gated_hard_sigmoid_mul(
    multiplicand: &Tensor,
    gate: &Tensor,
    bias: &Tensor,
    alpha: f32,
    beta: f32,
) -> Result<Tensor> {
    let operation = BiasedGatedHardSigmoid::new(
        multiplicand.layout(),
        gate.layout(),
        bias.layout(),
        alpha,
        beta,
    )?;
    multiplicand.apply_op3_no_bwd(gate, bias, &operation)
}

#[derive(Clone)]
struct ChannelBiasActivation {
    shape: Shape,
    elements: usize,
    index_divisors: [u32; 5],
    activation: Activation,
}

#[derive(Clone, Copy)]
enum Activation {
    Bias,
    Relu,
    GeluErf {
        divisor: f32,
        offset: f32,
        scale: f32,
    },
}

impl ChannelBiasActivation {
    fn new(input: &Layout, bias: &Layout, activation: Activation) -> Result<Self> {
        let dimensions = validate_channel_bias(input, bias)?;
        let elements = input.shape().elem_count();
        if elements == 0 || u32::try_from(elements).is_err() {
            candle_core::bail!("biased activation exceeds the reviewed launch bound")
        }
        let spatial_elements = dimensions[2]
            .checked_mul(dimensions[3])
            .ok_or_else(|| candle_core::Error::Msg("activation spatial size overflowed".into()))?;
        let channels = u32::try_from(dimensions[1])
            .map_err(|_| candle_core::Error::Msg("channel count exceeds u32".into()))?;
        let spatial_elements = u32::try_from(spatial_elements)
            .map_err(|_| candle_core::Error::Msg("spatial size exceeds u32".into()))?;
        Ok(Self {
            shape: input.shape().clone(),
            elements,
            index_divisors: channel_index_divisors(channels, spatial_elements)?,
            activation,
        })
    }

    fn launch_into(
        &self,
        output: &mut CudaSlice<f32>,
        input: &CudaStorage,
        input_layout: &Layout,
        bias: &CudaStorage,
        bias_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        if output.len() != self.elements {
            candle_core::bail!(
                "biased activation destination length does not match the input geometry"
            )
        }
        let input_values = input.as_cuda_slice::<f32>()?;
        let bias_values = bias.as_cuda_slice::<f32>()?;
        let device = input.device();
        let (function_name, divisor, offset, scale) = match self.activation {
            Activation::Bias => (BIAS_FUNCTION, 1.0, 0.0, 1.0),
            Activation::Relu => (RELU_FUNCTION, 1.0, 0.0, 1.0),
            Activation::GeluErf {
                divisor,
                offset,
                scale,
            } => (GELU_FUNCTION, divisor, offset, scale),
        };
        let function = device.get_or_load_custom_func(
            function_name,
            MODULE_NAME,
            kernels::BIASED_ACTIVATION,
        )?;
        let input_offset = layout_offset(input_layout, "activation input")?;
        let bias_offset = layout_offset(bias_layout, "activation bias")?;
        let element_count = u32::try_from(self.elements)
            .map_err(|_| candle_core::Error::Msg("activation size exceeds u32".into()))?;
        let mut builder = function.builder();
        builder.arg(input_values);
        builder.arg(bias_values);
        builder.arg(&*output);
        builder.arg(&input_offset);
        builder.arg(&bias_offset);
        builder.arg(&element_count);
        for value in &self.index_divisors {
            builder.arg(value);
        }
        builder.arg(&divisor);
        builder.arg(&offset);
        builder.arg(&scale);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: (element_count.div_ceil(THREADS_PER_BLOCK), 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
                .w()?
        };
        Ok(())
    }
}

impl CustomOp2 for ChannelBiasActivation {
    fn name(&self) -> &'static str {
        "a3s-fused-channel-bias-activation"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _bias: &CpuStorage,
        _bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused channel-bias activation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        bias: &CudaStorage,
        bias_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let device = input.device();
        let mut output = unsafe { device.alloc::<f32>(self.elements)? };
        self.launch_into(&mut output, input, input_layout, bias, bias_layout)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            self.shape.clone(),
        ))
    }
}

#[cfg(test)]
impl InplaceOp3 for ChannelBiasActivation {
    fn name(&self) -> &'static str {
        "a3s-fused-channel-bias-activation-into"
    }

    fn cpu_fwd(
        &self,
        _output: &mut CpuStorage,
        _output_layout: &Layout,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _bias: &CpuStorage,
        _bias_layout: &Layout,
    ) -> Result<()> {
        candle_core::bail!("the preallocated biased activation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        output: &mut CudaStorage,
        output_layout: &Layout,
        input: &CudaStorage,
        input_layout: &Layout,
        bias: &CudaStorage,
        bias_layout: &Layout,
    ) -> Result<()> {
        if !output_layout.is_contiguous()
            || output_layout.start_offset() != 0
            || output_layout.shape() != &self.shape
        {
            candle_core::bail!(
                "biased activation destination must be one exact contiguous output tensor"
            )
        }
        self.launch_into(
            output.as_cuda_slice_mut::<f32>()?,
            input,
            input_layout,
            bias,
            bias_layout,
        )
    }
}

#[derive(Clone)]
struct ChannelBiasResidual {
    shape: Shape,
    elements: usize,
    index_divisors: [u32; 5],
}

impl ChannelBiasResidual {
    fn new(input: &Layout, residual: &Layout, bias: &Layout) -> Result<Self> {
        let dimensions = validate_channel_bias(input, bias)?;
        if !residual.is_contiguous() || residual.shape() != input.shape() {
            candle_core::bail!(
                "biased residual requires a contiguous addend matching the NCHW input"
            )
        }
        let elements = input.shape().elem_count();
        if elements == 0 || u32::try_from(elements).is_err() {
            candle_core::bail!("biased residual exceeds the reviewed launch bound")
        }
        let spatial_elements = dimensions[2]
            .checked_mul(dimensions[3])
            .ok_or_else(|| candle_core::Error::Msg("residual spatial size overflowed".into()))?;
        let channels = u32::try_from(dimensions[1])
            .map_err(|_| candle_core::Error::Msg("channel count exceeds u32".into()))?;
        let spatial_elements = u32::try_from(spatial_elements)
            .map_err(|_| candle_core::Error::Msg("spatial size exceeds u32".into()))?;
        Ok(Self {
            shape: input.shape().clone(),
            elements,
            index_divisors: channel_index_divisors(channels, spatial_elements)?,
        })
    }
}

impl CustomOp3 for ChannelBiasResidual {
    fn name(&self) -> &'static str {
        "a3s-fused-channel-bias-residual"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _residual: &CpuStorage,
        _residual_layout: &Layout,
        _bias: &CpuStorage,
        _bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused channel-bias residual operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        residual: &CudaStorage,
        residual_layout: &Layout,
        bias: &CudaStorage,
        bias_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let input_values = input.as_cuda_slice::<f32>()?;
        let residual_values = residual.as_cuda_slice::<f32>()?;
        let bias_values = bias.as_cuda_slice::<f32>()?;
        let device = input.device();
        let function = device.get_or_load_custom_func(
            RESIDUAL_FUNCTION,
            MODULE_NAME,
            kernels::BIASED_ACTIVATION,
        )?;
        let output = unsafe { device.alloc::<f32>(self.elements)? };
        let input_offset = layout_offset(input_layout, "residual input")?;
        let residual_offset = layout_offset(residual_layout, "residual addend")?;
        let bias_offset = layout_offset(bias_layout, "residual bias")?;
        let element_count = u32::try_from(self.elements)
            .map_err(|_| candle_core::Error::Msg("residual size exceeds u32".into()))?;
        let mut builder = function.builder();
        builder.arg(input_values);
        builder.arg(residual_values);
        builder.arg(bias_values);
        builder.arg(&output);
        builder.arg(&input_offset);
        builder.arg(&residual_offset);
        builder.arg(&bias_offset);
        builder.arg(&element_count);
        for value in &self.index_divisors {
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

#[derive(Clone)]
struct BiasedGatedHardSigmoid {
    shape: Shape,
    elements: usize,
    index_divisors: [u32; 5],
    layout: GateLayout,
    alpha: f32,
    beta: f32,
}

#[derive(Clone, Copy)]
enum GateLayout {
    SameShape,
    ChannelGate,
}

impl BiasedGatedHardSigmoid {
    fn new(
        multiplicand: &Layout,
        gate: &Layout,
        bias: &Layout,
        alpha: f32,
        beta: f32,
    ) -> Result<Self> {
        if !multiplicand.is_contiguous() || !gate.is_contiguous() || !bias.is_contiguous() {
            candle_core::bail!("biased gated HardSigmoid requires contiguous inputs")
        }
        let multiplicand_dimensions = four_dimensions(multiplicand)?;
        let gate_dimensions = validate_channel_bias(gate, bias)?;
        let layout = if multiplicand_dimensions == gate_dimensions {
            GateLayout::SameShape
        } else if multiplicand_dimensions[0] == gate_dimensions[0]
            && multiplicand_dimensions[1] == gate_dimensions[1]
            && gate_dimensions[2] == 1
            && gate_dimensions[3] == 1
        {
            GateLayout::ChannelGate
        } else {
            candle_core::bail!(
                "biased gated HardSigmoid requires equal shapes or an exact NCHW channel gate"
            )
        };
        let elements = multiplicand.shape().elem_count();
        if elements == 0 || u32::try_from(elements).is_err() {
            candle_core::bail!("biased gated HardSigmoid exceeds the reviewed launch bound")
        }
        let spatial_elements = multiplicand_dimensions[2]
            .checked_mul(multiplicand_dimensions[3])
            .ok_or_else(|| {
                candle_core::Error::Msg("gated activation spatial size overflowed".into())
            })?;
        let channels = u32::try_from(multiplicand_dimensions[1])
            .map_err(|_| candle_core::Error::Msg("channel count exceeds u32".into()))?;
        let spatial_elements = u32::try_from(spatial_elements)
            .map_err(|_| candle_core::Error::Msg("spatial size exceeds u32".into()))?;
        Ok(Self {
            shape: multiplicand.shape().clone(),
            elements,
            index_divisors: channel_index_divisors(channels, spatial_elements)?,
            layout,
            alpha,
            beta,
        })
    }
}

impl CustomOp3 for BiasedGatedHardSigmoid {
    fn name(&self) -> &'static str {
        "a3s-fused-biased-gated-hard-sigmoid"
    }

    fn cpu_fwd(
        &self,
        _multiplicand: &CpuStorage,
        _multiplicand_layout: &Layout,
        _gate: &CpuStorage,
        _gate_layout: &Layout,
        _bias: &CpuStorage,
        _bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused biased gated HardSigmoid is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        multiplicand: &CudaStorage,
        multiplicand_layout: &Layout,
        gate: &CudaStorage,
        gate_layout: &Layout,
        bias: &CudaStorage,
        bias_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let multiplicand_values = multiplicand.as_cuda_slice::<f32>()?;
        let gate_values = gate.as_cuda_slice::<f32>()?;
        let bias_values = bias.as_cuda_slice::<f32>()?;
        let device = multiplicand.device();
        let function_name = match self.layout {
            GateLayout::SameShape => SAME_SHAPE_GATE_FUNCTION,
            GateLayout::ChannelGate => CHANNEL_GATE_FUNCTION,
        };
        let function = device.get_or_load_custom_func(
            function_name,
            MODULE_NAME,
            kernels::BIASED_ACTIVATION,
        )?;
        let output = unsafe { device.alloc::<f32>(self.elements)? };
        let multiplicand_offset = layout_offset(multiplicand_layout, "multiplicand")?;
        let gate_offset = layout_offset(gate_layout, "gate")?;
        let bias_offset = layout_offset(bias_layout, "gate bias")?;
        let element_count = u32::try_from(self.elements)
            .map_err(|_| candle_core::Error::Msg("gated activation size exceeds u32".into()))?;
        let mut builder = function.builder();
        builder.arg(multiplicand_values);
        builder.arg(gate_values);
        builder.arg(bias_values);
        builder.arg(&output);
        builder.arg(&multiplicand_offset);
        builder.arg(&gate_offset);
        builder.arg(&bias_offset);
        builder.arg(&element_count);
        for value in &self.index_divisors {
            builder.arg(value);
        }
        builder.arg(&self.alpha);
        builder.arg(&self.beta);
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

fn channel_index_divisors(channels: u32, spatial_elements: u32) -> Result<[u32; 5]> {
    let [spatial_multiplier, spatial_shift, _] = FastDivisorU32::new(spatial_elements)
        .ok_or_else(|| candle_core::Error::Msg("spatial size is zero".into()))?
        .launch_parameters();
    let [channels_multiplier, channels_shift, channels] = FastDivisorU32::new(channels)
        .ok_or_else(|| candle_core::Error::Msg("channel count is zero".into()))?
        .launch_parameters();
    Ok([
        spatial_multiplier,
        spatial_shift,
        channels_multiplier,
        channels_shift,
        channels,
    ])
}

fn validate_channel_bias(input: &Layout, bias: &Layout) -> Result<[usize; 4]> {
    if !input.is_contiguous() || !bias.is_contiguous() {
        candle_core::bail!("biased activation requires contiguous inputs")
    }
    let dimensions = four_dimensions(input)?;
    let bias_dimensions = four_dimensions(bias)?;
    if bias_dimensions != [1, dimensions[1], 1, 1] {
        candle_core::bail!("biased activation requires an exact NCHW channel bias")
    }
    Ok(dimensions)
}

fn four_dimensions(layout: &Layout) -> Result<[usize; 4]> {
    let [batch, channels, height, width] = layout.shape().dims() else {
        candle_core::bail!("biased activation requires rank-four NCHW inputs")
    };
    Ok([*batch, *channels, *height, *width])
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
    fn biased_identity_relu_and_gelu_are_byte_exact_with_existing_cuda_sequences() {
        let cuda = Device::new_cuda(0).unwrap();
        let input = Tensor::new(&[-17.0_f32, -3.0, -1.0, -0.0, 0.0, 0.25, 1.0, 3.0], &cuda)
            .unwrap()
            .reshape((1, 2, 2, 2))
            .unwrap();
        let bias = Tensor::new(&[-0.25_f32, 0.5], &cuda)
            .unwrap()
            .reshape((1, 2, 1, 1))
            .unwrap();
        let biased = input.broadcast_add(&bias).unwrap();

        let actual_bias = super::bias(&input, &bias).unwrap();
        assert_eq!(bits(&actual_bias), bits(&biased));

        let expected_relu = biased.relu().unwrap();
        let actual_relu = relu(&input, &bias).unwrap();
        assert_eq!(bits(&actual_relu), bits(&expected_relu));

        let divisor = Tensor::new(&[std::f32::consts::SQRT_2], &cuda).unwrap();
        let offset = Tensor::new(&[1.0_f32], &cuda).unwrap();
        let scale = Tensor::new(&[0.5_f32], &cuda).unwrap();
        let expected_gelu = biased
            .broadcast_div(&divisor)
            .and_then(|value| value.erf())
            .and_then(|value| value.broadcast_add(&offset))
            .and_then(|value| biased.broadcast_mul(&value))
            .and_then(|value| value.broadcast_mul(&scale))
            .unwrap();
        let actual_gelu = gelu_erf(&input, &bias, std::f32::consts::SQRT_2, 1.0, 0.5).unwrap();
        assert_eq!(bits(&actual_gelu), bits(&expected_gelu));
    }

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn biased_channel_gate_is_byte_exact_with_existing_cuda_sequence() {
        let cuda = Device::new_cuda(0).unwrap();
        let multiplicand = Tensor::new(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &cuda)
            .unwrap()
            .reshape((1, 2, 2, 2))
            .unwrap();
        let gate = Tensor::new(&[-3.0_f32, 2.0], &cuda)
            .unwrap()
            .reshape((1, 2, 1, 1))
            .unwrap();
        let bias = Tensor::new(&[0.5_f32, -0.25], &cuda)
            .unwrap()
            .reshape((1, 2, 1, 1))
            .unwrap();
        let alpha = 0.2_f32;
        let beta = 0.5_f32;
        let expected = gate
            .broadcast_add(&bias)
            .and_then(|value| &value * f64::from(alpha))
            .and_then(|value| value.affine(1.0, f64::from(beta)))
            .and_then(|value| value.clamp(0.0, 1.0))
            .and_then(|value| multiplicand.broadcast_mul(&value))
            .unwrap();
        let actual = gated_hard_sigmoid_mul(&multiplicand, &gate, &bias, alpha, beta).unwrap();

        assert_eq!(bits(&actual), bits(&expected));
    }

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn biased_residual_is_byte_exact_with_existing_cuda_sequence() {
        let cuda = Device::new_cuda(0).unwrap();
        let input = Tensor::new(&[-17.0_f32, -3.0, -1.0, -0.0, 0.0, 0.25, 1.0, 3.0], &cuda)
            .unwrap()
            .reshape((1, 2, 2, 2))
            .unwrap();
        let residual = Tensor::new(&[0.5_f32, -0.25, 7.0, -11.0, 0.0, -0.0, 2.5, -4.0], &cuda)
            .unwrap()
            .reshape((1, 2, 2, 2))
            .unwrap();
        let bias = Tensor::new(&[-0.25_f32, 0.5], &cuda)
            .unwrap()
            .reshape((1, 2, 1, 1))
            .unwrap();
        let expected = input
            .broadcast_add(&bias)
            .and_then(|value| value.broadcast_add(&residual))
            .unwrap();
        let actual = bias_residual(&input, &residual, &bias).unwrap();

        assert_eq!(bits(&actual), bits(&expected));
    }
}
