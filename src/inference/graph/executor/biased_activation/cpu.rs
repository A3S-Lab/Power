use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

pub(super) fn relu(input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let operation = ChannelBiasActivation::new(input.layout(), bias.layout(), Activation::Relu)?;
    input.apply_op2_no_bwd(bias, &operation)
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
    channels: usize,
    spatial: usize,
    elements: usize,
    activation: Activation,
}

#[derive(Clone, Copy)]
enum Activation {
    Relu,
    GeluErf {
        divisor: f32,
        offset: f32,
        scale: f32,
    },
}

impl ChannelBiasActivation {
    fn new(input: &Layout, bias: &Layout, activation: Activation) -> Result<Self> {
        if !input.is_contiguous() || !bias.is_contiguous() {
            candle_core::bail!("fused CPU channel-bias activation requires contiguous inputs")
        }
        let (batch, channels, height, width) = input.shape().dims4()?;
        if bias.shape().dims4()? != (1, channels, 1, 1) {
            candle_core::bail!("fused CPU channel bias must have exact [1, channels, 1, 1] shape")
        }
        let spatial = height.checked_mul(width).ok_or_else(|| {
            candle_core::Error::Msg("biased activation spatial size overflowed".into())
        })?;
        let elements = input.shape().elem_count();
        if batch == 0 || channels == 0 || spatial == 0 || elements == 0 {
            candle_core::bail!("fused CPU channel-bias activation requires non-empty NCHW input")
        }
        Ok(Self {
            shape: input.shape().clone(),
            channels,
            spatial,
            elements,
            activation,
        })
    }
}

impl CustomOp2 for ChannelBiasActivation {
    fn name(&self) -> &'static str {
        "a3s-fused-cpu-channel-bias-activation"
    }

    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        input_layout: &Layout,
        bias: &CpuStorage,
        bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let input_values = input.as_slice::<f32>()?;
        let bias_values = bias.as_slice::<f32>()?;
        let input_start = input_layout.start_offset();
        let bias_start = bias_layout.start_offset();
        let input_values = input_values
            .get(input_start..input_start + self.elements)
            .ok_or_else(|| {
                candle_core::Error::Msg("biased activation input layout is out of bounds".into())
            })?;
        let bias_values = bias_values
            .get(bias_start..bias_start + self.channels)
            .ok_or_else(|| {
                candle_core::Error::Msg("biased activation bias layout is out of bounds".into())
            })?;
        let mut output = vec![0.0_f32; self.elements];
        match self.activation {
            Activation::Relu => apply_channel_bias(
                input_values,
                bias_values,
                &mut output,
                self.channels,
                self.spatial,
                |input, bias| (input + bias).max(0.0),
            ),
            Activation::GeluErf {
                divisor,
                offset,
                scale,
            } => apply_channel_bias(
                input_values,
                bias_values,
                &mut output,
                self.channels,
                self.spatial,
                |input, bias| {
                    let biased = *input + bias;
                    let divided = biased / divisor;
                    let activated = candle_core::cpu::erf::erf_f32(divided);
                    let shifted = activated + offset;
                    let product = biased * shifted;
                    product * scale
                },
            ),
        }
        Ok((CpuStorage::F32(output), self.shape.clone()))
    }
}

fn apply_channel_bias(
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
    channels: usize,
    spatial: usize,
    transform: impl Fn(&f32, f32) -> f32 + Sync,
) {
    output
        .par_chunks_mut(spatial)
        .enumerate()
        .for_each(|(batch_channel, output)| {
            let channel = batch_channel % channels;
            let start = batch_channel * spatial;
            let end = start + spatial;
            let bias = bias[channel];
            for (output, input) in output.iter_mut().zip(&input[start..end]) {
                *output = transform(input, bias);
            }
        });
}

#[derive(Clone)]
struct BiasedGatedHardSigmoid {
    shape: Shape,
    elements: usize,
    channels: usize,
    spatial: usize,
    gate_spatial: usize,
    alpha: f32,
    beta: f32,
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
            candle_core::bail!("fused CPU biased HardSigmoid requires contiguous inputs")
        }
        let (batch, channels, height, width) = multiplicand.shape().dims4()?;
        let gate_dimensions = gate.shape().dims4()?;
        if bias.shape().dims4()? != (1, channels, 1, 1)
            || (gate_dimensions != (batch, channels, height, width)
                && gate_dimensions != (batch, channels, 1, 1))
        {
            candle_core::bail!(
                "fused CPU biased HardSigmoid requires equal shapes or an exact NCHW channel gate"
            )
        }
        let spatial = height.checked_mul(width).ok_or_else(|| {
            candle_core::Error::Msg("gated activation spatial size overflowed".into())
        })?;
        let gate_spatial = gate_dimensions
            .2
            .checked_mul(gate_dimensions.3)
            .ok_or_else(|| {
                candle_core::Error::Msg("gated activation gate size overflowed".into())
            })?;
        let elements = multiplicand.shape().elem_count();
        if batch == 0 || channels == 0 || spatial == 0 || gate_spatial == 0 || elements == 0 {
            candle_core::bail!("fused CPU biased HardSigmoid requires non-empty NCHW inputs")
        }
        Ok(Self {
            shape: multiplicand.shape().clone(),
            elements,
            channels,
            spatial,
            gate_spatial,
            alpha,
            beta,
        })
    }
}

impl CustomOp3 for BiasedGatedHardSigmoid {
    fn name(&self) -> &'static str {
        "a3s-fused-cpu-channel-bias-gated-hard-sigmoid"
    }

    fn cpu_fwd(
        &self,
        multiplicand: &CpuStorage,
        multiplicand_layout: &Layout,
        gate: &CpuStorage,
        gate_layout: &Layout,
        bias: &CpuStorage,
        bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let multiplicand = contiguous_values(
            multiplicand.as_slice::<f32>()?,
            multiplicand_layout,
            self.elements,
            "gated activation multiplicand",
        )?;
        let gate_elements = self.elements / self.spatial * self.gate_spatial;
        let gate = contiguous_values(
            gate.as_slice::<f32>()?,
            gate_layout,
            gate_elements,
            "gated activation gate",
        )?;
        let bias = contiguous_values(
            bias.as_slice::<f32>()?,
            bias_layout,
            self.channels,
            "gated activation bias",
        )?;
        let mut output = vec![0.0_f32; self.elements];
        output
            .par_chunks_mut(self.spatial)
            .enumerate()
            .for_each(|(batch_channel, output)| {
                let channel = batch_channel % self.channels;
                let multiplicand_start = batch_channel * self.spatial;
                let gate_start = batch_channel * self.gate_spatial;
                for (pixel, output) in output.iter_mut().enumerate() {
                    let gate_index = gate_start + pixel.min(self.gate_spatial - 1);
                    let biased = gate[gate_index] + bias[channel];
                    let scaled = biased * self.alpha + 0.0;
                    let shifted = scaled * 1.0 + self.beta;
                    let lower_bounded = if shifted < 0.0 { 0.0 } else { shifted };
                    let bounded = if lower_bounded > 1.0 {
                        1.0
                    } else {
                        lower_bounded
                    };
                    *output = multiplicand[multiplicand_start + pixel] * bounded;
                }
            });
        Ok((CpuStorage::F32(output), self.shape.clone()))
    }
}

fn contiguous_values<'a>(
    storage: &'a [f32],
    layout: &Layout,
    elements: usize,
    label: &str,
) -> Result<&'a [f32]> {
    let start = layout.start_offset();
    storage
        .get(start..start + elements)
        .ok_or_else(|| candle_core::Error::Msg(format!("{label} layout is out of bounds")))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    #[test]
    fn fused_cpu_channel_bias_relu_preserves_the_explicit_graph_formula() {
        let device = Device::Cpu;
        let input = Tensor::from_iter(
            (0..3 * 3 * 2 * 4).map(|value| (value as f32 - 31.0) / 11.0),
            &device,
        )
        .unwrap()
        .reshape((3, 3, 2, 4))
        .unwrap()
        .narrow(0, 1, 2)
        .unwrap();
        let bias = Tensor::new(&[-9.0_f32, 0.25, -0.5, 1.25, 7.0, 8.0], &device)
            .unwrap()
            .reshape((2, 3, 1, 1))
            .unwrap()
            .narrow(0, 1, 1)
            .unwrap();
        let expected = input.broadcast_add(&bias).unwrap().relu().unwrap();
        let actual = relu(&input, &bias).unwrap();

        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn fused_cpu_channel_bias_gelu_preserves_the_explicit_graph_formula() {
        let device = Device::Cpu;
        let input = Tensor::from_iter(
            (0..3 * 3 * 2 * 4).map(|value| (value as f32 - 31.0) / 11.0),
            &device,
        )
        .unwrap()
        .reshape((3, 3, 2, 4))
        .unwrap()
        .narrow(0, 1, 2)
        .unwrap();
        let bias = Tensor::new(&[-9.0_f32, 0.25, -0.5, 1.25, 7.0, 8.0], &device)
            .unwrap()
            .reshape((2, 3, 1, 1))
            .unwrap()
            .narrow(0, 1, 1)
            .unwrap();
        assert!(input.is_contiguous());
        assert!(bias.is_contiguous());
        assert_ne!(input.layout().start_offset(), 0);
        assert_ne!(bias.layout().start_offset(), 0);
        // Preserve the exact f32 initializer published by the graph rather
        // than substituting a mathematically equivalent higher-precision
        // constant that could round differently.
        let divisor = f32::from_bits(0x3fb5_04f3);
        let offset = 1.0_f32;
        let scale = 0.5_f32;
        let biased = input.broadcast_add(&bias).unwrap();
        let expected = biased
            .broadcast_div(&Tensor::new(divisor, &device).unwrap())
            .unwrap()
            .erf()
            .unwrap()
            .affine(1.0, offset as f64)
            .unwrap()
            .broadcast_mul(&biased)
            .unwrap()
            .affine(scale as f64, 0.0)
            .unwrap();
        let actual = gelu_erf(&input, &bias, divisor, offset, scale).unwrap();

        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn fused_cpu_biased_gate_preserves_same_shape_and_channel_broadcast_formulas() {
        let device = Device::Cpu;
        let multiplicand = Tensor::from_iter(
            (0..3 * 3 * 2 * 4).map(|value| (value as f32 - 29.0) / 13.0),
            &device,
        )
        .unwrap()
        .reshape((3, 3, 2, 4))
        .unwrap()
        .narrow(0, 1, 2)
        .unwrap();
        let same_shape_gate = Tensor::from_iter(
            (0..3 * 3 * 2 * 4).map(|value| (value as f32 - 41.0) / 17.0),
            &device,
        )
        .unwrap()
        .reshape((3, 3, 2, 4))
        .unwrap()
        .narrow(0, 1, 2)
        .unwrap();
        let channel_gate =
            Tensor::from_iter((0..3 * 3).map(|value| (value as f32 - 5.0) / 7.0), &device)
                .unwrap()
                .reshape((3, 3, 1, 1))
                .unwrap()
                .narrow(0, 1, 2)
                .unwrap();
        let bias = Tensor::new(&[-9.0_f32, 0.25, -0.5, 1.25, 7.0, 8.0], &device)
            .unwrap()
            .reshape((2, 3, 1, 1))
            .unwrap()
            .narrow(0, 1, 1)
            .unwrap();
        let alpha = 0.2_f32;
        let beta = 0.5_f32;

        for gate in [&same_shape_gate, &channel_gate] {
            let biased = gate.broadcast_add(&bias).unwrap();
            let bounded = biased
                .affine(alpha as f64, 0.0)
                .unwrap()
                .affine(1.0, beta as f64)
                .unwrap()
                .clamp(0.0, 1.0)
                .unwrap();
            let expected = multiplicand.broadcast_mul(&bounded).unwrap();
            let actual = gated_hard_sigmoid_mul(&multiplicand, gate, &bias, alpha, beta).unwrap();

            assert_eq!(
                actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
        }
    }
}
