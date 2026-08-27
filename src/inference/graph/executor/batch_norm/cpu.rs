use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

use super::Activation;

pub(super) fn prepare_statistics(mean_and_variance: &Tensor, epsilon: f32) -> Result<Tensor> {
    let (sets, channels) = mean_and_variance.dims2()?;
    if sets != 2
        || channels == 0
        || !mean_and_variance.is_contiguous()
        || !mean_and_variance.device().is_cpu()
        || !epsilon.is_finite()
        || epsilon < 0.0
    {
        candle_core::bail!(
            "CPU BatchNormalization preparation requires finite contiguous [2, channels] statistics"
        )
    }
    let values = mean_and_variance.flatten_all()?.to_vec1::<f32>()?;
    let (means, variances) = values.split_at(channels);
    let elements = channels.checked_mul(2).ok_or_else(|| {
        candle_core::Error::Msg("CPU BatchNormalization statistic size overflowed".into())
    })?;
    let mut prepared = Vec::with_capacity(elements);
    prepared.extend_from_slice(means);
    prepared.extend(variances.iter().map(|variance| (variance + epsilon).sqrt()));
    Tensor::from_vec(prepared, (2, channels), mean_and_variance.device())
}

pub(super) fn execute(
    input: &Tensor,
    scale_and_bias: &Tensor,
    mean_and_stddev: &Tensor,
    activation: Activation,
) -> Result<Tensor> {
    let operation = BatchNorm::new(
        input.layout(),
        scale_and_bias.layout(),
        mean_and_stddev.layout(),
        activation,
    )?;
    input.apply_op3_no_bwd(scale_and_bias, mean_and_stddev, &operation)
}

#[derive(Clone, Copy)]
struct BatchNorm {
    channels: usize,
    spatial: usize,
    elements: usize,
    activation: Activation,
}

impl BatchNorm {
    fn new(
        input: &Layout,
        scale_and_bias: &Layout,
        mean_and_stddev: &Layout,
        activation: Activation,
    ) -> Result<Self> {
        if !input.is_contiguous()
            || !scale_and_bias.is_contiguous()
            || !mean_and_stddev.is_contiguous()
        {
            candle_core::bail!("fused CPU BatchNormalization requires contiguous inputs")
        }
        let dimensions = input.shape().dims();
        if dimensions.len() < 2 {
            candle_core::bail!("fused CPU BatchNormalization requires [N, C, D...] input")
        }
        let batch = dimensions[0];
        let channels = dimensions[1];
        if scale_and_bias.shape().dims2()? != (2, channels)
            || mean_and_stddev.shape().dims2()? != (2, channels)
        {
            candle_core::bail!(
                "fused CPU BatchNormalization parameters must have exact [2, channels] shape"
            )
        }
        let spatial = dimensions[2..]
            .iter()
            .try_fold(1_usize, |total, dimension| {
                total.checked_mul(*dimension).ok_or_else(|| {
                    candle_core::Error::Msg("batch norm spatial size overflowed".into())
                })
            })?;
        let elements = input.shape().elem_count();
        if batch == 0 || channels == 0 || spatial == 0 || elements == 0 {
            candle_core::bail!("fused CPU BatchNormalization requires non-empty [N, C, D...] input")
        }
        Ok(Self {
            channels,
            spatial,
            elements,
            activation,
        })
    }

    fn transform(&self, input: f32, scale: f32, bias: f32, mean: f32, stddev: f32) -> f32 {
        let normalized = (((input - mean) / stddev) * scale) + bias;
        match self.activation {
            Activation::Identity => normalized,
            Activation::Relu => normalized.max(0.0),
            Activation::HardSwish { alpha, beta } => {
                let gate = ((normalized * alpha) + beta).clamp(0.0, 1.0);
                normalized * gate
            }
            Activation::Sigmoid => 1.0 / (1.0 + (-normalized).exp()),
            Activation::Swish => normalized * (1.0 / (1.0 + (-normalized).exp())),
            Activation::GeluErf {
                divisor,
                offset,
                scale,
            } => {
                let divided = normalized / divisor;
                let activated = candle_core::cpu::erf::erf_f32(divided);
                let shifted = activated + offset;
                let product = normalized * shifted;
                product * scale
            }
        }
    }
}

impl CustomOp3 for BatchNorm {
    fn name(&self) -> &'static str {
        "a3s-fused-cpu-batch-normalization"
    }

    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        input_layout: &Layout,
        scale_and_bias: &CpuStorage,
        scale_and_bias_layout: &Layout,
        mean_and_stddev: &CpuStorage,
        mean_and_stddev_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let input = input.as_slice::<f32>()?;
        let scale_and_bias = scale_and_bias.as_slice::<f32>()?;
        let mean_and_stddev = mean_and_stddev.as_slice::<f32>()?;
        let input_start = input_layout.start_offset();
        let parameter_start = scale_and_bias_layout.start_offset();
        let statistics_start = mean_and_stddev_layout.start_offset();
        let input = input
            .get(input_start..input_start + self.elements)
            .ok_or_else(|| {
                candle_core::Error::Msg("batch norm input layout is out of bounds".into())
            })?;
        let parameters = scale_and_bias
            .get(parameter_start..parameter_start + 2 * self.channels)
            .ok_or_else(|| {
                candle_core::Error::Msg("batch norm parameter layout is out of bounds".into())
            })?;
        let statistics = mean_and_stddev
            .get(statistics_start..statistics_start + 2 * self.channels)
            .ok_or_else(|| {
                candle_core::Error::Msg("batch norm statistic layout is out of bounds".into())
            })?;
        let (scales, biases) = parameters.split_at(self.channels);
        let (means, stddevs) = statistics.split_at(self.channels);
        let mut output = vec![0.0_f32; self.elements];
        output
            .par_chunks_mut(self.spatial)
            .enumerate()
            .for_each(|(batch_channel, output)| {
                let batch = batch_channel / self.channels;
                let channel = batch_channel % self.channels;
                let start = (batch * self.channels + channel) * self.spatial;
                let end = start + self.spatial;
                let scale = scales[channel];
                let bias = biases[channel];
                let mean = means[channel];
                let stddev = stddevs[channel];
                for (output, input) in output.iter_mut().zip(&input[start..end]) {
                    *output = self.transform(*input, scale, bias, mean, stddev);
                }
            });
        Ok((CpuStorage::F32(output), input_layout.shape().clone()))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    fn values(tensor: &Tensor) -> Vec<f32> {
        tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn fused_cpu_batch_norm_preserves_the_explicit_graph_formula() {
        let device = Device::Cpu;
        let input = Tensor::from_iter(
            (0..3 * 3 * 4 * 5).map(|value| (value as f32 - 41.0) / 17.0),
            &device,
        )
        .unwrap()
        .reshape((3, 3, 4, 5))
        .unwrap()
        .narrow(0, 1, 2)
        .unwrap();
        assert!(input.is_contiguous());
        assert_ne!(input.layout().start_offset(), 0);
        let scale = Tensor::new(&[0.75_f32, -1.25, 2.0], &device).unwrap();
        let bias = Tensor::new(&[-0.5_f32, 0.125, 1.5], &device).unwrap();
        let mean = Tensor::new(&[0.25_f32, -0.75, 1.25], &device).unwrap();
        let variance = Tensor::new(&[0.5_f32, 1.5, 2.5], &device).unwrap();
        let epsilon = 0.000_01_f32;
        let channel_shape = (1, 3, 1, 1);
        let expected = input
            .broadcast_sub(&mean.reshape(channel_shape).unwrap())
            .unwrap()
            .broadcast_div(
                &variance
                    .reshape(channel_shape)
                    .unwrap()
                    .affine(1.0, epsilon as f64)
                    .unwrap()
                    .sqrt()
                    .unwrap(),
            )
            .unwrap()
            .broadcast_mul(&scale.reshape(channel_shape).unwrap())
            .unwrap()
            .broadcast_add(&bias.reshape(channel_shape).unwrap())
            .unwrap();
        let scale_and_bias = Tensor::stack(&[&scale, &bias], 0).unwrap();
        let mean_and_variance = Tensor::stack(&[&mean, &variance], 0).unwrap();
        let mean_and_stddev = prepare_statistics(&mean_and_variance, epsilon).unwrap();

        let actual = execute(
            &input,
            &scale_and_bias,
            &mean_and_stddev,
            Activation::Identity,
        )
        .unwrap();
        let expected_gate = (&expected * 0.2)
            .unwrap()
            .affine(1.0, 0.5)
            .unwrap()
            .clamp(0.0, 1.0)
            .unwrap();
        let expected_hard_swish = expected.broadcast_mul(&expected_gate).unwrap();
        let actual_hard_swish = execute(
            &input,
            &scale_and_bias,
            &mean_and_stddev,
            Activation::HardSwish {
                alpha: 0.2,
                beta: 0.5,
            },
        )
        .unwrap();
        let expected_relu = expected.relu().unwrap();
        let actual_relu =
            execute(&input, &scale_and_bias, &mean_and_stddev, Activation::Relu).unwrap();
        let expected_swish = expected
            .broadcast_mul(&candle_nn::ops::sigmoid(&expected).unwrap())
            .unwrap();
        let expected_sigmoid = candle_nn::ops::sigmoid(&expected).unwrap();
        let actual_sigmoid = execute(
            &input,
            &scale_and_bias,
            &mean_and_stddev,
            Activation::Sigmoid,
        )
        .unwrap();
        let actual_swish =
            execute(&input, &scale_and_bias, &mean_and_stddev, Activation::Swish).unwrap();
        let divisor = std::f32::consts::SQRT_2;
        let offset = 1.0_f32;
        let gelu_scale = 0.5_f32;
        let divisor_tensor = Tensor::new(&[divisor], &device).unwrap();
        let offset_tensor = Tensor::new(&[offset], &device).unwrap();
        let gelu_scale_tensor = Tensor::new(&[gelu_scale], &device).unwrap();
        let expected_gelu = expected
            .broadcast_div(&divisor_tensor)
            .and_then(|value| value.erf())
            .and_then(|value| value.broadcast_add(&offset_tensor))
            .and_then(|value| expected.broadcast_mul(&value))
            .and_then(|value| value.broadcast_mul(&gelu_scale_tensor))
            .unwrap();
        let actual_gelu = execute(
            &input,
            &scale_and_bias,
            &mean_and_stddev,
            Activation::GeluErf {
                divisor,
                offset,
                scale: gelu_scale,
            },
        )
        .unwrap();

        assert_eq!(values(&actual), values(&expected));
        assert_eq!(values(&actual_relu), values(&expected_relu));
        assert_eq!(values(&actual_hard_swish), values(&expected_hard_swish));
        assert_eq!(values(&actual_sigmoid), values(&expected_sigmoid));
        assert_eq!(values(&actual_swish), values(&expected_swish));
        assert_eq!(values(&actual_gelu), values(&expected_gelu));
    }

    #[test]
    fn fused_cpu_batch_norm_supports_rank_three_sequence_features() {
        let device = Device::Cpu;
        let input = Tensor::new(&[[[1.0_f32, 3.0], [2.0, 5.0]]], &device).unwrap();
        let scale = Tensor::new(&[2.0_f32, 3.0], &device).unwrap();
        let bias = Tensor::zeros(2, candle_core::DType::F32, &device).unwrap();
        let mean = Tensor::new(&[1.0_f32, 2.0], &device).unwrap();
        let variance = Tensor::new(&[4.0_f32, 9.0], &device).unwrap();
        let scale_and_bias = Tensor::stack(&[&scale, &bias], 0).unwrap();
        let mean_and_variance = Tensor::stack(&[&mean, &variance], 0).unwrap();
        let mean_and_stddev = prepare_statistics(&mean_and_variance, 0.0).unwrap();

        let actual = execute(
            &input,
            &scale_and_bias,
            &mean_and_stddev,
            Activation::Identity,
        )
        .unwrap();

        assert_eq!(actual.dims(), [1, 2, 2]);
        assert_eq!(values(&actual), [0.0, 2.0, 0.0, 3.0]);
    }
}
