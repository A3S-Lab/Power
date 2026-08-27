use std::sync::Arc;

#[cfg(feature = "embedded-cuda")]
use candle_core::{DType, Tensor};

mod vectorized;

#[derive(Clone, Debug)]
pub(super) enum ConvolutionPostOperation {
    Identity,
    #[cfg(test)]
    Relu,
    #[cfg(test)]
    GeluErf {
        divisor: f32,
        offset: f32,
        scale: f32,
    },
    BatchNormalization {
        parameters: Arc<BatchNormParameters>,
        activation: BatchNormActivation,
    },
    #[cfg(feature = "embedded-cuda")]
    CudaBatchNormalization(Arc<CudaBatchNormPostOperation>),
}

impl ConvolutionPostOperation {
    #[cfg(test)]
    pub(super) fn batch_normalization(
        scale: &[f32],
        bias: &[f32],
        mean: &[f32],
        variance: &[f32],
        epsilon: f32,
        hard_swish: Option<(f32, f32)>,
    ) -> Option<Self> {
        if !epsilon.is_finite() || epsilon < 0.0 {
            return None;
        }
        let activation = hard_swish.map_or(BatchNormActivation::Identity, |(alpha, beta)| {
            BatchNormActivation::HardSwish { alpha, beta }
        });
        let stddev = variance
            .iter()
            .map(|variance| (variance + epsilon).sqrt())
            .collect::<Vec<_>>();
        Self::batch_normalization_with_prepared_statistics(scale, bias, mean, &stddev, activation)
    }

    pub(super) fn batch_normalization_with_prepared_statistics(
        scale: &[f32],
        bias: &[f32],
        mean: &[f32],
        stddev: &[f32],
        activation: BatchNormActivation,
    ) -> Option<Self> {
        let channels = scale.len();
        if channels == 0
            || bias.len() != channels
            || mean.len() != channels
            || stddev.len() != channels
        {
            return None;
        }
        let mut values = Vec::with_capacity(4 * channels);
        values.extend_from_slice(scale);
        values.extend_from_slice(bias);
        values.extend_from_slice(mean);
        values.extend_from_slice(stddev);
        Some(Self::BatchNormalization {
            parameters: Arc::new(BatchNormParameters {
                channels,
                values: values.into_boxed_slice(),
            }),
            activation,
        })
    }

    #[cfg(feature = "embedded-cuda")]
    pub(super) fn cuda_batch_normalization(
        scale_and_bias: &Tensor,
        mean_and_stddev: &Tensor,
        activation: CudaBatchNormActivation,
    ) -> Option<Self> {
        let (parameter_sets, channels) = scale_and_bias.dims2().ok()?;
        if parameter_sets != 2
            || channels == 0
            || mean_and_stddev.dims2().ok()? != (2, channels)
            || scale_and_bias.dtype() != DType::F32
            || mean_and_stddev.dtype() != DType::F32
            || !scale_and_bias.device().is_cuda()
            || !mean_and_stddev
                .device()
                .same_device(scale_and_bias.device())
            || !scale_and_bias.is_contiguous()
            || !mean_and_stddev.is_contiguous()
        {
            return None;
        }
        Some(Self::CudaBatchNormalization(Arc::new(
            CudaBatchNormPostOperation {
                scale_and_bias: scale_and_bias.clone(),
                mean_and_stddev: mean_and_stddev.clone(),
                channels,
                activation,
            },
        )))
    }

    #[cfg(feature = "embedded-cuda")]
    pub(super) fn cuda_batch_normalization_parameters(
        &self,
    ) -> Option<&CudaBatchNormPostOperation> {
        match self {
            Self::CudaBatchNormalization(parameters) => Some(parameters),
            _ => None,
        }
    }

    pub(super) fn supports_cuda_spatial(&self) -> bool {
        #[cfg(all(test, feature = "embedded-cuda"))]
        {
            matches!(
                self,
                Self::Relu | Self::GeluErf { .. } | Self::CudaBatchNormalization(_)
            )
        }
        #[cfg(all(not(test), feature = "embedded-cuda"))]
        {
            matches!(self, Self::CudaBatchNormalization(_))
        }
        #[cfg(not(feature = "embedded-cuda"))]
        {
            false
        }
    }

    pub(super) fn supports_channels(&self, channels: usize) -> bool {
        match self {
            Self::BatchNormalization { parameters, .. } => parameters.channels == channels,
            #[cfg(feature = "embedded-cuda")]
            Self::CudaBatchNormalization(parameters) => parameters.channels == channels,
            _ => true,
        }
    }

    pub(super) fn is_identity(&self) -> bool {
        matches!(self, Self::Identity)
    }

    pub(super) fn apply_channel(
        &self,
        channel: usize,
        values: &mut [f32],
        convolution_bias: Option<f32>,
    ) -> candle_core::Result<()> {
        let operation = match self {
            Self::Identity => ChannelPostOperation::Identity,
            #[cfg(test)]
            Self::Relu => ChannelPostOperation::Relu,
            #[cfg(test)]
            Self::GeluErf {
                divisor,
                offset,
                scale,
            } => ChannelPostOperation::GeluErf {
                divisor: *divisor,
                offset: *offset,
                scale: *scale,
            },
            Self::BatchNormalization {
                parameters,
                activation,
            } => parameters.channel(channel, *activation).ok_or_else(|| {
                candle_core::Error::Msg(
                    "convolution BatchNormalization channel is out of bounds".into(),
                )
            })?,
            #[cfg(feature = "embedded-cuda")]
            Self::CudaBatchNormalization(_) => {
                candle_core::bail!(
                    "CUDA BatchNormalization post-operation cannot execute on the CPU"
                )
            }
        };
        vectorized::apply(operation, values, convolution_bias);
        Ok(())
    }

    pub(super) fn add_identity_residual(
        &self,
        values: &mut [f32],
        residual: &[f32],
        convolution_bias: Option<f32>,
    ) -> candle_core::Result<()> {
        if !self.is_identity() {
            candle_core::bail!("convolution residual fusion requires an identity post-operation")
        }
        if values.len() != residual.len() {
            candle_core::bail!("convolution residual length does not match its output")
        }
        vectorized::add_bias_and_residual(values, residual, convolution_bias);
        Ok(())
    }
}

#[cfg(feature = "embedded-cuda")]
#[derive(Clone, Debug)]
pub(super) struct CudaBatchNormPostOperation {
    pub(super) scale_and_bias: Tensor,
    pub(super) mean_and_stddev: Tensor,
    channels: usize,
    pub(super) activation: CudaBatchNormActivation,
}

#[cfg(feature = "embedded-cuda")]
#[derive(Clone, Copy, Debug)]
pub(super) enum CudaBatchNormActivation {
    Identity,
    Relu,
    HardSwish {
        alpha: f32,
        beta: f32,
    },
    Swish,
    GeluErf {
        divisor: f32,
        offset: f32,
        scale: f32,
    },
}

#[cfg(feature = "embedded-cuda")]
impl CudaBatchNormActivation {
    pub(super) fn launch_parameters(self) -> CudaBatchNormLaunchParameters {
        match self {
            Self::Identity => CudaBatchNormLaunchParameters::new(0, 0.0, 0.0, 0.0),
            Self::HardSwish { alpha, beta } => {
                CudaBatchNormLaunchParameters::new(1, alpha, beta, 0.0)
            }
            Self::Relu => CudaBatchNormLaunchParameters::new(2, 0.0, 0.0, 0.0),
            Self::Swish => CudaBatchNormLaunchParameters::new(3, 0.0, 0.0, 0.0),
            Self::GeluErf {
                divisor,
                offset,
                scale,
            } => CudaBatchNormLaunchParameters::new(4, divisor, offset, scale),
        }
    }
}

#[cfg(feature = "embedded-cuda")]
#[derive(Clone, Copy, Debug)]
pub(super) struct CudaBatchNormLaunchParameters {
    pub(super) kind: u32,
    pub(super) first: f32,
    pub(super) second: f32,
    pub(super) third: f32,
}

#[cfg(feature = "embedded-cuda")]
impl CudaBatchNormLaunchParameters {
    fn new(kind: u32, first: f32, second: f32, third: f32) -> Self {
        Self {
            kind,
            first,
            second,
            third,
        }
    }

    pub(super) fn sigmoid() -> Self {
        Self::new(5, 0.0, 0.0, 0.0)
    }
}

#[derive(Debug)]
pub(super) struct BatchNormParameters {
    channels: usize,
    values: Box<[f32]>,
}

impl BatchNormParameters {
    fn channel(
        &self,
        channel: usize,
        activation: BatchNormActivation,
    ) -> Option<ChannelPostOperation> {
        Some(ChannelPostOperation::BatchNormalization {
            scale: *self.values.get(channel)?,
            bias: *self.values.get(self.channels + channel)?,
            mean: *self.values.get(2 * self.channels + channel)?,
            stddev: *self.values.get(3 * self.channels + channel)?,
            activation,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum BatchNormActivation {
    Identity,
    Relu,
    HardSwish { alpha: f32, beta: f32 },
}

#[derive(Clone, Copy, Debug)]
pub(super) enum ChannelPostOperation {
    Identity,
    #[cfg(test)]
    Relu,
    #[cfg(test)]
    GeluErf {
        divisor: f32,
        offset: f32,
        scale: f32,
    },
    BatchNormalization {
        scale: f32,
        bias: f32,
        mean: f32,
        stddev: f32,
        activation: BatchNormActivation,
    },
}

impl ChannelPostOperation {
    pub(super) fn apply(self, value: f32) -> f32 {
        match self {
            Self::Identity => value,
            #[cfg(test)]
            Self::Relu => value.max(0.0),
            #[cfg(test)]
            Self::GeluErf {
                divisor,
                offset,
                scale,
            } => {
                let divided = value / divisor;
                let activated = candle_core::cpu::erf::erf_f32(divided);
                let shifted = activated + offset;
                let product = value * shifted;
                product * scale
            }
            Self::BatchNormalization {
                scale,
                bias,
                mean,
                stddev,
                activation,
            } => {
                let normalized = (((value - mean) / stddev) * scale) + bias;
                match activation {
                    BatchNormActivation::Identity => normalized,
                    BatchNormActivation::Relu => normalized.max(0.0),
                    BatchNormActivation::HardSwish { alpha, beta } => {
                        let gate = ((normalized * alpha) + beta).clamp(0.0, 1.0);
                        normalized * gate
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectorized_channel_operations_match_scalar_bits_for_all_value_classes() {
        let source = [
            f32::NEG_INFINITY,
            -f32::MAX,
            -17.25,
            -0.0,
            0.0,
            0.125,
            f32::MAX,
            f32::INFINITY,
            f32::from_bits(0x7fc0_0001),
            f32::from_bits(0xffc0_1234),
            -3.5,
            11.75,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.0,
            -1.0,
            7.0,
        ];
        let operations = [
            ChannelPostOperation::Identity,
            ChannelPostOperation::Relu,
            ChannelPostOperation::GeluErf {
                divisor: std::f32::consts::SQRT_2,
                offset: 1.0,
                scale: 0.5,
            },
            ChannelPostOperation::BatchNormalization {
                scale: 0.75,
                bias: -0.125,
                mean: 0.5,
                stddev: 1.25,
                activation: BatchNormActivation::Identity,
            },
            ChannelPostOperation::BatchNormalization {
                scale: 0.75,
                bias: -0.125,
                mean: 0.5,
                stddev: 1.25,
                activation: BatchNormActivation::Relu,
            },
            ChannelPostOperation::BatchNormalization {
                scale: 0.75,
                bias: -0.125,
                mean: 0.5,
                stddev: 1.25,
                activation: BatchNormActivation::HardSwish {
                    alpha: 1.0 / 6.0,
                    beta: 0.5,
                },
            },
        ];
        for operation in operations {
            for convolution_bias in [None, Some(0.375)] {
                let expected = source
                    .iter()
                    .map(|value| {
                        let value = convolution_bias.map_or(*value, |bias| *value + bias);
                        operation.apply(value).to_bits()
                    })
                    .collect::<Vec<_>>();
                let mut actual = source;
                vectorized::apply(operation, &mut actual, convolution_bias);
                assert_eq!(
                    actual.map(f32::to_bits).as_slice(),
                    expected,
                    "operation={operation:?} convolution_bias={convolution_bias:?}",
                );
            }
        }
    }

    #[test]
    fn vectorized_gelu_erf_matches_scalar_bits_across_dense_numeric_samples() {
        const SAMPLES: usize = 65_536;
        const LOWER_BITS: u32 = 0x2f00_0000;
        const UPPER_BITS: u32 = 0x4110_0000;

        let mut source = Vec::with_capacity(SAMPLES * 2);
        for sample in 0..SAMPLES {
            let bits = LOWER_BITS
                + (((UPPER_BITS - LOWER_BITS) as u64 * sample as u64) / (SAMPLES - 1) as u64)
                    as u32;
            source.push(f32::from_bits(bits));
            source.push(f32::from_bits(bits | 0x8000_0000));
        }

        let operation = ChannelPostOperation::GeluErf {
            divisor: std::f32::consts::SQRT_2,
            offset: 1.0,
            scale: 0.5,
        };
        let expected = source
            .iter()
            .map(|value| operation.apply(*value).to_bits())
            .collect::<Vec<_>>();
        vectorized::apply(operation, &mut source, None);

        let actual = source.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn vectorized_bias_and_residual_matches_scalar_bits() {
        let mut actual = [
            f32::NEG_INFINITY,
            -f32::MAX,
            -3.5,
            -0.0,
            0.0,
            0.125,
            f32::MAX,
            f32::INFINITY,
            f32::from_bits(0x7fc0_0001),
            f32::from_bits(0xffc0_1234),
            11.75,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.0,
            -1.0,
            7.0,
            17.0,
        ];
        let residual = [
            1.0,
            -1.0,
            0.25,
            -0.0,
            0.0,
            -0.375,
            f32::NEG_INFINITY,
            f32::INFINITY,
            3.0,
            -5.0,
            f32::MAX,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            -2.0,
            2.0,
            f32::from_bits(0x7fc0_4321),
            19.0,
        ];
        let bias = 0.375;
        let expected = actual
            .iter()
            .zip(residual)
            .map(|(value, residual)| ((*value + bias) + residual).to_bits())
            .collect::<Vec<_>>();

        vectorized::add_bias_and_residual(&mut actual, &residual, Some(bias));

        assert_eq!(actual.map(f32::to_bits).as_slice(), expected);
    }
}
