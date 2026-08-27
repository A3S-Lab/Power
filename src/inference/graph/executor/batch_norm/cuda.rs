use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::{cudarc::driver::CudaSlice, CudaDevice};
use candle_core::{
    CpuStorage, CudaStorage, CustomOp1, CustomOp3, Layout, Result, Shape, Storage, Tensor,
};

use super::super::convolution_post::{
    CudaBatchNormActivation, CudaBatchNormLaunchParameters, CudaBatchNormPostOperation,
};
use super::super::cuda_fast_divisor::FastDivisorU32;
use super::Activation;

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/batch_norm_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_batch_norm_f32_v8";
const PREPARE_FUNCTION_NAME: &str = "prepare_batch_norm_mean_stddev_f32";
const FUNCTION_NAME: &str = "batch_norm_f32";
const IN_PLACE_FUNCTION_NAME: &str = "batch_norm_in_place_f32";
const THREADS_PER_BLOCK: u32 = 512;

pub(super) fn prepare_statistics(mean_and_variance: &Tensor, epsilon: f32) -> Result<Tensor> {
    let operation = PreparedStatistics::new(mean_and_variance.layout(), epsilon)?;
    mean_and_variance.apply_op1_no_bwd(&operation)
}

#[derive(Clone, Copy)]
struct PreparedStatistics {
    channels: usize,
    epsilon: f32,
}

impl PreparedStatistics {
    fn new(mean_and_variance: &Layout, epsilon: f32) -> Result<Self> {
        let (sets, channels) = mean_and_variance.shape().dims2()?;
        if !mean_and_variance.is_contiguous()
            || sets != 2
            || channels == 0
            || channels.checked_mul(2).is_none()
            || u32::try_from(channels).is_err()
            || !epsilon.is_finite()
            || epsilon < 0.0
        {
            candle_core::bail!(
                "CUDA BatchNormalization preparation requires finite contiguous [2, channels] statistics"
            )
        }
        Ok(Self { channels, epsilon })
    }
}

impl CustomOp1 for PreparedStatistics {
    fn name(&self) -> &'static str {
        "a3s-prepare-batch-normalization-statistics"
    }

    fn cpu_fwd(&self, _input: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA BatchNormalization preparation is CUDA-only")
    }

    fn cuda_fwd(&self, input: &CudaStorage, input_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let device = input.device();
        let function = device.get_or_load_custom_func(
            PREPARE_FUNCTION_NAME,
            MODULE_NAME,
            kernels::BATCH_NORM,
        )?;
        let elements = self.channels.checked_mul(2).ok_or_else(|| {
            candle_core::Error::Msg("batch norm statistic size overflowed".into())
        })?;
        let mut output = unsafe { device.alloc::<f32>(elements)? };
        let input_offset = as_u64(input_layout.start_offset())?;
        let channels = u32::try_from(self.channels)
            .map_err(|_| candle_core::Error::Msg("batch norm channels exceed u32".into()))?;
        let mut builder = function.builder();
        builder.arg(input.as_cuda_slice::<f32>()?);
        builder.arg(&mut output);
        builder.arg(&input_offset);
        builder.arg(&channels);
        builder.arg(&self.epsilon);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: (channels.div_ceil(THREADS_PER_BLOCK), 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
                .w()?
        };
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            input_layout.shape().clone(),
        ))
    }
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
    index_divisors: [u32; 5],
    elements: usize,
    activation: Activation,
}

#[derive(Clone, Copy)]
struct CudaBatchNormParameters<'a> {
    scale_and_bias: &'a CudaStorage,
    scale_and_bias_layout: &'a Layout,
    mean_and_stddev: &'a CudaStorage,
    mean_and_stddev_layout: &'a Layout,
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
            candle_core::bail!("fused BatchNormalization requires contiguous inputs")
        }
        let dimensions = input.shape().dims();
        if dimensions.len() < 2 {
            candle_core::bail!("fused BatchNormalization requires [N, C, D...] input")
        }
        let batch = dimensions[0];
        let channels = dimensions[1];
        if scale_and_bias.shape().dims2()? != (2, channels)
            || mean_and_stddev.shape().dims2()? != (2, channels)
        {
            candle_core::bail!(
                "fused BatchNormalization parameters must have exact [2, channels] shape"
            )
        }
        let spatial = dimensions[2..]
            .iter()
            .try_fold(1_usize, |total, dimension| {
                total.checked_mul(*dimension).ok_or_else(|| {
                    candle_core::Error::Msg("batch norm spatial size overflowed".into())
                })
            })?;
        Self::from_geometry(
            batch,
            channels,
            spatial,
            scale_and_bias,
            mean_and_stddev,
            activation,
        )
    }

    fn from_geometry(
        batch: usize,
        channels: usize,
        spatial: usize,
        scale_and_bias: &Layout,
        mean_and_stddev: &Layout,
        activation: Activation,
    ) -> Result<Self> {
        if !scale_and_bias.is_contiguous() || !mean_and_stddev.is_contiguous() {
            candle_core::bail!("fused BatchNormalization requires contiguous parameters")
        }
        if scale_and_bias.shape().dims2()? != (2, channels)
            || mean_and_stddev.shape().dims2()? != (2, channels)
        {
            candle_core::bail!(
                "fused BatchNormalization parameters must have exact [2, channels] shape"
            )
        }
        let elements = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(spatial))
            .ok_or_else(|| candle_core::Error::Msg("batch norm element count overflowed".into()))?;
        if batch == 0
            || channels == 0
            || spatial == 0
            || elements == 0
            || u32::try_from(elements).is_err()
        {
            candle_core::bail!("fused BatchNormalization exceeds the reviewed launch bound")
        }
        let [spatial_multiplier, spatial_shift, _] =
            FastDivisorU32::new(u32::try_from(spatial).map_err(|_| {
                candle_core::Error::Msg("batch norm spatial size exceeds u32".into())
            })?)
            .ok_or_else(|| candle_core::Error::Msg("batch norm spatial size is zero".into()))?
            .launch_parameters();
        let [channels_multiplier, channels_shift, channels] = FastDivisorU32::new(
            u32::try_from(channels)
                .map_err(|_| candle_core::Error::Msg("batch norm channels exceed u32".into()))?,
        )
        .ok_or_else(|| candle_core::Error::Msg("batch norm channel count is zero".into()))?
        .launch_parameters();
        Ok(Self {
            index_divisors: [
                spatial_multiplier,
                spatial_shift,
                channels_multiplier,
                channels_shift,
                channels,
            ],
            elements,
            activation,
        })
    }

    fn launch_into(
        &self,
        device: &CudaDevice,
        input: &CudaSlice<f32>,
        input_offset: u64,
        parameters: CudaBatchNormParameters<'_>,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let function =
            device.get_or_load_custom_func(FUNCTION_NAME, MODULE_NAME, kernels::BATCH_NORM)?;
        let scale_and_bias_offset = as_u64(parameters.scale_and_bias_layout.start_offset())?;
        let mean_and_stddev_offset = as_u64(parameters.mean_and_stddev_layout.start_offset())?;
        let mut builder = function.builder();
        builder.arg(input);
        builder.arg(parameters.scale_and_bias.as_cuda_slice::<f32>()?);
        builder.arg(parameters.mean_and_stddev.as_cuda_slice::<f32>()?);
        builder.arg(&mut *output);
        builder.arg(&input_offset);
        builder.arg(&scale_and_bias_offset);
        builder.arg(&mean_and_stddev_offset);
        let elements = u32::try_from(self.elements)
            .map_err(|_| candle_core::Error::Msg("batch norm launch size exceeds u32".into()))?;
        builder.arg(&elements);
        for value in &self.index_divisors {
            builder.arg(value);
        }
        let activation = activation_launch_parameters(self.activation);
        builder.arg(&activation.kind);
        builder.arg(&activation.first);
        builder.arg(&activation.second);
        builder.arg(&activation.third);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: (elements.div_ceil(THREADS_PER_BLOCK), 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
                .w()?
        };
        Ok(())
    }

    fn launch_in_place(
        &self,
        device: &CudaDevice,
        values: &mut CudaSlice<f32>,
        parameters: CudaBatchNormParameters<'_>,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let function = device.get_or_load_custom_func(
            IN_PLACE_FUNCTION_NAME,
            MODULE_NAME,
            kernels::BATCH_NORM,
        )?;
        let scale_and_bias_offset = as_u64(parameters.scale_and_bias_layout.start_offset())?;
        let mean_and_stddev_offset = as_u64(parameters.mean_and_stddev_layout.start_offset())?;
        let mut builder = function.builder();
        builder.arg(&mut *values);
        builder.arg(parameters.scale_and_bias.as_cuda_slice::<f32>()?);
        builder.arg(parameters.mean_and_stddev.as_cuda_slice::<f32>()?);
        builder.arg(&scale_and_bias_offset);
        builder.arg(&mean_and_stddev_offset);
        let elements = u32::try_from(self.elements)
            .map_err(|_| candle_core::Error::Msg("batch norm launch size exceeds u32".into()))?;
        builder.arg(&elements);
        for value in &self.index_divisors {
            builder.arg(value);
        }
        let activation = activation_launch_parameters(self.activation);
        builder.arg(&activation.kind);
        builder.arg(&activation.first);
        builder.arg(&activation.second);
        builder.arg(&activation.third);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: (elements.div_ceil(THREADS_PER_BLOCK), 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
                .w()?
        };
        Ok(())
    }
}

impl CustomOp3 for BatchNorm {
    fn name(&self) -> &'static str {
        "a3s-fused-batch-normalization"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _scale_and_bias: &CpuStorage,
        _scale_and_bias_layout: &Layout,
        _mean_and_variance: &CpuStorage,
        _mean_and_variance_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused BatchNormalization operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        scale_and_bias: &CudaStorage,
        scale_and_bias_layout: &Layout,
        mean_and_stddev: &CudaStorage,
        mean_and_stddev_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let device = input.device();
        let mut output = unsafe { device.alloc::<f32>(self.elements)? };
        let input_offset = as_u64(input_layout.start_offset())?;
        self.launch_into(
            device,
            input.as_cuda_slice::<f32>()?,
            input_offset,
            CudaBatchNormParameters {
                scale_and_bias,
                scale_and_bias_layout,
                mean_and_stddev,
                mean_and_stddev_layout,
            },
            &mut output,
        )?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            input_layout.shape().clone(),
        ))
    }
}

pub(super) fn execute_post_in_place(
    output: &mut CudaSlice<f32>,
    device: &CudaDevice,
    batch: usize,
    channels: usize,
    spatial: usize,
    post_operation: &CudaBatchNormPostOperation,
) -> Result<()> {
    let (scale_and_bias_storage, scale_and_bias_layout) =
        post_operation.scale_and_bias.storage_and_layout();
    let Storage::Cuda(scale_and_bias) = &*scale_and_bias_storage else {
        candle_core::bail!("pointwise BatchNormalization parameters are not CUDA-resident")
    };
    let (mean_and_stddev_storage, mean_and_stddev_layout) =
        post_operation.mean_and_stddev.storage_and_layout();
    let Storage::Cuda(mean_and_stddev) = &*mean_and_stddev_storage else {
        candle_core::bail!("pointwise BatchNormalization statistics are not CUDA-resident")
    };
    let operation = BatchNorm::from_geometry(
        batch,
        channels,
        spatial,
        scale_and_bias_layout,
        mean_and_stddev_layout,
        activation_from_cuda(post_operation.activation),
    )?;
    if output.len() != operation.elements {
        candle_core::bail!("pointwise BatchNormalization output length does not match its geometry")
    }
    operation.launch_in_place(
        device,
        output,
        CudaBatchNormParameters {
            scale_and_bias,
            scale_and_bias_layout,
            mean_and_stddev,
            mean_and_stddev_layout,
        },
    )
}

fn activation_from_cuda(activation: CudaBatchNormActivation) -> Activation {
    match activation {
        CudaBatchNormActivation::Identity => Activation::Identity,
        CudaBatchNormActivation::Relu => Activation::Relu,
        CudaBatchNormActivation::HardSwish { alpha, beta } => Activation::HardSwish { alpha, beta },
        CudaBatchNormActivation::Swish => Activation::Swish,
        CudaBatchNormActivation::GeluErf {
            divisor,
            offset,
            scale,
        } => Activation::GeluErf {
            divisor,
            offset,
            scale,
        },
    }
}

fn activation_launch_parameters(activation: Activation) -> CudaBatchNormLaunchParameters {
    match activation {
        Activation::Identity => CudaBatchNormActivation::Identity.launch_parameters(),
        Activation::Relu => CudaBatchNormActivation::Relu.launch_parameters(),
        Activation::HardSwish { alpha, beta } => {
            CudaBatchNormActivation::HardSwish { alpha, beta }.launch_parameters()
        }
        Activation::Sigmoid => CudaBatchNormLaunchParameters::sigmoid(),
        Activation::Swish => CudaBatchNormActivation::Swish.launch_parameters(),
        Activation::GeluErf {
            divisor,
            offset,
            scale,
        } => CudaBatchNormActivation::GeluErf {
            divisor,
            offset,
            scale,
        }
        .launch_parameters(),
    }
}

fn as_u64(value: usize) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| candle_core::Error::Msg("batch norm dimension exceeds u64".into()))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{execute, prepare_statistics, Activation};

    fn values(tensor: &Tensor) -> Vec<f32> {
        tensor
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn fused_cuda_batch_norm_matches_the_explicit_graph_formula() {
        let device = Device::new_cuda_with_stream(0).unwrap();
        let input = Tensor::from_iter(
            (0..2 * 3 * 4 * 5).map(|value| (value as f32 - 41.0) / 17.0),
            &device,
        )
        .unwrap()
        .reshape((2, 3, 4, 5))
        .unwrap();
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
        let gate = (&expected * 0.2)
            .unwrap()
            .affine(1.0, 0.5)
            .unwrap()
            .clamp(0.0, 1.0)
            .unwrap();
        let expected_hard_swish = expected.broadcast_mul(&gate).unwrap();
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
        actual.device().synchronize().unwrap();

        let expected = values(&expected);
        let actual = values(&actual);

        assert_eq!(actual.len(), expected.len());
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
        assert!(values(&actual_hard_swish)
            .iter()
            .zip(values(&expected_hard_swish))
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
        assert!(values(&actual_sigmoid)
            .iter()
            .zip(values(&expected_sigmoid))
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
        assert!(values(&actual_swish)
            .iter()
            .zip(values(&expected_swish))
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
        assert!(values(&actual_gelu)
            .iter()
            .zip(values(&expected_gelu))
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
    }
}
