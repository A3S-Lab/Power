use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp3, Layout, Result, Shape, Storage, Tensor};

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/layer_norm_affine_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_layer_norm_affine_f32_v2";
const TAIL_FUNCTION_NAME: &str = "layer_norm_affine_tail_f32";
const FULL_FUNCTION_NAME: &str = "layer_norm_full_f32";

pub(super) fn execute_full(
    input: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
    epsilon: f32,
) -> Result<Tensor> {
    // A preceding transpose can expose a valid last-axis LayerNorm through a
    // strided view. Materialize that boundary once, then replace the remaining
    // arithmetic/reduction chain with the fused kernel. Contiguous inputs stay
    // zero-copy because Candle returns the existing tensor.
    let input = input.contiguous()?;
    let operation = FullLayerNorm::new(input.layout(), scale.layout(), bias.layout(), epsilon)?;
    input.apply_op3_no_bwd(scale, bias, &operation)
}

#[derive(Clone)]
struct FullLayerNorm {
    shape: Shape,
    elements: usize,
    rows: u32,
    features: u32,
    block_size: u32,
    mean_scale: f32,
    epsilon: f32,
}

impl FullLayerNorm {
    fn new(input: &Layout, scale: &Layout, bias: &Layout, epsilon: f32) -> Result<Self> {
        if !input.is_contiguous() || !scale.is_contiguous() || !bias.is_contiguous() {
            candle_core::bail!("fused full LayerNorm requires contiguous inputs")
        }
        let Some(&features) = input.shape().dims().last() else {
            candle_core::bail!("fused full LayerNorm requires a non-scalar input")
        };
        if features == 0 || scale.dims() != [features] || bias.dims() != [features] {
            candle_core::bail!("fused full LayerNorm requires exact last-axis affine shapes")
        }
        let elements = input.shape().elem_count();
        if elements == 0 || !elements.is_multiple_of(features) || u32::try_from(elements).is_err() {
            candle_core::bail!("fused full LayerNorm exceeds the reviewed launch bound")
        }
        let rows = elements / features;
        let block_size = features.min(1024).next_power_of_two();
        Ok(Self {
            shape: input.shape().clone(),
            elements,
            rows: u32::try_from(rows)
                .map_err(|_| candle_core::Error::Msg("LayerNorm row count exceeds u32".into()))?,
            features: u32::try_from(features)
                .map_err(|_| candle_core::Error::Msg("feature count exceeds u32".into()))?,
            block_size: u32::try_from(block_size)
                .map_err(|_| candle_core::Error::Msg("block size exceeds u32".into()))?,
            mean_scale: (1.0_f64 / features as f64) as f32,
            epsilon,
        })
    }
}

impl CustomOp3 for FullLayerNorm {
    fn name(&self) -> &'static str {
        "a3s-fused-full-layer-norm"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _scale: &CpuStorage,
        _scale_layout: &Layout,
        _bias: &CpuStorage,
        _bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused full LayerNorm is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        scale: &CudaStorage,
        scale_layout: &Layout,
        bias: &CudaStorage,
        bias_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let input_values = input.as_cuda_slice::<f32>()?;
        let scale_values = scale.as_cuda_slice::<f32>()?;
        let bias_values = bias.as_cuda_slice::<f32>()?;
        let device = input.device();
        let function = device.get_or_load_custom_func(
            FULL_FUNCTION_NAME,
            MODULE_NAME,
            kernels::LAYER_NORM_AFFINE,
        )?;
        let output = unsafe { device.alloc::<f32>(self.elements)? };
        let input_offset = layout_offset(input_layout, "LayerNorm input")?;
        let scale_offset = layout_offset(scale_layout, "LayerNorm scale")?;
        let bias_offset = layout_offset(bias_layout, "LayerNorm bias")?;
        let element_count = u64::try_from(self.elements)
            .map_err(|_| candle_core::Error::Msg("LayerNorm size exceeds u64".into()))?;
        let mut builder = function.builder();
        builder.arg(input_values);
        builder.arg(scale_values);
        builder.arg(bias_values);
        builder.arg(&output);
        builder.arg(&input_offset);
        builder.arg(&scale_offset);
        builder.arg(&bias_offset);
        builder.arg(&element_count);
        builder.arg(&self.features);
        builder.arg(&self.mean_scale);
        builder.arg(&self.epsilon);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: (self.rows, 1, 1),
                    block_dim: (self.block_size, 1, 1),
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

pub(super) fn execute(
    centered: &Tensor,
    variance: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
    epsilon: f32,
) -> Result<Tensor> {
    let operation = LayerNormAffineTail::new(
        centered.layout(),
        variance.layout(),
        scale.layout(),
        bias,
        epsilon,
    )?;
    centered.apply_op3_no_bwd(variance, scale, &operation)
}

#[derive(Clone)]
struct LayerNormAffineTail {
    shape: Shape,
    elements: usize,
    features: u32,
    bias: Tensor,
    epsilon: f32,
}

impl LayerNormAffineTail {
    fn new(
        centered: &Layout,
        variance: &Layout,
        scale: &Layout,
        bias: &Tensor,
        epsilon: f32,
    ) -> Result<Self> {
        if !centered.is_contiguous()
            || !variance.is_contiguous()
            || !scale.is_contiguous()
            || !bias.is_contiguous()
        {
            candle_core::bail!("fused LayerNorm affine tail requires contiguous inputs")
        }
        let centered_dimensions = centered.shape().dims();
        let variance_dimensions = variance.shape().dims();
        let Some((&features, centered_prefix)) = centered_dimensions.split_last() else {
            candle_core::bail!("fused LayerNorm affine tail requires a non-scalar input")
        };
        let Some((&variance_features, variance_prefix)) = variance_dimensions.split_last() else {
            candle_core::bail!("fused LayerNorm variance requires a non-scalar input")
        };
        if features == 0
            || centered_prefix != variance_prefix
            || variance_features != 1
            || scale.shape().dims() != [features]
            || bias.dims() != [features]
        {
            candle_core::bail!(
                "fused LayerNorm affine tail requires exact last-axis broadcast shapes"
            )
        }
        let elements = centered.shape().elem_count();
        if elements == 0 || u32::try_from(elements).is_err() {
            candle_core::bail!("fused LayerNorm affine tail exceeds the reviewed launch bound")
        }
        Ok(Self {
            shape: centered.shape().clone(),
            elements,
            features: u32::try_from(features)
                .map_err(|_| candle_core::Error::Msg("feature count exceeds u32".into()))?,
            bias: bias.clone(),
            epsilon,
        })
    }
}

impl CustomOp3 for LayerNormAffineTail {
    fn name(&self) -> &'static str {
        "a3s-fused-layer-norm-affine-tail"
    }

    fn cpu_fwd(
        &self,
        _centered: &CpuStorage,
        _centered_layout: &Layout,
        _variance: &CpuStorage,
        _variance_layout: &Layout,
        _scale: &CpuStorage,
        _scale_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused LayerNorm affine tail is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        centered: &CudaStorage,
        centered_layout: &Layout,
        variance: &CudaStorage,
        variance_layout: &Layout,
        scale: &CudaStorage,
        scale_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let centered_values = centered.as_cuda_slice::<f32>()?;
        let variance_values = variance.as_cuda_slice::<f32>()?;
        let scale_values = scale.as_cuda_slice::<f32>()?;
        let (bias_storage, bias_layout) = self.bias.storage_and_layout();
        let Storage::Cuda(bias) = &*bias_storage else {
            candle_core::bail!("fused LayerNorm affine bias is not CUDA-resident")
        };
        let bias_values = bias.as_cuda_slice::<f32>()?;
        let device = centered.device();
        let function = device.get_or_load_custom_func(
            TAIL_FUNCTION_NAME,
            MODULE_NAME,
            kernels::LAYER_NORM_AFFINE,
        )?;
        let output = unsafe { device.alloc::<f32>(self.elements)? };
        let centered_offset = layout_offset(centered_layout, "centered input")?;
        let variance_offset = layout_offset(variance_layout, "variance")?;
        let scale_offset = layout_offset(scale_layout, "affine scale")?;
        let bias_offset = layout_offset(bias_layout, "affine bias")?;
        let element_count = u64::try_from(self.elements)
            .map_err(|_| candle_core::Error::Msg("LayerNorm size exceeds u64".into()))?;
        let mut builder = function.builder();
        builder.arg(centered_values);
        builder.arg(variance_values);
        builder.arg(scale_values);
        builder.arg(bias_values);
        builder.arg(&output);
        builder.arg(&centered_offset);
        builder.arg(&variance_offset);
        builder.arg(&scale_offset);
        builder.arg(&bias_offset);
        builder.arg(&element_count);
        builder.arg(&self.features);
        builder.arg(&self.epsilon);
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
    fn fused_full_layer_norm_is_byte_exact_with_the_existing_cuda_sequence() {
        let cuda = Device::new_cuda(0).unwrap();
        let epsilon = 0.00001_f32;
        let epsilon_tensor = Tensor::new(&[epsilon], &cuda).unwrap();

        for features in [4_usize, 120, 320, 1025] {
            let element_count = 6 * features;
            let input_values = (0..element_count)
                .map(|index| {
                    let numerator = ((index * 37 + 11) % 257) as f32 - 128.0;
                    numerator / 17.0 + (index % 7) as f32 / 31.0
                })
                .collect::<Vec<_>>();
            let scale_values = (0..features)
                .map(|index| ((index * 13 + 5) % 43) as f32 / 19.0 - 0.75)
                .collect::<Vec<_>>();
            let bias_values = (0..features)
                .map(|index| ((index * 17 + 3) % 47) as f32 / 23.0 - 1.0)
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(input_values, (2, 3, features), &cuda).unwrap();
            let scale = Tensor::from_vec(scale_values, features, &cuda).unwrap();
            let bias = Tensor::from_vec(bias_values, features, &cuda).unwrap();

            let mean = input.mean_keepdim(2).unwrap();
            let centered = input.broadcast_sub(&mean).unwrap();
            let variance = centered.sqr().unwrap().mean_keepdim(2).unwrap();
            let expected = variance
                .broadcast_add(&epsilon_tensor)
                .and_then(|value| value.sqrt())
                .and_then(|denominator| centered.broadcast_div(&denominator))
                .and_then(|value| value.broadcast_mul(&scale))
                .and_then(|value| value.broadcast_add(&bias))
                .unwrap();
            let actual = execute_full(&input, &scale, &bias, epsilon).unwrap();

            assert_eq!(
                bits(&actual),
                bits(&expected),
                "feature width {features} must preserve every f32 rounding boundary"
            );
        }

        let features = 120_usize;
        let input_values = (0..6 * features)
            .map(|index| {
                let numerator = ((index * 29 + 7) % 251) as f32 - 125.0;
                numerator / 13.0 + (index % 11) as f32 / 37.0
            })
            .collect::<Vec<_>>();
        let scale_values = (0..features)
            .map(|index| ((index * 11 + 3) % 41) as f32 / 17.0 - 0.5)
            .collect::<Vec<_>>();
        let bias_values = (0..features)
            .map(|index| ((index * 19 + 1) % 53) as f32 / 29.0 - 0.875)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input_values, (1, features, 6), &cuda)
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        assert!(!input.is_contiguous());
        let scale = Tensor::from_vec(scale_values, features, &cuda).unwrap();
        let bias = Tensor::from_vec(bias_values, features, &cuda).unwrap();
        let mean = input.mean_keepdim(2).unwrap();
        let centered = input.broadcast_sub(&mean).unwrap();
        let variance = centered.sqr().unwrap().mean_keepdim(2).unwrap();
        let expected = variance
            .broadcast_add(&epsilon_tensor)
            .and_then(|value| value.sqrt())
            .and_then(|denominator| centered.broadcast_div(&denominator))
            .and_then(|value| value.broadcast_mul(&scale))
            .and_then(|value| value.broadcast_add(&bias))
            .unwrap();
        let actual = execute_full(&input, &scale, &bias, epsilon).unwrap();
        assert_eq!(
            bits(&actual),
            bits(&expected),
            "a transposed source view must preserve every f32 rounding boundary"
        );
    }

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn fused_full_layer_norm_is_repeatable_across_many_rows() {
        let cuda = Device::new_cuda(0).unwrap();
        let features = 120_usize;
        let rows = 4_096_usize;
        let input = Tensor::from_iter(
            (0..rows * features).map(|index| {
                let centered = ((index * 37 + index / features * 11) % 509) as f32 - 254.0;
                centered / 131.0 + (index % 13) as f32 / 257.0
            }),
            &cuda,
        )
        .unwrap()
        .reshape((rows, features))
        .unwrap();
        let scale = Tensor::from_iter(
            (0..features).map(|index| ((index * 13 + 5) % 43) as f32 / 19.0 - 0.75),
            &cuda,
        )
        .unwrap();
        let bias = Tensor::from_iter(
            (0..features).map(|index| ((index * 17 + 3) % 47) as f32 / 23.0 - 1.0),
            &cuda,
        )
        .unwrap();
        let first = bits(&execute_full(&input, &scale, &bias, 0.00001).unwrap());

        for repetition in 0..8 {
            let repeated = bits(&execute_full(&input, &scale, &bias, 0.00001).unwrap());
            assert_eq!(
                repeated, first,
                "full LayerNorm changed on repetition {repetition}"
            );
        }
    }

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn fused_tail_is_byte_exact_with_the_existing_cuda_sequence() {
        let cuda = Device::new_cuda(0).unwrap();
        let centered = Tensor::new(
            &[
                -4.0_f32, -1.5, 0.25, 3.0, -7.0, -0.0, 0.5, 8.0, -2.0, -0.25, 1.5, 5.0, -9.0, -3.0,
                2.0, 11.0, -6.0, -2.5, 0.75, 7.75, -1.0, -0.5, 4.0, 10.0,
            ],
            &cuda,
        )
        .unwrap()
        .reshape((2, 3, 4))
        .unwrap();
        let variance = Tensor::new(&[2.0_f32, 5.0, 0.125, 17.0, 1.0, 9.0], &cuda)
            .unwrap()
            .reshape((2, 3, 1))
            .unwrap();
        let scale = Tensor::new(&[0.5_f32, -1.25, 2.0, 0.125], &cuda).unwrap();
        let bias = Tensor::new(&[-0.75_f32, 1.0, 0.25, -2.0], &cuda).unwrap();
        let epsilon = 0.00001_f32;
        let epsilon_tensor = Tensor::new(&[epsilon], &cuda).unwrap();
        let expected = variance
            .broadcast_add(&epsilon_tensor)
            .and_then(|value| value.sqrt())
            .and_then(|denominator| centered.broadcast_div(&denominator))
            .and_then(|value| value.broadcast_mul(&scale))
            .and_then(|value| value.broadcast_add(&bias))
            .unwrap();
        let actual = execute(&centered, &variance, &scale, &bias, epsilon).unwrap();
        assert_eq!(bits(&actual), bits(&expected));
    }
}
