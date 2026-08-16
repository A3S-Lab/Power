use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp3, Layout, Result, Shape, Storage, Tensor};

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/layer_norm_affine_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_layer_norm_affine_f32_v1";
const FUNCTION_NAME: &str = "layer_norm_affine_tail_f32";

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
            FUNCTION_NAME,
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
