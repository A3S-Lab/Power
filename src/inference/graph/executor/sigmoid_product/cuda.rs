use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Result, Shape, Tensor};

use super::{multiplier_layout, MultiplierLayout};
use crate::inference::graph::executor::cuda_fast_divisor::FastDivisorU32;

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/sigmoid_product_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_sigmoid_product_f32_v2";
const PRODUCT_FUNCTION: &str = "sigmoid_product_f32";
const MUL_SAME_SHAPE_FUNCTION: &str = "sigmoid_mul_f32";
const MUL_PER_CHANNEL_FUNCTION: &str = "sigmoid_mul_nchw_per_channel_f32";
const MUL_PER_SPATIAL_POSITION_FUNCTION: &str = "sigmoid_mul_nchw_per_spatial_position_f32";
const THREADS_PER_BLOCK: u32 = 512;

pub(super) fn execute_product(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    let operation = SigmoidProduct::new(left.layout(), right.layout())?;
    left.apply_op2_no_bwd(right, &operation)
}

pub(super) fn execute_sigmoid_mul(input: &Tensor, multiplier: &Tensor) -> Result<Tensor> {
    let operation = SigmoidMul::new(input.layout(), multiplier.layout())?;
    input.apply_op2_no_bwd(multiplier, &operation)
}

#[derive(Clone)]
struct SigmoidProduct {
    shape: Shape,
    elements: usize,
}

impl SigmoidProduct {
    fn new(left: &Layout, right: &Layout) -> Result<Self> {
        if !left.is_contiguous() || !right.is_contiguous() {
            candle_core::bail!("sigmoid product requires contiguous inputs")
        }
        if left.shape() != right.shape() {
            candle_core::bail!("sigmoid product requires equal input shapes")
        }
        let elements = left.shape().elem_count();
        validate_elements(elements, "sigmoid product")?;
        Ok(Self {
            shape: left.shape().clone(),
            elements,
        })
    }
}

impl CustomOp2 for SigmoidProduct {
    fn name(&self) -> &'static str {
        "a3s-fused-sigmoid-product"
    }

    fn cpu_fwd(
        &self,
        _left: &CpuStorage,
        _left_layout: &Layout,
        _right: &CpuStorage,
        _right_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused sigmoid product operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        left: &CudaStorage,
        left_layout: &Layout,
        right: &CudaStorage,
        right_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
        use candle_core::cuda_backend::WrapErr;

        let device = left.device();
        let function = device.get_or_load_custom_func(
            PRODUCT_FUNCTION,
            MODULE_NAME,
            kernels::SIGMOID_PRODUCT,
        )?;
        let output = unsafe { device.alloc::<f32>(self.elements)? };
        let left_offset = layout_offset(left_layout, "sigmoid-product left input")?;
        let right_offset = layout_offset(right_layout, "sigmoid-product right input")?;
        let element_count = element_count(self.elements, "sigmoid-product")?;
        let mut builder = function.builder();
        builder.arg(left.as_cuda_slice::<f32>()?);
        builder.arg(right.as_cuda_slice::<f32>()?);
        builder.arg(&output);
        builder.arg(&left_offset);
        builder.arg(&right_offset);
        builder.arg(&element_count);
        unsafe { builder.launch(launch_config(element_count)).w()? };
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            self.shape.clone(),
        ))
    }
}

#[derive(Clone)]
struct SigmoidMul {
    shape: Shape,
    elements: usize,
    layout: MultiplierLayout,
    index_parameters: [u32; 6],
}

impl SigmoidMul {
    fn new(input: &Layout, multiplier: &Layout) -> Result<Self> {
        if !input.is_contiguous() || !multiplier.is_contiguous() {
            candle_core::bail!("sigmoid multiplication requires contiguous inputs")
        }
        let layout = multiplier_layout(input.dims(), multiplier.dims()).ok_or_else(|| {
            candle_core::Error::Msg(
                "sigmoid multiplication requires equal shapes or a reviewed NCHW broadcast".into(),
            )
        })?;
        let elements = input.shape().elem_count();
        validate_elements(elements, "sigmoid multiplication")?;
        let index_parameters = match layout {
            MultiplierLayout::SameShape => [0; 6],
            MultiplierLayout::NchwPerChannel | MultiplierLayout::NchwPerSpatialPosition => {
                let (_, channels, height, width) = input.shape().dims4()?;
                let spatial = height.checked_mul(width).ok_or_else(|| {
                    candle_core::Error::Msg("sigmoid-mul spatial size overflowed".into())
                })?;
                nchw_index_parameters(channels, spatial)?
            }
        };
        Ok(Self {
            shape: input.shape().clone(),
            elements,
            layout,
            index_parameters,
        })
    }
}

impl CustomOp2 for SigmoidMul {
    fn name(&self) -> &'static str {
        "a3s-fused-sigmoid-mul"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _multiplier: &CpuStorage,
        _multiplier_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the fused sigmoid multiplication operation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        multiplier: &CudaStorage,
        multiplier_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
        use candle_core::cuda_backend::WrapErr;

        let device = input.device();
        let function_name = match self.layout {
            MultiplierLayout::SameShape => MUL_SAME_SHAPE_FUNCTION,
            MultiplierLayout::NchwPerChannel => MUL_PER_CHANNEL_FUNCTION,
            MultiplierLayout::NchwPerSpatialPosition => MUL_PER_SPATIAL_POSITION_FUNCTION,
        };
        let function =
            device.get_or_load_custom_func(function_name, MODULE_NAME, kernels::SIGMOID_PRODUCT)?;
        let output = unsafe { device.alloc::<f32>(self.elements)? };
        let input_offset = layout_offset(input_layout, "sigmoid-mul input")?;
        let multiplier_offset = layout_offset(multiplier_layout, "sigmoid-mul multiplier")?;
        let element_count = element_count(self.elements, "sigmoid-mul")?;
        let mut builder = function.builder();
        builder.arg(input.as_cuda_slice::<f32>()?);
        builder.arg(multiplier.as_cuda_slice::<f32>()?);
        builder.arg(&output);
        builder.arg(&input_offset);
        builder.arg(&multiplier_offset);
        builder.arg(&element_count);
        if self.layout != MultiplierLayout::SameShape {
            for value in &self.index_parameters {
                builder.arg(value);
            }
        }
        unsafe { builder.launch(launch_config(element_count)).w()? };
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            self.shape.clone(),
        ))
    }
}

fn validate_elements(elements: usize, label: &str) -> Result<()> {
    if elements == 0 || u32::try_from(elements).is_err() {
        candle_core::bail!("{label} exceeds the reviewed launch bound")
    }
    Ok(())
}

fn element_count(elements: usize, label: &str) -> Result<u32> {
    u32::try_from(elements)
        .map_err(|_| candle_core::Error::Msg(format!("{label} size exceeds u32")))
}

fn nchw_index_parameters(channels: usize, spatial: usize) -> Result<[u32; 6]> {
    let channels = u32::try_from(channels)
        .map_err(|_| candle_core::Error::Msg("sigmoid-mul channel count exceeds u32".into()))?;
    let spatial = u32::try_from(spatial)
        .map_err(|_| candle_core::Error::Msg("sigmoid-mul spatial size exceeds u32".into()))?;
    let [spatial_multiplier, spatial_shift, spatial] = FastDivisorU32::new(spatial)
        .ok_or_else(|| candle_core::Error::Msg("sigmoid-mul spatial size is zero".into()))?
        .launch_parameters();
    let [channels_multiplier, channels_shift, channels] = FastDivisorU32::new(channels)
        .ok_or_else(|| candle_core::Error::Msg("sigmoid-mul channel count is zero".into()))?
        .launch_parameters();
    Ok([
        spatial_multiplier,
        spatial_shift,
        spatial,
        channels_multiplier,
        channels_shift,
        channels,
    ])
}

fn layout_offset(layout: &Layout, label: &str) -> Result<u64> {
    u64::try_from(layout.start_offset())
        .map_err(|_| candle_core::Error::Msg(format!("{label} offset exceeds u64")))
}

fn launch_config(element_count: u32) -> candle_core::cuda_backend::cudarc::driver::LaunchConfig {
    candle_core::cuda_backend::cudarc::driver::LaunchConfig {
        grid_dim: (element_count.div_ceil(THREADS_PER_BLOCK), 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
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
    fn fused_product_is_byte_exact_across_unrelated_geometries_and_offsets() {
        let cuda = Device::new_cuda_with_stream(0).unwrap();
        for shape in [vec![2, 3, 5], vec![7, 11], vec![3, 17, 29]] {
            let elements = shape.iter().product::<usize>();
            let left_storage = Tensor::from_iter(
                (0..elements + 5).map(|index| ((index * 37 % 509) as f32 - 254.0) / 29.0),
                &cuda,
            )
            .unwrap();
            let right_storage = Tensor::from_iter(
                (0..elements + 7).map(|index| ((index * 19 % 397) as f32 - 198.0) / 23.0),
                &cuda,
            )
            .unwrap();
            let left = left_storage
                .narrow(0, 5, elements)
                .unwrap()
                .reshape(shape.as_slice())
                .unwrap();
            let right = right_storage
                .narrow(0, 7, elements)
                .unwrap()
                .reshape(shape.as_slice())
                .unwrap();
            let expected = candle_nn::ops::sigmoid(&left)
                .and_then(|left| {
                    candle_nn::ops::sigmoid(&right).and_then(|right| left.broadcast_mul(&right))
                })
                .unwrap();
            let actual = execute_product(&left, &right).unwrap();
            cuda.synchronize().unwrap();

            assert_eq!(actual.dims(), shape);
            assert_eq!(bits(&actual), bits(&expected), "shape={shape:?}");
        }
    }

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn fused_sigmoid_mul_is_byte_exact_for_generic_nchw_broadcasts_and_offsets() {
        let cuda = Device::new_cuda_with_stream(0).unwrap();
        for full_shape in [[2, 3, 5, 7], [1, 11, 3, 13], [3, 2, 1, 17]] {
            let [batch, channels, height, width] = full_shape;
            let full_elements = full_shape.iter().product::<usize>();
            let input_storage = Tensor::from_iter(
                (0..full_elements + 7).map(|index| ((index * 43 % 607) as f32 - 303.0) / 31.0),
                &cuda,
            )
            .unwrap();
            let input = input_storage
                .narrow(0, 7, full_elements)
                .unwrap()
                .reshape(&full_shape)
                .unwrap();

            let multiplier_shapes = [
                full_shape,
                [batch, channels, 1, 1],
                [batch, 1, height, width],
            ];
            for (case, multiplier_shape) in multiplier_shapes.into_iter().enumerate() {
                let multiplier_elements = multiplier_shape.iter().product::<usize>();
                let multiplier_storage = Tensor::from_iter(
                    (0..multiplier_elements + 5)
                        .map(|index| ((index * 29 % 431) as f32 - 215.0) / 27.0),
                    &cuda,
                )
                .unwrap();
                let multiplier = multiplier_storage
                    .narrow(0, 5, multiplier_elements)
                    .unwrap()
                    .reshape(&multiplier_shape)
                    .unwrap();
                let expected = candle_nn::ops::sigmoid(&input)
                    .and_then(|input| input.broadcast_mul(&multiplier))
                    .unwrap();
                let actual = execute_sigmoid_mul(&input, &multiplier).unwrap();
                cuda.synchronize().unwrap();

                assert_eq!(actual.dims(), full_shape);
                assert_eq!(
                    bits(&actual),
                    bits(&expected),
                    "shape={full_shape:?} case={case} multiplier={multiplier_shape:?}"
                );
            }
        }
    }
}
