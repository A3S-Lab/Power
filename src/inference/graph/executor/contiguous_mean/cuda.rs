use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Tensor};

mod kernels {
    include!(concat!(env!("OUT_DIR"), "/contiguous_mean_ptx.rs"));
}

const MODULE_NAME: &str = "a3s_power_contiguous_mean_f32_v1";
const FUNCTION_NAME: &str = "contiguous_suffix_mean_f32";
const MAXIMUM_THREADS: usize = 1024;

pub(super) fn try_execute(
    input: &Tensor,
    first_axis: usize,
    keep_dimensions: bool,
) -> Result<Option<Tensor>> {
    let Some(operation) = ContiguousMean::new(input, first_axis, keep_dimensions) else {
        return Ok(None);
    };
    input.apply_op1_no_bwd(&operation).map(Some)
}

#[derive(Clone)]
struct ContiguousMean {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    rows: u32,
    reduced_elements: u32,
    block_size: u32,
    scale: f32,
}

impl ContiguousMean {
    fn new(input: &Tensor, first_axis: usize, keep_dimensions: bool) -> Option<Self> {
        let dimensions = input.dims();
        let rows = dimensions[..first_axis]
            .iter()
            .try_fold(1_usize, |total, dimension| total.checked_mul(*dimension))?;
        let reduced_elements = dimensions[first_axis..]
            .iter()
            .try_fold(1_usize, |total, dimension| total.checked_mul(*dimension))?;
        let total_elements = rows.checked_mul(reduced_elements)?;
        if rows == 0
            || reduced_elements == 0
            || rows > u32::MAX as usize
            || reduced_elements > u32::MAX as usize
            || total_elements > u32::MAX as usize
        {
            return None;
        }
        let block_size = reduced_elements.min(MAXIMUM_THREADS).next_power_of_two();
        let mut output_shape = dimensions[..first_axis].to_vec();
        if keep_dimensions {
            output_shape.resize(dimensions.len(), 1);
        }
        Some(Self {
            input_shape: dimensions.to_vec(),
            output_shape,
            rows: u32::try_from(rows).ok()?,
            reduced_elements: u32::try_from(reduced_elements).ok()?,
            block_size: u32::try_from(block_size).ok()?,
            scale: (1_f64 / reduced_elements as f64) as f32,
        })
    }
}

impl CustomOp1 for ContiguousMean {
    fn name(&self) -> &'static str {
        "a3s-cuda-contiguous-suffix-mean"
    }

    fn cpu_fwd(&self, _input: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("the contiguous suffix mean is CUDA-only")
    }

    fn cuda_fwd(&self, input: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        if !layout.is_contiguous() || layout.dims() != self.input_shape {
            candle_core::bail!("contiguous suffix mean input layout changed before execution")
        }
        let device = input.device();
        let function =
            device.get_or_load_custom_func(FUNCTION_NAME, MODULE_NAME, kernels::CONTIGUOUS_MEAN)?;
        let output = unsafe { device.alloc::<f32>(self.rows as usize)? };
        let input_offset = u64::try_from(layout.start_offset()).map_err(|_| {
            candle_core::Error::Msg("contiguous mean input offset exceeds u64".into())
        })?;
        let mut builder = function.builder();
        builder.arg(input.as_cuda_slice::<f32>()?);
        builder.arg(&output);
        builder.arg(&input_offset);
        builder.arg(&self.reduced_elements);
        builder.arg(&self.scale);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.rows, 1, 1),
                block_dim: (self.block_size, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from_dims(&self.output_shape),
        ))
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
    fn suffix_mean_matches_candle_bits_across_reduction_sizes() {
        let device = Device::new_cuda_with_stream(0).unwrap();
        for (shape, first_axis) in [
            (vec![2, 3, 4], 2),
            (vec![2, 3, 4, 5], 2),
            (vec![2, 3, 33, 35], 2),
        ] {
            let elements = shape.iter().product::<usize>();
            let input = Tensor::from_iter(
                (0..elements).map(|index| ((index * 17 % 251) as f32 - 125.0) / 127.0),
                &device,
            )
            .unwrap()
            .reshape(shape.as_slice())
            .unwrap();
            let axes = (first_axis..shape.len()).collect::<Vec<_>>();
            let expected = input.mean_keepdim(axes.as_slice()).unwrap();
            let actual = try_execute(&input, first_axis, true).unwrap().unwrap();
            actual.device().synchronize().unwrap();

            assert_eq!(actual.dims(), expected.dims(), "shape={shape:?}");
            assert_eq!(bits(&actual), bits(&expected), "shape={shape:?}");
        }
    }
}
