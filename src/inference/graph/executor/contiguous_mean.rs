use candle_core::{DType, Result, Tensor};

#[cfg(feature = "embedded-cuda")]
mod cuda;

/// Uses an exact contiguous suffix reduction when the selected device has a
/// reviewed implementation. Other layouts, dtypes, axes, and devices remain
/// on Candle's authoritative mean path.
pub(super) fn try_execute(
    input: &Tensor,
    axes: &[usize],
    _keep_dimensions: bool,
) -> Result<Option<Tensor>> {
    if input.dtype() != DType::F32 || axes.is_empty() || !input.is_contiguous() {
        return Ok(None);
    }
    let first_axis = axes[0];
    if first_axis >= input.rank() || axes.iter().copied().ne(first_axis..input.rank()) {
        return Ok(None);
    }

    #[cfg(feature = "embedded-cuda")]
    if input.device().is_cuda() {
        return cuda::try_execute(input, first_axis, _keep_dimensions);
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    #[test]
    fn cpu_and_non_suffix_reductions_remain_on_candle() {
        let input = Tensor::arange(0_f32, 24_f32, &Device::Cpu)
            .unwrap()
            .reshape((2, 3, 4))
            .unwrap();

        assert!(try_execute(&input, &[2], true).unwrap().is_none());
        assert!(try_execute(&input, &[1], true).unwrap().is_none());
        assert!(try_execute(&input, &[], true).unwrap().is_none());
    }
}
