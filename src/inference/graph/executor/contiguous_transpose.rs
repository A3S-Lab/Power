use candle_core::{Result, Tensor};

#[cfg(feature = "embedded-cuda")]
mod cuda;

/// Materializes a tensor for a dense consumer while preserving its logical
/// element order. CUDA last-two-axis transpose views use a bounded tiled copy;
/// every other layout remains on Candle's authoritative contiguous path.
pub(super) fn materialize(input: &Tensor) -> Result<Tensor> {
    if input.is_contiguous() {
        return Ok(input.clone());
    }
    #[cfg(feature = "embedded-cuda")]
    if input.device().is_cuda() {
        if let Some(output) = cuda::try_materialize(input)? {
            return Ok(output);
        }
    }
    input.contiguous()
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

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
    fn cpu_transpose_remains_on_the_authoritative_contiguous_path() {
        let source = Tensor::arange(0_f32, 2.0 * 3.0 * 5.0, &Device::Cpu)
            .unwrap()
            .reshape((2, 3, 5))
            .unwrap();
        let view = source.transpose(1, 2).unwrap();
        let expected = view.contiguous().unwrap();
        let actual = materialize(&view).unwrap();

        assert_eq!(actual.dims(), expected.dims());
        assert_eq!(bits(&actual), bits(&expected));
    }

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn tiled_cuda_copy_matches_candle_bits_for_general_prefixes_and_offsets() {
        let device = Device::new_cuda_with_stream(0).unwrap();
        for shape in [vec![5, 7], vec![3, 5, 7], vec![2, 3, 5, 7]] {
            let elements = shape.iter().product::<usize>();
            let source = Tensor::from_iter(
                (0..elements).map(|value| ((value * 17 % 251) as f32 - 125.0) / 127.0),
                &device,
            )
            .unwrap()
            .reshape(shape.as_slice())
            .unwrap();
            let rank = source.rank();
            let view = source.transpose(rank - 2, rank - 1).unwrap();
            let expected = view.contiguous().unwrap();
            let actual = materialize(&view).unwrap();
            actual.device().synchronize().unwrap();

            assert!(actual.is_contiguous());
            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(bits(&actual), bits(&expected), "shape={shape:?}");
        }

        let source = Tensor::from_iter((0..4 * 5 * 7).map(|value| value as f32 / 13.0), &device)
            .unwrap()
            .reshape((4, 5, 7))
            .unwrap();
        let view = source.narrow(0, 1, 2).unwrap().transpose(1, 2).unwrap();
        let expected = view.contiguous().unwrap();
        let actual = materialize(&view).unwrap();
        actual.device().synchronize().unwrap();
        assert_eq!(bits(&actual), bits(&expected));

        let nonmatching = source.transpose(0, 1).unwrap();
        let expected = nonmatching.contiguous().unwrap();
        let actual = materialize(&nonmatching).unwrap();
        actual.device().synchronize().unwrap();
        assert_eq!(bits(&actual), bits(&expected));

        assert_eq!(actual.dtype(), DType::F32);
    }
}
