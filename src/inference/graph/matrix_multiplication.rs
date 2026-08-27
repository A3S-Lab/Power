#[cfg(any(test, feature = "embedded-cuda"))]
use candle_core::DType;
use candle_core::{Result, Tensor};

/// Applies ONNX-style matrix broadcasting while retaining a shared CUDA
/// rank-two right-hand matrix as a zero-stride batch view when possible.
///
/// Candle's authoritative broadcast path materializes broadcast operands.
/// CUDA strided-batched GEMM accepts a zero batch stride, so every batch can
/// read the same immutable matrix without allocating or copying replicas. The
/// batch count and individual GEMM geometry remain unchanged.
pub(super) fn broadcast(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "embedded-cuda")]
    if let Some(output) = cuda_shared_rank_two_rhs(left, right)? {
        return Ok(output);
    }

    left.broadcast_matmul(right)
}

/// Executes one reviewed CUDA layout directly when a dense consumer would
/// otherwise materialize a last-two-axis transpose first.
///
/// The gate describes only cuBLAS' existing strided-matrix capability. It does
/// not select on a model, tensor value, or measured geometry. Every other
/// layout retains the authoritative materialization path at the executor
/// boundary.
pub(super) fn try_cuda_transposed_lhs(left: &Tensor, right: &Tensor) -> Result<Option<Tensor>> {
    #[cfg(feature = "embedded-cuda")]
    {
        cuda_transposed_rank_three_lhs(left, right)
    }

    #[cfg(not(feature = "embedded-cuda"))]
    {
        let _ = (left, right);
        Ok(None)
    }
}

#[cfg(feature = "embedded-cuda")]
fn cuda_transposed_rank_three_lhs(left: &Tensor, right: &Tensor) -> Result<Option<Tensor>> {
    if left.dtype() != DType::F32
        || right.dtype() != DType::F32
        || !left.device().is_cuda()
        || !right.device().is_cuda()
        || !left.device().same_device(right.device())
        || left.rank() != 3
        || right.rank() != 2
        || left.is_contiguous()
        || !right.is_contiguous()
    {
        return Ok(None);
    }

    let [batch, rows, inner] = left.dims() else {
        return Ok(None);
    };
    let [right_inner, columns] = right.dims() else {
        return Ok(None);
    };
    let Some(batch_stride) = rows.checked_mul(*inner) else {
        return Ok(None);
    };
    if *batch == 0
        || *rows == 0
        || *inner == 0
        || *columns == 0
        || inner != right_inner
        || left.stride() != [batch_stride, 1, *rows]
    {
        return Ok(None);
    }

    let right_broadcast = right
        .unsqueeze(0)?
        .broadcast_as((*batch, *inner, *columns))?;
    left.matmul(&right_broadcast).map(Some)
}

#[cfg(feature = "embedded-cuda")]
fn cuda_shared_rank_two_rhs(left: &Tensor, right: &Tensor) -> Result<Option<Tensor>> {
    if !left.device().is_cuda()
        || !right.device().is_cuda()
        || !left.device().same_device(right.device())
        || left.rank() <= 2
        || right.rank() != 2
        || !right.is_contiguous()
    {
        return Ok(None);
    }

    // A graph-derived transpose can leave the comparatively small dynamic
    // operand strided. Materialize that operand first so the immutable rank-2
    // matrix can still remain one zero-batch-stride view instead of copying a
    // full broadcast replica. This preserves logical order and GEMM geometry.
    let left = if left.is_contiguous() {
        left.clone()
    } else {
        left.contiguous()?
    };

    let left_dimensions = left.dims();
    let right_dimensions = right.dims();
    let matrix_rows = left_dimensions[left_dimensions.len() - 2];
    let inner = left_dimensions[left_dimensions.len() - 1];
    let classes = right_dimensions[1];
    if inner == 0 || inner != right_dimensions[0] {
        return Ok(None);
    }
    let Some(batch) = left_dimensions[..left_dimensions.len() - 2]
        .iter()
        .try_fold(1_usize, |total, dimension| total.checked_mul(*dimension))
    else {
        return Ok(None);
    };
    if batch == 0 {
        return Ok(None);
    }

    let left_batched = left.reshape((batch, matrix_rows, inner))?;
    let right_broadcast = right.unsqueeze(0)?.broadcast_as((batch, inner, classes))?;
    let output = left_batched.matmul(&right_broadcast)?;
    let mut output_dimensions = left_dimensions[..left_dimensions.len() - 1].to_vec();
    output_dimensions.push(classes);
    output.reshape(output_dimensions).map(Some)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

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
    fn cpu_uses_the_authoritative_broadcast_path() {
        let left = Tensor::zeros((3, 5, 7), candle_core::DType::F32, &Device::Cpu).unwrap();
        let right = Tensor::zeros((7, 11), candle_core::DType::F32, &Device::Cpu).unwrap();

        let actual = broadcast(&left, &right).unwrap();

        assert_eq!(actual.dims(), [3, 5, 11]);
    }

    #[test]
    fn nonmatching_geometry_uses_authoritative_broadcast() {
        let left = Tensor::zeros((2, 3), candle_core::DType::F32, &Device::Cpu).unwrap();
        let right = Tensor::zeros((4, 3, 5), candle_core::DType::F32, &Device::Cpu).unwrap();

        let actual = broadcast(&left, &right).unwrap();

        assert_eq!(actual.dims(), [4, 2, 5]);
    }

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn shared_cuda_rhs_matches_authoritative_broadcast_bits() {
        let device = Device::new_cuda_with_stream(0).unwrap();
        for shape in [vec![3, 5, 7], vec![2, 3, 5, 7]] {
            let elements = shape.iter().product::<usize>();
            let left = Tensor::from_iter(
                (0..elements).map(|value| ((value * 13 % 101) as f32 - 50.0) / 37.0),
                &device,
            )
            .unwrap()
            .reshape(shape.as_slice())
            .unwrap();
            let right = Tensor::from_iter(
                (0..7 * 11).map(|value| ((value * 19 % 89) as f32 - 44.0) / 31.0),
                &device,
            )
            .unwrap()
            .reshape((7, 11))
            .unwrap();
            let expected = left.broadcast_matmul(&right).unwrap();
            let actual = broadcast(&left, &right).unwrap();
            device.synchronize().unwrap();

            let mut expected_shape = shape[..shape.len() - 1].to_vec();
            expected_shape.push(11);
            assert_eq!(actual.dims(), expected_shape);
            assert_eq!(bits(&actual), bits(&expected), "shape={shape:?}");
        }

        let source = Tensor::from_iter(
            (0..2 * 5 * 3 * 7).map(|value| ((value * 23 % 127) as f32 - 63.0) / 41.0),
            &device,
        )
        .unwrap()
        .reshape((2, 5, 3, 7))
        .unwrap();
        let left = source.transpose(1, 2).unwrap();
        assert!(!left.is_contiguous());
        let right = Tensor::from_iter(
            (0..7 * 11).map(|value| ((value * 19 % 89) as f32 - 44.0) / 31.0),
            &device,
        )
        .unwrap()
        .reshape((7, 11))
        .unwrap();
        let expected = left.contiguous().unwrap().broadcast_matmul(&right).unwrap();
        let actual = broadcast(&left, &right).unwrap();
        device.synchronize().unwrap();

        assert_eq!(actual.dims(), [2, 3, 5, 11]);
        assert_eq!(bits(&actual), bits(&expected));
    }

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn transposed_cuda_lhs_matches_materialized_gemm_bits() {
        let device = Device::new_cuda_with_stream(0).unwrap();
        for (batch, rows, inner, columns) in [(1, 3, 5, 7), (3, 17, 11, 29), (2, 41, 15, 67)] {
            let source = Tensor::from_iter(
                (0..batch * inner * rows).map(|value| ((value * 23 % 127) as f32 - 63.0) / 41.0),
                &device,
            )
            .unwrap()
            .reshape((batch, inner, rows))
            .unwrap();
            let left = source.transpose(1, 2).unwrap();
            let right = Tensor::from_iter(
                (0..inner * columns).map(|value| ((value * 19 % 89) as f32 - 44.0) / 31.0),
                &device,
            )
            .unwrap()
            .reshape((inner, columns))
            .unwrap();
            let expected = left.contiguous().unwrap().broadcast_matmul(&right).unwrap();
            let actual = try_cuda_transposed_lhs(&left, &right)
                .unwrap()
                .expect("an exact rank-three last-two-axis transpose must be eligible");
            device.synchronize().unwrap();

            assert_eq!(actual.dims(), [batch, rows, columns]);
            assert_eq!(
                bits(&actual),
                bits(&expected),
                "geometry={batch}x{rows}x{inner}x{columns}"
            );
        }

        let contiguous = Tensor::zeros((2, 3, 5), DType::F32, &device).unwrap();
        let right = Tensor::zeros((5, 7), DType::F32, &device).unwrap();
        assert!(try_cuda_transposed_lhs(&contiguous, &right)
            .unwrap()
            .is_none());
    }
}
