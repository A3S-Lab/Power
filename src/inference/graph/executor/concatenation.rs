use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

/// Concatenates two contiguous CPU F32 tensors by copying independent prefix
/// blocks in parallel. Concatenation is a byte-preserving operation, so this
/// changes scheduling only and cannot change numeric semantics.
pub(super) fn concat_two(left: &Tensor, right: &Tensor, axis: usize) -> Result<Tensor> {
    let operation = ContiguousConcat2::new(left.layout(), right.layout(), axis)?;
    left.apply_op2_no_bwd(right, &operation)
}

#[derive(Clone)]
struct ContiguousConcat2 {
    output_shape: Shape,
    left_block: usize,
    right_block: usize,
    output_block: usize,
    output_elements: usize,
    left_elements: usize,
    right_elements: usize,
}

impl ContiguousConcat2 {
    fn new(left: &Layout, right: &Layout, axis: usize) -> Result<Self> {
        if !left.is_contiguous() || !right.is_contiguous() {
            candle_core::bail!("direct CPU concatenation requires contiguous inputs")
        }
        let left_dimensions = left.shape().dims();
        let right_dimensions = right.shape().dims();
        if left_dimensions.len() != right_dimensions.len() || axis >= left_dimensions.len() {
            candle_core::bail!("direct CPU concatenation has an invalid axis or rank")
        }
        for (index, (&left, &right)) in left_dimensions.iter().zip(right_dimensions).enumerate() {
            if index != axis && left != right {
                candle_core::bail!(
                    "direct CPU concatenation requires equal non-concatenated dimensions"
                )
            }
        }
        let outer = checked_product(&left_dimensions[..axis], "prefix")?;
        let suffix = checked_product(&left_dimensions[axis + 1..], "suffix")?;
        let left_block = left_dimensions[axis]
            .checked_mul(suffix)
            .ok_or_else(|| concat_error("left block cardinality overflowed"))?;
        let right_block = right_dimensions[axis]
            .checked_mul(suffix)
            .ok_or_else(|| concat_error("right block cardinality overflowed"))?;
        let output_block = left_block
            .checked_add(right_block)
            .ok_or_else(|| concat_error("output block cardinality overflowed"))?;
        let left_elements = outer
            .checked_mul(left_block)
            .ok_or_else(|| concat_error("left tensor cardinality overflowed"))?;
        let right_elements = outer
            .checked_mul(right_block)
            .ok_or_else(|| concat_error("right tensor cardinality overflowed"))?;
        let output_elements = outer
            .checked_mul(output_block)
            .ok_or_else(|| concat_error("output tensor cardinality overflowed"))?;
        let mut output_dimensions = left_dimensions.to_vec();
        output_dimensions[axis] = left_dimensions[axis]
            .checked_add(right_dimensions[axis])
            .ok_or_else(|| concat_error("output axis overflowed"))?;
        Ok(Self {
            output_shape: Shape::from_dims(&output_dimensions),
            left_block,
            right_block,
            output_block,
            output_elements,
            left_elements,
            right_elements,
        })
    }

    fn execute(&self, left: &[f32], right: &[f32]) -> (CpuStorage, Shape) {
        let mut output = vec![0.0_f32; self.output_elements];
        if output.is_empty() {
            return (CpuStorage::F32(output), self.output_shape.clone());
        }
        output
            .par_chunks_mut(self.output_block)
            .enumerate()
            .for_each(|(outer, output)| {
                let left_start = outer * self.left_block;
                let right_start = outer * self.right_block;
                output[..self.left_block]
                    .copy_from_slice(&left[left_start..left_start + self.left_block]);
                output[self.left_block..]
                    .copy_from_slice(&right[right_start..right_start + self.right_block]);
            });
        (CpuStorage::F32(output), self.output_shape.clone())
    }
}

impl CustomOp2 for ContiguousConcat2 {
    fn name(&self) -> &'static str {
        "a3s-direct-cpu-contiguous-concat-two"
    }

    fn cpu_fwd(
        &self,
        left: &CpuStorage,
        left_layout: &Layout,
        right: &CpuStorage,
        right_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let left = contiguous_values(
            left.as_slice::<f32>()?,
            left_layout,
            self.left_elements,
            "left",
        )?;
        let right = contiguous_values(
            right.as_slice::<f32>()?,
            right_layout,
            self.right_elements,
            "right",
        )?;
        Ok(self.execute(left, right))
    }
}

fn checked_product(dimensions: &[usize], label: &str) -> Result<usize> {
    dimensions.iter().try_fold(1_usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| concat_error(format!("{label} cardinality overflowed")))
    })
}

fn contiguous_values<'a>(
    storage: &'a [f32],
    layout: &Layout,
    elements: usize,
    label: &str,
) -> Result<&'a [f32]> {
    let start = layout.start_offset();
    let end = start
        .checked_add(elements)
        .ok_or_else(|| concat_error(format!("{label} layout range overflowed")))?;
    storage
        .get(start..end)
        .ok_or_else(|| concat_error(format!("{label} layout is out of bounds")))
}

fn concat_error(message: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(message.into())
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn direct_concat_matches_candle_across_axes_and_offset_layouts() {
        for (left_shape, right_shape, axis) in [
            (vec![3, 2, 4, 7], vec![3, 5, 4, 7], 1),
            (vec![2, 3, 5], vec![2, 3, 11], 2),
            (vec![4, 3, 2], vec![4, 7, 2], 1),
            (vec![2, 3], vec![5, 3], 0),
        ] {
            let left_elements = left_shape.iter().product::<usize>();
            let right_elements = right_shape.iter().product::<usize>();
            let left = Tensor::from_iter(
                (0..left_elements).map(|value| (value as f32 - 17.0) / 31.0),
                &Device::Cpu,
            )
            .unwrap()
            .reshape(left_shape.as_slice())
            .unwrap();
            let right = Tensor::from_iter(
                (0..right_elements).map(|value| (value as f32 + 13.0) / 29.0),
                &Device::Cpu,
            )
            .unwrap()
            .reshape(right_shape.as_slice())
            .unwrap();
            let expected = Tensor::cat(&[&left, &right], axis).unwrap();
            let actual = concat_two(&left, &right, axis).unwrap();
            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(
                actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            );
        }

        let left = Tensor::from_iter((0..3 * 2 * 5).map(|value| value as f32), &Device::Cpu)
            .unwrap()
            .reshape((3, 2, 5))
            .unwrap()
            .narrow(0, 1, 2)
            .unwrap();
        let right = Tensor::from_iter(
            (0..3 * 4 * 5).map(|value| (value as f32) * -0.5),
            &Device::Cpu,
        )
        .unwrap()
        .reshape((3, 4, 5))
        .unwrap()
        .narrow(0, 1, 2)
        .unwrap();
        assert!(left.is_contiguous() && right.is_contiguous());
        assert_ne!(left.layout().start_offset(), 0);
        assert_ne!(right.layout().start_offset(), 0);
        let expected = Tensor::cat(&[&left, &right], 1).unwrap();
        let actual = concat_two(&left, &right, 1).unwrap();
        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
    }
}
