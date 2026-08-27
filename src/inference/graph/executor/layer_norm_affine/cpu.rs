use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Storage, Tensor};
use rayon::prelude::*;

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
    rows: usize,
    features: usize,
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
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            candle_core::bail!(
                "fused CPU LayerNorm affine tail requires contiguous inputs and positive finite epsilon"
            )
        }
        let centered_dimensions = centered.shape().dims();
        let variance_dimensions = variance.shape().dims();
        let Some((&features, centered_prefix)) = centered_dimensions.split_last() else {
            candle_core::bail!("fused CPU LayerNorm affine tail requires a non-scalar input")
        };
        let Some((&variance_features, variance_prefix)) = variance_dimensions.split_last() else {
            candle_core::bail!("fused CPU LayerNorm variance requires a non-scalar input")
        };
        if features == 0
            || centered_prefix != variance_prefix
            || variance_features != 1
            || scale.shape().dims() != [features]
            || bias.dims() != [features]
        {
            candle_core::bail!(
                "fused CPU LayerNorm affine tail requires exact last-axis broadcast shapes"
            )
        }
        let elements = centered.shape().elem_count();
        let rows = elements.checked_div(features).ok_or_else(|| {
            candle_core::Error::Msg("LayerNorm feature count must be non-zero".into())
        })?;
        if rows == 0 || variance.shape().elem_count() != rows {
            candle_core::bail!("fused CPU LayerNorm affine tail requires non-empty exact rows")
        }
        Ok(Self {
            shape: centered.shape().clone(),
            rows,
            features,
            bias: bias.clone(),
            epsilon,
        })
    }

    #[inline]
    fn transform_row(
        &self,
        output: &mut [f32],
        centered: &[f32],
        variance: f32,
        scale: &[f32],
        bias: &[f32],
    ) {
        // Keep a named statement for every original graph node. This retains
        // the f32 rounding boundary of Add, Sqrt, Div, Mul, and Add.
        let shifted_variance = variance + self.epsilon;
        let denominator = shifted_variance.sqrt();
        for (((output, centered), scale), bias) in
            output.iter_mut().zip(centered).zip(scale).zip(bias)
        {
            let normalized = *centered / denominator;
            let scaled = normalized * *scale;
            *output = scaled + *bias;
        }
    }
}

impl CustomOp3 for LayerNormAffineTail {
    fn name(&self) -> &'static str {
        "a3s-fused-cpu-layer-norm-affine-tail"
    }

    fn cpu_fwd(
        &self,
        centered: &CpuStorage,
        centered_layout: &Layout,
        variance: &CpuStorage,
        variance_layout: &Layout,
        scale: &CpuStorage,
        scale_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let centered = contiguous_values(
            centered.as_slice::<f32>()?,
            centered_layout,
            self.rows * self.features,
            "centered input",
        )?;
        let variance = contiguous_values(
            variance.as_slice::<f32>()?,
            variance_layout,
            self.rows,
            "variance",
        )?;
        let scale = contiguous_values(
            scale.as_slice::<f32>()?,
            scale_layout,
            self.features,
            "affine scale",
        )?;
        let (bias_storage, bias_layout) = self.bias.storage_and_layout();
        let Storage::Cpu(bias_storage) = &*bias_storage else {
            candle_core::bail!("fused CPU LayerNorm affine bias is not CPU-resident")
        };
        let bias = contiguous_values(
            bias_storage.as_slice::<f32>()?,
            bias_layout,
            self.features,
            "affine bias",
        )?;

        let mut output = vec![0.0_f32; self.rows * self.features];
        if rayon::current_thread_index().is_some() || rayon::current_num_threads() <= 1 {
            for (row, output) in output.chunks_mut(self.features).enumerate() {
                let start = row * self.features;
                self.transform_row(
                    output,
                    &centered[start..start + self.features],
                    variance[row],
                    scale,
                    bias,
                );
            }
        } else {
            output
                .par_chunks_mut(self.features)
                .enumerate()
                .for_each(|(row, output)| {
                    let start = row * self.features;
                    self.transform_row(
                        output,
                        &centered[start..start + self.features],
                        variance[row],
                        scale,
                        bias,
                    );
                });
        }
        Ok((CpuStorage::F32(output), self.shape.clone()))
    }
}

fn contiguous_values<'a>(
    storage: &'a [f32],
    layout: &Layout,
    elements: usize,
    label: &str,
) -> Result<&'a [f32]> {
    if !layout.is_contiguous() || layout.shape().elem_count() != elements {
        candle_core::bail!("fused CPU LayerNorm affine {label} has an incompatible layout")
    }
    let start = layout.start_offset();
    storage.get(start..start + elements).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "fused CPU LayerNorm affine {label} layout is out of bounds"
        ))
    })
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    fn bits(tensor: &Tensor) -> Vec<u32> {
        tensor
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    #[test]
    fn fused_cpu_tail_is_bit_exact_with_the_existing_operator_sequence() {
        let device = Device::Cpu;
        for (rows, features) in [(1, 1), (2, 7), (19, 120), (3, 256)] {
            let centered = Tensor::from_iter(
                (0..(rows + 1) * features).map(|index| ((index * 97 % 503) as f32 - 251.0) / 37.0),
                &device,
            )
            .unwrap()
            .reshape((rows + 1, features))
            .unwrap()
            .narrow(0, 1, rows)
            .unwrap();
            let variance = Tensor::from_iter(
                (0..rows + 1).map(|index| ((index * 29 % 113) as f32 + 1.0) / 19.0),
                &device,
            )
            .unwrap()
            .narrow(0, 1, rows)
            .unwrap()
            .reshape((rows, 1))
            .unwrap();
            let scale = Tensor::from_iter(
                (0..features).map(|index| ((index * 31 % 127) as f32 - 63.0) / 23.0),
                &device,
            )
            .unwrap();
            let bias = Tensor::from_iter(
                (0..features).map(|index| ((index * 43 % 137) as f32 - 68.0) / 29.0),
                &device,
            )
            .unwrap();
            let epsilon = f32::from_bits(0x3727_c5ac);
            let expected = variance
                .broadcast_add(&Tensor::new(epsilon, &device).unwrap())
                .and_then(|value| value.sqrt())
                .and_then(|denominator| centered.broadcast_div(&denominator))
                .and_then(|value| value.broadcast_mul(&scale))
                .and_then(|value| value.broadcast_add(&bias))
                .unwrap();
            let actual = execute(&centered, &variance, &scale, &bias, epsilon).unwrap();

            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(bits(&actual), bits(&expected));
        }
    }
}
