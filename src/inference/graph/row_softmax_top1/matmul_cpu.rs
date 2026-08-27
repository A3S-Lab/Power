use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};
use gemm::{gemm, Parallelism};
use rayon::prelude::*;

use super::cpu::project_row;

const MAX_LOGIT_TILE_ELEMENTS: usize = 1 << 20;

pub(super) fn execute(input: &Tensor, weights: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let operation = RowMatmulBiasSoftmaxTop1::new(input.layout(), weights.layout(), bias.layout())?;
    input.apply_op3_no_bwd(weights, bias, &operation)
}

#[derive(Clone)]
struct RowMatmulBiasSoftmaxTop1 {
    output_shape: Shape,
    rows: usize,
    features: usize,
    classes: usize,
    tile_rows: usize,
}

impl RowMatmulBiasSoftmaxTop1 {
    fn new(input: &Layout, weights: &Layout, bias: &Layout) -> Result<Self> {
        if !input.is_contiguous() || !weights.is_contiguous() || !bias.is_contiguous() {
            candle_core::bail!("bounded CPU classifier projection requires contiguous inputs")
        }
        let mut output_dimensions = input.shape().dims().to_vec();
        let features = output_dimensions.last().copied().unwrap_or_default();
        let (weight_features, classes) = weights.shape().dims2()?;
        if output_dimensions.len() < 2
            || features == 0
            || features != weight_features
            || classes == 0
            || classes > (1 << 24)
            || bias.shape().dims1()? != classes
        {
            candle_core::bail!("bounded CPU classifier projection received incompatible shapes")
        }
        let rows = input.shape().elem_count() / features;
        if rows == 0 {
            candle_core::bail!("bounded CPU classifier projection requires at least one row")
        }
        let last_dimension = output_dimensions.len() - 1;
        output_dimensions[last_dimension] = 3;
        let tile_rows = (MAX_LOGIT_TILE_ELEMENTS / classes).clamp(1, rows);
        Ok(Self {
            output_shape: Shape::from_dims(&output_dimensions),
            rows,
            features,
            classes,
            tile_rows,
        })
    }

    fn project(&self, input: &[f32], weights: &[f32], bias: &[f32]) -> Result<(CpuStorage, Shape)> {
        let output_elements = self
            .rows
            .checked_mul(3)
            .ok_or_else(|| candle_core::Error::Msg("classifier output size overflowed".into()))?;
        let mut output = vec![0.0_f32; output_elements];
        let threads = rayon::current_num_threads();
        let parallelism = if threads > 1 {
            Parallelism::Rayon(threads)
        } else {
            Parallelism::None
        };
        for row_start in (0..self.rows).step_by(self.tile_rows) {
            let rows = self.tile_rows.min(self.rows - row_start);
            let mut logits = vec![0.0_f32; rows * self.classes];
            unsafe {
                gemm(
                    rows,
                    self.classes,
                    self.features,
                    logits.as_mut_ptr(),
                    1,
                    self.classes as isize,
                    false,
                    input.as_ptr().add(row_start * self.features),
                    1,
                    self.features as isize,
                    weights.as_ptr(),
                    1,
                    self.classes as isize,
                    0.0_f32,
                    1.0_f32,
                    false,
                    false,
                    false,
                    parallelism,
                );
            }
            let projected = &mut output[row_start * 3..][..rows * 3];
            logits
                .par_chunks(self.classes)
                .zip(projected.par_chunks_mut(3))
                .for_each_init(
                    || vec![0.0_f32; self.classes],
                    |exponentials, (logits, output)| {
                        project_row(self.classes, exponentials, output, |index| {
                            logits[index] + bias[index]
                        });
                    },
                );
        }
        Ok((CpuStorage::F32(output), self.output_shape.clone()))
    }
}

impl CustomOp3 for RowMatmulBiasSoftmaxTop1 {
    fn name(&self) -> &'static str {
        "a3s-bounded-cpu-row-matmul-bias-softmax-top1"
    }

    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        input_layout: &Layout,
        weights: &CpuStorage,
        weights_layout: &Layout,
        bias: &CpuStorage,
        bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let input = contiguous_values(
            input.as_slice::<f32>()?,
            input_layout,
            self.rows * self.features,
            "classifier input",
        )?;
        let weights = contiguous_values(
            weights.as_slice::<f32>()?,
            weights_layout,
            self.features * self.classes,
            "classifier weights",
        )?;
        let bias = contiguous_values(
            bias.as_slice::<f32>()?,
            bias_layout,
            self.classes,
            "classifier bias",
        )?;
        self.project(input, weights, bias)
    }
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
        .ok_or_else(|| candle_core::Error::Msg(format!("{label} range overflowed")))?;
    storage
        .get(start..end)
        .ok_or_else(|| candle_core::Error::Msg(format!("{label} is out of bounds")))
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::Instant;

    use candle_core::{DType, Device};

    use super::super::{
        row_bias_softmax_top1_last_finite, row_matmul_bias_softmax_top1_last_finite,
    };
    use super::*;

    #[test]
    fn bounded_classifier_matches_explicit_graph_bits_and_offsets() {
        let device = Device::Cpu;
        for (batch, rows, features, classes) in
            [(2, 3, 5, 17), (3, 7, 120, 257), (1, 2, 120, 18_710)]
        {
            let input = Tensor::from_iter(
                (0..(batch + 1) * rows * features)
                    .map(|index| ((index * 37 % 509) as f32 - 254.0) / 113.0),
                &device,
            )
            .unwrap()
            .reshape((batch + 1, rows, features))
            .unwrap()
            .narrow(0, 1, batch)
            .unwrap();
            let weights = Tensor::from_iter(
                (0..(features + 1) * classes)
                    .map(|index| ((index * 53 % 257) as f32 - 128.0) / 97.0),
                &device,
            )
            .unwrap()
            .reshape((features + 1, classes))
            .unwrap()
            .narrow(0, 1, features)
            .unwrap();
            let bias = Tensor::from_iter(
                (0..classes + 1).map(|index| ((index * 43 % 271) as f32 - 135.0) / 97.0),
                &device,
            )
            .unwrap()
            .narrow(0, 1, classes)
            .unwrap();
            assert_ne!(input.layout().start_offset(), 0);
            assert_ne!(weights.layout().start_offset(), 0);
            assert_ne!(bias.layout().start_offset(), 0);

            let logits = input.broadcast_matmul(&weights).unwrap();
            let expected = row_bias_softmax_top1_last_finite(&logits, &bias).unwrap();
            let actual = row_matmul_bias_softmax_top1_last_finite(&input, &weights, &bias).unwrap();

            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(
                actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "input={batch}x{rows}x{features} classes={classes}"
            );
        }
    }

    #[test]
    #[ignore = "diagnostic bounded CPU classifier comparison"]
    fn compare_bounded_and_explicit_classifier() {
        let device = Device::Cpu;
        for (batch, rows, features, classes) in [
            (2, 80, 120, 18_710),
            (8, 50, 120, 18_710),
            (32, 40, 120, 18_710),
        ] {
            let input = Tensor::zeros((batch, rows, features), DType::F32, &device).unwrap();
            let weights = Tensor::zeros((features, classes), DType::F32, &device).unwrap();
            let bias = Tensor::zeros(classes, DType::F32, &device).unwrap();
            let explicit = || {
                let logits = input.broadcast_matmul(&weights).unwrap();
                row_bias_softmax_top1_last_finite(&logits, &bias).unwrap()
            };
            black_box(explicit());
            black_box(row_matmul_bias_softmax_top1_last_finite(&input, &weights, &bias).unwrap());
            let iterations = 3_u32;
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(explicit());
            }
            let explicit_elapsed = started.elapsed();
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(
                    row_matmul_bias_softmax_top1_last_finite(&input, &weights, &bias).unwrap(),
                );
            }
            let bounded_elapsed = started.elapsed();
            eprintln!(
                "CPU classifier profile: input={batch}x{rows}x{features} classes={classes} explicit_ms={:.3} bounded_ms={:.3}",
                explicit_elapsed.as_secs_f64() * 1_000.0 / f64::from(iterations),
                bounded_elapsed.as_secs_f64() * 1_000.0 / f64::from(iterations),
            );
        }
    }
}
