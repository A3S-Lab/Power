use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

pub(super) fn execute(
    input: &Tensor,
    kernel: (usize, usize),
    strides: (usize, usize),
    pads: (usize, usize, usize, usize),
) -> Result<Tensor> {
    let operation = MaxPool2d::new(input.layout(), kernel, strides, pads)?;
    input.apply_op1_no_bwd(&operation)
}

#[derive(Clone)]
struct MaxPool2d {
    output_shape: Shape,
    batch: usize,
    channels: usize,
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
    kernel: (usize, usize),
    strides: (usize, usize),
    pads: (usize, usize, usize, usize),
    planes: usize,
    input_spatial: usize,
    output_spatial: usize,
    input_elements: usize,
    output_elements: usize,
    comparison_work: usize,
}

impl MaxPool2d {
    fn new(
        input: &Layout,
        kernel: (usize, usize),
        strides: (usize, usize),
        pads: (usize, usize, usize, usize),
    ) -> Result<Self> {
        if !input.is_contiguous() {
            candle_core::bail!("direct CPU MaxPool requires a contiguous input")
        }
        let (batch, channels, input_height, input_width) = input.shape().dims4()?;
        if batch == 0
            || channels == 0
            || kernel.0 == 0
            || kernel.1 == 0
            || strides.0 == 0
            || strides.1 == 0
        {
            candle_core::bail!("direct CPU MaxPool requires non-empty positive geometry")
        }
        let padded_height = input_height
            .checked_add(pads.0)
            .and_then(|height| height.checked_add(pads.2))
            .ok_or_else(|| candle_core::Error::Msg("MaxPool input height overflowed".into()))?;
        let padded_width = input_width
            .checked_add(pads.1)
            .and_then(|width| width.checked_add(pads.3))
            .ok_or_else(|| candle_core::Error::Msg("MaxPool input width overflowed".into()))?;
        let output_height = padded_height
            .checked_sub(kernel.0)
            .map(|height| height / strides.0 + 1)
            .ok_or_else(|| candle_core::Error::Msg("MaxPool kernel exceeds input height".into()))?;
        let output_width = padded_width
            .checked_sub(kernel.1)
            .map(|width| width / strides.1 + 1)
            .ok_or_else(|| candle_core::Error::Msg("MaxPool kernel exceeds input width".into()))?;
        let planes = batch
            .checked_mul(channels)
            .ok_or_else(|| candle_core::Error::Msg("MaxPool plane count overflowed".into()))?;
        let input_spatial = input_height
            .checked_mul(input_width)
            .ok_or_else(|| candle_core::Error::Msg("MaxPool input area overflowed".into()))?;
        let output_spatial = output_height
            .checked_mul(output_width)
            .ok_or_else(|| candle_core::Error::Msg("MaxPool output area overflowed".into()))?;
        let input_elements = planes
            .checked_mul(input_spatial)
            .ok_or_else(|| candle_core::Error::Msg("MaxPool input size overflowed".into()))?;
        let output_elements = planes
            .checked_mul(output_spatial)
            .ok_or_else(|| candle_core::Error::Msg("MaxPool output size overflowed".into()))?;
        let kernel_area = kernel
            .0
            .checked_mul(kernel.1)
            .ok_or_else(|| candle_core::Error::Msg("MaxPool kernel area overflowed".into()))?;
        let comparison_work = output_elements
            .checked_mul(kernel_area)
            .ok_or_else(|| candle_core::Error::Msg("MaxPool work estimate overflowed".into()))?;
        Ok(Self {
            output_shape: Shape::from_dims(&[batch, channels, output_height, output_width]),
            batch,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel,
            strides,
            pads,
            planes,
            input_spatial,
            output_spatial,
            input_elements,
            output_elements,
            comparison_work,
        })
    }

    fn source_value(
        &self,
        input: &[f32],
        input_base: usize,
        output_y: usize,
        output_x: usize,
        kernel_y: usize,
        kernel_x: usize,
    ) -> f32 {
        let padded_y = output_y * self.strides.0 + kernel_y;
        let padded_x = output_x * self.strides.1 + kernel_x;
        padded_y
            .checked_sub(self.pads.0)
            .filter(|input_y| *input_y < self.input_height)
            .zip(
                padded_x
                    .checked_sub(self.pads.1)
                    .filter(|input_x| *input_x < self.input_width),
            )
            .map_or(0.0, |(input_y, input_x)| {
                input[input_base + input_y * self.input_width + input_x]
            })
    }

    fn fill_plane(&self, input: &[f32], input_base: usize, output: &mut [f32]) {
        if self.pads == (0, 0, 0, 0) {
            self.fill_unpadded_plane(input, input_base, output);
            return;
        }
        for output_y in 0..self.output_height {
            for output_x in 0..self.output_width {
                let mut largest = self.source_value(input, input_base, output_y, output_x, 0, 0);
                for kernel_y in 0..self.kernel.0 {
                    for kernel_x in 0..self.kernel.1 {
                        let value = self.source_value(
                            input, input_base, output_y, output_x, kernel_y, kernel_x,
                        );
                        if largest < value {
                            largest = value;
                        }
                    }
                }
                output[output_y * self.output_width + output_x] = largest;
            }
        }
    }

    fn fill_unpadded_plane(&self, input: &[f32], input_base: usize, output: &mut [f32]) {
        for output_y in 0..self.output_height {
            let input_y = output_y * self.strides.0;
            for output_x in 0..self.output_width {
                let input_x = output_x * self.strides.1;
                let first = input_base + input_y * self.input_width + input_x;
                let mut largest = input[first];
                for kernel_y in 0..self.kernel.0 {
                    let row = first + kernel_y * self.input_width;
                    for kernel_x in 0..self.kernel.1 {
                        let value = input[row + kernel_x];
                        if largest < value {
                            largest = value;
                        }
                    }
                }
                output[output_y * self.output_width + output_x] = largest;
            }
        }
    }

    fn should_parallelize(&self) -> bool {
        const COMPARISONS_PER_WORKER: usize = 1_024;
        self.planes > 1
            && self.comparison_work
                >= rayon::current_num_threads().saturating_mul(COMPARISONS_PER_WORKER)
    }
}

impl CustomOp1 for MaxPool2d {
    fn name(&self) -> &'static str {
        "a3s-direct-cpu-max-pool-2d"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous()
            || layout.shape().dims4()?
                != (
                    self.batch,
                    self.channels,
                    self.input_height,
                    self.input_width,
                )
        {
            candle_core::bail!("direct CPU MaxPool input layout changed after validation")
        }
        let start = layout.start_offset();
        let end = start
            .checked_add(self.input_elements)
            .ok_or_else(|| candle_core::Error::Msg("MaxPool input range overflowed".into()))?;
        let input = storage
            .as_slice::<f32>()?
            .get(start..end)
            .ok_or_else(|| candle_core::Error::Msg("MaxPool input is out of bounds".into()))?;
        let mut output = vec![0.0_f32; self.output_elements];
        if self.should_parallelize() {
            output
                .par_chunks_mut(self.output_spatial)
                .enumerate()
                .for_each(|(plane, output)| {
                    self.fill_plane(input, plane * self.input_spatial, output);
                });
        } else {
            output
                .chunks_mut(self.output_spatial)
                .enumerate()
                .for_each(|(plane, output)| {
                    self.fill_plane(input, plane * self.input_spatial, output);
                });
        }
        Ok((CpuStorage::F32(output), self.output_shape.clone()))
    }
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::Instant;

    use candle_core::Device;

    use super::*;

    #[test]
    fn direct_max_pool_matches_materialized_padding_bits() {
        let device = Device::Cpu;
        for (batch, channels, height, width, kernel, strides, pads) in [
            (2, 3, 7, 11, (2, 2), (1, 1), (0, 0, 1, 1)),
            (3, 5, 9, 13, (3, 2), (2, 1), (1, 2, 0, 1)),
            (2, 4, 8, 10, (1, 3), (1, 2), (0, 1, 0, 1)),
        ] {
            let input = Tensor::from_iter(
                (0..(batch + 1) * channels * height * width)
                    .map(|value| ((value * 17 % 251) as f32 - 180.0) / 61.0),
                &device,
            )
            .unwrap()
            .reshape((batch + 1, channels, height, width))
            .unwrap()
            .narrow(0, 1, batch)
            .unwrap();
            assert!(input.is_contiguous());
            assert_ne!(input.layout().start_offset(), 0);
            let expected = input
                .pad_with_zeros(2, pads.0, pads.2)
                .unwrap()
                .pad_with_zeros(3, pads.1, pads.3)
                .unwrap()
                .max_pool2d_with_stride(kernel, strides)
                .unwrap();
            let actual = execute(&input, kernel, strides, pads).unwrap();

            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(
                actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "input={batch}x{channels}x{height}x{width} kernel={kernel:?} strides={strides:?} pads={pads:?}"
            );
        }
    }

    #[test]
    #[ignore = "diagnostic CPU MaxPool kernel comparison"]
    fn compare_direct_and_materialized_max_pool() {
        let device = Device::Cpu;
        for (batch, channels, height, width, kernel, strides, pads) in [
            (1, 16, 16, 64, (2, 2), (2, 2), (0, 0, 0, 0)),
            (2, 32, 48, 320, (2, 2), (2, 2), (0, 0, 0, 0)),
            (8, 64, 24, 160, (2, 2), (2, 2), (0, 0, 0, 0)),
            (24, 128, 12, 80, (2, 2), (2, 2), (0, 0, 0, 0)),
            (8, 96, 12, 130, (3, 3), (2, 2), (1, 1, 1, 1)),
            (2, 8, 7, 11, (3, 2), (2, 1), (1, 2, 0, 1)),
        ] {
            let input = Tensor::zeros(
                (batch, channels, height, width),
                candle_core::DType::F32,
                &device,
            )
            .unwrap();
            let materialized = || {
                input
                    .pad_with_zeros(2, pads.0, pads.2)?
                    .pad_with_zeros(3, pads.1, pads.3)?
                    .max_pool2d_with_stride(kernel, strides)
            };
            black_box(materialized().unwrap());
            black_box(execute(&input, kernel, strides, pads).unwrap());
            let iterations = 10_u32;
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(materialized().unwrap());
            }
            let padded = started.elapsed();
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(execute(&input, kernel, strides, pads).unwrap());
            }
            let direct = started.elapsed();
            eprintln!(
                "CPU MaxPool profile: batch={batch} channels={channels} input={height}x{width} kernel={kernel:?} strides={strides:?} pads={pads:?} materialized_ms={:.3} direct_ms={:.3}",
                padded.as_secs_f64() * 1_000.0 / f64::from(iterations),
                direct.as_secs_f64() * 1_000.0 / f64::from(iterations),
            );
        }
    }
}
