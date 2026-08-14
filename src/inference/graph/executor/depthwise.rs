use candle_core::{IndexOp, Result, Tensor};

/// Executes a multiplier-one NCHW depthwise convolution without lowering it
/// into one independent graph call per channel.
///
/// Candle's generic grouped path splits `groups > 1` into scalar-channel
/// convolutions. A shift/multiply accumulation keeps the same tensor semantics
/// while issuing a bounded number of device-wide operations based on kernel
/// area instead of channel count. Padding is applied by the graph executor;
/// strided tensor views ensure that multiplication and accumulation materialize
/// only final output elements for strided convolutions.
pub(super) fn conv2d(
    input: &Tensor,
    kernel: &Tensor,
    strides: (usize, usize),
    dilation: usize,
) -> Result<Tensor> {
    let (_, channels, input_height, input_width) = input.dims4()?;
    let (output_channels, kernel_channels, kernel_height, kernel_width) = kernel.dims4()?;
    if channels == 0
        || output_channels != channels
        || kernel_channels != 1
        || kernel_height == 0
        || kernel_width == 0
        || strides.0 == 0
        || strides.1 == 0
        || dilation == 0
    {
        candle_core::bail!("depthwise convolution requires one non-empty kernel per input channel")
    }
    let effective_height = dilation
        .checked_mul(kernel_height - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| candle_core::Error::Msg("depthwise kernel height overflowed".into()))?;
    let effective_width = dilation
        .checked_mul(kernel_width - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| candle_core::Error::Msg("depthwise kernel width overflowed".into()))?;
    let output_height = input_height
        .checked_sub(effective_height)
        .map(|remaining| remaining / strides.0 + 1)
        .ok_or_else(|| candle_core::Error::Msg("depthwise kernel exceeds input height".into()))?;
    let output_width = input_width
        .checked_sub(effective_width)
        .map(|remaining| remaining / strides.1 + 1)
        .ok_or_else(|| candle_core::Error::Msg("depthwise kernel exceeds input width".into()))?;
    let mut output = None;
    for kernel_y in 0..kernel_height {
        for kernel_x in 0..kernel_width {
            let source = sampled_axis(input, 2, kernel_y * dilation, output_height, strides.0)?;
            let source = sampled_axis(&source, 3, kernel_x * dilation, output_width, strides.1)?;
            let weight = kernel
                .i((.., 0, kernel_y, kernel_x))?
                .unsqueeze(0)?
                .unsqueeze(2)?
                .unsqueeze(3)?;
            let term = source.broadcast_mul(&weight)?;
            output = Some(match output {
                Some(accumulator) => (&accumulator + &term)?,
                None => term,
            });
        }
    }
    output.ok_or_else(|| candle_core::Error::Msg("depthwise convolution produced no terms".into()))
}

fn sampled_axis(
    input: &Tensor,
    axis: usize,
    start: usize,
    output_length: usize,
    stride: usize,
) -> Result<Tensor> {
    let span = output_length
        .checked_sub(1)
        .and_then(|length| length.checked_mul(stride))
        .and_then(|length| length.checked_add(1))
        .ok_or_else(|| candle_core::Error::Msg("depthwise sample span overflowed".into()))?;
    let narrowed = input.narrow(axis, start, span)?;
    if stride == 1 {
        Ok(narrowed)
    } else {
        let appended_axis = narrowed.rank();
        narrowed.unfold(axis, 1, stride)?.squeeze(appended_axis)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::conv2d;

    #[test]
    fn shift_accumulation_matches_grouped_depthwise_convolution() {
        let device = Device::Cpu;
        let input = Tensor::from_iter(
            (0..2 * 3 * 5 * 7).map(|value| (value as f32 - 40.0) / 17.0),
            &device,
        )
        .unwrap()
        .reshape((2, 3, 5, 7))
        .unwrap()
        .pad_with_zeros(2, 1, 1)
        .unwrap()
        .pad_with_zeros(3, 1, 1)
        .unwrap();
        let kernel = Tensor::from_iter(
            (0..3 * 3 * 3).map(|value| (value as f32 - 12.0) / 13.0),
            &device,
        )
        .unwrap()
        .reshape((3, 1, 3, 3))
        .unwrap();

        let expected = input.conv2d(&kernel, 0, 1, 1, 3).unwrap();
        let actual = conv2d(&input, &kernel, (1, 1), 1).unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(actual.len(), expected.len());
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= 0.000_01));
    }

    #[test]
    fn shift_accumulation_preserves_dilated_shape_and_values() {
        let device = Device::Cpu;
        let input = Tensor::from_iter((0..2 * 7 * 9).map(|value| value as f32 / 11.0), &device)
            .unwrap()
            .reshape((1, 2, 7, 9))
            .unwrap();
        let kernel = Tensor::from_iter(
            (0..2 * 2 * 3).map(|value| (value as f32 + 1.0) / 7.0),
            &device,
        )
        .unwrap()
        .reshape((2, 1, 2, 3))
        .unwrap();

        let expected = input.conv2d(&kernel, 0, 1, 2, 2).unwrap();
        let actual = conv2d(&input, &kernel, (1, 1), 2).unwrap();

        assert_eq!(actual.dims(), expected.dims());
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= 0.000_01));
    }

    #[test]
    fn shift_accumulation_applies_stride_before_materialization() {
        let device = Device::Cpu;
        let input = Tensor::from_iter((0..2 * 8 * 10).map(|value| value as f32 / 19.0), &device)
            .unwrap()
            .reshape((1, 2, 8, 10))
            .unwrap();
        let kernel = Tensor::from_iter(
            (0..2 * 3 * 3).map(|value| (value as f32 - 4.0) / 9.0),
            &device,
        )
        .unwrap()
        .reshape((2, 1, 3, 3))
        .unwrap();

        let expected = input.conv2d(&kernel, 0, 2, 1, 2).unwrap();
        let actual = conv2d(&input, &kernel, (2, 2), 1).unwrap();

        assert_eq!(actual.dims(), expected.dims());
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= 0.000_01));
    }
}
