use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

pub(super) fn mul(multiplicand: &Tensor, gate: &Tensor, alpha: f32, beta: f32) -> Result<Tensor> {
    let operation = GatedHardSigmoid::new(multiplicand.layout(), gate.layout(), alpha, beta)?;
    multiplicand.apply_op2_no_bwd(gate, &operation)
}

#[derive(Clone)]
struct GatedHardSigmoid {
    shape: Shape,
    elements: usize,
    spatial: usize,
    gate_spatial: usize,
    alpha: f32,
    beta: f32,
}

impl GatedHardSigmoid {
    fn new(multiplicand: &Layout, gate: &Layout, alpha: f32, beta: f32) -> Result<Self> {
        if !multiplicand.is_contiguous() || !gate.is_contiguous() {
            candle_core::bail!("fused CPU gated HardSigmoid requires contiguous inputs")
        }
        if !alpha.is_finite() || !beta.is_finite() {
            candle_core::bail!("fused CPU gated HardSigmoid requires finite attributes")
        }
        let (batch, channels, height, width) = multiplicand.shape().dims4()?;
        let gate_dimensions = gate.shape().dims4()?;
        if gate_dimensions != (batch, channels, height, width)
            && gate_dimensions != (batch, channels, 1, 1)
        {
            candle_core::bail!(
                "fused CPU gated HardSigmoid requires equal shapes or an exact NCHW channel gate"
            )
        }
        let spatial = height.checked_mul(width).ok_or_else(|| {
            candle_core::Error::Msg("gated HardSigmoid spatial size overflowed".into())
        })?;
        let gate_spatial = gate_dimensions
            .2
            .checked_mul(gate_dimensions.3)
            .ok_or_else(|| {
                candle_core::Error::Msg("gated HardSigmoid gate size overflowed".into())
            })?;
        let elements = multiplicand.shape().elem_count();
        if batch == 0 || channels == 0 || spatial == 0 || gate_spatial == 0 || elements == 0 {
            candle_core::bail!("fused CPU gated HardSigmoid requires non-empty NCHW inputs")
        }
        Ok(Self {
            shape: multiplicand.shape().clone(),
            elements,
            spatial,
            gate_spatial,
            alpha,
            beta,
        })
    }
}

impl CustomOp2 for GatedHardSigmoid {
    fn name(&self) -> &'static str {
        "a3s-fused-cpu-gated-hard-sigmoid"
    }

    fn cpu_fwd(
        &self,
        multiplicand: &CpuStorage,
        multiplicand_layout: &Layout,
        gate: &CpuStorage,
        gate_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let multiplicand = contiguous_values(
            multiplicand.as_slice::<f32>()?,
            multiplicand_layout,
            self.elements,
            "gated HardSigmoid multiplicand",
        )?;
        let gate_elements = self.elements / self.spatial * self.gate_spatial;
        let gate = contiguous_values(
            gate.as_slice::<f32>()?,
            gate_layout,
            gate_elements,
            "gated HardSigmoid gate",
        )?;
        let mut output = vec![0.0_f32; self.elements];
        output
            .par_chunks_mut(self.spatial)
            .enumerate()
            .for_each(|(batch_channel, output)| {
                let multiplicand_start = batch_channel * self.spatial;
                let gate_start = batch_channel * self.gate_spatial;
                for (pixel, output) in output.iter_mut().enumerate() {
                    let gate_index = gate_start + pixel.min(self.gate_spatial - 1);
                    let scaled = gate[gate_index] * self.alpha + 0.0;
                    let shifted = scaled * 1.0 + self.beta;
                    let lower_bounded = if shifted < 0.0 { 0.0 } else { shifted };
                    let bounded = if lower_bounded > 1.0 {
                        1.0
                    } else {
                        lower_bounded
                    };
                    *output = multiplicand[multiplicand_start + pixel] * bounded;
                }
            });
        Ok((CpuStorage::F32(output), self.shape.clone()))
    }
}

fn contiguous_values<'a>(
    storage: &'a [f32],
    layout: &Layout,
    elements: usize,
    label: &str,
) -> Result<&'a [f32]> {
    let start = layout.start_offset();
    storage
        .get(start..start + elements)
        .ok_or_else(|| candle_core::Error::Msg(format!("{label} layout is out of bounds")))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    #[test]
    fn equal_shape_gate_matches_explicit_hard_swish() {
        let input = Tensor::from_vec(
            vec![-6.0_f32, -3.0, 0.0, 3.0, 6.0, 9.0],
            (1, 1, 2, 3),
            &Device::Cpu,
        )
        .unwrap();
        let fused = mul(&input, &input, 1.0 / 6.0, 0.5).unwrap();
        let explicit = (&input
            * ((&input * (1.0 / 6.0)).unwrap() + 0.5)
                .unwrap()
                .clamp(0.0, 1.0))
        .unwrap();
        assert_eq!(
            fused.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            explicit.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn channel_gate_broadcasts_over_exact_nchw_spatial_extent() {
        let multiplicand =
            Tensor::from_vec(vec![2.0_f32, 3.0, 4.0, 5.0], (1, 1, 2, 2), &Device::Cpu).unwrap();
        let gate = Tensor::from_vec(vec![0.0_f32], (1, 1, 1, 1), &Device::Cpu).unwrap();
        let fused = mul(&multiplicand, &gate, 0.2, 0.5).unwrap();
        assert_eq!(
            fused.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 1.5, 2.0, 2.5]
        );
    }
}
