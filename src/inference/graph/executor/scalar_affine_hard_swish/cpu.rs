use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

pub(super) fn execute(
    input: &Tensor,
    pre_scale: f32,
    pre_bias: f32,
    alpha: f32,
    beta: f32,
    post_scale: f32,
    post_bias: f32,
) -> Result<Tensor> {
    input.apply_op1_no_bwd(&AffineHardSwishAffine {
        shape: input.shape().clone(),
        elements: input.elem_count(),
        pre_scale,
        pre_bias,
        alpha,
        beta,
        post_scale,
        post_bias,
    })
}

struct AffineHardSwishAffine {
    shape: Shape,
    elements: usize,
    pre_scale: f32,
    pre_bias: f32,
    alpha: f32,
    beta: f32,
    post_scale: f32,
    post_bias: f32,
}

impl CustomOp1 for AffineHardSwishAffine {
    fn name(&self) -> &'static str {
        "a3s-fused-cpu-affine-hard-swish-affine"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() || layout.shape() != &self.shape {
            candle_core::bail!("fused CPU affine-HardSwish-affine requires a contiguous input")
        }
        let start = layout.start_offset();
        let input = storage
            .as_slice::<f32>()?
            .get(start..start + self.elements)
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    "affine-HardSwish-affine input layout is out of bounds".into(),
                )
            })?;
        let mut output = vec![0.0_f32; self.elements];
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, output)| {
                let pre_scaled = input[index] * self.pre_scale + 0.0;
                let activated_input = pre_scaled * 1.0 + self.pre_bias;
                let gate_scaled = activated_input * self.alpha + 0.0;
                let gate_shifted = gate_scaled * 1.0 + self.beta;
                let lower_bounded = if gate_shifted < 0.0 {
                    0.0
                } else {
                    gate_shifted
                };
                let gate = if lower_bounded > 1.0 {
                    1.0
                } else {
                    lower_bounded
                };
                let activated = activated_input * gate;
                let post_scaled = activated * self.post_scale + 0.0;
                *output = post_scaled * 1.0 + self.post_bias;
            });
        Ok((CpuStorage::F32(output), self.shape.clone()))
    }
}
