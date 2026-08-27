use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

pub(super) fn execute(input: &Tensor, divisor: f32, offset: f32, scale: f32) -> Result<Tensor> {
    input.apply_op1_no_bwd(&GeluErf {
        elements: input.elem_count(),
        divisor,
        offset,
        scale,
    })
}

#[derive(Clone, Copy)]
struct GeluErf {
    elements: usize,
    divisor: f32,
    offset: f32,
    scale: f32,
}

impl CustomOp1 for GeluErf {
    fn name(&self) -> &'static str {
        "a3s-fused-cpu-gelu-erf"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() || layout.shape().elem_count() != self.elements {
            candle_core::bail!("fused CPU GeluErf requires one contiguous input")
        }
        let values = storage.as_slice::<f32>()?;
        let start = layout.start_offset();
        let values = values.get(start..start + self.elements).ok_or_else(|| {
            candle_core::Error::Msg("fused CPU GeluErf input layout is out of bounds".into())
        })?;
        let mut output = vec![0.0_f32; self.elements];
        output
            .par_iter_mut()
            .zip(values)
            .for_each(|(output, input)| {
                let divided = *input / self.divisor;
                let activated = candle_core::cpu::erf::erf_f32(divided);
                let shifted = activated + self.offset;
                let product = *input * shifted;
                *output = product * self.scale;
            });
        Ok((CpuStorage::F32(output), layout.shape().clone()))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    #[test]
    fn fused_cpu_gelu_preserves_the_explicit_graph_formula() {
        let device = Device::Cpu;
        let input = Tensor::new(
            &[-7.0_f32, -2.5, -1.0, -0.0, 0.0, 0.125, 1.0, 3.5, 9.0],
            &device,
        )
        .unwrap()
        .narrow(0, 1, 7)
        .unwrap();
        assert!(input.is_contiguous());
        assert_ne!(input.layout().start_offset(), 0);
        // Preserve the exact f32 initializer published by the graph rather
        // than substituting a mathematically equivalent higher-precision
        // constant that could round differently.
        let divisor = f32::from_bits(0x3fb5_04f3);
        let offset = 1.0_f32;
        let scale = 0.5_f32;
        // Build the exact graph; multiplying by a reciprocal may round at a
        // different boundary than its true division node.
        let expected = input
            .broadcast_div(&Tensor::new(divisor, &device).unwrap())
            .unwrap()
            .erf()
            .unwrap()
            .affine(1.0, offset as f64)
            .unwrap()
            .broadcast_mul(&input)
            .unwrap()
            .affine(scale as f64, 0.0)
            .unwrap();
        let actual = execute(&input, divisor, offset, scale).unwrap();

        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }
}
