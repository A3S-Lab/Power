use candle_core::cpu::kernels::VecOps;
use candle_core::{CpuStorage, CustomOp1, CustomOp2, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

pub(super) fn execute(input: &Tensor) -> Result<Tensor> {
    let operation = RowSoftmaxTop1::new(input.layout())?;
    input.apply_op1_no_bwd(&operation)
}

pub(super) fn execute_with_bias(input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let operation = RowSoftmaxTop1::new(input.layout())?;
    operation.validate_bias(bias.layout())?;
    input.apply_op2_no_bwd(bias, &operation)
}

#[derive(Clone)]
struct RowSoftmaxTop1 {
    output_shape: Shape,
    rows: usize,
    classes: usize,
}

impl RowSoftmaxTop1 {
    fn new(input: &Layout) -> Result<Self> {
        if !input.is_contiguous() {
            candle_core::bail!("row Softmax top-1 CPU projection requires contiguous input")
        }
        let mut dimensions = input.shape().dims().to_vec();
        let classes = dimensions.last().copied().unwrap_or_default();
        if dimensions.len() < 2 || classes == 0 || classes > (1 << 24) {
            candle_core::bail!("row Softmax top-1 CPU projection received an invalid shape")
        }
        let rows = input.shape().elem_count() / classes;
        let last_dimension = dimensions.len() - 1;
        dimensions[last_dimension] = 3;
        Ok(Self {
            output_shape: Shape::from_dims(&dimensions),
            rows,
            classes,
        })
    }

    fn validate_bias(&self, bias: &Layout) -> Result<()> {
        if !bias.is_contiguous() || bias.shape().dims1()? != self.classes {
            candle_core::bail!(
                "row Softmax top-1 CPU bias requires exact contiguous [classes] shape"
            )
        }
        Ok(())
    }

    fn project(&self, input: &[f32], bias: Option<&[f32]>) -> Result<(CpuStorage, Shape)> {
        let output_elements = self
            .rows
            .checked_mul(3)
            .ok_or_else(|| candle_core::Error::Msg("row output size overflowed".into()))?;
        let mut output = vec![0.0_f32; output_elements];

        input
            .par_chunks(self.classes)
            .zip(output.par_chunks_mut(3))
            .for_each_init(
                || vec![0.0_f32; self.classes],
                |exponentials, (input, output)| match bias {
                    Some(bias) => project_row(self.classes, exponentials, output, |index| {
                        input[index] + bias[index]
                    }),
                    None => project_row(self.classes, exponentials, output, |index| input[index]),
                },
            );

        Ok((CpuStorage::F32(output), self.output_shape.clone()))
    }
}

impl CustomOp1 for RowSoftmaxTop1 {
    fn name(&self) -> &'static str {
        "a3s-row-softmax-top1-last-finite-cpu"
    }

    fn cpu_fwd(&self, input: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let input = input.as_slice::<f32>()?;
        let start = layout.start_offset();
        let elements = self
            .rows
            .checked_mul(self.classes)
            .ok_or_else(|| candle_core::Error::Msg("row input size overflowed".into()))?;
        let input = input.get(start..start + elements).ok_or_else(|| {
            candle_core::Error::Msg("row Softmax top-1 CPU input is out of bounds".into())
        })?;
        self.project(input, None)
    }
}

impl CustomOp2 for RowSoftmaxTop1 {
    fn name(&self) -> &'static str {
        "a3s-row-bias-softmax-top1-last-finite-cpu"
    }

    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        input_layout: &Layout,
        bias: &CpuStorage,
        bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        self.validate_bias(bias_layout)?;
        let input = input.as_slice::<f32>()?;
        let input_start = input_layout.start_offset();
        let elements = self
            .rows
            .checked_mul(self.classes)
            .ok_or_else(|| candle_core::Error::Msg("row input size overflowed".into()))?;
        let input = input
            .get(input_start..input_start + elements)
            .ok_or_else(|| {
                candle_core::Error::Msg("row Softmax top-1 CPU input is out of bounds".into())
            })?;
        let bias = bias.as_slice::<f32>()?;
        let bias_start = bias_layout.start_offset();
        let bias = bias
            .get(bias_start..bias_start + self.classes)
            .ok_or_else(|| {
                candle_core::Error::Msg("row Softmax top-1 CPU bias is out of bounds".into())
            })?;
        self.project(input, Some(bias))
    }
}

pub(super) fn project_row(
    classes: usize,
    exponentials: &mut [f32],
    output: &mut [f32],
    value_at: impl Fn(usize) -> f32,
) {
    debug_assert_eq!(exponentials.len(), classes);
    debug_assert_eq!(output.len(), 3);
    let mut maximum = f32::NEG_INFINITY;
    let mut all_finite = true;
    for (index, value_slot) in exponentials.iter_mut().enumerate() {
        let value = value_at(index);
        *value_slot = value;
        all_finite &= value.is_finite();
        maximum = maximum.max(value);
    }
    if !all_finite {
        output[0] = (classes - 1) as f32;
        output[1] = 0.0;
        output[2] = 0.0;
        return;
    }

    let mut best_index = 0_usize;
    let mut best_exponential = f32::NEG_INFINITY;
    for (index, exponential) in exponentials.iter_mut().enumerate() {
        *exponential = (*exponential - maximum).exp();
        // The explicit graph selects top-1 after Softmax. Comparing the
        // rounded exponentials preserves its last-index tie behavior even
        // when distinct adjacent logits round to the same exponential.
        if *exponential >= best_exponential {
            best_exponential = *exponential;
            best_index = index;
        }
    }
    let mut sum = 0.0_f32;
    // Candle's generic last-axis Sum uses this same F32 reduction primitive.
    // Keep its SIMD lane and rounding order so the probability is bit exact.
    unsafe {
        f32::vec_reduce_sum(exponentials.as_ptr(), &mut sum, classes);
    }
    output[0] = best_index as f32;
    output[1] = best_exponential / sum;
    output[2] = 1.0;
}
