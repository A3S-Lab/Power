use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::ptr::NonNull;
use std::sync::Mutex;

use candle_core::{CpuStorage, CustomOp1, DType, Layout, Shape};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};
use crate::inference::InferenceLimits;

use super::super::plan::GraphPlan;
use super::super::value::GraphValue;

mod matching;

const ERROR_CAPACITY: usize = 1_024;
// Shape-specialized primitives must earn enough reuse to amortize descriptor
// construction and parameter packing. Tracking is deliberately bounded so an
// adversarial dynamic-shape stream cannot turn admission evidence into state.
const MINIMUM_OPTIMIZED_SHAPE_OBSERVATIONS: u8 = 8;
const MAXIMUM_TRACKED_SHAPES: usize = 128;

pub(super) struct FusedOutput {
    pub(super) value: GraphValue,
    pub(super) consumed_nodes: usize,
}

pub(super) struct PreparedSegment {
    input: String,
    consumed_nodes: usize,
    blocks: Vec<Block>,
    native: NativeSegment,
    shape_admission: Mutex<ShapeAdmission>,
    execution_state_budget_bytes: u64,
}

#[derive(Default)]
struct ShapeAdmission {
    clock: u64,
    observations: HashMap<[usize; 4], ShapeObservation>,
}

#[derive(Clone, Copy)]
struct ShapeObservation {
    count: u8,
    last_use: u64,
}

#[derive(Clone, Debug)]
struct Block {
    input_channels: usize,
    output_channels: usize,
    groups: usize,
    kernel: (usize, usize),
    strides: (usize, usize),
    dilations: (usize, usize),
    padding: Padding,
    activation: Activation,
    residual: Option<ResidualSource>,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
enum Padding {
    Explicit((usize, usize, usize, usize)),
    SameUpper,
}

#[derive(Clone, Copy, Debug)]
enum Activation {
    Bias,
    Relu,
    GeluErf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidualSource {
    Input,
    Block(usize),
}

#[repr(C)]
struct NativeBlock {
    input_channels: u64,
    output_channels: u64,
    groups: u64,
    kernel_height: u64,
    kernel_width: u64,
    stride_height: u64,
    stride_width: u64,
    dilation_height: u64,
    dilation_width: u64,
    padding_kind: u32,
    padding_top: u64,
    padding_left: u64,
    padding_bottom: u64,
    padding_right: u64,
    activation_kind: u32,
    residual_source: i64,
    weights: *const f32,
    weight_count: u64,
    bias: *const f32,
    bias_count: u64,
}

struct NativeSegment(NonNull<c_void>);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
enum InternalPrecision {
    F32 = 0,
    Bf16 = 1,
}

// The native segment owns immutable convolution descriptions and a bounded
// shape cache. Each cached shape reserves one protected activation context;
// packed parameters and that context are charged to the cache together.
unsafe impl Send for NativeSegment {}
unsafe impl Sync for NativeSegment {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RuntimeVersion {
    major: i32,
    minor: i32,
    patch: i32,
    cpu_runtime: u32,
}

extern "C" {
    fn a3s_power_cpu_graph_runtime_version(
        major: *mut i32,
        minor: *mut i32,
        patch: *mut i32,
        cpu_runtime: *mut u32,
    ) -> i32;
    fn a3s_power_cpu_graph_segment_create(
        blocks: *const NativeBlock,
        block_count: usize,
        cache_budget_bytes: u64,
        context_cache_budget_bytes: u64,
        precision_kind: u32,
        output: *mut *mut c_void,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn a3s_power_cpu_graph_segment_execute(
        segment: *mut c_void,
        input: *const f32,
        input_count: usize,
        input_dimensions: *const u64,
        output: *mut f32,
        output_count: usize,
        state_budget_bytes: u64,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn a3s_power_cpu_graph_segment_destroy(segment: *mut c_void);
}

pub(super) fn prepare(
    plan: &GraphPlan,
    constants: &HashMap<String, GraphValue>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
    limits: &InferenceLimits,
) -> Result<HashMap<usize, PreparedSegment>> {
    let version = runtime_version()?;
    if (version.major, version.minor, version.patch) != (3, 13, 1)
        || !matches!(version.cpu_runtime, 1 | 2)
    {
        return Err(PowerError::BackendNotAvailable(format!(
            "optimized CPU graph runtime has unsupported identity {version:?}"
        )));
    }
    matching::prepare(plan, constants, use_counts, retained_output, limits)
}

impl PreparedSegment {
    fn new(
        input: String,
        consumed_nodes: usize,
        mut blocks: Vec<Block>,
        cache_budget_bytes: u64,
        context_cache_budget_bytes: u64,
        execution_state_budget_bytes: u64,
    ) -> Result<Self> {
        let native = NativeSegment::new(&blocks, cache_budget_bytes, context_cache_budget_bytes)?;
        for block in &mut blocks {
            block.weights = Vec::new();
            block.bias = Vec::new();
        }
        Ok(Self {
            input,
            consumed_nodes,
            blocks,
            native,
            shape_admission: Mutex::new(ShapeAdmission::default()),
            execution_state_budget_bytes,
        })
    }

    pub(super) fn try_execute(
        &self,
        values: &HashMap<String, GraphValue>,
        limits: &InferenceLimits,
        cancellation: &CancellationToken,
    ) -> Result<Option<FusedOutput>> {
        let Some(GraphValue::Tensor(input)) = values.get(&self.input) else {
            return Ok(None);
        };
        if input.dtype() != DType::F32 || !input.device().is_cpu() || !input.is_contiguous() {
            return Ok(None);
        }
        if cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let shape: [usize; 4] = input.dims().try_into().map_err(|_| {
            PowerError::InferenceFailed(
                "optimized CPU graph segment requires rank-four NCHW input".to_string(),
            )
        })?;
        let admitted = self
            .shape_admission
            .lock()
            .map_err(|_| {
                PowerError::InferenceFailed(
                    "optimized CPU graph shape admission state is unavailable".to_string(),
                )
            })?
            .observe(shape);
        if !admitted {
            return Ok(None);
        }
        let output_shape = self.output_shape(input.dims(), limits)?;
        let operation = SegmentOperation {
            native: &self.native,
            input_shape: input.dims(),
            output_shape: Shape::from(output_shape),
            state_budget_bytes: self.execution_state_budget_bytes,
        };
        let output = input.apply_op1_no_bwd(&operation).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "optimized CPU graph segment execution failed: {error}"
            ))
        })?;
        if cancellation.is_cancelled() {
            return Err(cancelled());
        }
        Ok(Some(FusedOutput {
            value: GraphValue::Tensor(output),
            consumed_nodes: self.consumed_nodes,
        }))
    }

    fn output_shape(&self, input: &[usize], limits: &InferenceLimits) -> Result<Vec<usize>> {
        let [batch, channels, mut height, mut width] = *input else {
            return Err(PowerError::InferenceFailed(
                "optimized CPU graph segment requires rank-four NCHW input".to_string(),
            ));
        };
        if channels != self.blocks[0].input_channels {
            return Err(PowerError::InferenceFailed(format!(
                "optimized CPU graph segment expected {} input channels, found {channels}",
                self.blocks[0].input_channels
            )));
        }
        let original_shape = [batch, channels, height, width];
        let mut output_shapes = Vec::with_capacity(self.blocks.len());
        let mut output_channels = channels;
        for block in &self.blocks {
            let pads = match block.padding {
                Padding::Explicit(pads) => pads,
                Padding::SameUpper => {
                    let vertical = same_upper_padding(
                        height,
                        block.kernel.0,
                        block.strides.0,
                        block.dilations.0,
                    );
                    let horizontal = same_upper_padding(
                        width,
                        block.kernel.1,
                        block.strides.1,
                        block.dilations.1,
                    );
                    (vertical.0, horizontal.0, vertical.1, horizontal.1)
                }
            };
            height = output_dimension(
                height,
                block.kernel.0,
                block.strides.0,
                block.dilations.0,
                pads.0,
                pads.2,
            )?;
            width = output_dimension(
                width,
                block.kernel.1,
                block.strides.1,
                block.dilations.1,
                pads.1,
                pads.3,
            )?;
            output_channels = block.output_channels;
            limits.checked_elements(
                &[batch, output_channels, height, width],
                "optimized CPU graph segment output",
            )?;
            let output_shape = [batch, output_channels, height, width];
            if let Some(source) = block.residual {
                let residual_shape = match source {
                    ResidualSource::Input => original_shape,
                    ResidualSource::Block(index) => *output_shapes.get(index).ok_or_else(|| {
                        PowerError::InferenceFailed(
                            "optimized CPU graph residual references a future block".to_string(),
                        )
                    })?,
                };
                if output_shape != residual_shape {
                    return Err(PowerError::InferenceFailed(
                        "optimized CPU graph residual tensors have different shapes".to_string(),
                    ));
                }
            }
            output_shapes.push(output_shape);
        }
        Ok(vec![batch, output_channels, height, width])
    }
}

impl ShapeAdmission {
    fn observe(&mut self, shape: [usize; 4]) -> bool {
        self.clock = self.clock.saturating_add(1);
        if let Some(observation) = self.observations.get_mut(&shape) {
            observation.count = observation.count.saturating_add(1);
            observation.last_use = self.clock;
            return observation.count >= MINIMUM_OPTIMIZED_SHAPE_OBSERVATIONS;
        }
        if self.observations.len() >= MAXIMUM_TRACKED_SHAPES {
            if let Some(oldest) = self
                .observations
                .iter()
                .min_by_key(|(_, observation)| observation.last_use)
                .map(|(shape, _)| *shape)
            {
                self.observations.remove(&oldest);
            }
        }
        self.observations.insert(
            shape,
            ShapeObservation {
                count: 1,
                last_use: self.clock,
            },
        );
        false
    }
}

struct SegmentOperation<'a> {
    native: &'a NativeSegment,
    input_shape: &'a [usize],
    output_shape: Shape,
    state_budget_bytes: u64,
}

impl CustomOp1 for SegmentOperation<'_> {
    fn name(&self) -> &'static str {
        "a3s-persistent-f32-cpu-graph-segment"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() || layout.shape().dims() != self.input_shape {
            candle_core::bail!("optimized CPU graph segment requires contiguous NCHW input")
        }
        let values = storage.as_slice::<f32>()?;
        let elements = layout.shape().elem_count();
        let start = layout.start_offset();
        let input = values.get(start..start + elements).ok_or_else(|| {
            candle_core::Error::Msg("optimized CPU graph segment input is out of bounds".into())
        })?;
        let mut output = vec![0.0_f32; self.output_shape.elem_count()];
        self.native
            .execute(
                input,
                self.input_shape,
                &mut output,
                self.state_budget_bytes,
            )
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        Ok((CpuStorage::F32(output), self.output_shape.clone()))
    }
}

impl NativeSegment {
    fn new(
        blocks: &[Block],
        cache_budget_bytes: u64,
        context_cache_budget_bytes: u64,
    ) -> Result<Self> {
        let native_blocks = blocks
            .iter()
            .map(NativeBlock::try_from)
            .collect::<Result<Vec<_>>>()?;
        let mut output = std::ptr::null_mut();
        let mut error = [0_i8; ERROR_CAPACITY];
        // SAFETY: Every pointer covers its declared immutable slice for the
        // duration of this call. The native constructor copies all weights and
        // biases and returns an independently owned handle.
        let status = unsafe {
            a3s_power_cpu_graph_segment_create(
                native_blocks.as_ptr(),
                native_blocks.len(),
                cache_budget_bytes,
                context_cache_budget_bytes,
                configured_internal_precision() as u32,
                &mut output,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let output = NonNull::new(output);
        if status != 0 || output.is_none() {
            return Err(native_error("create optimized CPU graph segment", &error));
        }
        Ok(Self(output.unwrap()))
    }

    fn execute(
        &self,
        input: &[f32],
        input_shape: &[usize],
        output: &mut [f32],
        state_budget_bytes: u64,
    ) -> Result<()> {
        let dimensions = input_shape
            .iter()
            .copied()
            .map(|value| {
                u64::try_from(value).map_err(|_| {
                    PowerError::InferenceFailed(
                        "optimized CPU graph segment dimension exceeds u64".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut error = [0_i8; ERROR_CAPACITY];
        // SAFETY: The native handle remains live through this call. Input and
        // output point to disjoint initialized F32 slices of the declared
        // lengths, and the four input dimensions remain live through return.
        let status = unsafe {
            a3s_power_cpu_graph_segment_execute(
                self.0.as_ptr(),
                input.as_ptr(),
                input.len(),
                dimensions.as_ptr(),
                output.as_mut_ptr(),
                output.len(),
                state_budget_bytes,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(native_error("execute optimized CPU graph segment", &error));
        }
        Ok(())
    }
}

const fn configured_internal_precision() -> InternalPrecision {
    if cfg!(feature = "embedded-cpu-bf16-experimental") {
        InternalPrecision::Bf16
    } else {
        InternalPrecision::F32
    }
}

impl Drop for NativeSegment {
    fn drop(&mut self) {
        // SAFETY: This handle was returned by the matching constructor and is
        // destroyed exactly once after all Rust references have ended.
        unsafe { a3s_power_cpu_graph_segment_destroy(self.0.as_ptr()) };
    }
}

impl TryFrom<&Block> for NativeBlock {
    type Error = PowerError;

    fn try_from(block: &Block) -> Result<Self> {
        let (padding_kind, pads) = match block.padding {
            Padding::Explicit(pads) => (0, pads),
            Padding::SameUpper => (1, (0, 0, 0, 0)),
        };
        Ok(Self {
            input_channels: u64_value(block.input_channels)?,
            output_channels: u64_value(block.output_channels)?,
            groups: u64_value(block.groups)?,
            kernel_height: u64_value(block.kernel.0)?,
            kernel_width: u64_value(block.kernel.1)?,
            stride_height: u64_value(block.strides.0)?,
            stride_width: u64_value(block.strides.1)?,
            dilation_height: u64_value(block.dilations.0)?,
            dilation_width: u64_value(block.dilations.1)?,
            padding_kind,
            padding_top: u64_value(pads.0)?,
            padding_left: u64_value(pads.1)?,
            padding_bottom: u64_value(pads.2)?,
            padding_right: u64_value(pads.3)?,
            activation_kind: match block.activation {
                Activation::Bias => 0,
                Activation::Relu => 1,
                Activation::GeluErf => 2,
            },
            residual_source: match block.residual {
                None => -1,
                Some(ResidualSource::Input) => -2,
                Some(ResidualSource::Block(index)) => i64::try_from(index).map_err(|_| {
                    PowerError::InvalidFormat(
                        "optimized CPU graph residual index exceeds i64".to_string(),
                    )
                })?,
            },
            weights: block.weights.as_ptr(),
            weight_count: u64_value(block.weights.len())?,
            bias: block.bias.as_ptr(),
            bias_count: u64_value(block.bias.len())?,
        })
    }
}

fn runtime_version() -> Result<RuntimeVersion> {
    let mut version = RuntimeVersion {
        major: 0,
        minor: 0,
        patch: 0,
        cpu_runtime: 0,
    };
    // SAFETY: The function writes four scalars through valid non-null pointers.
    let status = unsafe {
        a3s_power_cpu_graph_runtime_version(
            &mut version.major,
            &mut version.minor,
            &mut version.patch,
            &mut version.cpu_runtime,
        )
    };
    if status != 0 {
        return Err(PowerError::BackendNotAvailable(
            "optimized CPU graph runtime did not report its identity".to_string(),
        ));
    }
    Ok(version)
}

fn output_dimension(
    input: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    before: usize,
    after: usize,
) -> Result<usize> {
    let effective = dilation
        .checked_mul(kernel.saturating_sub(1))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            PowerError::InferenceFailed("optimized CPU graph kernel overflowed".to_string())
        })?;
    input
        .checked_add(before)
        .and_then(|value| value.checked_add(after))
        .and_then(|value| value.checked_sub(effective))
        .map(|value| value / stride + 1)
        .ok_or_else(|| {
            PowerError::InferenceFailed("optimized CPU graph kernel exceeds its input".to_string())
        })
}

fn same_upper_padding(
    input: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> (usize, usize) {
    let output = input.div_ceil(stride);
    let effective = dilation * (kernel.saturating_sub(1)) + 1;
    let total = ((output.saturating_sub(1)) * stride + effective).saturating_sub(input);
    (total / 2, total - total / 2)
}

fn native_error(action: &str, buffer: &[c_char]) -> PowerError {
    let detail = if buffer.first().copied() == Some(0) {
        "native runtime returned no detail".to_string()
    } else {
        // SAFETY: Native functions always terminate the fixed error buffer.
        unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    };
    PowerError::InferenceFailed(format!("failed to {action}: {detail}"))
}

fn u64_value(value: usize) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| PowerError::InvalidFormat("optimized CPU graph value exceeds u64".to_string()))
}

fn cancelled() -> PowerError {
    PowerError::InferenceFailed("static graph execution was cancelled".to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;

    #[test]
    fn runtime_uses_the_pinned_onednn_release() {
        let version = runtime_version().unwrap();
        eprintln!("A3S_POWER_CPU_GRAPH_RUNTIME {version:?}");
        assert_eq!((version.major, version.minor, version.patch), (3, 13, 1));
        assert!(matches!(version.cpu_runtime, 1 | 2));
    }

    #[test]
    fn shape_specialization_requires_bounded_reuse_evidence() {
        let mut admission = ShapeAdmission::default();
        let hot = [1, 96, 12, 80];
        for _ in 1..MINIMUM_OPTIMIZED_SHAPE_OBSERVATIONS {
            assert!(!admission.observe(hot));
        }
        assert!(admission.observe(hot));
        assert!(admission.observe(hot));

        for width in 1..=MAXIMUM_TRACKED_SHAPES {
            admission.observe([1, 3, 48, width]);
        }
        assert_eq!(admission.observations.len(), MAXIMUM_TRACKED_SHAPES);
        assert!(!admission.observe(hot));
        assert_eq!(admission.observations.len(), MAXIMUM_TRACKED_SHAPES);
    }

    #[test]
    fn concurrent_same_shape_segments_preserve_output_bits() {
        let channels = 32;
        let height = 8;
        let width = 64;
        let blocks = (0..2)
            .map(|block| Block {
                input_channels: channels,
                output_channels: channels,
                groups: 1,
                kernel: (1, 1),
                strides: (1, 1),
                dilations: (1, 1),
                padding: Padding::Explicit((0, 0, 0, 0)),
                activation: if block == 0 {
                    Activation::Relu
                } else {
                    Activation::Bias
                },
                residual: None,
                weights: (0..channels * channels)
                    .map(|index| {
                        let output = index / channels;
                        let input = index % channels;
                        if output == input {
                            0.75
                        } else {
                            ((index * 29 % 257) as f32 - 128.0) / 4_096.0
                        }
                    })
                    .collect(),
                bias: (0..channels)
                    .map(|channel| (channel as f32 - 15.0) / 97.0)
                    .collect(),
            })
            .collect::<Vec<_>>();
        let budget = 16 * 1024 * 1024;
        let segment = Arc::new(NativeSegment::new(&blocks, budget, budget).unwrap());
        let input = Arc::new(
            (0..channels * height * width)
                .map(|index| ((index * 37 % 1_021) as f32 - 510.0) / 511.0)
                .collect::<Vec<_>>(),
        );
        let shape = [1, channels, height, width];
        let mut expected = vec![0.0_f32; input.len()];
        segment
            .execute(&input, &shape, &mut expected, budget)
            .unwrap();

        let workers = 4;
        let barrier = Arc::new(Barrier::new(workers));
        let handles = (0..workers)
            .map(|_| {
                let segment = Arc::clone(&segment);
                let input = Arc::clone(&input);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let mut output = vec![0.0_f32; input.len()];
                    barrier.wait();
                    segment
                        .execute(&input, &shape, &mut output, budget)
                        .unwrap();
                    output
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    }
}
