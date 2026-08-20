use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::graph::GraphExecutor;
use super::super::{ExecutionPermit, TensorInput, TensorOutput};
use super::{
    GraphExecutionBoundaryMeasurement, HostAllocationCounter, TensorBatchBenchmarkConfig,
    TensorBatchBenchmarkMode, TensorBatchBenchmarkOrder, TensorBatchBenchmarkReport,
    TensorBatchBenchmarkSample,
};

struct ModeResult {
    sample: TensorBatchBenchmarkSample,
    outputs: Vec<TensorOutput>,
}

/// Measures generic individual and leading-axis batch execution in alternating
/// order and returns digest-bound, named-hardware evidence.
///
/// The graph, input tensors, and output tensors remain caller-owned and are
/// never serialized into the report. Run this in an isolated process whose
/// global allocator implements [`HostAllocationCounter`]; background activity
/// in the same process would otherwise contaminate process-wide counters.
pub fn run_tensor_batch_benchmark<C: HostAllocationCounter>(
    graph: &GraphExecutor,
    inputs: &[TensorInput],
    config: &TensorBatchBenchmarkConfig,
    allocations: &C,
    cancellation: &CancellationToken,
) -> Result<TensorBatchBenchmarkReport> {
    super::validation::validate_config(config)?;
    validate_inputs(graph, inputs)?;
    let binding = graph.benchmark_binding(&config.power_commit, &config.system)?;
    let permit = graph.runtime().begin(cancellation)?;

    for round in 0..config.warmup_rounds {
        let order = TensorBatchBenchmarkOrder::for_round(round);
        let first = execute_mode(graph, inputs, order.modes()[0], &permit, cancellation)?;
        let second = execute_mode(graph, inputs, order.modes()[1], &permit, cancellation)?;
        require_exact_outputs(&first.1, &second.1)?;
    }

    let mut samples = Vec::with_capacity(config.measured_rounds.saturating_mul(2));
    let mut canonical_output_sha256: Option<String> = None;
    for round in 0..config.measured_rounds {
        let order = TensorBatchBenchmarkOrder::for_round(round);
        let modes = order.modes();
        let first = measure_mode(
            graph,
            inputs,
            round,
            modes[0],
            allocations,
            &permit,
            cancellation,
        )?;
        let second = measure_mode(
            graph,
            inputs,
            round,
            modes[1],
            allocations,
            &permit,
            cancellation,
        )?;
        require_exact_outputs(&first.outputs, &second.outputs)?;
        if first.sample.output_sha256 != second.sample.output_sha256 {
            return Err(PowerError::IntegrityCheckFailed {
                model: "tensor batch benchmark output".to_string(),
                expected: first.sample.output_sha256,
                actual: second.sample.output_sha256,
            });
        }
        match &canonical_output_sha256 {
            Some(expected) if expected != &first.sample.output_sha256 => {
                return Err(PowerError::IntegrityCheckFailed {
                    model: "tensor batch benchmark repeated output".to_string(),
                    expected: expected.clone(),
                    actual: first.sample.output_sha256,
                });
            }
            None => canonical_output_sha256 = Some(first.sample.output_sha256.clone()),
            Some(_) => {}
        }
        samples.push(first.sample);
        samples.push(second.sample);
    }

    let mut report = TensorBatchBenchmarkReport {
        schema: TensorBatchBenchmarkReport::SCHEMA.to_string(),
        binding,
        runtime_artifact_sha256: config.runtime_artifact_sha256.clone(),
        system: config.system.clone(),
        warmup_rounds: config.warmup_rounds,
        measured_rounds: config.measured_rounds,
        item_count: inputs.len(),
        input_sequence_sha256: super::digest::input_sequence_sha256(inputs),
        output_sha256: canonical_output_sha256.ok_or_else(|| {
            PowerError::InferenceFailed(
                "tensor batch benchmark produced no measured output".to_string(),
            )
        })?,
        exact_output_parity: true,
        summaries: super::validation::summaries(&samples)?,
        samples,
        sha256: String::new(),
    };
    report.sha256 = super::digest::report_sha256(&report)?;
    report.verify()?;
    Ok(report)
}

fn validate_inputs(graph: &GraphExecutor, inputs: &[TensorInput]) -> Result<()> {
    if inputs.len() < 2 || inputs.len() > super::validation::MAX_BENCHMARK_ITEMS {
        return Err(PowerError::InvalidRequest(format!(
            "tensor batch benchmark requires between 2 and {} input items",
            super::validation::MAX_BENCHMARK_ITEMS
        )));
    }
    for input in inputs {
        input.validate(graph.runtime().limits())?;
    }
    TensorInput::stack_leading(inputs.to_vec(), graph.runtime().limits())?;
    Ok(())
}

fn measure_mode<C: HostAllocationCounter>(
    graph: &GraphExecutor,
    inputs: &[TensorInput],
    round: usize,
    mode: TensorBatchBenchmarkMode,
    allocations: &C,
    permit: &ExecutionPermit,
    cancellation: &CancellationToken,
) -> Result<ModeResult> {
    let order = TensorBatchBenchmarkOrder::for_round(round);
    // Benchmark setup allocations are intentionally outside the measured
    // interval. The interval includes batch assembly, graph execution,
    // boundary materialization, and leading-axis output partitioning.
    let prepared_inputs = inputs.to_vec();
    let started_allocations = allocations.snapshot();
    let started = Instant::now();
    let (boundary, outputs) =
        execute_prepared_mode(graph, prepared_inputs, mode, permit, cancellation)?;
    let elapsed_nanos = duration_nanos(started.elapsed());
    let host_allocations = allocations
        .snapshot()
        .checked_measurement_since(started_allocations)?;
    let output_sha256 = super::digest::output_sequence_sha256(&outputs);
    Ok(ModeResult {
        sample: TensorBatchBenchmarkSample {
            round,
            order,
            mode,
            item_count: inputs.len(),
            execution_count: match mode {
                TensorBatchBenchmarkMode::Individual => inputs.len(),
                TensorBatchBenchmarkMode::LeadingBatch => 1,
            },
            elapsed_nanos,
            host_allocations,
            boundary,
            output_sha256,
        },
        outputs,
    })
}

fn execute_mode(
    graph: &GraphExecutor,
    inputs: &[TensorInput],
    mode: TensorBatchBenchmarkMode,
    permit: &ExecutionPermit,
    cancellation: &CancellationToken,
) -> Result<(GraphExecutionBoundaryMeasurement, Vec<TensorOutput>)> {
    execute_prepared_mode(graph, inputs.to_vec(), mode, permit, cancellation)
}

fn execute_prepared_mode(
    graph: &GraphExecutor,
    inputs: Vec<TensorInput>,
    mode: TensorBatchBenchmarkMode,
    permit: &ExecutionPermit,
    cancellation: &CancellationToken,
) -> Result<(GraphExecutionBoundaryMeasurement, Vec<TensorOutput>)> {
    match mode {
        TensorBatchBenchmarkMode::Individual => {
            let mut boundary = GraphExecutionBoundaryMeasurement::default();
            let mut outputs = Vec::with_capacity(inputs.len());
            for input in inputs {
                let (output, measured) = graph.run_measured(input, permit, cancellation)?;
                boundary = boundary.checked_add(measured)?;
                outputs.push(output);
            }
            Ok((boundary, outputs))
        }
        TensorBatchBenchmarkMode::LeadingBatch => {
            let leading_partitions = inputs
                .iter()
                .map(|input| input.shape[0])
                .collect::<Vec<_>>();
            let batched = TensorInput::stack_leading(inputs, graph.runtime().limits())?;
            let (output, boundary) = graph.run_measured(batched, permit, cancellation)?;
            let outputs = output.split_leading(&leading_partitions, graph.runtime().limits())?;
            Ok((boundary, outputs))
        }
    }
}

fn require_exact_outputs(left: &[TensorOutput], right: &[TensorOutput]) -> Result<()> {
    if left == right {
        Ok(())
    } else {
        Err(PowerError::IntegrityCheckFailed {
            model: "tensor batch benchmark exact output parity".to_string(),
            expected: super::digest::output_sequence_sha256(left),
            actual: super::digest::output_sequence_sha256(right),
        })
    }
}

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
