//! Environment-driven numerical conformance probe for reviewed graph bundles.
//!
//! The probe is intentionally model-neutral. It admits an explicit graph
//! bundle plus reference input/output tensors and compares the Power result
//! with a bounded floating-point error contract.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use a3s_power::inference::graph::{
    row_matmul_bias_softmax_top1_last_finite, GraphExecutor, GraphIdentity, GraphPlan,
};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, TensorInput, TensorOutput, WeightStore,
};
use rayon::prelude::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

const ROOT_ENV: &str = "A3S_POWER_GRAPH_PROBE_ROOT";
const GRAPH_ENV: &str = "A3S_POWER_GRAPH_PROBE_GRAPH";
const INPUT_ENV: &str = "A3S_POWER_GRAPH_PROBE_INPUT_F32LE";
const INPUT_SHAPE_ENV: &str = "A3S_POWER_GRAPH_PROBE_INPUT_SHAPE";
const EXPECTED_ENV: &str = "A3S_POWER_GRAPH_PROBE_EXPECTED_F32LE";
const EXPECTED_SHAPE_ENV: &str = "A3S_POWER_GRAPH_PROBE_EXPECTED_SHAPE";
const BENCH_SHAPE_ENV: &str = "A3S_POWER_GRAPH_PROBE_BENCH_SHAPE";
const BENCH_REPEATS_ENV: &str = "A3S_POWER_GRAPH_PROBE_BENCH_REPEATS";
const BENCH_PARALLEL_JOBS_ENV: &str = "A3S_POWER_GRAPH_PROBE_PARALLEL_JOBS";
#[cfg(feature = "embedded-cuda")]
const PREFIX_BATCH_ENV: &str = "A3S_POWER_GRAPH_PROBE_PREFIX_BATCH";
const CAPTURE_INPUT_ENV: &str = "A3S_POWER_GRAPH_PROBE_CAPTURE_INPUT_F32LE";
const CAPTURE_OUTPUT_ENV: &str = "A3S_POWER_GRAPH_PROBE_CAPTURE_OUTPUT_F32LE";
const CAPTURE_NODE_COUNT_ENV: &str = "A3S_POWER_GRAPH_PROBE_CAPTURE_NODE_COUNT";

// The contract is expressed independently of any model or fixture. The
// relative term covers normal f32 accumulation error while the absolute term
// keeps values near zero bounded.
const ABSOLUTE_TOLERANCE: f32 = 2.0e-5;
const RELATIVE_TOLERANCE: f32 = 2.0e-4;

#[test]
#[ignore = "requires an explicit reviewed graph bundle and reference tensors"]
fn external_graph_matches_reference_tensor() {
    let root = canonical_directory(required_path(ROOT_ENV));
    let graph_source = read_graph_source(&root);
    let identity = identity_from_graph(&graph_source);
    let input_shape = required_shape(INPUT_SHAPE_ENV);
    let expected_shape = required_shape(EXPECTED_SHAPE_ENV);
    let input = read_f32_tensor(&required_path(INPUT_ENV), &input_shape);
    let expected = read_f32_tensor(&required_path(EXPECTED_ENV), &expected_shape);

    let limits = InferenceLimits::default();
    let weights = Arc::new(WeightStore::open(&root, &limits).unwrap());
    eprintln!(
        "A3S_POWER_GRAPH_BUNDLE model_collection_sha256={} bytes={}",
        weights.sha256(),
        weights.bytes(),
    );
    let plan = GraphPlan::parse(&graph_source, &identity, &weights, &limits).unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, weights, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let output = graph
        .run(
            TensorInput::new(input_shape, input, &limits).unwrap(),
            &permit,
            &cancellation,
        )
        .unwrap();

    assert_eq!(output.shape, expected_shape);
    assert_eq!(output.values.len(), expected.len());
    let mut maximum_absolute = 0.0_f32;
    let mut absolute_sum = 0.0_f64;
    let mut first_out_of_tolerance = None;
    for (index, (&actual, &reference)) in output.values.iter().zip(&expected).enumerate() {
        assert!(
            actual.is_finite() && reference.is_finite(),
            "non-finite graph value at flat index {index}: actual={actual:?}, reference={reference:?}"
        );
        let absolute = (actual - reference).abs();
        maximum_absolute = maximum_absolute.max(absolute);
        absolute_sum += f64::from(absolute);
        let tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * reference.abs();
        if absolute > tolerance && first_out_of_tolerance.is_none() {
            first_out_of_tolerance = Some((index, actual, reference, absolute, tolerance));
        }
    }
    eprintln!(
        "A3S_POWER_GRAPH_PARITY elements={} max_abs={maximum_absolute:.9e} mean_abs={:.9e}",
        expected.len(),
        absolute_sum / expected.len() as f64,
    );
    assert!(
        first_out_of_tolerance.is_none(),
        "graph value first exceeds tolerance at {:?}",
        first_out_of_tolerance.unwrap()
    );
}

#[test]
#[ignore = "requires an explicit reviewed graph bundle with a terminal classifier"]
fn external_graph_reports_projected_cpu_throughput() {
    let root = canonical_directory(required_path(ROOT_ENV));
    let graph_source = read_graph_source(&root);
    let identity = identity_from_graph(&graph_source);
    let input_shape = required_shape(BENCH_SHAPE_ENV);
    let input_elements = input_shape
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .unwrap();
    let batch = input_shape[0];
    let repeats = std::env::var(BENCH_REPEATS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=20).contains(value))
        .unwrap_or(3);
    let limits = InferenceLimits::default();
    let weights = Arc::new(WeightStore::open(&root, &limits).unwrap());
    let plan = GraphPlan::parse(&graph_source, &identity, &weights, &limits).unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, weights, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let execute = || {
        graph
            .run_with_terminal_matmul_bias_softmax_projection(
                TensorInput::new(input_shape.clone(), vec![0.0; input_elements], &limits).unwrap(),
                &permit,
                &cancellation,
                row_matmul_bias_softmax_top1_last_finite,
            )
            .unwrap()
    };

    let reference = execute();
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = std::time::Instant::now();
        let output = execute();
        samples.push(started.elapsed());
        assert_projected_task_parity(&reference, &output);
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    eprintln!(
        "A3S_POWER_GRAPH_PROJECTED_CPU_BENCH input_shape={input_shape:?} output_shape={:?} repeats={repeats} median_ms={:.3} items_per_second={:.3}",
        reference.shape,
        median.as_secs_f64() * 1_000.0,
        batch as f64 / median.as_secs_f64(),
    );
}

#[cfg(feature = "embedded-cuda")]
#[test]
#[ignore = "requires an explicit reviewed graph bundle and CUDA device"]
fn external_graph_reports_projected_cuda_profile() {
    let root = canonical_directory(required_path(ROOT_ENV));
    let graph_source = read_graph_source(&root);
    let identity = identity_from_graph(&graph_source);
    let input_shape = required_shape(BENCH_SHAPE_ENV);
    let input_elements = input_shape
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .unwrap();

    let limits = InferenceLimits::default();
    let weights = Arc::new(WeightStore::open(&root, &limits).unwrap());
    let plan = GraphPlan::parse(&graph_source, &identity, &weights, &limits).unwrap();
    let runtime =
        EmbeddedRuntime::new(DevicePreference::Cuda { ordinal: 0 }, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, weights, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let execute = || {
        graph
            .run_with_terminal_matmul_bias_softmax_projection(
                TensorInput::new(input_shape.clone(), vec![0.0; input_elements], &limits).unwrap(),
                &permit,
                &cancellation,
                row_matmul_bias_softmax_top1_last_finite,
            )
            .unwrap()
    };

    let reference = execute();
    let started = std::time::Instant::now();
    let output = execute();
    let elapsed = started.elapsed();
    assert_projected_task_parity(&reference, &output);
    eprintln!(
        "A3S_POWER_GRAPH_PROJECTED_CUDA_PROFILE input_shape={input_shape:?} output_shape={:?} elapsed_ms={:.3}",
        output.shape,
        elapsed.as_secs_f64() * 1_000.0,
    );
}

#[cfg(feature = "embedded-cuda")]
#[test]
#[ignore = "requires an explicit reviewed graph bundle and CUDA device"]
fn external_graph_reports_cuda_throughput() {
    let root = canonical_directory(required_path(ROOT_ENV));
    let graph_source = read_graph_source(&root);
    let identity = identity_from_graph(&graph_source);
    let input_shape = required_shape(BENCH_SHAPE_ENV);
    let input_elements = input_shape
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .unwrap();
    let batch = input_shape[0];
    let repeats = std::env::var(BENCH_REPEATS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=20).contains(value))
        .unwrap_or(3);

    let limits = InferenceLimits::default();
    let weights = Arc::new(WeightStore::open(&root, &limits).unwrap());
    let plan = GraphPlan::parse(&graph_source, &identity, &weights, &limits).unwrap();
    let runtime =
        EmbeddedRuntime::new(DevicePreference::Cuda { ordinal: 0 }, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, weights, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let execute = || {
        graph
            .run(
                TensorInput::new(input_shape.clone(), vec![0.0; input_elements], &limits).unwrap(),
                &permit,
                &cancellation,
            )
            .unwrap()
    };

    let reference = execute();
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = std::time::Instant::now();
        let output = execute();
        samples.push(started.elapsed());
        assert_eq!(output.shape, reference.shape);
        assert_eq!(output.values.len(), reference.values.len());
        assert!(output
            .values
            .iter()
            .zip(&reference.values)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    eprintln!(
        "A3S_POWER_GRAPH_CUDA_BENCH input_shape={input_shape:?} output_shape={:?} repeats={repeats} median_ms={:.3} items_per_second={:.3}",
        reference.shape,
        median.as_secs_f64() * 1_000.0,
        batch as f64 / median.as_secs_f64(),
    );
}

#[cfg(feature = "embedded-cuda")]
#[test]
#[ignore = "requires an explicit reviewed graph bundle and CUDA device"]
fn external_graph_reports_cuda_batch_prefix_parity() {
    let root = canonical_directory(required_path(ROOT_ENV));
    let graph_source = capture_graph_source(&graph_path(&root));
    let identity = identity_from_graph(&graph_source);
    let large_shape = required_shape(BENCH_SHAPE_ENV);
    let prefix_batch = std::env::var(PREFIX_BATCH_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1);
    assert!(
        large_shape[0] > prefix_batch,
        "the compared batch must be larger than the prefix batch"
    );
    let large_elements = large_shape.iter().product::<usize>();
    assert_eq!(large_elements % large_shape[0], 0);
    let input_elements_per_item = large_elements / large_shape[0];
    let large_input = deterministic_input(large_elements);

    let limits = InferenceLimits::default();
    let weights = Arc::new(WeightStore::open(&root, &limits).unwrap());
    let plan = GraphPlan::parse(&graph_source, &identity, &weights, &limits).unwrap();
    let runtime =
        EmbeddedRuntime::new(DevicePreference::Cuda { ordinal: 0 }, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, weights, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let execute = |shape: Vec<usize>, values: Vec<f32>| {
        graph
            .run(
                TensorInput::new(shape, values, &limits).unwrap(),
                &permit,
                &cancellation,
            )
            .unwrap()
    };

    let large = execute(large_shape.clone(), large_input.clone());
    assert_eq!(large.shape[0], large_shape[0]);
    assert_eq!(large.values.len() % large_shape[0], 0);
    let output_values_per_item = large.values.len() / large_shape[0];

    let mut different_bits = 0_usize;
    let mut maximum_absolute = 0.0_f32;
    for batch_offset in (0..large_shape[0]).step_by(prefix_batch) {
        let batch_items = (large_shape[0] - batch_offset).min(prefix_batch);
        let mut partition_shape = large_shape.clone();
        partition_shape[0] = batch_items;
        let input_start = batch_offset * input_elements_per_item;
        let input_end = input_start + batch_items * input_elements_per_item;
        let partition = execute(
            partition_shape,
            large_input[input_start..input_end].to_vec(),
        );
        assert_eq!(&partition.shape[1..], &large.shape[1..]);
        assert_eq!(partition.shape[0], batch_items);
        let output_start = batch_offset * output_values_per_item;
        let output_end = output_start + partition.values.len();
        for (&expected, &actual) in partition
            .values
            .iter()
            .zip(&large.values[output_start..output_end])
        {
            if expected.to_bits() != actual.to_bits() {
                different_bits += 1;
                maximum_absolute = maximum_absolute.max((expected - actual).abs());
            }
        }
    }
    let compared_values = large.values.len();
    eprintln!(
        "A3S_POWER_GRAPH_CUDA_BATCH_PREFIX_PARITY partition_batch={prefix_batch} large_shape={large_shape:?} output_shape={:?} compared_values={compared_values} different_bits={different_bits} max_abs={maximum_absolute:.9e}",
        large.shape,
    );
}

#[test]
#[ignore = "requires an explicit reviewed graph bundle with a terminal classifier"]
fn external_graph_reports_parallel_projected_cpu_throughput() {
    let root = canonical_directory(required_path(ROOT_ENV));
    let graph_source = read_graph_source(&root);
    let identity = identity_from_graph(&graph_source);
    let input_shape = required_shape(BENCH_SHAPE_ENV);
    let input_elements = input_shape
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .unwrap();
    let batch = input_shape[0];
    let repeats = std::env::var(BENCH_REPEATS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=20).contains(value))
        .unwrap_or(3);
    let parallel_jobs = std::env::var(BENCH_PARALLEL_JOBS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=64).contains(value))
        .unwrap_or_else(rayon::current_num_threads);

    let limits = InferenceLimits::default();
    let weights = Arc::new(WeightStore::open(&root, &limits).unwrap());
    let plan = GraphPlan::parse(&graph_source, &identity, &weights, &limits).unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, weights, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let inputs = || {
        (0..parallel_jobs)
            .map(|_| {
                TensorInput::new(input_shape.clone(), vec![0.0; input_elements], &limits).unwrap()
            })
            .collect::<Vec<_>>()
    };
    let execute = |inputs: Vec<TensorInput>| {
        inputs
            .into_par_iter()
            .map(|input| {
                graph
                    .run_with_terminal_matmul_bias_softmax_projection(
                        input,
                        &permit,
                        &cancellation,
                        row_matmul_bias_softmax_top1_last_finite,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };

    let reference = execute(inputs());
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let prepared = inputs();
        let started = std::time::Instant::now();
        let outputs = execute(prepared);
        samples.push(started.elapsed());
        for (expected, actual) in reference.iter().zip(&outputs) {
            assert_projected_task_parity(expected, actual);
        }
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    eprintln!(
        "A3S_POWER_GRAPH_PARALLEL_PROJECTED_CPU_BENCH input_shape={input_shape:?} output_shape={:?} parallel_jobs={parallel_jobs} repeats={repeats} median_ms={:.3} items_per_second={:.3}",
        reference[0].shape,
        median.as_secs_f64() * 1_000.0,
        (parallel_jobs * batch) as f64 / median.as_secs_f64(),
    );
}

fn assert_projected_task_parity(expected: &TensorOutput, actual: &TensorOutput) {
    assert_eq!(actual.shape, expected.shape);
    assert_eq!(actual.values.len(), expected.values.len());
    assert_eq!(actual.values.len() % 3, 0);
    for (row, (expected, actual)) in expected
        .values
        .chunks_exact(3)
        .zip(actual.values.chunks_exact(3))
        .enumerate()
    {
        assert_eq!(actual[0], expected[0], "top-1 class changed at row {row}");
        assert_eq!(actual[2], expected[2], "finite flag changed at row {row}");
        assert!(
            actual[1].is_finite() && (0.0..=1.0).contains(&actual[1]),
            "top-1 probability is invalid at row {row}: {}",
            actual[1]
        );
    }
}

#[test]
#[ignore = "requires an explicit reviewed graph bundle"]
fn external_graph_reports_cpu_throughput() {
    let root = canonical_directory(required_path(ROOT_ENV));
    let graph_source = capture_graph_source(&graph_path(&root));
    let identity = identity_from_graph(&graph_source);
    let input_shape = required_shape(BENCH_SHAPE_ENV);
    let input_elements = input_shape
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .unwrap();
    let batch = input_shape[0];
    let repeats = std::env::var(BENCH_REPEATS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=20).contains(value))
        .unwrap_or(3);

    let limits = InferenceLimits::default();
    let weights = Arc::new(WeightStore::open(&root, &limits).unwrap());
    let plan = GraphPlan::parse(&graph_source, &identity, &weights, &limits).unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, weights, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let execute = || {
        graph
            .run(
                TensorInput::new(input_shape.clone(), vec![0.0; input_elements], &limits).unwrap(),
                &permit,
                &cancellation,
            )
            .unwrap()
    };

    let reference = execute();
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = std::time::Instant::now();
        let output = execute();
        samples.push(started.elapsed());
        assert_eq!(output.shape, reference.shape);
        assert_eq!(output.values, reference.values);
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    eprintln!(
        "A3S_POWER_GRAPH_CPU_BENCH input_shape={input_shape:?} output_shape={:?} repeats={repeats} median_ms={:.3} items_per_second={:.3}",
        reference.shape,
        median.as_secs_f64() * 1_000.0,
        batch as f64 / median.as_secs_f64(),
    );
}

#[test]
#[ignore = "requires an explicit reviewed graph bundle and new capture paths"]
fn external_graph_captures_deterministic_reference_tensors() {
    let root = canonical_directory(required_path(ROOT_ENV));
    let graph_source = capture_graph_source(&graph_path(&root));
    let identity = identity_from_graph(&graph_source);
    let input_shape = required_shape(BENCH_SHAPE_ENV);
    let input_elements = input_shape
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .unwrap();
    let input = deterministic_input(input_elements);

    let limits = InferenceLimits::default();
    let weights = Arc::new(WeightStore::open(&root, &limits).unwrap());
    let plan = GraphPlan::parse(&graph_source, &identity, &weights, &limits).unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, weights, runtime.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let output = graph
        .run(
            TensorInput::new(input_shape.clone(), input.clone(), &limits).unwrap(),
            &permit,
            &cancellation,
        )
        .unwrap();

    let input_bytes = f32_le_bytes(&input);
    let output_bytes = f32_le_bytes(&output.values);
    write_new_file(&required_path(CAPTURE_INPUT_ENV), &input_bytes);
    write_new_file(&required_path(CAPTURE_OUTPUT_ENV), &output_bytes);
    eprintln!(
        "A3S_POWER_GRAPH_CAPTURE input_shape={input_shape:?} output_shape={:?} input_sha256={} output_sha256={}",
        output.shape,
        hex::encode(Sha256::digest(&input_bytes)),
        hex::encode(Sha256::digest(&output_bytes)),
    );
}

fn capture_graph_source(path: &Path) -> String {
    let source = std::fs::read_to_string(path).unwrap();
    let Some(node_count) = std::env::var(CAPTURE_NODE_COUNT_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        return source;
    };
    let mut document: Value = serde_json::from_str(&source).unwrap();
    let nodes = document
        .get_mut("nodes")
        .and_then(Value::as_array_mut)
        .unwrap();
    assert!((1..=nodes.len()).contains(&node_count));
    nodes.truncate(node_count);
    let output_name = nodes
        .last()
        .and_then(|node| node.get("outputs"))
        .and_then(Value::as_array)
        .and_then(|outputs| outputs.first())
        .and_then(Value::as_str)
        .unwrap()
        .to_string();
    document["outputs"][0]["name"] = Value::String(output_name);
    serde_json::to_string(&document).unwrap()
}

fn deterministic_input(elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| ((index * 37 + 17) % 1_024) as f32 / 511.5 - 1.0)
        .collect()
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn write_new_file(path: &Path, bytes: &[u8]) {
    assert!(path.is_absolute());
    let parent = path.parent().unwrap();
    assert!(parent.is_dir());
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    std::io::Write::write_all(&mut file, bytes).unwrap();
    file.sync_all().unwrap();
}

fn identity_from_graph(source: &str) -> GraphIdentity {
    let document: Value = serde_json::from_str(source).unwrap();
    let source_identity = document.get("source").and_then(Value::as_object).unwrap();
    GraphIdentity::new(
        required_text(&document, "family"),
        required_text(&document, "role"),
        required_object_text(source_identity, "format"),
        required_object_text(source_identity, "sha256"),
        source_identity
            .get("opset")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap(),
    )
}

fn required_text<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap()
}

fn required_object_text<'a>(value: &'a serde_json::Map<String, Value>, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap()
}

fn required_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must be set"))
}

fn graph_path(root: &Path) -> PathBuf {
    std::env::var_os(GRAPH_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("graph.json"))
}

fn read_graph_source(root: &Path) -> String {
    std::fs::read_to_string(graph_path(root)).unwrap()
}

fn canonical_directory(path: PathBuf) -> PathBuf {
    let path = std::fs::canonicalize(path).unwrap();
    assert!(path.is_dir());
    path
}

fn required_shape(name: &str) -> Vec<usize> {
    let source = std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"));
    let shape: Vec<usize> = serde_json::from_str(&source).unwrap();
    assert!(!shape.is_empty() && shape.iter().all(|dimension| *dimension > 0));
    shape
}

fn read_f32_tensor(path: &Path, shape: &[usize]) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let elements = shape
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .unwrap();
    assert_eq!(bytes.len(), elements.checked_mul(4).unwrap());
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect()
}
