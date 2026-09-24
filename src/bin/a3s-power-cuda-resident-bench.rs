//! End-to-end CUDA device-resident graph chain harness.
//!
//! Contract (docs/perf-first-principles.md, docs/device-resident-graphs.md):
//! - resident N-graph chain must byte-match the owned host round-trip path
//! - resident path must report exactly one H→D and one D→H boundary copy
//! - owned path must report N H→D and N D→H copies
//! - latency is measured on ≥1M-element tensors (refuse unit-shape overfitting)

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use a3s_power::error::{PowerError, Result};
use a3s_power::inference::graph::{GraphExecutor, GraphIdentity, GraphPlan};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, GraphExecutionBoundaryMeasurement, InferenceLimits,
    TensorInput, WeightStore,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

const SOURCE_SHA256: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const MIN_ELEMENTS: usize = 1_048_576;
const DEFAULT_ELEMENTS: usize = 4_194_304;
const DEFAULT_CHAIN_GRAPHS: usize = 2;
const MIN_CHAIN_GRAPHS: usize = 2;
const MAX_CHAIN_GRAPHS: usize = 16;

#[derive(Serialize)]
struct PathReport {
    median_ns: u64,
    output_digest: String,
    host_to_device_copy_operations: u64,
    device_to_host_copy_operations: u64,
    input_materializations: u64,
    output_materializations: u64,
}

#[derive(Serialize)]
struct BenchReport {
    schema: &'static str,
    device: String,
    elements: usize,
    chain_graphs: usize,
    warmup_rounds: usize,
    measured_rounds: usize,
    resident: PathReport,
    owned_roundtrip: PathReport,
    speedup_resident_vs_owned: f64,
    parity: bool,
    copy_contract_ok: bool,
    accepted: bool,
    reject_reason: Option<String>,
}

fn add_plan(elements: usize, family: &str, role: &str) -> String {
    serde_json::json!({
        "schemaVersion": 1,
        "family": family,
        "role": role,
        "source": {
            "format": "reviewed-json",
            "sha256": SOURCE_SHA256,
            "opset": 1
        },
        "inputs": [{"name": "input", "shape": [1, elements]}],
        "outputs": [{"name": "output", "shape": [1, elements]}],
        "initializers": [{"name": "bias", "dtype": "float32", "shape": [1]}],
        "nodes": [{
            "name": "add-bias",
            "op": "Add",
            "inputs": ["input", "bias"],
            "outputs": ["output"],
            "attributes": {}
        }]
    })
    .to_string()
}

fn build_shared_executor(
    runtime: &EmbeddedRuntime,
    weight_dir: &PathBuf,
    elements: usize,
    family: &str,
    role: &str,
) -> Result<GraphExecutor> {
    let store = Arc::new(WeightStore::open(weight_dir, runtime.limits())?);
    let identity = GraphIdentity::new(family, role, "reviewed-json", SOURCE_SHA256, 1);
    let plan = GraphPlan::parse(
        &add_plan(elements, family, role),
        &identity,
        &store,
        runtime.limits(),
    )?;
    Ok(GraphExecutor::new(plan, store, runtime.clone())?)
}

fn materialize_bias(dir: &PathBuf, bias: f32) -> Result<()> {
    std::fs::create_dir_all(dir)
        .map_err(|error| PowerError::InvalidFormat(format!("weight dir: {error}")))?;
    let bytes = bias.to_le_bytes();
    let view = TensorView::new(Dtype::F32, vec![1], &bytes)
        .map_err(|error| PowerError::InvalidFormat(error.to_string()))?;
    serialize_to_file(vec![("bias", view)], None, &dir.join("model.safetensors"))
        .map_err(|error| PowerError::InvalidFormat(error.to_string()))?;
    Ok(())
}

fn median_ns(samples: &[u64]) -> u64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn digest(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn input_values(elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| ((index % 257) as f32) * 0.01 - 1.0)
        .collect()
}

fn merge_boundary(
    left: GraphExecutionBoundaryMeasurement,
    right: GraphExecutionBoundaryMeasurement,
) -> GraphExecutionBoundaryMeasurement {
    GraphExecutionBoundaryMeasurement {
        input_materializations: left
            .input_materializations
            .saturating_add(right.input_materializations),
        input_host_bytes: left.input_host_bytes.saturating_add(right.input_host_bytes),
        host_to_device_copy_operations: left
            .host_to_device_copy_operations
            .saturating_add(right.host_to_device_copy_operations),
        input_materialization_nanos: left
            .input_materialization_nanos
            .saturating_add(right.input_materialization_nanos),
        output_materializations: left
            .output_materializations
            .saturating_add(right.output_materializations),
        output_host_bytes: left
            .output_host_bytes
            .saturating_add(right.output_host_bytes),
        device_to_host_copy_operations: left
            .device_to_host_copy_operations
            .saturating_add(right.device_to_host_copy_operations),
        output_materialization_nanos: left
            .output_materialization_nanos
            .saturating_add(right.output_materialization_nanos),
    }
}

fn run_resident_chain(
    runtime: &EmbeddedRuntime,
    stages: &[GraphExecutor],
    values: &[f32],
    elements: usize,
) -> Result<(u64, String, GraphExecutionBoundaryMeasurement)> {
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation)?;
    let input = TensorInput::new(vec![1, elements], values.to_vec(), runtime.limits())?;
    let started = Instant::now();
    let mut handle = stages[0].run_to_resident(input, &permit, &cancellation)?;
    for stage in stages.iter().skip(1) {
        handle = stage.run_resident(handle, &permit, &cancellation)?;
    }
    let completed = handle.materialize(&cancellation)?;
    let elapsed = started.elapsed().as_nanos() as u64;
    Ok((
        elapsed,
        digest(&completed.output.values),
        completed.boundary,
    ))
}

fn run_owned_roundtrip(
    runtime: &EmbeddedRuntime,
    stages: &[GraphExecutor],
    values: &[f32],
    elements: usize,
) -> Result<(u64, String, GraphExecutionBoundaryMeasurement)> {
    use a3s_power::inference::ExecutionDigest;

    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation)?;
    let mut input = TensorInput::new(vec![1, elements], values.to_vec(), runtime.limits())?;
    let started = Instant::now();
    // Digest endpoints inside the timed window so owned matches the resident
    // chain contract (input digest at admit + output digest at materialize).
    let _input_digest = ExecutionDigest::f32_tensor(&input.shape, &input.values);
    let mut boundary = GraphExecutionBoundaryMeasurement::default();
    let mut out = None;
    for stage in stages {
        let (completed, stage_boundary) = stage.run_measured(input, &permit, &cancellation)?;
        boundary = merge_boundary(boundary, stage_boundary);
        out = Some(completed);
        input = out.as_ref().unwrap().clone().into_input(runtime.limits())?;
    }
    let out = out.expect("chain_graphs >= 2");
    let _output_digest = ExecutionDigest::f32_tensor(&out.shape, &out.values);
    let elapsed = started.elapsed().as_nanos() as u64;
    Ok((elapsed, digest(&out.values), boundary))
}

fn time_path<F>(rounds_warmup: usize, rounds_measured: usize, mut run: F) -> Result<PathReport>
where
    F: FnMut() -> Result<(u64, String, GraphExecutionBoundaryMeasurement)>,
{
    let mut samples = Vec::with_capacity(rounds_measured);
    let mut last_digest = String::new();
    let mut last_boundary = GraphExecutionBoundaryMeasurement::default();
    for round in 0..(rounds_warmup + rounds_measured) {
        let (elapsed, digest_hex, boundary) = run()?;
        if round >= rounds_warmup {
            samples.push(elapsed);
        }
        last_digest = digest_hex;
        last_boundary = boundary;
    }
    Ok(PathReport {
        median_ns: median_ns(&samples),
        output_digest: last_digest,
        host_to_device_copy_operations: last_boundary.host_to_device_copy_operations,
        device_to_host_copy_operations: last_boundary.device_to_host_copy_operations,
        input_materializations: last_boundary.input_materializations,
        output_materializations: last_boundary.output_materializations,
    })
}

fn parse_args() -> Result<(usize, usize, usize, usize, Option<PathBuf>, f64)> {
    let mut elements = DEFAULT_ELEMENTS;
    let mut chain_graphs = DEFAULT_CHAIN_GRAPHS;
    let mut warmup = 3;
    let mut measured = 9;
    let mut out: Option<PathBuf> = None;
    let mut min_speedup = 0.0;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--elements" => {
                elements = args
                    .get(index + 1)
                    .ok_or_else(|| PowerError::InvalidRequest("missing --elements".into()))?
                    .parse()
                    .map_err(|_| PowerError::InvalidRequest("bad --elements".into()))?;
                index += 2;
            }
            "--chain-graphs" => {
                chain_graphs = args
                    .get(index + 1)
                    .ok_or_else(|| PowerError::InvalidRequest("missing --chain-graphs".into()))?
                    .parse()
                    .map_err(|_| PowerError::InvalidRequest("bad --chain-graphs".into()))?;
                index += 2;
            }
            "--warmup-rounds" => {
                warmup = args
                    .get(index + 1)
                    .ok_or_else(|| PowerError::InvalidRequest("missing --warmup-rounds".into()))?
                    .parse()
                    .map_err(|_| PowerError::InvalidRequest("bad --warmup-rounds".into()))?;
                index += 2;
            }
            "--measured-rounds" => {
                measured = args
                    .get(index + 1)
                    .ok_or_else(|| PowerError::InvalidRequest("missing --measured-rounds".into()))?
                    .parse()
                    .map_err(|_| PowerError::InvalidRequest("bad --measured-rounds".into()))?;
                index += 2;
            }
            "--out" => {
                out = Some(PathBuf::from(args.get(index + 1).ok_or_else(|| {
                    PowerError::InvalidRequest("missing --out".into())
                })?));
                index += 2;
            }
            "--min-speedup" => {
                min_speedup = args
                    .get(index + 1)
                    .ok_or_else(|| PowerError::InvalidRequest("missing --min-speedup".into()))?
                    .parse()
                    .map_err(|_| PowerError::InvalidRequest("bad --min-speedup".into()))?;
                index += 2;
            }
            other => {
                return Err(PowerError::InvalidRequest(format!("unknown arg: {other}")));
            }
        }
    }
    if elements < MIN_ELEMENTS {
        return Err(PowerError::InvalidRequest(format!(
            "elements must be >= {MIN_ELEMENTS} (refuse unit-shape overfitting)"
        )));
    }
    if !(MIN_CHAIN_GRAPHS..=MAX_CHAIN_GRAPHS).contains(&chain_graphs) {
        return Err(PowerError::InvalidRequest(format!(
            "--chain-graphs must be in {MIN_CHAIN_GRAPHS}..={MAX_CHAIN_GRAPHS}"
        )));
    }
    if measured == 0 {
        return Err(PowerError::InvalidRequest(
            "--measured-rounds must be positive".into(),
        ));
    }
    Ok((elements, chain_graphs, warmup, measured, out, min_speedup))
}

fn main() -> Result<()> {
    let (elements, chain_graphs, warmup, measured, out, min_speedup) = parse_args()?;

    let mut temp_dirs = Vec::with_capacity(chain_graphs);
    let mut stages = Vec::with_capacity(chain_graphs);
    let limits = InferenceLimits {
        max_tensor_elements: elements.saturating_mul(4).max(elements),
        ..InferenceLimits::default()
    };
    let runtime = EmbeddedRuntime::new(DevicePreference::Cuda { ordinal: 0 }, limits)?;

    for index in 0..chain_graphs {
        let dir = tempfile::tempdir()
            .map_err(|error| PowerError::InvalidFormat(format!("tempdir: {error}")))?;
        let bias = (index + 1) as f32;
        materialize_bias(&dir.path().to_path_buf(), bias)?;
        let role = format!("stage-{index}");
        let executor = build_shared_executor(
            &runtime,
            &dir.path().to_path_buf(),
            elements,
            "cuda-resident-bench",
            &role,
        )?;
        stages.push(executor);
        temp_dirs.push(dir);
    }
    let values = input_values(elements);
    let n = chain_graphs as u64;

    let resident_report = time_path(warmup, measured, || {
        run_resident_chain(&runtime, &stages, &values, elements)
    })?;
    let owned_report = time_path(warmup, measured, || {
        run_owned_roundtrip(&runtime, &stages, &values, elements)
    })?;

    let parity = resident_report.output_digest == owned_report.output_digest;
    let copy_contract_ok = resident_report.host_to_device_copy_operations == 1
        && resident_report.device_to_host_copy_operations == 1
        && owned_report.host_to_device_copy_operations == n
        && owned_report.device_to_host_copy_operations == n;
    let speedup = owned_report.median_ns as f64 / resident_report.median_ns.max(1) as f64;

    let mut reject_reason = None;
    if !parity {
        reject_reason = Some("resident and owned CUDA outputs diverge".into());
    } else if !copy_contract_ok {
        reject_reason = Some(format!(
            "copy contract failed: resident H2D/D2H={}/{} owned={}/{} (want 1/1 and {n}/{n})",
            resident_report.host_to_device_copy_operations,
            resident_report.device_to_host_copy_operations,
            owned_report.host_to_device_copy_operations,
            owned_report.device_to_host_copy_operations
        ));
    } else if speedup + f64::EPSILON < min_speedup {
        reject_reason = Some(format!(
            "resident median {} ns is not >= {min_speedup}x owned {} ns (speedup={speedup:.3})",
            resident_report.median_ns, owned_report.median_ns
        ));
    }
    let accepted = reject_reason.is_none();

    let report = BenchReport {
        schema: "a3s.power.cuda-resident-bench.v1",
        device: "cuda:0".into(),
        elements,
        chain_graphs,
        warmup_rounds: warmup,
        measured_rounds: measured,
        resident: resident_report,
        owned_roundtrip: owned_report,
        speedup_resident_vs_owned: speedup,
        parity,
        copy_contract_ok,
        accepted,
        reject_reason: reject_reason.clone(),
    };
    let json = serde_json::to_string_pretty(&report)
        .map_err(|error| PowerError::InvalidFormat(error.to_string()))?;
    println!("{json}");
    if let Some(path) = out {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| PowerError::InvalidFormat(error.to_string()))?;
        }
        std::fs::write(&path, &json)
            .map_err(|error| PowerError::InvalidFormat(error.to_string()))?;
        eprintln!("WROTE {}", path.display());
    }

    if !accepted {
        return Err(PowerError::InferenceFailed(
            reject_reason.unwrap_or_else(|| "cuda resident bench rejected".into()),
        ));
    }
    Ok(())
}
