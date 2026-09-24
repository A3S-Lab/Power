//! End-to-end GraphExecutor CUDA fusion harness.
//!
//! Contract (docs/perf-first-principles.md):
//! - fused vs intentionally unfused CUDA plans must byte-match
//! - latency is measured on ≥1M-element tensors (not unit-test shapes)
//! - exit non-zero if parity fails or fused loses by more than the tolerance

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use a3s_power::error::{PowerError, Result};
use a3s_power::inference::graph::{GraphExecutor, GraphIdentity, GraphPlan};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, TensorInput, WeightStore,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

const SOURCE_SHA256: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const MIN_ELEMENTS: usize = 1_048_576;
const DEFAULT_ELEMENTS: usize = 4_194_304;

#[derive(Serialize)]
struct BenchReport {
    schema: &'static str,
    device: String,
    elements: usize,
    warmup_rounds: usize,
    measured_rounds: usize,
    fused_median_ns: u64,
    unfused_median_ns: u64,
    speedup: f64,
    output_digest: String,
    parity: bool,
    accepted: bool,
    reject_reason: Option<String>,
}

fn gelu_plan(fused: bool) -> String {
    let mut nodes = vec![
        serde_json::json!({
            "name": "divide",
            "op": "Div",
            "inputs": ["input", "divisor"],
            "outputs": ["divided"],
            "attributes": {}
        }),
    ];
    if !fused {
        // Breaks the exact Div/Erf/Add/Mul/Mul fusion window on purpose.
        nodes.push(serde_json::json!({
            "name": "identity-barrier",
            "op": "Identity",
            "inputs": ["divided"],
            "outputs": ["divided_barrier"],
            "attributes": {}
        }));
    }
    let erf_in = if fused { "divided" } else { "divided_barrier" };
    nodes.extend([
        serde_json::json!({
            "name": "erf",
            "op": "Erf",
            "inputs": [erf_in],
            "outputs": ["activated"],
            "attributes": {}
        }),
        serde_json::json!({
            "name": "add",
            "op": "Add",
            "inputs": ["activated", "offset"],
            "outputs": ["shifted"],
            "attributes": {}
        }),
        serde_json::json!({
            "name": "multiply-input",
            "op": "Mul",
            "inputs": ["input", "shifted"],
            "outputs": ["product"],
            "attributes": {}
        }),
        serde_json::json!({
            "name": "multiply-scale",
            "op": "Mul",
            "inputs": ["product", "scale"],
            "outputs": ["output"],
            "attributes": {}
        }),
    ]);

    serde_json::json!({
        "schemaVersion": 1,
        "family": "cuda-fusion-bench",
        "role": "activation",
        "source": {
            "format": "synthetic",
            "sha256": SOURCE_SHA256,
            "opset": 17
        },
        "inputs": [{"name": "input", "shape": [1, DEFAULT_ELEMENTS]}],
        "outputs": [{"name": "output", "shape": [1, DEFAULT_ELEMENTS]}],
        "initializers": [
            {"name": "divisor", "dtype": "float32", "shape": [1]},
            {"name": "offset", "dtype": "float32", "shape": [1]},
            {"name": "scale", "dtype": "float32", "shape": [1]}
        ],
        "nodes": nodes
    })
    .to_string()
}

fn build_executor(
    plan_json: &str,
    elements: usize,
    device: DevicePreference,
    weight_dir: &PathBuf,
) -> Result<(EmbeddedRuntime, GraphExecutor)> {
    let limits = InferenceLimits {
        max_tensor_elements: elements.saturating_mul(4).max(elements),
        ..InferenceLimits::default()
    };
    let store = Arc::new(WeightStore::open(weight_dir, &limits)?);
    let identity = GraphIdentity::new(
        "cuda-fusion-bench",
        "activation",
        "synthetic",
        SOURCE_SHA256,
        17,
    );
    let mut plan_doc: serde_json::Value = serde_json::from_str(plan_json)
        .map_err(|error| PowerError::InvalidFormat(error.to_string()))?;
    plan_doc["inputs"][0]["shape"] = serde_json::json!([1, elements]);
    plan_doc["outputs"][0]["shape"] = serde_json::json!([1, elements]);
    let plan_json = plan_doc.to_string();
    let plan = GraphPlan::parse(&plan_json, &identity, &store, &limits)?;
    let runtime = EmbeddedRuntime::new(device, limits)?;
    let graph = GraphExecutor::new(plan, store, runtime.clone())?;
    Ok((runtime, graph))
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

fn materialize_weights(dir: &PathBuf) -> Result<()> {
    std::fs::create_dir_all(dir)
        .map_err(|error| PowerError::InvalidFormat(format!("weight dir: {error}")))?;
    let divisor = std::f32::consts::SQRT_2.to_le_bytes();
    let offset = 1.0_f32.to_le_bytes();
    let scale = 0.5_f32.to_le_bytes();
    let divisor_view = TensorView::new(Dtype::F32, vec![1], &divisor)
        .map_err(|error| PowerError::InvalidFormat(error.to_string()))?;
    let offset_view = TensorView::new(Dtype::F32, vec![1], &offset)
        .map_err(|error| PowerError::InvalidFormat(error.to_string()))?;
    let scale_view = TensorView::new(Dtype::F32, vec![1], &scale)
        .map_err(|error| PowerError::InvalidFormat(error.to_string()))?;
    serialize_to_file(
        vec![
            ("divisor", divisor_view),
            ("offset", offset_view),
            ("scale", scale_view),
        ],
        None,
        &dir.join("model.safetensors"),
    )
    .map_err(|error| PowerError::InvalidFormat(error.to_string()))?;
    Ok(())
}

fn input_values(elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| ((index % 257) as f32) * 0.01 - 1.0)
        .collect()
}

fn time_plan(
    runtime: &EmbeddedRuntime,
    graph: &GraphExecutor,
    values: &[f32],
    elements: usize,
    warmup: usize,
    measured: usize,
) -> Result<(u64, String, Vec<f32>)> {
    let limits = runtime.limits().clone();
    let cancellation = CancellationToken::new();
    let mut samples = Vec::with_capacity(measured);
    let mut last_values = Vec::new();
    for round in 0..(warmup + measured) {
        let permit = runtime.begin(&cancellation)?;
        let input = TensorInput::new(vec![1, elements], values.to_vec(), &limits)?;
        let start = Instant::now();
        let output = graph.run(input, &permit, &cancellation)?;
        let elapsed = start.elapsed().as_nanos() as u64;
        if round >= warmup {
            samples.push(elapsed);
        }
        last_values = output.values;
    }
    Ok((median_ns(&samples), digest(&last_values), last_values))
}

fn parse_args() -> Result<(usize, usize, usize, Option<PathBuf>, f64)> {
    let mut elements = DEFAULT_ELEMENTS;
    let mut warmup = 3;
    let mut measured = 9;
    let mut out: Option<PathBuf> = None;
    // Default 0: parity-only gate. Claiming a fusion *win* requires an explicit
    // --min-speedup (for example 1.05) so noise cannot create false rejects.
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
    if measured == 0 {
        return Err(PowerError::InvalidRequest(
            "--measured-rounds must be positive".into(),
        ));
    }
    Ok((elements, warmup, measured, out, min_speedup))
}

fn main() -> Result<()> {
    let (elements, warmup, measured, out, min_speedup) = parse_args()?;
    let weight_dir = tempfile::tempdir()
        .map_err(|error| PowerError::InvalidFormat(format!("tempdir: {error}")))?;
    materialize_weights(&weight_dir.path().to_path_buf())?;

    let device = DevicePreference::Cuda { ordinal: 0 };
    let values = input_values(elements);

    let (fused_runtime, fused_graph) = build_executor(
        &gelu_plan(true),
        elements,
        device.clone(),
        &weight_dir.path().to_path_buf(),
    )?;
    let (unfused_runtime, unfused_graph) = build_executor(
        &gelu_plan(false),
        elements,
        device,
        &weight_dir.path().to_path_buf(),
    )?;

    let (fused_median, fused_digest, fused_values) = time_plan(
        &fused_runtime,
        &fused_graph,
        &values,
        elements,
        warmup,
        measured,
    )?;
    let (unfused_median, unfused_digest, unfused_values) = time_plan(
        &unfused_runtime,
        &unfused_graph,
        &values,
        elements,
        warmup,
        measured,
    )?;

    let parity = fused_digest == unfused_digest
        && fused_values
            .iter()
            .zip(&unfused_values)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    let speedup = unfused_median as f64 / fused_median.max(1) as f64;
    let mut reject_reason = None;
    if !parity {
        reject_reason = Some("fused and unfused CUDA outputs diverge".into());
    } else if speedup + f64::EPSILON < min_speedup {
        reject_reason = Some(format!(
            "fused median {fused_median} ns is not >= {min_speedup}x unfused {unfused_median} ns (speedup={speedup:.3})"
        ));
    }
    let accepted = reject_reason.is_none();

    let report = BenchReport {
        schema: "a3s.power.cuda-fusion-bench.v1",
        device: "cuda:0".into(),
        elements,
        warmup_rounds: warmup,
        measured_rounds: measured,
        fused_median_ns: fused_median,
        unfused_median_ns: unfused_median,
        speedup,
        output_digest: fused_digest,
        parity,
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
            reject_reason.unwrap_or_else(|| "cuda fusion bench rejected".into()),
        ));
    }
    Ok(())
}
