use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::inference::graph::{GraphExecutor, GraphIdentity, GraphPlan};
use crate::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, StorageBenchmarkSystem, TensorInput,
    WeightStore,
};

const SOURCE_SHA256: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const POWER_COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const RUNTIME_ARTIFACT_SHA256: &str =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

struct StepCounter {
    step: AtomicU64,
}

impl StepCounter {
    fn new() -> Self {
        Self {
            step: AtomicU64::new(0),
        }
    }
}

impl HostAllocationCounter for StepCounter {
    fn snapshot(&self) -> HostAllocationSnapshot {
        let step = self.step.fetch_add(1, Ordering::Relaxed);
        HostAllocationSnapshot {
            allocation_count: step * 10,
            allocated_bytes: step * 100,
            reallocation_count: step * 2,
            reallocated_bytes: step * 20,
        }
    }
}

fn system() -> StorageBenchmarkSystem {
    StorageBenchmarkSystem {
        os: "test-os".to_string(),
        architecture: "test-architecture".to_string(),
        cpu_model: "test-cpu".to_string(),
        logical_cpus: 8,
        ram_bytes: 16 * 1024 * 1024 * 1024,
        filesystem_class: "test-filesystem".to_string(),
        device_class: "test-device".to_string(),
    }
}

fn plan_json(family: &str) -> String {
    serde_json::json!({
        "schemaVersion": 1,
        "family": family,
        "role": "generic-transform",
        "source": {
            "format": "reviewed-json",
            "sha256": SOURCE_SHA256,
            "opset": 1
        },
        "inputs": [{"name": "input", "shape": ["batch", 2]}],
        "outputs": [{"name": "output", "shape": ["batch", 2]}],
        "initializers": [{"name": "bias", "dtype": "float32", "shape": [2]}],
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

fn graph(family: &str) -> (tempfile::TempDir, GraphExecutor, InferenceLimits) {
    let directory = tempfile::tempdir().unwrap();
    let bias = [1_f32, 2_f32]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let view = TensorView::new(Dtype::F32, vec![2], &bias).unwrap();
    serialize_to_file(
        vec![("bias", view)],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();

    let limits = InferenceLimits::default();
    let store = Arc::new(WeightStore::open(directory.path(), &limits).unwrap());
    let identity = GraphIdentity::new(
        family,
        "generic-transform",
        "reviewed-json",
        SOURCE_SHA256,
        1,
    );
    let plan = GraphPlan::parse(&plan_json(family), &identity, &store, &limits).unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let graph = GraphExecutor::new(plan, store, runtime).unwrap();
    (directory, graph, limits)
}

fn inputs(limits: &InferenceLimits) -> Vec<TensorInput> {
    vec![
        TensorInput::new(vec![1, 2], vec![3.0, 4.0], limits).unwrap(),
        TensorInput::new(vec![1, 2], vec![5.0, 6.0], limits).unwrap(),
    ]
}

fn report(family: &str) -> (tempfile::TempDir, TensorBatchBenchmarkReport) {
    let (directory, graph, limits) = graph(family);
    let report = run_tensor_batch_benchmark(
        &graph,
        &inputs(&limits),
        &TensorBatchBenchmarkConfig {
            power_commit: POWER_COMMIT.to_string(),
            runtime_artifact_sha256: RUNTIME_ARTIFACT_SHA256.to_string(),
            system: system(),
            warmup_rounds: 1,
            measured_rounds: 2,
        },
        &StepCounter::new(),
        &CancellationToken::new(),
    )
    .unwrap();
    (directory, report)
}

#[test]
fn benchmark_binds_generic_graph_and_named_hardware_with_exact_parity() {
    let (_directory, report) = report("family-alpha");

    report.verify().unwrap();
    assert_eq!(report.schema, TensorBatchBenchmarkReport::SCHEMA);
    assert_eq!(report.binding.graph_source_sha256, SOURCE_SHA256);
    assert_eq!(report.binding.runtime_device.name(), "cpu");
    assert_eq!(report.runtime_artifact_sha256, RUNTIME_ARTIFACT_SHA256);
    assert_eq!(report.system.device_class, "test-device");
    assert_eq!(report.samples.len(), 4);
    assert!(report.exact_output_parity);
    assert_eq!(report.summaries.len(), 2);
    assert_eq!(
        report
            .samples
            .iter()
            .map(|sample| sample.mode)
            .collect::<Vec<_>>(),
        [
            TensorBatchBenchmarkMode::Individual,
            TensorBatchBenchmarkMode::LeadingBatch,
            TensorBatchBenchmarkMode::LeadingBatch,
            TensorBatchBenchmarkMode::Individual,
        ]
    );
    assert!(report.samples.iter().all(|sample| {
        sample.host_allocations
            == HostAllocationMeasurement {
                allocation_count: 10,
                allocated_bytes: 100,
                reallocation_count: 2,
                reallocated_bytes: 20,
            }
    }));

    let individual = report
        .samples
        .iter()
        .find(|sample| sample.mode == TensorBatchBenchmarkMode::Individual)
        .unwrap();
    let batch = report
        .samples
        .iter()
        .find(|sample| sample.mode == TensorBatchBenchmarkMode::LeadingBatch)
        .unwrap();
    assert_eq!(individual.boundary.input_materializations, 2);
    assert_eq!(individual.boundary.output_materializations, 2);
    assert_eq!(batch.boundary.input_materializations, 1);
    assert_eq!(batch.boundary.output_materializations, 1);
    assert_eq!(individual.boundary.host_to_device_copy_operations, 0);
    assert_eq!(individual.boundary.device_to_host_copy_operations, 0);
    assert_eq!(
        individual.boundary.input_host_bytes,
        batch.boundary.input_host_bytes
    );
    assert_eq!(
        individual.boundary.output_host_bytes,
        batch.boundary.output_host_bytes
    );
}

#[test]
fn evidence_shape_is_model_neutral_and_omits_paths_families_and_values() {
    let (alpha_directory, alpha) = report("family-alpha");
    let (beta_directory, beta) = report("family-beta");

    let alpha_json = serde_json::to_value(&alpha).unwrap();
    let beta_json = serde_json::to_value(&beta).unwrap();
    assert_eq!(
        alpha_json.as_object().unwrap().keys().collect::<Vec<_>>(),
        beta_json.as_object().unwrap().keys().collect::<Vec<_>>()
    );
    let encoded = serde_json::to_string(&alpha).unwrap();
    assert!(!encoded.contains("family-alpha"));
    assert!(!encoded.contains(alpha_directory.path().to_string_lossy().as_ref()));
    assert!(!encoded.contains(beta_directory.path().to_string_lossy().as_ref()));
    assert!(!encoded.contains("3.0"));
    assert!(!encoded.contains("6.0"));
}

#[test]
fn report_digest_rejects_tampered_cost_evidence() {
    let (_directory, mut report) = report("family-alpha");
    report.samples[0].host_allocations.allocation_count += 1;

    assert!(report.verify().is_err());
}

#[test]
fn canonical_digest_cannot_hide_a_platform_or_order_mismatch() {
    let (_directory, mut platform) = report("family-alpha");
    platform.system.device_class = "different-device".to_string();
    platform.sha256 = super::digest::report_sha256(&platform).unwrap();
    assert!(platform.verify().is_err());

    let (_directory, mut order) = report("family-alpha");
    order.samples.swap(0, 1);
    order.summaries = super::validation::summaries(&order.samples).unwrap();
    order.sha256 = super::digest::report_sha256(&order).unwrap();
    assert!(order.verify().is_err());
}

#[test]
fn counters_and_configuration_fail_closed_at_their_bounds() {
    let backwards = HostAllocationSnapshot {
        allocation_count: 1,
        allocated_bytes: 1,
        reallocation_count: 1,
        reallocated_bytes: 1,
    }
    .checked_measurement_since(HostAllocationSnapshot {
        allocation_count: 2,
        allocated_bytes: 1,
        reallocation_count: 1,
        reallocated_bytes: 1,
    });
    assert!(backwards.is_err());

    let mut config = TensorBatchBenchmarkConfig {
        power_commit: POWER_COMMIT.to_string(),
        runtime_artifact_sha256: RUNTIME_ARTIFACT_SHA256.to_string(),
        system: system(),
        warmup_rounds: 0,
        measured_rounds: 0,
    };
    assert!(super::validation::validate_config(&config).is_err());
    config.measured_rounds = super::validation::MAX_MEASURED_ROUNDS + 1;
    assert!(super::validation::validate_config(&config).is_err());
    config.measured_rounds = 1;
    config.warmup_rounds = super::validation::MAX_WARMUP_ROUNDS + 1;
    assert!(super::validation::validate_config(&config).is_err());
}

#[test]
fn public_evidence_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<GraphExecutionBoundaryMeasurement>();
    assert_send_sync::<HostAllocationMeasurement>();
    assert_send_sync::<HostAllocationSnapshot>();
    assert_send_sync::<TensorBatchBenchmarkReport>();
    assert_send_sync::<TensorBatchBenchmarkSample>();
    assert_send_sync::<TensorBatchBenchmarkSummary>();
}

#[test]
fn incompatible_items_fail_before_measurement() {
    let (_directory, graph, limits) = graph("family-alpha");
    let incompatible = vec![
        TensorInput::new(vec![1, 2], vec![1.0, 2.0], &limits).unwrap(),
        TensorInput::new(vec![1, 3], vec![1.0, 2.0, 3.0], &limits).unwrap(),
    ];

    assert!(run_tensor_batch_benchmark(
        &graph,
        &incompatible,
        &TensorBatchBenchmarkConfig {
            power_commit: POWER_COMMIT.to_string(),
            runtime_artifact_sha256: RUNTIME_ARTIFACT_SHA256.to_string(),
            system: system(),
            warmup_rounds: 0,
            measured_rounds: 1,
        },
        &StepCounter::new(),
        &CancellationToken::new(),
    )
    .is_err());
}
