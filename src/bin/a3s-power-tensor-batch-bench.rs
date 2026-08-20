use std::path::{Path, PathBuf};
use std::{fs::File, io::Read};

use a3s_power::error::{PowerError, Result};
use a3s_power::inference::graph::{GraphExecutor, GraphIdentity, GraphPlan};
use a3s_power::inference::{
    run_tensor_batch_benchmark, DevicePreference, EmbeddedRuntime, InferenceLimits,
    StorageBenchmarkSystem, TensorBatchBenchmarkConfig, TensorInput, WeightStore,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

const MAX_INPUT_DOCUMENT_BYTES: u64 = 64 * 1024 * 1024;
const FIXTURE_PREFIX: &str = "a3s-power-tensor-batch-fixture-";

#[path = "tensor_batch_bench/allocator.rs"]
mod allocator;
#[path = "tensor_batch_bench/arguments.rs"]
mod arguments;
#[path = "tensor_batch_bench/output.rs"]
mod output;
#[path = "tensor_batch_bench/release_contract.rs"]
mod release_contract;
#[path = "tensor_batch_bench/release_fixture.rs"]
mod release_fixture;
#[path = "tensor_batch_bench/release_run.rs"]
mod release_run;

use allocator::ProcessAllocationCounter;
use arguments::Arguments;
use output::write_json_output;

const USAGE: &str = r#"A3S Power model-neutral tensor batch benchmark

Run a caller-owned reviewed graph:
  a3s-power-tensor-batch-bench run \
    --weights <directory> \
    --plan <reviewed-plan.json> \
    --inputs <tensor-items.json> \
    --family <model-owned-label> \
    --role <model-owned-label> \
    --source-format <model-owned-label> \
    --source-sha256 <lowercase-sha256> \
    --opset <number> \
    --device <cpu|cuda:N|metal:N> \
    --power-commit <lowercase-git-revision> \
    --filesystem-class <label> \
    --device-class <named-hardware-label> \
    --cpu-model <label> \
    --ram-bytes <bytes> \
    [--warmup-rounds <count>] \
    [--measured-rounds <count>]

Capture the complete contract for a caller-owned reviewed graph:
  a3s-power-tensor-batch-bench release-run \
    <all run arguments above> \
    --reference-output <tensor-output.json> \
    --profile-implementation-sha256 <lowercase-sha256> \
    --profile-shape-class-sha256 <lowercase-sha256> \
    --fallback-implementation-sha256 <lowercase-sha256> \
    --fallback-request-class-sha256 <lowercase-sha256> \
    --tee-policy-sha256 <lowercase-sha256> \
    --host-fixed-bytes <bytes> \
    --host-scratch-bytes <bytes> \
    --device-fixed-bytes <bytes> \
    --device-scratch-bytes <bytes>

Run the built-in generic Add graph fixture:
  a3s-power-tensor-batch-bench fixture \
    --device <cpu|cuda:N|metal:N> \
    --power-commit <lowercase-git-revision> \
    --filesystem-class <label> \
    --device-class <named-hardware-label> \
    --cpu-model <label> \
    --ram-bytes <bytes> \
    [--items <count>] \
    [--width <elements>] \
    [--warmup-rounds <count>] \
    [--measured-rounds <count>]

Capture the complete model-neutral runtime contract with the same fixture:
  a3s-power-tensor-batch-bench release-fixture \
    --device <cpu|cuda:N|metal:N> \
    --power-commit <lowercase-git-revision> \
    --filesystem-class <label> \
    --device-class <named-hardware-label> \
    --cpu-model <label> \
    --ram-bytes <bytes> \
    --tee-policy-sha256 <lowercase-sha256> \
    --host-fixed-bytes <bytes> \
    --host-scratch-bytes <bytes> \
    --device-fixed-bytes <bytes> \
    --device-scratch-bytes <bytes> \
    [--items <count>] \
    [--width <elements>] \
    [--warmup-rounds <count>] \
    [--measured-rounds <count>]

The JSON report is written to stdout. It contains named hardware and digests,
but no model path, graph path, tensor values, tensor names, or model-family
label. Allocation counters cover successful host heap allocations in this
isolated process; they do not claim visibility into device or driver allocators.

Add --output <new-json-file> to any command to write UTF-8 JSON directly. The
runner creates the file once and refuses to replace an existing path.
"#;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct InputDocument {
    items: Vec<TensorInput>,
}

struct CommonOptions {
    device: DevicePreference,
    power_commit: String,
    runtime_artifact_sha256: String,
    system: StorageBenchmarkSystem,
    warmup_rounds: usize,
    measured_rounds: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("tensor batch benchmark failed: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<()> {
    let mut values = std::env::args().skip(1).collect::<Vec<_>>();
    if values
        .iter()
        .any(|value| matches!(value.as_str(), "--help" | "-h"))
    {
        print!("{USAGE}");
        return Ok(());
    }
    let command = values.first().cloned().ok_or_else(|| {
        PowerError::InvalidRequest("tensor batch benchmark command is required".to_string())
    })?;
    values.remove(0);
    let mut arguments = Arguments::new(values);
    let output_path = arguments.optional_path("--output")?;
    let common = parse_common(&mut arguments)?;
    let output = match command.as_str() {
        "run" => serde_json::to_value(run_reviewed_graph(&mut arguments, &common)?)?,
        "fixture" => serde_json::to_value(run_fixture(&mut arguments, &common)?)?,
        "release-run" => serde_json::to_value(release_run::run(&mut arguments, &common)?)?,
        "release-fixture" => serde_json::to_value(release_fixture::run(&mut arguments, &common)?)?,
        _ => {
            return Err(PowerError::InvalidRequest(format!(
                "unsupported tensor batch benchmark command '{command}'"
            )))
        }
    };
    arguments.finish()?;
    write_json_output(&output, output_path.as_deref())?;
    Ok(())
}

fn parse_common(arguments: &mut Arguments) -> Result<CommonOptions> {
    let device = parse_device(&arguments.required("--device")?)?;
    let power_commit = arguments.required("--power-commit")?;
    let filesystem_class = arguments.required("--filesystem-class")?;
    let device_class = arguments.required("--device-class")?;
    let cpu_model = arguments.required("--cpu-model")?;
    let ram_bytes = arguments.required_number("--ram-bytes")?;
    let warmup_rounds = arguments.optional_number("--warmup-rounds")?.unwrap_or(2);
    let measured_rounds = arguments.optional_number("--measured-rounds")?.unwrap_or(9);
    Ok(CommonOptions {
        device,
        power_commit,
        runtime_artifact_sha256: current_executable_sha256()?,
        system: StorageBenchmarkSystem {
            os: std::env::consts::OS.to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            cpu_model,
            logical_cpus: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            ram_bytes,
            filesystem_class,
            device_class,
        },
        warmup_rounds,
        measured_rounds,
    })
}

fn run_reviewed_graph(
    arguments: &mut Arguments,
    common: &CommonOptions,
) -> Result<a3s_power::inference::TensorBatchBenchmarkReport> {
    let workload = build_reviewed_graph(arguments, common)?;
    benchmark(&workload.graph, &workload.inputs, common)
}

fn build_reviewed_graph(
    arguments: &mut Arguments,
    common: &CommonOptions,
) -> Result<BenchmarkWorkload> {
    let weights = arguments.required_path("--weights")?;
    let plan_path = arguments.required_path("--plan")?;
    let inputs_path = arguments.required_path("--inputs")?;
    let identity = GraphIdentity::new(
        arguments.required("--family")?,
        arguments.required("--role")?,
        arguments.required("--source-format")?,
        arguments.required("--source-sha256")?,
        arguments.required_number("--opset")?,
    );
    let limits = InferenceLimits::default();
    let plan_source = read_bounded_regular_utf8(
        &plan_path,
        limits.max_graph_plan_bytes as u64,
        "reviewed graph plan",
    )?;
    let input_source = read_bounded_regular(
        &inputs_path,
        MAX_INPUT_DOCUMENT_BYTES.min(limits.max_input_bytes as u64),
        "tensor input document",
    )?;
    let inputs = serde_json::from_slice::<InputDocument>(&input_source)?.items;
    let store = std::sync::Arc::new(WeightStore::open(weights, &limits)?);
    let plan = GraphPlan::parse(&plan_source, &identity, &store, &limits)?;
    let runtime = EmbeddedRuntime::new(common.device, limits)?;
    let graph = GraphExecutor::new(plan, store, runtime.clone())?;
    Ok(BenchmarkWorkload {
        _fixture_directory: None,
        graph,
        runtime,
        inputs,
    })
}

fn run_fixture(
    arguments: &mut Arguments,
    common: &CommonOptions,
) -> Result<a3s_power::inference::TensorBatchBenchmarkReport> {
    let fixture = build_fixture(arguments, common)?;
    benchmark(&fixture.graph, &fixture.inputs, common)
}

struct BenchmarkWorkload {
    _fixture_directory: Option<FixtureDirectory>,
    graph: GraphExecutor,
    runtime: EmbeddedRuntime,
    inputs: Vec<TensorInput>,
}

fn build_fixture(arguments: &mut Arguments, common: &CommonOptions) -> Result<BenchmarkWorkload> {
    let item_count = arguments.optional_number("--items")?.unwrap_or(8_usize);
    let width = arguments.optional_number("--width")?.unwrap_or(4_096_usize);
    if !(2..=1_024).contains(&item_count) || width == 0 {
        return Err(PowerError::InvalidRequest(
            "fixture items must be between 2 and 1024 and width must be positive".to_string(),
        ));
    }
    let limits = InferenceLimits::default();
    limits.checked_elements(&[item_count, width], "fixture batch")?;
    let fixture = FixtureDirectory::create()?;
    let bias = vec![0.25_f32; width]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let view = TensorView::new(Dtype::F32, vec![width], &bias).map_err(|error| {
        PowerError::InvalidFormat(format!("failed to build fixture tensor: {error}"))
    })?;
    serialize_to_file(
        vec![("bias", view)],
        None,
        &fixture.path.join("fixture.safetensors"),
    )
    .map_err(|error| {
        PowerError::InvalidFormat(format!("failed to serialize fixture weights: {error}"))
    })?;
    let source_sha256 = fixture_source_sha256();
    let plan_source = fixture_plan(width, &source_sha256);
    let identity = GraphIdentity::new(
        "generic-fixture",
        "elementwise-transform",
        "reviewed-json",
        source_sha256,
        1,
    );
    let store = std::sync::Arc::new(WeightStore::open(&fixture.path, &limits)?);
    let plan = GraphPlan::parse(&plan_source, &identity, &store, &limits)?;
    let runtime = EmbeddedRuntime::new(common.device, limits.clone())?;
    let graph = GraphExecutor::new(plan, store, runtime.clone())?;
    let inputs = (0..item_count)
        .map(|item| {
            let values = (0..width)
                .map(|element| item as f32 + element as f32 / width as f32)
                .collect::<Vec<_>>();
            TensorInput::new(vec![1, width], values, &limits)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(BenchmarkWorkload {
        _fixture_directory: Some(fixture),
        graph,
        runtime,
        inputs,
    })
}

fn benchmark(
    graph: &GraphExecutor,
    inputs: &[TensorInput],
    common: &CommonOptions,
) -> Result<a3s_power::inference::TensorBatchBenchmarkReport> {
    run_tensor_batch_benchmark(
        graph,
        inputs,
        &TensorBatchBenchmarkConfig {
            power_commit: common.power_commit.clone(),
            runtime_artifact_sha256: common.runtime_artifact_sha256.clone(),
            system: common.system.clone(),
            warmup_rounds: common.warmup_rounds,
            measured_rounds: common.measured_rounds,
        },
        &ProcessAllocationCounter,
        &CancellationToken::new(),
    )
}

fn current_executable_sha256() -> Result<String> {
    let executable = std::env::current_exe()?;
    let metadata = std::fs::symlink_metadata(&executable)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        return Err(PowerError::InvalidRequest(
            "benchmark executable must be a non-empty regular non-symlink file".to_string(),
        ));
    }
    let mut file = File::open(executable)?;
    let mut buffer = [0_u8; 64 * 1024];
    let mut hasher = Sha256::new();
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn fixture_source_sha256() -> String {
    format!("{:x}", Sha256::digest(b"a3s-power-generic-add-fixture-v1"))
}

fn fixture_plan(width: usize, source_sha256: &str) -> String {
    serde_json::json!({
        "schemaVersion": 1,
        "family": "generic-fixture",
        "role": "elementwise-transform",
        "source": {
            "format": "reviewed-json",
            "sha256": source_sha256,
            "opset": 1
        },
        "inputs": [{"name": "input", "shape": ["batch", width]}],
        "outputs": [{"name": "output", "shape": ["batch", width]}],
        "initializers": [{"name": "bias", "dtype": "float32", "shape": [width]}],
        "nodes": [{
            "name": "elementwise-add",
            "op": "Add",
            "inputs": ["input", "bias"],
            "outputs": ["output"],
            "attributes": {}
        }]
    })
    .to_string()
}

fn parse_device(value: &str) -> Result<DevicePreference> {
    if value == "cpu" {
        return Ok(DevicePreference::Cpu);
    }
    let (kind, ordinal) = value.split_once(':').ok_or_else(|| {
        PowerError::InvalidRequest(
            "device must be cpu, cuda:<ordinal>, or metal:<ordinal>".to_string(),
        )
    })?;
    let ordinal = ordinal.parse::<usize>().map_err(|_| {
        PowerError::InvalidRequest("device ordinal must be a non-negative integer".to_string())
    })?;
    match kind {
        "cuda" => Ok(DevicePreference::Cuda { ordinal }),
        "metal" => Ok(DevicePreference::Metal { ordinal }),
        _ => Err(PowerError::InvalidRequest(
            "device must be cpu, cuda:<ordinal>, or metal:<ordinal>".to_string(),
        )),
    }
}

fn read_bounded_regular_utf8(path: &Path, maximum: u64, label: &str) -> Result<String> {
    String::from_utf8(read_bounded_regular(path, maximum, label)?)
        .map_err(|error| PowerError::InvalidFormat(format!("{label} must contain UTF-8: {error}")))
}

fn read_bounded_regular(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() == 0
        || metadata.len() > maximum
    {
        return Err(PowerError::InvalidRequest(format!(
            "{label} must be a non-empty regular non-symlink file of at most {maximum} bytes"
        )));
    }
    Ok(std::fs::read(path)?)
}

struct FixtureDirectory {
    path: PathBuf,
}

impl FixtureDirectory {
    fn create() -> Result<Self> {
        let unique = format!(
            "{FIXTURE_PREFIX}{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|error| PowerError::InferenceFailed(error.to_string()))?
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        std::fs::create_dir(&path)?;
        Ok(Self { path })
    }
}

impl Drop for FixtureDirectory {
    fn drop(&mut self) {
        let expected_parent = std::env::temp_dir();
        let safe_name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(FIXTURE_PREFIX));
        if self.path.parent() == Some(expected_parent.as_path()) && safe_name {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_devices_require_explicit_ordinals() {
        assert_eq!(parse_device("cpu").unwrap(), DevicePreference::Cpu);
        assert_eq!(
            parse_device("cuda:2").unwrap(),
            DevicePreference::Cuda { ordinal: 2 }
        );
        assert!(parse_device("auto").is_err());
        assert!(parse_device("cuda").is_err());
        assert!(parse_device("cuda:-1").is_err());
    }

    #[test]
    fn direct_output_is_utf8_and_never_overwrites() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("capture.json");
        let value = serde_json::json!({"label": "model-neutral"});

        write_json_output(&value, Some(&output)).unwrap();
        assert_eq!(
            std::fs::read_to_string(&output).unwrap(),
            "{\n  \"label\": \"model-neutral\"\n}\n"
        );
        assert!(write_json_output(&serde_json::json!({}), Some(&output)).is_err());
        assert_eq!(
            std::fs::read_to_string(&output).unwrap(),
            "{\n  \"label\": \"model-neutral\"\n}\n"
        );
    }

    #[test]
    fn unknown_and_incomplete_arguments_fail_closed() {
        assert!(Arguments::new(vec!["--unknown".to_string()])
            .finish()
            .is_err());
        assert!(Arguments::new(vec!["--device".to_string()])
            .optional("--device")
            .is_err());
    }

    #[test]
    fn current_runner_artifact_has_a_canonical_sha256() {
        let digest = current_executable_sha256().unwrap();
        assert_eq!(digest.len(), 64);
        assert!(digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
    }
}
