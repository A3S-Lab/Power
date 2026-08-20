use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::error::{PowerError, Result};

use super::super::{ExecutionDigest, TensorInput, TensorOutput};
use super::TensorBatchBenchmarkReport;

const MAX_CANONICAL_REPORT_BYTES: usize = 16 * 1024 * 1024;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReportPayload<'a> {
    schema: &'a str,
    binding: &'a super::super::HardwareEvidenceBinding,
    system: &'a super::super::StorageBenchmarkSystem,
    warmup_rounds: usize,
    measured_rounds: usize,
    item_count: usize,
    input_sequence_sha256: &'a str,
    output_sha256: &'a str,
    exact_output_parity: bool,
    samples: &'a [super::TensorBatchBenchmarkSample],
    summaries: &'a [super::TensorBatchBenchmarkSummary],
}

pub(super) fn input_sequence_sha256(inputs: &[TensorInput]) -> String {
    sequence_sha256(
        b"a3s-power-tensor-batch-input-sequence-v1\0",
        inputs
            .iter()
            .map(|input| ExecutionDigest::f32_tensor(&input.shape, &input.values)),
    )
}

pub(super) fn output_sequence_sha256(outputs: &[TensorOutput]) -> String {
    sequence_sha256(
        b"a3s-power-tensor-batch-output-sequence-v1\0",
        outputs
            .iter()
            .map(|output| ExecutionDigest::f32_tensor(&output.shape, &output.values)),
    )
}

fn sequence_sha256(
    domain: &[u8],
    digests: impl ExactSizeIterator<Item = ExecutionDigest>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((digests.len() as u64).to_le_bytes());
    for digest in digests {
        hasher.update((digest.sha256.len() as u64).to_le_bytes());
        hasher.update(digest.sha256.as_bytes());
        hasher.update((digest.byte_length as u64).to_le_bytes());
        hasher.update((digest.item_count as u64).to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub(super) fn report_sha256(report: &TensorBatchBenchmarkReport) -> Result<String> {
    let payload = ReportPayload {
        schema: &report.schema,
        binding: &report.binding,
        system: &report.system,
        warmup_rounds: report.warmup_rounds,
        measured_rounds: report.measured_rounds,
        item_count: report.item_count,
        input_sequence_sha256: &report.input_sequence_sha256,
        output_sha256: &report.output_sha256,
        exact_output_parity: report.exact_output_parity,
        samples: &report.samples,
        summaries: &report.summaries,
    };
    let bytes = serde_json::to_vec(&payload)?;
    if bytes.len() > MAX_CANONICAL_REPORT_BYTES {
        return Err(PowerError::InvalidRequest(format!(
            "tensor batch benchmark contains {} canonical bytes, exceeding the {MAX_CANONICAL_REPORT_BYTES} byte limit",
            bytes.len()
        )));
    }
    let mut hasher = Sha256::new();
    hasher.update(b"a3s-power-tensor-batch-benchmark-v1\0");
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}
