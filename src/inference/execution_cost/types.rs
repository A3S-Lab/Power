use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

use super::super::{HardwareEvidenceBinding, StorageBenchmarkSystem};

/// Process-wide host allocator counters sampled by an isolated benchmark
/// runner. Device and driver allocations are deliberately outside this scope.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HostAllocationSnapshot {
    pub allocation_count: u64,
    pub allocated_bytes: u64,
    pub reallocation_count: u64,
    pub reallocated_bytes: u64,
}

impl HostAllocationSnapshot {
    pub(super) fn checked_measurement_since(
        self,
        earlier: Self,
    ) -> Result<HostAllocationMeasurement> {
        Ok(HostAllocationMeasurement {
            allocation_count: checked_delta(
                self.allocation_count,
                earlier.allocation_count,
                "host allocation count",
            )?,
            allocated_bytes: checked_delta(
                self.allocated_bytes,
                earlier.allocated_bytes,
                "host allocated byte count",
            )?,
            reallocation_count: checked_delta(
                self.reallocation_count,
                earlier.reallocation_count,
                "host reallocation count",
            )?,
            reallocated_bytes: checked_delta(
                self.reallocated_bytes,
                earlier.reallocated_bytes,
                "host reallocated byte count",
            )?,
        })
    }
}

fn checked_delta(current: u64, earlier: u64, label: &str) -> Result<u64> {
    current.checked_sub(earlier).ok_or_else(|| {
        PowerError::InferenceFailed(format!("tensor batch benchmark {label} moved backwards"))
    })
}

/// Successful host heap allocation activity within one measured sample.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HostAllocationMeasurement {
    pub allocation_count: u64,
    pub allocated_bytes: u64,
    pub reallocation_count: u64,
    pub reallocated_bytes: u64,
}

/// Read-only allocation counter implemented by an isolated benchmark process.
///
/// Implementations must not allocate while taking a snapshot. Power does not
/// install a global allocator in the library or production runtime.
pub trait HostAllocationCounter: Send + Sync {
    fn snapshot(&self) -> HostAllocationSnapshot;
}

/// Measured work at the owned-host-tensor/execution-device boundary.
///
/// On CPU, Candle adopts an input `Vec` without copying it, so device-copy
/// counts are zero even though input materialization is timed. Output host
/// materialization always creates an owned `Vec`; accelerator timings also
/// include provider synchronization and the device-to-host transfer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GraphExecutionBoundaryMeasurement {
    pub input_materializations: u64,
    pub input_host_bytes: u64,
    pub host_to_device_copy_operations: u64,
    pub input_materialization_nanos: u64,
    pub output_materializations: u64,
    pub output_host_bytes: u64,
    pub device_to_host_copy_operations: u64,
    pub output_materialization_nanos: u64,
}

impl GraphExecutionBoundaryMeasurement {
    pub(super) fn checked_add(self, other: Self) -> Result<Self> {
        Ok(Self {
            input_materializations: add(
                self.input_materializations,
                other.input_materializations,
                "input materialization count",
            )?,
            input_host_bytes: add(
                self.input_host_bytes,
                other.input_host_bytes,
                "input host byte count",
            )?,
            host_to_device_copy_operations: add(
                self.host_to_device_copy_operations,
                other.host_to_device_copy_operations,
                "host-to-device copy count",
            )?,
            input_materialization_nanos: add(
                self.input_materialization_nanos,
                other.input_materialization_nanos,
                "input materialization duration",
            )?,
            output_materializations: add(
                self.output_materializations,
                other.output_materializations,
                "output materialization count",
            )?,
            output_host_bytes: add(
                self.output_host_bytes,
                other.output_host_bytes,
                "output host byte count",
            )?,
            device_to_host_copy_operations: add(
                self.device_to_host_copy_operations,
                other.device_to_host_copy_operations,
                "device-to-host copy count",
            )?,
            output_materialization_nanos: add(
                self.output_materialization_nanos,
                other.output_materialization_nanos,
                "output materialization duration",
            )?,
        })
    }
}

fn add(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_add(right).ok_or_else(|| {
        PowerError::InferenceFailed(format!("tensor batch benchmark {label} overflowed"))
    })
}

/// Explicit controls for one scalar-versus-leading-batch benchmark.
#[derive(Clone)]
pub struct TensorBatchBenchmarkConfig {
    pub power_commit: String,
    pub system: StorageBenchmarkSystem,
    pub warmup_rounds: usize,
    pub measured_rounds: usize,
}

impl std::fmt::Debug for TensorBatchBenchmarkConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorBatchBenchmarkConfig")
            .field("power_commit", &"revision")
            .field("system", &self.system)
            .field("warmup_rounds", &self.warmup_rounds)
            .field("measured_rounds", &self.measured_rounds)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TensorBatchBenchmarkMode {
    Individual,
    LeadingBatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TensorBatchBenchmarkOrder {
    IndividualThenLeadingBatch,
    LeadingBatchThenIndividual,
}

impl TensorBatchBenchmarkOrder {
    pub(super) fn for_round(round: usize) -> Self {
        if round.is_multiple_of(2) {
            Self::IndividualThenLeadingBatch
        } else {
            Self::LeadingBatchThenIndividual
        }
    }

    pub(super) fn modes(self) -> [TensorBatchBenchmarkMode; 2] {
        match self {
            Self::IndividualThenLeadingBatch => [
                TensorBatchBenchmarkMode::Individual,
                TensorBatchBenchmarkMode::LeadingBatch,
            ],
            Self::LeadingBatchThenIndividual => [
                TensorBatchBenchmarkMode::LeadingBatch,
                TensorBatchBenchmarkMode::Individual,
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TensorBatchBenchmarkSample {
    pub round: usize,
    pub order: TensorBatchBenchmarkOrder,
    pub mode: TensorBatchBenchmarkMode,
    pub item_count: usize,
    pub execution_count: usize,
    pub elapsed_nanos: u64,
    pub host_allocations: HostAllocationMeasurement,
    pub boundary: GraphExecutionBoundaryMeasurement,
    pub output_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TensorBatchBenchmarkSummary {
    pub mode: TensorBatchBenchmarkMode,
    pub sample_count: usize,
    pub median_elapsed_nanos: u64,
    pub median_host_allocation_count: u64,
    pub median_host_allocated_bytes: u64,
    pub median_host_reallocation_count: u64,
    pub median_host_reallocated_bytes: u64,
    pub median_input_materialization_nanos: u64,
    pub median_output_materialization_nanos: u64,
}

/// Canonical, path-free evidence for generic individual and leading-batch
/// execution of one reviewed static graph on named hardware.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TensorBatchBenchmarkReport {
    pub schema: String,
    pub binding: HardwareEvidenceBinding,
    pub system: StorageBenchmarkSystem,
    pub warmup_rounds: usize,
    pub measured_rounds: usize,
    pub item_count: usize,
    pub input_sequence_sha256: String,
    pub output_sha256: String,
    pub exact_output_parity: bool,
    pub samples: Vec<TensorBatchBenchmarkSample>,
    pub summaries: Vec<TensorBatchBenchmarkSummary>,
    pub sha256: String,
}

impl TensorBatchBenchmarkReport {
    pub const SCHEMA: &'static str = "a3s.power.tensor-batch-benchmark.v1";

    /// Revalidates canonical structure, derived summaries, platform binding,
    /// exact output parity, and the embedded report digest.
    pub fn verify(&self) -> Result<()> {
        super::validation::verify_report(self)
    }
}

impl std::fmt::Debug for TensorBatchBenchmarkReport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorBatchBenchmarkReport")
            .field("schema", &self.schema)
            .field("binding", &self.binding)
            .field("system", &self.system)
            .field("warmup_rounds", &self.warmup_rounds)
            .field("measured_rounds", &self.measured_rounds)
            .field("item_count", &self.item_count)
            .field("sample_count", &self.samples.len())
            .field("exact_output_parity", &self.exact_output_parity)
            .field("sha256", &self.sha256)
            .finish()
    }
}
