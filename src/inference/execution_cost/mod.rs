mod benchmark;
mod digest;
#[cfg(test)]
mod tests;
mod types;
mod validation;

pub use benchmark::run_tensor_batch_benchmark;
pub use types::{
    GraphExecutionBoundaryMeasurement, HostAllocationCounter, HostAllocationMeasurement,
    HostAllocationSnapshot, TensorBatchBenchmarkConfig, TensorBatchBenchmarkMode,
    TensorBatchBenchmarkOrder, TensorBatchBenchmarkReport, TensorBatchBenchmarkSample,
    TensorBatchBenchmarkSummary,
};

#[cfg(all(test, feature = "server"))]
pub(crate) fn recompute_test_report_sha256(report: &TensorBatchBenchmarkReport) -> String {
    digest::report_sha256(report).expect("test report must have a canonical SHA-256")
}
