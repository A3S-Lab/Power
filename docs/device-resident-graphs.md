# Device-Resident Reviewed Graph Chains

`ResidentGraphTensor` removes avoidable host round trips between adjacent
reviewed static graphs without moving model architecture into A3S Power. The
same contract is available to language, vision, OCR, embedding, scientific, and
multimodal model crates.

## First-principles contract

Three facts determine the API:

1. An accelerator-to-host-to-accelerator boundary transfers every tensor byte
   twice and can synchronize the device.
2. A retained tensor is still live memory and must remain inside the request's
   admission, cancellation, dtype, shape, device, and aggregate byte bounds.
3. An exact cryptographic output digest requires reading the output bytes. Power
   therefore hashes the initial owned input and the one final owned output; it
   never claims an intermediate digest that it did not compute.

The result is an affine chain:

```text
owned F32 input
  |  validate + canonical input digest + one host/device boundary
  v
reviewed graph A
  |
  +-- ResidentGraphTensor -- same request/runtime/device --> reviewed graph B
                                                          |
                                                          v
                                             one owned materialization
                                                          |
                                                          v
                                      canonical input/output digests
```

The handle is deliberately non-cloneable. It retains an `ExecutionPermit` and
one runtime-level byte reservation until the next graph consumes it, the caller
materializes it, or it is dropped after an error or cancellation.

## API flow

```rust,no_run
use a3s_power::inference::{EmbeddedRuntime, ExecutionPermit, TensorInput};
use a3s_power::inference::graph::GraphExecutor;
use tokio_util::sync::CancellationToken;

fn execute_chain(
    runtime: &EmbeddedRuntime,
    first: &GraphExecutor,
    second: &GraphExecutor,
    input: TensorInput,
    permit: &ExecutionPermit,
    cancellation: &CancellationToken,
) -> Result<(), a3s_power::error::PowerError> {
    let resident = first.run_to_resident(input, permit, cancellation)?;
    let resident = second.run_resident(resident, permit, cancellation)?;
    let completed = resident.materialize(cancellation)?;

    assert_eq!(completed.boundary.input_materializations, 1);
    assert_eq!(completed.boundary.output_materializations, 1);
    let _canonical_input = completed.input_digest;
    let _canonical_output = completed.output_digest;
    let _owned_output = completed.output;
    let _ = runtime.resident_tensor_snapshot();
    Ok(())
}
```

Both executors must contain reviewed plans and clones of the same
`EmbeddedRuntime`. `run_resident` also requires the exact same logical request
permit, not merely another permit from the same runtime.

## Validation and bounds

| Boundary | Enforced by Power |
| --- | --- |
| Plan load | One input and output; each shape dimension is a positive integer, a bounded symbol, or `null` |
| Owned input | Finite F32 values, positive bounded shape, reviewed fixed/symbolic shape match, exact runtime permit |
| Resident input | F32 dtype, exact current shape, reviewed shape match, same Candle device, same runtime, same request permit |
| Resident output | F32 dtype, reviewed output shape match, same device, per-tensor element bound |
| Live handles | Shared runtime budget of `max_tensor_elements * sizeof(f32)` bytes; non-cloneable reservations resize atomically |
| Final output | One owned copy, finite-value validation, canonical F32 tensor digest, cancellation before and after copy and hashing |

Repeated shape symbols bind exact dimensions across a graph's input and output.
For example, `["batch", 4]` to `["batch", 8]` preserves the same runtime batch
dimension while leaving the meaning of that dimension with the model crate.
`null` is an anonymous dynamic dimension. Power does not select padding,
buckets, sequence lengths, image sizes, or semantic layouts.

`EmbeddedRuntime::resident_tensor_snapshot` exposes only aggregate capacity,
current/peak bytes, active handle count, and rejection count. It is opt-in and
is not exported as automatic telemetry. Handle debug output omits values,
shape, boundary timings, graph identity, and the canonical input digest.

## Explicit owned fallback

Power never silently copies a handle across runtimes, request permits, or
devices. Materialize the source first, then reuse its owned allocation:

```rust,no_run
# use a3s_power::inference::EmbeddedRuntime;
# use a3s_power::inference::graph::ResidentGraphTensor;
# use tokio_util::sync::CancellationToken;
fn prepare_fallback(
    resident: ResidentGraphTensor,
    target: &EmbeddedRuntime,
    cancellation: &CancellationToken,
) -> Result<(), a3s_power::error::PowerError> {
    let completed = resident.materialize(cancellation)?;
    let intermediate_digest = completed.output_digest;
    let target_input = completed.output.into_input(target.limits())?;

    // Acquire the target permit after the source handle has been materialized
    // when both runtimes can share a single-capacity physical-device gate.
    let _ = (intermediate_digest, target_input);
    Ok(())
}
```

`TensorOutput::into_input` moves the existing host allocation and revalidates
the target limits; it does not allocate a second tensor payload. The next
`run_to_resident` recomputes the same canonical digest, so the intermediate
boundary is reproducible. If a caller needs a receipt for an intermediate graph
rather than the logical chain, that caller must materialize the intermediate
output and use its returned digest.

## Ownership boundary

Power owns the opaque handle, reviewed generic graph operators, validation,
budget, cancellation, and digest boundary. Model crates still own:

- architecture and topology;
- tokenization, decoding, generation, and recurrent/KV semantics;
- image resize/normalization policy and model-specific projections;
- OCR ROI geometry, DB postprocessing, CTC decoding, and document semantics;
- the decision to compose particular reviewed graphs.

No model name, format name, quantization mode, attention implementation, or
model-family dispatch appears in the resident-tensor path.

## Reproduce the contract tests

Run from the Power crate checkout:

```bash
cargo test --no-default-features --features embedded-inference \
  inference::graph::resident_tests -- --nocapture

cargo test --no-default-features --features embedded-inference \
  inference::runtime::resident_budget::tests -- --nocapture

CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 \
  cargo clippy --all-targets --no-default-features \
  --features embedded-inference -- -D warnings
```

The tests prove exact output/digest parity, one owned input and output boundary
for a two-graph chain, fixed/symbolic shape checks, F32 and runtime/permit
rejection, aggregate budget rejection, cancellation cleanup, and explicit
cross-runtime fallback continuity. The accelerator copy-count test uses a
logical test device with CPU storage; it validates accounting only. Real CUDA,
Metal, and confidential-accelerator performance captures remain a separate
release gate and must not be inferred from unit tests.
