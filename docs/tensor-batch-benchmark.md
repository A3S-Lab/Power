# Tensor Batch Cost Benchmark Protocol

`a3s-power-tensor-batch-bench` measures the generic execution cost of running
the same reviewed single-input/single-output F32 graph as individual items and
as one leading-axis batch. It exists to test Power's tensor boundary, not a
particular language model, tokenizer, vocabulary, decoder, or model family.

The benchmark refuses to infer that batching is faster. It records raw samples,
requires exact output parity, and preserves negative results.

## What the benchmark measures

Each measured round contains both modes:

| Mode | Execution shape |
| --- | --- |
| `individual` | One admitted graph call per input item |
| `leading-batch` | `TensorInput::stack_leading`, one graph call, then `TensorOutput::split_leading` |

Round order alternates between individual-first and batch-first. Warmups are
unmeasured. Every measured sample records:

- end-to-end elapsed nanoseconds for batch assembly, graph execution, host
  boundary materialization, and output partitioning;
- successful host heap allocation and reallocation counts and requested bytes;
- input and output materialization counts, host-visible bytes, and durations;
- logical host-to-device and device-to-host copy counts for accelerator runs;
  and
- a digest of the exact ordered outputs.

On CPU, Candle adopts the owned input `Vec<f32>` without copying it, so the
host-to-device count is zero. Output materialization still creates an owned
host vector. On CUDA and Metal, the input and output boundary counts are one per
graph call. These are reviewed API-boundary operations, not claims about the
number of driver allocations, DMA packets, kernel launches, or internal backend
copies. Accelerator output timing includes provider synchronization, transfer,
and construction of the owned host vector.

The process-wide counting allocator is installed only by the benchmark binary.
The library and production runtime do not replace the application's allocator.
Run the benchmark in an otherwise idle process because unrelated threads in the
same process would contribute to the host allocation counters.

## Evidence binding and privacy

`TensorBatchBenchmarkReport` binds the raw samples to:

- the exact Power version and lowercase Git revision;
- the SHA-256 of the exact benchmark runner executable (or equivalent
  caller-owned runtime artifact);
- the verified weight-collection SHA-256;
- the model-owned reviewed graph/source SHA-256;
- the typed CPU, CUDA, or Metal device identity; and
- the named OS, architecture, CPU, RAM, filesystem, and device class.

The report includes no model path, graph path, tensor name, tensor value, model
family, graph role, tokenizer, prompt, or output value. The model family and
role are required while validating a caller-owned graph but deliberately stay
outside the report. The report's canonical SHA-256 detects mutation relative to
a pinned digest; it does not prove who ran the benchmark. Release evidence must
authenticate the report digest through a signed revision, attestation, or an
equivalent caller-owned trust root.

The executable digest is mandatory because one source revision can be built
with different Cargo features, compiler flags, link settings, or native
dependencies. Reports from CPU-only and accelerator-enabled binaries therefore
cannot be mistaken for the same runtime artifact even when their Git revision
matches.

Deserialization is fail-closed. `verify()` reconstructs the named-hardware
binding, checks canonical alternating order, requires two modes per round,
recomputes every median summary, verifies exact output-digest parity, and then
recomputes the report digest.

## Reproduce the generic fixture

The built-in fixture is a model-neutral broadcast Add graph with eight inputs
of shape `[1, 4096]`. It creates temporary verified SafeTensors weights, runs the
same public graph executor and report builder used by caller-owned graphs, and
removes the scoped temporary directory after the run.

Build from the Power crate directory and bind the capture to the checked-out
revision:

```powershell
$powerCommit = (git rev-parse HEAD).Trim()
cargo run --release --no-default-features `
  --features embedded-inference `
  --bin a3s-power-tensor-batch-bench -- fixture `
  --device cpu `
  --power-commit $powerCommit `
  --filesystem-class ntfs `
  --device-class "Intel Xeon w5-2445 CPU" `
  --cpu-model "Intel(R) Xeon(R) w5-2445" `
  --ram-bytes 137071693824 `
  --items 8 `
  --width 4096 `
  --warmup-rounds 2 `
  --measured-rounds 9 > tensor-batch-cpu.json
```

For CUDA, select the accelerator feature and typed ordinal explicitly:

```powershell
$powerCommit = (git rev-parse HEAD).Trim()
cargo run --release --no-default-features `
  --features embedded-cuda `
  --bin a3s-power-tensor-batch-bench -- fixture `
  --device cuda:0 `
  --power-commit $powerCommit `
  --filesystem-class ntfs `
  --device-class "NVIDIA GeForce RTX 4090 24 GiB; driver 610.74" `
  --cpu-model "Intel(R) Xeon(R) w5-2445" `
  --ram-bytes 137071693824 `
  --items 8 `
  --width 4096 `
  --warmup-rounds 2 `
  --measured-rounds 9 > tensor-batch-cuda.json
```

Linux and macOS use the same arguments with shell line continuations and honest
platform labels. Metal requires `--features embedded-metal --device metal:0`.
Do not label an emulated or fallback device as an accelerator capture.

The fixture proves that the benchmark and evidence contract are reproducible.
Its tiny Add workload is not evidence that batching improves a real model and
must not be reported as token throughput, request throughput, or an end-to-end
model gain.

## Run a caller-owned graph

The input document is a bounded JSON object containing at least two compatible
owned F32 tensors:

```json
{
  "items": [
    { "shape": [1, 2], "values": [1.0, 2.0] },
    { "shape": [1, 2], "values": [3.0, 4.0] }
  ]
}
```

Run the reviewed graph with explicit model-owned identity fields:

```bash
power_commit="$(git rev-parse HEAD)"
cargo run --release --no-default-features \
  --features embedded-inference \
  --bin a3s-power-tensor-batch-bench -- run \
  --weights /verified/model/root \
  --plan /reviewed/graph-plan.json \
  --inputs /private/tensor-items.json \
  --family model-owned-family \
  --role model-owned-role \
  --source-format reviewed-format \
  --source-sha256 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --opset 1 \
  --device cpu \
  --power-commit "$power_commit" \
  --filesystem-class ext4 \
  --device-class "named CPU host" \
  --cpu-model "exact CPU model" \
  --ram-bytes 68719476736 \
  --warmup-rounds 2 \
  --measured-rounds 9 > tensor-batch-report.json
```

The graph must accept the combined leading dimension and return outputs that
can be partitioned by the original leading dimensions. Power does not choose
padding, buckets, sequence semantics, image geometry, vocabulary, generation
policy, or a model-specific projection. A model integration that applies a
device-side output projection can use
`GraphExecutor::run_with_output_projection_measured` and must bind that
projection into its own reviewed execution identity.

## Capture the complete runtime contract

`fixture` and `run` emit only tensor batch reports. Their corresponding
`release-fixture` and `release-run` commands additionally execute and verify
peak memory, active cancellation cleanup, queue deadline expiry, replica
retirement/reconstruction, and exact explicit fallback.

The generic calibration command is reproducible from an isolated checkout:

```powershell
$powerCommit = (git rev-parse HEAD).Trim()
$teePolicySha256 = "<64 lowercase hex characters for the reviewed policy>"
cargo run --release --no-default-features `
  --features embedded-inference `
  --bin a3s-power-tensor-batch-bench -- release-fixture `
  --output release-cpu.json `
  --device cpu `
  --power-commit $powerCommit `
  --filesystem-class ntfs `
  --device-class "named CPU host" `
  --cpu-model "exact CPU model" `
  --ram-bytes 68719476736 `
  --tee-policy-sha256 $teePolicySha256 `
  --host-fixed-bytes 67108864 `
  --host-scratch-bytes 67108864 `
  --device-fixed-bytes 0 `
  --device-scratch-bytes 0 `
  --items 8 `
  --width 4096 `
  --warmup-rounds 2 `
  --measured-rounds 9
```

Memory values are predeclared bounds, not observed values copied back into the
command. A capture fails if its measured additional peak exceeds
fixed-plus-scratch or if its final retained increase exceeds fixed. CUDA and
Metal require positive device bounds and emit sampled device-pool evidence;
CPU requires both device values to be zero.

For any caller-owned reviewed graph, prepare a bounded `TensorOutput` JSON file
from an independent reviewed implementation for the first input:

```json
{ "shape": [1, 2], "values": [1.5, 2.5] }
```

Then run the same collector. The four profile values are opaque SHA-256
identities owned by the integrating crate; Power validates and binds them but
does not interpret their architecture or modality.

```bash
power_commit="$(git rev-parse HEAD)"
cargo run --release --no-default-features \
  --features embedded-inference \
  --bin a3s-power-tensor-batch-bench -- release-run \
  --output release-capture.json \
  --weights /verified/model/root \
  --plan /reviewed/graph-plan.json \
  --inputs /private/tensor-items.json \
  --reference-output /private/reference-output.json \
  --family model-owned-family \
  --role model-owned-role \
  --source-format reviewed-format \
  --source-sha256 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --opset 1 \
  --profile-implementation-sha256 1111111111111111111111111111111111111111111111111111111111111111 \
  --profile-shape-class-sha256 2222222222222222222222222222222222222222222222222222222222222222 \
  --fallback-implementation-sha256 3333333333333333333333333333333333333333333333333333333333333333 \
  --fallback-request-class-sha256 4444444444444444444444444444444444444444444444444444444444444444 \
  --tee-policy-sha256 5555555555555555555555555555555555555555555555555555555555555555 \
  --host-fixed-bytes 67108864 \
  --host-scratch-bytes 67108864 \
  --device-fixed-bytes 0 \
  --device-scratch-bytes 0 \
  --device cpu \
  --power-commit "$power_commit" \
  --filesystem-class ext4 \
  --device-class "named CPU host" \
  --cpu-model "exact CPU model" \
  --ram-bytes 68719476736 \
  --warmup-rounds 2 \
  --measured-rounds 9
```

The emitted JSON includes digests and aggregate counters, not any supplied
path, tensor value, family, role, tokenizer, model format, or architecture
switch. `ReleaseCapture::verify()` can replay it after deserialization. A
production policy uses one `ReleasePlatformBinding` per platform because shape
profiles and TEE policies are device-specific.

`--output` writes UTF-8 JSON with create-new semantics and refuses to overwrite
an existing capture. Omit it to keep stdout output for pipelines.

## Interpretation and promotion gate

Compare raw samples and lower medians rather than one best run. Fewer allocation
operations can coexist with more allocated bytes, and one large batch can be
slower for a small CPU graph. That is a valid negative result.

A batching policy may be promoted only after the owning model integration also
publishes:

1. scalar-versus-batch parity on pinned model inputs and outputs;
2. end-to-end latency and throughput on named target hardware;
3. bounded peak host and device memory;
4. cancellation, fallback, and resource-limit behavior; and
5. an authenticated evidence digest bound to the exact model, graph, weights,
   runtime, device, and build revision.

No fixture or Power-level microbenchmark establishes a universal model speedup.

## Published named-hardware evidence

The checked-in [Windows CPU and RTX 4090 capture](benchmarks/tensor-batch-cost-windows-20260820/README.md)
contains both raw reports, exact runtime-artifact and report pins, environment
details, median summaries, reproduction commands, and the retained CPU negative
result. Repository tests deserialize and replay `verify()` for both reports.
