# Magnitude Native Backend Production Guide

The optional `magnitude` feature embeds Magnitude's Engine and Seismic native
runtime in `a3s-power`. It provides an in-process GGUF Chat backend while Power
retains HTTP authentication, admission, artifact verification, model lifecycle,
metrics, and inference receipts. No second OpenAI server is required.

This guide defines the supported production profile. It does not claim that
Power v1, every model family, every accelerator, or every Power API is ready.

## Build and configure

Use the pinned source and lockfile. Do not substitute a moving Magnitude branch.
Release archives already include this backend. To build from the Aster source
workspace, run:

~~~bash
CARGO_TARGET_DIR=target/magnitude-release \
cargo build --manifest-path crates/power/Cargo.toml --locked --release \
  -p a3s-power --no-default-features --features magnitude --bin a3s-power
~~~

A standalone Power checkout additionally needs the pinned `../acl` and
`../runtime` sources used by its manifest; release CI checks out those exact
revisions before running the locked build.

Start with [`examples/magnitude-production/power.acl`](examples/magnitude-production/power.acl):

~~~bash
A3S_POWER_HOME=/var/lib/a3s-power \
  target/magnitude-release/release/a3s-power serve \
  --config /var/lib/a3s-power/config.acl
~~~

`magnitude_device` accepts `auto`, `cpu`, `metal`, `cuda`, `vulkan`, or an exact
Seismic selector. An explicit unavailable device fails instead of silently
moving execution elsewhere. `gpu_layers` and llama.cpp-specific placement
overrides do not select Magnitude's device.

For non-loopback listeners, configure `api_keys`, TLS as applicable, network
policy, and tenant isolation before exposing the service. Keep the example's
loopback default for local agent use.

## Supported production profile

| Area | Supported and verified | Boundary |
| --- | --- | --- |
| Artifacts | Recognized GGUF targets with exact size and SHA-256 verification; optional projector and bound external draft | Other containers and unrecognized architectures fail closed |
| Chat | Text, reasoning, streaming and non-streaming, exact usage and native timings | Completion is not implemented |
| Tools | Function definitions, required and named choices, single-tool mode, streamed argument deltas, tool results in later turns | Parallel behavior depends on the model; malformed streams fail |
| Structured output | `json_object` and schema-constrained `json_schema` | Unsupported fields and invalid schemas are rejected |
| Devices | CPU and Metal with cold/warm Prepare, load, inference, cancellation, unload, and reload | CUDA and Vulkan selectors exist but have not passed hardware qualification |
| Context | 4K on CPU and Metal; 8K on CPU; 16K on Metal | Larger contexts require separate memory, latency, and quality qualification |
| Recovery | Request-local errors, SSE cancellation, model unload/reload, and warm restart | Worker/process crash recovery is not a persistence guarantee |
| Operations | Bounded HTTP admission, queue timeouts, model hashes, source provenance, release binary, and full Power unit tests | Production rollout still requires environment-specific limits and monitoring |

The verified model is `MiniCPM5-1B-Q4_K_M.gguf` with SHA-256
`81b64d05a23b17b34c475f42b3e72fbde62d4b92cc34541f7a8031d0752deafa`.
Another GGUF is admitted only when Magnitude's exact family inspection accepts
it; format or filename similarity is not support evidence.

## Explicitly unsupported in this profile

- Text Completion, embeddings, and SystemOne scoring.
- Keyed prompt caching (`prompt_cache_key`). Requests carrying it fail rather
  than sharing unnamespaced state.
- Images and other multimodal input, LoRA loading, manifest template overrides,
  and manifest-provided prompts.
- Confidential in-memory model loading and the strict TEE profile.
- Arbitrary GPU layer, tensor placement, or llama.cpp sampler overrides.

These are capability boundaries, not silent fallbacks. Use another registered
Power backend when a different capability is required.

## Loading and resource behavior

The first load of an exact model/device profile runs Magnitude Prepare in a
short-lived worker process. Normal inference remains in-process. Kernel choices
are cached below `$A3S_POWER_HOME/kernels/magnitude`; warm loads reuse matching
records without another Prepare. Do not share that cache between different
model bytes, engine builds, or device identities.

Power's HTTP limits provide outer admission control. Magnitude provides inner
batching, fairness, capacity negotiation, and memory reclamation. A memory
shortfall can reject or evict work; this is intended protection, not a reason to
raise limits blindly. Measure cold readiness separately from warm request
latency and do not treat Prepare time as steady-state TTFT.

## Acceptance evidence

Evidence is retained under
`docs/evidence/power/2026-10-30-magnitude-native-backend/`:

- `power-full-test.log`: all 2,028 Power unit tests pass.
- `magnitude-release-build.log`, `release-version.log`, and
  `release-linkage.log`: optimized build and runtime identity.
- `release-comparison.json`: short-request CPU/Metal comparison, three warm
  samples per backend.
- `agent-workload-comparison.json`: approximately 4K prefill and two concurrent
  sessions.
- `live-semantic-*.json` and `live-stream-*.json`: tools, streamed arguments,
  and JSON Schema on both backends.
- `live-multiturn-*.json`: tool results consumed correctly in the next turn.
- `live-long-*.json`: 8K CPU and 16K Metal context, unsupported-request
  isolation, unload, and reload.
- `live-soak-*.json`: 20 successful requests per backend with four workers.

The measurements apply to one Apple M5 Max 64 GiB host and one model. They are
not cross-hardware, cross-model, or statistical performance guarantees.

## Rollout requirements

1. Use the pinned release build and exact model hashes; never admit unreviewed
   model substitutions.
2. Start with `magnitude_device` explicit for the target host, bounded queues,
   and a single loaded model. Tune limits only after observing memory pressure.
3. Verify cold and warm Chat, tool round-trips, JSON Schema, cancellation,
   unload, and restart against the deployed artifact before serving traffic.
4. Run a soak test with the actual model, context, concurrency, and quality
   suite. Qualify CUDA, Vulkan, larger contexts, and additional models
   separately before adding them to the support matrix.
5. Preserve the Apache-2.0 license, source manifest, and native tuning evidence.
   Do not describe this scoped backend as a complete Power v1 or TEE release.
