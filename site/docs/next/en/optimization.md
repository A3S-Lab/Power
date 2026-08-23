---
title: Optimization playbook
description: The complete model-neutral optimization map for A3S Power, from graph shapes and tensor residency to scheduling, storage, speculation, and evidence.
---

# Optimization playbook

Power does not ship one magic fast mode. It separates portable runtime
mechanisms from model-owned algorithms, backend kernels, host controls, and
artifact conversion. That separation lets the same runtime serve language,
vision, OCR, embedding, audio, multimodal, scientific, and caller-owned graphs.

## Read the ownership label first

| Label | Owns the decision |
| --- | --- |
| Runtime | Power implements a model-neutral bound or execution mechanism. |
| Model crate | The integration owns topology, shape meaning, state layout, arithmetic, and quality policy. |
| Backend | mistral.rs, llama.cpp, picolm, Candle, CUDA, or Metal owns the kernel path. |
| Host profile | A machine-specific setting that must be measured again on another host. |
| Artifact profile | Conversion or quantization that changes or augments model bytes. |
| Client gate | Acceptance policy outside the service that produced the result. |

Power may constrain and record model-owned choices. It never derives model
semantics from a filename, model family, tensor name, sequence length, image
geometry, or backend label.

## Complete optimization map

| Execution layer | Available mechanisms | Ownership |
| --- | --- | --- |
| Shapes and launch | Finite shape profiles, stable graph shapes, last-use tensor release, reviewed CUDA fusion | Runtime + model crate + backend |
| Attention and decode | Flash Attention, full device offload, batched GPU sampling, CUDA Graph reuse | Backend + model crate |
| Tensor path | Deterministic microbatching, leading-axis stack/split, device-resident reviewed graph chains | Runtime |
| Speculation | Prompt lookup, n-gram, native MTP, exact target verification, snapshots, rollback guard, FR projection | Runtime + adapter + artifact profile |
| Prompt reuse | Explicit cache keys, authenticated namespaces, bounded KV lifetime, hit/miss/token metrics | Runtime + backend |
| Scheduling | Model/device admission, continuous batches, session replicas, cancellation, monotonic deadlines | Runtime |
| Host scheduling | CUDA stream priority, physical-core affinity, process priority, clock policy, single-service GPU use | Backend + host profile |
| Weight I/O | Verified mmap, positional and direct reads, shards, replicas, encrypted/lossless sources, partial mirrors | Runtime |
| Residency | LFRU/LRU, prefetch, ordered staging, hot-set plans, hardware budgets, live adaptation | Runtime + model crate |
| Accelerator topology | Residency-bound fused batches, exact fallback, bounded multi-device meshes | Runtime + model crate + backend |
| Proof | Two-order A/B, output parity, quality gates, receipts, hardware bundles, offline hash verification | Runtime + client gate |

These mechanisms are composable, not automatically enabled together.

## Tune in lossless-first order

For Agent, RAG, and coding workloads, first improve generated tokens per target
pass with exact speculation, then remove repeated prefill with keyed prefix
reuse. Consider quantization only after both paths have workload-wide evidence.
Speculation targets decode; prefix reuse targets prefill and TTFT. Quantization
changes the artifact and its value depends on concurrency, kernels, and memory
pressure.

## Graph shapes and kernels

Model crates declare a small set of opaque optimized shape identities. Power
checks aggregate batch, tensor, scratch, device, artifact, and security bounds.
Unsupported classes fail closed or select a digest-bound dynamic path.

Stable shapes matter because graph capture is only reusable when later work
matches the captured geometry. Draft width, target verification capacity,
batch, context bucket, and parallel-request count can all create another graph.
An adaptive setting that improves proposal acceptance can still lose overall
throughput by fragmenting graph reuse.

Power's opt-in request-local controller therefore starts inside the rollback
window and exposes only a small hot-shape set. It opens the wider shape after a
fully accepted first probe, closes that path after a partial first round, and
moves sustained low-yield work through a one-way target-only circuit. The
current DSpark capture kept K6/K10 reusable shapes, cleared 160 token/s in all
three peak samples, and recorded zero replay; its paired losses still prevent
it from becoming the default.

Flash Attention is profile-specific. It usually helps context-heavy attention,
but setup and layout work can lose on short target/draft batches. The model and
backend profile owns that decision.

Power also releases static-graph intermediates after their last consumer. Its
reviewed CUDA lowerings cover exact depthwise convolution, gated HardSigmoid,
error-function activation, convolution/bias/activation, and LayerNorm-tail
patterns. Unreviewed shapes and dtypes stay on the ordinary path.

## Tensor path

Deterministic microbatching preserves caller order and checks point-in-time host
and device memory before launch. Canonical leading-axis stack/split helpers
validate trailing shapes, partitions, finite values, order, and tensor limits.

Adjacent reviewed graphs can pass an affine `ResidentGraphTensor` without an
intermediate host copy. Runtime, logical device, request permit, dtype, and
reviewed shape must match. Power hashes the initial input and final materialized
output; it does not invent an intermediate digest or silently copy between
runtimes.

## Exact speculation

Prompt lookup, n-gram context, native MTP, DFlash, and DSpark adapters share
one transaction: checkpoint, propose, target-verify, commit the matching
prefix, emit one correction or bonus token, and restore the last commit on
failure.

- `spec_draft_max` bounds proposal width.
- `spec_mtp_recurrent_snapshots` bounds resident rollback state.
- The target batch must hold the anchor and proposal rows.
- FR limits only draft-head projection rows and is workload-sensitive.
- TBQ4 and dynamic/mixed quantization are artifact choices, not generic runtime
  switches.
- The request-local width controller is shared by native MTP, DFlash, and
  DSpark adapters; `spec_mtp_adaptive` is a compatibility name, not a
  Qwen-specific implementation.
- DFlash2 has a separate selector/convolution artifact contract. Native Power
  reached 144.453 token/s median decode versus 33.075 target-only on the high-
  acceptance prompt; the fixed 12-task workload reached 45.143 versus 29.702
  token/s and kept only 7/12 complete-output parity, so the mode remains opt-in.

Acceptance rate alone is not the objective. Measure emitted tokens per target
pass against draft, target verification, synchronization, sampling, replay,
and graph-shape cost.

## Scheduling and replicas

One cancellation-safe admission path covers model queues, physical devices,
active requests, and unfinished loads. A monotonic deadline can span sequential
waits without exposing request timing or identity.

Continuous batches admit new members only at the next execution step and commit
each step atomically. Exclusive session-replica leases serve KV caches,
recurrent state, OCR contexts, embedding sessions, and multimodal state without
a model-family branch.

The hosted API also accepts an explicit `prompt_cache_key`. Power validates and
hash-scopes it by authentication, endpoint, and model, then forwards it only to
a backend that advertises exact prefix reuse. llama.cpp text requests currently
use a per-model bounded LRU/TTL context map and turn an unprovable recurrent
rollback into a measured miss; other paths fail closed. Native MTP
and cached sessions do not compose yet. See the
[canonical cache contract](https://github.com/A3S-Lab/Power/blob/main/docs/prompt-prefix-cache.md).

High-priority CUDA streams, physical-core affinity, process priority, power and
clock policy, and a single loaded/concurrent model are host profiles. They can
reduce variance but do not physically reserve a shared WDDM display GPU.

## Weight I/O and residency

All paths preserve canonical tensor identity, dtype, shape, byte range, and
collection digest:

- verified mmap, buffered positional, aligned direct, and honest macOS
  cache-bypass reads;
- disjoint shard roots and complete or partial read-only replicas;
- seekable authenticated encryption and typed lossless rANS replicas;
- validation-throughput source weighting without a hidden extra probe;
- LFRU or LRU caching with separate manual and plan pin provenance;
- bounded deduplicated prefetch, per-key load serialization, and ordered
  current-layer staging;
- repeated-expert union loads and evaluated cross-layer route hints;
- hardware-aware residency budgets, whole-group plans, hysteretic live
  adaptation, and usage-ranked partial mirrors.

The zero-cache, no-adaptation path remains the safe default. Model crates decide
atomic groups, safe adaptation boundaries, and whether a prefetch hint is worth
using.

## Fused accelerator batches and meshes

Fused batches acquire only groups already pinned by the active residency plan.
An unavailable kernel may select only its declared exact fallback; arithmetic
or policy errors remain failures.

Heterogeneous meshes bound device count, directed transfer size, launches, and
aggregate traffic. Power supplies guarded transfers and evidence. The model
crate owns partitioning, activation movement, reduction order, and the decision
that peer traffic is cheaper than local execution.

## Evidence is part of optimization

Candidate settings run in baseline→candidate and candidate→baseline order for
at least two rounds. Selection uses the lower order-specific median, enforces
p99 and cache-hit regression limits, requires identical output, and retains the
baseline on an exact tie.

Publish request-wide throughput beside steady decode. Bind speed to output
identity, quality, truncation, proposal acceptance, verified tokens per target
pass, replay, memory, device path, environment, and source revision. A peak
prompt is not a service-level objective.

## Backend boundary

| Path | Intended use | Who owns model execution |
| --- | --- | --- |
| `mistralrs` | GGUF/SafeTensors language, vision, and embedding service | mistral.rs |
| `llamacpp` / `llamacpp-cuda` | Mature GGUF execution and native MTP | llama.cpp |
| `picolm` | Pure-Rust layer-streamed GGUF for constrained or TEE memory | picolm |
| `embedded-inference` | Vision, OCR, embedding, audio, scientific, and custom reviewed graphs | The model crate |

Power owns the common artifact, admission, device, scheduling, state, and
evidence layer. Adding a model means adding or selecting an adapter; it does not
mean adding a model-name branch to the scheduler.

## What the current Q6_K case proves

The checked-in RTX 4090 integration keeps the exact Q6_K bytes unchanged.

- K7/S6/B11 peak profile: 174.413 token/s median, 172.723 minimum, and
  177.150 maximum across nine exact-build samples.
- Earlier quiet-host high-water mark: 176.611 token/s median.
- Active full-vocabulary 3x100 quality profile: 41.035 token/s request-wide
  versus 23.642 target-only, a 73.57% gain.
- The active profile kept its 67/100 lenient score but moved from 60/100 to
  58/100 strict, so it remains opt-in rather than a lossless default.

The peak profile combines fixed graph shapes, short-batch Flash Attention off,
normal CUDA Graphs, exact MTP verification, a measured target batch,
high-priority streams, physical-core affinity, and single-service scheduling.
It is one backend case study, not a universal default or a general-intelligence
claim.

Read [Performance evidence](./performance) for the measurements and
[Reproduction](./reproduction) for the exact commands. The
[canonical playbook](https://github.com/A3S-Lab/Power/blob/main/docs/optimization-playbook.md)
contains the API-level boundaries and source links.
