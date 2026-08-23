# Model-Neutral Inference Optimization Playbook

A3S Power is a general inference runtime. It does not optimize by branching on
a model name. It exposes bounded execution, tensor, scheduling, storage, and
evidence contracts that a language, vision, OCR, embedding, audio, multimodal,
scientific, or caller-owned graph integration can compose.

This playbook is the inventory of those optimization mechanisms. It also
separates portable runtime features from model-owned algorithms, backend
implementation details, host-specific controls, and experimental artifact
work. Those layers must not be presented as interchangeable switches.

## Ownership legend

| Owner | Meaning |
| --- | --- |
| **Runtime** | Model-neutral API and invariant implemented by Power. |
| **Model crate** | Shape, topology, state layout, numerical policy, or graph choice owned by an integration. |
| **Backend** | Kernel or inference-engine behavior provided by mistral.rs, llama.cpp, picolm, Candle, CUDA, or Metal. |
| **Host profile** | Machine-specific scheduling or topology setting that must be measured again on another host. |
| **Artifact profile** | A conversion or quantization decision that changes or augments model bytes. |
| **Client gate** | Acceptance policy evaluated outside the service that produced the result. |

The portable rule is simple: Power may constrain and record model-owned
decisions, but it does not infer model semantics from a filename, model family,
tensor name, sequence length, image geometry, or backend label.

## Optimization map

| Layer | Mechanisms | Owner | Portable default? |
| --- | --- | --- | --- |
| Shape and launch | Finite shape profiles, stable graph shapes, last-use tensor release, reviewed CUDA fusion | Runtime + model crate + backend | Finite bounds are portable; captured shapes and kernels are not. |
| Attention and decode | Flash Attention, full device offload, batched sampling, CUDA Graph reuse | Backend + model crate | No. Select per workload and backend. |
| Tensor path | Deterministic microbatching, leading-axis stack/split, device-resident graph chains | Runtime | Yes, within declared limits and compatible graph contracts. |
| Speculation | Prompt lookup, n-gram, native MTP, exact target verification, snapshots, rollback guard, FR projection | Runtime + adapter + artifact profile | Exact transaction is portable; strategy availability and shapes are not. |
| Prompt reuse | Explicit keyed prefix matching, authenticated namespaces, bounded KV lifetime, hit/miss/token metrics | Runtime + backend | Opt-in and fail-closed; llama.cpp text requests currently implement it. |
| Scheduling | Model/device admission, continuous batches, session replicas, cancellation, monotonic deadlines | Runtime | Yes, with caller-owned limits. |
| Host scheduling | High-priority CUDA streams, physical-core affinity, process priority, clock policy, single-service GPU use | Backend + host profile | No. Every host needs a new A/B capture. |
| Weight I/O | Verified mmap/positional/direct reads, shards, replicas, partial mirrors, encrypted and lossless sources | Runtime | Explicit and fail-closed; the selected path is workload- and storage-specific. |
| Residency | LFRU/LRU, grouped staging, prefetch, hot-set plans, hardware budgets, lossless live adaptation | Runtime + model crate | Zero-cache and no adaptation remain safe defaults. |
| Accelerator topology | Residency-bound fused batches, exact fallback, bounded heterogeneous meshes | Runtime + model crate + backend | Only after parity and device evidence pass. |
| Tuning and proof | Two-order A/B, output parity, quality gates, receipts, hardware bundles, offline hash verification | Runtime + client gate | Evidence format is portable; thresholds are deployment policy. |

## Optimization order

For repeated language-model workloads, exhaust lossless execution changes
before changing model bytes:

1. increase effective output per target pass with an exact speculative adapter;
2. eliminate repeated prefill with explicit prompt-prefix reuse; and
3. consider quantization only after the first two paths have workload-wide
   evidence.

Speculation and prefix reuse optimize different terms. Speculation targets
decode; a prefix cache targets prefill and time to first token. Quantization can
reduce bandwidth or capacity pressure, but it changes the artifact and its
benefit depends on concurrency, kernels, and the memory hierarchy.

## 1. Shape and kernel work

### Model-owned finite shape profiles

`ShapeProfileDeclaration` lets a model crate name a bounded set of optimized
shape classes with opaque SHA-256 identities. Each class binds an exact
implementation to aggregate batch, tensor-element, scratch, device, artifact,
and TEE-policy limits. Power does not inspect what the class means.

Finite classes make graph capture, kernel selection, and memory planning
repeatable. An unsupported class must either fail closed or select an explicit
dynamic implementation whose digest and fallback reason enter receipt v5.

### Stable graph shapes

CUDA Graph reuse helps only when repeated requests reach the same captured
shape. Draft width, target verification capacity, batch capacity, context
bucket, and parallel-request count can all change that shape. Increasing a
nominal acceptance rate is not a win when it creates enough graph variants to
lose launch reuse.

The Q6_K case study demonstrates this boundary:

- fixed K6/S6/B8 reached 46.923 token/s request-wide;
- an earlier unrestricted adaptive K7/S6 profile reported 50.07% acceptance
  but fell to 35.178 token/s because variable verification shapes reduced
  graph reuse;
- the current DSpark controller limits its hot proposal shapes to K6 and K10,
  reached 164.756 token/s median and 160.881 minimum on the controlled peak,
  and recorded zero replay in both that capture and the paired 100-task run;
- disabling CUDA Graphs reduced the peak workload to 133.876 token/s;
- the experimental `GGML_CUDA_GRAPH_OPT=1` path reached only 160.613 token/s.

These values describe one llama.cpp/CUDA integration. Power's generic feature
is the finite-profile and evidence contract, not a hard-coded K or B value.

### Profile-specific Flash Attention

Flash Attention reduces attention memory traffic for many context-heavy
workloads, but it is not a universal decode switch. Kernel setup and layout
cost can dominate a short hybrid target/draft batch. The current RTX 4090
profiles therefore keep Flash Attention enabled for portable long-context
profiles and disable it only for the measured short-batch K6/K7 profiles.

The decision belongs to the backend profile and must be repeated when context,
batch, quantization, driver, GPU, or backend revision changes.

### Reviewed fusion and activation liveness

Power's eager static-graph executor releases an intermediate after its last
declared consumer. That bounds live activation memory by graph dependencies
rather than total node count.

Reviewed CUDA lowerings currently cover these exact F32 patterns while
preserving explicit arithmetic boundaries and keeping the ordinary path as the
fallback:

- multiplier-one depthwise convolution;
- private `HardSigmoid` followed by gated multiplication;
- the reviewed `Div`/`Erf`/`Add`/`Mul`/`Mul` activation chain;
- convolution plus channel bias and a reviewed activation tail; and
- the final pointwise tail of decomposed last-axis LayerNorm.

Model integrations still lock eligible graph inventory and complete-output
parity. Power never rewrites an unreviewed graph merely because its nodes have
similar names.

## 2. Tensor data path

### Deterministic microbatching

Microbatch planning preserves caller order and enforces input, state, host
memory, device memory, topology, and concurrency limits before launch. One
execution permit covers the aggregate model call; the concurrency limit counts
calls rather than rows inside the call.

`TensorInput::stack_leading` and `TensorOutput::split_leading` are the canonical
leading-axis boundary. They check trailing shapes, positive partitions, finite
values, order, and shared tensor limits. Padding, bucketing, valid extents, and
slot semantics remain model-owned.

### Device-resident reviewed graph chains

`GraphExecutor::run_to_resident` and `run_resident` pass an affine
`ResidentGraphTensor` between adjacent reviewed graphs without an intermediate
host copy. The handle retains the exact request permit and a shared byte
reservation. Runtime, logical device, permit, dtype, and reviewed shape must
match.

Only the initial owned input and final materialized output are hashed. Power
does not invent a digest for device bytes it did not read and does not silently
copy across runtimes. The mechanism is useful for transformer blocks, vision
encoders, OCR pipelines, embedding stages, audio graphs, and scientific graphs;
it contains no language-model topology.

## 3. Exact speculative execution

Power provides a common speculative transaction. Backends provide compatible
proposal graphs and state access.

Available adapter families include:

- prompt lookup and n-gram context for compatible picolm paths;
- native MTP when a llama.cpp artifact exposes prediction tensors;
- verified external DFlash or DSpark GGUF artifacts for llama.cpp; the first
  native DSpark Q4 acceptance capture is published, while DFlash still awaits
  a compatible accepted artifact;
- the shared `draft-model` protocol identity, which still awaits a production
  llama.cpp adapter.

DFlash and DSpark are alternative draft contracts, not layers in one graph.
Power loads at most one external drafter, validates its declared tensor family,
and refuses a kind mismatch instead of silently relabeling the proposal path.

Every emitted token remains target-authoritative. The transaction checkpoints
target, draft, sampler, and decoder state; verifies the anchor and proposal
block; commits only the matching prefix and one correction or bonus token; and
restores the last committed state on failure or cancellation.

### Proposal width, snapshots, and target batch

`spec_draft_max` bounds proposal width. `spec_mtp_recurrent_snapshots` bounds
resident rollback state. The target batch must accommodate the anchor and
proposal rows. These values form one execution shape and must be tuned
together rather than independently.

K7/S7 keeps a resident rollback point for every proposal. K7/S6 can reduce
state work on a high-acceptance path, while the guarded implementation permits
one exact replay and then clamps later rounds to six proposals. A representative
workload, not the peak prompt alone, decides between them.

The opt-in request-local controller avoids using replay as its first feedback.
It starts at `min(K, S)`, opens the wider K shape only after a fully accepted
first probe, closes that path after a partial first round, and moves sustained
low-yield requests through a one-way target-only circuit. Healthy partial
rounds retain a captured graph shape instead of continuously resizing K. This
shared controller applies to native MTP, DFlash, and DSpark adapters even
though its compatibility ACL key remains `spec_mtp_adaptive`.

### Full vocabulary, FR, and artifact quantization

FR limits only the rows projected by an MTP draft head. It can improve a narrow
high-coverage workload and regress a broader language or domain distribution.
Full vocabulary is the balanced choice when proposal coverage is unknown.

TBQ4 is an artifact-construction strategy, not a runtime flag. Dynamic or mixed
quantization also belongs to the artifact/model layer because it changes
storage, bandwidth, kernels, and possibly output. Power can bind and compare
those artifacts but must not describe them as a lossless execution-only speedup.

## 4. Scheduling and mutable state

### One admission path

Finite model queues, physical-device queues, active permits, and unfinished
loads use cancellation-safe ownership. A monotonic deadline may cover
sequential model and device waits. Expiry releases earlier permits and records
only aggregate counters.

Device admission prevents independent models from overcommitting an
accelerator. `ExecutionBatchLifecycle` supports continuously changing batches
without importing a model loop: the model crate still chooses token policy,
row shapes, KV layout, kernels, and completion.

### Session replicas

Stateful integrations can lease a bounded number of lazy, anonymous session
replicas for one exact model and execution identity. Each lease is exclusive,
all replicas share device admission, and worst-case resident memory is reserved
before loading. A model-owned health check can retire a damaged generation
without disturbing healthy replicas.

The same contract serves KV caches, recurrent state, OCR sessions, embedding
contexts, and multimodal state. No model family appears in the scheduler.

### Keyed prompt-prefix reuse

The hosted API accepts the explicit `prompt_cache_key` extension for chat and
text completions. Power validates it, derives an opaque identity scoped by API
identity, endpoint, and model, and forwards it only when the selected backend
advertises exact prefix matching. Unsupported backends fail before model load.

The current llama.cpp path uses a per-model TTL- and capacity-bounded LRU map.
It compares token IDs, restores prompt-boundary recurrent/SWA state, clears KV
rows only across an exact backend rollback, and evaluates the suffix. A hybrid
recurrent context that cannot prove an older divergent rollback becomes a
measured miss; strict prompt extension remains the portable fast path. Health
reports support and configured bounds; Prometheus reports requests, hits,
misses, reused and evaluated tokens, evictions, and resident entries.

Native llama.cpp MTP and cached sessions do not yet share one reviewed state
transaction. Explicit MTP plus a cache key fails closed; `auto` selects exact
target-only decoding for that request. See
[Keyed Prompt-Prefix Cache](prompt-prefix-cache.md) for the API, security,
memory, and reproduction contract.

### Host-specific GPU scheduling

The pinned llama.cpp integration can request high-priority CUDA streams. The
benchmark runners can also bind a tested physical-core mask, process priority,
power scheme, clock policy, and a single loaded/concurrent model.

Those controls reduce avoidable host and WDDM variance; they do not reserve the
physical GPU. A shared Windows display GPU remains preemptible by desktop
clients. A stable service floor near a measured peak requires a quiet or
dedicated GPU rather than another undocumented environment variable.

## 5. Weight I/O and residency

All storage paths retain canonical tensor identity, dtype, shape, byte range,
and collection digest.

### Read and source paths

- mmap is the default SafeTensors path;
- buffered positional reads avoid a collection-wide mapping;
- aligned direct positional reads are explicit on supported Linux and Windows
  filesystems and never masquerade after a buffered fallback;
- macOS cache bypass is labeled honestly and does not claim a proven cold
  cache;
- one logical collection may span verified disjoint shard roots;
- complete and partial read-only replicas route only tensors they contain and
  fall back to the primary on recoverable failure;
- seekable encrypted sources decrypt only covering authenticated chunks into
  zeroizing buffers; and
- the typed lossless rANS replica decodes bounded records and byte-compares
  them with the canonical primary before routing.

Validation-throughput routing can derive bounded source weights from work that
the integrity pass already had to perform. It does not run an extra hidden
probe.

### Cache, prefetch, and staged execution

The weight hierarchy spans storage, host memory, and accelerator memory. The
default LFRU policy combines bounded frequency with recency and heat decay;
plain LRU remains available.

Power supports:

- explicit and residency-plan pins with separate provenance;
- de-duplicated, bounded asynchronous prefetch;
- per-key load serialization so demand and prefetch cannot materialize the
  same tensor twice;
- ordered current-layer grouped staging;
- union loading of repeated routed experts across a batch;
- evaluated cross-layer route hints that report predicted, actual, and matched
  selections; and
- telemetry that distinguishes useful prefetched bytes from unused evicted
  bytes.

### Plans, budgets, and mirrors

`plan_residency` selects whole model-owned atomic groups under exact host,
device, byte, and per-layer-entry budgets. Hardware-aware budgeting subtracts
caller-declared fixed and peak-scratch reservations from native memory
snapshots before assigning cache capacity.

Live residency adaptation uses hysteresis, bounded replacements, equal-footprint
groups, and a model-owned safe boundary. It changes tier only; it cannot change
weights, dtype, gate mass, arithmetic, or receipts.

Usage-ranked partial mirrors copy deterministic whole-file subsets to a
caller-authorized faster tier, re-hash each file, and publish atomically. They
extend the same `WeightStore`; they are not a second cache or integrity path.

## 6. Accelerator batches and device meshes

Residency-bound fused batches acquire only groups already pinned by the active
plan. An unavailable reviewed kernel may select only its digest-declared exact
fallback. Arithmetic and policy errors remain failures.

Heterogeneous meshes bound device count, directed transfer sizes, launch count,
and aggregate traffic. Power supplies the guarded transfer substrate and
evidence; the model crate owns partitioning, activation movement, reduction
order, and the decision that peer traffic is worthwhile. A single-device mesh
is valid for parity before a peer edge is introduced.

## 7. Measure before promotion

`evaluate_tuning_profile` accepts only digest-bound, output-identical evidence.
Every candidate runs in both baseline-to-candidate and candidate-to-baseline
order for at least two rounds. Selection uses the lower order-specific median,
applies p99 and cache-hit regression limits, and retains the baseline on an
exact tie.

The tensor-batch benchmark adds isolated host-allocation and boundary-copy
measurements. Hardware evidence bundles combine storage reports, tuning
evidence, model parity artifacts, exact source revision, typed device, and
named environment. Clients pin the resulting digest and decide whether it is
acceptable.

For language generation, publish both steady decode and request-wide
throughput. Pair speed with output identity, truncation, task score, acceptance,
verified tokens per target pass, fallback/replay counts, environment receipts,
and idle-device contention. A peak prompt is not a service-level objective.

## Backend and workload boundary

| Path | Typical workload | Formats and devices | Optimization ownership |
| --- | --- | --- | --- |
| `mistralrs` | Language, vision, embedding | GGUF and SafeTensors through the backend | Backend owns graph and kernels; Power owns service, admission, artifacts, and evidence. |
| `llamacpp` / `llamacpp-cuda` | GGUF language and multimodal models | CPU or CUDA | llama.cpp owns kernels, KV layout, Flash Attention, CUDA Graphs, and native MTP; Power owns policy, lifecycle, scheduling controls, and receipts. |
| `picolm` | Pure-Rust, memory-constrained or TEE language inference | Layer-streamed GGUF on CPU | picolm owns transformer arithmetic; Power owns bounded loading, service, and evidence. |
| `embedded-inference` | Vision, OCR, embedding, audio, scientific, and custom reviewed graphs | SafeTensors with typed CPU/CUDA/Metal devices | Model crate owns graph semantics; Power supplies the generic execution and optimization substrate. |

Adding a new architecture means adding or selecting an adapter/model crate. It
does not mean adding a model-name branch to the scheduler.

## Current Q6_K case study

The checked-in RTX 4090 integration keeps the exact 22,884,408,288-byte Q6_K
artifact with SHA-256
`562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727`.

- peak profile: fixed K7/S6/B11, prefix-FR8192, short-batch Flash Attention off,
  normal CUDA Graphs, high-priority CUDA streams, physical-core affinity, one
  loaded model, and one concurrent request;
- clean contended-host result: 172.835 token/s median, 171.298 minimum, 175.533
  maximum across nine measured 1,024-token requests;
- earlier quiet-host high-water mark: 176.611 token/s median;
- general profile: fixed K6/S6/B8, 46.923 token/s request-wide versus 28.713
  token/s target-only, a 63.42% gain;
- paired calibration: 12/12 final answers and the 9/12 score matched, 8/12 full
  content digests matched, proposal acceptance was 26.81%, verified tokens per
  target pass were 2.591, and fallback replay was zero.

The 12-task calibration is not a general-intelligence proof. The archived
100-task matrices remain the broader quality evidence. The complete raw data,
environment receipts, rejected candidates, and reproduction commands are in
[the RTX 4090 benchmark record](benchmarks/qwen3.8-27b-q6k-rtx4090/README.md).

## Selection procedure

1. Freeze model bytes, backend revision, prompt or task set, device identity,
   and output policy.
2. Measure the target-only baseline request-wide and at steady state.
3. Remove avoidable host/device boundaries before changing numerical format.
4. Choose a small finite shape set and verify capture reuse.
5. Tune backend kernels, attention, speculation, and host scheduling one layer
   at a time with reverse-order A/B runs.
6. Reject any candidate that changes required output, quality, fallback,
   memory, or tail-latency bounds.
7. Archive the raw report, environment/preflight receipt, hashes, and replay
   command before promoting a profile.

This procedure keeps Power general: the runtime standardizes how an
optimization is bounded and proved, while each model integration remains free
to supply the graph and algorithms that fit its workload.
