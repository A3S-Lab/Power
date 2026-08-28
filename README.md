# A3S Power

<p align="center">
  <a href="https://a3s-lab.github.io/Power/"><img src="./site/docs/public/a3s-os-logo.png" width="72" alt="A3S OS"></a>
</p>

<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="A3S Power routes model-owned graphs and hosted API requests through bounded admission, accelerator execution, canonical receipts, and caller-owned verification">
</p>

<p align="center">
  <a href="https://github.com/A3S-Lab/Power/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Power/ci.yml?branch=main&amp;style=flat-square&amp;label=CI" alt="CI status"></a>
  <a href="https://github.com/A3S-Lab/Power/actions/workflows/pages.yml"><img src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Power/pages.yml?branch=main&amp;style=flat-square&amp;label=docs" alt="Documentation deployment status"></a>
  <a href="https://a3s-lab.github.io/Power/"><img src="https://img.shields.io/badge/docs-ZH-2864e8?style=flat-square" alt="A3S Power Chinese documentation"></a>
  <a href="https://a3s-lab.github.io/Power/en/"><img src="https://img.shields.io/badge/docs-EN-2864e8?style=flat-square" alt="A3S Power English documentation"></a>
  <a href="https://crates.io/crates/a3s-power"><img src="https://img.shields.io/crates/v/a3s-power?style=flat-square&amp;color=2864e8" alt="a3s-power on crates.io"></a>
  <a href="https://docs.rs/a3s-power"><img src="https://img.shields.io/docsrs/a3s-power?style=flat-square" alt="a3s-power API documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-17181a?style=flat-square" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#measured-not-promised">Evidence</a> &middot;
  <a href="#choose-the-boundary">Surfaces</a> &middot;
  <a href="#quick-start">Quick start</a> &middot;
  <a href="#one-runtime-contract">Architecture</a> &middot;
  <a href="#optimization-without-hidden-shortcuts">Optimization</a> &middot;
  <a href="#verification-and-release-gates">Verification</a> &middot;
  <a href="https://a3s-lab.github.io/Power/">Documentation</a>
</p>

A3S Power is a model-neutral Rust execution layer for inference. Model crates
keep their topology, tokenizer, preprocessing, mutable state, and quality
policy. Power supplies the shared boundary around them: artifact identity,
device placement, admission, cancellation, bounded state, execution receipts,
and independent verification.

The same contracts support a listener-free embedded library, an
OpenAI-compatible service, verified artifact provisioning, and a minimal
layer-streaming TEE profile. The runtime core does not dispatch on Qwen or any
other model family.

> [!IMPORTANT]
> `main` contains the v1.0.0 source candidate. The latest published crate and
> API documentation are still
> [v0.9.0](https://crates.io/crates/a3s-power/0.9.0); use the source-based
> commands below for current v1 APIs. A v1 tag is not published until the
> strict four-platform evidence bundle and verified annotated tag both pass.

## Measured, not promised

Power records performance through its real streaming API and publishes the
inputs, raw samples, environment receipts, output identities, and offline
verifiers. The Qwen3.8-27B results below are one pinned llama.cpp/CUDA
integration on Windows 11, an RTX 4090, and an Intel Xeon w5-2445. They are
evidence for those exact workloads, not the scope of the engine or a service
SLA.

The target in every active row is the same untouched 22,884,408,288-byte Q6_K
artifact. A Q4 file appears only as an auxiliary proposer where stated.

| Path | Measured result | Acceptance boundary |
| --- | ---: | --- |
| Q6_K autoregressive control | 23.642 request-wide token/s | 3 x 100 fixed tasks; 67/100 lenient and 60/100 strict |
| Q6_K full-vocabulary MTP | **41.035 request-wide token/s; 1.736x** | Same 3 x 100 tasks and same target bytes; 67/100 lenient, 58/100 strict; opt-in |
| Q6_K MTP/FR peak shape | **174.413 token/s median; 172.723 minimum** | Nine 1,024-token runs with one output digest; not a stable 175 token/s floor |
| Target-only prefix reuse | **23.5299x backend prefill; 13.1593x TTFT** | Five cold/warm pairs reusing 9,740 prompt tokens; repeated-context latency only |
| Q6_K + DFlash2 proposer | **144.453 decode; 63.182 end-to-end token/s** | Five exact synthetic pairs; broader 12-task run kept 12/12 answers but only 7/12 complete outputs |

These rows use different workload shapes and must not be compared as if they
were one benchmark. Fixed-task scores are quality proxies, not intelligence
measurements. Exact target verification proves target authority for committed
tokens; it does not promise byte-identical prose.

[Benchmark index](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/README.md) ·
[Q6_K-only offline evidence](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/pure-q6-rtx4090-3x.evidence.json) ·
[Exact reproduction](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md) ·
[Performance documentation](https://a3s-lab.github.io/Power/en/performance)

## Choose the boundary

Pick the narrowest surface that fits the product.

| Surface | Use it when | Network behavior |
| --- | --- | --- |
| **Embedded runtime** | A Rust model crate owns a reviewed graph and needs shared devices, scheduling, state, and receipts. | No listener, model hub, download, or child process. |
| **Hosted service** | Existing clients need chat, completions, embeddings, model lifecycle, metrics, or attestation. | Explicit HTTP, RA-TLS, or vsock transport. |
| **Artifact provisioner** | A product installs an exact revision-locked model bundle. | Policy-gated download; verified blobs remain reusable offline. |
| **Minimal TEE service** | A constrained enclave needs a pure-Rust, layer-streaming GGUF path. | Transport remains an explicit feature choice. |

## Quick start

### Run the current hosted service

Install directly from `main` while v1 remains a release candidate:

~~~bash
cargo install --git https://github.com/A3S-Lab/Power.git --locked a3s-power
a3s-power serve --host 127.0.0.1 --port 11434
~~~

In another terminal:

~~~bash
a3s-power models pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
a3s-power chat Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
~~~

For the published v0.9.0 CLI instead:

~~~bash
cargo install a3s-power --version 0.9.0 --locked
~~~

Models and content-addressed blobs live under `~/.a3s/power` by default. Set
`A3S_POWER_HOME` to move the store.

### Embed the current runtime

~~~toml
[dependencies]
a3s-power = { git = "https://github.com/A3S-Lab/Power.git", branch = "main", default-features = false, features = ["embedded-inference"] }
~~~

~~~rust
use a3s_power::inference::{DevicePreference, EmbeddedRuntime, InferenceLimits};

fn main() -> Result<(), a3s_power::error::PowerError> {
    let runtime = EmbeddedRuntime::new(
        DevicePreference::Auto,
        InferenceLimits::default(),
    )?;

    println!("execution device: {}", runtime.device().name());
    Ok(())
}
~~~

Constructing `EmbeddedRuntime` never opens a listener or downloads a model.
The caller supplies a reviewed graph and retains semantic state; Power supplies
the bounded execution and evidence contracts.

### Send an OpenAI-compatible request

~~~bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "Explain capability-based security."}],
    "prompt_cache_key": "shared-agent-prefix-v1",
    "stream": true
  }'
~~~

Streaming responses emit the execution receipt before `[DONE]`.

## One runtime contract

~~~text
model-owned graph                   OpenAI-compatible client
        |                                      |
   embedded API                           hosted service
        \                                      /
         +--------- shared Power core --------+
                            |
           artifact identity + bounded admission
                            |
              placement + reviewed execution
                            |
                   CPU / CUDA / Metal
                            |
                  canonical receipt
                            |
               independent verification
~~~

| Power owns | Model-owning crates own |
| --- | --- |
| Typed devices, bounded graph execution, opaque resident tensors, and aggregate memory budgets | Architecture, topology, layers, kernels, and arithmetic |
| Admission, microbatching, session replicas, cancellation, deadlines, and placement | Tokenization, preprocessing, postprocessing, and generation policy |
| Artifact identities, mirrors, residency plans, and execution receipts | Asset revisions, conversion, tensor contracts, and quality gates |
| TEE privacy, attestation binding, sealed state, and verifier inputs | KV or recurrent layout and semantic mutable state |

This split is deliberate. Shape-profile names, model-family strings, and
session identities are opaque to the core. Language, vision, OCR, embedding,
audio, multimodal, scientific, and caller-owned graphs can share the runtime
without adding architecture dispatch to Power.

### Three first principles

| Constraint | Runtime consequence |
| --- | --- |
| Memory, compute, queues, and transfers are finite. | Requests enter through explicit limits, share one physical-device gate, remain cancellable, and cannot silently overcommit declared state. |
| A model name does not identify an execution. | Receipts bind artifact, policy, device path, input, output, and fallback identities with canonical digests. |
| A server cannot be the trust root for its own claims. | The client chooses acceptable measurements, hashes, runtime policy, GPU evidence, and receipt fields. |

Read [Embedded Inference Architecture](docs/embedded-inference-architecture.md)
for the complete contract model.

## Optimization without hidden shortcuts

Power does not expose one vague “fast mode.” Each layer has its own owner,
measurement, fallback, and receipt identity.

| Layer | Mechanism | Rule |
| --- | --- | --- |
| Graphs and kernels | Finite shape profiles, CUDA Graph reuse, profile-specific Flash Attention, and reviewed fused paths | The model defines shape meaning; missing or oversized shapes fail or use an explicit digest-bound fallback. |
| Tensor movement | Deterministic microbatching, execution batches, device-resident graph chains, and one final materialization | Incompatible devices never trigger a hidden cross-device copy. |
| Speculative decoding | Prompt lookup, n-gram, draft model, MTP, DFlash, DFlash2, or DSpark with exact target verification | Only accepted tokens commit; unsupported explicit strategies fail closed. |
| Prefix reuse | Tenant-, endpoint-, and model-scoped KV/recurrent contexts with bounded LRU and TTL | `prompt_cache_key` is explicit; unsupported backends return an error instead of ignoring it. |
| Distributed state transfer | Typed prepare/publish/consume/abort port with exact model, execution, layout, epoch, size, protocol, expiry, and receipt binding | The default server advertises no P/D capability; a reviewed injected adapter must own registered memory, transport integrity, and cleanup. |
| Scheduling | Shared device admission, bounded queues, cancellation, deadlines, session replicas, and host controls | Replica declarations reserve their full resident budget before loading. |
| Weights and storage | Content-addressed artifacts, mmap/mlock policy, verified mirrors, prefetch, and bounded residency | Fallback returns to the original artifact without changing tensor identity. |
| Rollout | Two-order A/B runs, output hashes, representative quality gates, hardware receipts, and offline replay | A faster profile does not become the default until its acceptance policy passes. |

Native MTP and keyed llama.cpp prompt-cache sessions do not compose yet:
explicit MTP fails closed for keyed requests, while `auto` selects target-only.
DFlash, DFlash2, and DSpark are alternative external-draft contracts, not
stackable modes.

[Optimization playbook](docs/optimization-playbook.md) ·
[Speculative decoding](docs/speculative-decoding.md) ·
[Prompt-prefix cache](docs/prompt-prefix-cache.md) ·
[Shape profiles](docs/shape-profiles.md) ·
[Session replicas](docs/session-replicas.md) ·
[Device-resident graphs](docs/device-resident-graphs.md)

## Backends are capabilities

Backends implement model-specific work behind the same resource and evidence
contracts. They do not define the architecture of the Power core.

| Feature | Capability | Native requirement |
| --- | --- | --- |
| `mistralrs` | Default GGUF, SafeTensors, vision, and embedding backend | No C++ inference engine |
| `llamacpp` | Mature GGUF backend with native MTP support | CMake, C++ compiler, and libclang |
| `llamacpp-cuda` | CUDA execution for llama.cpp | CUDA toolkit |
| `llamacpp-external-draft` | Typed DFlash, DFlash2, and DSpark artifact contracts | Reviewed patches for the pinned llama.cpp source |
| `llamacpp-mtp-fr` | Experimental reduced-vocabulary draft projection | Reviewed patch for the pinned llama.cpp source |
| `picolm` | Pure-Rust layer-streaming GGUF for constrained TEE memory | No C or C++ inference engine |
| `embedded-cuda` / `embedded-metal` | Accelerators for model-owned embedded graphs | Platform toolkit |
| `tls` / `vsock` / `hw-verify` | RA-TLS, guest-host transport, and AMD SEV-SNP verification | Platform-specific dependencies and trust roots |

<details>
<summary>Build profiles</summary>

~~~bash
# Default hosted service
cargo build --locked --release

# Listener-free embedded runtime
cargo build --locked --release --no-default-features --features embedded-inference

# Pure-Rust layer-streaming TEE service
cargo build --locked --release --no-default-features --features tee-minimal

# llama.cpp with CUDA
cargo build --locked --release --no-default-features --features llamacpp-cuda

# Strict verifier and release-promotion path
cargo build --locked --release --no-default-features \
  --features server,embedded-inference,hw-verify \
  --bin a3s-power-verify
~~~

On Windows, enable Git long paths before the first pinned llama.cpp checkout:

~~~powershell
git config --global core.longpaths true
~~~

</details>

## Policy is explicit

The service reads A3S ACL from `~/.a3s/power/config.acl` or from the path
passed to `a3s-power serve --config`.

~~~acl
host = "127.0.0.1"
port = 11434
max_loaded_models = 1
prompt_cache_max_entries = 1
prompt_cache_ttl_seconds = 300
worker_observation_ttl_seconds = 15
keep_alive = "5m"

flash_attention = true
num_parallel = 1

gpu {
  gpu_layers = -1
  main_gpu = 0
}
~~~

Invalid ACL, ranges, hashes, strategies, or unsupported explicit backends fail
before inference. TEE deployments add verifier-owned model hashes,
measurements, and strict policy; simulated attestation never passes strict
verification.

## API surface

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Readiness, loaded models, TEE status, and the versioned worker observation |
| `POST` | `/v1/chat/completions` | Chat, tools, structured output, vision, and SSE |
| `POST` | `/v1/completions` | Text completion and SSE |
| `POST` | `/v1/embeddings` | Embedding inference |
| `GET` | `/v1/models` | Registered models |
| `POST` | `/v1/models` | Register weights and typed auxiliary artifacts |
| `POST` | `/v1/models/pull` | Resumable ModelScope or Hugging Face pull |
| `GET` | `/v1/attestation` | Nonce- and model-bound TEE evidence |
| `GET` | `/metrics` | Prometheus metrics |

GGUF registration accepts typed adapter, projector, and external-draft objects.
Power measures the files itself and records exact lengths and SHA-256
identities; strict TEE startup rejects legacy path-only auxiliary artifacts.

## Verification and release gates

~~~text
model bytes + resolved runtime policy
                  |
        canonical claims + fresh nonce
                  |
       CPU TEE report + GPU evidence
                  |
       request + prompt + output receipt
                  |
       independent client accepts or rejects
~~~

~~~bash
a3s-power-verify \
  --url http://127.0.0.1:11434 \
  --nonce <client-nonce-hex> \
  --model-hash <artifact-sha256-hex> \
  --inference-execution-digest <resolved-policy-sha256> \
  --auxiliary-artifacts-digest <portable-auxiliary-set-sha256> \
  --expected-measurement <launch-measurement-hex>
~~~

Power verifies AMD SEV-SNP signatures, binds policy-visible fields to the exact
signed report, checks nonce freshness and RA-TLS binding, and can verify
NVIDIA GPU/NVSwitch evidence and NRAS verdicts. Intel TDX currently emits a
local TDREPORT but fails strict verification until a reviewed DCAP Quote/QVL
path exists.

The strict v1 release policy requires four distinct captures: CPU, CUDA,
native Apple Silicon Metal, and proof-promoted SEV-SNP/NVIDIA confidential
GPU. The tag must point to an evidence-only child of the frozen source, and it
must be an annotated signature that GitHub verifies. Hosted or virtual Metal,
local CUDA relabeled as confidential, mixed source/evidence commits, and
lightweight or unverified tags all fail closed.

[Production release gate](docs/release-evidence-gate.md) ·
[v1 support matrix](docs/v1-support-matrix.md) ·
[External hardware capture](docs/external-release-capture.md) ·
[Hardware verifier operations](docs/hardware-verifier-operations.md)

### Security boundaries

- Simulated TEE mode is development-only.
- CPU TEE placement does not make ordinary GPU offload confidential.
- Effective-prompt digests exist only for deterministic text paths.
- Reduced-vocabulary FR and auxiliary proposers are performance techniques,
  not universal quality guarantees.
- Performance evidence from one model, host, or workload does not transfer to
  another without a new capture.

## Documentation

| Start here | What it answers |
| --- | --- |
| [Documentation home - Chinese](https://a3s-lab.github.io/Power/) | Default `next` documentation |
| [Documentation home - English](https://a3s-lab.github.io/Power/en/) | English `next` documentation |
| [Getting started](https://a3s-lab.github.io/Power/en/getting-started) | Installation, embedded use, service use, and first request |
| [Architecture](https://a3s-lab.github.io/Power/en/architecture) | Ownership, execution, state, and receipt boundaries |
| [Optimization](https://a3s-lab.github.io/Power/en/optimization) | Graph, tensor, speculation, scheduling, storage, and evidence methods |
| [Performance](https://a3s-lab.github.io/Power/en/performance) | Current measurements, quality boundaries, and limitations |
| [Reproduction](https://a3s-lab.github.io/Power/en/reproduction) | Offline verification and full hardware replay |
| [Verification](https://a3s-lab.github.io/Power/en/verification) | Attestation claims, client policy, and failure behavior |
| [Operations](https://a3s-lab.github.io/Power/en/operations) | Feature profiles and deployment controls |

The repository also keeps the detailed contracts close to the code:

- [Model-owned shape profiles](docs/shape-profiles.md)
- [Model-neutral session replicas](docs/session-replicas.md)
- [Device-resident reviewed graph chains](docs/device-resident-graphs.md)
- [Keyed prompt-prefix cache](docs/prompt-prefix-cache.md)
- [Distributed-serving worker observation](docs/distributed-serving-observation.md)
- [Model-neutral speculative decoding](docs/speculative-decoding.md)
- [Supply-chain audit](docs/supply-chain.md)
- [Storage benchmark](docs/storage-benchmark.md)
- [Tensor-batch cost benchmark](docs/tensor-batch-benchmark.md)
- [Roadmap](ROADMAP.md) and [changelog](CHANGELOG.md)

The checked-in `site/docs/v1.0.0` tree is a candidate documentation snapshot,
not evidence that the v1 tag has already passed the production gate.

## Development

Run checks from this crate, not from the monorepo root:

~~~bash
cargo fmt --all -- --check
cargo test --locked --lib
cargo test --locked --no-default-features --features embedded-inference --lib
cargo test --locked --no-default-features --features picolm --lib
cargo clippy --locked --all-targets -- -D warnings

npm ci --prefix site
npm run typecheck --prefix site
npm run build --prefix site
npm run check:site --prefix site
~~~

CI checks formatting, Clippy feature matrices, focused and integration tests,
listener-free embedded builds, release contracts, benchmark evidence, and the
GitHub Pages artifact.

## A3S ecosystem

| Project | Relationship |
| --- | --- |
| [A3S Box](https://github.com/A3S-Lab/Box) | Hosts Power inside SEV-SNP or TDX MicroVMs |
| [A3S Gateway](https://github.com/A3S-Lab/Gateway) | Routes inference traffic |
| [A3S Runtime](https://github.com/A3S-Lab/Runtime) | Supplies deployment-unit contracts |
| [A3S Code](https://github.com/A3S-Lab/Code) | Consumes local inference and verified bundles |
| [A3S Event](https://github.com/A3S-Lab/Event) | Distributes platform events |

Questions and design discussions are welcome on
[Discord](https://discord.gg/XVg6Hu6H).

## License

[MIT](LICENSE)
