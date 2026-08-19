# A3S Power

<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="A3S Power joins an embedded Rust inference library and a TEE-aware OpenAI-compatible service through shared execution and verification contracts">
</p>

<p align="center">
  <strong>Model-neutral Rust inference with bounded execution, canonical receipts, and hardware-backed verification.</strong>
</p>

<p align="center">
  <a href="https://github.com/A3S-Lab/Power/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Power/ci.yml?branch=main&amp;style=flat-square&amp;label=CI" alt="CI status"></a>
  <a href="https://github.com/A3S-Lab/Power/actions/workflows/release.yml"><img src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Power/release.yml?style=flat-square&amp;label=release" alt="Release status"></a>
  <a href="https://crates.io/crates/a3s-power"><img src="https://img.shields.io/crates/v/a3s-power?style=flat-square&amp;color=2563eb" alt="a3s-power on crates.io"></a>
  <a href="https://docs.rs/a3s-power"><img src="https://img.shields.io/docsrs/a3s-power?style=flat-square" alt="a3s-power documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-17181a?style=flat-square" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#measured-proof">Measured proof</a> &middot;
  <a href="#choose-a-surface">Surfaces</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#quick-start">Quick start</a> &middot;
  <a href="#verifiable-execution">Verification</a> &middot;
  <a href="#documentation">Documentation</a>
</p>

A3S Power is a Rust inference foundation with two inference entry points: a
listener-free embedded library for model-owning crates and an optional
OpenAI-compatible service for hosted LLMs. A third, independent surface safely
provisions consumer-owned, revision-locked artifact bundles.

All three reuse explicit integrity and resource contracts. Power owns devices,
admission, placement, cancellation, privacy, and evidence; model repositories
continue to own topology, tokenization, preprocessing, state semantics, and
quality policy.

## Measured proof

The repository publishes raw, machine-readable captures from Power's real
streaming API, not a standalone llama.cpp microbenchmark.

| Qwen3.8-27B mode | Median decode | Acceptance evidence |
| --- | ---: | --- |
| Autoregressive, untouched Q6_K | 35.5793 token/s | 35.4812 minimum; same-artifact baseline |
| Native MTP, untouched Q6_K | 140.1600 token/s | 139.4793 minimum; 3.9394x; exact greedy parity |
| TBQ4 + MTP, full vocabulary, physical-core affinity | **177.3062 token/s** | 175.5958 minimum; 9 / 9 captured-binary samples passed 175 |
| TBQ4 + MTP, historical shared-WDDM range | **159.8593–188.2972 token/s** | Exposes quiet and contended display-GPU boundaries |

The current captured binary reached a 177.3062 token/s median and 175.5958 minimum
across nine 1,024-token samples. All nine passed 175 token/s. An order-balanced
A/B pinned ten worker threads to one logical processor per physical Xeon
W5-2445 core and raised the combined median from 173.0114 to 176.8276 token/s
(2.21%). The topology-specific mask is recorded by the benchmark runner and is
not a portable product default. Earlier quiet-WDDM captures reached 187.6094
and 188.2972 token/s, while a contended 256-token run fell to 159.8593, so 175
remains an observed boundary rather than a guaranteed floor on this shared
display GPU. These results used an RTX 4090, Flash Attention, full CUDA layer
offload, a high-performance host power plan, and a Q6_K-derived artifact whose
main FFN tensors were requantized to Q4_0. It is **not** an untouched 6-bit
result. The full target output and MTP block remain Q6_K, the separate draft
head retains all vocabulary rows in Q4_K, the target verifies every proposal,
and every archived sample produced its expected deterministic output digest.

A separate representative matrix fixes 100 MMLU, GSM8K, and C-Eval tasks,
rotates mode order, and repeats every mode three times. Its throughput includes
prompt processing, generation, and request overhead rather than measuring only
steady-state decode:

| Qwen3.8-27B mode | Lenient score | Strict score | Mean workload throughput |
| --- | ---: | ---: | ---: |
| Autoregressive, untouched Q6_K | 66/100 | 59/100 | 34.551 token/s |
| Autoregressive, TBQ4 mixed artifact | 72/100 | 64/100 | **41.745 token/s** |
| TBQ4 + MTP + prefix FR (earlier build) | 72/100 | 60/100 | 27.951 token/s |

TBQ4 without speculation was 20.8% faster than untouched Q6_K on that
workload. The earlier prefix-FR MTP build was 33.0% slower than TBQ4-off
overall: it improved GSM8K throughput by 6.9%, but C-Eval draft acceptance fell
to 14.21%. The repeated matrix therefore selects TBQ4-off for that
representative workload. A newer three-run, 12-task calibration of the
full-vocabulary path reached 68.211 token/s with fixed K7/S7 versus 35.048
token/s with TBQ4-off, with zero fallback replays and the same 5/12 lenient and
3/12 strict scores. Fixed K7/S6 instead triggered 46 exact prefix replays per
run and fell to 28.226 token/s. This establishes the rollback-safe
configuration but remains too small to replace the 100-task release matrix;
the 188.2972 token/s long-window result is still a workload-sensitive peak.

The larger 31.46 GB UD-Q8_K_XL artifact also ran through exact CPU/GPU tensor
placement, reaching 9.7577 token/s with MTP. Its cross-mode output hashes
differed, so that capture is documented as a boundary result rather than a
parity acceptance.

- [Q6_K and TBQ4/MTP/FR benchmark, protocol, artifacts, and raw samples](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/README.md)
- [Repeated quality/workload matrix and reproducible environment](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/README.md)
- [UD-Q8_K_XL heterogeneous-placement boundary](docs/benchmarks/qwen3.8-27b-ud-q8-k-xl-rtx4090/README.md)
- [Model-neutral speculative decoding design](docs/speculative-decoding.md)

## Choose a surface

- **Artifact provisioner** — enable `artifact-provisioning` when a product owns
  a revision-locked bundle and needs safe first-use installation. Network access
  is policy-gated; verified files are reused offline.
- **Embedded runtime** — enable `embedded-inference` when a model crate owns the
  graph and needs shared execution, placement, and evidence. It opens no
  listener and excludes the model hub.
- **Hosted service** — use the default `server,mistralrs` profile for chat,
  completions, embeddings, lifecycle, metrics, and attestation over explicit
  HTTP, TLS, or vsock transports.
- **Minimal TEE service** — enable `tee-minimal` for the pure-Rust,
  layer-streaming GGUF path in a constrained enclave. Transport selection stays
  explicit.

The embedded library and hosted service are entry points into shared contracts,
not two copies of the same model implementation.

## Architecture

```text
bundle       model graph       API client
  |              |                 |
provision      embed            service
   \             |                /
    +-------- shared core -------+
                  |
      admission / placement / state
                  |
        privacy / evidence / receipts
                  |
          verifier acceptance
```

### Responsibility boundary

| Power owns | Model-owning crates own |
| --- | --- |
| Typed CPU, CUDA, and Metal devices; bounded graph execution | Architecture, topology, layers, kernels, and arithmetic |
| Admission, session pools, microbatching, cancellation, and limits | Tokenizer, preprocessing, postprocessing, and generation policy |
| SafeTensors identity, replicas, mirrors, placement, and residency | Model assets, revision pins, conversion, and tensor contracts |
| Artifact download bounds, SHA-256 admission, locking, and atomic install | Artifact URLs, expected digests, filenames, and first-use policy |
| TEE privacy, attestation binding, sealed state, and receipts | KV/recurrent layout, state semantics, and product quality gates |

This boundary is deliberate: Power contains no product model assets and does
not absorb model-specific topology from OCR, vision, language, or embedding
repositories.

### Shared runtime contracts

- **Bounded execution:** admission control, exact tensor limits, deterministic
  microbatch plans, cancellation-safe queues, and bounded model/device sessions.
- **Verified weights:** canonical identities, SHA-256 checks, optional Ed25519
  signatures, complete or partial mirrors, and storage-to-device residency.
- **Accelerator evidence:** explicit device identity, placement and fallback
  receipts, multi-device meshes, and confidential GPU/NVSwitch claim binding.
- **Recoverable state:** authenticated sealed-state envelopes, primary/backup
  recovery, export authorization, and zeroization.
- **Private observability:** digest-only receipts and telemetry policies that do
  not require prompt or response content.

See [Embedded Inference Architecture](docs/embedded-inference-architecture.md)
for the complete contract model.

## Quick start

### Embed the model-neutral runtime

```toml
[dependencies]
a3s-power = { version = "0.9.0", default-features = false, features = ["embedded-inference"] }
```

```rust
use a3s_power::inference::{DevicePreference, EmbeddedRuntime, InferenceLimits};

fn main() -> Result<(), a3s_power::error::PowerError> {
    let _runtime = EmbeddedRuntime::new(
        DevicePreference::Auto,
        InferenceLimits::default(),
    )?;
    Ok(())
}
```

The caller supplies the reviewed model plan and keeps ownership of semantic
state. Power supplies the execution boundary.

### Run the OpenAI-compatible service

```bash
cargo install a3s-power
a3s-power serve --host 127.0.0.1 --port 11434
```

In another terminal:

```bash
a3s-power models pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
a3s-power chat Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
```

Power stores manifests and content-addressed model blobs under
`~/.a3s/power` by default. Override the root with `A3S_POWER_HOME`.

### Provision a revision-locked bundle

```toml
[dependencies]
a3s-power = { version = "0.9.0", default-features = false, features = ["artifact-provisioning"] }
```

```rust
use a3s_power::artifact_bundle::{
    provision_artifact_bundle, ArtifactBundle, BundleArtifact,
    BundleProvisionPolicy,
};

async fn install() -> Result<(), Box<dyn std::error::Error>> {
    let bundle = ArtifactBundle::new(
        "example/embedding-model",
        "locked-revision",
        vec![BundleArtifact::remote(
            "model.onnx",
            "https://models.example/model.onnx",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            32 * 1024 * 1024,
        )?],
    )?;
    let policy = BundleProvisionPolicy::new(".a3s/power/artifacts/example")
        .with_network(true);
    provision_artifact_bundle(&bundle, &policy).await?;
    Ok(())
}
```

Every file has a hard size limit, streams into a private staging file, passes
SHA-256 admission, and commits atomically under a cross-process lock. Offline
mode fails closed if a verified artifact is unavailable.

## Backends and build profiles

| Feature | Role | Native dependency |
| --- | --- | --- |
| `mistralrs` | Default Candle-based GGUF, SafeTensors, vision, and embedding backend | No C++ inference engine |
| `llamacpp` | Mature GGUF backend with native MTP support | CMake, C++ compiler, and libclang |
| `llamacpp-cuda` | CUDA execution for llama.cpp | CUDA toolkit |
| `llamacpp-mtp-fr` | Experimental reduced-vocabulary MTP draft projection | Reviewed Power patch to the pinned llama.cpp source |
| `picolm` | Pure-Rust layer-streaming GGUF backend for constrained TEE memory | No C/C++ inference engine |
| `embedded-cuda` / `embedded-metal` | Accelerator support for model-owned embedded graphs | Platform toolkit |
| `tls` / `vsock` | RA-TLS and A3S Box guest-host transports | Platform-specific |
| `hw-verify` | AMD KDS and Intel PCS signature verification | Platform crypto dependencies |

Typical builds:

```bash
# Default hosted service
cargo build --release

# Listener-free embedded runtime
cargo build --release --no-default-features --features embedded-inference

# Pure-Rust layer-streaming TEE service
cargo build --release --no-default-features --features tee-minimal

# llama.cpp with CUDA
cargo build --release --no-default-features --features llamacpp-cuda
```

The FR-Spec-inspired path is intentionally separate because it modifies the
pinned llama.cpp source. Follow the patch and build procedure in
[speculative-decoding.md](docs/speculative-decoding.md); ordinary
`llamacpp` builds do not require that patch.

## Speculative decoding

Power exposes model-neutral strategy selection, capability negotiation,
adaptive draft lengths, exact target verification, rollback, and metrics.
Backends advertise support; an explicitly requested unsupported strategy fails
closed.

Available strategy names are `off`, `prompt-lookup`, `ngram-context`,
`draft-model`, `mtp`, `dflash`, and `dspark`, with `auto` selecting a
backend-supported default.

```acl
spec_mode = "mtp"
spec_draft_max = 7
spec_mtp_recurrent_snapshots = 6

# Optional and experimental; requires the patched llamacpp-mtp-fr build.
# A compact d2t head uses ranked rows; omit this key for the full draft head.
# spec_mtp_fr_vocab_size = 8192
```

Recurrent snapshots trade memory for rollback cost. When a rejected suffix is
longer than the resident snapshot window, Power replays the exact accepted
target prefix. Reduced-vocabulary projection affects only the MTP draft head;
it is workload-sensitive and must be retuned across languages and domains.
The current 175 token/s gate keeps the full draft vocabulary.

TBQ4 is an artifact construction choice, not a generic runtime switch. The
current long-window 188.2972 token/s capture combines that mixed artifact with
native MTP, full-vocabulary drafting, batched target/draft greedy CUDA sampling, Flash
Attention, and host/GPU tuning.

## Configuration

The service reads A3S ACL from `~/.a3s/power/config.acl`, or from the path
passed to `a3s-power serve --config`.

```acl
host = "127.0.0.1"
port = 11434
max_loaded_models = 1
keep_alive = "5m"

flash_attention = true
num_parallel = 1

gpu {
  gpu_layers = -1
  main_gpu = 0

  # Optional exact placement when the artifact exceeds VRAM.
  # cpu_tensors = ["output.weight", "blk.3.attn_q.weight"]
}
```

Production TEE deployments add strict policy and verifier-owned pins:

```acl
tee_mode = true
tee_policy_mode = "strict"
redact_logs = true

expected_measurement "sev-snp" {
  digest = "<96-character measurement hex>"
}

model_hash "your-model" {
  digest = "sha256:<64-character artifact digest>"
}
```

Strict mode rejects simulated reports. GPU-confidential mode additionally
requires nonce-fresh NVIDIA evidence, a pinned NRAS verdict, structured device
claims, exact topology policy, and an attested GPU execution digest.

Core settings can also be supplied through `A3S_POWER_*` environment
overrides. Invalid ACL, ranges, strategies, hashes, or unsupported explicit
backend choices fail before inference.

## OpenAI-compatible API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Readiness, loaded models, backend capabilities, and TEE status |
| `POST` | `/v1/chat/completions` | Chat, tools, structured output, vision, and SSE streaming |
| `POST` | `/v1/completions` | Text completion and SSE streaming |
| `POST` | `/v1/embeddings` | Embedding inference |
| `GET` | `/v1/models` | List registered models |
| `POST` | `/v1/models/pull` | Resumable ModelScope or Hugging Face model pull |
| `GET` | `/v1/attestation` | Nonce- and model-bound TEE evidence |
| `GET` | `/metrics` | Prometheus metrics |

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "Explain capability-based security."}],
    "stream": true
  }'
```

Chat and completion responses include an `attestation_receipt` and its
SHA-256 digest. Streaming responses emit the receipt before `[DONE]`.

## Verifiable execution

```text
model bytes + applied runtime policy
                 |
                 v
       canonical claims digest
                 |
                 v
nonce + CPU TEE report + optional GPU evidence
                 |
                 v
request policy + effective prompt + output receipt
                 |
                 v
        independent client verifier
```

The verifier—not the server operator—chooses the acceptable launch
measurement, artifact hash, runtime policy, GPU evidence, and receipt fields.

```bash
a3s-power-verify \
  --url http://127.0.0.1:11434 \
  --nonce <client-nonce-hex> \
  --model-hash <artifact-sha256-hex> \
  --expected-measurement <launch-measurement-hex>
```

The verification surface covers:

- AMD SEV-SNP and Intel TDX hardware signatures and launch measurements.
- Nonce freshness and canonical model/runtime claims.
- RA-TLS attestation transport.
- Request input, decoding, streaming, tool, output-policy, effective-prompt,
  and response digests.
- Optional NVIDIA GPU/NVSwitch identity, firmware, topology, and NRAS verdict
  binding.
- Encrypted model provenance, zeroization, log redaction, and sealed state.

See [Hardware Verifier Operations](docs/hardware-verifier-operations.md) for
production certificate caching, KDS/PCS behavior, and failure policy.

## Security boundaries

- Simulated TEE mode is for development and is rejected by strict verification.
- GPU offload is not confidential merely because the CPU runs in a TEE; use
  `gpu-confidential` policy with verified NVIDIA evidence.
- `picolm` can release mapped plaintext layer pages after use. The current
  AES-GCM encrypted artifact format is not seekable, so streaming decrypt still
  retains locked plaintext while its handle is live.
- Effective-prompt digests are available for deterministic text paths. Opaque
  multimodal renderer paths leave the claim absent instead of fabricating it.
- FR-Spec-inspired vocabulary reduction and mixed quantization are experimental
  performance techniques, not universal quality guarantees.

## Documentation

| Document | Scope |
| --- | --- |
| [Embedded Inference Architecture](docs/embedded-inference-architecture.md) | Ownership, graph execution, placement, scheduling, state, and receipts |
| [Model-neutral Speculative Decoding](docs/speculative-decoding.md) | Strategies, native MTP, patching, benchmark protocol, and acceptance |
| [Qwen3.8-27B Q6_K benchmark](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/README.md) | Peak gates, artifact identity, and representative-workload boundary |
| [Qwen3.8-27B reproduction guide](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md) | CUDA build, pinned inputs, performance replay, evidence audit, and validation commands |
| [Qwen3.8-27B repeated quality matrix](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/README.md) | Three-mode quality, workload throughput, evidence, and replay protocol |
| [Qwen3.8-27B Q8 boundary](docs/benchmarks/qwen3.8-27b-ud-q8-k-xl-rtx4090/README.md) | Heterogeneous placement and parity limitation |
| [Hardware Verifier Operations](docs/hardware-verifier-operations.md) | Production hardware-signature verification |
| [Supply-chain Audit](docs/supply-chain.md) | Feature profiles, native code, and threat model |
| [Storage Benchmark](docs/storage-benchmark.md) | Verified storage and residency measurements |
| [Roadmap](ROADMAP.md) | Acceptance gates and remaining work |
| [Changelog](CHANGELOG.md) | Released behavior |

API documentation is published at [docs.rs/a3s-power](https://docs.rs/a3s-power).

## Development

Run checks from this crate, not from the monorepo root:

```bash
cargo fmt --all -- --check
cargo test --lib
cargo test --no-default-features --features embedded-inference --lib
cargo test --no-default-features --features picolm --lib
cargo clippy --all-targets -- -D warnings
```

The `llamacpp` profile additionally needs CMake, a C++ compiler, and libclang.
The experimental `llamacpp-mtp-fr` profile also needs the reviewed patch
described above.

CI checks formatting, Clippy feature matrices, tests, the listener-free
embedded dependency boundary, Windows/macOS embedded builds, and release builds
for Apple Silicon, Intel macOS, Linux ARM64, and Linux x86_64.

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
