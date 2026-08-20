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

| Qwen3.8-27B artifact and mode | Fixed-task quality proxy | Mean request-wide throughput | Median steady decode |
| --- | --- | ---: | ---: |
| Untouched Q6_K, autoregressive | 67/100 lenient; 60/100 strict (100 tasks, 3x) | 30.883 token/s | 35.5793 token/s (earlier capture) |
| Untouched Q6_K, native MTP | Matrix not run; fixed peak prompt has exact greedy parity | -- | 140.1600 token/s |
| TBQ4 mixed, autoregressive | 70/100 lenient; 64/100 strict (100 tasks, 3x) | 38.724 token/s | -- |
| TBQ4 mixed + full-vocabulary fixed MTP, K7/S7 | **76/100 lenient; 66/100 strict** (100 tasks, 3x) | **83.228 token/s** | **175.2089 token/s** |
| TBQ4 mixed + full-vocabulary guarded MTP, K7/S6 | 5/12 lenient; 3/12 strict (12 tasks, 3x) | 54.060 token/s | **177.7165 token/s** |
| TBQ4 mixed + MTP + prefix FR (historical) | 72/100 lenient; 60/100 strict (100 tasks, 3x) | 27.951 token/s | 184.3665 token/s |
| UD-Q8_K_XL, autoregressive heterogeneous placement | Matrix not run | -- | 6.3484 token/s |
| UD-Q8_K_XL, native MTP K4/S4 heterogeneous placement | Matrix not run; cross-mode output hashes differ | -- | 9.7577 token/s |

The quality column is a task-accuracy proxy rather than a general intelligence
or IQ measurement. The 100-task release matrix and 12-task rollback calibration
have different denominators and must not be compared directly. Request-wide
throughput includes prefill, generation, and request overhead, while steady
decode is a warmed-up repetitive long-output measurement. See the
[consolidated mode table](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/README.md#quality-and-speed-by-mode)
for the omitted calibration modes, limitations, and evidence links.

The current rollback-complete K7/S7 profile reached a 175.2089 token/s median
across nine 1,024-token samples, with a 174.2211 token/s minimum and five of
nine samples at or above 175. The guarded K7/S6 peak profile reached a 177.7165
token/s median and a 176.7287 token/s minimum, with all nine samples above 175.
K7/S6 preserves its high-acceptance fast path; after the first exact replay in
a low-acceptance request, the request-local guard permanently clamps proposals
to the six-snapshot rollback window. K7/S7 is the balanced mixed-workload
default because it needs no replay or clamp. The topology-specific `0x55555`
mask pins ten worker threads to one logical processor per physical Xeon
W5-2445 core and is not a portable product default.

These are observed steady-decode boundaries, not a 175 token/s service floor.
Earlier quiet-WDDM captures reached 187.6094 and 188.2972 token/s, while a
contended 256-token run fell to 159.8593. All current results used an RTX 4090,
Flash Attention, full CUDA layer offload, a high-performance host power plan,
and a Q6_K-derived artifact whose main FFN tensors were requantized to Q4_0.
It is **not** an untouched 6-bit result. The target output and MTP block remain
Q6_K, the separate draft head retains all vocabulary rows in Q4_K, the target
verifies every proposal, and every current peak sample produced the expected
deterministic output digest.

A separate representative matrix fixes 100 MMLU, GSM8K, and C-Eval tasks,
rotates mode order, and repeats every mode three times. Its throughput includes
prompt processing, generation, and request overhead rather than measuring only
steady-state decode:

| Qwen3.8-27B mode | Lenient score | Strict score | Mean workload throughput |
| --- | ---: | ---: | ---: |
| Autoregressive, untouched Q6_K | 67/100 | 60/100 | 30.883 token/s |
| Autoregressive, TBQ4 mixed artifact | 70/100 | 64/100 | 38.724 token/s |
| TBQ4 + full-vocabulary MTP, fixed K7/S7 | **76/100** | **66/100** | **83.228 token/s** |

On the current binary, TBQ4-off was 25.4% faster than untouched Q6_K and K7/S7
was 114.9% faster than TBQ4-off on the same request-wide workload. K7/S7 moved
from 70 to 76 lenient answers and from 64 to 66 strict answers. The paired
TBQ4-off comparison had seven lenient gains and one loss (`p=0.0703`) and three
strict gains and one loss (`p=0.625`, exact McNemar). This is **no observed
intelligence regression** on the fixed sample, but neither comparison reaches
the conventional 0.05 threshold, so it is not proof that MTP improves general
model intelligence. All 100 task predictions were stable across the three
repetitions, all 900 requests completed without errors, acceptance was 51.33%,
and fallback replay and rollback-guard activation were both zero.

The earlier prefix-FR matrix remains archived as a rejected universal default:
its workload-wide acceptance fell to 25.55% and its 27.951 token/s request-wide
rate was 33.0% below that historical TBQ4-off capture. Full-vocabulary K7/S7
removes that domain-coverage failure while retaining exact target verification.

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
spec_mtp_recurrent_snapshots = 7

# Optional and experimental; requires the patched llamacpp-mtp-fr build.
# A compact d2t head uses ranked rows; omit this key for the full draft head.
# spec_mtp_fr_vocab_size = 8192
```

Recurrent snapshots trade memory for rollback cost. K7/S7 is the balanced
default because every proposal has a resident rollback point. A deliberately
narrow K7/S6 profile may replay the exact accepted target prefix once; its
request-local guard then clamps later rounds to six proposals. Reduced-
vocabulary projection affects only the MTP draft head; it is workload-sensitive
and must be retuned across languages and domains. The current gates keep the
full draft vocabulary.

TBQ4 is an artifact construction choice, not a generic runtime switch. The
current K7/S7 capture combines that mixed artifact with native MTP,
full-vocabulary drafting, batched target/draft greedy CUDA sampling, Flash
Attention, and host/GPU tuning. It reached 175.2089 token/s steady decode and
83.228 token/s request-wide throughput on the fixed 100-task workload. The
188.2972 token/s capture is retained as a historical quiet-WDDM high-water mark.

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
