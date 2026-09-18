# A3S Power — Next Phase Development Plan

## First Principles Analysis

### Core Mission
A3S Power provides a **model-neutral, bounded, and verifiable inference runtime**.
Language, vision, OCR, embedding, audio, multimodal, and scientific model crates
retain their own topology and semantics. Power owns shared execution concerns:
artifact integrity, devices, admission, placement, mutable state, cancellation,
privacy, evidence, and independent verification. Hardware TEE support is one
deployment boundary, not a restriction to LLMs or to a particular model family.

### What Makes Power Unique (The Moat)
1. **Hardware-enforced privacy** — not policy promises, but CPU-level memory encryption
2. **Verifiable inference** — clients can bind exact artifacts, reviewed graphs,
   runtime policy, devices, inputs, and outputs
3. **tee-minimal build** — ~1,220 deps, no C++ inference engine, reduced auditable supply chain
4. **Layer-streaming** — O(layer_size) peak RAM, runs 7B+ models in 512MB EPC
5. **Model-neutral execution contracts** — one bounded runtime serves language,
   vision, OCR, embedding, audio, multimodal, and custom reviewed graphs without
   family dispatch in the core

### Current State (v1.0.0 source freeze; production tag pending evidence)
- Changelog `[1.0.0] - 2026-09-10` is frozen with an empty `[Unreleased]`
  section; package version remains `1.0.0`. Production release still requires
  the exact-parent evidence-only child (CPU, CUDA, Metal, SEV-SNP
  confidential GPU) and a GitHub-verified annotated tag.
- 3 backends (mistralrs, llamacpp, picolm) — all functional
- TEE runtime stack (attestation collection, encrypted models, RA-TLS, privacy,
  audit); production verifier coverage is qualified below
- OpenAI-compatible API with streaming
- Listener-free embedded runtime with typed CPU, CUDA, and Metal devices
- Model-owned reviewed graphs, finite shape profiles, session replicas,
  cancellation-safe execution batches, and device-resident graph boundaries
- A model-neutral release-evidence gate and complete-contract collector;
  a production tag is admitted only when its evidence-only child authenticates
  exact-revision CPU, CUDA, Metal, and confidential-GPU captures
- One canonical SafeTensors collection identity now joins startup integrity,
  model-bound attestation, embedded execution, persistent cross-host fixtures,
  and confidential accelerator declarations; mismatched directory hashing can
  no longer make a real promotion impossible
- Local speculative drafts, LoRA adapters, and multimodal projectors now have
  content identities independent of host paths. Strict startup, backend load,
  signature verification, model-bound attestation, request receipts, and the
  independent verifier all bind the same canonical auxiliary-artifacts digest.
- Model-bound attestation and request receipts now bind one canonical
  inference-execution digest for speculative/MTP/FR policy, prompt-cache
  bounds, model residency, mmap/mlock, threads, Flash Attention, and parallel
  slots. Strict clients derive the expected value from the resolved ACL and
  explicit speculative modes remain fail-closed.
- Proof-promoted confidential captures preserve that accepted inference digest
  and any accepted auxiliary-artifacts digest as explicit release fields. Final
  bundle replay therefore audits the exact execution and proposer identities,
  in addition to the aggregate attestation-claims digest.
- Strict SEV-SNP verification now binds policy fields to the exact signed raw
  report. Intel TDX fails closed until a DCAP Quote/QVL path is implemented.

### 2026 Shared Document Inference Substrate (TO2)

The model-neutral Power layer now provides the scheduling substrate required by
`a3s-ocr` and `a3s-parser`: finite cancellation-aware model queues, an exact
device/model session pool with declared resident-byte bounds, a shared physical
device gate, policy-bounded exclusive mutable session replicas with worst-case
residency admission, safe-boundary health retirement with lazy reconstruction,
monotonic queue deadlines with aggregate expiry evidence, deterministic
memory-aware microbatch plans with live-pressure revalidation, same-request
device-resident reviewed-graph chains with aggregate handle-byte admission, and
digest-only receipt evidence. Language,
vision, OCR, embedding, and multimodal crates use the same runtime contracts.
OCR stage semantics, image
pre/postprocessing, document target identity, cross-page structure, and retry
policy remain in their owning crates. Ready pool entries intentionally have no
implicit eviction; the owning service controls pool lifetime.
Resident graph handles contain no family dispatch or model topology: they
validate only reviewed F32 shape contracts, runtime/device/permit identity,
aggregate bytes, cancellation, and canonical initial/final tensor digests.

### 2026 Attestation Soundness Reopen

The implementation has a production attestation hardening follow-up after an
external review identified fail-open and claim-binding gaps in the current TEE
flow. The detailed remediation plan is tracked in
[`docs/attestation-hardening-plan.md`](docs/attestation-hardening-plan.md).
The current remediation has added strict fail-closed policy defaults, v2
CPU/GPU/runtime attestation claims, strict verifier defaults, and request-level
inference receipts. Encrypted GGUF models can now load from locked
`MemoryDecryptedModel` plaintext and `LayerStreamingDecryptedModel` plaintext
through `picolm`, while unsupported backends fail closed. Local deterministic chat
renderers can include rendered-prompt `effective_prompt` digests in receipts;
mistralrs text chat can include a domain-separated prompt-token-ID digest, and
proxy backends can include an upstream-declared digest through an explicit
opt-in endpoint. NVIDIA GPU confidential-computing support now has configured
evidence/verdict binding, a live `nvattest-cli` provider, and a direct
`nras-rest` provider that share the CPU/GPU attestation nonce, extract
structured NVIDIA device identity/freshness claims from the NRAS/NVAT verdict,
exposes verifier policy checks for those device claims, and supports
deployment-specific GPU/NVSwitch topology, NVIDIA claims-version, UEID,
OEM ID, hwmodel, driver, firmware, secure-boot, debug-state, and NVSwitch
identity/version pinning, with RIM schema validation enforced for accepted
device claims. A native NRAS SDK client is optional and not required for
v1. Multimodal prompt claims have a closed `EffectivePromptClaimKind`
(reserved `chat.multimodal-rendered-prompt`) with fail-closed receipt binding
that abstains until backends expose the exact representation; inventing
text-only stand-in digests for image-bearing chat remains rejected. Attested
private-fabric serving privacy is retained for AAD wire binding but is
explicitly excluded from v1 prefill/decode advertisement. See
[`docs/attestation-hardening-plan.md`](docs/attestation-hardening-plan.md) and
[`docs/v1-support-matrix.md`](docs/v1-support-matrix.md).

### First Principles Question: What Should We Build Next?

Every candidate feature must pass this filter:
1. Does it directly strengthen the privacy/verifiability moat?
2. Does it make Power deployable in real TEE production environments?
3. Does it close a gap that blocks actual adoption?

Features that fail this filter get rejected, no matter how "nice to have" they are.

Phases 4–7 below are completed history. The answer after the P0–P6 software
exit is [Next Plan](#next-plan-after-the-software-exit): publish the frozen
parent, then prove Cloud can host that revision. Do not open another execution
milestone.

---

## Phase 4: TEE Runtime Components Complete; Remote Verification v1 Boundary Set

**Goal**: Make Power deployable in real AMD SEV-SNP and Intel TDX environments
with independently verifiable evidence. The runtime components below are
implemented. Production SEV-SNP still needs release-evidence captures for a
tagged production release. Intel TDX is explicitly unsupported on the v1
production matrix until a reviewed DCAP Quote/QVL path exists (same
OR-exclusion pattern as HSN DirectDeviceMemoryPull advertisement).

### 4.1 — ✅ picolm Multi-Turn Session KV Cache
- `Arc<Mutex<Option<KvCache>>>` return path via `tokio::spawn` background task
- KV cache positions correctly maintained across turns
- Session map insert/remove with eviction on unload

### 4.2 — ✅ picolm Chat Template from GGUF Metadata
- Read `tokenizer.chat_template` from GGUF metadata
- Jinja2 rendering via minijinja with ChatML fallback
- Tested with Llama 3 and invalid template fallback

### 4.3 — ✅ picolm Configurable Context Length
- Read `context_length` from GGUF metadata, capped at 32K
- KV cache allocation scaled accordingly

### 4.4 — ✅ picolm Stop Sequence Support
- Check generated text against `stop` sequences after each token
- Trim output at stop boundary, set `finish_reason: "stop"`

### 4.5 — ✅ Remove Unused `candle-core` from picolm Feature
- Removed from Cargo.toml, picolm feature now: `["dep:memmap2", "dep:half", "dep:rayon"]`

### 4.6 — ✅ Integration Tests
- `tests/integration.rs`: 14 tests (HTTP API, router, registry, auth, error paths)
- `tests/picolm_tee.rs`: 8 tests (load/unload cycle, TEE mode, deterministic output)
- `tests/picolm_real.rs`: Real model inference (gated by model file presence)

---

## Phase 5: Performance & Scalability ✅

**Goal**: Close the remaining performance gaps vs. llama.cpp for TEE-constrained environments.

### 5.1 — ✅ SIMD-Accelerated vec_dot Kernels
- AVX2+FMA kernels for F32, Q8_0, Q4_K, Q6_K
- Runtime feature detection via `is_x86_feature_detected!`
- Scalar fallback for non-AVX2 platforms (aarch64)
- Parity tests: AVX2 vs scalar for all kernel types

### 5.2 — ✅ NEON-Accelerated vec_dot for Apple Silicon
- `#[cfg(target_arch = "aarch64")]` NEON paths for F32, Q8_0, Q4_K, Q6_K
- F32: vfmaq_f32 4-wide accumulation
- Q8_0: vmull_s8 + vpadalq_s16 with scalar scale accumulation
- Q4_K: nibble extract + vmull with min/scale dequant
- Q6_K: 6-bit reconstruct (ql low4 + qh high2) + vmull
- Parity tests for all 4 kernel types

### 5.3 — ✅ Batch Prefill
- Layer-outer, token-inner loop ordering: O(n_layers) page faults instead of O(n_layers × n_tokens)
- Each layer's mmap pages loaded once for all tokens, then released
- `matmul_batch` function for batched matrix-vector multiply
- Hidden states matrix `[n_tokens × n_embd]` flows through layers

### 5.4 — ✅ Speculative Decoding (Prompt-Lookup)
- Prompt-lookup decoding: matches n-grams (2–5) from generated text against input tokens
- Zero draft cost — no extra model or layer-skipping needed
- `count_accepted` greedy verification against full-model logits
- KV cache rollback via `truncate()` for rejected draft tokens
- Wired into decode loop with hidden-state backup/restore
- Disabled during grammar-constrained generation (structured output)
- Works well for tasks where output overlaps input (summarization, JSON, code completion)

---

## Phase 6: TEE Hardening ✅

**Goal**: Close security gaps for production TEE deployment.

### 6.1 — ✅ Timing Side-Channel Mitigation
- `timing_padding_ms` wired into both streaming and non-streaming chat paths
- ±20% jitter via existing `timing_padding()` method

### 6.2 — ✅ Memory Zeroization Audit
- `Drop` impl for `ForwardBuffers`: zeroizes all 12 Vec<f32> buffers
- `Drop` impl for `LayerKvCache`: zeroizes K/V f16 data
- `KvCache::clear()` now zeroizes data instead of just resetting length

### 6.3 — ✅ Startup Self-Test
- Embedded test vectors for rms_norm, vec_dot_f32, vec_dot_q8_0
- Runs at model load time, fails fast with clear error on mismatch
- Catches memory corruption in TEE before inference begins

---

## Phase 7: Ecosystem Integration ✅

**Goal**: Make Power useful in the broader A3S platform.

### 7.1 — ✅ picolm Tool/Function Calling
- Wire `tool_parser::parse_tool_calls()` into picolm response stream
- Accumulate full generated text, parse tool calls on final chunk (EOS/stop/max_tokens)
- `has_tools` flag in GenerateParams, set from ChatRequest.tools
- Matches llamacpp/mistralrs backend pattern

### 7.2 — ✅ picolm Structured Output (JSON Grammar)
- `JsonGrammarSampler`: stack-based JSON validator for grammar-constrained sampling
- Tracks structural state (object/array/string/number/keyword nesting)
- `mask_logits`: filters tokens whose first character violates grammar
- Wired into decode loop via `response_format` field from ChatRequest
- Auto-stops generation when complete JSON value is produced

### 7.3 — ✅ picolm Repeat/Frequency Penalty
- 64-token ring buffer tracking recent generated tokens
- `repeat_penalty` (multiplicative, llama.cpp style)
- `frequency_penalty` (proportional to count, OpenAI style)
- `presence_penalty` (flat if appeared, OpenAI style)
- Applied to logits before sampling

---

## What We Will NOT Build (First Principles Rejection)

### ❌ GPU Support inside picolm
**Why not**: picolm is the CPU/EPC-constrained pure Rust backend. Adding GPU memory to picolm itself would undermine that backend's minimal CPU TEE security model. Power still supports NVIDIA GPU acceleration through the other backends, and production NVIDIA GPU Confidential Computing support is tracked separately through `tee_policy_mode = "gpu-confidential"` and bound GPU evidence claims.

### ❌ Embeddings in picolm
**Why not**: Embedding models are small, don't need layer-streaming, and don't process sensitive user prompts (they process documents at indexing time, not query time). Use mistralrs for embeddings. Adding embedding support to picolm adds complexity without strengthening the privacy moat.

### ❌ Vision/Multimodal in picolm
**Why not**: Vision models require image encoders (ViT) that are architecturally different from text transformers. The complexity cost is high, and the TEE use case for vision is niche. Use mistralrs for vision.

### ❌ Model Quantization in picolm
**Why not**: Quantization is a one-time offline operation. It doesn't need to run inside TEE. Users quantize models before deployment.

### ❌ LoRA/Adapter Support in picolm
**Why not**: LoRA adds complexity to the forward pass and the supply-chain audit story. In TEE, you want to verify one specific model — not a base + N adapters. If needed, merge LoRA into the base model before deployment.

---

## Completion Summary

The picolm, performance, hardening-component, ecosystem, attestation v1
remediation, and ROADMAP P0–P6 development milestones above are implemented
and covered by the repository test profiles (including first-principles
distributed-serving and env-gated live llama.cpp evidence). That is the
development/testing plan exit for this repository. It is not a claim that
Power is ready to tag a production v1.0.0 release.

The Rust release API now requires an opaque exact-report proof for confidential
promotion, and the checked-in external capture runbook carries that API through
raw vendor-evidence preservation and create-new CLI output. The generic runner
also materializes persistent fixture weights and creates the local CUDA
source/declaration pair atomically; caller-owned graphs retain their model-owned
declaration path. The release CLI now also verifies each transferred capture
against an exact platform, version, and source revision before assembly while
marking that result as single-capture scope rather than production eligibility.
A production release
exists only when its signed evidence child proves that all of these gates pass:

1. one immutable revision supplies CPU, CUDA, Metal, and confidential-GPU
   complete-contract captures;
2. the checked-in external capture runbook is executed on named hardware and
   its raw vendor evidence, report, declaration, environment, artifact hashes,
   promoted capture, and release trust-root material are preserved;
3. Intel TDX either gains a reviewed DCAP Quote/QVL path or remains explicitly
   unsupported by the v1 production support matrix;
4. the full default, embedded, accelerator, verifier, documentation, and release
   checks pass for the frozen source parent and its evidence-only child; and
5. the GitHub-verified annotated tag and root-monorepo gitlink bind those exact
   revisions.

See [ROADMAP.md](ROADMAP.md) and
[Production Release Evidence Gate](docs/release-evidence-gate.md) for the
authoritative remaining acceptance evidence.

---

## Next Plan After the Software Exit

Status date: 2026-09-18. This section is the development plan. It is not a
production-release claim, and it is not permission to add runtime features.

### What is already true

The mission is a model-neutral, bounded, verifiable inference runtime. The
software that serves that mission inside this repository has exited:

- P0–P5 bounded execution, receipts, and the release-evidence machinery exist.
- P6 distributed serving exited by shipping buffered-host loopback prefill/decode
  and by excluding high-speed-network advertisement, attested-private-fabric
  readiness, and Intel TDX from the v1 matrix. Exclusion is not capability.
- `PowerRuntimeServiceProfile` already compiles one digest-pinned deployment
  into the shared `a3s-runtime` Service contract. Placement, accelerator
  leases, rollout, authorization, and routing stay outside Power.
- Attestation v1 remediation is complete under the same exclusions: multimodal
  prompt digests abstain, a native NRAS SDK is unnecessary, and private-fabric
  readiness is not advertised.

### What is not true

Two gaps still fail the filter above. Both block the mission. Neither is a new
kernel, backend, or protocol.

| Gap | Evidence that it is still open | Why it blocks the mission |
| --- | --- | --- |
| Production verifiability | No `release/v1.0.0/` evidence child. No GitHub-verified `v1.0.0` tag. crates.io remains v0.9.0. Metal and SEV-SNP confidential-GPU captures are absent. | The verifier exists, but the moat is not a release fact until one immutable revision authenticates CPU, CUDA, Metal, and confidential GPU. |
| Adoption | Cloud `PW0` is Planned. Clean-host audits print `power_revision=UNBOUND`. `compat/cloud-stack.acl` has no `power` component. | Power is not yet the sole local inference boundary. An OpenAI data plane cannot be claimed. |

Windows CPU and CUDA complete-contract captures under
`docs/benchmarks/release-contract-windows-20260910/` are regression evidence for
frozen parent `514031dc74edd72da7c3bfee40144a38d2d91434` only. They do not
authorize the tag, and they cannot be relabeled onto a later commit.

### Freeze invariant

v1.0.0 evidence binds that parent and no other. As of 2026-09-18, `main` is
`3c92da571ecd533c7105ae716a6d6ec089f904a6`, two documentation commits after that
parent. Phase R checks out the parent, not `HEAD`.

- Do not commit documentation, refactors, or this plan onto that parent before
  its evidence-only child exists. Any source commit after it invalidates the
  checked-in CPU and CUDA captures. A new parent requires all four platforms
  again, not the two missing ones.
- Do not retag, amend, or rebuild v1.0.0 from a tree that contains post-freeze
  edits.
- Open the next source line only after the annotated tag exists, or accept
  explicitly that v1.0.0 will not be cut from the current parent.

Operator entry points already exist. Do not write a second capture CLI unless
one of these commands fails closed on the frozen parent:

- `tools/release-capture/capture-macos-metal.sh`
- `tools/release-capture/capture-confidential-source.ps1` plus the proof-backed
  promotion in `docs/external-release-capture.md`
- `tools/release-capture/assemble-evidence-child.sh`
- `tools/verify-release-candidate.sh`

### Phase R — Publish the frozen parent

Goal: make v1.0.0 a production release, or leave it untagged. No Power code.

Work, in order:

1. On a clean checkout of `514031dc74edd72da7c3bfee40144a38d2d91434`, capture
   native Apple Silicon Metal evidence. Virtual, translated, or paravirtual
   GPUs fail the existing gate and are not a shortcut.
2. On confidential hardware, promote a distinct local CUDA capture with a
   strict SEV-SNP report and NVIDIA NRAS proof. A raw report or a caller label
   cannot mint the confidential-GPU class. Intel TDX stays unsupported.
3. Assemble `release/v1.0.0/release-evidence.json` and the single-line SHA-256
   pin as the only diff of an evidence-only child. Preserve the raw vendor
   evidence outside that commit.
4. Run the existing candidate verifier. Publish binaries from the frozen
   parent. Point a GitHub-verified annotated tag at the evidence child.
5. Update the root gitlink only to that tagged revision. Do not publish a
   compatibility-lock entry in the same step.

Exit: the support-matrix release decision is true for v1.0.0. If Metal or
confidential hardware is unavailable, the honest state remains "source freeze,
not a production release." Do not weaken `strict_v1` to ship.

### Phase A — Host the tagged revision

Goal: Cloud deploys the tagged Power revision as an ordinary Box-hosted
Runtime Service. Start this phase only after Phase R, on a new post-tag
revision. Do not rebuild the profile that already exists.

Power may change code only when the current `PowerRuntimeServiceProfile`, ACL
configuration, `/health`, and worker observation contract cannot satisfy the
PW0.1 exit. Prove that hole with a failing Cloud/Box run before editing Power.

Cloud-owned exit, which Power must not reimplement:

- Compile the existing profile through Inference into the current Workload,
  Flow, Fleet Claim, Runtime, Box, Gateway, and audit paths.
- Deploy, become healthy, serve one bounded streaming request and one
  non-streaming request, update, roll back, and stop.
- Recover Power process death, Agent death, and Box VM loss.
- Persist no prompt, response, secret, credential, or second configuration.
- Then, and only then, add a sorted `power` component to
  `compat/cloud-stack.acl` with the matching Cloud and Gateway pins, and write
  `apps/cloud/tools/power-conformance/power-revision`. Absence of that pin is
  intentional until this exit passes.

Forbidden in Power: a second scheduler, node channel, device allocator, route
authority, authorization authority, usage ledger, queue, or lifecycle store.
Gateway consumes the existing observation facts. It does not import Power
internals.

`PW0.tee` is a stronger profile on top of this software exit plus the Phase R
confidential-GPU evidence. It is not a substitute for Phase A, and it does not
reopen TDX.

### Rejected until a named deployment is blocked

These fail the filter. Do not schedule them as the next milestone.

| Proposal | Why it is rejected now |
| --- | --- |
| More CUDA fusion, speculative paths, or backends | Does not make the frozen revision releasable or hostable. |
| High-speed `DirectDeviceMemoryPull` | v1 excluded it on purpose. Loopback prefill/decode is the supported software path. |
| Intel TDX Quote/QVL | v1 excluded it on purpose. SEV-SNP is the confidential-GPU class. |
| Attested-private-fabric advertisement | Digest binding exists. Advertising fabric without TEE-export evidence would weaken the moat. |
| Native NRAS SDK | `nvattest-cli` and `nras-rest` already fail closed. |
| Emitible multimodal prompt digests | v1 abstains until a backend exposes the exact representation. Inventing one is a false claim. |
| picolm GPU, embeddings, vision, quantization, or LoRA | Already rejected. They weaken the minimal CPU TEE backend or move offline work into the enclave. |
| A Power-owned deployment controller | Breaks the Runtime/Box boundary the profile was built to preserve. |

### Order

1. Phase R on the frozen parent, with no source changes.
2. Phase A on the tagged revision, changing Power only to close a proven
   contract hole.
3. `PW0.tee` only after both exits.
4. Reopen an excluded capability only when a named deployment is blocked by
   that exclusion, and the replacement has the same machine-enforced evidence
   bar the exclusion replaced.

### Issue and PR triage (2026-09-18)

Authoritative queue check: `A3S-Lab/Power` had **zero** open issues and **zero**
open pull requests. Closed Power#3 already delivered `PowerRuntimeServiceProfile`;
cross-repo PW0 acceptance stays on Cloud#85. Tracking issues opened for the real
blockers:

| Issue | Role |
| --- | --- |
| [#55](https://github.com/A3S-Lab/Power/issues/55) | Phase R Metal capture on the freeze parent |
| [#56](https://github.com/A3S-Lab/Power/issues/56) | Phase R SEV-SNP confidential-GPU promotion |
| [#57](https://github.com/A3S-Lab/Power/issues/57) | Phase A prep: air-gapped embedding load + observation TTL |

Rejected as overfitting or out of order:

| Candidate | Decision |
| --- | --- |
| Local `codex/v1-metal-capture-*` branches | Not ancestors of `514031dc…`; cannot authorize v1.0.0 |
| Local `perf/*` CUDA fusion branches | Behind `main`; do not unblock release or hosting |
| Renaming mistralrs `from_hf_cache_pathf` | Real upstream API name on text/vision builders; not a typo |
| More speculative/CUDA micro-optimizations | Fail the three-question filter until Phase R exits |
