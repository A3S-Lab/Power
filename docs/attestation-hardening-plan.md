# A3S Power Attestation Hardening Plan

Status: immediate remediation in progress
Scope: A3S Power TEE and NVIDIA GPU confidential-computing attestation soundness
Current code baseline reviewed locally: `4efcd1c`

## Purpose

This plan responds to the coordinated disclosure report covering model hash
enforcement, cached attestation state, optional hardware verification, simulated
TEE acceptance, encrypted model hash semantics, CPU/GPU attestation binding, and
runtime policy claims.

The goal is to make the production attestation path fail closed by default and
to make the verifier's claim set match what Power actually runs.

NVIDIA GPU support is a production requirement for Power. This plan therefore
separates ordinary CUDA acceleration from NVIDIA GPU confidential-computing
attestation: ordinary CUDA can remain a supported performance mode, while
`gpu-confidential` must fail closed unless NVIDIA evidence is configured,
verified by the deployment's NVIDIA flow, and cryptographically bound to the
CPU TEE evidence. It must also fail closed when the final runtime GPU execution
configuration is CPU-only.

## Patch Status

Landed in the current working tree:

- Added `tee_policy_mode` with `development`, `strict`, and
  `gpu-confidential` modes. The default is `strict`.
- Made strict policy reject simulated TEE by default through the effective TEE
  allowlist.
- Made strict and `gpu-confidential` startup require a 48-byte
  `expected_measurements` launch-measurement pin for the detected hardware TEE
  type, then self-check that the current attestation report matches it.
- Made strict TEE startup require `model_hashes` or `model_signing_key`, and
  require every registered local model to be pinned when no signing key is
  configured.
- Added NVIDIA GPU confidential-computing evidence providers for configured
  evidence/verdict bytes, live `nvattest-cli`, and direct `nras-rest`
  attestation. `gpu-confidential` now fails closed unless GPU evidence and an
  NRAS/NVAT verdict digest are bound into v2 claims and the final runtime
  configuration enables GPU execution/offload.
- Added production startup validation for NVIDIA GPU confidential-computing
  metadata: `gpu-confidential` requires the canonical `nvidia-nras` provider
  label, an absolute executable `nvattest_path` for the live CLI path, HTTPS
  custom NRAS/RIM/OCSP service URLs without embedded credentials, and an
  absolute existing non-empty relying-party policy file when one is configured;
  file-backed configured and direct NRAS REST evidence/verdict sources must
  also use absolute paths to existing non-empty regular files.
- Changed `/v1/attestation?model=...` to re-hash the current local model
  artifact and fail on missing, malformed, or stale model hashes instead of
  silently omitting the model binding.
- Added deterministic directory-manifest hashing for HuggingFace/Vision-style
  model directories and emits `directory-manifest-sha256` model claims.
- Added typed encrypted model digest semantics. Explicit `model_hashes` pins
  encrypted models by decrypted plaintext SHA-256 and model claims include both
  `plaintext_digest` and `ciphertext_digest`; manifest-only fallback emits an
  explicit `ciphertext-artifact-sha256` claim.
- Added `VerificationPolicy` and `verify_report_strict()` so production
  verifier callers fail when hardware signature verification is missing and
  reject simulated TEE reports; strict verification now also requires an
  operator-pinned launch measurement.
- Made `a3s-power-verify` default to strict verification; skipping hardware
  signatures and measurement pinning now requires explicit `--allow-offline`,
  while strict mode requires `--expected-measurement`.
- Added `AttestationClaimsV2`, model digest claims, GPU evidence claims,
  canonical JSON hashing, and CPU TEE `report_data` binding for model-bound
  and GPU-confidential attestations.
- Updated verifier logic so reports with v2 claims verify nonce/model/GPU
  digests and GPU evidence provider/format/count pins from the canonical claims
  and reject mismatched claims/report_data bindings.
- Added runtime policy claims for model-bound attestations. Power now hashes
  applied chat templates and canonical GPU execution/offload parameters into
  `AttestationClaimsV2`; unapplied manifest system prompts, pre-seeded
  messages, and default generation parameters are intentionally not claimed.
- Added verifier and CLI checks for runtime policy digests via
  `VerificationPolicy::require_runtime_policy()` and
  `a3s-power-verify --require-runtime-policy`.
- Added `VerificationPolicy::gpu_confidential()` /
  `require_gpu_confidential()` and `a3s-power-verify --gpu-confidential` so
  production NVIDIA GPU confidential-computing verifiers can require v2 claims,
  nonce freshness, an expected NVIDIA NRAS verdict digest, verifier-pinned GPU
  provider/format/count policy, structured device claims, verifier-pinned GPU
  topology, claims schema version, identity/version policy, secure-boot/debug
  state, runtime policy, and GPU execution/offload digest pinning as one
  profile. The production GPU profile no longer accepts raw evidence digest
  pinning as a substitute for pinning the NRAS verdict digest.
- Added per-request inference receipts for `/v1/chat/completions` and
  `/v1/completions`. Non-streaming responses include `attestation_receipt` and
  `attestation_receipt_sha256`; streaming responses emit a final receipt event
  before `[DONE]`. Receipt v2 includes the same runtime
  chat-template/GPU execution policy claim semantics used by model-bound
  attestation.
- OpenAI-compatible chat and completion requests now expose extended local
  sampling controls (`top_k`, `min_p`, `repeat_penalty`, `repeat_last_n`,
  `penalize_newline`, `num_ctx`, `mirostat*`, `tfs_z`, and `typical_p`), pass
  them through to backend inference requests, and include them in receipt
  decoding-parameter hashes.
- OpenAI-compatible chat requests preserve `parallel_tool_calls` through the
  backend/proxy inference path and include it in receipt decoding-parameter
  hashes.
- Added opt-in proxy upstream effective-prompt digest support. Proxy receipts
  remain non-overclaiming by default; when configured, Power asks the upstream
  digest endpoint for a `chat.rendered-prompt` SHA-256 before inference and can
  fail closed if the endpoint is required.
- Added mistralrs text-chat effective-prompt coverage using a
  domain-separated SHA-256 over the exact prompt token ID sequence returned by
  mistralrs' own tokenization path. This is emitted as
  `kind = "chat.prompt-token-ids"` rather than pretending to be rendered bytes.
- Fixed OpenAI-compatible message-level `images` forwarding into backend chat
  requests and mistralrs multimodal image detection/decoding for those
  message-level payloads, with regression coverage, so Ollama-native base64
  multimodal payloads are not dropped before inference.
- Fixed llama.cpp chat image collection to include request-level `images` and
  made llama.cpp multimodal chat leave `effective_prompt` absent instead of
  overclaiming a text-only rendered prompt digest.
- Made picolm reject image-bearing chat requests fail-closed and leave
  `effective_prompt` absent for image-bearing requests instead of silently
  rendering only the text parts.
- Fixed proxy chat request bodies to preserve structured multimodal content,
  message images, tools, tool choice, parallel tool-call policy, response
  format, and extended sampling controls when forwarding inference and when
  requesting an upstream
  `effective_prompt` digest.
- Added regression tests for strict policy, runtime model-hash binding, stale
  model-file rejection, GPU confidential fail-closed behavior, runtime policy
  digest binding, proxy and mistralrs effective-prompt digest handling, request
  receipts, multimodal receipt input-digest coverage, and strict verifier
  behavior.
- Fixed live `a3s-power-verify --url` attestation fetches to build URLs with a
  URL parser, preserving deployment base paths, dropping stale query/fragment
  components, and percent-encoding query parameters so model names containing
  query-special characters are not misparsed during verifier-side nonce/model
  binding.
- Fixed `a3s-power` model-management CLI URL construction to percent-encode
  model names as path segments for `models show` / `models rm`, so names with
  `/`, `:`, spaces, or query-special characters resolve to the intended model.
- Fixed remote model hub URL construction to use a URL parser for download and
  file-list endpoints, preserving intended repo/file subdirectories while
  percent-encoding spaces and query-special characters and rejecting empty or
  dot path segments in local references.
- Fixed remote hub token resolution to use the selected source's token
  environment variable plus the generic `A3S_POWER_HUB_TOKEN`, avoiding
  accidental cross-use of HuggingFace and ModelScope credentials.
- Made `A3S_POWER_MODEL_SOURCE` fail closed for unknown configured values
  instead of silently falling back to ModelScope.
- Extended receipt policy pins so `ExpectedReceipt` / `verify_receipt_policy`
  can directly require a matching `effective_prompt` digest, while still
  detecting conflicts with explicit effective-prompt absence.
- Added receipt coverage and verifier pins for `stream_options`, so clients can
  distinguish streaming output protocol choices such as `include_usage`.
- Added `verify_receipt_matches_chat_request()` and
  `verify_receipt_matches_completion_request()` so SDK verifiers can recompute
  and compare all request-derived receipt fields from the original request,
  including multimodal input digests and exact presence/absence of optional
  output-policy digests, before separately checking attestation runtime policy
  or effective-prompt pins.
- Exposed the same original-request receipt comparison through
  `a3s-power-verify --receipt-chat-request-file` and
  `--receipt-completion-request-file`.
- Added strict hardware-verifier operations guidance for AMD SEV-SNP and Intel
  TDX, covering `hw-verify` builds, raw report requirements, AMD KDS / Intel
  PCS outbound access, measurement pins, and production failure handling.
- Added `a3s-power-verify --hw-cert-cache-ttl-secs` so strict verifier
  operators can tune AMD KDS / Intel PCS certificate cache lifetime without
  weakening hardware-signature or measurement requirements.
- Tightened the GPU confidential verifier profile so it requires the top-level
  GPU evidence nonce claim in addition to structured device `eat_nonce`
  freshness checks.
- Consolidated image-bearing chat request detection in
  `ChatRequest::has_image_inputs()` and switched llama.cpp, picolm, and
  mistralrs effective-prompt/rejection gates to that single source of truth, so
  opaque multimodal paths keep `effective_prompt` absent consistently.
- Tightened direct `nras-rest` evidence normalization so DeviceEvidence
  `evidence` and `certificate` fields must be non-empty base64/base64url before
  Power posts them to NVIDIA NRAS.
- Added a 64 MiB cap for configured GPU evidence and verdict file/hex byte
  sources so malformed deployments fail before unbounded reads or hex decodes.
- Tightened `nras-rest` bearer-token configuration so
  `nras_bearer_token_env` is treated as an environment variable name, trimmed,
  and rejected when empty or not a portable ASCII identifier before Power sends
  NRAS requests; token values are also trimmed and rejected when empty or not
  visible ASCII before they are placed in the Authorization header.
- Tightened direct `nras-rest` endpoint normalization so custom URLs must be an
  HTTPS service root/base path or the full `/v4/attest/gpu` endpoint, without
  embedded credentials, query strings, fragments, or unsupported API-version
  paths; `gpu-confidential` startup validation now applies the same endpoint
  policy before the provider is constructed.
- Tightened direct `nras-rest` detached EAT parsing so only explicit EAT fields
  are decoded as JWTs, malformed token payloads fail closed, and ordinary
  version-like strings elsewhere in the response are ignored.

Still open:

- Expose exact post-template prompt representations for remaining opaque
  multimodal paths, or keep requiring those paths to leave `effective_prompt`
  absent.
- Native NVIDIA GPU confidential-computing NRAS SDK integration. The current
  implementation supports configured evidence/verdict bytes, live
  `nvattest-cli` collection, and direct `nras-rest` attestation, hashes the
  evidence and verdict, extracts structured device claims when verdict claims
  are exposed, and binds them into the CPU TEE report.
- Effective post-template-render request receipts are partial. The current
  receipt covers prompt-bearing API input, model runtime
  chat-template/GPU execution policy claims, normalized exposed
  decoding parameters, stop tokens, response format, tools, tool choice, and
  parallel tool-call policy.
  llama.cpp and picolm text-only chat additionally emit an `effective_prompt`
  digest for the exact rendered chat prompt; proxy can emit an
  upstream-declared digest through an explicit opt-in endpoint; mistralrs text
  chat emits a prompt-token-ID digest. Remaining opaque multimodal paths leave
  that field absent.

## Baseline Findings

### Model hash enforcement is optional

At the reviewed baseline, `tee_mode = true` did not require `model_hashes` or
model signatures. Startup only called `verify_all_models` when the map was
non-empty, and model load only rechecked models that appeared in the map.

Risk: a production deployment can expose `/v1/attestation` while serving a model
that has no pinned integrity policy.

### `/v1/attestation` can bind stale or missing model state

At the reviewed baseline, `GET /v1/attestation?model=...` used
`config.model_hashes[model]` when present, falling back to `manifest.sha256`.
It did not re-hash the loaded file or the in-memory model buffer on the
attestation request path.

If the selected hash was empty or invalid, the baseline code could silently omit the
model binding instead of failing the request.

Risk: a verifier may accept registration-time metadata as runtime model state.

### Launch measurement does not include model weights

SEV-SNP and TDX launch measurements cover the guest image and boot-time state,
not model files read from disk after boot.

Risk: the post-boot model integrity chain must be explicit, mandatory in
production policy, and reflected in the attestation claims.

### Encrypted model hash semantics are ambiguous

Power supports encrypted `.enc` models and multiple decryption modes, but the
reviewed baseline did not explicitly type the attested "model hash" as
ciphertext hash, plaintext hash, or a composite claim.

There was also an implementation gap to address: in the reviewed baseline,
`MemoryDecryptedModel` and `LayerStreamingDecryptedModel` were stored in
`AppState`, but the backend `load()` interface still received a `ModelManifest`
path and backends loaded from that path. Current remediation adds an explicit
locked-memory backend load path; `picolm` consumes `MemoryDecryptedModel`
plaintext and `LayerStreamingDecryptedModel` plaintext for GGUF models, and
unsupported backends fail closed.

Risk: different paths can hash or serve different bytes if a backend mode does
not explicitly declare which plaintext source it consumes. The current backend
trait therefore requires explicit support for each decrypted plaintext source.

### Hardware signature verification is optional

`VerifyOptions.hardware_verifier` is optional and the verification CLI currently
uses `None`, reporting a skipped hardware signature check as an otherwise
successful attestation.

Risk: integrators can accidentally deploy an offline or partial verifier as if
it were a production verifier.

### Simulated TEE can pass unless policy rejects it

At the reviewed baseline, `A3S_TEE_SIMULATE=1` enabled simulated reports. The
default policy allowlist was empty, which meant all TEE types were accepted
unless strict policy was explicitly enabled.

Risk: development evidence can be accepted in production.

### CPU and NVIDIA GPU evidence are not bound

Power supports NVIDIA CUDA execution paths, and NVIDIA GPU support is a
production requirement. However, the current `report_data` layout is only
`[nonce(32)][model_hash(32)]` and there is no NVIDIA GPU confidential-computing
evidence or NRAS verdict hash in the CPU TEE report.

Risk: CPU TEE evidence and GPU confidential-computing evidence can be checked as
separate statements without cryptographic proof that they describe the same
computation.

### Runtime prompt policy is outside attested claims

Chat templates, system prompts, stop sequences, and sampling parameters affect
the computation but are not included in the attestation payload or a request
receipt.

Risk: "the model ran" does not imply "the model ran with the verifier's intended
prompt construction and decoding policy."

## Remediation Requirements

1. Production verification must fail closed.
2. A missing, malformed, stale, or unverified model digest must fail the
   attestation request.
3. Every attested digest must state exactly what bytes it covers.
4. Simulated TEE evidence must be rejected by default outside explicit
   development mode.
5. Hardware signature verification must be mandatory for production verifier
   paths.
6. NVIDIA GPU confidential-computing evidence must be represented and bound to
   the CPU TEE evidence before Power claims a unified CPU/GPU attestation chain.
7. Runtime prompt construction and decoding policy must be hashable, stable, and
   available to verifiers.
8. Documentation must distinguish development, CPU-only TEE, CPU TEE plus
   ordinary CUDA, and CPU TEE plus NVIDIA confidential GPU modes.

## Implementation Plan

### Phase 0: Coordination and disclosure tracking

Deliverables:

- Open a private security advisory or private issue tracking the disclosure.
- Acknowledge the findings and maintain a coordinated disclosure timeline.
- Add this plan to the repository as the working remediation checklist.

Acceptance:

- The advisory or internal issue references every finding from the report.
- The public docs are not changed to overclaim before fixes land.

### Phase 1: Strict production policy

Code changes:

- Add an explicit `attestation_policy` or `tee_policy_mode` config with values
  such as `development`, `strict`, and `gpu-confidential`.
- In `strict` and `gpu-confidential` modes, reject:
  - empty `model_hashes` or missing model signatures for local models;
  - `tee_type = "simulated"`;
  - missing, malformed, or non-48-byte expected measurements for the detected
    hardware TEE type;
  - remote proxy models unless they have a separate declared trust mode.
- Make `A3S_POWER_TEE_STRICT=1` map to strict policy, but do not make
  environment variables the only way to get safe behavior.

Tests:

- Server startup fails when `tee_mode = true` and strict policy has no pinned
  local model digest.
- Server startup fails for `A3S_TEE_SIMULATE=1` under strict policy.
- Existing development tests still pass when policy is explicitly development.

### Phase 2: Model digest source of truth

Code changes:

- Introduce typed digest claims:
  - `ModelDigestKind::PlaintextWeightsSha256`
  - `ModelDigestKind::CiphertextArtifactSha256`
  - `ModelDigestKind::DirectoryManifestSha256`
  - `ModelDigestKind::RemoteUnattested`
- Add a canonical model claim builder that computes the digest from the loaded
  artifact or verified decrypted buffer.
- Make `/v1/attestation?model=...` fail if it cannot produce a valid typed
  32-byte digest claim.
- Remove silent `decode_hex_nonce(...).ok()` downgrade behavior.
- For HuggingFace directory models, add a deterministic directory manifest hash
  over file paths, file modes where relevant, sizes, and file SHA-256 values.

Tests:

- `/v1/attestation?model=...` returns an error for empty or malformed hashes.
- Tampering with a model file after registration is detected before issuing an
  attestation claim in strict mode.
- Directory models produce stable hashes independent of directory traversal
  order.

### Phase 3: Encrypted model loading and digest semantics

Code changes:

- Split encrypted model claims into ciphertext and plaintext digests.
- Compute plaintext digest after successful AES-GCM decryption.
- Update backend loading interfaces so in-memory decrypted models can be loaded
  from verified plaintext buffers, not from the encrypted path.
- If a backend cannot consume in-memory plaintext, fail closed rather than
  silently falling back.
- Document which digest verifiers should pin for each deployment mode.

Current status:

- Plaintext and ciphertext encrypted-model digest claims are implemented.
- File-backed `DecryptedModel` loading remains available for file-based
  backends.
- Backend trait/source changes for verified plaintext buffers are implemented.
  `in_memory_decrypt` loads GGUF plaintext through `picolm` locked-memory
  loading; unsupported backends fail closed before backend load.
- `streaming_decrypt` loads GGUF plaintext through `picolm` from
  `LayerStreamingDecryptedModel`; unsupported backends fail closed before
  backend load. The current AES-GCM artifact format is still a single
  non-seekable ciphertext, so the decrypted plaintext remains locked in RAM
  while the handle is live.

Tests:

- `.enc + in_memory_decrypt` loads from decrypted bytes on a supporting backend
  and fails before backend load on unsupported backends.
- `.enc + streaming_decrypt` loads a real encrypted synthetic GGUF through
  `picolm`, verifies the plaintext digest pin, does not materialize a `.dec`
  file, and runs chat from the decrypted backend-owned source.
- Ciphertext tampering fails during decryption.
- Plaintext digest mismatch fails before attestation.
- File-backed `.dec` mode reports and verifies the same plaintext digest as
  in-memory mode.

### Phase 4: Attestation claim schema v2

Code changes:

- Replace the fixed `[nonce(32)][model_hash(32)]` semantic with a canonical
  `AttestationClaimsV2` JSON or CBOR structure. Initial JSON support is in
  place for model-bound attestations.
- Bind `sha256(canonical_claims_v2)` into CPU TEE `report_data`. Initial
  binding is in place for model-bound attestations.
- Include at minimum:
  - schema version;
  - nonce hash or nonce bytes;
  - tee type;
  - model digest kind and digest;
  - encrypted artifact digest, when applicable;
  - launch measurement;
  - runtime policy mode;
  - chat template hash, when known at model load time;
  - decoding policy hash, when tied to a request receipt;
  - GPU evidence hash, when GPU confidential mode is active.
- Preserve backward-compatible verification for legacy reports but mark legacy
  binding as development or compatibility only. Verifier compatibility is in
  place: legacy reports use the old offsets, v2 reports verify claims first.

Tests:

- Canonical claim serialization is deterministic.
- CPU report data equals the claim digest.
- Verifier rejects reports whose claims do not match CPU `report_data`.
- Legacy reports do not satisfy strict v2 policy.

### Phase 5: Production verifier API and CLI

Code changes:

- Add `verify_report_strict` or a `VerificationPolicy` object that requires:
  - hardware signature verification;
  - nonce verification when policy declares nonce mandatory;
  - model digest verification when policy declares model digest mandatory;
  - launch measurement verification in strict mode;
  - rejection of simulated TEE.
- Make `a3s-power-verify` default to strict verification unless an explicit
  offline/development flag is provided, and require `--expected-measurement`
  in strict mode.
- Add `--hw-cert-cache-ttl-secs` to tune the in-memory AMD KDS / Intel PCS
  certificate cache for long-running verifier processes.
- Add CLI flags for AMD KDS, Intel PCS, and NVIDIA GPU evidence verification
  modes where needed.
- Make skipped checks visible as non-success status in strict mode.

Tests:

- `verify_report_strict` fails when `hardware_verifier` is missing.
- `verify_report_strict` fails when the expected launch measurement is missing.
- `a3s-power-verify` exits non-zero when hardware verification is skipped in
  strict mode or when the launch measurement pin is missing.
- Simulated reports require an explicit development/offline flag.

### Phase 6: NVIDIA GPU confidential-computing chain

Code changes:

- Added a GPU attestation module with a provider trait, `GpuEvidenceProvider`.
  The compatibility provider consumes configured evidence and NRAS verdict byte
  sources. For nonce-bound attestations, configured verdicts must be parseable
  NVIDIA nvattest/NRAS JSON whose `eat_nonce` matches the CPU attestation
  nonce; stale verdicts fail closed.
- Added `gpu_attestation.source = "nvattest-cli"` to invoke NVIDIA's
  `nvattest` CLI for live GPU evidence collection and NRAS/local attestation.
  The provider passes the same nonce used by the CPU TEE attestation request to
  `nvattest collect-evidence` and `nvattest attest`, then binds SHA-256 digests
  of the raw evidence and verdict JSON. `gpu-confidential` startup requires
  `nvattest_verifier = "remote"` so NVIDIA NRAS performs production evidence
  verification.
- Added `gpu_attestation.source = "nras-rest"` to send configured NVIDIA
  DeviceEvidenceV2 JSON directly to the NVIDIA NRAS REST `/v4/attest/gpu` API.
  The provider requires a 32-byte nonce, GPU architecture (`HOPPER` or
  `BLACKWELL`), and claims version (`2.0` or `3.0`), then binds SHA-256 digests
  of the raw evidence JSON and returned detached EAT JSON.
- Added `GpuEvidenceClaim` with:
  - provider name;
  - evidence and verdict byte-format labels;
  - raw evidence entry count when the provider exposes it;
  - raw evidence digest;
  - verdict digest;
  - nonce linkage through the shared CPU/GPU attestation nonce;
  - structured NVIDIA device claims from `nvattest` verdicts: device type,
    `eat_nonce`, hardware model, UEID/OEM ID, driver/firmware versions,
    measurement result, secure-boot/debug status, and normalized
    validation booleans for nonce match, report signature, FWID match, RIM
    schema validation, RIM signature, RIM version match, and measurement
    availability.
- In `gpu-confidential` policy mode, require a GPU evidence claim and bind its
  evidence/verdict digests into `AttestationClaimsV2`, which is then bound into
  CPU TEE `report_data`.
- In `gpu-confidential` policy mode, `/v1/attestation` rejects missing or
  non-32-byte nonce values so CPU TEE evidence and NVIDIA GPU evidence share
  one 32-byte freshness value.
- In `gpu-confidential` policy mode, startup rejects CPU-only execution policy
  (`gpu.gpu_layers = 0`) after GPU auto-configuration. Use `strict` mode for
  CPU-only TEE deployments.
- In `gpu-confidential` policy mode, startup rejects custom NVIDIA NRAS, RIM,
  and OCSP URLs that are not HTTPS for live `nvattest-cli`; direct `nras-rest`
  also rejects custom NRAS URLs that are not HTTPS. Both startup validation and
  the direct NRAS REST provider reject embedded URL credentials so bearer tokens
  stay in `nras_bearer_token_env`.
- Make ordinary CUDA acceleration a separate declared mode that does not claim
  GPU confidential-computing guarantees.

Tests:

- `gpu-confidential` policy refuses to start or attest when no GPU evidence
  provider is configured.
- `gpu-confidential` policy refuses to start when GPU execution/offload remains
  disabled.
- `gpu-confidential` policy refuses to start with `nvattest-cli` local
  verification.
- `gpu-confidential` policy refuses to start when a custom NVIDIA NRAS, RIM, or
  OCSP URL uses HTTP instead of HTTPS.
- `gpu-confidential` policy refuses to start when a direct `nras-rest`
  endpoint includes query strings, fragments, embedded credentials, or
  unsupported API-version paths.
- `gpu-confidential` attestation refuses to proceed without a 32-byte nonce.
- The configured evidence provider rejects nonce-bound attestations when the
  configured NVIDIA verdict is stale or not parseable as nvattest/NRAS JSON.
- `nvattest-cli` provider tests simulate the CLI and verify 32-byte nonce
  enforcement and propagation, temporary evidence-file handling,
  evidence/verdict digest binding, structured device-claim extraction,
  owner-only temporary evidence-file permissions on Unix, and rejection of
  failed core NVIDIA validation booleans.
- `nras-rest` provider tests verify official request-shape construction,
  evidence normalization, 32-byte nonce enforcement, NRAS verdict digest
  binding, structured device-claim extraction when claims are present, and
  fail-closed config validation, including oversized NRAS response bodies and
  malformed detached EAT JWTs.
- Verifier policy `require_gpu_evidence()` now also requires an expected nonce
  and rejects GPU claim nonce mismatches.
- Verifier policy `require_gpu_device_claims()` requires structured NVIDIA
  device claims and validates device nonce, `measres`, `secboot`, `dbgstat`,
  and core NVIDIA validation booleans, including RIM schema validation.
- Verifier `ExpectedGpuEvidence` and `verify_claims_expected_gpu_evidence()`
  pin the GPU evidence provider label, evidence/verdict byte-format labels, and
  raw evidence entry count. The `a3s-power-verify` CLI exposes the same checks
  through `--gpu-provider`, `--gpu-evidence-format`, `--gpu-verdict-format`,
  and `--gpu-evidence-count`.
- Verifier `ExpectedGpuDevices` and `verify_claims_expected_gpu_devices()` pin
  exact GPU/NVSwitch device counts, exact GPU UEID sets, allow-list NVIDIA GPU
  and NVSwitch UEID sets, allow-list NVIDIA GPU/NVSwitch OEM IDs and claims
  schema versions, and allow-list expected GPU/NVSwitch hardware models and
  firmware versions plus GPU driver versions. The `a3s-power-verify` CLI
  exposes the same checks through `--gpu-count`, `--nvswitch-count`,
  `--gpu-claims-version`, `--gpu-ueid`, `--gpu-oemid`, `--gpu-hwmodel`,
  `--gpu-driver-version`, `--gpu-firmware-version`,
  `--nvswitch-claims-version`, `--nvswitch-ueid`, `--nvswitch-oemid`,
  `--nvswitch-hwmodel`, and `--nvswitch-firmware-version`.
- Verifier `VerificationPolicy::gpu_confidential()` and CLI
  `--gpu-confidential` bundle the production NVIDIA GPU CC policy so callers do
  not have to remember each individual GPU evidence, GPU provider/format/count
  pin, 32-byte nonce requirement, structured device-claim, exact GPU topology
  pin, GPU claims-version pin, GPU identity/version pin, required NVSwitch
  identity/version pins when NVSwitch claims are present, secure-boot/debug-state
  check, runtime-policy, and GPU execution/offload requirement.
- CPU report data changes when GPU evidence changes.
- Verifier rejects mismatched CPU claims and GPU evidence.
- Ordinary CUDA mode remains usable but is reported as outside GPU CC
  attestation.

Remaining gap:

- Add a native NRAS SDK client, if needed, so production deployments can choose
  between the current REST provider, `nvattest-cli`, or a vendor SDK binding.

### Phase 7: Runtime receipt for prompts and decoding policy

Current implementation:

- Model-bound `/v1/attestation?model=...` includes `runtime.prompt` digests for
  applied manifest chat template overrides or readable GGUF
  `tokenizer.chat_template` metadata. Manifest system prompts and pre-seeded
  messages are not claimed until an execution path actually injects them into
  inference requests.
- Model-bound attestation does not currently emit
  `runtime.decoding.parameters_sha256` for manifest default generation
  parameters, because the OpenAI-compatible inference path does not apply those
  defaults automatically. Request-level receipts cover the decoding parameters
  that are actually sent on each request.
- Model-bound attestation includes `runtime.execution.gpu_sha256` for the
  canonical GPU execution/offload configuration: `gpu_layers`, `main_gpu`, and
  `tensor_split`.
- Verifiers can pin these digests with
  `VerificationPolicy::require_runtime_policy()` or
  `a3s-power-verify --require-runtime-policy`; GPU execution/offload pins use
  `--gpu-execution-digest`, and the CLI can compute the matching pin with
  `--print-gpu-execution-digest` from `gpu_layers`, `main_gpu`, and
  `tensor_split`.
- `/v1/chat/completions` and `/v1/completions` build an
  `a3s.power.inference-receipt.v2` receipt from prompt-bearing API input,
  model runtime chat-template/GPU execution policy claims, and
  normalized exposed decoding parameters.
- Non-streaming responses return `attestation_receipt` and
  `attestation_receipt_sha256`.
- Streaming responses emit a final receipt event before `[DONE]`; if usage
  reporting is requested or token metrics are suppressed, the same final event
  also carries rounded `usage`.
- Verifiers can check response receipt integrity with
  `verify::verify_receipt_digest()` or `verify::verify_receipt_digest_hex()`.
- Verifiers can reject malformed receipts with
  `verify::verify_receipt_well_formed()`, which checks schema,
  request-type/input-kind pairing, and SHA-256 hex fields before policy
  comparison.
- Verifiers can pin receipt-level request policy with
  `verify::verify_receipt_policy()` / `ExpectedReceipt` or CLI flags for
  receipt model, request type, input digest, request decoding parameters,
  stream-options / stop-token / response-format / tools / tool-choice digests,
  and effective-prompt digest/backend/kind or explicit effective-prompt absence
  for opaque multimodal paths. The lower-level
  `verify::verify_receipt_effective_prompt_digest()` and
  `verify::verify_receipt_effective_prompt_digest_hex()` helpers remain
  available for direct digest-only checks.
- SDK verifiers that still have the original request can use
  `verify::verify_receipt_matches_chat_request()` or
  `verify::verify_receipt_matches_completion_request()` to recompute and compare
  every request-derived receipt field, including exact optional output-policy
  digest presence or absence.
- Verifiers can compare a response receipt against the attested runtime policy
  and optional receipt/effective-prompt digest pins with
  `verify::verify_receipt_against_attestation()`.
- `a3s-power-verify` exposes the same attestation-to-receipt check with
  `--receipt-file`, `--receipt-digest`, receipt policy digest pins,
  `--receipt-chat-request-file`, `--receipt-completion-request-file`, and
  `--effective-prompt-digest`.
- Proxy backends can ask upstreams for an explicit rendered prompt digest before
  inference with `proxy_effective_prompt_digest = true`. The upstream endpoint
  receives the OpenAI-compatible chat body with `stream = false` and returns a
  `chat.rendered-prompt` SHA-256 claim. Unsupported endpoints leave
  `effective_prompt` absent unless
  `proxy_effective_prompt_digest_required = true`; malformed digests fail
  closed.
- mistralrs text chat emits `kind = "chat.prompt-token-ids"` with a
  domain-separated SHA-256 over the exact token IDs produced by
  `Model::tokenize(..., add_special_tokens = true, add_generation_prompt =
  true)`, matching the path used before generation. mistralrs vision/multimodal
  requests still leave `effective_prompt` absent because image prefixing and
  multimodal preprocessing are not exposed as a stable rendered-byte claim.

Remaining gap:

- Effective prompt digests are implemented for local deterministic text-only
  chat renderers (llama.cpp and picolm), and for proxy upstreams that implement
  the explicit digest endpoint. mistralrs text chat is covered by
  prompt-token-ID digest. llama.cpp, picolm, and mistralrs vision/multimodal
  paths still leave `effective_prompt` absent until they can expose the exact
  prompt representation submitted to the model.

Remaining code changes:

- Extend effective prompt digest support only to backends that can prove the
  exact prompt representation submitted to the model.
- Continue extending the receipt whenever new request-visible decoding controls
  are exposed by the OpenAI-compatible API.
- If manifest system prompts, pre-seeded messages, or default generation
  parameters become execution defaults, apply them in the request path first and
  then add matching runtime-policy claims and receipt tests.

Tests:

- Same request and same template produce identical receipt hashes.
- Changing any sampling or stop-token value changes the receipt hash.
- Changing the attested GPU execution/offload configuration changes the runtime
  policy digest.
- Template override changes the effective prompt-policy or receipt hash for
  backends that expose `effective_prompt`.
- Receipt generation is available for streaming and non-streaming responses.
- llama.cpp, picolm, and mistralrs image-bearing chat paths keep
  `effective_prompt` absent unless the backend can expose the exact
  post-template multimodal representation.
- Shared image-input detection covers top-level request images, per-message
  image arrays, and OpenAI `image_url` content parts.

### Phase 8: Documentation and release notes

Docs to update:

- `README.md`
- `PLAN.md`
- `docs/supply-chain.md`
- Power docs under `apps/docs/content/docs/en/power/` and
  `apps/docs/content/docs/cn/power/`

Required documentation changes:

- Remove or qualify claims that current default settings prove "which model ran"
  without strict model digest policy.
- Explain legacy attestation versus v2 claim binding.
- Document the difference between:
  - development simulation;
  - CPU-only TEE;
  - CPU TEE plus ordinary CUDA;
  - CPU TEE plus NVIDIA GPU confidential computing.
- Add production configuration examples using strict policy.
- Add verifier examples that fail closed.

Tests:

- Documentation examples compile or run where practical.
- CLI help text matches the strict defaults.

## Suggested Work Order

1. Phase 1 and Phase 2 first. They close the easiest fail-open paths.
2. Phase 5 next, so the verifier and CLI stop producing false confidence.
3. Keep expanding Phase 3 validation when additional encrypted-model artifact
   formats or backends consume decrypted plaintext sources.
4. Phase 4 as the schema migration foundation.
5. Phase 6 for NVIDIA GPU production support.
6. Phase 7 for request-level receipts.
7. Phase 8 continuously, with final documentation updates before release.

## Immediate Patch Targets

- `src/config.rs`: add policy mode and strict validation.
- `src/server/mod.rs`: fail closed during TEE startup under strict policy.
- `src/api/openai/attestation.rs`: replace cached hash fallback with typed
  digest claim builder and fail on invalid model binding.
- `src/tee/attestation.rs`: add v2 claim digest binding.
- `src/verify/mod.rs`: continue evolving strict verification policy toward v2
  claim verification.
- `src/bin/a3s-power-verify.rs`: add production configuration examples and
  operational flags for vendor verifier settings.
- `src/tee/encrypted_model.rs` and backend load interfaces: complete chunked
  decrypted loading semantics.
- GPU attestation module: add native NVIDIA NRAS SDK collection in addition to
  the direct REST and live `nvattest-cli` providers.

## Completion Criteria

The remediation is complete only when all of the following are true:

- Strict TEE mode cannot start without pinned local model integrity policy.
- Simulated reports are rejected unless development mode is explicit.
- `/v1/attestation?model=...` never silently omits model binding.
- Verifiers can prove that CPU `report_data` binds the full canonical claim set.
- Hardware signature verification is required in production verifier paths.
- Encrypted model claims identify plaintext and ciphertext digests precisely.
- NVIDIA GPU confidential-computing evidence is verified and bound into CPU TEE
  evidence in GPU confidential mode.
- Chat template and decoding policy can be independently hashed and verified.
- Documentation no longer overstates legacy or non-strict guarantees.
