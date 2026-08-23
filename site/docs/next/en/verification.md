---
title: Verification
description: Client-owned verification of A3S Power model identity, runtime policy, TEE reports, accelerator evidence, and response receipts.
---

# Verification

Attestation is useful only when the party relying on an answer controls the
acceptance policy. Power therefore separates evidence production from evidence
acceptance.

```text
model bytes + runtime policy
             |
             v
canonical claims digest + fresh nonce
             |
             v
CPU TEE report + optional GPU evidence
             |
             v
request, effective prompt, and output receipt
             |
             v
independent client verifier accepts or rejects
```

## What the receipt binds

A response receipt can commit to:

- the exact model artifact identity;
- the runtime and effective policy;
- input, effective-prompt, decoding, tool, output, and response digests;
- the selected accelerator or exact fallback path;
- fused-batch or heterogeneous device-mesh evidence;
- the CPU TEE report and optional GPU confidential-computing claims.

Fields that cannot be derived honestly remain absent. For example, opaque
multimodal renderer paths do not fabricate an effective-prompt digest.

## Build the strict verifier

```bash
cargo build --release --bin a3s-power-verify --features hw-verify
```

Without `hw-verify`, strict signature verification fails closed. The explicit
`--allow-offline` bypass exists for fixtures and offline inspection, not
production acceptance.

Confidential release promotion also needs the model-neutral embedded contract
types:

```bash
cargo build --locked --release --no-default-features \
  --features server,embedded-inference,hw-verify \
  --bin a3s-power-verify
```

## Verify a running service

```bash
a3s-power-verify \
  --url https://power.example.com \
  --model your-model \
  --nonce <fresh-client-nonce-hex> \
  --model-hash <64-character-artifact-sha256> \
  --inference-execution-digest <resolved-power-policy-sha256> \
  --auxiliary-artifacts-digest <portable-auxiliary-set-sha256> \
  --expected-measurement <96-character-launch-measurement-hex>
```

Derive the server policy pin from the reviewed ACL. If the model also uses an
external draft, LoRA adapter, or multimodal projector, derive its
path-independent expected value from the reviewed deployment manifest:

```bash
a3s-power-verify --print-inference-execution-digest power.acl
a3s-power-verify --print-auxiliary-artifacts-digest model-manifest.json
```

New local-model reports always declare the inference policy, so strict clients
pin its digest. Strict verification requires the auxiliary pin whenever the
attested runtime declares auxiliary artifacts. Roles, decoder contracts, byte lengths,
artifact hashes, and external-draft target binding are covered; local paths are
excluded.

The inference-execution digest covers normalized speculative/MTP/FR controls,
prompt-cache bounds, model residency, mmap/mlock, threads, Flash Attention, and
parallel request slots from the fully resolved ACL. Environment overrides
therefore change the digest. Configure an explicit `spec_mode` when acceptance
must prove one decoder; `auto` deliberately proves backend selection policy,
not a decoder chosen in advance.

The verifier - not the server operator - chooses the accepted launch
measurement, artifact hash, runtime policy, GPU evidence, and receipt fields.

## Hardware evidence

| TEE | Strict verification |
| --- | --- |
| AMD SEV-SNP | Raw report parsing, nonce and measurement binding, VCEK retrieval, and ECDSA P-384 signature verification |
| Intel TDX | Fails closed: the current local TDREPORT is not a remotely verifiable DCAP Quote; Quote generation and QVL are pending |
| NVIDIA confidential GPU | Fresh device claims, firmware and topology policy, pinned NRAS verdict, GPU placement digest, and inference-execution digest |

Power caches fetched AMD KDS certificate material in memory for one hour by
default. Operators can tune that cache, but a network or certificate failure
remains production-blocking unless an explicit reviewed offline certificate
design exists. No cache setting enables TDX verification.

## Configure strict policy

```text
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

Strict policy rejects simulated reports. CPU TEE placement also does not make
GPU offload confidential; a GPU path needs verified `gpu-confidential` claims.

## Model-neutral runtime release contracts

The release gate is separate from model-specific quality evaluation. It binds
one Power revision, exact weights, and one reviewed graph to platform-specific
shape profiles and TEE policies, then verifies scalar/batch parity, bounded
peak memory, active cancellation cleanup, queue expiry, replica recovery, and
an explicit exact fallback. The schema contains no Qwen, GGUF, tokenizer,
decoder, or model-family dispatch field.

The trusted construction path is type-safe. Strict confidential-GPU
verification returns an opaque proof bound to the exact report; only
`ReleaseCapture::promote_confidential_gpu` can use that proof to promote a valid
local CUDA capture. A raw report, deserialized label, or caller-authored boolean
cannot mint confidential release evidence. Promotion preserves the accepted
48-byte launch measurement, SHA-256 identity of the exact raw signed report,
inference-execution digest, and optional auxiliary-artifacts digest as explicit
capture fields; final bundle replay validates all of them. The resulting bundle
still needs an authenticated release trust root.

Validate each transferred capture before assembly:

```bash
a3s-power-tensor-batch-bench verify-release-capture \
  --capture <file> \
  --platform <cpu|cuda|metal|confidential-gpu> \
  --power-version <version> \
  --power-commit <revision>
```

The command checks bounded JSON, canonical digests, platform, and source
identity. Its receipt remains explicitly single-capture scope; only the strict
four-platform bundle can authorize a production release.

The current [clean-revision CPU/CUDA captures](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/release-contract-windows-20260821/README.md)
replay as one verified two-platform partial bundle. Metal and confidential-GPU
captures remain required before the strict four-platform v1 policy can pass.

## Verify benchmark evidence offline

Model-specific performance evidence is not a release attestation, but it is
still fail-closed and independently checkable. The DSpark quality package pins
the clean source and server, target and draft artifacts, task and ACL inputs,
six raw-report hashes, GPU admission windows, aggregates, and paired task
vectors:

```bash
python3 tools/qwen38_quality_evidence.py verify \
  --evidence docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark/quality/evidence.json \
  --json
```

The evidence passes integrity verification while declaring itself ineligible
as a production default. `--require-production-default` rejects it because
exact target/DSpark output parity is 54/100. Integrity, quality observation,
and deployment acceptance are separate decisions.

The current adaptive package binds the controlled peak and paired 100-task run
in one path-free document, including host-control attestations, raw-capture
hashes, runtime telemetry, and every paired task vector:

```bash
python3 tools/dspark_adaptive_evidence.py verify \
  --evidence docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark/adaptive/evidence.json \
  --json
```

It verifies a 164.756 token/s peak median, 160.881 minimum, zero replay, and a
1.358x request-wide quality-workload speedup. Production-default verification
still rejects it because the matrix contains three paired lenient losses and
only 55/100 complete-output parity.

The loss-focused follow-up independently binds the hash-locked five-task
selection, 512- and 1,024-token request identities, benchmark-tool hashes,
host controls, raw report hashes, and compact task vectors:

```bash
python3 tools/dspark_quality_followup_evidence.py verify \
  --evidence docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark/quality/followup-evidence.json \
  --json
```

It verifies zero paired answer losses and 5/5 untruncated answer parity at
1,024 tokens. Exact output parity remains 0/5, so the production-default gate
stays closed even though the selected-answer diagnostic passes.

## Production release trust chain

For v1 and later, hardware captures bind a frozen source commit. Its direct
child may add only `release/v<version>/release-evidence.json` and the matching
SHA-256 pin. That evidence child must already be on `main`, and a
GitHub-verified annotated tag must point directly to it. Release CI verifies the
two-commit layout and the four-platform bundle, then builds binaries and the
crate from the frozen parent. Lightweight, unverified, detached, or
extra-change tags fail before publication.

This split avoids a self-referential commit hash: the bundle authenticates the
source parent, while the signed child authenticates the bundle. Source code or
historical benchmark files alone are never sufficient evidence that a release
passed.

Run the same fail-closed candidate gate locally before creating the tag:

```bash
git fetch --no-tags origin +refs/heads/main:refs/remotes/origin/main
bash tools/verify-release-candidate.sh \
  --evidence-ref HEAD \
  --main-ref refs/remotes/origin/main
```

It accepts only a clean non-`0.x` evidence child with a finalized changelog,
remote-main containment, and a successfully replayed strict four-platform
bundle. Native Metal and proof-promoted SEV-SNP/NVIDIA evidence are still
mandatory; the preflight never substitutes synthetic or partial captures.

## Reproduce external hardware capture

The checked-in workflow runs the complete contract on a real Metal device, then
uses one fresh nonce to bind preserved NVIDIA evidence, the remote NRAS verdict,
the CPU TEE report, the canonical GPU and inference execution policies, any
auxiliary-artifact set, and a model-owned accelerator declaration.
`a3s-power-verify --promote-capture` consumes the strict proof in-process and
creates a new confidential capture without replacing an existing file.

Stage each host's complete raw artifact set under one dedicated read-only
directory and keep its manifest outside that directory. Build the exact
portable inventory before transfer, then replay it on the review host:

```bash
cargo run --locked --release --no-default-features \
  --features embedded-inference \
  --bin a3s-power-tensor-batch-bench -- \
  build-release-handoff --root ./metal-handoff --platform metal \
  --power-version "$power_version" --power-commit "$power_commit" \
  --output ./metal-handoff.manifest.json

cargo run --locked --release --no-default-features \
  --features embedded-inference \
  --bin a3s-power-tensor-batch-bench -- \
  verify-release-handoff --root ./metal-handoff --platform metal \
  --manifest ./metal-handoff.manifest.json \
  --power-version "$power_version" --power-commit "$power_commit"
```

Verification rejects changed, missing, extra, unsafe, symlinked/reparse, or
relabeled files. The path-free manifest still needs release-root
authentication and never replaces capture verification or the four-platform
bundle.

Read [External Metal and Confidential-GPU Release Capture](https://github.com/A3S-Lab/Power/blob/main/docs/external-release-capture.md)
for the exact commands, ACL, device pins, failure conditions, and artifact
inventory. Every production tag must carry same-parent Metal and
confidential-GPU proofs; absent or invalid hardware evidence blocks publication.

## Production-blocking failures

- Missing hardware-verification support in a strict verifier build.
- Missing or malformed launch-measurement and artifact pins.
- Missing or mismatched auxiliary-artifacts pins when the runtime declares a
  draft, adapter, or projector.
- Missing raw report bytes in saved evidence.
- Vendor certificate retrieval, parsing, or signature failures.
- Stale nonces or mismatched model, policy, input, output, or device digests.
- Simulated or `tee_type=none` reports on a strict path.

Read [Hardware Verifier Operations](https://github.com/A3S-Lab/Power/blob/main/docs/hardware-verifier-operations.md)
for certificate services, cache behavior, saved evidence, and failure policy.
