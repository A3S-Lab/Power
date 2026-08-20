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

## Verify a running service

```bash
a3s-power-verify \
  --url https://power.example.com \
  --model your-model \
  --nonce <fresh-client-nonce-hex> \
  --model-hash <64-character-artifact-sha256> \
  --expected-measurement <96-character-launch-measurement-hex>
```

The verifier - not the server operator - chooses the accepted launch
measurement, artifact hash, runtime policy, GPU evidence, and receipt fields.

## Hardware evidence

| TEE | Strict verification |
| --- | --- |
| AMD SEV-SNP | Raw report parsing, nonce and measurement binding, VCEK retrieval, and ECDSA P-384 signature verification |
| Intel TDX | Fails closed: the current local TDREPORT is not a remotely verifiable DCAP Quote; Quote generation and QVL are pending |
| NVIDIA confidential GPU | Fresh device claims, firmware and topology policy, pinned NRAS verdict, and GPU execution digest |

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
cannot mint confidential release evidence. The resulting bundle still needs an
authenticated release trust root.

The current [clean-revision CPU/CUDA captures](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/release-contract-windows-20260821/README.md)
replay as one verified two-platform partial bundle. Metal and confidential-GPU
captures remain required before the strict four-platform v1 policy can pass.

## Production-blocking failures

- Missing hardware-verification support in a strict verifier build.
- Missing or malformed launch-measurement and artifact pins.
- Missing raw report bytes in saved evidence.
- Vendor certificate retrieval, parsing, or signature failures.
- Stale nonces or mismatched model, policy, input, output, or device digests.
- Simulated or `tee_type=none` reports on a strict path.

Read [Hardware Verifier Operations](https://github.com/A3S-Lab/Power/blob/main/docs/hardware-verifier-operations.md)
for certificate services, cache behavior, saved evidence, and failure policy.
