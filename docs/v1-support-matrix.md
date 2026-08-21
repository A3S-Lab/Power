# v1 Production Support Matrix

This document is the release boundary for A3S Power v1. A feature is production
supported only when the tagged revision's checked-in release bundle passes the
machine-enforced evidence gate. Build support, unit tests, or a single fast
benchmark do not by themselves establish production support.

## Execution platforms

| Platform class | v1 status | Required evidence |
| --- | --- | --- |
| CPU | Required | Complete contract capture on named hardware |
| CUDA | Required | Complete local CUDA contract capture on named hardware |
| Metal | Required | Complete Metal contract capture on named Apple hardware |
| Confidential GPU | Required | Distinct CUDA capture promoted by a strict SEV-SNP and NVIDIA NRAS proof |

All four captures must bind one exact Power version and commit, weights digest,
and reviewed graph identity. Platform-specific executable, device, topology,
shape-profile, memory, and TEE-policy bindings remain distinct. One tensor
benchmark cannot be reused for two platform classes.

The release contract is model-, format-, and backend-neutral. A model
integration may use GGUF, SafeTensors, llama.cpp, a native Rust backend, or
another reviewed implementation, but it must present the same bounded and
digest-bound evidence.

## Confidential-computing boundary

| CPU TEE | v1 status | Reason |
| --- | --- | --- |
| AMD SEV-SNP | Supported for the confidential-GPU release class | Power verifies the signed report and binds canonical claims, nonce, model, runtime policy, and NVIDIA evidence |
| Intel TDX | Explicitly unsupported | A local TDREPORT is not a remotely verifiable DCAP Quote; reviewed Quote generation, QVL verification, collateral freshness, and exact REPORTDATA/MRTD binding are not implemented |
| Simulated or no TEE | Unsupported | Development modes cannot mint production security evidence |

Generic evidence schemas retain a typed TDX value for forward-compatible
development and later policy revisions. The v1 verifier nevertheless requires
`sev-snp` in the confidential release binding and fails closed for TDX. Enabling
a custom hardware verifier cannot bypass that release rule.

## Release artifact contract

Every non-`0.x` release currently uses the strict v1 gate and must check in:

```text
release/v<crate-version>/
|-- release-evidence.json
`-- release-evidence.sha256
```

The pin file contains exactly one lowercase SHA-256 digest, optionally followed
by one LF or CRLF line ending. The bundle is a bounded, unknown-field-denying
JSON document. From the tagged clean checkout, reproduce the release decision
with:

```bash
version="$(cargo metadata --locked --no-deps --format-version 1 \
  | jq -r '.packages[] | select(.name == "a3s-power") | .version')"
commit="$(git rev-parse HEAD)"

cargo run --locked --release --no-default-features \
  --features embedded-inference \
  --bin a3s-power-tensor-batch-bench -- \
  verify-release-bundle \
  --bundle "release/v${version}/release-evidence.json" \
  --expected-sha256-file "release/v${version}/release-evidence.sha256" \
  --power-version "${version}" \
  --power-commit "${commit}"
```

Verification recomputes the bundle and nested digests, requires the exact four
platform classes, checks the external digest pin, compares the exact version and
Git revision, and requires SEV-SNP for the confidential-GPU capture. The release
workflow runs this command before any `v1.x` or later tag can publish artifacts.

The checked-in digest is a mutation-detection pin. Release authorship still
requires the repository's signed tag/release trust root and preserved raw
hardware evidence described in the external capture guide.

## Current readiness

The verifier, capture contracts, and fail-closed workflow are implemented.
Actual same-revision Metal and SEV-SNP confidential-GPU captures have not yet
been published, so v1.0.0 is not ready to tag. CPU/CUDA pre-release captures are
useful regression evidence but cannot satisfy the four-platform gate.

See [Production Release Evidence Gate](release-evidence-gate.md) for schema and
contract details and [External Metal and Confidential-GPU Capture](external-release-capture.md)
for the hardware procedure.
