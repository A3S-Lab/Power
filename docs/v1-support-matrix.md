# v1 Production Support Matrix

This document is the release boundary for A3S Power v1. A feature is production
supported only when the tag's evidence-only commit and its frozen source parent
pass the machine-enforced evidence gate. Build support, unit tests, or a single
fast benchmark do not by themselves establish production support.

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

Strict construction rejects Metal records that do not name native Apple
Silicon/macOS execution or that contain virtual, emulated, translated,
fallback, software-renderer, or unnamed-device markers. This is a fail-closed
consistency check, not Apple hardware attestation; the signed raw host inventory
remains mandatory. In particular, `Apple Paravirtual device` is useful CI
preflight evidence but is not v1 production Metal evidence.

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
JSON document.

The release uses two commits so the evidence can bind an immutable source
revision without trying to contain its own Git hash:

```text
frozen source commit S
  `-- evidence commit E: adds exactly the JSON and SHA-256 files above
        `-- signed tag v<crate-version>
```

`E` must have exactly one parent, `S`; the evidence directory must be absent
from `S`; and the two regular files must be the complete diff. `E` must be
reachable from `main`. Release binaries and the crates.io package are built from
`S`. A GitHub-verified annotated tag and the GitHub source archive point to `E`,
which carries the authenticated evidence pair. Lightweight or unverified tags
fail the release workflow. Cargo excludes `release/` from the published crate.

From a clean checkout of `S`, assemble both create-new artifacts from the four
independently reviewed captures. The command rejects mislabeled platforms,
revision drift, reused tensor evidence, and non-SEV-SNP confidential evidence,
and rolls back ordinary partial-output failures:

```bash
version="$(cargo metadata --locked --no-deps --format-version 1 \
  | jq -r '.packages[] | select(.name == "a3s-power") | .version')"
source_commit="$(git rev-parse HEAD)"

for platform_capture in \
  "cpu:/reviewed/cpu.json" \
  "cuda:/reviewed/cuda.json" \
  "metal:/reviewed/metal.json" \
  "confidential-gpu:/reviewed/confidential-gpu.json"
do
  platform="${platform_capture%%:*}"
  capture="${platform_capture#*:}"
  cargo run --locked --release --no-default-features \
    --features embedded-inference \
    --bin a3s-power-tensor-batch-bench -- \
    verify-release-capture \
    --capture "$capture" \
    --platform "$platform" \
    --power-version "$version" \
    --power-commit "$source_commit"
done

mkdir -p "release/v${version}"

cargo run --locked --release --no-default-features \
  --features embedded-inference \
  --bin a3s-power-tensor-batch-bench -- \
  build-release-bundle \
  --cpu-capture /reviewed/cpu.json \
  --cuda-capture /reviewed/cuda.json \
  --metal-capture /reviewed/metal.json \
  --confidential-gpu-capture /reviewed/confidential-gpu.json \
  --power-version "${version}" \
  --power-commit "${source_commit}" \
  --output "release/v${version}/release-evidence.json" \
  --sha256-output "release/v${version}/release-evidence.sha256"

git add -- \
  "release/v${version}/release-evidence.json" \
  "release/v${version}/release-evidence.sha256"
git diff --cached --check
git commit -m "release: add v${version} production evidence"

evidence_commit="$(git rev-parse HEAD)"
test "$(bash tools/verify-release-evidence-commit.sh \
  "${version}" "${evidence_commit}")" = "${source_commit}"
git push origin HEAD:main
git tag -s "v${version}" -m "A3S Power v${version}"
git push origin "v${version}"
```

From the tagged clean checkout, reproduce the release decision with:

```bash
version="$(cargo metadata --locked --no-deps --format-version 1 \
  | jq -r '.packages[] | select(.name == "a3s-power") | .version')"
source_commit="$(bash tools/verify-release-evidence-commit.sh \
  "${version}" HEAD)"

cargo run --locked --release --no-default-features \
  --features embedded-inference \
  --bin a3s-power-tensor-batch-bench -- \
  verify-release-bundle \
  --bundle "release/v${version}/release-evidence.json" \
  --expected-sha256-file "release/v${version}/release-evidence.sha256" \
  --power-version "${version}" \
  --power-commit "${source_commit}"
```

Verification recomputes the bundle and nested digests, requires the exact four
platform classes, checks the external digest pin, compares the exact version and
Git source revision, validates the exact evidence-only child layout, and
requires SEV-SNP for the confidential-GPU capture. The release workflow runs
these checks before any `v1.x` or later tag can publish artifacts.

The checked-in digest is a mutation-detection pin. Release authorship still
requires the repository's signed tag/release trust root and preserved raw
hardware evidence described in the external capture guide.
For non-`0.x` tags, the release workflow also publishes the verified bundle and
pin beside the platform archives instead of leaving production evidence only
inside the source tree.

## Release decision

The source tree alone never claims that a production release passed. The
frozen source parent intentionally has no `release/v1.0.0` directory; only its
direct evidence child may add that directory. A v1.0.0 tag is production
supported exactly when that child is on `main`, its GitHub-verified annotated
tag points directly to it, and the checked-in bundle passes the four-platform
gate against the parent revision. An absent or invalid child means there is no
production v1.0.0 release, regardless of source-build or historical benchmark
results.

The checked-in CPU/CUDA pre-release captures remain useful regression evidence,
but they cannot substitute for exact-parent CPU, CUDA, Metal, and proof-promoted
SEV-SNP confidential-GPU captures authenticated by the tagged bundle.

See [Production Release Evidence Gate](release-evidence-gate.md) for schema and
contract details and [External Metal and Confidential-GPU Capture](external-release-capture.md)
for the hardware procedure.
