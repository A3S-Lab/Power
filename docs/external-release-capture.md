# External Metal and Confidential-GPU Release Capture

This guide closes the operational gap between Power's model-neutral release
contracts and hardware that is not available on the ordinary development host.
It covers two artifacts required by `ReleaseEvidencePolicy::strict_v1`:

- a local Metal complete-contract capture; and
- a confidential-GPU capture promoted from a distinct local CUDA capture only
  after strict CPU TEE and NVIDIA evidence verification.

The workflow is model-, format-, and backend-neutral. Power binds exact weights,
a reviewed graph, typed devices, resource limits, outputs, and policy digests.
It does not branch on an architecture name, tokenizer, quantization scheme,
container format, or generation algorithm. A model integration supplies the
loader, graph, tensors, independent reference output, and opaque implementation
identities.

## Trust boundary

Every capture must come from one immutable Power revision and one common
weights/graph identity. Platform binaries and shape profiles are expected to
differ. A release capture proves internal consistency and detects mutation; it
does not prove who ran the command. The final bundle digest must be authenticated
by a caller-owned signing or attestation trust root.

Use a clean, detached checkout for every hardware run:

```bash
set -euo pipefail
test -z "$(git status --porcelain)"
power_commit="$(git rev-parse HEAD)"
test "${#power_commit}" -eq 40
cargo metadata --locked --no-deps --format-version 1 >/dev/null
```

Record `Cargo.lock`, `rustc -Vv`, `cargo -V`, the operating-system build, driver
and firmware versions, the exact command line, and SHA-256 hashes for every
input and output. Do not rebuild from another commit after a capture has begun.

## Metal capture

Run this path on named Apple hardware with a real Metal device. An emulated,
CPU-fallback, or translated device is not Metal release evidence. Prepare at
least two compatible F32 inputs plus a typed output for the first input from an
independent reviewed implementation.

The predeclared device bounds must be positive and must not be copied from the
observed result:

```bash
set -euo pipefail
set -o noclobber
power_commit="$(git rev-parse HEAD)"

cargo run --locked --release --no-default-features \
  --features embedded-metal \
  --bin a3s-power-tensor-batch-bench -- release-run \
  --output metal.json \
  --weights /verified/model/root \
  --plan /reviewed/graph-plan.json \
  --inputs /private/tensor-items.json \
  --reference-output /private/independent-reference-output.json \
  --family model-owned-family \
  --role model-owned-role \
  --source-format reviewed-format \
  --source-sha256 <reviewed-source-sha256> \
  --opset 1 \
  --profile-implementation-sha256 <metal-profile-implementation-sha256> \
  --profile-shape-class-sha256 <metal-shape-class-sha256> \
  --fallback-implementation-sha256 <metal-fallback-implementation-sha256> \
  --fallback-request-class-sha256 <metal-fallback-request-class-sha256> \
  --tee-policy-sha256 <reviewed-local-metal-policy-sha256> \
  --host-fixed-bytes <positive-predeclared-host-fixed-bound> \
  --host-scratch-bytes <positive-predeclared-host-scratch-bound> \
  --device-fixed-bytes <positive-predeclared-metal-fixed-bound> \
  --device-scratch-bytes <positive-predeclared-metal-scratch-bound> \
  --device metal:0 \
  --power-commit "$power_commit" \
  --filesystem-class apfs \
  --device-class "<exact Apple GPU, memory, and OS build>" \
  --cpu-model "<exact Apple SoC model>" \
  --ram-bytes <physical-memory-bytes> \
  --warmup-rounds 2 \
  --measured-rounds 9
```

`--output` uses create-new semantics. The runner verifies exact scalar/batch
parity, bounded peak host and sampled Metal memory, cancellation cleanup, queue
expiry, replica recovery, and explicit fallback before it commits `metal.json`.
The reference output must not be copied from the Power run.

Save the host record beside the capture:

```bash
sw_vers > macos.txt
uname -a > uname.txt
system_profiler SPHardwareDataType SPDisplaysDataType > apple-hardware.txt
rustc -Vv > rustc.txt
cargo -V > cargo.txt
shasum -a 256 metal.json Cargo.lock /reviewed/graph-plan.json \
  /private/tensor-items.json /private/independent-reference-output.json \
  > metal-inputs.sha256
```

## Confidential-GPU capture

Use supported NVIDIA confidential-computing hardware inside a production CPU
TEE. Power currently supports the production SEV-SNP verification path. The
built-in Intel TDX verifier fails closed because a local TDREPORT is not a DCAP
Quote; do not use TDX for this release class until Quote generation and QVL
verification are implemented.

The confidential workflow uses one fresh 32-byte nonce for all three layers:

1. NVIDIA evidence collection and the remote NRAS verdict;
2. the CPU TEE report returned by `/v1/attestation`; and
3. strict verifier acceptance and release promotion.

### 1. Preserve vendor evidence

Create the nonce before starting the configured-evidence Power process. Preserve
the exact stdout bytes; reformatting JSON changes the digest.

```bash
set -euo pipefail
set -o noclobber
umask 077

nonce="$(openssl rand -hex 32)"
test "${#nonce}" -eq 64
printf '%s\n' "$nonce" > nonce.hex

nvattest collect-evidence \
  --device gpu \
  --nonce "$nonce" \
  --gpu-evidence-source nvml \
  --format json > gpu-evidence.json

nvattest attest \
  --device gpu \
  --verifier remote \
  --nonce "$nonce" \
  --gpu-evidence-source file \
  --gpu-evidence-file gpu-evidence.json \
  --format json > gpu-verdict.json

gpu_evidence_sha256="$(sha256sum gpu-evidence.json | cut -d ' ' -f 1)"
gpu_verdict_sha256="$(sha256sum gpu-verdict.json | cut -d ' ' -f 1)"
```

If the reviewed deployment pins custom NRAS, RIM, OCSP, or relying-party policy
inputs, add the corresponding `nvattest attest` arguments and preserve those
inputs too. Production service URLs must use HTTPS and credentials must not be
embedded in URLs or committed with the evidence.

### 2. Bind the exact server execution policy

Build the independent verifier from the same revision and compute the digest of
the exact `GpuConfig` that the server will use:

```bash
cargo build --locked --release --no-default-features \
  --features server,embedded-inference,hw-verify \
  --bin a3s-power-verify

gpu_execution_sha256="$(target/release/a3s-power-verify \
  --print-gpu-execution-digest \
  --gpu-layers -1 \
  --main-gpu 0 \
  --tensor-split 1.0)"
test "${#gpu_execution_sha256}" -eq 64
```

Include every `cpu_tensors` and `gpu_tensors` override with repeated
`--cpu-tensor` and `--gpu-tensor` flags. The digest command and the server ACL
must describe exactly the same values.

Configure Power with absolute paths to the preserved bytes:

```acl
tee_mode = true
tee_policy_mode = "gpu-confidential"
redact_logs = true

gpu {
  gpu_layers = -1
  main_gpu = 0
  tensor_split = [1.0]
}

gpu_attestation {
  source = "configured"
  provider = "nvidia-nras"
  evidence_path = "/absolute/release/gpu-evidence.json"
  verdict_path = "/absolute/release/gpu-verdict.json"
}

expected_measurement "sev-snp" {
  digest = "<96-character-reviewed-launch-measurement>"
}

model_hash "your-model" {
  digest = "sha256:<same-weights-sha256-used-by-the-release-capture>"
}
```

For a nonce-bound request, the configured provider now parses the saved
`nvattest` evidence and verdict, requires both to carry the exact nonce, records
the evidence entry count, extracts the device claims, and labels the raw formats
as `nvidia-nvattest-evidence-json` and
`nvidia-nvattest-attestation-json`. A stale or unstructured document fails
before the CPU TEE report is issued.

### 3. Capture the report and local CUDA contract

Start the deployment's model/backend build with the ACL above, then save the
exact model-bound report without changing it:

```bash
set -euo pipefail
set -o noclobber
power_url="https://power.example.com"
model_name="your-model"

curl --fail --show-error --silent --get \
  --data-urlencode "nonce=$nonce" \
  --data-urlencode "model=$model_name" \
  "$power_url/v1/attestation" > report.json
```

On the same confidential host, run `release-run` with the same common
weights/graph and independent reference artifacts. The platform policy field is
the exact canonical GPU execution digest that the CPU TEE report also binds:

```bash
set -euo pipefail
set -o noclobber
power_commit="$(git rev-parse HEAD)"

cargo run --locked --release --no-default-features \
  --features embedded-cuda \
  --bin a3s-power-tensor-batch-bench -- release-run \
  --output confidential-source-cuda.json \
  --weights /verified/model/root \
  --plan /reviewed/graph-plan.json \
  --inputs /private/tensor-items.json \
  --reference-output /private/independent-reference-output.json \
  --family model-owned-family \
  --role model-owned-role \
  --source-format reviewed-format \
  --source-sha256 <reviewed-source-sha256> \
  --opset 1 \
  --profile-implementation-sha256 <confidential-profile-implementation-sha256> \
  --profile-shape-class-sha256 <confidential-shape-class-sha256> \
  --fallback-implementation-sha256 <confidential-fallback-implementation-sha256> \
  --fallback-request-class-sha256 <confidential-fallback-request-class-sha256> \
  --tee-policy-sha256 "$gpu_execution_sha256" \
  --host-fixed-bytes <positive-predeclared-host-fixed-bound> \
  --host-scratch-bytes <positive-predeclared-host-scratch-bound> \
  --device-fixed-bytes <positive-predeclared-cuda-fixed-bound> \
  --device-scratch-bytes <positive-predeclared-cuda-scratch-bound> \
  --device cuda:0 \
  --power-commit "$power_commit" \
  --filesystem-class <exact-filesystem-class> \
  --device-class "<exact confidential GPU, driver, firmware, and guest build>" \
  --cpu-model "<exact confidential host CPU model>" \
  --ram-bytes <guest-physical-memory-bytes> \
  --warmup-rounds 2 \
  --measured-rounds 9
```

Write that distinct local source to `confidential-source-cuda.json`. Do not also
use its tensor report as the ordinary CUDA platform capture in the same strict
bundle; strict v1 rejects reused platform evidence.

### 4. Create the accelerator declaration

The integrating model crate creates a declaration from the active Power
residency hierarchy. The server execution digest and declaration digest are
independent identities: the former binds the real `GpuConfig`; the latter also
binds weights, groups, kernels, fallback, device, and optional mesh.

```rust
use a3s_power::api::prompt_policy::canonical_gpu_execution_digest;
use a3s_power::config::GpuConfig;
use a3s_power::inference::{
    AcceleratorFusedBatchSpec, AcceleratorSecurityRequirement,
};

let gpu_config = GpuConfig {
    gpu_layers: -1,
    main_gpu: 0,
    tensor_split: vec![1.0],
    cpu_tensors: Vec::new(),
    gpu_tensors: Vec::new(),
};
let execution_policy_sha256 =
    hex::encode(canonical_gpu_execution_digest(&gpu_config)?);
let spec = AcceleratorFusedBatchSpec::new(
    fused_kernel_sha256,
    exact_fallback_sha256,
    execution_policy_sha256,
    residency_group_ids,
)
.with_security(AcceleratorSecurityRequirement::ConfidentialGpu);
let declaration = hierarchy.declare_accelerator_residency(&spec)?;
declaration.verify()?;

// Serialize declaration to accelerator.json with private, create-new I/O.
// Do not log the document or infer any field from a model name.
```

The declaration's weights, runtime device, and execution-policy digest must
match `confidential-source-cuda.json` and the model-bound CPU TEE claims exactly.

### 5. Verify and promote in one process

Hash the preserved vendor bytes, then pass independently reviewed device and
firmware pins. This example is for one GPU and no NVSwitch; repeat the UEID and
policy flags for the exact expected topology.

```bash
target/release/a3s-power-verify \
  --file report.json \
  --promote-capture confidential-source-cuda.json \
  --accelerator-declaration accelerator.json \
  --promoted-output confidential-gpu.json \
  --expected-measurement <96-character-reviewed-launch-measurement> \
  --nonce "$nonce" \
  --model-hash <same-weights-sha256-used-by-the-release-capture> \
  --gpu-confidential \
  --gpu-evidence-digest "$gpu_evidence_sha256" \
  --gpu-verdict-digest "$gpu_verdict_sha256" \
  --gpu-provider nvidia-nras \
  --gpu-evidence-format nvidia-nvattest-evidence-json \
  --gpu-verdict-format nvidia-nvattest-attestation-json \
  --gpu-evidence-count 1 \
  --gpu-execution-digest "$gpu_execution_sha256" \
  --gpu-count 1 \
  --nvswitch-count 0 \
  --gpu-ueid <reviewed-exact-gpu-ueid> \
  --gpu-oemid <reviewed-gpu-oem-id> \
  --gpu-claims-version <reviewed-claims-version> \
  --gpu-hwmodel "<reviewed-hardware-model>" \
  --gpu-driver-version <reviewed-driver-version> \
  --gpu-firmware-version <reviewed-firmware-version>
```

Promotion uses the fixed `verify_confidential_gpu_attestation` profile and
consumes its opaque exact-report proof immediately. It rejects `--allow-offline`,
live `--url` input, missing evidence pins, a non-local or non-CUDA source,
mismatched weights/device/policy, malformed declarations, and builds without
`embedded-inference` or `hw-verify`. `confidential-gpu.json` is synchronized and
committed with same-directory create-new semantics; an existing path is never
replaced.

## Artifact inventory

Preserve at least these files for review:

| Artifact | Required binding |
| --- | --- |
| Clean source record, `Cargo.lock`, build log, binary hash | Exact Power commit and feature profile |
| Weights, graph plan/declaration, inputs, independent output | Common revision/workload identity |
| Platform ACL and canonical policy input | Exact runtime and security policy |
| `metal.json` | Named Metal complete-contract evidence |
| `confidential-source-cuda.json` | Distinct local CUDA source used only for promotion |
| `accelerator.json` | Weights, real GPU execution policy, residency, kernel/fallback, device/mesh |
| `gpu-evidence.json`, `gpu-verdict.json`, `nonce.hex` | Exact raw NVIDIA freshness and verdict bytes |
| `report.json` | Raw CPU TEE report and canonical model/runtime/GPU claims |
| `confidential-gpu.json` | Proof-backed promoted capture |
| OS, CPU/GPU, driver, firmware, `nvattest`, Rust, and Cargo records | Named execution environment |
| SHA-256 manifest and external signature/attestation | Mutation detection and caller-owned authorship |

Never include bearer tokens, decrypted model values, tensor values, prompts, or
private filesystem paths in the published canonical release bundle. Store
private raw inputs under the release operator's access policy and publish only
the evidence artifacts intended by the schema.

## Acceptance

Before building the four-platform bundle:

1. call `verify()` on each deserialized capture and declaration;
2. confirm CPU, ordinary CUDA, Metal, and confidential GPU use one Power commit,
   weights identity, and graph identity;
3. confirm platform-specific executable, topology, profile, memory, and policy
   bindings remain distinct and honest;
4. confirm the confidential source is not reused as the ordinary CUDA capture;
5. recompute the artifact manifest from read-only copies; and
6. authenticate the final bundle digest through the release trust root.

The workflow and verifier control path are implemented and covered by
deterministic tests. Those tests do not substitute for a hardware capture.
Actual Metal and confidential-GPU captures remain release blockers until they
are produced on supported named hardware from the same immutable revision as
the CPU and ordinary CUDA captures.
