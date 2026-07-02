# Hardware Verifier Operations

This guide covers the production verifier path for AMD SEV-SNP and Intel TDX
CPU TEE reports.

Power's strict verifier path requires two independent checks:

- Hardware signature verification, enabled by building `a3s-power-verify` with
  the `hw-verify` feature.
- An operator-pinned 48-byte launch measurement supplied with
  `--expected-measurement`.

`--allow-offline` skips hardware signatures and measurement pinning. Use it only
for development, fixture tests, or offline inspection.

## Build

Build the verifier with hardware signature support:

```bash
cargo build --release --bin a3s-power-verify --features hw-verify
```

Without `hw-verify`, strict verification fails closed with an error explaining
that hardware signature verification is unavailable. The only bypass is the
explicit `--allow-offline` development flag.

## Evidence Inputs

The verifier needs the full raw CPU TEE report bytes in the attestation JSON:

- SEV-SNP verification requires the SNP raw report so the verifier can extract
  the TCB version, chip ID, signed report body, and ECDSA P-384 signature.
- TDX verification requires the raw TDX report bytes so the verifier can extract
  the platform certificate selector and verify the reported signature/MAC path
  implemented by Power.

Reports fetched from a running Power server include the raw report fields.
Saved report files must preserve those fields exactly.

## Network Access

The `hw-verify` verifier fetches vendor certificate material on demand:

| TEE | Vendor service used by Power | Purpose |
| --- | --- | --- |
| AMD SEV-SNP | `https://kdsintf.amd.com/vcek/v1/...` | Fetch VCEK certificate material for the report TCB and chip ID |
| Intel TDX | `https://api.trustedservices.intel.com/tdx/certification/v4/...` | Fetch Intel PCS certificate material |

Allow outbound HTTPS from the verification environment to the relevant vendor
service. Power caches fetched certificate material in memory for one hour per
verifier process, so long-running verifier processes avoid repeated vendor
requests. Short-lived CI jobs should expect one fetch per cold verifier run.
Use `--hw-cert-cache-ttl-secs <N>` to tune that in-memory cache. The default is
`3600`; `0` disables reuse and refetches vendor certificate material on every
verification attempt.

## Production Command Shape

For CPU-only strict verification:

```bash
a3s-power-verify \
  --url https://power.example.com \
  --model llama3 \
  --nonce <nonce-hex> \
  --model-hash <64-char-model-sha256> \
  --expected-measurement <96-char-launch-measurement-hex>
```

For saved evidence:

```bash
a3s-power-verify \
  --file report.json \
  --nonce <nonce-hex> \
  --model-hash <64-char-model-sha256> \
  --hw-cert-cache-ttl-secs 3600 \
  --expected-measurement <96-char-launch-measurement-hex>
```

For NVIDIA GPU confidential-computing deployments, add the GPU confidential
profile pins described in the README, including `--gpu-confidential`,
`--gpu-verdict-digest`, GPU/NVSwitch topology pins, claims-version pins, and
`--gpu-execution-digest`.

## Failure Modes

Treat these failures as production-blocking:

- Missing `hw-verify` feature in a strict verifier build.
- Missing or malformed `--expected-measurement`.
- Missing raw report bytes in saved evidence.
- Failed AMD KDS or Intel PCS fetches.
- Certificate parse failures.
- Hardware signature verification failures.
- Simulated or `tee_type=none` reports on a strict path.

Do not paper over those failures with `--allow-offline` in production. A
deployment that cannot reach vendor certificate services should add an explicit
offline certificate-bundle design before claiming production hardware
verification.
