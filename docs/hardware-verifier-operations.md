# Hardware Verifier Operations

This guide covers the production AMD SEV-SNP verifier path and the current
fail-closed Intel TDX boundary. Power does not claim production TDX remote
verification until it can generate an Intel DCAP Quote and verify that Quote
with QVL or an equivalently reviewed service.

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

The verifier needs the full raw CPU TEE evidence in the attestation JSON:

- SEV-SNP verification requires the SNP raw report so the verifier can extract
  the TCB version, chip ID, signed report body, and ECDSA P-384 signature. Before
  invoking the signature verifier, Power requires the exposed 64-byte
  `report_data` and 48-byte `measurement` to exactly match their signed raw
  report fields.
- The current TDX provider returns a 1024-byte local TDREPORT. Its REPORTMAC is
  authenticated to the platform quoting component, but it is not a remotely
  verifiable ECDSA Quote. `TdxVerifier` therefore rejects it even when the
  exposed REPORTDATA and MRTD fields match.

Reports fetched from a running Power server include the raw report fields.
Saved report files must preserve those fields exactly.

## Network Access

The `hw-verify` verifier fetches AMD certificate material on demand:

| TEE | Vendor service used by Power | Purpose |
| --- | --- | --- |
| AMD SEV-SNP | `https://kdsintf.amd.com/vcek/v1/...` | Fetch VCEK certificate material for the report TCB and chip ID |
| Intel TDX | Not contacted | Fails closed until DCAP Quote generation and QVL verification are implemented |

Allow outbound HTTPS from the SEV-SNP verification environment to AMD KDS.
Power caches fetched VCEK material in memory for one hour per verifier process,
so long-running verifier processes avoid repeated requests. Short-lived CI jobs
should expect one fetch per cold verifier run. Use
`--hw-cert-cache-ttl-secs <N>` to tune that AMD cache. The default is `3600`;
`0` disables reuse and refetches on every SEV-SNP verification attempt. The
option is accepted for TDX CLI compatibility but cannot enable TDX verification.

## Production Command Shape

For CPU-only strict SEV-SNP verification:

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
- Failed AMD KDS fetches.
- Certificate parse failures.
- Hardware signature verification failures.
- Any mismatch between exposed `report_data` / `measurement` and the signed raw
  SEV-SNP fields.
- Every current Intel TDX verification attempt; a TDREPORT is not a DCAP Quote.
- Simulated or `tee_type=none` reports on a strict path.

Do not paper over those failures with `--allow-offline` in production. A
SEV-SNP deployment that cannot reach AMD KDS should add an explicit offline
certificate-bundle design before claiming production hardware verification.
TDX deployments must add reviewed Quote generation, QVL verification, collateral
freshness handling, and exact Quote REPORTDATA/MRTD extraction before enabling
strict verification.

The Intel distinction follows the official TDX attestation flow: a TDREPORT is
converted into a Quote by a quoting component, and remote verification uses the
DCAP Quote Verification Library. See the
[Intel TDX documentation](https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/documentation.html)
and [Intel TDX enabling guide](https://cc-enabling.trustedservices.intel.com/intel-tdx-enabling-guide/07/trust_domain_at_runtime/).
