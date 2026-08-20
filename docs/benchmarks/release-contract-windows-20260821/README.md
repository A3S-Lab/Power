# Windows CPU and CUDA Complete Runtime-Contract Captures — 2026-08-21

These captures are valid partial inputs to the model-neutral P5 release gate.
They were produced by the same immutable Power revision and prove the complete
runtime contract on this named CPU and CUDA host. They do **not** complete the
strict v1 policy: independent Metal and confidential-GPU captures are still
required.

The workload is a reviewed broadcast Add graph over opaque F32 tensors. It has
no tokenizer, decoder, prompt, model family, or architecture branch. The fixture
calibrates the shared runtime contract; it is not a token-throughput benchmark
and does not claim performance for Qwen or any other model.

## Immutable inputs

| Item | Value |
| --- | --- |
| Power revision | `6b7d6e5265b34c3e9e812c830ce22cc4a35940e5` |
| Power version | `0.9.0` |
| Rust toolchain | `rustc 1.97.1 (8bab26f4f 2026-07-14)`, MSVC target |
| Host | Windows 11 23H2, x86_64 |
| CPU | Intel Xeon w5-2445, 20 logical CPUs |
| RAM | 137,071,693,824 bytes |
| CUDA device | NVIDIA GeForce RTX 4090 24 GiB, compute capability 8.9, ordinal 0 |
| NVIDIA driver / CUDA | 610.74 / 12.6.68 |
| Local execution policy | [`local-execution-policy.json`](local-execution-policy.json), SHA-256 `e8706c8becf1dad80a5ff83004eb1028e1ef9c67ff3fa7fc978416dc8be4d3bd` |
| Fixture | 8 inputs, shape `[1, 4096]`, 2 warmups, 9 measured rounds |
| Host reservation | 64 MiB fixed + 64 MiB scratch on both platforms |
| CUDA reservation | 64 MiB fixed + 64 MiB scratch |

The policy explicitly declares local, unattested execution and makes no TEE or
confidential-GPU claim. Its directory is marked `-text -diff` in
`.gitattributes`, so the policy and evidence SHA-256 values survive clones
without newline conversion.

## Observed results

Times are medians across the nine raw alternating samples retained in each
capture. Lower is better.

| Platform | Individual median | Leading-batch median | Host additional peak | Device additional peak | Final additional memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 67.2 µs | 254.5 µs | 34,661 bytes | Not applicable | 0 bytes host |
| CUDA | 400.3 µs | 195.3 µs | 17,053 bytes | 33,554,432 bytes | 0 bytes host; 33,554,432 bytes device |

The CPU leading batch is 3.79× slower for this tiny graph; that negative result
is retained. CUDA leading batch reduces median elapsed time by 51.2% versus
eight individual executions. The retained 32 MiB CUDA allocation is within the
predeclared 64 MiB fixed reservation and reflects the accelerator runtime pool,
not leaked resident graph handles.

Both captures additionally prove:

- exact scalar/leading-batch output parity;
- one active-work cancellation with zero admission and resident handles after
  cleanup;
- one real monotonic queue deadline expiration;
- one replica retirement followed by one lazy reconstruction; and
- an explicit dynamic fallback whose typed output digest exactly equals the
  independently calculated fixture reference.

## Artifact identities

| Capture | Runtime executable SHA-256 | Tensor report SHA-256 | Capture SHA-256 | JSON file SHA-256 |
| --- | --- | --- | --- | --- |
| CPU | `04c90d829beff7b6752d1bc7471a2c5be369b0f5699d6f6ad471fc88b573dfa5` | `c25a45cd4d3f03c82b02efae66a286026c5828a35bd415b262875e461d005431` | `579ac88294a22c44f9dd9467a78a13c8d2567f7feb39e5fca7d7a98f147b8bdc` | `58ce9799449f36f1b60e9db868ebc109eab490304bb4fcfc817dde7b0f365de6` |
| CUDA | `8975345018d141bf669c9ebdbdcf8b66c7a903100ba5633f8e3dcf46dafd1750` | `3c8b63aa1db24622f87b60efaee353d3eb7698fc26c3f9c152182b6f954f6d18` | `7ed8b965039890138f318879ccc22067f3f74144539f5ce71c1fcae2aa1983dc` | `d89f7e326118d4139fbecda7b5a6556366bfafef044d41becd8a20085797b73d` |

Common workload identities:

| Identity | SHA-256 |
| --- | --- |
| Fixture weights | `0f3de53015794403689b151f172b05e0bd115ed1b61290318ec4eab882dca443` |
| Reviewed graph source | `e894ac8daa23b2caaf3031af5ef287dc0a3dc15dbe580338ca552817ed5f92c3` |
| Reviewed graph declaration | `0400ec37caa58a74731fb78c9cceb0e318bd44be71e925f094bce50cba82fb3c` |

The CPU shape-profile declaration is
`54865497f48aac854c16f9ba6c442a463f39d11d64fc6c014f0f1a474ece2792`;
the CUDA declaration is
`4eb8b51d60e876c7679c7f8f30bad74bd99087612413108de89555ff17e242fd`.
They differ intentionally because typed device topology and reservations are
part of each platform binding.

## Reproduce

Use a detached worktree at the exact revision. Keep capture output outside that
worktree so the source tree remains clean throughout execution.

```powershell
$revision = "6b7d6e5265b34c3e9e812c830ce22cc4a35940e5"
$captureRoot = "D:\code\a3s-power-release-capture-6b7d6e5"
$outputRoot = "D:\captures\a3s-power-6b7d6e5"
git worktree add --detach $captureRoot $revision
New-Item -ItemType Directory -Path $outputRoot
Set-Location $captureRoot
if (git status --porcelain) { throw "capture worktree is not clean" }
$policyHash = (Get-FileHash `
  .\docs\benchmarks\release-contract-windows-20260821\local-execution-policy.json `
  -Algorithm SHA256).Hash.ToLowerInvariant()
```

CPU:

```powershell
cargo run --locked --release --no-default-features `
  --features embedded-inference `
  --bin a3s-power-tensor-batch-bench -- release-fixture `
  --output "$outputRoot\cpu.json" `
  --device cpu `
  --power-commit $revision `
  --filesystem-class ntfs `
  --device-class "Intel Xeon w5-2445 CPU" `
  --cpu-model "Intel(R) Xeon(R) w5-2445" `
  --ram-bytes 137071693824 `
  --tee-policy-sha256 $policyHash `
  --host-fixed-bytes 67108864 `
  --host-scratch-bytes 67108864 `
  --device-fixed-bytes 0 `
  --device-scratch-bytes 0 `
  --items 8 --width 4096 `
  --warmup-rounds 2 --measured-rounds 9
```

CUDA, from an x64 Visual Studio developer shell:

```powershell
cargo run --locked --release --no-default-features `
  --features embedded-cuda `
  --bin a3s-power-tensor-batch-bench -- release-fixture `
  --output "$outputRoot\cuda.json" `
  --device cuda:0 `
  --power-commit $revision `
  --filesystem-class ntfs `
  --device-class "NVIDIA GeForce RTX 4090 24 GiB; driver 610.74" `
  --cpu-model "Intel(R) Xeon(R) w5-2445" `
  --ram-bytes 137071693824 `
  --tee-policy-sha256 $policyHash `
  --host-fixed-bytes 67108864 `
  --host-scratch-bytes 67108864 `
  --device-fixed-bytes 67108864 `
  --device-scratch-bytes 67108864 `
  --items 8 --width 4096 `
  --warmup-rounds 2 --measured-rounds 9
```

Replay semantic verification and compare file hashes:

```powershell
cargo test --locked --no-default-features `
  --features embedded-inference `
  published_complete_cpu_cuda_contracts_replay_as_one_partial_policy --lib
Get-FileHash "$outputRoot\cpu.json" -Algorithm SHA256
Get-FileHash "$outputRoot\cuda.json" -Algorithm SHA256
if (git status --porcelain) { throw "capture worktree changed" }
```

`ReleaseCapture::verify()` recomputes nested and canonical digests. The replay
test also constructs and verifies one two-platform `ReleaseEvidenceBundle`,
proving that both captures share the same revision, weights, and reviewed graph
while retaining their distinct platform bindings. Authenticity still requires
the final bundle digest to be pinned by a signed release or equivalent
caller-owned trust root.
