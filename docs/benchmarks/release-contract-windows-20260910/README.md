# Windows Exact-Parent CPU and CUDA Captures — 2026-09-10

These captures bind the frozen v1.0.0 **source parent**

`514031dc74edd72da7c3bfee40144a38d2d91434`

(`Cargo.toml` `1.0.0`, changelog `[1.0.0] - 2026-09-10`, empty
`[Unreleased]`). They prove the complete model-neutral runtime contract on
this named CPU and CUDA host.

They are **partial** production inputs only. They do **not** complete
`ReleaseEvidencePolicy::strict_v1`. Same-parent Metal and proof-promoted
SEV-SNP confidential-GPU captures are still required before an evidence child
or annotated `v1.0.0` tag may exist.

## Immutable inputs

| Item | Value |
| --- | --- |
| Power revision | `514031dc74edd72da7c3bfee40144a38d2d91434` |
| Power version | `1.0.0` |
| Rust toolchain | `rustc 1.97.1 (8bab26f4f 2026-07-14)`, MSVC target |
| Host | Windows 11, x86_64 |
| CPU | Intel Xeon w5-2445, 20 logical CPUs |
| RAM | 137,071,693,824 bytes |
| CUDA device | NVIDIA GeForce RTX 4090 24 GiB, compute capability 8.9, ordinal 0 |
| NVIDIA driver | 610.74 |
| Local execution policy | [`local-execution-policy.json`](local-execution-policy.json), SHA-256 `e8706c8becf1dad80a5ff83004eb1028e1ef9c67ff3fa7fc978416dc8be4d3bd` |
| Fixture | 8 inputs, shape `[1, 4096]`, 2 warmups, 9 measured rounds |
| Host reservation | 64 MiB fixed + 64 MiB scratch |
| CUDA reservation | 64 MiB fixed + 64 MiB scratch |

The policy declares local, unattested execution and makes no confidential-GPU
claim.

## Artifact identities

| Capture | Capture SHA-256 | JSON file SHA-256 |
| --- | --- | --- |
| CPU | `ec218d03a86be3d6eda964f87ad8b31e31520591c600ce2b64944209800e23cc` | `b5abdb45a9dd7890774e408aa303f5e414222a0ee5782056afd3b6372b0557b0` |
| CUDA | `408c31d2d41a1fc2a1f399560dad0adeafaa2ab88ee4036e25c71f6d8326f71d` | `1e225c72e632eb6623d0ca66e1ad9cf308ba78fe6f832dcd899c706090348e5c` |

Common workload identities match the generic Add fixture:

| Identity | SHA-256 |
| --- | --- |
| Fixture weights | `0f3de53015794403689b151f172b05e0bd115ed1b61290318ec4eab882dca443` |
| Reviewed graph source | `e894ac8daa23b2caaf3031af5ef287dc0a3dc15dbe580338ca552817ed5f92c3` |
| Reviewed graph declaration | `0400ec37caa58a74731fb78c9cceb0e318bd44be71e925f094bce50cba82fb3c` |
| CPU shape-profile declaration | `54865497f48aac854c16f9ba6c442a463f39d11d64fc6c014f0f1a474ece2792` |
| CUDA shape-profile declaration | `4eb8b51d60e876c7679c7f8f30bad74bd99087612413108de89555ff17e242fd` |

## Verify

From a clean checkout of this repository (any descendant of the source parent):

```powershell
$revision = "514031dc74edd72da7c3bfee40144a38d2d91434"
cargo run --locked --release --no-default-features `
  --features embedded-inference `
  --bin a3s-power-tensor-batch-bench -- verify-release-capture `
  --capture docs/benchmarks/release-contract-windows-20260910/cpu.json `
  --platform cpu --power-version 1.0.0 --power-commit $revision

# CUDA verify requires an x64 VS developer shell (cl.exe on PATH):
cargo run --locked --release --no-default-features `
  --features embedded-cuda `
  --bin a3s-power-tensor-batch-bench -- verify-release-capture `
  --capture docs/benchmarks/release-contract-windows-20260910/cuda.json `
  --platform cuda --power-version 1.0.0 --power-commit $revision
```

## Reproduce

Use [`tools/release-capture/capture-windows-cpu-cuda.ps1`](../../../tools/release-capture/capture-windows-cpu-cuda.ps1)
against a clean detached checkout of `$revision`. Keep outputs outside the
worktree so the source tree stays clean.

External Metal and confidential-GPU hosts: see
[`docs/external-release-capture.md`](../../external-release-capture.md) and
[`tools/release-capture/`](../../../tools/release-capture/).
