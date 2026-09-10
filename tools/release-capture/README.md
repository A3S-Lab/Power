# Release capture helpers for the v1 production evidence gate

These scripts encode the fail-closed operator path. They do **not** substitute
for Metal hardware, SEV-SNP/NRAS promotion, or a signed annotated tag.

| Script | Host | Output |
| --- | --- | --- |
| [`capture-windows-cpu-cuda.ps1`](capture-windows-cpu-cuda.ps1) | Windows + CUDA + VS x64 | `cpu.json`, `cuda.json` |
| [`capture-macos-metal.sh`](capture-macos-metal.sh) | Apple Silicon macOS | `metal.json` + host inventory |
| [`capture-confidential-source.ps1`](capture-confidential-source.ps1) | Windows CUDA (dev) | local CUDA source + declaration (**not** confidential-gpu) |
| [`assemble-evidence-child.sh`](assemble-evidence-child.sh) | Review host at frozen `S` | `release/v*/release-evidence.{json,sha256}` |

Frozen source parent for the 2026-09-10 Windows CPU/CUDA pair:

`514031dc74edd72da7c3bfee40144a38d2d91434`

Checked-in partial evidence:
[`docs/benchmarks/release-contract-windows-20260910/`](../../docs/benchmarks/release-contract-windows-20260910/).

Full operator guide:
[`docs/external-release-capture.md`](../../docs/external-release-capture.md).
