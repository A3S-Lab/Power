# Release capture helpers for the v1 production evidence gate

These scripts encode the fail-closed operator path. They do **not** substitute
for Metal hardware, SEV-SNP/NRAS promotion, or a signed annotated tag.

| Script | Host | Output |
| --- | --- | --- |
| [`capture-windows-cpu-cuda.ps1`](capture-windows-cpu-cuda.ps1) | Windows + CUDA + VS x64 | `cpu.json`, `cuda.json` |
| [`capture-macos-metal.sh`](capture-macos-metal.sh) | Apple Silicon macOS | `metal.json` + host inventory |
| [`capture-confidential-source.ps1`](capture-confidential-source.ps1) | Windows CUDA (dev) | local CUDA source + declaration (**not** confidential-gpu) |
| [`ensure-nvcc-ccbin.ps1`](ensure-nvcc-ccbin.ps1) | Windows VS Build Tools | sets `NVCC_CCBIN` via space-free `C:\vsbt` junction |
| [`assemble-evidence-child.sh`](assemble-evidence-child.sh) | Review host at frozen `S` | `release/v*/release-evidence.{json,sha256}` |

Frozen source parent for the 2026-09-10 Windows CPU/CUDA pair:

`514031dc74edd72da7c3bfee40144a38d2d91434`

Capture and assemble scripts default to that parent and fail closed on any
other `HEAD`. Override only when intentionally recutting evidence:

- macOS / assemble: `A3S_POWER_RELEASE_SOURCE_PARENT=<40-hex>`
- Windows: `-ExpectedSourceParent <40-hex>` or `-AllowAnySourceParent`

### Freeze-parent checkout

Helper scripts and
`docs/benchmarks/release-contract-windows-20260910/local-execution-policy.json`
landed **after** the freeze parent. A clean checkout of `514031dc…` therefore
does not contain them. Do not copy the scripts into that tree.

Use a second worktree. Build and capture only in the freeze-parent worktree;
invoke the scripts by path from `main`:

```bash
git worktree add ../power-freeze 514031dc74edd72da7c3bfee40144a38d2d91434
git show main:docs/benchmarks/release-contract-windows-20260910/local-execution-policy.json \
  > /outside/local-execution-policy.json
cd ../power-freeze
POLICY_PATH=/outside/local-execution-policy.json \
  bash /path/to/main/tools/release-capture/capture-macos-metal.sh /path/to/output
```

Windows is the same shape: `cd` the freeze-parent worktree, pass
`-PolicyPath` to the exported blob, and run the `.ps1` from the `main`
checkout. For `embedded-cuda` on VS Build Tools, `ensure-nvcc-ccbin.ps1` sets
`NVCC_CCBIN` via `C:\vsbt`. Do not nest an outer `VsDevCmd` before `cargo`.

Checked-in partial evidence:
[`docs/benchmarks/release-contract-windows-20260910/`](../../docs/benchmarks/release-contract-windows-20260910/).

Full operator guide:
[`docs/external-release-capture.md`](../../docs/external-release-capture.md).
