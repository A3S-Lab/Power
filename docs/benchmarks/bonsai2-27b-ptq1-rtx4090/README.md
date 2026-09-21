# Bonsai 2 27B PTQ1_0 — Power Prism acceleration test evidence (RTX 4090)

Path: `a3s-power` (`prism_profile=baseline`) → Prism `llama-server` →
`Ternary-Bonsai-2-27B-PTQ1_0.gguf`.

## Reproducible baseline bench (3×128 tokens)

| Metric | Value |
| --- | --- |
| Artifact | [power-baseline-bench-20260921-180201.json](power-baseline-bench-20260921-180201.json) |
| `predicted_per_second` mean | **80.97** tok/s |
| min / max | 78.13 / 82.73 |
| `prism_upstream.source` | `upstream-reported` |
| `draft_n` | null (baseline; DSpark not engaged) |

Reproduce:

```powershell
.\tools\run-prism-baseline-bench.ps1 -Samples 3 -OutDir docs\benchmarks\bonsai2-27b-ptq1-rtx4090
```

## Automated gates

```text
cargo test --no-default-features --features server --lib prism
# 14 passed — includes Power spec_mode fail-closed + dspark drafter fail-closed
```

## DSpark compare (G3)

**Skipped / fail-closed:** Bonsai 2 has no pinned official `*dspark-dflash*`
drafter in this environment. `prism_profile=dspark` without `prism_drafter`
rejects at admit time (unit-tested). Do not treat Qwen speculative benches as
Bonsai evidence.
