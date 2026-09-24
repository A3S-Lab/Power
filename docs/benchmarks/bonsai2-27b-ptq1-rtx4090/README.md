# Bonsai 2 27B — Power Prism acceleration test evidence (RTX 4090)

Path: `a3s-power` (`prism_profile=…`) → Prism / patched Prism+DFlash2
`llama-server` → Ternary-Bonsai-2 27B GGUF.

## Peak result (2026-09-24)

| Profile | Artifact | Mean tok/s | Accept | Notes |
| --- | --- | --- | --- | --- |
| **dflash SM89 n=3 + reasoning off** | [power-spec-dflash-sm89-n3-nothink-20260924-092416.json](power-spec-dflash-sm89-n3-nothink-20260924-092416.json) | **137.72** (max 151.5) | **98.6%** | **host peak** |
| same-binary PQ2 baseline | [power-spec-pq2-sm89-baseline-nothink-20260924-092512.json](power-spec-pq2-sm89-baseline-nothink-20260924-092512.json) | **74.21** | — | **1.86×** dflash vs this |
| stock Prism MTP n=1 + reasoning off | [power-spec-mtp-n1-nothink-20260924-065837.json](power-spec-mtp-n1-nothink-20260924-065837.json) | **89.89** | **97.8%** | no patched binary |
| stock Prism PTQ1 thinking-on | [power-baseline-bench-20260924-043700.json](power-baseline-bench-20260924-043700.json) | **79.63** | — | prior ~80 serve |

**DFlash stack:** ProCreations `prism-dflash2-source.tar.gz` built for
`CMAKE_CUDA_ARCHITECTURES=89` →
`…/bonsai2-dflash2-runtime/runtime/source/llama/build-sm89/bin`,
target `Ternary-Bonsai-2-27B-PQ2_0.gguf`, draft
`Bonsai-2-27B-DFlash2-Q8_0.gguf`, `--spec-type draft-dflash --spec-draft-n-max 3
--reasoning off`, Power `prism_profile=dflash`.

## Reproducible stock-Prism baseline (3×128 tokens)

| Metric | Value |
| --- | --- |
| Artifact (2026-09-24 stream-path refresh) | [power-baseline-bench-20260924-043700.json](power-baseline-bench-20260924-043700.json) |
| `predicted_per_second` mean | **79.63** tok/s |
| min / max | 77.45 / 80.92 |
| Prior capture (2026-09-23) | [power-baseline-bench-20260923-235831.json](power-baseline-bench-20260923-235831.json) ≈78.43 tok/s |
| `prism_upstream.source` | `upstream-reported` |
| `draft_n` | null (baseline; DSpark not engaged) |

```powershell
.\tools\run-prism-baseline-bench.ps1 -Samples 3 -OutDir docs\benchmarks\bonsai2-27b-ptq1-rtx4090
```

## Streaming TTFT (client-observed)

Tool-free Power chat forwards Prism SSE. TTFT counts first non-empty
`content` / `reasoning_content` / `thinking` delta.

| Metric | Value |
| --- | --- |
| Artifact | [power-ttft-bench-20260924-043908.json](power-ttft-bench-20260924-043908.json) |
| Samples / max_tokens | 5 × 64 |
| Stream TTFT median | **2318 ms** |
| Stream wall median | 3117 ms |
| Buffered wall median | 3147 ms |
| Client first-token vs buffered completion | ~26% earlier (TTFT / buffered wall) |

## Ceiling vs serving gap (reasoning tax)

Raw `llama-bench` PTQ1 (ngl 99, FA on, tg128): **89.09** tok/s. Default chat
with thinking sat near **~80**. `--reasoning off` closes most of that gap
before speculation.

| Profile | Artifact | Mean tok/s | Accept | Notes |
| --- | --- | --- | --- | --- |
| llama-bench PTQ1 | `.tmp-perf-goal/llama-bench-ptq1.txt` | **89.09** | — | raw tg128 ceiling |
| baseline (thinking on) | [power-baseline-bench-20260924-043700.json](power-baseline-bench-20260924-043700.json) | **79.63** | — | prior default serve |
| baseline `--reasoning off` | [power-spec-baseline-nothink-20260924-065654.json](power-spec-baseline-nothink-20260924-065654.json) | **85.31** | — | closes ~70% of ceiling gap |
| mtp n=1 + reasoning off | [power-spec-mtp-n1-nothink-20260924-065837.json](power-spec-mtp-n1-nothink-20260924-065837.json) | **89.89** | **97.8%** | stock Prism PTQ1 MTP graft |
| **dflash SM89 n=3 + reasoning off** | [power-spec-dflash-sm89-n3-nothink-20260924-092416.json](power-spec-dflash-sm89-n3-nothink-20260924-092416.json) | **137.72** | **98.6%** | **peak** |
| pq2 SM89 baseline (no draft) | [power-spec-pq2-sm89-baseline-nothink-20260924-092512.json](power-spec-pq2-sm89-baseline-nothink-20260924-092512.json) | **74.21** | — | paired with dflash |

## Speculation detail (Bonsai-2; official DSpark unavailable)

Official Bonsai-2 `*dspark-dflash*` does **not** exist (HF tree +
`Bonsai-demo/setup.ps1`). Power keeps `prism_profile=dspark` fail-closed.
Do **not** attach a Bonsai-1 DSpark sidecar (HF reports large regressions).

| Profile | Artifact | Mean tok/s | Accept | Notes |
| --- | --- | --- | --- | --- |
| baseline PTQ1 (thinking on) | [power-baseline-bench-20260924-043700.json](power-baseline-bench-20260924-043700.json) | **79.63** | — | no draft |
| baseline `--reasoning off` | [power-spec-baseline-nothink-20260924-065654.json](power-spec-baseline-nothink-20260924-065654.json) | **85.31** | — | no draft |
| mtp `n-max=1` (thinking on) | [power-spec-mtp-20260924-065307.json](power-spec-mtp-20260924-065307.json) | **83.32** | **72.6%** | stock Prism |
| mtp `n-max=1` + reasoning off | [power-spec-mtp-n1-nothink-20260924-065837.json](power-spec-mtp-n1-nothink-20260924-065837.json) | **89.89** | **97.8%** | stock Prism best |
| mtp `n-max=2` | [power-spec-mtp-nmax2-20260924-065348.json](power-spec-mtp-nmax2-20260924-065348.json) | 79.69 | 60% | worse than n-max=1 |
| ext Qwen MTP `-md` n=2 | [power-spec-ext-mtp-n2-nothink-20260924-090105.json](power-spec-ext-mtp-n2-nothink-20260924-090105.json) | **19.10** | 97% | **reject** |
| stock Prism + DFlash2 GGUF | [dflash-stock-prism-fail-20260924.txt](dflash-stock-prism-fail-20260924.txt) | — | — | tensor 81≠58 |
| **dflash patched SM89** | [power-spec-dflash-sm89-n3-nothink-20260924-092416.json](power-spec-dflash-sm89-n3-nothink-20260924-092416.json) | **137.72** | **98.6%** | **1.86×** vs PQ2 same binary |

### Build patched Prism+DFlash2 for RTX 4090 (SM89)

```powershell
# Source: ProCreations/Ternary-Bonsai-2-27B-DFlash2 runtime/prism-dflash2-source.tar.gz
# (source_is_already_patched=true — do not re-apply bonsai-dflash2.patch)
$llama = "D:\Bonsai-demo\models\bonsai2-dflash2-runtime\runtime\source\llama"
$cmake = "C:\vsbt\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
# Via VsDevCmd + Ninja:
# cmake -S $llama -B $llama\build-sm89 -G Ninja -DGGML_CUDA=ON `
#   -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release ...
# cmake --build $llama\build-sm89 -j 8 --target llama-server

.\tools\start-prism-upstream.ps1 -Profile dflash -Alias bonsai2-pq2-dflash `
  -Model "…\Ternary-Bonsai-2-27B-PQ2_0.gguf" `
  -Drafter "…\Bonsai-2-27B-DFlash2-Q8_0.gguf" `
  -BinDir "…\build-sm89\bin" -DraftNMax 3 -Reasoning off -ReasoningBudget 0
.\tools\run-prism-spec-compare.ps1 -ProfileLabel dflash-sm89-n3-nothink `
  -Model bonsai2-pq2-dflash -OutDir docs\benchmarks\bonsai2-27b-ptq1-rtx4090
```

**Verdict:** deepest measured acceleration on this RTX 4090 host is
**patched Prism+DFlash2** → **~138 tok/s** mean through Power (~1.86× vs same
binary PQ2 baseline; ~1.73× vs prior ~80 thinking-on serve). Stock Prism
fallback remains MTP n-max=1 + reasoning off (~90 tok/s). Official DSpark
still absent for Bonsai-2 — fail-closed.

## Automated gates

```text
cargo test --no-default-features --features server --lib prism
# 19 passed — includes upstream SSE, health TTL, mtp/dspark/dflash profile rules,
# Power spec_mode fail-closed, drafter fail-closed
```

## DSpark compare (G3)

**Skipped / fail-closed:** Bonsai 2 has no pinned official `*dspark-dflash*`
drafter. Prefer `prism_profile=dflash` (patched bin) or `mtp` (stock Prism).
Do not attach a Bonsai-1 DSpark sidecar to Bonsai-2.
