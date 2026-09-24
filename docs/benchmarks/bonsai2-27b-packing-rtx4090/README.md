# Bonsai 2 packing compare (PTQ1_0 vs PQ2_0) — RTX 4090

Same-host decode/prefill compare through `a3s-power` + Prism upstream
(sequential single-GPU restart).

## Gate

See [perf-first-principles.md](../../perf-first-principles.md).

## Capture (2026-09-24)

| Artifact | [power-packing-compare-20260924-003352.json](power-packing-compare-20260924-003352.json) |
| --- | --- |
| Mode | sequential (`run-prism-packing-compare-sequential.ps1`) |
| Workload | 3×128 tokens, code prompt, `prism_profile=baseline` |
| Decode winner | **PTQ1_0** — 81.42 tok/s mean vs PQ2 77.60 |
| Prefill winner | **PQ2_0** — 153.5 prompt tok/s mean vs PTQ1 123.5 |

Steady-state note: sample 1 prompt rates include cold-cache spikes. Samples 2–3
prefill still favor PQ2 (~37.5 vs ~30.5 prompt tok/s). Decode ranking is stable
across all three samples.

**Ada advisory (this host):** prefer `PTQ1_0` for decode-bound agentic chat;
prefer `PQ2_0` when prefill/TTFT dominates.

## Streaming TTFT cross-check (same prompt, 5×64, Power SSE)

| Packing | Artifact | Stream TTFT median | Decode mean (3×128) |
| --- | --- | --- | --- |
| PTQ1_0 | [../bonsai2-27b-ptq1-rtx4090/power-ttft-bench-20260924-043908.json](../bonsai2-27b-ptq1-rtx4090/power-ttft-bench-20260924-043908.json) | 2318 ms | 79.63 tok/s |
| PQ2_0 | [power-ttft-bench-20260924-044132.json](power-ttft-bench-20260924-044132.json) | **2107 ms** (~9% lower) | 75.71 tok/s ([power-baseline-bench-20260924-044100.json](power-baseline-bench-20260924-044100.json)) |

Confirms packing advisory under client-observed TTFT: PQ2 wins first-token
latency; PTQ1 wins steady decode.

## KV4 profile tax (PQ2_0, short chat)

| Profile | Artifact | Decode mean (3×128) |
| --- | --- | --- |
| baseline | [power-baseline-bench-20260924-044100.json](power-baseline-bench-20260924-044100.json) | 75.71 tok/s |
| kv4 | [power-kv4-baseline-bench-20260924-044305.json](power-kv4-baseline-bench-20260924-044305.json) | 75.90 tok/s |

On this short workload KV4 is **noise-level** vs baseline decode. Claim KV4 as
VRAM relief for long context, not as a tok/s win.

## Reproduce

```powershell
.\tools\run-prism-packing-compare-sequential.ps1 `
  -Ptq1ModelPath "D:\Bonsai-demo\models\bonsai2-gguf\27B\Ternary-Bonsai-2-27B-PTQ1_0.gguf" `
  -Pq2ModelPath "D:\Bonsai-demo\models\bonsai2-gguf\27B\Ternary-Bonsai-2-27B-PQ2_0.gguf" `
  -BinDir "D:\Bonsai-demo\bin\cuda" `
  -OutDir docs\benchmarks\bonsai2-27b-packing-rtx4090
```

Dual-upstream form: `run-prism-packing-compare.ps1` (ports 8080/8081).

## Related

- PTQ1-only baseline tree: [../bonsai2-27b-ptq1-rtx4090/](../bonsai2-27b-ptq1-rtx4090/)
