# DSpark 100-task quality evidence

[`evidence.json`](evidence.json) is the path-free, machine-readable package for
the context-1024 Q6_K target-only versus external-DSpark K10/S6 capture. It
contains 600 successful API requests: 100 fixed MMLU/GSM8K/C-Eval tasks, two
modes, and three alternating repetitions.

| Mode | Lenient | Strict | Mean request-wide throughput |
| --- | ---: | ---: | ---: |
| Q6_K target-only | 67/100 | 58/100 | 22.618 token/s |
| Q6_K + DSpark Q4 K10/S6 | 73/100 | 59/100 | 32.678 token/s |

The capture is deterministic within each mode and records a 1.445x workload
speedup. Complete target/DSpark outputs match on only 54/100 tasks, so the
package is classified as diagnostic output divergence and is not eligible as
a lossless production default.

Verify every pinned identity and derived metric without a model or GPU:

```powershell
py -3.13 .\tools\qwen38_quality_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\quality\evidence.json `
  --json
```

`--require-production-default` is expected to reject this capture. See the
[parent report](../README.md#representative-100-task-diagnostic) for analysis,
raw-capture reproduction, and the distinction from the 169.324 token/s peak
prompt.
