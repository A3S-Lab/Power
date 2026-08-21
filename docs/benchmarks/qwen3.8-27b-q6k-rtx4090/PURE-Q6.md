# Untouched Q6_K performance boundary

Yes: the untouched 22.88 GB Q6_K artifact can exceed 175 token/s on the
acceptance host. The measured result is a peak steady-decode boundary, not a
claim that every prompt or service workload sustains that rate.

## Accepted result

Both rows below use the exact same GGUF, server binary, prompt, greedy sampling
policy, batch size, and host controls. Each capture has one warm-up followed by
nine measured 1,024-token requests through Power's streaming API.

| Pure Q6_K mode | Median decode | Minimum decode | Median end to end | Samples at least 175 | Output SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| [Full-vocabulary MTP, K7/S7](pure-q6-full-vocabulary-1024-9x.json) | 147.0207 token/s | 146.0917 token/s | 140.2573 token/s | 0 / 9 | `a54538ea...90523` |
| [Prefix-FR8192 MTP, K7/S6](pure-q6-fr8192-1024-9x.json) | **176.6109 token/s** | 173.2630 token/s | **167.3519 token/s** | **7 / 9** | `a54538ea...90523` |

Prefix-FR8192 improved median steady decode by **20.13%** and median
end-to-end throughput by **19.32%**. The model bytes were not converted or
requantized. Every measured output retained the same deterministic digest.

| Identity or control | Fixed value |
| --- | --- |
| Power source revision | `eb6aeda59561eff3e4e7592704cab6fc863b72c7`, clean worktree |
| GGUF bytes | `22,884,408,288` |
| GGUF SHA-256 | `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` |
| Server SHA-256 | `c6ba312db786b45d81e8feaa286df7793ad2f072a81e4d0ab37ad39756ec95fa` |
| Benchmark client SHA-256 | `d1b5760849f31d5d9b76e64548dd85110db3b961cd032eb818917d88e4d452da` |
| GPU | RTX 4090, driver 610.74, requested 2,745 MHz clock lock |
| CPU | Xeon w5-2445, ten workers, affinity `0x55555` |
| Runtime | Full CUDA offload, Flash Attention, one parallel slot, batch 14 |
| Sampling | Greedy, seed 42, `temperature=0`, `top_p=1` |

The complete machine receipts are
[full vocabulary](pure-q6-full-vocabulary-1024-9x.environment.json) and
[prefix FR](pure-q6-fr8192-1024-9x.environment.json). The reviewed runtime
profiles are [K7/S7 full vocabulary](pure-q6-mtp7-snap7-host-staged.acl) and
[K7/S6 prefix FR](pure-q6-mtp7-snap6-fr8192-host-staged.acl).

## Why this shape is fast

The optimization follows the data path rather than the model name:

1. A 27B Q6 target is primarily a weight-traffic problem during
   autoregressive decode. Reducing sampler overhead alone cannot create a
   fivefold gain.
2. Native MTP proposes seven tokens and lets the target verify a block, so one
   target pass can emit several exact target tokens when proposal acceptance
   is high.
3. The current full draft head projects 248,320 token rows. On the fixed peak
   prompt, limiting that draft-only projection to the first 8,192 token IDs
   retained 93.617% proposal acceptance in the 256-token diagnostic while
   cutting draft projection time.
4. Six resident recurrent snapshots avoid one snapshot allocation. The peak
   prompt's longest rejected suffix was five, so it did not replay. This is an
   observed property of that prompt, not a general guarantee.
5. Batched target and draft greedy selection, Flash Attention, full offload,
   host-staged recurrent state, batch 14, physical-core affinity, and an
   idle-enough WDDM GPU remove smaller synchronization and launch costs around
   the dominant graph work.

The target still verifies every proposal. Prefix FR can change proposal
coverage and acceptance, but it cannot commit an unverified draft token.

The checked-in full head has no corpus-frequency `d2t` map, so
`spec_mtp_fr_vocab_size = 8192` means a **target-token-ID prefix**, not a
frequency-ranked vocabulary. A separately constructed compact head may use a
real frequency order, but it is a different artifact and must be measured
under its own identity.

## Why 176.61 token/s is not the workload average

A fixed one-pass, 12-task MMLU/GSM8K/C-Eval calibration used the same untouched
Q6_K artifact, batch 14, K7/S6, and 128-token cap:

| Mode | Request-wide throughput | Proposal acceptance | Lenient | Strict |
| --- | ---: | ---: | ---: | ---: |
| Autoregressive | 29.7127 token/s | - | 4 / 12 | 3 / 12 |
| Full-vocabulary MTP | **47.0324 token/s** | **52.3041%** | **5 / 12** | 3 / 12 |
| Prefix-FR8192 MTP | 37.2900 token/s | 24.8249% | 4 / 12 | 3 / 12 |

Full-vocabulary MTP was 58.29% faster than autoregressive mode. Prefix-FR8192
was 25.50% faster than autoregressive mode but 20.71% slower than
full-vocabulary MTP because its proposal coverage fell on the mixed,
multilingual tasks.

Eleven of twelve tasks in every mode hit the 128-token cap. The capture is a
throughput and acceptance calibration, not a defensible intelligence score.
The repeated 100-task matrix remains the primary quality evidence, and it has
not yet been rerun for this exact pure-Q6_K prefix-FR profile. See the
[raw calibration](quality/pure-q6-fr8192-calibration-rtx4090-1x.json) and its
[environment receipt](quality/pure-q6-fr8192-calibration-rtx4090-1x.environment.json).

The operational conclusion is:

- use full-vocabulary MTP as the balanced pure-Q6_K profile;
- use K7/S6 prefix-FR8192 only for measured, high-coverage workloads where
  peak steady decode matters;
- select between them from workload evidence, not from the fixed-prompt peak.

## Dynamic activation quantization result

The pinned llama.cpp CUDA backend already dynamically quantizes activations to
Q8_1 for quantized matrix-matrix kernels. The relevant question was therefore
whether the eight-row Q6 target-verification shape should be forced from its
default matrix-vector kernel into that Q8_1 matrix-matrix path.

The answer on Ada SM89 was no:

| Q6 verification routing, 256 tokens, 5x | Median decode | Change |
| --- | ---: | ---: |
| Default MMVQ | 143.6024 token/s | control |
| Force Q8_1 MMQ for 5-8 rows | 132.2581 token/s | -7.90% |
| Force Q8_1 MMQ for exactly 8 rows | 116.1835 token/s | -19.09% |

All three variants emitted
`584e2b93ba21d7c727456567762c6bbacc150d43156c73ed91c1c0cbb13be6eb`.
The dynamic-quantization experiments changed only the fetched third-party
llama.cpp checkout. Those experimental changes were reverted and are not
shipped because quantization and matrix-kernel setup cost exceeded the benefit
at this small verification row count.

Dynamic quantization remains a valid optimization dimension for larger
matrix-matrix shapes. It is not a universal accelerator, and routing must be
selected by architecture, quantization type, and row shape.

## Reproduce and verify

Use the [step-by-step reproduction guide](REPRODUCE.md#current-untouched-q6-k-target)
to rebuild the SM89 profile, verify the model identity, run both nine-sample
captures, and replay the 12-task calibration.

The model-free verifier checks all raw evidence, environment receipts, ACL
hashes, medians, minima, output identities, and calibration statistics:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\verify-qwen38-q6k-evidence.ps1 -Json
```

A passing result reports `verified_file_hashes: 14`, the 176.6109 token/s
pure-Q6_K peak, the 147.0207 token/s full-vocabulary control, and all three
request-wide calibration rates.
