# Qwen3.8-27B Q6_K RTX 4090 acceptance capture

This directory records Power's Qwen3.8 CUDA performance gates and subsequent
boundary tuning. Every accepted result was measured through Power's streaming
`POST /v1/completions` API. No native llama.cpp timing is used as an acceptance
result.

## Quality and speed by mode

`Quality proxy` below is fixed-task answer accuracy, not a claim about general
intelligence or an IQ score. The 100-task matrix and 12-task calibration have
different purposes and denominators, so their scores must not be compared
directly. `Request-wide` throughput includes prompt processing, generation,
and HTTP overhead; `steady decode` isolates a long repetitive generation after
warm-up. A dash means that no defensible apples-to-apples capture exists for
that cell.

| Artifact and runtime mode | Quality proxy | Request-wide throughput | Median steady decode | Interpretation |
| --- | --- | ---: | ---: | --- |
| Untouched Q6_K, autoregressive | 67/100 lenient; 60/100 strict (100 tasks, 3x) | 30.883 token/s | 35.5793 token/s (earlier capture) | Current fixed-task baseline; steady column is a separate historical shape |
| Untouched Q6_K, native DFlash2 paired control | Exact output in all five paired 256-token samples | 25.744 token/s median | 33.075 token/s | Same Q6_K target, request, build, and host controls as the native candidate |
| **Untouched Q6_K + native DFlash2 Q4 proposer, K7/S6** | **Exact output in all five paired samples** | **63.182 token/s median** | **144.453 token/s median; 141.267 minimum** | 4.367x decode and 2.454x end-to-end speedup; 98.230% acceptance; zero replay; opt-in synthetic boundary |
| Untouched Q6_K + DFlash2 Q4 proposer, representative quality matrix | 9/12 lenient and strict; **12/12 answer parity, 7/12 complete-output parity** | **45.143 token/s mean** | 108.429 token/s historical standalone | 1.520x mixed-workload speedup; quality evidence keeps the native profile out of lossless-default status |
| Untouched Q6_K, paired DSpark control | Exact 256-token greedy output in paired 3x capture | 25.171 token/s median | 32.249 token/s | Same request shape and clean binary as the external-draft candidate |
| **Untouched Q6_K + external DSpark Q4, K10/S6 (peak prompt)** | Exact paired 256-token output and receipt hashes | **65.825 token/s median** | **169.324 token/s** | 5.250x decode speedup; 90.873% acceptance; zero replay; all samples above 160 token/s |
| Untouched Q6_K + external DSpark Q4, K10/S6 (100 tasks, 3x) | 73/100 lenient; 59/100 strict; **54/100 exact-output parity** versus target-only | **32.678 token/s** | -- | 1.445x paired workload speedup; deterministic, but not eligible as a lossless production default |
| **Untouched Q6_K + adaptive external DSpark Q4, K10/S6 (controlled peak)** | Identical output and receipt hashes in all 3 samples | **63.535 token/s median** | **164.756 token/s median; 160.881 minimum** | Rollback-safe K6 start; 92.713% acceptance; zero replay; all samples above 160 token/s |
| Untouched Q6_K + adaptive external DSpark Q4, K10/S6 (100 tasks, 1x) | 69/100 lenient; 56/100 strict; **55/100 exact-output parity** versus the 67/100 and 58/100 control | **31.052 token/s** | -- | 1.358x paired speedup and zero replay, but three paired lenient losses keep it opt-in |
| **Untouched Q6_K + prefix-FR8192 MTP, fixed K6/S6, B8** | 9/12 lenient and strict in both off/MTP modes (1x; 3 truncated) | **46.923 token/s** | -- | General short-task profile; 63.42% faster than its 28.713 token/s paired off control |
| **Untouched Q6_K + prefix-FR8192 MTP, fixed K7/S6, B11, high-priority CUDA** | Same fixed-prompt digest as the controls | -- | **172.835 token/s** under a contended desktop | Current clean 9x peak profile; the earlier quiet-host capture remains 176.6109 token/s |
| Untouched Q6_K, full-vocabulary MTP, K7/S7 | Exact parity on the fixed peak prompt | -- | 147.0207 token/s | Current balanced steady-decode control |
| Untouched Q6_K, full-vocabulary MTP, K7/S6 | 5/12 lenient; 3/12 strict (1x; 11 truncated) | **47.032 token/s** | -- | Current small mixed-workload calibration winner |
| **Untouched Q6_K + prefix-FR8192 MTP, K7/S6** | 4/12 lenient; 3/12 strict (1x; 11 truncated) | 37.290 token/s | **176.6109 token/s** | Peak-only profile; proposal coverage is workload-sensitive |
| TBQ4 mixed artifact, autoregressive | 70/100 lenient; 64/100 strict (100 tasks, 3x) | 38.724 token/s | -- | Current non-speculative mixed-artifact control |
| TBQ4 mixed + full-vocabulary fixed MTP, K7/S7 | **76/100 lenient; 66/100 strict (100 tasks, 3x)** | **83.228 token/s** | **175.2089 token/s** | Balanced default: complete rollback window and zero replay |
| TBQ4 mixed + full-vocabulary guarded MTP, K7/S6 | 5/12 lenient; 3/12 strict (12 tasks, 3x) | 54.060 token/s | **177.7165 token/s** | Peak profile: one replay at most before request-local clamp |
| TBQ4 mixed + MTP + prefix FR, K7/S6 (historical) | 72/100 lenient; 60/100 strict (100 tasks, 3x) | 27.951 token/s | 184.3665 token/s | Rejected universal default: 25.55% mixed-workload acceptance |
| TBQ4 mixed + full-vocabulary fixed MTP, K7/S6 (pre-guard) | 4/12 lenient; 3/12 strict (12 tasks, 3x) | 28.226 token/s | -- | Historical failure: 46 exact prefix replays per run |
| TBQ4 mixed + full-vocabulary adaptive MTP, K7/S6 | 5/12 lenient; 3/12 strict (12 tasks, 3x) | 60.031 token/s | -- | Rollback-safe, but slower than fixed K7/S7 on the calibration |
| UD-Q8_K_XL, autoregressive heterogeneous placement | Quality matrix not run | -- | 6.3484 token/s | Fits through exact CPU/GPU tensor placement, but is bandwidth-bound |
| UD-Q8_K_XL, native MTP K4/S4 heterogeneous placement | Quality matrix not run; cross-mode output hashes differ | -- | 9.7577 token/s | Performance boundary only; not a parity or quality acceptance result |

The complete protocols and raw evidence are in the
[untouched-Q6_K report](PURE-Q6.md), the
[100-task and 12-task quality report](quality/README.md), the sections below,
[Q6_K-only native DFlash2 evidence](dflash2/README.md),
and the sibling [UD-Q8_K_XL boundary capture](../qwen3.8-27b-ud-q8-k-xl-rtx4090/README.md).

The [DFlash2 capture](dflash2/README.md) fixes the target to the same
22,884,408,288-byte Q6_K artifact in both modes. Its Q4 file is only a 1.14 GB
proposal model; no Q4 target result is included. Native Power reached 144.453
token/s median decode versus 33.075 target-only and 63.182 versus 25.744
token/s median end-to-end. All five paired outputs matched, acceptance was
98.230%, and replay was zero. This high-acceptance prompt still did not
demonstrate a stable 175 token/s boundary. The separate three-order 12-task
quality calibration retained 9/12 in both modes and all 12 extracted answers,
while only 7/12 complete response hashes matched; its 45.143 versus 29.702
token/s request-wide result keeps the profile opt-in.

The native [DSpark Q4 paired capture](dspark/README.md) is deliberately
separate from the MTP and TBQ4 matrices below. It uses a content-addressed
1.10 GB external drafter, keeps the 22.88 GB target unchanged, and records a
single-request context-512 boundary. Its exact output match is evidence for
that deterministic request. The newer context-1024 100-task matrix measures
the same profile across MMLU, GSM8K, and C-Eval; it retains the score but only
54/100 complete responses match target-only byte for byte, so the profile is
diagnostic rather than a lossless production default.

The current request-local follow-up starts inside the S6 rollback window and
opens K10 only after a fully accepted first probe. Its controlled peak passed
the 160 token/s median and all-sample gates at 164.756 and 160.881 token/s,
with 92.713% acceptance, 9.8077 verified tokens per target pass, and zero
replay. The paired 100-task run reached 31.052 versus 22.872 token/s (1.358x),
but contained five lenient gains and three losses, one strict gain and three
losses, and only 55/100 complete-output parity. The
[path-free adaptive evidence](dspark/adaptive/evidence.json) consequently marks
the profile opt-in rather than lossless-by-default.

## Prompt-prefix cache, 2026-08-22

An independent target-only capture isolates repeated-prefix reuse from decode
throughput. Across five fresh cold/warm pairs, the exact Q6_K model reused
9,740 tokens and reduced evaluated prompt tokens by 99.3843%.

| Paired metric | Cold median | Warm median | Speedup |
| --- | ---: | ---: | ---: |
| Backend prompt evaluation | 786.1375 ms | 33.4102 ms | **23.5299x** |
| Time to first token | 950.0142 ms | 72.1932 ms | **13.1593x** |

Flash Attention was enabled and speculation was disabled. The
[protocol, exact commands, ACL, and raw report](prompt-cache/README.md) bind
commit `84e1eec`, the unchanged model digest, health policy, metric deltas, and
each result receipt. This is prefill/TTFT evidence, not a steady-decode or
external-draft result.

## Q6_K deep-optimization result, 2026-08-22

The target GGUF remained byte-for-byte unchanged. The optimized server used
the same 22,884,408,288-byte Q6_K file and SHA-256
`562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727`.
Only execution shape and scheduling changed.

### Peak steady decode

All rows below used K7/S6, prefix-FR8192, one warm-up, three measured
1,024-token greedy requests, ten physical-core threads (`0x55555`), and the
same output SHA-256 `a54538ea...90523`. The shared Windows desktop reported
roughly 5--8% GPU utilization before these captures. The final row uses nine
measured requests from clean commit `f6326bb05bb8101c2335ec7c3c2f1e261fd86071`.

| Shape or scheduler | Median decode | Minimum | Result |
| --- | ---: | ---: | --- |
| Previous B14, Flash Attention on, ordinary CUDA streams | 160.932 token/s | 160.706 token/s | Loaded-desktop control |
| B11, Flash Attention off, rebuilt ordinary streams | 170.184 token/s | 166.895 token/s | Stable graph-shape and kernel-path gain |
| B11, Flash Attention off, high-priority CUDA streams, first order | 171.854 token/s | 168.920 token/s | Peak sample 174.452 token/s |
| B11, Flash Attention off, high-priority CUDA streams, reverse order | 172.252 token/s | 171.250 token/s | Development repeat |
| [B11, Flash Attention off, high-priority CUDA streams, clean 9x](deepopt-20260822/peak/deepopt-final-f6326bb-k7s6-b11-faoff-cudahigh-1024-9x.json) | **172.835 token/s** | **171.298 token/s** | Final clean capture; peak sample 175.533 token/s |
| B11 with CUDA graphs disabled | 133.876 token/s | -- | Rejected; graph launch reuse is essential |
| B11 with `GGML_CUDA_GRAPH_OPT=1` | 160.613 token/s | -- | Rejected; extra concurrent-graph work regressed this hybrid model |
| B11 with `CUDA_DEVICE_MAX_CONNECTIONS=32` | 168.900 token/s | -- | Rejected; CUDA's default connection count was faster |

The clean peak evidence includes the [environment receipt](deepopt-20260822/peak/deepopt-final-f6326bb-k7s6-b11-faoff-cudahigh-1024-9x.environment.json)
and [preflight receipt](deepopt-20260822/peak/deepopt-final-f6326bb-k7s6-b11-faoff-cudahigh-1024-9x.preflight.json).

The default CUDA graph implementation is therefore retained. B11 is the
selected fixed verification capacity for K7: one anchor plus seven proposals
forms an eight-row target graph, while the extra physical capacity satisfies
llama.cpp's recurrent splitter without inflating the B14 allocation. Flash
Attention remains enabled for long-context and portable default profiles, but is
disabled in the RTX 4090 short-batch throughput profiles because the recurrent
and small attention regions do not amortize its setup cost. This is a measured
architecture-specific choice, not a model-neutral default.

The optional `GGML_CUDA_HIGH_PRIORITY=1` backend patch creates all llama.cpp
CUDA streams at the device's greatest available priority. It improved median
and tail behavior under WDDM contention without changing weights, sampling, or
the fixed-prompt output. It is priority isolation, not physical GPU
exclusivity: `max_loaded_models=1`, `max_concurrent_requests=1`, and
`num_parallel=1` prevent competing Power work, but Windows display and other
processes can still preempt the GPU.

### Mixed-task profile

The best peak width is not the best general-task width. A paired 12-task,
256-token-cap run used the same Q6_K model, B8, Flash Attention off, and
high-priority CUDA streams:

| Mode | Request-wide throughput | Draft acceptance | Verified tokens / target pass | Score | Replays |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Speculation off, B8](deepopt-20260822/quality/r01-o01-off-b8.json) | 28.713 token/s | -- | -- | 9/12 lenient and strict | -- |
| [Fixed prefix-FR8192 K6/S6, B8](deepopt-20260822/quality/r01-o02-fr8192-k6-s6-b8-fixed.json) | **46.923 token/s** | 26.81% | 2.591 | 9/12 lenient and strict | 0 |
| Fixed prefix-FR8192 K7/S6, B11 | 40.095 token/s | 24.90% | 2.490 | 9/12 lenient; 9/12 strict | 12 |
| Adaptive prefix-FR8192 K7/S6, B11 | 35.178 token/s | 50.07% | 2.529 | 8/12 lenient and strict | 0 |

The clean paired [aggregate](deepopt-20260822/quality/sweep.json) and
[environment receipt](deepopt-20260822/quality/environment.json) bind the task
selection, server, model, ACL, affinity, and CUDA stream priority.

K6/S6 B8 was 63.42% faster than its paired target-only control, retained all
12 final answers, and matched the score; eight of twelve complete response
digests were identical. The remaining content differences are expected
floating-point trajectory differences between serial and batched target
graphs, so this small calibration is evidence of no observed score regression,
not proof of general intelligence parity.

The adaptive controller demonstrates why acceptance percentage is not the
objective. It raised measured acceptance by shortening proposals and sent 8 of
12 requests through a one-token target-only circuit, but variable K shapes
lost CUDA graph reuse and reduced throughput by 12.3% versus fixed K7/S6.
Fixed K7/S7 removed replay and reached 45.543 token/s; fixed K6/S6 removed the
same replay with a smaller stable graph and reached 46.579 token/s at B11. A
two-order B8/B10 follow-up measured 48.096 and 48.178 token/s respectively;
the 0.17% difference is noise, so B8 wins on the higher minimum, higher
acceptance, and smallest legal allocation.

The two checked-in profiles make that workload split explicit:

- [`pure-q6-mtp7-snap6-fr8192-rtx4090-throughput.acl`](pure-q6-mtp7-snap6-fr8192-rtx4090-throughput.acl): peak K7/S6 profile; send `num_batch=11`.
- [`pure-q6-mtp6-snap6-fr8192-rtx4090-general.acl`](pure-q6-mtp6-snap6-fr8192-rtx4090-general.acl): mixed-task K6/S6 profile; send `num_batch=8`.

From first principles, steady throughput is approximately emitted tokens per
target pass divided by draft, verification, synchronization, and sampling
time. On the peak prompt K7 already accepts 96.64% of proposals and emits 7.75
tokens per target pass, leaving only about 3.2% perfect-acceptance headroom.
The remaining stable-175 gap on this machine is host contention: the current
software reaches 171--176 token/s with a 5--8% busy WDDM desktop, while the
earlier quiet-host run reached a 176.611 token/s median. A defensible 175+
service floor therefore requires a quiet/dedicated GPU or headless compute
host; it cannot be guaranteed by another inference flag.

## Current untouched-Q6_K boundary

The clean `eb6aeda59561eff3e4e7592704cab6fc863b72c7` capture pins the
original 22,884,408,288-byte GGUF with SHA-256
`562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727`.
No target or draft-head weight was requantized.

| Pure Q6_K mode, 1 warm-up + 9x1,024 tokens | Median decode | Minimum | Median end to end | At least 175 |
| --- | ---: | ---: | ---: | ---: |
| [Full vocabulary, K7/S7](pure-q6-full-vocabulary-1024-9x.json) | 147.0207 token/s | 146.0917 token/s | 140.2573 token/s | 0 / 9 |
| [Prefix FR8192, K7/S6](pure-q6-fr8192-1024-9x.json) | **176.6109 token/s** | 173.2630 token/s | **167.3519 token/s** | **7 / 9** |

The prefix-FR capture passes a 175 token/s median gate, not a 175 token/s
all-sample gate. Its two sub-threshold samples are part of the result. Use the
[reproduction guide](REPRODUCE.md#promote-a-stable-175-tokens-floor) for the
separate median, minimum-sample, and idle-GPU promotion checks.

The 20.13% steady-decode gain retained the same deterministic output SHA-256,
but it did not generalize uniformly. In a one-pass 12-task calibration,
full-vocabulary K7/S6 reached 47.032 token/s request-wide at 52.30% acceptance;
prefix-FR8192 reached 37.290 token/s at 24.82% acceptance. Eleven tasks per mode
hit the output cap, so those scores are calibration diagnostics rather than an
intelligence result. Full vocabulary remains the balanced profile; prefix FR
is a measured high-coverage peak profile.

For first-principles analysis, raw environment receipts, the dynamic-Q8_1
activation experiment, limitations, and exact reproduction commands, see the
[untouched-Q6_K performance boundary](PURE-Q6.md) and
[reproduction guide](REPRODUCE.md).

## Representative workload, repeated three times

The peak gate below is intentionally complemented by a fixed 100-task quality
and workload-throughput matrix. It uses 50 MMLU, 20 GSM8K, and 30 C-Eval
tasks, runs each mode three times in cyclic order, and publishes the task
cache, environment, nine per-request reports, and paired statistics.

| Mode | Lenient score | Strict score | Mean workload throughput |
| --- | ---: | ---: | ---: |
| Untouched Q6_K, speculation off | 67/100 | 60/100 | 30.883 token/s |
| TBQ4 mixed artifact, speculation off | 70/100 | 64/100 | 38.724 token/s |
| TBQ4 + full-vocabulary MTP, fixed K7/S7 | **76/100** | **66/100** | **83.228 token/s** |

TBQ4-off was 25.4% faster than untouched Q6_K on this current workload. Fixed
K7/S7 was 114.9% faster than TBQ4-off, accepted 51.33% of proposals, verified
4.543 tokens per target pass, and recorded zero fallback replay or guard
activation. Relative to TBQ4-off it had seven lenient gains and one loss
(`p=0.0703`) and three strict gains and one loss (`p=0.625`). The measured
scores did not fall, but the differences do not establish a statistically
significant general-intelligence improvement. All predictions were stable
across repetitions and all 900 requests completed without error.

The historical prefix-FR matrix remains useful negative evidence: MTP was
33.0% slower than its TBQ4-off control because workload-wide acceptance fell
to 25.55%, including 14.21% on C-Eval. Full-vocabulary K7/S7 removes that draft
coverage bottleneck. See the [quality and workload matrix](quality/README.md)
for the full protocol, paired statistics, limitations, replay command, and
machine-readable evidence.

A post-change calibration then replayed a fixed 12-task subset three times with
the full 248,320-row draft vocabulary. It is intentionally smaller than the
100-task release matrix, but it isolates the rollback-window decision on the
current binary:

| Full-vocabulary mode | Mean workload throughput | Draft acceptance | Fallback replays per run | Lenient score |
| --- | ---: | ---: | ---: | ---: |
| TBQ4, speculation off | 35.048 token/s | -- | -- | 5/12 |
| Fixed K7, six snapshots (pre-guard) | 28.226 token/s | 48.54% | 46 | 4/12 |
| Adaptive K7, six snapshots | 60.031 token/s | 65.50% | 0 | 5/12 |
| Fixed K7, seven snapshots | **68.211 token/s** | 49.67% | 0 | 5/12 |
| Guarded fixed K7, six snapshots (current) | 54.060 token/s | 53.07% | 11 | 5/12 |

The complete K7 rollback window made fixed full-vocabulary MTP 94.6% faster
than speculation-off on this calibration and 13.6% faster than adaptive S6.
The current request-local guard makes six snapshots bounded: after one exact
replay, that request remains clamped to the six-snapshot window. It improved
the short fixed-S6 calibration by 91.5% over the pre-guard result, but current
S7 still reached 68.205 token/s with no replay and is the balanced default.
See the
[calibration evidence](quality/full-vocab-rollback-calibration-rtx4090-3x.json)
and the [current compact evidence](quality/full-vocabulary-s7-current-rtx4090-3x.json)
with reproduction commands in the quality README.

## Mixed-artifact full-vocabulary batched-greedy boundary

The mixed-artifact release profile removes FR from its performance gate and
retains all 248,320 draft-head rows. The current binary passes topology-pinned
nine-sample median gates with both guarded K7/S6 and rollback-complete K7/S7;
two earlier quiet-WDDM builds provide independent high-water captures.
Predecessor captures are retained to show the range introduced by a shared
WDDM display GPU:

| Capture | Median decode | Minimum decode | Median end-to-end | Gate |
| --- | ---: | ---: | ---: | --- |
| [Current guarded K7/S6, affinity, 1,024 tokens, 9x](quality/full-vocabulary-s7-current-rtx4090-3x.json) | **177.7165 token/s** | **176.7287 token/s** | 169.0438 token/s | 175 passed, 9 / 9 |
| [Current rollback-complete K7/S7, affinity, 1,024 tokens, 9x](quality/full-vocabulary-s7-current-rtx4090-3x.json) | **175.2089 token/s** | 174.2211 token/s | 166.3457 token/s | 175 median passed, 5 / 9 |
| [Earlier physical-core-affinity gate, 1,024 tokens, 9x](final-affinity-1024-9x.json) | **177.3062 token/s** | **175.5958 token/s** | 168.9809 token/s | 175 passed, 9 / 9 |
| [Earlier 1,024-token control, 9x](final-draft-greedy-pmin-guard-1024-9x.json) | **187.6094 token/s** | **183.7360 token/s** | 178.6574 token/s | 175 passed, 9 / 9 |
| [Earlier 1,024-token repeat, 9x](final-draft-greedy-pmin-guard-1024-9x-repeat.json) | **188.2972 token/s** | **184.4913 token/s** | 179.2680 token/s | 175 passed, 9 / 9 |
| [1,024 tokens, unlocked, 5x](batched-greedy-1024-5x.json) | **182.1530 token/s** | **180.7195 token/s** | 173.4201 token/s | 175 passed |
| [256 tokens, 2,745 MHz control, 9x](batched-greedy-256-locked2745-9x.json) | **175.9190 token/s** | 169.3528 token/s | 152.0363 token/s | 175 median passed |
| [256 tokens, contended desktop, 9x](batched-greedy-256-contended-9x.json) | 159.8593 token/s | 152.4855 token/s | 139.8559 token/s | 175 failed |
| [Post-removal 256-token regression, 3x](post-device-carry-removal-256-3x.json) | 167.3977 token/s | 165.0082 token/s | 140.8385 token/s | Output regression passed |

The current binary has two explicit host-staged CPU-embedding profiles. Guarded
K7/S6 preserves the 177.7165 token/s peak and kept all nine long samples above
175; rollback-complete K7/S7 trades 1.41% median steady decode for zero replay
on mixed workloads and passed the median gate at 175.2089 token/s. Both pin ten
worker threads to one logical processor per physical Xeon W5-2445 core with
mask `0x55555`. That mask is specific to this CPU topology and remains an
explicit runner option rather than a product default.

The earlier order-balanced affinity A/B measured a 176.8276 token/s combined
median with the mask versus 173.0114 token/s across all 20 logical processors,
a 2.21% gain. Its complete A/B and environment remain recorded in
[the compact comparison](affinity-ab-1024-12x.json), the
[earlier final report](final-affinity-1024-9x.json), and its
[environment capture](final-affinity-1024-9x.environment.json).

All samples in every listed capture produced the expected deterministic digest:
`584e2b93...be6eb` at 256 tokens and `a54538ea...90523` at 1,024 tokens. The two
historical high-water environment captures pin server SHA-256
`e982060f...a93f7`, client
SHA-256 `0d19efb0...af2f1`, the exclusive `llama.cpp` backend assertion, model,
ACL, requested 2,745 MHz clock lock, power plan, and GPU processes:
[control](final-draft-greedy-pmin-guard-1024-9x.environment.json) and
[repeat](final-draft-greedy-pmin-guard-1024-9x-repeat.environment.json).
The predecessor captures pin server `d7e6fa2e...c324d1` and client
`6b1b650b...f2f8a7`: [long](batched-greedy-1024-5x.environment.json),
[locked](batched-greedy-256-locked2745-9x.environment.json), and
[contended](batched-greedy-256-contended-9x.environment.json). The contended
capture also records all 47 `nvidia-smi` process
snapshot lines. The locked capture's environment contains an explicit
post-capture annotation because native clock-control provenance was added to
the runner immediately afterward. The wrapper reset clocks in its `finally`
block; the final repeat returned to an idle 210 MHz after reset.

The guarded [K7/S6 host-staged ACL](mtp7-snap6-full-vocab-cpu-embedding.acl)
uses six recurrent snapshots. The balanced
[K7/S7 host-staged ACL](mtp7-snap7-full-vocab-cpu-embedding.acl) retains the
complete seven-token rollback window. Both disable the recurrent device chain,
enable Flash Attention, use ten host threads, retain the complete draft
vocabulary, and use batch 14 for the long gate. The historical
[device-chain ACL](mtp7-snap6-full-vocab.acl) remains only for the optional
recurrent experiment, while [the earlier S7 ACL](mtp7-snap7-full-vocab.acl)
records the GPU-embedding calibration shape.

For fixed configurations where proposal width exceeds resident snapshots, the
runtime now owns a request-local rollback guard. The first rejection that
cannot be restored from a snapshot performs the exact legacy prefix replay;
the guard then clamps every later round in that request to the rollback-complete
width. Metrics expose guarded requests, activation count, activation round, and
the resulting draft limit. No global policy changes, and a high-acceptance
request that never needs replay follows the original K7/S6 fast path exactly.

The new backend fast path is deliberately narrow: one active sequence, dense
output rows, and a sampler chain containing only backend greedy sampling. It
runs one matrix `argmax` for all verification rows and performs one contiguous
device-to-host token copy. Other sampler chains and sparse or multi-sequence
graphs retain llama.cpp's ordinary per-row path. In a same-session 1,024-token
A/B, the five-sample median moved from 180.7365 to 183.4028 token/s (1.5%),
while the minimum moved from 174.5714 to 183.0585 token/s. GPU telemetry showed
2,725 versus 2,718.5 MHz average active clocks, so the gain was not produced by
a higher batched-run clock.

Pure-greedy requests now extend the same contract to the MTP draft context.
When `spec_draft_p_min = 0`, llama.cpp installs a backend greedy sampler and
reads the selected token after one draft-step synchronization. It does not
construct the former Top-K-10 probability candidates or invoke the common CPU
sampler. Any grammar, penalty, stochastic filter, or positive draft-probability
threshold retains the general path.

The implementation was measured with two interleaved pre/post blocks, three
1,024-token samples per block after one warm-up, batch 14, a requested 2,745 MHz
clock lock, and the same pure-llama.cpp benchmark client:

| Draft sampler | Run 1 median | Run 2 median | Combined mean | Combined median |
| --- | ---: | ---: | ---: | ---: |
| Top-K candidate path ([r1](ab-draft-greedy-v2-pre-r1-1024-3x.json), [r2](ab-draft-greedy-v2-pre-r2-1024-3x.json)) | 169.7460 token/s | 173.6984 token/s | 170.6841 token/s | 170.2021 token/s |
| Backend greedy ([r1](ab-draft-greedy-v2-post-r1-1024-3x.json), [r2](ab-draft-greedy-v2-post-r2-1024-3x.json)) | 172.3998 token/s | 172.9412 token/s | **172.7103 token/s** | **172.6705 token/s** |

The combined mean improved by 1.19% and the combined median by 1.45%. This is a
small causal hot-path reduction; WDDM variance is larger than the per-block
delta, so it does not by itself explain the later 188.2972 token/s boundary.
All twelve measured samples emitted the same
`a54538ea...90523` digest. The baseline server SHA-256 was
`b0768be3...b09fb1`, the candidate was `144cbd40...b978d`, and both used client
`f24ffe42...e69d8`. Companion environment captures pin the binaries, model,
ACL, power plan, requested clock lock, and contemporaneous GPU processes:
[pre r1](ab-draft-greedy-v2-pre-r1-1024-3x.environment.json),
[post r1](ab-draft-greedy-v2-post-r1-1024-3x.environment.json),
[pre r2](ab-draft-greedy-v2-pre-r2-1024-3x.environment.json), and
[post r2](ab-draft-greedy-v2-post-r2-1024-3x.environment.json).

### CUDA control-plane profile

A request-scoped Nsight Systems inspection of the current full-vocabulary
backend-greedy path exposed the synchronization and transfer structure of the
remaining critical path. The measured 256-token window contained 236 CUDA
graph launches and
6,050 `cudaStreamSynchronize` calls with 1.3297 seconds of cumulative API time.
Each of the 236 recurrent observations copied one 20,480-byte row from device
to host and one four-byte sampled token from device to host. The row is exactly
5,120 FP32 hidden elements. Matching 20,480-byte host-to-device copies then
feed the staged hidden state back into subsequent draft graphs.

| Profile observation | Sample-window value |
| --- | ---: |
| Traced request duration | 1.8035 s |
| Decode throughput under profiler | 145.69 token/s |
| CUDA graph launches | 236 |
| Stream synchronizations | 6,050 calls / 1.3297 s cumulative |
| Async-copy API time | 57.68 ms cumulative |
| Backend argmax | 38 calls / 2.81 ms cumulative |
| GPU kernel + memcpy interval union | 164.91 ms |

The compact, machine-readable evidence is retained in
[nsys-draft-greedy-256-summary.json](nsys-draft-greedy-256-summary.json). The
369.9 MB report and its 5.34 GB SQLite export are deliberately not versioned or
retained. Recreate the report with the checked-in
[`profile-qwen38-cuda.ps1`](../../../tools/profile-qwen38-cuda.ps1), the archived
ACL, model hash, server hash, and benchmark hash recorded in the summary. The
GPU interval union is diagnostic only: profiler overhead plus WDDM/CUPTI graph
accounting prevents treating it as a raw utilization percentage. Transfer and
API-call counts expose the control-flow boundary without that assumption.

The trace motivated a backend-resident hidden-row carry experiment. An
order-reversed eight-sample A/B found a 0.09% median gain but a 0.38% mean
regression, with identical output and acceptance paths. The result did not
reproduce beyond WDDM noise, so the extra cross-context state was removed. The
[compact A/B evidence](device-carry-ab-256-8x.json) prevents that rejected path
from being rediscovered. The next useful backend optimization must reduce or
fuse target/draft graph work itself, such as an exact device-side recurrent
draft loop; moving the same 20 KiB row without reducing graph execution is not
enough.

This is steady-state decode evidence, not representative application
throughput or a guaranteed 175 token/s service level. The controlled result
and repeat show that the final software path can remain above the requested
boundary for 18 / 18 long samples; the failed predecessor run shows that a busy
Windows display GPU can still erase that margin. A dedicated
GPU, headless Linux, or an otherwise exclusive benchmark window is required
before promoting 175 token/s to a reproducible floor. The current 100-task,
three-repetition matrix now covers this binary and K7/S7 profile: it recorded
76/100 lenient, 66/100 strict, 83.228 token/s request-wide throughput, zero
replay, and zero guard activation. That is evidence of no observed regression
on the fixed sample, not a general model-intelligence guarantee.

The complete build, input verification, offline evidence audit, quality replay,
and source-validation procedure is in [REPRODUCE.md](REPRODUCE.md). Its runner
resolves the current Git revision, records executable and input hashes plus
host controls, rejects backend drift, retains failed-threshold JSON, and resets
an explicit GPU clock lock even when the run fails.

## Historical prefix-FR 175 token/s gate

Before the batched-greedy path, four five-sample captures of the experimental
8,192-token-ID-prefix path passed the declared 175
token/s median gate. The last two use the final rebuilt binaries; the earlier
two are retained to expose run-to-run WDDM variation:

| Capture | Median decode | Minimum decode | Samples at least 175 token/s |
| --- | ---: | ---: | ---: |
| [FR-Spec 5x](fr-spec-5x.json) | 185.4103 token/s | 176.6460 token/s | 5 / 5 |
| [FR-Spec 5x repeat](fr-spec-5x-repeat.json) | 182.1038 token/s | 180.6591 token/s | 5 / 5 |
| [Final binary 5x](fr-spec-final-5x.json) | 176.6444 token/s | 172.2354 token/s | 4 / 5 |
| [Final binary 5x repeat](fr-spec-final-5x-repeat.json) | 184.3665 token/s | 182.5627 token/s | 5 / 5 |

All twenty outputs have SHA-256
`584e2b93ba21d7c727456567762c6bbacc150d43156c73ed91c1c0cbb13be6eb`.
Nineteen of twenty measured samples exceeded 175 token/s; the one 172.2354
token/s sample is retained rather than discarded. The final-binary hot repeat
places all five samples between 182.5627 and 187.9386 token/s.
Each request used 35 target passes, drafted 235 tokens, accepted 220, reached
93.6170% draft acceptance, and required no target-prefix replay. The target
context retained the full vocabulary and verified every committed token.

This development gate combines the following first-principles reductions:

- The 22.88 GB Q6_K source was converted into a 19.19 GB mixed-precision
  runtime artifact: the main-block FFN down, gate, and up matrices use Q4_0,
  the MTP block remains Q6_K, and the separate MTP head uses Q4_K. This reduces
  the dominant weight-memory traffic per target and draft pass. Target and MTP
  contexts share the single loaded model allocation; only the draft-specific
  head is added.
- Native MTP uses a width of seven and six resident recurrent snapshots, with
  exact replay available beyond the snapshot window. Batch 14 was the fastest
  stable CUDA graph shape in the final sweep.
- The experimental `spec_mtp_fr_vocab_size = 8192` path projects only the
  leading 8,192 vocabulary rows in the draft LM head and pads the remaining
  draft logits with negative infinity. This is an FR-Spec-inspired prefix
  specialization, not a corpus-derived frequency map. It is workload-sensitive
  and must be revalidated for other languages and domains.
- llama.cpp backend sampling remains inside the CUDA graph. Power's pinned
  llama.cpp patch skips full-vocabulary CPU row swaps when raw logits or
  sampler buffers were not populated, while preserving swaps for live rows.
- Flash Attention, full CUDA layer offload, ten host threads, the Windows
  high-performance power plan, and High process priority were active.

The optimized GGUF is derived from Q6_K but is **not** an untouched pure-Q6_K
artifact. The same-artifact result below remains the reference for the original
22.88 GB Q6_K file. Full-target verification prevents unverified draft tokens
from being committed, but the serial and block execution paths are not
bitwise-identical. In the representative matrix, TBQ4-off and TBQ4 + MTP + FR
had 88/100 answer parity, 34/100 output-hash parity, and 59/59 answer parity
where neither response was truncated. Selective weight requantization and MTP
therefore retain separate paired quality gates.

The binding and llama.cpp changes are captured in
[`patches/llama-cpp-rs-dfd12e4-mtp-dynamic-k.patch`](../../../patches/llama-cpp-rs-dfd12e4-mtp-dynamic-k.patch)
and
[`patches/llama-cpp-rs-dfd12e4-mtp-fr-spec.patch`](../../../patches/llama-cpp-rs-dfd12e4-mtp-fr-spec.patch).
After `cargo fetch`, apply them idempotently on Windows with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\apply-llamacpp-power-patches.ps1
```

The development GGUF can be reproduced with the pinned llama.cpp quantizer.
The first command changes exactly 192 FFN tensors in blocks 0 through 63; the
second creates a source for the separate draft head, and the Python tool copies
that head into the mixed target without changing its other tensors:

```powershell
llama-quantize.exe --allow-requantize `
  --tensor-type '^blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_(down|gate|up)\.weight$=Q4_0' `
  Qwen3.8-27B-Q6_K.gguf qwen38-tbq4-ffn.gguf Q6_K 10

llama-quantize.exe --allow-requantize --output-tensor-type Q4_K `
  Qwen3.8-27B-Q6_K.gguf qwen38-q4k-head-source.gguf Q6_K 10

py -3 tools\add-gguf-mtp-head.py `
  qwen38-tbq4-ffn.gguf qwen38-q4k-head-source.gguf `
  qwen38-tbq4-ffn-mtp-head-q4k.gguf
```

## Historical same-artifact Q6_K result

| Metric | Explicit off | Native MTP |
| --- | ---: | ---: |
| Median steady-state decode | 35.5793 token/s | 140.1600 token/s |
| Minimum measured decode | 35.4812 token/s | 139.4793 token/s |
| Median speedup | 1.0000x | 3.9394x |
| Samples after warm-up | 5 | 5 |
| Completion tokens per sample | 256 | 256 |

This earlier MTP median is 40.1600% above the 100 token/s acceptance floor, and
every individual MTP sample is at least 39.4793% above it. All ten measured
outputs have SHA-256
`584e2b93ba21d7c727456567762c6bbacc150d43156c73ed91c1c0cbb13be6eb`,
so greedy output parity passes both within and across modes. The verified
comparison is in [final-comparison.json](final-comparison.json); the full raw
reports are [final-baseline.json](final-baseline.json) and
[final-mtp.json](final-mtp.json).

The current untouched-Q6_K capture supersedes this historical performance
boundary at the same model identity: full-vocabulary K7/S7 now reaches
147.0207 token/s, while the workload-sensitive prefix-FR8192 K7/S6 profile
reaches 176.6109 token/s. The historical reports remain immutable evidence for
the original implementation stage.

### Post-safety rebuild confirmation

After adding the `draft_max + 2` recurrent-batch safety check, the current
release binaries were rebuilt and the original balanced-power, ten-thread,
unlocked-clock A/B was repeated. That conservative follow-up reached a
32.1710 token/s baseline median and a 129.7065 token/s MTP median, with a
125.8369 token/s MTP minimum, 4.0318x speedup, and the same output digest.
The comparison is in
[post-safety-comparison.json](post-safety-comparison.json); its raw reports are
[post-safety-baseline.json](post-safety-baseline.json) and
[post-safety-mtp.json](post-safety-mtp.json). This confirms that the safety
fix still clears the gate; it does not replace the 140.1600 token/s best
controlled capture above with a slower result collected while the Windows
desktop had several active WDDM clients.

Each final MTP request used 35 target passes, drafted 235 tokens, accepted 220,
achieved a 93.6170% draft acceptance rate, and emitted 7.2857 tokens per
target pass. None of the six requests, including warm-up, required a target
prefix replay; the longest rejected suffix was five tokens. The server log also
confirmed CUDA fused Gated Delta Net execution and the Qwen3.8 native
prediction tensors.

The previous 130.9225 token/s tuned checkpoint remains available as
[tuned-comparison.json](tuned-comparison.json),
[tuned-baseline.json](tuned-baseline.json), and [tuned-mtp.json](tuned-mtp.json).
The earlier conservative checkpoint remains available as
[comparison.json](comparison.json), [baseline.json](baseline.json), and
[mtp.json](mtp.json). It recorded a 109.3003 token/s median, 101.4822 token/s
minimum, and 3.0834x speedup with `draft_max=6` and `num_batch=8`.

## Optimization boundary

- The three-sample median width curve at `num_batch=24` was 100.2956,
  110.0070, 119.9584, and 127.2871 token/s for widths 3 through 6. Width 7
  remained the clear optimum at roughly 140--142 token/s. Widths 8 through 15
  were slower as acceptance fell and verification work grew; the widest cases
  dropped below 50 token/s.
- Diagnostic three-sample batch medians for width 7 were 139.7589, 139.1133,
  141.4218, 138.6824, and 142.0743 token/s for batch sizes 12, 16, 24, 32,
  and 48. The apparent batch-48 gain was below run-to-run noise and did not
  reproduce strongly enough to replace the established batch-24 capture.
- Six recurrent snapshots can match seven on a high-acceptance peak prompt.
  Before the guard, however, the three-run mixed-workload calibration recorded
  exactly 46 fallback replays in every fixed K7/S6 run and only 28.226 token/s.
  The request-local guard reduced this to 11 replays and raised throughput to
  54.060 token/s. Current K7/S7 still removed every replay and reached 68.205
  token/s, so K7/S6 remains peak-only and K7/S7 remains the fixed
  mixed-workload default. Four or fewer snapshots cross the replay boundary
  even more often. Raising `draft_p_min` above 0.7 also increased target-pass
  count and slowed decoding.
- The 2026-08-22 B9--B16 peak sweep supersedes the earlier batch-14 guidance.
  B11 was the stable K7 CUDA graph shape: on 1,024-token repeats it composed
  with Flash Attention off and high-priority CUDA streams to reach a 172.835
  token/s clean nine-run median under desktop contention. The mixed-task K6
  profile instead
  uses the minimum legal B8 shape; B8 and B10 differed by only 0.17% across a
  two-order 256-token sweep.
- Flash Attention is now profile-specific. Disabling it for the K7/B11
  short-batch hybrid path raised a repeated 1,024-token median from 164.678 to
  169.500 token/s with the same output digest. Long-context profiles
  retain Flash Attention because this result does not characterize prompt
  ingestion or large attention regions. Temporary instrumentation also
  observed 531 llama.cpp output-reorder calls in a 64-token request and zero
  pending row swaps, ruling out CPU vocabulary row reordering as the hot spot.
- CUDA graphs remain enabled: disabling them fell to 133.876 token/s, while
  `GGML_CUDA_GRAPH_OPT=1` regressed to 160.613 token/s. A reviewed optional
  high-priority-stream patch produced a 172.835 token/s median and 171.298
  minimum in the clean nine-run capture. This mitigates WDDM
  tail latency but does not reserve the physical GPU from other processes.
- Ranked reduced-vocabulary follow-ups did not recover representative
  throughput. A 65,536-row candidate reached 69.264 token/s at 43.2% draft
  acceptance, while a 131,072-row candidate reached 74.980 token/s at 50.4%.
  The full-vocabulary control reached 90.654 token/s at 69.1% acceptance on
  the same calibration task. Lower projection traffic was outweighed by extra
  target passes, so ranked FR is not part of the current optimization path.
- Requantizing the target `output.weight` also regressed this CUDA kernel mix.
  In the same earlier 256-token configuration, the Q6_K-head control reached
  a 160.687 token/s median and Q5_K reached 154.840 token/s; a clean 64-token
  Q4_0 diagnostic reached 149.591 token/s. These were routing experiments, not
  quality approvals, and their generated model files were removed.
- Single-sample peaks around 143--145 token/s were observed during tuning, but
  they are not presented as the acceptance result. The reproducible claim is
  the final five-sample median and minimum above.
- A follow-up host-control sweep found that the Windows high-performance power
  plan raised one unlocked ten-thread median from 129.7065 to 132.7665 token/s.
  Requested 2850, 3000, and 3105 MHz graphics-clock runs reached medians of
  134.9730, 137.4782, and 136.6937 token/s, respectively, but the driver only
  exposed up to 2745 MHz during sampled work and the gain did not compose with
  the best thread result. Every clock request was reset after its run.
- A three-sample high-performance-plan thread sweep produced medians of
  127.3598, 129.4400, 130.5746, 125.6633, 134.4634, 136.5069, and 136.8842
  token/s for 4, 6, 8, 10, 12, 16, and 20 threads. Five-sample 20-thread
  repeats ranged from 131.8561 to 137.3431 token/s and did not beat the
  ten-thread 140.1600 capture. Because desktop contention exceeded the
  apparent gain, neither 20 threads, process priority, fixed clocks, nor the
  high-performance plan became the documented default; the original balanced
  plan and unlocked clocks were restored.

## Artifact and build identity

| Item | Value |
| --- | --- |
| GGUF source | `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q6_K.gguf` |
| Hugging Face revision | `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe` |
| ModelScope mirror revision | `3bce06d3ab9ceadbca9f5b7f496adbf6835b2f08` |
| GGUF byte length | `22884408288` |
| GGUF SHA-256 | `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` |
| Deep-optimization Power commit | `f6326bb05bb8101c2335ec7c3c2f1e261fd86071` |
| Deep-optimization server SHA-256 | `a2b1ef3eab435dca02ca6dc41415f21c91c0f84d424ebfd0c7c589a992c555cc` |
| Clean peak report SHA-256 | `9d8d767eaccdbea5c3ad09783556ed940a6d5e66ecfea482e80b58db631492ca` |
| Clean peak environment SHA-256 | `2c43d2ad8703aee64051b363fd58735965950bdd6fffb6607b09660a90934c63` |
| Clean peak preflight SHA-256 | `6ca1259c687a4dd08c3759cc59f3a74fd4a5f7f2a0dea96a6de3da6df0995c2a` |
| Clean general paired sweep SHA-256 | `05f29c83397e664d02563ef45396bfbcfabaca91ae2fbde853063cf91a9b4e7f` |
| Optimized GGUF byte length | `19187686464` |
| Optimized GGUF SHA-256 | `5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6` |
| Final affinity server SHA-256 | `3e7b990c511934a35e2886e73742846901e9dcdefa02a7324592a78aa4a5b24f` |
| Final affinity benchmark client SHA-256 | `6665ae6a0fd7a14245445161cc1468d4b092a96691d496388db46917c8b041e0` |
| Final affinity ACL SHA-256 | `2f348cca96282a22650d9766cffa81251ea10a5e34a089bcc91b0822ab5c1d0e` |
| Final affinity prompt SHA-256 | `d95a5e4dad822ba9c84138f7a120017318bcb3a6a90e77246a8ec4ede0e65d89` |
| Final affinity report SHA-256 | `ad478dfdb3df7d3560fb39eeb2be854715bbb483a985c8e10793df053c0c2d72` |
| Final affinity environment SHA-256 | `c0cb352e99dcdd2c9bd487cb7501a1f231c97ef5a661c26446ef3c2e4f5eda8d` |
| Final affinity Power source base | `955b4552ca091af07818573e803f9369488a63f9` (dirty worktree disclosed in environment) |
| Pre-affinity batched-greedy server SHA-256 | `d7e6fa2e883f140096c4ab930519535363224eee662d7eb30956873128c324d1` |
| Pre-affinity benchmark client SHA-256 | `6b1b650b73fd738bddacd5a85acef7cd7708dc6b39987c4f75210bcacdf2f8a7` |
| Pre-validation optimized server SHA-256 | `97cd56cbfc5cca5f3fc3d1e969cd4af804b47b9c00b60786b36879e39014229d` |
| Pre-validation optimized benchmark SHA-256 | `faf0bbc824c6b27e62eea3909d9a84480e987caf5e72e7b6efe278c2b9338e75` |
| Final optimized server SHA-256 | `3ec225e412f27eda4677288332988977201c038dfb09a76eed7c0593e9db7eea` |
| Final optimized benchmark SHA-256 | `0e752978ca9521319a50b8eb78232f342680d5d7ae1f4bae521541697ea45075` |
| Historical Power source base | `491184ada54699ddfc4b40246cd6aee92d7550dd` |
| 140.1600 capture server executable SHA-256 | `bfba63bad8b2d6af148b092b75e784de0e4fd7f31109c7001625f3236841e2c1` |
| 140.1600 capture benchmark executable SHA-256 | `eed7b1da30eef87363d95d96ee67b971a9bb7c8ba7cea91f999090e4260dc24e` |
| Post-safety server executable SHA-256 | `e46c8261e8fee1f8b738d29c2f2cc79c328bb1bbb16bbf0c0bab126caab54e74` |
| Post-safety benchmark executable SHA-256 | `1fd2a3dd646ca2ebcd2dfa05380a4684b615b86dc01776ec2ed0afa29394b5a7` |
| Previous tuned server executable SHA-256 | `dfed5dab4e4cbe380ce933b9cdd5ddb276fc20409612b70adab52387ad70616f` |
| Initial server executable SHA-256 | `8c06148132b8bd4dd16209b487d8f893dcfe4152a98f03f5408811a3a5528876` |
| llama-cpp-rs revision | `dfd12e4d334846367e4284a2a7763fe92c1bf676` (llama.cpp b10405 compatibility update) |
| Toolchain | Rust 1.97.1, CUDA 12.6 |
| Final capture time | 2026-08-18T06:50Z |
| Post-safety capture time | 2026-08-18T10:00Z |
| Optimized captures time | 2026-08-18T20:57Z--21:11Z |
| Current batched-greedy captures time | 2026-08-19T10:52Z--11:11Z |
| Final affinity capture time | 2026-08-19T20:49Z |

The 140.1600 capture predates the validation-only recurrent-batch safety fix;
the post-safety reports pin the rebuilt implementation from that stage. The
original, optimized, and current batched-greedy executables were built from a
dirty working tree based on their recorded Power commits. Therefore the
executable digests above, rather than a base commit alone, are the exact binary
identities. Release evidence should repeat the capture from the eventual clean
commit.

### Historical prefix-FR capture controls

- Both captures used one warm-up followed by five measured requests.
- `max_tokens=256`, `num_ctx=4096`, `num_batch=14`, seed 42,
  `temperature=0`, and `top_p=1`.
- `draft_max=7`, `mtp_recurrent_snapshots=6`,
  `spec_mtp_fr_vocab_size=8192`, `draft_min=0`, and `draft_p_min=0.0`.
- Prompt SHA-256 was
  `d95a5e4dad822ba9c84138f7a120017318bcb3a6a90e77246a8ec4ede0e65d89`;
  request SHA-256 was
  `2744b65126aa7004d9d675596aac0c9ec5f3ba593c77e846221b02faaeae92ab`.
- No concurrent CUDA compute process from the repository test suite was
  present before or after either capture.

With the optimized artifact registered under the documented Power home, the
historical prefix-FR gate can be repeated from the crate root with:

```powershell
.\tools\run-qwen38-q6k-benchmark.ps1 `
  -Label fr-final `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\fr-spec-mtp7-snap6.acl `
  -PowerHome D:\models\a3s-power\qwen38\power-home-tbq4-0-ffn `
  -ModelHash 5f578b395f61dcaac9698fe222d988f461fd902ce9494e8a06d8b9aae4e7e2a6 `
  -NumBatch 14 -Samples 5 -MinimumTokensPerSecond 175 `
  -ProcessPriority High -RequireHighPerformancePowerPlan
```

## Original Q6_K hardware and controls

- NVIDIA GeForce RTX 4090, 25,757,220,864 reported VRAM bytes.
- Intel Xeon w5-2445 (10 physical cores / 20 logical processors), 128 GiB RAM.
- NVIDIA driver 610.74 on Windows x86_64; the host used the Windows balanced
  power plan, so no locked-clock result is claimed.
- Full CUDA layer offload, main GPU 0, Flash Attention enabled, ten CPU
  inference threads, one parallel slot, and memory locking disabled.
- One warm-up followed by five samples in each mode.
- `max_tokens=256`, `num_ctx=4096`, `num_batch=24`, seed 42,
  `temperature=0`, and `top_p=1`.
- Prompt SHA-256
  `d95a5e4dad822ba9c84138f7a120017318bcb3a6a90e77246a8ec4ede0e65d89`.
- Canonical request SHA-256
  `a32f870bbf052d383a7356f31c923cb9f3f557cb22c2de2369dbcf498b7646e7`.
- Both ACL files used `draft_max=7`, `mtp_recurrent_snapshots=7`,
  `draft_min=0`, and `draft_p_min=0.0`; only `spec_mode` changed from `off`
  to `mtp`.
- The acceptance threshold is the median server-reported steady-state decode
  rate. The final opted-in SSE usage event supplied the timing evidence.

Regenerate [final-comparison.json](final-comparison.json) from the two final raw
reports:

```console
a3s-power-speculative-bench compare final-baseline.json final-mtp.json
```
