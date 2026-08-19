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
| Untouched Q6_K, autoregressive | 66/100 lenient; 59/100 strict (100 tasks, 3x) | 34.551 token/s | 35.5793 token/s | Same-artifact quality and speed baseline |
| Untouched Q6_K, native MTP | Quality matrix not run; exact parity on the fixed peak prompt | -- | 140.1600 token/s | Same-artifact MTP ceiling, not a broad quality result |
| TBQ4 mixed artifact, autoregressive | 72/100 lenient; 64/100 strict (100 tasks, 3x) | **41.745 token/s** | -- | Representative-workload winner in the 100-task matrix |
| TBQ4 mixed + MTP + prefix FR, K7/S6 | 72/100 lenient; 60/100 strict (100 tasks, 3x) | 27.951 token/s | 184.3665 token/s | High repetitive-prompt peak, but low 25.55% mixed-workload acceptance |
| TBQ4 mixed + full-vocabulary fixed MTP, K7/S6 | 4/12 lenient; 3/12 strict (12 tasks, 3x) | 28.226 token/s | -- | Rejected default: 46 exact prefix replays per run |
| TBQ4 mixed + full-vocabulary adaptive MTP, K7/S6 | 5/12 lenient; 3/12 strict (12 tasks, 3x) | 60.031 token/s | -- | Rollback-safe, but slower than fixed K7/S7 on the calibration |
| TBQ4 mixed + full-vocabulary fixed MTP, K7/S7 | 5/12 lenient; 3/12 strict (12 tasks, 3x) | **68.211 token/s** | -- | Rollback-safe calibration winner; the 100-task matrix is still pending |
| TBQ4 mixed + full-vocabulary fixed MTP, host-staged K7/S6 with physical-core affinity | Quality matrix not run | -- | **177.3062 token/s** | Current controlled 9x peak gate; 175.5958 token/s minimum |
| UD-Q8_K_XL, autoregressive heterogeneous placement | Quality matrix not run | -- | 6.3484 token/s | Fits through exact CPU/GPU tensor placement, but is bandwidth-bound |
| UD-Q8_K_XL, native MTP K4/S4 heterogeneous placement | Quality matrix not run; cross-mode output hashes differ | -- | 9.7577 token/s | Performance boundary only; not a parity or quality acceptance result |

The complete protocols and raw evidence are in the
[100-task and 12-task quality report](quality/README.md), the sections below,
and the sibling [UD-Q8_K_XL boundary capture](../qwen3.8-27b-ud-q8-k-xl-rtx4090/README.md).

## Representative workload, repeated three times

The peak gate below is intentionally complemented by a fixed 100-task quality
and workload-throughput matrix. It uses 50 MMLU, 20 GSM8K, and 30 C-Eval
tasks, runs each mode three times in cyclic order, and publishes the task
cache, environment, nine per-request reports, and paired statistics.

| Mode | Lenient score | Strict score | Mean workload throughput |
| --- | ---: | ---: | ---: |
| Untouched Q6_K, speculation off | 66/100 | 59/100 | 34.551 token/s |
| TBQ4 mixed artifact, speculation off | 72/100 | 64/100 | **41.745 token/s** |
| TBQ4 + MTP + FR | 72/100 | 60/100 | 27.951 token/s |

TBQ4-off was 20.8% faster than untouched Q6_K on this workload. MTP + FR was
33.0% slower than TBQ4-off because its workload-wide draft acceptance fell to
25.55%; C-Eval acceptance was only 14.21%. That historical prefix-FR result is
a repetitive-prompt peak, not a representative mixed-workload rate.
See the [quality and workload matrix](quality/README.md) for the full protocol,
limitations, reproducibility command, and machine-readable evidence.

A post-change calibration then replayed a fixed 12-task subset three times with
the full 248,320-row draft vocabulary. It is intentionally smaller than the
100-task release matrix, but it isolates the rollback-window decision on the
current binary:

| Full-vocabulary mode | Mean workload throughput | Draft acceptance | Fallback replays per run | Lenient score |
| --- | ---: | ---: | ---: | ---: |
| TBQ4, speculation off | 35.048 token/s | -- | -- | 5/12 |
| Fixed K7, six snapshots | 28.226 token/s | 48.54% | 46 | 4/12 |
| Adaptive K7, six snapshots | 60.031 token/s | 65.50% | 0 | 5/12 |
| Fixed K7, seven snapshots | **68.211 token/s** | 49.67% | 0 | 5/12 |

The complete K7 rollback window made fixed full-vocabulary MTP 94.6% faster
than speculation-off on this calibration and 13.6% faster than adaptive S6.
Six snapshots remain useful for the high-acceptance peak prompt, but fixed
K7/S6 is not a safe mixed-workload default because a seven-token rejection
forces exact prefix replay. See the
[calibration evidence](quality/full-vocab-rollback-calibration-rtx4090-3x.json)
and its reproduction command in the quality README.

## Current full-vocabulary batched-greedy boundary

The current implementation removes FR from the performance gate and retains
all 248,320 draft-head rows. The current captured binary passes a topology-pinned
nine-sample gate, and two earlier quiet-WDDM builds provide independent
high-water captures. Predecessor captures are retained to show the range
introduced by a shared WDDM display GPU:

| Capture | Median decode | Minimum decode | Median end-to-end | Gate |
| --- | ---: | ---: | ---: | --- |
| [Current physical-core-affinity gate, 1,024 tokens, 9x](final-affinity-1024-9x.json) | **177.3062 token/s** | **175.5958 token/s** | 168.9809 token/s | 175 passed, 9 / 9 |
| [Current 1,024-token control, 9x](final-draft-greedy-pmin-guard-1024-9x.json) | **187.6094 token/s** | **183.7360 token/s** | 178.6574 token/s | 175 passed, 9 / 9 |
| [Current 1,024-token repeat, 9x](final-draft-greedy-pmin-guard-1024-9x-repeat.json) | **188.2972 token/s** | **184.4913 token/s** | 179.2680 token/s | 175 passed, 9 / 9 |
| [1,024 tokens, unlocked, 5x](batched-greedy-1024-5x.json) | **182.1530 token/s** | **180.7195 token/s** | 173.4201 token/s | 175 passed |
| [256 tokens, 2,745 MHz control, 9x](batched-greedy-256-locked2745-9x.json) | **175.9190 token/s** | 169.3528 token/s | 152.0363 token/s | 175 median passed |
| [256 tokens, contended desktop, 9x](batched-greedy-256-contended-9x.json) | 159.8593 token/s | 152.4855 token/s | 139.8559 token/s | 175 failed |
| [Post-removal 256-token regression, 3x](post-device-carry-removal-256-3x.json) | 167.3977 token/s | 165.0082 token/s | 140.8385 token/s | Output regression passed |

The current captured binary uses the host-staged CPU-embedding ACL and pins its ten
worker threads to one logical processor per physical Xeon W5-2445 core with
mask `0x55555`. An order-balanced A/B measured a 176.8276 token/s combined
median with that mask versus 173.0114 token/s across all 20 logical
processors, a 2.21% gain. The independent nine-sample confirmation then kept
every sample above 175 token/s. The complete A/B and final environment are
recorded in [the compact comparison](affinity-ab-1024-12x.json), the
[final report](final-affinity-1024-9x.json), and its
[environment capture](final-affinity-1024-9x.environment.json). The mask is
specific to this CPU topology and remains an explicit runner option rather
than a product default.

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

The current [host-staged ACL](mtp7-snap6-full-vocab-cpu-embedding.acl) uses MTP
width 7, six recurrent snapshots, an explicit disabled recurrent device chain,
Flash Attention, ten host threads, and no `spec_mtp_fr_vocab_size` override.
Short-window tuning used batch 20; the long capture used batch 14. The
historical [device-chain ACL](mtp7-snap6-full-vocab.acl) keeps the token
embedding on the GPU for the optional recurrent experiment. The rollback-safe
[mixed-workload ACL](mtp7-snap7-full-vocab.acl) retains seven snapshots.

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
before promoting 175 token/s to a reproducible floor. The 12-task calibration
above confirms the full-vocabulary direction, but the separate 100-task matrix
must still be repeated against this binary before making a release-scale
quality claim.

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

## Original same-artifact Q6_K result

| Metric | Explicit off | Native MTP |
| --- | ---: | ---: |
| Median steady-state decode | 35.5793 token/s | 140.1600 token/s |
| Minimum measured decode | 35.4812 token/s | 139.4793 token/s |
| Median speedup | 1.0000x | 3.9394x |
| Samples after warm-up | 5 | 5 |
| Completion tokens per sample | 256 | 256 |

The final MTP median is 40.1600% above the 100 token/s acceptance floor, and
every individual MTP sample is at least 39.4793% above it. All ten measured
outputs have SHA-256
`584e2b93ba21d7c727456567762c6bbacc150d43156c73ed91c1c0cbb13be6eb`,
so greedy output parity passes both within and across modes. The verified
comparison is in [final-comparison.json](final-comparison.json); the full raw
reports are [final-baseline.json](final-baseline.json) and
[final-mtp.json](final-mtp.json).

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
- Six recurrent snapshots can match seven on the 93.62%-acceptance peak prompt,
  but the three-run mixed-workload calibration recorded exactly 46 fallback
  replays in every fixed K7/S6 run and only 28.226 token/s. Fixed K7/S7 removed
  every replay and reached 68.211 token/s; adaptive K7/S6 also avoided replay
  and reached 60.031 token/s. K7/S6 is therefore peak-only, while K7/S7 is the
  fixed mixed-workload default. Four or fewer snapshots cross the replay
  boundary even more often. Raising `draft_p_min` above 0.7 also increased
  target-pass count and slowed decoding.
- Disabling Flash Attention overlapped the enabled result. Temporary
  instrumentation observed 531 llama.cpp output-reorder calls in a 64-token
  request and zero pending row swaps in every call, ruling out CPU vocabulary
  row reordering as the apparent hot spot. Phase timings instead place the
  remaining boundary in asynchronous CUDA draft/target execution and result
  synchronization.
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
