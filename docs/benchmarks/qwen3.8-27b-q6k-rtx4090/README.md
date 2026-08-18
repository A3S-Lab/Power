# Qwen3.8-27B Q6_K RTX 4090 acceptance capture

This directory records Power's Qwen3.8 CUDA performance gates and subsequent
boundary tuning. Every accepted result was measured through Power's streaming
`POST /v1/completions` API. No native llama.cpp timing is used as an acceptance
result.

## 175 token/s development gate

Four five-sample captures of the optimized path passed the declared 175
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
22.88 GB Q6_K file. The reduced draft vocabulary cannot change the committed
greedy sequence because the full target verifies proposals, but the selective
weight requantization has its own quality trade-off and needs a representative
accuracy evaluation before release.

The llama.cpp changes are captured in
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
- Five or six recurrent snapshots could match seven in isolated samples, but
  did not improve the five-sample median. Four or fewer crossed the exact
  replay boundary for this output and reduced throughput sharply. Raising
  `draft_p_min` above 0.7 also increased target-pass count and slowed decoding.
- Disabling Flash Attention overlapped the enabled result. Temporary
  instrumentation observed 531 llama.cpp output-reorder calls in a 64-token
  request and zero pending row swaps in every call, ruling out CPU vocabulary
  row reordering as the apparent hot spot. Phase timings instead place the
  remaining boundary in asynchronous CUDA draft/target execution and result
  synchronization.
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
| Pre-validation optimized server SHA-256 | `97cd56cbfc5cca5f3fc3d1e969cd4af804b47b9c00b60786b36879e39014229d` |
| Pre-validation optimized benchmark SHA-256 | `faf0bbc824c6b27e62eea3909d9a84480e987caf5e72e7b6efe278c2b9338e75` |
| Final optimized server SHA-256 | `3ec225e412f27eda4677288332988977201c038dfb09a76eed7c0593e9db7eea` |
| Final optimized benchmark SHA-256 | `0e752978ca9521319a50b8eb78232f342680d5d7ae1f4bae521541697ea45075` |
| Power source base | `491184ada54699ddfc4b40246cd6aee92d7550dd` |
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

The 140.1600 capture predates the validation-only recurrent-batch safety fix;
the post-safety reports pin the rebuilt implementation from that stage. The
original and optimized executables were built from a dirty working tree based
on the recorded Power commit. Therefore the executable digests above, rather
than the base commit alone, are the exact binary identities. Release evidence
should repeat the capture from the eventual clean commit.

### Optimized capture controls

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
final gate can be repeated from the crate root with:

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
