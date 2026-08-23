---
title: Speculative decoding
description: How A3S Power separates advisory proposal generation from exact target verification, rollback, and quality gates.
---

# Model-neutral Speculative Decoding

Speculative decoding is a runtime capability, not a Qwen-specific branch.
Proposal generation is advisory; the target model remains authoritative for
every emitted token.

Power provides the common transaction, scheduling, rollback, and evidence
protocol. A backend adapter provides the architecture-specific draft graph,
state snapshots, logits access, and compatibility metadata.

## The exactness transaction

Each model-backed round follows one order:

1. Retain a checkpoint for target, draft, sampler, and decoder state.
2. Ask the adapter for at most the scheduled number of proposal tokens.
3. Evaluate the anchor and proposal block in one target pass.
4. Sample target rows only through the first mismatch.
5. Commit the accepted prefix and discard every rejected state row.
6. Emit exactly one correction or bonus token when generation continues.
7. Restore the last committed transaction on cancellation or failure.

For greedy decoding, speculative and autoregressive token IDs must match
exactly. For stochastic decoding, target sampler state advances once per
emitted target sample and never for an unobserved rejected suffix.

## Capabilities fail closed

| Backend/model capability | Available strategies |
| --- | --- |
| mistral.rs or proxy | `off`; `auto` resolves to `off` |
| picolm | `off`, `prompt-lookup`, `ngram-context` |
| llama.cpp without native prediction tensors | `off` |
| llama.cpp with `*.nextn_predict_layers > 0` | `off`, `mtp` |
| llama.cpp with a verified external DFlash GGUF | `off`, `dflash` |
| llama.cpp with a verified external DFlash2 GGUF | `off`, `dflash2` |
| llama.cpp with a verified external DSpark GGUF | `off`, `dspark` |

`draft-model` remains a reserved shared strategy without a production
llama.cpp adapter. DFlash v1, DFlash2, and DSpark use different
external-artifact contracts; they are not interchangeable and do not stack.
Power parses and hashes both GGUF files, binds the draft to the target digest,
validates artifact-specific metadata and tensors, then compares the complete
target/draft vocabularies when their contexts bind. An explicit unsupported or
mismatched mode fails closed.

DFlash2 has a typed strategy and a strict selector/convolution tensor
validator. Power ports the reviewed executor changes to its pinned llama.cpp
source; malformed metadata and DFlash v1/DFlash2 mismatches fail before native
execution.

`auto` selects a verified external artifact when the model manifest contains
one; otherwise it considers native MTP. Power does not load both draft
mechanisms because they would compete for device memory and mutable state.

## Draft width and rollback width are different

```text
spec_mode = "mtp"
spec_draft_max = 7
spec_mtp_recurrent_snapshots = 7

# Experimental compact draft projection. Omit for full vocabulary.
# spec_mtp_fr_vocab_size = 8192
```

`spec_draft_max` bounds proposal width. `spec_mtp_recurrent_snapshots` bounds
resident target rollback state. K7/S7 keeps a rollback point for every proposed
token. K7/S6 can be faster on high-acceptance prompts, but a rejected suffix may
exceed the resident window.

Power's guarded K7/S6 path permits one exact replay, then clamps all later
rounds in that request to six proposals. The guard bounds replay without
changing requests that remain on the high-acceptance path. K7/S7 avoids the
replay condition entirely and is the balanced default.

The opt-in request-local controller avoids using replay as its first signal.
It starts at `min(K, S)`, opens the wider K shape only after a fully accepted
first probe, and closes that path after a partial first round. Healthy partial
rounds retain their graph shape; sustained low-yield rounds open a one-way
target-only circuit. The same scheduler is shared by native MTP, DFlash, and
DSpark model-backed adapters. The legacy ACL key remains
`spec_mtp_adaptive`. Native DFlash2 currently uses its fixed external-draft
shape; an adaptive profile requires separate evidence.

## Q6_K acceptance and FR

The current acceptance target keeps every main-model weight in the original
Q6_K artifact. It does not use a mixed-quantization target.

FR reduces only the rows projected by the MTP draft head. That can raise a peak
on a narrow vocabulary distribution without rewriting Q6_K weights, but its
acceptance is language- and domain-sensitive. The pure-Q6_K peak deliberately
enables an 8,192-token-ID prefix; the balanced workload profile keeps the full
draft vocabulary.

## Current measured profiles

The untouched Q6_K peak combines:

- the original 22,884,408,288-byte Q6_K artifact;
- native MTP with an 8,192-row draft-only token-ID prefix;
- seven proposals and six recurrent snapshots;
- fixed B11 target-verification capacity and normal CUDA Graphs;
- batched target and draft greedy CUDA sampling;
- short-batch Flash Attention off and full CUDA layer offload;
- high-priority CUDA streams, physical-core affinity, and single-model,
  single-request scheduling;
- exact target verification and deterministic output digests.

The latest exact-build nine-run capture reached 174.413 token/s median steady
decode, 172.723 minimum, and 177.150 maximum. Four of nine samples reached 175,
so this is not a stable 175 token/s floor. The earlier quiet-host high-water
mark is 176.6109 token/s; the same-artifact full-vocabulary K7/S7 control
reached 147.0207 token/s.

The archived short-task calibration uses fixed K6/S6/B8. In its paired
12-task, 256-token calibration it reached 46.923 token/s versus 28.713 token/s
with speculation off, a 63.42% gain. Both modes retained all 12 final answers
and the 9/12 score. Acceptance was 26.81%, verified tokens per target pass were
2.591, and replay was zero.

Stable shapes mattered more than nominal acceptance. An earlier unrestricted
adaptive-K experiment raised acceptance on the same representative workload
to 50.07% but fell to 35.178 token/s because changing verification shapes
reduced CUDA Graph reuse. Disabling CUDA Graphs also reduced the peak workload
to 133.876 token/s. K, S, target batch, and graph shape must be tuned as one
system.

The native external-DSpark acceptance capture is a separate context-512,
batch-12 boundary. The untouched Q6_K target-only control reached 32.249
token/s median; DSpark Q4 K10/S6 reached 169.324 token/s median and 167.102
minimum, a 5.250x decode gain. All three 256-token outputs and execution
receipts matched the control exactly, proposal acceptance was 90.873%, and
fallback replay was zero. Peak VRAM was 23,847 MiB, so this profile requires a
quiet 24 GB device. A genuine DFlash v1 artifact has not yet been benchmarked;
a DSpark artifact mislabeled as DFlash is rejected and is not a DFlash result.

The native DFlash2 capture keeps the Q6_K target unchanged and uses a 1.14 GB
Q4 model only as an auxiliary proposer. Five paired 256-token samples reached
144.453 token/s median decode versus 33.075 target-only (4.367x), and 63.182
versus 25.744 token/s median end-to-end. Every output matched, acceptance was
98.230%, and replay was zero. The separate three-order 12-task calibration
retained 9/12 in both modes, 12/12 extracted answers, and only 7/12 complete
outputs; mean request-wide throughput increased from 29.702 to 45.143 token/s.
The native peak is not a 175 token/s floor, and the quality result keeps the
profile opt-in.

The context-1024 quality diagnostic tells a different, workload-representative
story. Across 600 successful requests, DSpark K10/S6 delivered 32.678 token/s
request-wide versus 22.618 for target-only (1.445x), with no observed score
decrease. Exact complete-output parity was only 54/100 and every DSpark request
entered fallback replay. The profile is therefore deterministic and useful for
experimentation, but it is not a lossless production default.

The current adaptive follow-up begins inside the S6 rollback window. Its clean
controlled peak reached 164.756 token/s median and 160.881 minimum across
three samples, with identical output and receipt hashes, 92.713% acceptance,
9.8077 verified tokens per target pass, and zero replay. The paired 100-task
run reached 31.052 token/s versus 22.872 target-only (1.358x), with zero replay
and 24 requests moved to target-only.

That adaptive run moved from 67/58 to 69/56 lenient/strict scores and contained
five lenient gains, three lenient losses, one strict gain, and three strict
losses. Exact complete-output parity was 55/100; all 57 tasks untruncated in
both modes retained the same extracted answer. The controller removes the
known replay tax.

A clean five-task follow-up then selected every observed answer loss and one
positive control. It retained 5/5 paired answers with zero losses across three
512-token repetitions, then retained 5/5 untruncated answer parity when the
budget was raised to 1,024 tokens. Complete output parity remained 0/5. The
controller therefore remains opt-in because target-authoritative acceptance
does not promise byte-identical serial and batched CUDA trajectories; the
focused answer result does not replace a representative quality matrix.

See the [Optimization playbook](./optimization) for the complete execution
path, [Performance evidence](./performance) for the measurement interpretation, and
the [canonical speculative-decoding design](https://github.com/A3S-Lab/Power/blob/main/docs/speculative-decoding.md)
for adapter APIs and acceptance rules, and the
[DSpark evidence package](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark)
for raw paired reports and exact replay commands, and the
[DFlash2 package](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dflash2)
for native paired reports, the representative quality boundary, and exact
reproduction.
