# Model-neutral Speculative Decoding

Speculative decoding is a Power runtime capability. It is not owned by Qwen,
DeepSeek, Llama, or any other model family. Qwen3.8-27B is the first dense
hybrid acceptance target for the CUDA path, not a special case in the shared
runtime.

The design follows the separation used by
[DSpark](https://arxiv.org/abs/2607.05147): proposal generation is advisory,
while the target model remains the source of truth for every emitted token.
Power exposes one control plane for zero-weight lookup, independent draft
models, native MTP heads, DFlash, DSpark, and future proposal algorithms.

## Ownership boundary

Power owns model-independent behavior:

- strategy parsing and fail-closed capability negotiation;
- bounded draft and verification scheduling;
- lossless target sampling and accepted-prefix accounting;
- commit and rejected-suffix rollback orchestration;
- cancellation, output limits, stop sequences, and streaming;
- acceptance, target-pass, latency, throughput, and memory evidence.

Each backend/model adapter owns architecture-dependent behavior:

- checkpoint discovery, compatibility fingerprints, and tensor layouts;
- draft graphs, native heads, and accelerator kernels;
- target block evaluation and logits access;
- transactions for every mutable model state;
- tokenizer and vocabulary compatibility between target and drafter.

A conventional transformer adapter normally transactions KV state. A hybrid
adapter may also need recurrent, convolution, or state-space snapshots. Power
does not infer those layouts from a model name.

## Capability negotiation

`SpeculativeStrategy` is the stable configuration vocabulary. A backend/model
pair advertises `SpeculativeCapabilities`, then resolves `auto` to a safe local
default. An explicit unsupported strategy returns an error; it never silently
falls back to a cheaper algorithm and never relabels n-gram lookup as DSpark.

Current executable adapters are:

| Backend/model capability | Strategies |
| --- | --- |
| mistral.rs or proxy | `off` (`auto` resolves to `off`) |
| picolm | `off`, `prompt-lookup`, `ngram-context` |
| llama.cpp without native prediction tensors | `off` |
| llama.cpp with `*.nextn_predict_layers > 0` | `off`, `mtp` |

`draft-model`, `dflash`, and `dspark` are part of the shared protocol but stay
unavailable until the loaded model supplies a compatible adapter artifact and
the backend implements its graph. This is intentional fail-closed behavior.

## Verification transaction

One model-backed round has the following order:

1. Retain a checkpoint for all target, draft, sampler, and decoder state.
2. Ask the adapter for at most the scheduled number of draft tokens.
3. Evaluate the anchor and draft block in one target pass.
4. Sample target rows only through the first mismatch. If every draft matches,
   sample one target bonus token.
5. Commit the accepted prefix and discard every rejected state row.
6. Stream accepted tokens through a terminal token; otherwise append exactly
   one correction or bonus token.
7. On cancellation or failure, restore the last committed transaction.

For greedy decoding, the speculative and non-speculative token IDs must match
exactly. For stochastic decoding, target sampler state advances once for each
emitted target sample and never for an unobserved rejected suffix.

## Adapter families

- Prompt lookup and online n-grams are zero-weight integration baselines.
- Native MTP consumes prediction tensors shipped with a target checkpoint.
- A separate draft model uses a tokenizer-compatible smaller checkpoint.
- DFlash provides a parallel block-diffusion draft backbone.
- DSpark adds prefix-dependent Markov and confidence heads to parallel drafting.

DSpark adapters can target Qwen, Llama, DeepSeek, Gemma, MoE, or future
architectures. Compatibility is determined by adapter metadata and state
contracts, not by branches in the Power scheduler.

## Delivery and acceptance

1. Shared strategy, capability, exact-verification, adaptive-length, and metric
   primitives are implemented and covered by deterministic tests.
2. picolm consumes the shared strategy vocabulary while preserving its former
   adaptive draft-length default.
3. llama.cpp provides native MTP execution with transactional target and draft
   rollback for compatible models.
4. Qwen3.8-27B Q6_K is the first CUDA performance gate. Baseline and MTP runs
   use the same model digest, prompts, sampling settings, context, and hardware.
5. A separate DSpark artifact is admitted only after its target/tokenizer
   compatibility, provenance, peak memory, exactness, and speedup are measured.
6. At least one non-Qwen adapter must pass the same transaction and exactness
   suite before DSpark support is considered cross-architecture complete.

The Qwen3.8 performance gate is at least 100 generated tokens per second on the
acceptance host through Power's streaming API. Native-tool measurements are
diagnostic evidence; they do not replace the Power end-to-end result.
