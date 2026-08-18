# Qwen3.8-27B UD-Q8_K_XL RTX 4090 boundary capture

This capture answers whether the exact 31.46 GB UD-Q8_K_XL artifact can run on
a 24 GB RTX 4090 and records the observed long-generation boundary. It measures
Power's real streaming `POST /v1/completions` path, not a native llama.cpp
microbenchmark. `UD-Q8_K_XL` is the conversion policy name; the reviewed GGUF
contains Q8_0, BF16, and F32 tensors rather than one monolithic Q8_K tensor type.

## Result

| Metric | Explicit off | Native MTP |
| --- | ---: | ---: |
| Median steady-state decode | 6.3484 token/s | 9.7577 token/s |
| Minimum measured decode | 6.2861 token/s | 9.3088 token/s |
| Maximum measured decode | 6.3629 token/s | 9.9561 token/s |
| Median end-to-end rate | 5.8518 token/s | 8.5738 token/s |
| Median time to first token | 3.4509 s | 3.7038 s |
| Samples after warm-up | 3 | 3 |

MTP improved median decode throughput by 1.5370x (53.70%). The defensible
long-generation result for this machine is therefore approximately 9.8
token/s, with an observed peak of approximately 10.0 token/s. A 72-token
search run reached 11.95 token/s, but that number did not survive the
256-token workload and is not reported as the sustained boundary.

The two modes were deterministic within themselves, but their greedy output
hashes differed: explicit-off produced
`e60d9e6f255af911ef21c4db43321dea00473ffdf6e10729668b7f477fdc2d0e`,
while MTP produced
`f059369f44d22c0d81350cd1ded4a17c138f8947c53c5d4fe886ea9f5d1e77a2`.
Consequently this is a performance-boundary capture, not a cross-mode parity
acceptance result. The batched MTP verification path must not be represented as
bit-identical to autoregressive greedy decoding for this artifact.

The machine-readable samples and identities are in [results.json](results.json).

## Why the model fits

The GGUF is larger than physical VRAM, so full GPU residency is impossible.
Power used exact tensor placement instead of unified-memory oversubscription:

- Keep the output projection, MTP prediction layer, recurrent core, and all
  other latency-sensitive weights on the GPU.
- Place `attn_q`, `ffn_down`, `ffn_gate`, and `ffn_up` on the CPU for the 16
  full-attention blocks `3, 7, ..., 63`.
- Keep the model resident in 22,446.92 MiB of CUDA model buffers and 7,543.28
  MiB of CUDA-host model buffers. The target context adds a 748.12 MiB
  recurrent-state buffer and a 32 MiB attention KV buffer.
- Disable mmap for the dedicated loaded CPU buffers and do not enable CUDA
  unified memory.

This is heterogeneous tensor flow through each layer: CPU-resident matrix
operations exchange activations with GPU-resident operations. It is not
disk-backed whole-model layer swapping. Streaming all 31.46 GB of weights over
PCIe for every token would have a bandwidth ceiling below one token/s before
compute, so that interpretation of layer streaming cannot reach this result.

## MTP boundary

The tuned configuration used `draft_max=4` and four resident recurrent
rollback snapshots. Each measured request used 55 target passes, drafted 220
tokens, accepted 200 (90.91%), emitted 4.6364 tokens per target pass, and
required no prefix replay. The longest rejected suffix was four tokens.

One recurrent state plane costs approximately 149.6 MiB. Wider drafts improve
target-pass amortization but require more rollback planes, eventually forcing
additional weights onto the CPU or creating a memory-pressure cliff. The
measured search illustrates that trade-off:

| Draft / snapshots | Additional CPU FFN blocks | Decode result | Samples |
| ---: | ---: | ---: | ---: |
| 3 / 3 | 0 | 6.8627 token/s median | 3 |
| 4 / 4 | 0 | 9.7577 token/s median | 3 |
| 5 / 5 | 0 | 9.6554 token/s | 1 |
| 6 / 6 | 1 | 8.9599 token/s | 1 |
| 7 / 7 | 1 | 8.0011 token/s | 1 |
| 8 / 8 | 3 | 6.6152 token/s | 1 |

A memory-saving width-8 configuration with only two rollback snapshots looked
fast on short output, but six long-suffix rejections caused full-prefix replay;
its 256-token median fell to 1.9542 token/s. Host-serialized recurrent
checkpoints reduced that cost but changed the greedy trajectory, so the
experiment was rejected and the exact replay fallback was retained.

## Artifact, build, and controls

| Item | Value |
| --- | --- |
| GGUF source | `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf` |
| GGUF byte length | `31457991680` |
| GGUF SHA-256 | `af36ecb6b5db1407953345b746c14ac93f0657dda413910b4348683a2d990377` |
| Power source base | `491184ada54699ddfc4b40246cd6aee92d7550dd` |
| Server executable SHA-256 | `8a10bac67b783da4403e9c6651e165ee77fdac08a1ed04576c5f37479facc680` |
| Benchmark executable SHA-256 | `885e077d5f8d56447109ea232337443c48ddcb97945540c58d4540198982f9ae` |
| llama-cpp-rs revision | `dfd12e4d334846367e4284a2a7763fe92c1bf676` |
| Toolchain | Rust 1.97.1, CUDA 12.6, SM 89 native build |
| Capture date | 2026-08-18 Asia/Shanghai |

Hardware was an RTX 4090 with 25,757,220,864 reported VRAM bytes, driver
610.74, a 10-core/20-thread Intel Xeon w5-2445, and 137,071,693,824 bytes of
system RAM. The run used eight CPU inference threads, Flash Attention, one
parallel slot, `num_ctx=512`, `num_batch=24`, 256 generated tokens, seed 42,
temperature zero, and `top_p=1`. The native build enabled CUDA SM 89 and the
available AVX-VNNI, AVX-512 BF16, VBMI, and VNNI CPU paths.

The capture executables were built from a dirty working tree based on the
recorded Power commit. Their executable digests, not the base commit alone,
identify the exact tested build. Release evidence should repeat the capture
from the eventual clean commit.
