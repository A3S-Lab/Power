# Generic Tensor Batch Cost Evidence — Windows, 2026-08-20

This directory contains named-hardware evidence for Power's model-neutral
individual-versus-leading-batch execution boundary. The workload is the built-in
generic broadcast Add fixture, not a language model, vision model, tokenizer,
decoder, or token-throughput benchmark.

Both reports were captured from clean commit
`8537d7aad7b82d943c1698d985976a4e3dd40153`. Each run used two warmup rounds,
nine measured rounds with alternating order, eight inputs of shape `[1, 4096]`,
and exact ordered F32 output parity.

## Named environment

| Component | Captured value |
| --- | --- |
| OS / architecture | Windows x86_64 |
| CPU | Intel(R) Xeon(R) w5-2445, 20 logical CPUs |
| RAM | 137,071,693,824 bytes |
| Filesystem class | NTFS |
| CUDA device | NVIDIA GeForce RTX 4090, 24,564 MiB reported memory |
| NVIDIA driver | 610.74 |
| CUDA compiler | 12.6.68 |
| MSVC | 19.44.35228, x64 |
| Rust / Cargo | 1.97.1, `x86_64-pc-windows-msvc` |

The CPU report came from a release binary built with only
`embedded-inference`. The CUDA report came from a separate release binary built
with `embedded-cuda`. The reports bind the exact executable hashes, so these two
feature sets cannot be treated as one runtime artifact.

## Raw median results

| Device / mode | Median elapsed | Host allocations | Host allocated bytes | H2D / D2H operations | Input / output materialization |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU, individual | 94.6 µs | 217 | 274,288 | 0 / 0 | 2.6 / 28.2 µs |
| CPU, leading batch | 278.3 µs | 48 | 526,358 | 0 / 0 | 0.5 / 34.1 µs |
| RTX 4090, individual | 395.2 µs | 241 | 278,520 | 8 / 8 | 57.4 / 245.5 µs |
| RTX 4090, leading batch | 182.6 µs | 52 | 526,935 | 1 / 1 | 17.4 / 104.7 µs |

All 36 measured executions produced the same ordered output digest:
`5d8fa7101c46a39138cd4e3bb9b7591365d2c018c9c8080da8819a7a77727840`.

The CPU result is intentionally retained as negative evidence. Leading batching
reduced allocation operations by 77.88% but increased requested host allocation
bytes by 91.90% and made this tiny CPU graph 194.19% slower. On the RTX 4090,
leading batching reduced median elapsed time by 53.80%, allocation operations by
78.42%, and output materialization time by 57.35%, while requested host
allocation bytes still increased by 89.19%.

This proves the measured boundary can expose the relevant tradeoffs. It does
not prove that a real model will receive the same gain. Batch assembly,
partitioning, kernel shape, graph depth, memory pressure, and model-owned control
flow determine the end-to-end result.

## Artifact pins

| Artifact | Runtime/report SHA-256 | File SHA-256 |
| --- | --- | --- |
| [`cpu.json`](cpu.json) | Runtime `6f2713c744323ce203e0b5652d673be6f2a3d42a6487955b8c957c4d1683bea9`; report `bf1907ee8d4475e05aede9d776e98a10c6da54eba96fbf3a81b72f44b6a84bf6` | `47d3aa37b9590e49c45d90dae44a2c8dd094542735868445aee5a68316c6912d` |
| [`cuda.json`](cuda.json) | Runtime `85e546a396ebc5fd2e8ee835d97adfa5639b7d91d269abaac889e1e445b35006`; report `32b50c4e132d6e57a4611061d48fe5cdb67a9cbd27a2ee0192cc6102bc84fa3f` | `0c77cc257cd1d96450061fa959d7b19be358c89c78386f04f3adb864bfe41c24` |

The report SHA-256 is the canonical digest verified by
`TensorBatchBenchmarkReport::verify`. The file SHA-256 covers the compact JSON
encoding committed here. Unit tests deserialize and verify both reports. A
signed release or attestation must still authenticate the trusted digest; a
hash alone does not establish authorship.

## Reproduction

Follow the complete [Tensor Batch Cost Benchmark
Protocol](../../tensor-batch-benchmark.md). The exact build commands were:

```powershell
cargo build --release --no-default-features `
  --features embedded-inference `
  --bin a3s-power-tensor-batch-bench

# Run the protocol's CPU fixture command before replacing the binary.

# Enter the Visual Studio x64 developer environment before CUDA compilation.
cargo build --release --no-default-features `
  --features embedded-cuda `
  --bin a3s-power-tensor-batch-bench

# Run the protocol's cuda:0 fixture command.
```

The runner automatically hashes its currently executing image before warmup.
Do not rebuild or replace the binary between its hash and measurement. Keep the
machine otherwise idle; the allocation counter is process-wide, and GPU clocks,
power state, or unrelated host load were not locked in this capture.
