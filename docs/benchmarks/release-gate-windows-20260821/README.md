# Windows CPU and CUDA Release-Evidence Pre-Captures — 2026-08-21

These two reports are clean-revision inputs for the P5 release gate. They prove
the generic tensor boundary's scalar/batch parity and cost on this host. They do
not complete P5: peak-memory, cancellation, queue-expiry, replica-recovery,
explicit-fallback, Metal, and confidential-GPU captures are still required by
`ReleaseEvidencePolicy::strict_v1`.

The fixture is a reviewed broadcast Add graph over opaque F32 tensors. It does
not load Qwen, GGUF, a tokenizer, a decoder, or any language-model architecture.
Qwen workloads can consume the same contracts later, as can vision, embedding,
audio, scientific, and caller-defined graphs.

## Immutable inputs

| Item | Value |
| --- | --- |
| Power revision | `1a9504e58fc2751e016efede2fc006615a0b8cc2` |
| Power version | `0.9.0` |
| Host | Windows 11 x86_64 |
| CPU | Intel Xeon w5-2445, 20 logical CPUs |
| RAM | 137,071,693,824 bytes |
| CUDA device | NVIDIA GeForce RTX 4090 24 GiB, ordinal 0 |
| NVIDIA driver | 610.74 |
| CUDA toolkit | 12.6.68 |
| Fixture | 8 inputs, shape `[1, 4096]`, 2 warmups, 9 measured rounds |

Both executables were built from a clean detached worktree at the revision
above. The CPU and CUDA feature builds have different executable digests. The
temporary worktree was verified clean and removed after capture.

## Results

Times are medians across the nine raw measured rounds preserved in each JSON
file. Lower is better. Allocation bytes are requested host-allocator bytes, not
peak live memory.

| Device | Mode | Median elapsed | Host allocations | Allocated bytes | Input boundary | Output boundary | Exact parity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU | Individual | 119.0 µs | 233 | 277,272 | 2.8 µs | 37.1 µs | Yes |
| CPU | Leading batch | 328.5 µs | 50 | 526,731 | 0.4 µs | 83.9 µs | Yes |
| CUDA | Individual | 394.2 µs | 257 | 281,504 | 61.0 µs | 236.3 µs | Yes |
| CUDA | Leading batch | 191.8 µs | 54 | 527,308 | 17.9 µs | 109.2 µs | Yes |

The CPU leading batch is 2.76× slower for this tiny Add fixture even though it
uses far fewer allocation calls. That negative result is retained. On CUDA,
one leading batch reduces median elapsed time by 51.3% relative to eight
individual executions. Neither result is a token-per-second measurement or a
claim about an end-to-end model.

## Artifact identities

| Capture | Runtime executable SHA-256 | Canonical report SHA-256 | JSON file SHA-256 |
| --- | --- | --- | --- |
| CPU | `7f58a95f37b533afa484ce93d3878fbf844c82ccd0890f964da713ef4ec6fae8` | `31ef05556ddda4903582e5429ac4488fc7cf2198db1fa2c63b8566faeee448f8` | `8b60a5ab69754a4d9bd869104540b0e1a789b3fac2c2055493fb95b07101d6e5` |
| CUDA | `e1903fd14e9d79942cc042811929ad787afe2903a3643c9cc4604eb70d8a7eb1` | `a2ada015aba85be45796c9f5198f5a2c03f8be43df90532bf50a03f69b10ee7a` | `0f4af7a2c64b414dbd56720998a84422b30f17cf3adc21cfdc62eaec40f26027` |

The canonical report digest covers the path-free semantic report. The file
digest also covers JSON whitespace. Authenticity requires a signed release or
another caller-owned trust root; a digest alone identifies bytes but not who
produced them.

## Reproduce

Use a clean checkout or detached worktree. Do not capture from a source tree
with modified or untracked files.

```powershell
$revision = "1a9504e58fc2751e016efede2fc006615a0b8cc2"
$captureRoot = "D:\code\Power-release-capture-1a9504e"
git worktree add --detach $captureRoot $revision
Set-Location $captureRoot
if (git status --porcelain) { throw "capture worktree is not clean" }
```

Run the CPU fixture:

```powershell
cargo run --locked --release --no-default-features `
  --features embedded-inference `
  --bin a3s-power-tensor-batch-bench -- fixture `
  --device cpu `
  --power-commit $revision `
  --filesystem-class ntfs `
  --device-class "Intel Xeon w5-2445 CPU" `
  --cpu-model "Intel(R) Xeon(R) w5-2445" `
  --ram-bytes 137071693824 `
  --items 8 `
  --width 4096 `
  --warmup-rounds 2 `
  --measured-rounds 9 > cpu.json
```

Run the CUDA fixture from an x64 Visual Studio developer shell with CUDA 12.6
available:

```powershell
cargo run --locked --release --no-default-features `
  --features embedded-cuda `
  --bin a3s-power-tensor-batch-bench -- fixture `
  --device cuda:0 `
  --power-commit $revision `
  --filesystem-class ntfs `
  --device-class "NVIDIA GeForce RTX 4090 24 GiB; driver 610.74" `
  --cpu-model "Intel(R) Xeon(R) w5-2445" `
  --ram-bytes 137071693824 `
  --items 8 `
  --width 4096 `
  --warmup-rounds 2 `
  --measured-rounds 9 > cuda.json
```

Replay semantic verification through the crate tests:

```powershell
cargo test --locked --no-default-features `
  --features embedded-inference `
  published_clean_revision_reports_replay_successfully --lib
```

Finally compare the report and file digests, then remove only the explicit
temporary worktree after confirming it is clean:

```powershell
Get-FileHash cpu.json -Algorithm SHA256
Get-FileHash cuda.json -Algorithm SHA256
if (git status --porcelain) { throw "capture worktree changed" }
Set-Location D:\code\a3s\crates\power
git worktree remove $captureRoot
```

For the full protocol and caller-owned graph mode, see
[`../../tensor-batch-benchmark.md`](../../tensor-batch-benchmark.md).
