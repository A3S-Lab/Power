# Storage Benchmark Protocol

`a3s-power-storage-bench` compares Power's verified SafeTensors materialization
strategies without constructing a model graph, tokenizer, inference backend,
subprocess, network client, or listener. It extends the same `WeightStore`
integrity, replica routing, cancellation, and fallback path used by embedded
inference; it is not a second storage engine.

The protocol adapts the measured positional-I/O ideas in
[A3S-Lab/colibri](https://github.com/A3S-Lab/colibri/tree/b085b48888a88d9a1c00b151a9979774b72cdbfd),
especially `st.h` and `iobench.c`, while adding Power's integrity, privacy,
cross-platform, and reproducibility requirements.

## Read Strategies

| Strategy | Behavior | Default |
| --- | --- | --- |
| `mmap` | Candle's validated SafeTensors mmap path | Yes |
| `positional-buffered` | Exact indexed ranges through bounded page-cache-backed positional reads; no collection-wide mmap is retained | No |
| `positional-direct` | Aligned `O_DIRECT` on Linux or `FILE_FLAG_NO_BUFFERING` on Windows | No |

Every strategy returns the same tensor dtype, shape, and bytes. Direct mode
never silently falls back to buffered I/O. macOS reports direct mode as
unsupported because `F_NOCACHE` is not equivalent evidence for this contract.

## Timing and Evidence

Each report records:

- exact Power version and commit;
- primary collection digest and path-free source descriptors;
- OS, architecture, named CPU, RAM, filesystem class, device class, transfer
  alignment, concurrency, strategy, and cache procedure;
- deterministic sequence digest, tensor count, requested bytes, read bytes,
  latency, throughput, and fallback count;
- integrity-open time separately from measured demand reads;
- two output-validation passes outside the measured interval, one before cache
  preparation and one after all samples;
- one canonical output digest used for cross-strategy byte parity.

Reports are written only to stdout. They contain no filesystem path or tensor
name. Power does not persist or export a report automatically.

## Warm Runs

A warm run performs one complete unmeasured sequence immediately before the
measured samples:

```bash
cargo run --release --locked \
  --no-default-features --features embedded-inference \
  --bin a3s-power-storage-bench -- \
  --primary /models/collection \
  --strategy mmap \
  --power-commit 0123456789abcdef0123456789abcdef01234567 \
  --filesystem-class ext4 \
  --device-class local-nvme \
  --cpu-model "Named CPU" \
  --ram-bytes 68719476736 \
  --cache-state warm \
  --cache-preparation warm-sequence \
  --concurrency 1 \
  --samples 30
```

Repeat with `--strategy positional-buffered`. Use
`--strategy positional-direct` only when the platform and source support it.
Add `--replica` or `--partial-replica` for a separately captured multi-source
run; all sources in one run must use the same strategy.

## Verified Cold Runs

Integrity hashing necessarily reads every configured source. A cache procedure
applied only before process start would therefore produce a false cold label.
On Linux, Power applies `POSIX_FADV_DONTNEED` after integrity-open and then uses
`mincore` to verify every page backing every requested tensor range across the
primary and all eligible replicas:

```bash
target/release/a3s-power-storage-bench \
  --primary /models/collection \
  --strategy positional-buffered \
  --power-commit 0123456789abcdef0123456789abcdef01234567 \
  --filesystem-class ext4 \
  --device-class local-nvme \
  --cpu-model "Named CPU" \
  --ram-bytes 68719476736 \
  --cache-state cold \
  --cache-preparation linux-fadvise-dontneed \
  --concurrency 1 \
  --samples 1
```

One process may emit only one cold sample. Capture each additional sample in a
new process, then aggregate the reports:

```bash
target/release/a3s-power-storage-bench compare \
  mmap-cold-01.json mmap-cold-02.json positional-cold-01.json
```

`FADV_DONTNEED` proves requested Linux page-cache pages are non-resident; it
does not prove that an SSD controller cache is cold. macOS and Windows currently
refuse the verified cold label because this implementation has no equally
strong post-integrity page-residency proof on those platforms. Direct I/O
bypasses the OS page cache but is not automatically labeled cold.

## Official PP-OCRv6 Warm Result

The following storage-only result used the exact PP-OCRv6 bundle installed and
executed by `a3s-ocr/tools/check_official_ppocr_v6.sh`. It is not an end-to-end
OCR latency claim.

| Field | Value |
| --- | --- |
| Power commit | `f27adeace747542d22ec73749fdec0073715b07c` |
| Host | Apple M2 Pro, 10 logical CPUs, 16 GiB RAM, macOS/aarch64 |
| Storage | Internal Apple Fabric SSD, APFS |
| Build and sampling | Release, concurrency 1, two processes × 30 warm samples per strategy |
| Detection collection | 169 tensors, 9,813,472 requested bytes, digest `0439824a102e0b365ca905355553985a885773ca0ea9f6a526e5f7317fc15592` |
| Recognition collection | 241 tensors, 21,071,132 requested bytes, digest `e8bf34a6900addc8cd9ec1d1ea73ea56e97cb0d668c8c45508a885924078761f` |

| Collection | Strategy | p50 latency | p95 latency | p50 throughput |
| --- | --- | ---: | ---: | ---: |
| Detection | mmap | 12.78 ms | 13.97 ms | 767.5 MB/s |
| Detection | positional buffered | 14.01 ms | 15.27 ms | 689.0 MB/s |
| Recognition | mmap | 26.99 ms | 30.93 ms | 779.6 MB/s |
| Recognition | positional buffered | 29.84 ms | 34.32 ms | 706.2 MB/s |

Detection output digest was
`f427bbbedfbedcabdc711d8597e59ce61d9b1645a8e71c5c8dad26a75caea7fe`
for both strategies. Recognition output digest was
`2416672993f3e99941a5a80242804a1c1e33a1610adaa39c5c43956dceaef2f8`
for both strategies.

Buffered positional reads regressed p50 latency by 9.6% for detection and 10.6%
for recognition on this host. Direct mode returned an explicit unsupported
error, and no cold result was recorded because macOS lacks the required proof.
These negative results keep mmap as the default. Direct I/O must remain opt-in
until a named-hardware end-to-end model workload demonstrates a repeatable win.
