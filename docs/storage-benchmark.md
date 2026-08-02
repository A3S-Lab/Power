# Storage Benchmark Protocol

`a3s-power-storage-bench` compares Power's verified SafeTensors materialization
strategies without constructing a model graph, tokenizer, inference backend,
subprocess, network client, or listener. It extends the same `WeightStore`
integrity, replica routing, cancellation, and fallback path used by embedded
inference; it is not a second storage engine.

The same runner also measures optional lossless representations. Canonical
SafeTensors remains the primary; each compressed replica carries an explicit
physical artifact pin and must pass full decoded-byte admission before timing.

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

Every strategy and representation returns the same canonical tensor dtype,
shape, and bytes. Direct mode never silently falls back to buffered I/O. macOS
reports direct mode as unsupported because `F_NOCACHE` is not equivalent
evidence for this contract.

## Timing and Evidence

Each report records:

- exact Power version and commit;
- primary collection digest and path-free source descriptors, including typed
  canonical or lossless representation identity and the compressed artifact
  digest where applicable;
- OS, architecture, named CPU, RAM, filesystem class, device class, transfer
  alignment, concurrency, strategy, and cache procedure;
- deterministic sequence digest, tensor count, requested bytes, read bytes,
  latency, throughput, and fallback count;
- integrity-open time separately from measured demand reads;
- two output-validation passes outside the measured interval, one before cache
  preparation and one after all samples;
- one canonical output digest used for cross-strategy byte parity.

`verifiedBytes` in each source descriptor is the physical admitted collection
size. Sample `bytesRead`, requested-byte totals, and throughput use decoded
canonical bytes, so a lossless run measures effective model-byte delivery plus
decode cost. Representation identity participates in the stable source-profile
digest; reports from different compressed artifacts are never merged silently.

Reports are written only to stdout. They contain no filesystem path or tensor
name. Power does not persist or export a report automatically.

## Warm Runs

A warm run performs one complete unmeasured sequence immediately before the
measured samples:

```bash
cargo run --release \
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

For an already minted lossless collection, pass the independently reviewed
physical collection pin with the source. The `::` separator is part of the CLI
syntax:

```bash
target/release/a3s-power-storage-bench \
  --primary /models/canonical \
  --lossless-replica /models/compressed::0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --strategy mmap \
  --power-commit 0123456789abcdef0123456789abcdef01234567 \
  --filesystem-class ext4 \
  --device-class local-nvme \
  --cpu-model "Named CPU" \
  --ram-bytes 68719476736 \
  --cache-state warm \
  --cache-preparation warm-sequence \
  --samples 30
```

Use `--partial-lossless-replica` for a proper non-empty tensor subset. The
runner does not mint a digest from the same untrusted artifact on the caller's
behalf; an explicit lowercase SHA-256 pin is mandatory.

## Verified Cold Runs

Integrity hashing necessarily reads every configured source. A cache procedure
applied only before process start would therefore produce a false cold label.
On Linux, Power synchronizes each involved file, applies
`POSIX_FADV_DONTNEED` after integrity-open, and then uses `mincore` to verify
every page backing every requested tensor range across the primary and all
eligible replicas:

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

## Hosted-Runner Evidence Workflow

The manual [`Storage Evidence`](../.github/workflows/storage-evidence.yml)
workflow downloads the public PP-OCRv6 SafeTensors bundle only after checking
its pinned byte length and SHA-256. It benchmarks the detection and recognition
collections independently on `ubuntu-24.04` and `windows-latest`:

- Linux records warm mmap, positional buffered, and positional direct reads,
  then attempts separately verified-cold processes for each strategy;
- Windows records warm mmap, positional buffered, and unbuffered direct reads,
  while continuing to refuse an unverified cold label; and
- each job compares every admitted report for one collection and fails if
  output bytes differ across read strategies.

Linux publishes a path-free capability record for each cold strategy. If the
host retains any requested page after `FADV_DONTNEED`, the record contains the
stable `page-cache-pages-remained-resident` limitation and no rejected run is
emitted as a cold benchmark. Unexpected cold failures still fail the workflow.

The uploaded artifacts contain only the path-free benchmark reports, canonical
comparisons, and Linux cold-capability records. A report captures the actual
CPU, RAM, filesystem, GitHub runner image, Power commit, and workload digest.
GitHub-hosted hardware and storage may change between runs, so results must be
reviewed per report and must not be generalized into a stable hardware claim.
The workflow has only one ephemeral OS disk and therefore does not claim an
independent-controller multi-source result. Neither its workload nor its
reports are embedded in the Power crate or exported automatically.

## Reviewed Hosted-Runner Result

[Workflow run 30764609062](https://github.com/A3S-Lab/Power/actions/runs/30764609062)
completed against Power commit
`aae85338781746ddee2f06094ca5bc3e512e93e6`. Each warm group contains ten
samples. Each admitted Linux cold group contains three separately verified
one-sample processes.

| OS | Host and storage evidence |
| --- | --- |
| Linux | AMD EPYC 7763, 4 logical CPUs, 16,766,423,040 bytes RAM, ext4, `ubuntu24-20260720.247.2` ephemeral OS disk |
| Windows | AMD EPYC 9V74, 4 logical CPUs, 17,174,360,064 bytes RAM, NTFS, `win25-vs2026-20260714.173.1` ephemeral OS disk |

Warm storage-only results:

| OS | Collection | Strategy | p50 latency | p95 latency | p50 throughput |
| --- | --- | --- | ---: | ---: | ---: |
| Linux | Detection | mmap | 7.07 ms | 8.53 ms | 1,357.4 MB/s |
| Linux | Detection | positional buffered | 7.95 ms | 9.02 ms | 1,234.0 MB/s |
| Linux | Detection | positional direct | 273.43 ms | 275.23 ms | 35.9 MB/s |
| Linux | Recognition | mmap | 15.66 ms | 16.74 ms | 1,328.9 MB/s |
| Linux | Recognition | positional buffered | 16.89 ms | 17.72 ms | 1,231.7 MB/s |
| Linux | Recognition | positional direct | 397.65 ms | 408.72 ms | 52.9 MB/s |
| Windows | Detection | mmap | 6.90 ms | 7.92 ms | 1,410.5 MB/s |
| Windows | Detection | positional buffered | 8.55 ms | 10.99 ms | 1,117.5 MB/s |
| Windows | Detection | positional direct | 154.40 ms | 182.58 ms | 63.3 MB/s |
| Windows | Recognition | mmap | 16.24 ms | 21.58 ms | 1,293.5 MB/s |
| Windows | Recognition | positional buffered | 18.42 ms | 18.65 ms | 1,143.3 MB/s |
| Windows | Recognition | positional direct | 230.71 ms | 241.13 ms | 90.8 MB/s |

Verified Linux cold storage-only results:

| Collection | Strategy | Verified processes | p50 latency | p95 latency | p50 throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| Detection | positional buffered | 3/3 | 12.76 ms | 13.12 ms | 768.8 MB/s |
| Detection | positional direct | 3/3 | 271.85 ms | 274.12 ms | 36.1 MB/s |
| Recognition | positional buffered | 3/3 | 26.68 ms | 27.52 ms | 789.8 MB/s |
| Recognition | positional direct | 3/3 | 403.74 ms | 410.60 ms | 52.2 MB/s |

Linux mmap cold preparation was rejected for both collections: zero of three
requested processes were admitted because mapped pages remained resident after
`FADV_DONTNEED`. No mmap result was mislabeled cold. All admitted reports had
exact output-byte parity. Direct reports used the direct strategy with zero
primary fallbacks; they did not silently become buffered reads.

On these ephemeral hosts, warm positional buffered p50 latency was 7.8% to
23.9% slower than mmap. Direct p50 latency was 14.2 to 38.7 times mmap, and
verified-cold direct reads were also substantially slower than verified-cold
buffered reads. These are negative storage-path results, not end-to-end OCR
measurements. They keep mmap as the default and do not satisfy the stable
named-hardware or independent-controller evidence requirements.
