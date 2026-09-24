# Embedded CUDA GraphExecutor evidence (RTX 4090)

First-principles gate: [perf-first-principles.md](../../perf-first-principles.md).

## GELU-erf fusion (`a3s-power-cuda-fusion-bench`)

Fused Div/Erf/Add/Mul/Mul vs Identity-broken unfused plan. Parity required;
speedup is measured, not assumed.

| Elements | Fused median | Unfused median | Speedup | Parity | Artifact |
| --- | --- | --- | --- | --- | --- |
| 1,048,576 | 3.47 ms | 3.49 ms | 1.008× | yes | `cuda-fusion-bench-e1048576-*.json` |
| 4,194,304 | 15.54 ms | 14.73 ms | 0.948× | yes | `cuda-fusion-bench-20260923-234900.json` |
| 16,777,216 | 58.64 ms | 59.81 ms | 1.020× | yes | `cuda-fusion-bench-e16777216-*.json` |
| 67,108,864 | 237.19 ms | 241.30 ms | 1.017× | yes | `cuda-fusion-bench-e67108864-*.json` |

**Verdict:** byte-exact fusion holds. Wall-clock win for this elementwise window
on Ada is **noise-level (~0–2%)**. Do **not** claim a tok/s or product win from
GELU-erf fusion alone — bandwidth-bound chains do not justify further kernel
tuning without a larger resident/H↔D lever.

```powershell
.\target\release\a3s-power-cuda-fusion-bench.exe `
  --elements 16777216 --warmup-rounds 3 --measured-rounds 11 `
  --out docs\benchmarks\cuda-fusion-rtx4090\cuda-fusion-bench.json
```

## Device-resident chain (`a3s-power-cuda-resident-bench`)

Two-graph Add chain: resident (`run_to_resident` → `run_resident` →
`materialize`) vs owned host round-trip (`run` → `into_input` → `run`).

### Digest-fair capture (2026-09-24) — authoritative

Owned path now hashes chain endpoints inside the timed window (same digest
contract as residency). Artifact:
`cuda-resident-bench-digest-fair-20260924-043540.json`.

| Elements | Graphs | Resident median | Owned median | Speedup | Copy contract | Parity |
| --- | --- | --- | --- | --- | --- | --- |
| 4,194,304 | 2 | 71.0 ms | 88.9 ms | **1.25×** | 1/1 vs 2/2 | yes |
| 16,777,216 | 2 | 275.9 ms | 346.8 ms | **1.26×** | ok | yes |
| 4,194,304 | 4 | 64.2 ms | 135.8 ms | **2.12×** | 1/1 vs 4/4 | yes |
| 4,194,304 | 8 | 65.7 ms | 222.3 ms | **3.38×** | 1/1 vs 8/8 | yes |

Artifacts: `cuda-resident-bench-digest-fair-*.json`,
`cuda-resident-bench-digest-fair-g4-*.json`,
`cuda-resident-bench-digest-fair-g8-*.json`.

**Verdict:** digest-fair residency wins on Ada; speedup **scales with chain
depth** because owned pays N H↔D round-trips while resident stays at 1/1.
Earlier ~0.45× “losses” were a measurement artifact (owned skipped SHA-256).

### Pre-fair captures (historical, not claimed)

| Elements | Resident | Owned | Apparent speedup |
| --- | --- | --- | --- |
| 4M–67M | — | no endpoint digests | ~0.45× (invalid comparison) |

```powershell
.\target\release\a3s-power-cuda-resident-bench.exe `
  --elements 4194304 --chain-graphs 4 --warmup-rounds 5 --measured-rounds 21 `
  --out docs\benchmarks\cuda-fusion-rtx4090\cuda-resident-bench-digest-fair.json
```
