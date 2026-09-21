# System One demo: Qwen3.5-4B GGUF (Q8_0)

Run a local Jev-shaped decision scorer with a3s-power + llama.cpp.

## Honesty

This demo uses OpenJev-style **option-logit readout** on a frozen instruct
GGUF. It does **not** reproduce TypeSafe Jev weights or calibration.

## Quantization (RTX 4090 default)

On a 24 GB GPU (for example RTX 4090), use **Q8_0** (~4.48 GB). Do not use
Q4_K_M on this class of hardware unless you are deliberately matching a
phone/browser demo footprint.

| Quant | Approx size | When to use |
| --- | ---: | --- |
| **Q8_0** (default here) | 4.48 GB | Desktop / 4090 decision scoring |
| Q6_K | ~3.5 GB | Acceptable if Q8 is unavailable |
| Q4_K_M | ~2.7 GB | Low-VRAM / browser demos only |

Artifact: `unsloth/Qwen3.5-4B-GGUF` → `Qwen3.5-4B-Q8_0.gguf` (~4.48 GB)

Expected SHA-256:

```text
10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1
```

Local path used on this workstation: `D:\models\Qwen3.5-4B-Q8_0.gguf`

## Build

```powershell
# NVIDIA GPU (recommended on 4090)
cargo build --locked --release --no-default-features --features "server,llamacpp-cuda"

# CPU-only fallback
cargo build --locked --release --no-default-features --features "server,llamacpp"
```

## Serve

```powershell
$env:A3S_POWER_MODEL_SOURCE = "hf"
.\target\release\a3s-power.exe serve --config docs\examples\systemone-qwen35-4b\power.acl
```

In another shell, pull Q8_0 (or register a local file if already downloaded):

```powershell
$env:A3S_POWER_MODEL_SOURCE = "hf"
.\target\release\a3s-power.exe models pull unsloth/Qwen3.5-4B-GGUF:q8_0
```

Or register a local GGUF (example path used by this machine’s download):

```powershell
curl.exe -X POST http://127.0.0.1:11434/v1/models `
  -H "Content-Type: application/json" `
  -d "{\"name\":\"qwen35-4b-q8\",\"path\":\"D:/models/Qwen3.5-4B-Q8_0.gguf\",\"format\":\"gguf\"}"
```

`POST /v1/systemone` always loads via **llama.cpp** (not mistral.rs), even when
both backends are enabled. That is required for Qwen3.5 (`qwen35` GGUF arch)
and for option-label logit scoring.

## Score a decision

```powershell
curl.exe -X POST http://127.0.0.1:11434/v1/systemone `
  -H "Content-Type: application/json" `
  -d "@docs/examples/systemone-qwen35-4b/request.example.json"
```

The example request uses model name `qwen35-4b-q8` (the local register id above).
Replace `model` if you pulled from Hugging Face instead.
