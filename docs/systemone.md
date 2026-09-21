# System One (Jev-shaped) decision scoring

Power exposes a Jev-compatible decision endpoint at `POST /v1/systemone`.
It scores declared options by reading next-token logits for single-token
labels (`A`/`B`/`C`…, `yes`/`no`, or level digits), then applying a softmax
over that subset only.

This is **not** TypeSafe Jev. Probabilities are conditional over the declared
option-label tokens and are **not** calibrated confidence. Quality depends on
the loaded instruct GGUF (for example Qwen3.5-4B **Q8_0** on an RTX 4090)
and quantization choices. Prefer Q8_0 on 24 GB GPUs; Q4 is for low-VRAM demos
only.

## Requirements

- Build with the `llamacpp` feature (prefer `llamacpp-cuda` on NVIDIA hosts).
- `POST /v1/systemone` always selects a System One-capable backend (`llama.cpp`),
  even when mistral.rs has higher GGUF format priority for chat.
- If no System One backend is available, the API returns `backend_unavailable`.
- Option labels must tokenize to exactly one token; otherwise the request fails
  closed.

## Local Qwen3.5-4B demo (Q8_0)

See [`examples/systemone-qwen35-4b/`](examples/systemone-qwen35-4b/README.md).
Default artifact: `unsloth/Qwen3.5-4B-GGUF` → `Qwen3.5-4B-Q8_0.gguf` (~4.48 GB).

## Request / response

Same wire shape as TypeSafe `POST /v1/systemone`:

- `choice` → `choice`, `probabilities`, `confidence`
- `noul` → `noul` (`P(yes)`)
- `score` → `score` (argmax level index), `legend`, `probabilities`, `confidence`

`confidence = 1 - H(p) / ln(|options|)`. This is a documented Power formula, not
Jev’s internal statistic.

Receipts and attestation do not bind System One outputs in this slice.
