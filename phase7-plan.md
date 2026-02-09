# Phase 7: Ollama Registry Deep Integration & CLI Enhancement

## Context

We just added Ollama Registry as the primary model resolution source (Phase 6.5). Models pulled from Ollama now carry rich metadata (template, system prompt, params, license). However, several gaps remain to make a3s-power a true drop-in Ollama replacement.

## Goal

Complete the Ollama compatibility story with:
1. **CLI `show` enrichment** — Display template, system prompt, license from registry metadata
2. **CLI `pull` auto-run** — `run` command auto-pulls if model not found locally (like Ollama)
3. **Streaming pull progress** — Show per-layer progress like Ollama does during pull
4. **Default parameters application** — Apply registry default params (stop tokens) during inference

---

## Part A: CLI `show` Enrichment (~30 min)

### Problem
`cli/show.rs` only displays basic fields (name, format, size, sha256, parameters). It ignores the new `system_prompt`, `template_override`, `default_parameters`, and `license` fields from Ollama registry.

### Changes

**File: `src/cli/show.rs`**

Add display of new fields after the existing parameters section:

```rust
// After existing parameters display...

if let Some(system) = &manifest.system_prompt {
    println!("\nSystem:");
    println!("  {system}");
}

if let Some(template) = &manifest.template_override {
    println!("\nTemplate:");
    for line in template.lines() {
        println!("  {line}");
    }
}

if let Some(params) = &manifest.default_parameters {
    println!("\nDefault Parameters:");
    for (key, value) in params {
        println!("  {key}: {value}");
    }
}

if let Some(license) = &manifest.license {
    println!("\nLicense:");
    // Show first 5 lines + truncation notice
    let lines: Vec<&str> = license.lines().collect();
    let show = lines.len().min(5);
    for line in &lines[..show] {
        println!("  {line}");
    }
    if lines.len() > 5 {
        println!("  ... ({} more lines)", lines.len() - 5);
    }
}
```

**Tests:** Update existing show tests, add test for rich metadata display.

---

## Part B: Auto-Pull on `run` (~45 min)

### Problem
When a user runs `a3s-power run llama3.2:3b` and the model isn't local, it prints "Model not found locally" and exits. Ollama auto-pulls the model instead.

### Changes

**File: `src/cli/run.rs`**

In `execute_with_options()`, replace the "not found" early return with auto-pull logic:

```rust
let manifest = match registry.get(model) {
    Ok(m) => m,
    Err(_) => {
        println!("Model '{model}' not found locally. Pulling...");
        // Auto-pull the model
        crate::cli::pull::execute(model, registry).await?;
        registry.get(model)?
    }
};
```

**Tests:** Add test for auto-pull flow.

---

## Part C: Apply Default Parameters During Inference (~1 hr)

### Problem
Models pulled from Ollama registry carry `default_parameters` (e.g. stop tokens like `["<|eot_id|>"]`). These are stored in the manifest but never applied during inference. Without stop tokens, models generate garbage after the response.

### Changes

**File: `src/api/native/chat.rs`** and **`src/api/native/generate.rs`**

When building the backend `ChatRequest`/`CompletionRequest`, merge manifest default params as base, then let request-level params override:

```rust
// In chat handler, after loading manifest:
let default_stop = manifest.default_parameters
    .as_ref()
    .and_then(|p| p.get("stop"))
    .and_then(|v| v.as_array())
    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<_>>());

// Use request stop tokens if provided, otherwise fall back to defaults
let stop = request.options.as_ref()
    .and_then(|o| o.stop.clone())
    .or(default_stop);
```

Same pattern for OpenAI chat/completions endpoints.

**Files modified:**
- `src/api/native/chat.rs` — merge default stop tokens
- `src/api/native/generate.rs` — merge default stop tokens
- `src/api/openai/chat.rs` — merge default stop tokens
- `src/api/openai/completions.rs` — merge default stop tokens

**Tests:** Add tests verifying default params are applied and request params override them.

---

## Part D: System Prompt Application During Inference (~30 min)

### Problem
Models pulled from Ollama registry carry `system_prompt` (e.g. "You are a helpful assistant"). This is stored but never injected into inference requests. Without it, models may behave differently than expected.

### Changes

**File: `src/api/native/chat.rs`** and **`src/api/native/generate.rs`**

When building messages for the backend, prepend the system prompt if:
1. The manifest has a `system_prompt`
2. The request doesn't already have a system message
3. The request doesn't explicitly override with its own `system` field

```rust
// For chat: prepend system message if not already present
let has_system = messages.iter().any(|m| m.role == "system");
if !has_system {
    if let Some(system) = &manifest.system_prompt {
        messages.insert(0, ChatMessage {
            role: "system".to_string(),
            content: MessageContent::Text(system.clone()),
            ..Default::default()
        });
    }
}
```

**Tests:** Add tests for system prompt injection and override behavior.

---

## Files Changed Summary

| File | Change |
|------|--------|
| `src/cli/show.rs` | Display template, system, params, license |
| `src/cli/run.rs` | Auto-pull model if not found locally |
| `src/api/native/chat.rs` | Apply default stop tokens + system prompt |
| `src/api/native/generate.rs` | Apply default stop tokens + system prompt |
| `src/api/openai/chat.rs` | Apply default stop tokens + system prompt |
| `src/api/openai/completions.rs` | Apply default stop tokens |

## Verification

```bash
# 1. All existing tests pass
cargo test -p a3s-power --lib

# 2. CLI show displays rich metadata
cargo run -p a3s-power -- show llama3.2:3b
# Should show: Template, System, Default Parameters, License

# 3. Run auto-pulls
cargo run -p a3s-power -- run llama3.2:3b
# Should auto-pull if not local, then start chat
```
