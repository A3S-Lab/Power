//! A3S Box + Power Integration Example
//!
//! Proves the full stack works end-to-end:
//! build Power → launch MicroVM → start Power inside → call API → verify responses,
//! including real model inference with a GGUF model.
//!
//! Prerequisites:
//! - `a3s-box-shim` installed (`~/.a3s/bin/a3s-box-shim`)
//! - macOS (HVF) or Linux (KVM) with virtualization support
//! - Network access to pull `alpine:latest` OCI image
//! - A GGUF model file at `/tmp/test-models/qwen2.5-0.5b-q4_k_m.gguf`
//!   (download: `huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir /tmp/test-models`)
//!
//! Run with:
//! ```bash
//! cargo run --example box_integration -p a3s-power
//! ```

use a3s_box_sdk::{BoxSdk, MountSpec, SandboxOptions};
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

/// Model name used in the manifest and API requests.
const MODEL_NAME: &str = "qwen2.5:0.5b";
/// Path to the GGUF model file on the host.
const MODEL_HOST_PATH: &str = "/tmp/test-models/qwen2.5-0.5b-q4_k_m.gguf";
/// SHA-256 of the model file.
const MODEL_SHA256: &str = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db";
/// Model file size in bytes.
const MODEL_SIZE: u64 = 491400032;

/// Build the `a3s-power` release binary for Linux (aarch64-unknown-linux-musl).
/// The MicroVM runs Linux, so we must cross-compile from macOS.
/// Returns the path to the compiled binary.
fn build_power_binary() -> Result<PathBuf, Box<dyn std::error::Error>> {
    println!("[1/7] Building a3s-power binary (cross-compile for linux-musl)...");

    let target = "aarch64-unknown-linux-musl";

    let status = Command::new("cargo")
        .args(["build", "--release", "-p", "a3s-power", "--target", target])
        .status()?;

    if !status.success() {
        return Err(format!("cargo build failed with exit code: {}", status).into());
    }

    // Resolve the target dir — try cargo metadata first, then common locations.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Try cargo metadata to find the target dir.
    let output = Command::new("cargo")
        .args(["metadata", "--format-version=1", "--no-deps"])
        .output()?;
    if output.status.success() {
        let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        if let Some(target_dir) = metadata["target_directory"].as_str() {
            let meta_binary = PathBuf::from(target_dir)
                .join(target)
                .join("release")
                .join("a3s-power");
            if meta_binary.exists() {
                return Ok(meta_binary);
            }
        }
    }

    // Try local target dir.
    let binary = manifest_dir
        .join("target")
        .join(target)
        .join("release")
        .join("a3s-power");
    if binary.exists() {
        return Ok(binary);
    }

    // Try workspace root (two levels up from crates/power/).
    let workspace_binary = manifest_dir.parent().and_then(|p| p.parent()).map(|root| {
        root.join("target")
            .join(target)
            .join("release")
            .join("a3s-power")
    });

    if let Some(ref path) = workspace_binary {
        if path.exists() {
            return Ok(path.clone());
        }
    }

    Err(format!(
        "a3s-power binary not found. Checked:\n  - {}\n  - {:?}",
        binary.display(),
        workspace_binary.map(|p| p.display().to_string())
    )
    .into())
}

/// Install a tiny raw HTTP client script in the guest.
/// Uses busybox wget for GET and raw HTTP via awk for POST (since busybox
/// wget may not support --post-file/--post-data).
async fn install_http_helper(
    sandbox: &a3s_box_sdk::Sandbox,
) -> Result<(), Box<dyn std::error::Error>> {
    // The script uses wget for GET requests and awk's TCP support for POST.
    // Busybox awk doesn't support /inet/tcp, so for POST we write a raw
    // HTTP request to a file and pipe it through nc.
    //
    // If nc doesn't work (vsock TSI limitation), we fall back to wget
    // with --post-file (which may silently do GET instead).
    let script = concat!(
        "#!/bin/sh\n",
        "URL=\"http://127.0.0.1:11434$2\"\n",
        "if [ \"$3\" = \"post\" ]; then\n",
        "  BLEN=$(wc -c < /tmp/post_body | tr -d ' ')\n",
        "  printf 'POST %s HTTP/1.0\\r\\nHost: 127.0.0.1\\r\\nContent-Type: application/json\\r\\nContent-Length: %s\\r\\nConnection: close\\r\\n\\r\\n' \"$2\" \"$BLEN\" > /tmp/raw_req\n",
        "  cat /tmp/post_body >> /tmp/raw_req\n",
        "  RESP=$(cat /tmp/raw_req | nc -w 5 127.0.0.1 11434 2>/dev/null)\n",
        "  if [ -n \"$RESP\" ]; then\n",
        "    CODE=$(echo \"$RESP\" | head -1 | awk '{print $2}')\n",
        "    RBODY=$(echo \"$RESP\" | sed '1,/^\\r*$/d')\n",
        "    printf 'HTTP_STATUS=%s\\n%s\\n' \"$CODE\" \"$RBODY\"\n",
        "    exit 0\n",
        "  fi\n",
        "  B=$(wget -q -O - -T 3 --header='Content-Type: application/json' --post-file=/tmp/post_body \"$URL\" 2>/dev/null)\n",
        "  R=$?\n",
        "else\n",
        "  B=$(wget -q -O - -T 3 \"$URL\" 2>/dev/null)\n",
        "  R=$?\n",
        "fi\n",
        "if [ $R -eq 0 ]; then\n",
        "  printf 'HTTP_STATUS=200\\n%s\\n' \"$B\"\n",
        "  exit 0\n",
        "fi\n",
        "if [ \"$3\" = \"post\" ]; then\n",
        "  E=$(wget -O /dev/null -T 3 --header='Content-Type: application/json' --post-file=/tmp/post_body \"$URL\" 2>&1)\n",
        "else\n",
        "  E=$(wget -O /dev/null -T 3 \"$URL\" 2>&1)\n",
        "fi\n",
        "C=$(echo \"$E\" | grep -o 'HTTP/[^ ]* [0-9]*' | tail -1 | awk '{print $2}')\n",
        "if [ -z \"$C\" ]; then\n",
        "  C=$(echo \"$E\" | grep -o 'error: [0-9]*' | head -1 | grep -o '[0-9]*')\n",
        "fi\n",
        "if [ -n \"$C\" ]; then\n",
        "  printf 'HTTP_STATUS=%s\\n%s\\n' \"$C\" \"$B\"\n",
        "else\n",
        "  printf 'HTTP_STATUS=0\\nrc=%s err=%s\\n' \"$R\" \"$E\"\n",
        "fi\n",
    );

    let b64 = base64_encode(script.as_bytes());

    let write_cmd = format!(
        "echo '{}' | base64 -d > /tmp/httpreq && chmod +x /tmp/httpreq",
        b64
    );

    for attempt in 0..3 {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        let result = sandbox.exec("/bin/sh", &["-c", &write_cmd]).await?;
        if result.stderr.contains("No child process") {
            continue;
        }
        if result.exit_code == 0 {
            return Ok(());
        }
        return Err(format!(
            "Failed to install HTTP helper (attempt {}): exit={} stderr={}",
            attempt, result.exit_code, result.stderr
        )
        .into());
    }

    Err("Failed to install HTTP helper after 3 retries".into())
}

/// Simple base64 encoder (avoids adding a dependency just for this).
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Make an HTTP request inside the sandbox.
/// Uses curl if available (supports POST + status codes), falls back to wget helper.
/// Returns `(http_status_code, response_body)`.
///
/// Retries on transient guest agent errors ("No child process").
async fn http_request(
    sandbox: &a3s_box_sdk::Sandbox,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> Result<(u16, String), Box<dyn std::error::Error>> {
    let url = format!("http://127.0.0.1:11434{path}");

    // Build curl command: -s (silent), -w for status code, -X for method.
    let cmd = if let Some(content) = body {
        let escaped = content.replace('\'', "'\\''");
        format!(
            "curl -s -w '\\nHTTP_STATUS=%{{http_code}}' -X {method} -H 'Content-Type: application/json' -d '{escaped}' '{url}'"
        )
    } else {
        format!("curl -s -w '\\nHTTP_STATUS=%{{http_code}}' '{url}'")
    };

    // Retry up to 8 times for transient guest agent errors or vsock contention.
    let mut last_err = String::new();
    for attempt in 0..8 {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(1000 + attempt as u64 * 500)).await;
        }

        let result = sandbox.exec("/bin/sh", &["-c", &cmd]).await?;

        // Check for guest agent transient error.
        if result.stderr.contains("No child process") || result.stdout.contains("No child process")
        {
            last_err = format!("stderr={} stdout={}", result.stderr, result.stdout);
            continue;
        }

        // Parse output: body followed by \nHTTP_STATUS=<code>
        let output = &result.stdout;
        if let Some(pos) = output.rfind("HTTP_STATUS=") {
            let code_str = output[pos + 12..].trim();
            let resp_body = output[..pos].trim_end().to_string();
            let code = code_str.parse::<u16>().unwrap_or(0);

            if code == 0 {
                // curl couldn't connect — server may not be ready yet.
                last_err = format!("HTTP_STATUS=0 (connection failed): {output}");
                continue;
            }

            return Ok((code, resp_body));
        }

        // Empty output or missing HTTP_STATUS marker — treat as transient error and retry.
        if output.trim().is_empty() {
            last_err = "curl returned empty output (vsock contention?)".to_string();
            continue;
        }

        // curl might not be installed — fall back to wget helper.
        return http_request_wget(sandbox, method, path, body).await;
    }

    Err(format!("HTTP request failed after 8 retries for {method} {path}: {last_err}").into())
}

/// HTTP request with a custom timeout (in seconds) for slow operations like inference.
async fn http_request_with_timeout(
    sandbox: &a3s_box_sdk::Sandbox,
    method: &str,
    path: &str,
    body: Option<&str>,
    timeout_secs: u64,
) -> Result<(u16, String), Box<dyn std::error::Error>> {
    let url = format!("http://127.0.0.1:11434{path}");

    let cmd = if let Some(content) = body {
        let escaped = content.replace('\'', "'\\''");
        format!(
            "curl -s -m {timeout_secs} -w '\\nHTTP_STATUS=%{{http_code}}' -X {method} -H 'Content-Type: application/json' -d '{escaped}' '{url}'"
        )
    } else {
        format!("curl -s -m {timeout_secs} -w '\\nHTTP_STATUS=%{{http_code}}' '{url}'")
    };

    let mut last_err = String::new();
    for attempt in 0..5 {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_secs(3)).await;
        }

        let result = sandbox.exec("/bin/sh", &["-c", &cmd]).await?;

        if result.stderr.contains("No child process") || result.stdout.contains("No child process")
        {
            last_err = format!("stderr={} stdout={}", result.stderr, result.stdout);
            continue;
        }

        let output = &result.stdout;

        // Empty output — vsock contention, retry.
        if output.trim().is_empty() {
            last_err = "curl returned empty output (vsock contention?)".to_string();
            continue;
        }

        if let Some(pos) = output.rfind("HTTP_STATUS=") {
            let code_str = output[pos + 12..].trim();
            let resp_body = output[..pos].trim_end().to_string();
            let code = code_str.parse::<u16>().unwrap_or(0);

            if code == 0 {
                last_err = format!("HTTP_STATUS=0 (connection failed or timeout): {output}");
                continue;
            }

            return Ok((code, resp_body));
        }

        // Non-empty but no HTTP_STATUS marker — unexpected.
        last_err = format!(
            "curl output missing HTTP_STATUS marker: {}",
            output.chars().take(200).collect::<String>()
        );
        continue;
    }

    Err(format!(
        "HTTP request (timeout={timeout_secs}s) failed after 5 retries for {method} {path}: {last_err}"
    ).into())
}

/// Fallback HTTP request using wget helper script.
async fn http_request_wget(
    sandbox: &a3s_box_sdk::Sandbox,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> Result<(u16, String), Box<dyn std::error::Error>> {
    // For POST, write the body to a temp file first.
    if let Some(content) = body {
        let escaped = content.replace('\'', "'\\''");
        for _ in 0..3 {
            let wr = sandbox
                .exec(
                    "/bin/sh",
                    &["-c", &format!("printf '%s' '{escaped}' > /tmp/post_body")],
                )
                .await?;
            if !wr.stderr.contains("No child process") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    }

    let has_body_arg = if body.is_some() { " post" } else { "" };
    let cmd = format!("/tmp/httpreq '{method}' '{path}'{has_body_arg}");

    let mut last_err = String::new();
    for attempt in 0..5 {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(500 + attempt as u64 * 300)).await;
        }

        let result = sandbox.exec("/bin/sh", &["-c", &cmd]).await?;

        if result.stderr.contains("No child process") || result.stdout.contains("No child process")
        {
            last_err = format!("stderr={} stdout={}", result.stderr, result.stdout);
            continue;
        }

        let output = result.stdout.trim();
        if let Some(rest) = output.strip_prefix("HTTP_STATUS=") {
            let mut lines = rest.splitn(2, '\n');
            let code_str = lines.next().unwrap_or("0").trim();
            let resp_body = lines.next().unwrap_or("").to_string();

            let code = code_str.parse::<u16>().unwrap_or(0);
            if code == 0 {
                last_err = format!("HTTP_STATUS=0: {output}");
                continue;
            }

            return Ok((code, resp_body));
        }

        return Err(format!(
            "Unexpected httpreq output for {method} {path}\nexit_code: {}\nstdout: {}\nstderr: {}",
            result.exit_code, result.stdout, result.stderr
        )
        .into());
    }

    Err(
        format!("HTTP request (wget) failed after 5 retries for {method} {path}: {last_err}")
            .into(),
    )
}

/// Poll `GET /health` until Power responds with 200, or timeout.
async fn wait_for_ready(
    sandbox: &a3s_box_sdk::Sandbox,
    timeout: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Waiting for a3s-power to be ready...");

    let start = tokio::time::Instant::now();
    let mut attempts = 0u32;
    loop {
        if start.elapsed() > timeout {
            // Debug: check if the process is running at all
            let ps = sandbox
                .exec("/bin/sh", &["-c", "ps aux 2>/dev/null || ps"])
                .await;
            let ps_output = ps
                .map(|r| r.stdout)
                .unwrap_or_else(|e| format!("ps failed: {e}"));
            return Err(format!(
                "a3s-power did not become ready within {}s (after {attempts} attempts)\nProcesses:\n{ps_output}",
                timeout.as_secs()
            )
            .into());
        }

        attempts += 1;

        // Try health endpoint — ignore errors (Power may not be listening yet).
        match http_request(sandbox, "GET", "/health", None).await {
            Ok((status, _)) if status == 200 => return Ok(()),
            Ok((status, body)) => {
                if attempts <= 3 || attempts % 10 == 0 {
                    eprintln!("  attempt {attempts}: status={status} body={body}");
                }
            }
            Err(e) => {
                if attempts <= 3 || attempts % 10 == 0 {
                    eprintln!("  attempt {attempts}: {e}");
                }
            }
        }

        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

/// Run all API test cases. Returns the number of passed tests.
async fn run_tests(
    sandbox: &a3s_box_sdk::Sandbox,
    has_model: bool,
) -> Result<u32, Box<dyn std::error::Error>> {
    println!("[6/7] Running API tests...");

    let mut passed = 0u32;

    // Test 1: Health
    {
        let (status, body) = http_request(sandbox, "GET", "/health", None).await?;
        assert_eq!(status, 200, "GET /health expected 200, got {status}");
        assert!(
            body.contains("\"status\""),
            "Health response should contain status field, got: {body}"
        );
        println!("  ✓ GET /health → {status}");
        passed += 1;
    }

    // Test 2: Models list — should contain our model if mounted
    {
        let (status, body) = http_request(sandbox, "GET", "/v1/models", None).await?;
        assert_eq!(status, 200, "GET /v1/models expected 200, got {status}");
        assert!(
            body.contains("\"data\""),
            "Models response should contain data field"
        );
        if has_model {
            assert!(
                body.contains(MODEL_NAME),
                "Models list should contain '{MODEL_NAME}', got: {body}"
            );
            println!("  ✓ GET /v1/models → {status} (contains {MODEL_NAME})");
        } else {
            println!("  ✓ GET /v1/models → {status} (empty list)");
        }
        passed += 1;
    }

    // Test 3: Chat completions with unknown model → 200 with error body
    {
        let chat_body = r#"{"model":"nonexistent","messages":[{"role":"user","content":"hi"}]}"#;
        let (status, body) =
            http_request(sandbox, "POST", "/v1/chat/completions", Some(chat_body)).await?;
        assert_eq!(
            status, 200,
            "POST /v1/chat/completions expected 200, got {status}"
        );
        assert!(
            body.contains("model_not_found"),
            "POST /v1/chat/completions should contain model_not_found error, got: {body}"
        );
        println!("  ✓ POST /v1/chat/completions (unknown model) → {status} (model_not_found)");
        passed += 1;
    }

    // Test 4: Completions with unknown model → 200 with error body
    {
        let completions_body = r#"{"model":"nonexistent","prompt":"hello"}"#;
        let (status, body) =
            http_request(sandbox, "POST", "/v1/completions", Some(completions_body)).await?;
        assert_eq!(
            status, 200,
            "POST /v1/completions expected 200, got {status}"
        );
        assert!(
            body.contains("model_not_found"),
            "POST /v1/completions should contain model_not_found error, got: {body}"
        );
        println!("  ✓ POST /v1/completions (unknown model) → {status} (model_not_found)");
        passed += 1;
    }

    // Test 5: Metrics endpoint
    {
        let (status, body) = http_request(sandbox, "GET", "/metrics", None).await?;
        assert_eq!(status, 200, "GET /metrics expected 200, got {status}");
        assert!(
            body.contains("power_"),
            "Metrics response should contain power_ prefix"
        );
        println!("  ✓ GET /metrics → {status}");
        passed += 1;
    }

    // Test 6: Unknown route → 404
    {
        let (status, _) = http_request(sandbox, "GET", "/nonexistent", None).await?;
        assert_eq!(status, 404, "GET /nonexistent expected 404, got {status}");
        println!("  ✓ GET /nonexistent → {status}");
        passed += 1;
    }

    // Test 7: Real model inference — chat completion with the loaded model
    if has_model {
        println!("\n[7/7] Running inference test with {MODEL_NAME}...");
        let chat_body = format!(
            r#"{{"model":"{MODEL_NAME}","messages":[{{"role":"user","content":"What is 2+2? Answer with just the number."}}],"max_tokens":32}}"#
        );
        let (status, body) = http_request_with_timeout(
            sandbox,
            "POST",
            "/v1/chat/completions",
            Some(&chat_body),
            120,
        )
        .await?;
        assert_eq!(
            status, 200,
            "POST /v1/chat/completions (real model) expected 200, got {status}"
        );
        assert!(
            !body.contains("model_not_found"),
            "Real model inference should not return model_not_found, got: {body}"
        );
        assert!(
            body.contains("\"choices\""),
            "Real model inference should contain choices, got: {body}"
        );
        // Extract and display the model's response
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
            if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
                println!("  Model response: \"{content}\"");
                assert!(
                    !content.is_empty(),
                    "Model should generate non-empty response"
                );
            }
        }
        println!("  ✓ POST /v1/chat/completions ({MODEL_NAME}) → {status} (real inference)");
        passed += 1;
    } else {
        println!("  ⊘ Inference test skipped (no model file at {MODEL_HOST_PATH})");
    }

    Ok(passed)
}

/// Generate the model manifest JSON for the guest filesystem.
fn model_manifest_json(guest_model_path: &str) -> String {
    serde_json::json!({
        "name": MODEL_NAME,
        "format": "gguf",
        "size": MODEL_SIZE,
        "sha256": MODEL_SHA256,
        "parameters": {
            "context_length": 4096,
            "embedding_length": null,
            "parameter_count": 500_000_000u64,
            "quantization": "Q4_K_M"
        },
        "created_at": "2025-01-01T00:00:00Z",
        "path": guest_model_path,
        "family": "qwen2"
    })
    .to_string()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check if model file exists on host.
    let has_model = std::path::Path::new(MODEL_HOST_PATH).exists();
    if has_model {
        println!("Model file found: {MODEL_HOST_PATH}");
    } else {
        println!("Warning: No model file at {MODEL_HOST_PATH} — inference test will be skipped");
        println!("  Download with: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir /tmp/test-models");
    }

    // 1. Build the binary (with default features = mistralrs backend).
    let binary_path = build_power_binary()?;
    println!("  Binary: {}", binary_path.display());

    let binary_dir = binary_path
        .parent()
        .ok_or("Cannot determine binary parent directory")?
        .to_string_lossy()
        .to_string();

    // 2. Create MicroVM sandbox with model mounted.
    println!("[2/7] Creating MicroVM sandbox...");
    let sdk = BoxSdk::new().await?;

    let mut mounts = vec![MountSpec {
        host_path: binary_dir,
        guest_path: "/opt/bin".into(),
        readonly: true,
    }];

    // Mount the model directory if the model file exists.
    if has_model {
        let model_host_dir = std::path::Path::new(MODEL_HOST_PATH)
            .parent()
            .unwrap()
            .to_string_lossy()
            .to_string();
        mounts.push(MountSpec {
            host_path: model_host_dir,
            guest_path: "/opt/models".into(),
            readonly: true,
        });
    }

    let sandbox = sdk
        .create(SandboxOptions {
            image: "alpine:latest".into(),
            // 1536 MB: ~469MB model + mistral.rs runtime + KV cache + OS overhead
            memory_mb: if has_model { 1536 } else { 512 },
            network: true,
            name: Some("power-integration-test".into()),
            mounts,
            ..Default::default()
        })
        .await?;

    println!("  Sandbox: {} ({})", sandbox.name(), sandbox.id());

    // 3. Set up the guest environment.
    println!("[3/7] Setting up guest environment...");

    // Copy the binary to /tmp so it's on a writable+executable filesystem.
    for attempt in 0..3 {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        let cp = sandbox
            .exec(
                "/bin/sh",
                &[
                    "-c",
                    "cp /opt/bin/a3s-power /tmp/a3s-power && chmod +x /tmp/a3s-power",
                ],
            )
            .await?;
        if !cp.stderr.contains("No child process") && cp.exit_code == 0 {
            break;
        }
        if attempt == 2 {
            return Err(format!("Failed to copy binary: {}", cp.stderr).into());
        }
    }

    // Install curl for full HTTP testing.
    println!("  Installing curl...");
    let mut curl_installed = false;
    for attempt in 0..3 {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        let install = sandbox
            .exec("/bin/sh", &["-c", "apk add --no-cache curl 2>&1"])
            .await?;
        if install.stderr.contains("No child process") {
            continue;
        }
        if install.exit_code == 0 {
            curl_installed = true;
            println!("  curl installed");
            break;
        }
        eprintln!(
            "  apk failed (attempt {attempt}): {}",
            install.stdout.trim()
        );
    }
    if !curl_installed {
        println!("  Warning: curl not available, falling back to wget");
    }

    // Install the wget-based HTTP helper script as fallback.
    install_http_helper(&sandbox).await?;

    // 4. Set up model manifest inside the guest if model is available.
    if has_model {
        println!("[4/7] Setting up model manifest...");

        // Create A3S_POWER_HOME directory structure inside the guest.
        let setup_dirs = "mkdir -p /tmp/power-home/models/manifests /tmp/power-home/models/blobs";
        exec_retry(&sandbox, setup_dirs).await?;

        // The model file is mounted at /opt/models/<filename>.
        let model_filename = std::path::Path::new(MODEL_HOST_PATH)
            .file_name()
            .unwrap()
            .to_string_lossy();
        let guest_model_path = format!("/opt/models/{model_filename}");

        // Write the manifest JSON using base64 to avoid shell escaping issues.
        let manifest_json = model_manifest_json(&guest_model_path);
        let manifest_b64 = base64_encode(manifest_json.as_bytes());
        let manifest_filename = MODEL_NAME.replace([':', '/'], "-");
        let write_manifest = format!(
            "echo '{}' | base64 -d > /tmp/power-home/models/manifests/{manifest_filename}.json",
            manifest_b64
        );
        exec_retry(&sandbox, &write_manifest).await?;

        // Verify the manifest was written.
        let verify = sandbox
            .exec(
                "/bin/sh",
                &[
                    "-c",
                    "cat /tmp/power-home/models/manifests/*.json | head -c 200",
                ],
            )
            .await?;
        println!(
            "  Manifest: {}",
            verify.stdout.trim().chars().take(120).collect::<String>()
        );

        // Verify the model file is accessible.
        let check_model = sandbox
            .exec(
                "/bin/sh",
                &["-c", &format!("ls -lh {guest_model_path} 2>&1")],
            )
            .await?;
        println!("  Model: {}", check_model.stdout.trim());
    } else {
        println!("[4/7] Skipping model setup (no model file)");
    }

    // 5. Start Power as a background daemon.
    println!("[5/7] Starting a3s-power...");

    // Start with A3S_POWER_HOME set so it finds the model manifest.
    let start_cmd = if has_model {
        "A3S_POWER_HOME=/tmp/power-home nohup /tmp/a3s-power > /tmp/power.log 2>&1 &"
    } else {
        "nohup /tmp/a3s-power > /tmp/power.log 2>&1 &"
    };

    for attempt in 0..3 {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        let start = sandbox.exec("/bin/sh", &["-c", start_cmd]).await?;
        if !start.stderr.contains("No child process") {
            break;
        }
        if attempt == 2 {
            return Err(format!("Failed to start a3s-power: {}", start.stderr).into());
        }
    }

    // Give Power time to start up. The vsock muxer floods with
    // "EventSet::OUT while not connecting" errors during startup — waiting
    // longer lets that settle before we start making exec calls.
    tokio::time::sleep(Duration::from_secs(12)).await;

    // Verify Power is running and check its log.
    let log_result = sandbox
        .exec(
            "/bin/sh",
            &["-c", "cat /tmp/power.log 2>/dev/null | head -10"],
        )
        .await?;
    let log_trimmed = log_result.stdout.trim();
    if log_trimmed.contains("Server listening") {
        println!("  Power is listening");
    } else {
        // Power might not have started yet or crashed — check process list.
        let ps_result = sandbox
            .exec("/bin/sh", &["-c", "ps 2>/dev/null || true"])
            .await?;
        if !log_trimmed.is_empty() {
            println!(
                "  Power log: {}",
                log_trimmed.chars().take(300).collect::<String>()
            );
        }
        // Don't fail here — wait_for_ready will handle the timeout.
        eprintln!(
            "  Process list: {}",
            ps_result
                .stdout
                .trim()
                .chars()
                .take(200)
                .collect::<String>()
        );
    }

    if has_model && log_trimmed.contains("Loaded model registry") {
        println!("  Model registry loaded");
    }

    // Wait for Power to be ready.
    if let Err(e) = wait_for_ready(&sandbox, Duration::from_secs(60)).await {
        let log = sandbox
            .exec(
                "/bin/sh",
                &["-c", "cat /tmp/power.log 2>/dev/null | tail -30"],
            )
            .await
            .map(|r| r.stdout)
            .unwrap_or_default();
        eprintln!("Power log:\n{log}");

        let debug = sandbox
            .exec("/bin/sh", &["-c", "ls -la /tmp/httpreq 2>&1; echo '---'; wget -q -O - -T 3 http://127.0.0.1:11434/health 2>&1"])
            .await
            .map(|r| format!("stdout={} stderr={}", r.stdout, r.stderr))
            .unwrap_or_else(|e| format!("debug failed: {e}"));
        eprintln!("Debug:\n{debug}");

        sandbox.stop().await?;
        return Err(e);
    }

    // 6. Run API tests (and inference test if model is available).
    let result = run_tests(&sandbox, has_model).await;

    // 7. Stop sandbox (always, even on test failure).
    match result {
        Ok(passed) => {
            let total = if has_model { 7 } else { 6 };
            println!("\nAll {passed}/{total} tests passed!");
            println!("Stopping sandbox...");
            sandbox.stop().await?;
            println!("Done.");
            Ok(())
        }
        Err(e) => {
            eprintln!("\nTest failed: {e}");
            // Dump power log on failure for debugging.
            let log = sandbox
                .exec(
                    "/bin/sh",
                    &["-c", "cat /tmp/power.log 2>/dev/null | tail -30"],
                )
                .await
                .map(|r| r.stdout)
                .unwrap_or_default();
            eprintln!("Power log:\n{log}");
            println!("Stopping sandbox...");
            sandbox.stop().await?;
            Err(e)
        }
    }
}

/// Execute a shell command inside the sandbox with retry on transient errors.
async fn exec_retry(
    sandbox: &a3s_box_sdk::Sandbox,
    cmd: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for attempt in 0..3 {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        let result = sandbox.exec("/bin/sh", &["-c", cmd]).await?;
        if result.stderr.contains("No child process") {
            continue;
        }
        if result.exit_code == 0 {
            return Ok(());
        }
        if attempt == 2 {
            return Err(format!(
                "Command failed after 3 attempts: {cmd}\nexit={} stdout={} stderr={}",
                result.exit_code, result.stdout, result.stderr
            )
            .into());
        }
    }
    Err("Command failed after 3 retries (No child process)".into())
}
