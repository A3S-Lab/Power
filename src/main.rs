use clap::Parser;
use tracing_subscriber::EnvFilter;

use a3s_power::cli::{ChatArgs, Cli, Command, ModelsCommand, PsArgs};
use a3s_power::config::PowerConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        // No subcommand = serve (backward compat)
        None => {
            init_tracing();
            let config = PowerConfig::load()?;
            a3s_power::server::start(config).await?;
        }
        Some(Command::Serve(args)) => {
            init_tracing();
            let mut config = if let Some(ref path) = args.config {
                PowerConfig::load_from(path)?
            } else {
                PowerConfig::load()?
            };
            config.host = args.host;
            config.port = args.port;
            a3s_power::server::start(config).await?;
        }
        Some(Command::Models(cmd)) => {
            run_models_command(cmd).await?;
        }
        Some(Command::Chat(args)) => {
            run_chat(args).await?;
        }
        Some(Command::Ps(args)) => {
            run_ps(args).await?;
        }
    }

    Ok(())
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}

// ── Models subcommands ───────────────────────────────────────────────────────

async fn run_models_command(cmd: ModelsCommand) -> anyhow::Result<()> {
    match cmd {
        ModelsCommand::List(args) => {
            let resp: serde_json::Value = http_get(&format!("{}/v1/models", args.url)).await?;
            if let Some(data) = resp.get("data").and_then(|d| d.as_array()) {
                if data.is_empty() {
                    println!("No models registered.");
                    return Ok(());
                }
                println!("{:<40} {:<12} {:<10}", "NAME", "FORMAT", "SIZE");
                for m in data {
                    let name = m.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                    let format = m.get("format").and_then(|v| v.as_str()).unwrap_or("?");
                    let size = m
                        .get("size")
                        .and_then(|v| v.as_u64())
                        .map(|s| format_bytes(s))
                        .unwrap_or_else(|| "?".to_string());
                    println!("{:<40} {:<12} {:<10}", name, format, size);
                }
            } else {
                println!("{}", serde_json::to_string_pretty(&resp)?);
            }
            Ok(())
        }
        ModelsCommand::Pull(args) => {
            println!("Pulling {}...", args.name);
            let body = serde_json::json!({
                "name": args.name,
                "force": args.force,
            });
            let url = format!("{}/v1/models/pull", args.url);
            let client = reqwest::Client::new();
            let resp = client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to connect: {e}"))?;

            if !resp.status().is_success() {
                let text = resp.text().await.unwrap_or_default();
                anyhow::bail!("Pull failed: {text}");
            }

            // Stream SSE events
            use futures::StreamExt;
            let mut stream = resp.bytes_stream();
            let mut buf = String::new();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| anyhow::anyhow!("Stream error: {e}"))?;
                buf.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(pos) = buf.find("\n\n") {
                    let event = buf[..pos].to_string();
                    buf = buf[pos + 2..].to_string();
                    if let Some(data) = event.strip_prefix("data: ") {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                            let status = v.get("status").and_then(|s| s.as_str()).unwrap_or("");
                            match status {
                                "downloading" => {
                                    let completed =
                                        v.get("completed").and_then(|c| c.as_u64()).unwrap_or(0);
                                    let total =
                                        v.get("total").and_then(|t| t.as_u64()).unwrap_or(1);
                                    let pct = if total > 0 {
                                        completed * 100 / total
                                    } else {
                                        0
                                    };
                                    print!(
                                        "\rDownloading... {pct}% ({}/{})",
                                        format_bytes(completed),
                                        format_bytes(total)
                                    );
                                }
                                "success" => {
                                    println!("\rPull complete: {}", args.name);
                                }
                                _ => {
                                    println!("{status}");
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        }
        ModelsCommand::Remove(args) => {
            let url = format!("{}/v1/models/{}", args.url, args.name);
            let client = reqwest::Client::new();
            let resp = client
                .delete(&url)
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to connect: {e}"))?;

            if resp.status().is_success() {
                println!("Deleted model: {}", args.name);
            } else {
                let text = resp.text().await.unwrap_or_default();
                anyhow::bail!("Delete failed: {text}");
            }
            Ok(())
        }
        ModelsCommand::Show(args) => {
            let resp: serde_json::Value =
                http_get(&format!("{}/v1/models/{}", args.url, args.name)).await?;
            println!("{}", serde_json::to_string_pretty(&resp)?);
            Ok(())
        }
    }
}

// ── Chat ─────────────────────────────────────────────────────────────────────

async fn run_chat(args: ChatArgs) -> anyhow::Result<()> {
    use std::io::{self, BufRead, Write};

    println!("Chatting with {} (Ctrl+D to exit)", args.model);
    println!();

    let client = reqwest::Client::new();
    let mut messages: Vec<serde_json::Value> = Vec::new();

    if let Some(ref sys) = args.system {
        messages.push(serde_json::json!({"role": "system", "content": sys}));
    }

    let stdin = io::stdin();
    loop {
        print!(">>> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        messages.push(serde_json::json!({"role": "user", "content": input}));

        let body = serde_json::json!({
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "stream": true,
        });

        let resp = client
            .post(&format!("{}/v1/chat/completions", args.url))
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to connect: {e}"))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            eprintln!("Error: {text}");
            messages.pop(); // remove failed user message
            continue;
        }

        // Stream SSE response
        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        let mut assistant_content = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| anyhow::anyhow!("Stream error: {e}"))?;
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buf.find("\n\n") {
                let event = buf[..pos].to_string();
                buf = buf[pos + 2..].to_string();
                for line in event.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            continue;
                        }
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                            if let Some(delta) = v
                                .pointer("/choices/0/delta/content")
                                .and_then(|c| c.as_str())
                            {
                                print!("{delta}");
                                io::stdout().flush()?;
                                assistant_content.push_str(delta);
                            }
                        }
                    }
                }
            }
        }
        println!("\n");

        messages.push(serde_json::json!({"role": "assistant", "content": assistant_content}));
    }

    Ok(())
}

// ── Ps ───────────────────────────────────────────────────────────────────────

async fn run_ps(args: PsArgs) -> anyhow::Result<()> {
    let resp: serde_json::Value = http_get(&format!("{}/v1/models", args.url)).await?;
    if let Some(data) = resp.get("data").and_then(|d| d.as_array()) {
        let loaded: Vec<_> = data
            .iter()
            .filter(|m| m.get("loaded").and_then(|v| v.as_bool()).unwrap_or(false))
            .collect();
        if loaded.is_empty() {
            println!("No models currently loaded.");
            return Ok(());
        }
        println!("{:<40} {:<12} {:<10}", "NAME", "FORMAT", "SIZE");
        for m in loaded {
            let name = m.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let format = m.get("format").and_then(|v| v.as_str()).unwrap_or("?");
            let size = m
                .get("size")
                .and_then(|v| v.as_u64())
                .map(|s| format_bytes(s))
                .unwrap_or_else(|| "?".to_string());
            println!("{:<40} {:<12} {:<10}", name, format, size);
        }
    } else {
        println!("{}", serde_json::to_string_pretty(&resp)?);
    }
    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────────────

async fn http_get(url: &str) -> anyhow::Result<serde_json::Value> {
    let client = reqwest::Client::new();
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to connect to {url}: {e}"))?;
    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("HTTP {}: {text}", url);
    }
    let body = resp
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("Invalid JSON from {url}: {e}"))?;
    Ok(body)
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}
