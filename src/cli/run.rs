use std::io::{self, BufRead, Write};

use futures::StreamExt;

use crate::backend::types::{ChatMessage, ChatRequest, MessageContent};
use crate::backend::BackendRegistry;
use crate::error::Result;
use crate::model::registry::ModelRegistry;

/// Generation parameters passed from CLI flags.
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub num_predict: Option<u32>,
    pub num_ctx: Option<u32>,
    pub repeat_penalty: Option<f32>,
    pub seed: Option<i64>,
    /// Response format: "json" for JSON-constrained output.
    pub format: Option<String>,
    /// Override the system prompt for this session.
    pub system: Option<String>,
    /// Override the chat template.
    pub template: Option<String>,
    /// How long to keep the model loaded after the request (e.g. "5m", "-1").
    pub keep_alive: Option<String>,
    /// Show timing and token statistics after generation.
    pub verbose: bool,
    /// Skip TLS verification (reserved for future registry operations).
    pub insecure: bool,
}

/// Execute the `run` command: load a model and start interactive chat.
pub async fn execute(
    model: &str,
    prompt: Option<&str>,
    registry: &ModelRegistry,
    backends: &BackendRegistry,
) -> Result<()> {
    execute_with_options(model, prompt, registry, backends, RunOptions::default()).await
}

/// Build the response_format value from the CLI --format flag.
fn parse_format_flag(format: &str) -> Option<serde_json::Value> {
    match format {
        "json" => Some(serde_json::Value::String("json".to_string())),
        other => {
            // Try to parse as a JSON schema object
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(other) {
                Some(v)
            } else {
                eprintln!("Warning: unrecognized format '{other}', ignoring");
                None
            }
        }
    }
}

/// Print verbose timing statistics after generation.
fn print_verbose_stats(stats: &GenerationStats) {
    eprintln!();
    if let Some(count) = stats.prompt_eval_count {
        let rate = if let Some(dur) = stats.prompt_eval_duration_ns {
            if dur > 0 {
                format!(
                    " ({:.2} tokens/s)",
                    count as f64 / (dur as f64 / 1_000_000_000.0)
                )
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        eprintln!("prompt eval count:    {count} token(s){rate}");
    }
    if let Some(dur) = stats.prompt_eval_duration_ns {
        eprintln!(
            "prompt eval duration: {:.2}ms",
            dur as f64 / 1_000_000.0
        );
    }
    eprintln!("eval count:           {} token(s)", stats.eval_count);
    let total_ms = stats.total_duration.as_secs_f64() * 1000.0;
    eprintln!("total duration:       {total_ms:.2}ms");
    if stats.eval_count > 0 {
        let eval_rate =
            stats.eval_count as f64 / stats.total_duration.as_secs_f64();
        eprintln!("eval rate:            {eval_rate:.2} tokens/s");
    }
}

/// Accumulated statistics from a streaming generation.
#[derive(Debug, Default)]
struct GenerationStats {
    eval_count: u32,
    prompt_eval_count: Option<u32>,
    prompt_eval_duration_ns: Option<u64>,
    total_duration: std::time::Duration,
}

/// Execute the `run` command with generation options.
pub async fn execute_with_options(
    model: &str,
    prompt: Option<&str>,
    registry: &ModelRegistry,
    backends: &BackendRegistry,
    options: RunOptions,
) -> Result<()> {
    let manifest = match registry.get(model) {
        Ok(m) => m,
        Err(_) => {
            println!("Model '{model}' not found locally.");
            println!("Use `a3s-power pull {model}` to download it first.");
            return Ok(());
        }
    };

    let backend = backends.find_for_format(&manifest.format)?;
    tracing::info!(
        model = %manifest.name,
        backend = backend.name(),
        "Selected backend for model"
    );

    if options.insecure {
        tracing::info!("TLS verification disabled (--insecure)");
    }

    println!("Loading model '{}'...", manifest.name);
    if let Err(e) = backend.load(&manifest).await {
        println!("Failed to load model: {e}");
        return Ok(());
    }

    // Parse --format flag into response_format value
    let response_format = options
        .format
        .as_deref()
        .and_then(parse_format_flag);

    if let Some(prompt_text) = prompt {
        // Non-interactive: send a single prompt and print the streamed response
        let mut messages = Vec::new();

        // Prepend system message if --system is provided
        if let Some(ref sys) = options.system {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text(sys.clone()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
            });
        }

        messages.push(ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(prompt_text.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        });

        let request = ChatRequest {
            messages,
            temperature: options.temperature,
            top_p: options.top_p,
            max_tokens: options.num_predict,
            stop: None,
            stream: true,
            top_k: options.top_k,
            min_p: None,
            repeat_penalty: options.repeat_penalty,
            frequency_penalty: None,
            presence_penalty: None,
            seed: options.seed,
            num_ctx: options.num_ctx,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tfs_z: None,
            typical_p: None,
            response_format: response_format.clone(),
            tools: None,
            tool_choice: None,
            repeat_last_n: None,
            penalize_newline: None,
            num_batch: None,
            num_thread: None,
            num_thread_batch: None,
            flash_attention: None,
            num_gpu: None,
            main_gpu: None,
            use_mmap: None,
            use_mlock: None,
        };

        let start = std::time::Instant::now();
        let mut stats = GenerationStats::default();

        match backend.chat(&manifest.name, request).await {
            Ok(mut stream) => {
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            if !c.content.is_empty() {
                                print!("{}", c.content);
                                io::stdout().flush().ok();
                                stats.eval_count += 1;
                            }
                            if c.prompt_tokens.is_some() {
                                stats.prompt_eval_count = c.prompt_tokens;
                            }
                            if c.prompt_eval_duration_ns.is_some() {
                                stats.prompt_eval_duration_ns = c.prompt_eval_duration_ns;
                            }
                            if c.done {
                                println!();
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("\nError during generation: {e}");
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Error: {e}");
            }
        }

        stats.total_duration = start.elapsed();
        if options.verbose {
            print_verbose_stats(&stats);
        }
    } else {
        // Interactive chat mode
        interactive_chat(&manifest.name, &backend, &options, response_format).await;
    }

    let _ = backend.unload(&manifest.name).await;
    Ok(())
}

async fn interactive_chat(
    model_name: &str,
    backend: &std::sync::Arc<dyn crate::backend::Backend>,
    options: &RunOptions,
    response_format: Option<serde_json::Value>,
) {
    let mut messages: Vec<ChatMessage> = Vec::new();
    let stdin = io::stdin();

    // Prepend system message if --system is provided
    if let Some(ref sys) = options.system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: MessageContent::Text(sys.clone()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        });
    }

    println!("Interactive chat with '{model_name}' (type /exit to quit)\n");

    loop {
        print!(">>> ");
        io::stdout().flush().ok();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).is_err() || input.is_empty() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/exit" || input == "/quit" {
            break;
        }

        messages.push(ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(input.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        });

        let request = ChatRequest {
            messages: messages.clone(),
            temperature: options.temperature,
            top_p: options.top_p,
            max_tokens: options.num_predict,
            stop: None,
            stream: true,
            top_k: options.top_k,
            min_p: None,
            repeat_penalty: options.repeat_penalty,
            frequency_penalty: None,
            presence_penalty: None,
            seed: options.seed,
            num_ctx: options.num_ctx,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tfs_z: None,
            typical_p: None,
            response_format: response_format.clone(),
            tools: None,
            tool_choice: None,
            repeat_last_n: None,
            penalize_newline: None,
            num_batch: None,
            num_thread: None,
            num_thread_batch: None,
            flash_attention: None,
            num_gpu: None,
            main_gpu: None,
            use_mmap: None,
            use_mlock: None,
        };

        let mut assistant_response = String::new();
        let start = std::time::Instant::now();
        let mut stats = GenerationStats::default();

        match backend.chat(model_name, request).await {
            Ok(mut stream) => {
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            if !c.content.is_empty() {
                                print!("{}", c.content);
                                io::stdout().flush().ok();
                                assistant_response.push_str(&c.content);
                                stats.eval_count += 1;
                            }
                            if c.prompt_tokens.is_some() {
                                stats.prompt_eval_count = c.prompt_tokens;
                            }
                            if c.prompt_eval_duration_ns.is_some() {
                                stats.prompt_eval_duration_ns = c.prompt_eval_duration_ns;
                            }
                            if c.done {
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("\nError: {e}");
                            break;
                        }
                    }
                }
                println!("\n");
            }
            Err(e) => {
                eprintln!("Error: {e}\n");
            }
        }

        stats.total_duration = start.elapsed();
        if options.verbose {
            print_verbose_stats(&stats);
        }

        if !assistant_response.is_empty() {
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: MessageContent::Text(assistant_response),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
            });
        }
    }

    println!("Goodbye!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_options_default() {
        let opts = RunOptions::default();
        assert!(opts.temperature.is_none());
        assert!(opts.top_p.is_none());
        assert!(opts.top_k.is_none());
        assert!(opts.num_predict.is_none());
        assert!(opts.num_ctx.is_none());
        assert!(opts.repeat_penalty.is_none());
        assert!(opts.seed.is_none());
        assert!(opts.format.is_none());
        assert!(opts.system.is_none());
        assert!(opts.template.is_none());
        assert!(opts.keep_alive.is_none());
        assert!(!opts.verbose);
        assert!(!opts.insecure);
    }

    #[test]
    fn test_parse_format_flag_json() {
        let result = parse_format_flag("json");
        assert_eq!(result, Some(serde_json::Value::String("json".to_string())));
    }

    #[test]
    fn test_parse_format_flag_json_schema() {
        let schema = r#"{"type":"object","properties":{"name":{"type":"string"}}}"#;
        let result = parse_format_flag(schema);
        assert!(result.is_some());
        let v = result.unwrap();
        assert!(v.is_object());
        assert!(v["type"] == "object");
    }

    #[test]
    fn test_parse_format_flag_invalid() {
        let result = parse_format_flag("not-valid-format");
        assert!(result.is_none());
    }

    #[test]
    fn test_generation_stats_default() {
        let stats = GenerationStats::default();
        assert_eq!(stats.eval_count, 0);
        assert!(stats.prompt_eval_count.is_none());
        assert!(stats.prompt_eval_duration_ns.is_none());
        assert_eq!(stats.total_duration, std::time::Duration::ZERO);
    }

    #[test]
    fn test_run_options_with_all_fields() {
        let opts = RunOptions {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            num_predict: Some(100),
            num_ctx: Some(2048),
            repeat_penalty: Some(1.1),
            seed: Some(42),
            format: Some("json".to_string()),
            system: Some("You are helpful.".to_string()),
            template: Some("custom".to_string()),
            keep_alive: Some("10m".to_string()),
            verbose: true,
            insecure: true,
        };
        assert_eq!(opts.temperature, Some(0.7));
        assert_eq!(opts.format.as_deref(), Some("json"));
        assert_eq!(opts.system.as_deref(), Some("You are helpful."));
        assert_eq!(opts.template.as_deref(), Some("custom"));
        assert_eq!(opts.keep_alive.as_deref(), Some("10m"));
        assert!(opts.verbose);
        assert!(opts.insecure);
    }
}
