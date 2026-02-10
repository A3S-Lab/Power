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
    let mut manifest = match registry.get(model) {
        Ok(m) => m,
        Err(_) => {
            println!("Model '{model}' not found locally.");
            println!("Use `a3s-power pull {model}` to download it first.");
            return Ok(());
        }
    };

    // Apply --template override to the manifest so the backend uses it
    if let Some(ref tmpl) = options.template {
        manifest.template_override = Some(tmpl.clone());
    }

    let backend = backends.find_for_format(&manifest.format)?;
    tracing::info!(
        model = %manifest.name,
        backend = backend.name(),
        "Selected backend for model"
    );

    if options.insecure {
        tracing::info!("TLS verification disabled (--insecure)");
    }

    if let Some(ref ka) = options.keep_alive {
        tracing::info!(keep_alive = %ka, "Keep-alive set (applies to server mode)");
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

    // Mutable session state for /set commands
    let mut session_system = options.system.clone();
    let mut session_verbose = options.verbose;
    let mut session_format = response_format;
    let mut session_temperature = options.temperature;
    let mut session_top_p = options.top_p;
    let mut session_top_k = options.top_k;
    let mut session_num_predict = options.num_predict;
    let mut session_num_ctx = options.num_ctx;
    let mut session_repeat_penalty = options.repeat_penalty;
    let mut session_seed = options.seed;
    let mut wordwrap = true;

    // Prepend system message if --system is provided
    if let Some(ref sys) = session_system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: MessageContent::Text(sys.clone()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        });
    }

    println!("Interactive chat with '{model_name}' (type /help for commands)\n");

    loop {
        print!(">>> ");
        io::stdout().flush().ok();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).is_err() || input.is_empty() {
            break;
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Handle slash commands
        if trimmed == "/exit" || trimmed == "/quit" {
            break;
        }
        if trimmed == "/help" || trimmed == "/?" {
            print_help();
            continue;
        }
        if trimmed == "/clear" {
            // Keep system message if present, clear the rest
            let system_msg: Vec<_> = messages
                .iter()
                .filter(|m| m.role == "system")
                .cloned()
                .collect();
            messages = system_msg;
            println!("Conversation cleared.\n");
            continue;
        }

        // /show commands
        if trimmed == "/show" || trimmed == "/show info" {
            print_show_info(model_name, &messages, &session_system, &session_format);
            continue;
        }
        if trimmed == "/show system" {
            match &session_system {
                Some(sys) => println!("{sys}\n"),
                None => println!("(no system prompt set)\n"),
            }
            continue;
        }
        if trimmed == "/show template" {
            if let Some(ref tmpl) = options.template {
                println!("{tmpl}\n");
            } else {
                println!("(using model default template)\n");
            }
            continue;
        }
        if trimmed == "/show license" {
            println!("(license info not available in interactive mode)\n");
            continue;
        }
        if trimmed == "/show modelfile" {
            println!("(modelfile not available in interactive mode)\n");
            continue;
        }
        if trimmed == "/show parameters" {
            println!("Current parameters:");
            println!("  temperature:    {}", fmt_opt(session_temperature));
            println!("  top_p:          {}", fmt_opt(session_top_p));
            println!("  top_k:          {}", fmt_opt_i32(session_top_k));
            println!("  num_predict:    {}", fmt_opt_u32(session_num_predict));
            println!("  num_ctx:        {}", fmt_opt_u32(session_num_ctx));
            println!("  repeat_penalty: {}", fmt_opt(session_repeat_penalty));
            println!("  seed:           {}", fmt_opt_i64(session_seed));
            println!("  verbose:        {session_verbose}");
            println!("  wordwrap:       {wordwrap}");
            if let Some(ref fmt) = session_format {
                println!("  format:         {fmt}");
            }
            println!();
            continue;
        }

        // /set commands
        if let Some(rest) = trimmed.strip_prefix("/set ") {
            let rest = rest.trim();
            if rest == "verbose" {
                session_verbose = !session_verbose;
                println!("Verbose mode: {}\n", if session_verbose { "on" } else { "off" });
            } else if rest == "nowordwrap" {
                wordwrap = false;
                println!("Word wrap: off\n");
            } else if rest == "wordwrap" {
                wordwrap = true;
                println!("Word wrap: on\n");
            } else if let Some(sys) = rest.strip_prefix("system ") {
                let sys = sys.trim().trim_matches('"').to_string();
                // Remove old system message and add new one
                messages.retain(|m| m.role != "system");
                messages.insert(0, ChatMessage {
                    role: "system".to_string(),
                    content: MessageContent::Text(sys.clone()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    images: None,
                });
                session_system = Some(sys);
                println!("System prompt updated.\n");
            } else if let Some(fmt) = rest.strip_prefix("format ") {
                let fmt = fmt.trim();
                session_format = parse_format_flag(fmt);
                println!("Format set to: {fmt}\n");
            } else if let Some(param) = rest.strip_prefix("parameter ") {
                if let Some((key, val)) = param.trim().split_once(char::is_whitespace) {
                    match key.trim() {
                        "temperature" => {
                            if let Ok(v) = val.trim().parse::<f32>() {
                                session_temperature = Some(v);
                                println!("temperature = {v}\n");
                            } else {
                                println!("Invalid value for temperature\n");
                            }
                        }
                        "top_p" => {
                            if let Ok(v) = val.trim().parse::<f32>() {
                                session_top_p = Some(v);
                                println!("top_p = {v}\n");
                            } else {
                                println!("Invalid value for top_p\n");
                            }
                        }
                        "top_k" => {
                            if let Ok(v) = val.trim().parse::<i32>() {
                                session_top_k = Some(v);
                                println!("top_k = {v}\n");
                            } else {
                                println!("Invalid value for top_k\n");
                            }
                        }
                        "num_predict" => {
                            if let Ok(v) = val.trim().parse::<u32>() {
                                session_num_predict = Some(v);
                                println!("num_predict = {v}\n");
                            } else {
                                println!("Invalid value for num_predict\n");
                            }
                        }
                        "num_ctx" => {
                            if let Ok(v) = val.trim().parse::<u32>() {
                                session_num_ctx = Some(v);
                                println!("num_ctx = {v}\n");
                            } else {
                                println!("Invalid value for num_ctx\n");
                            }
                        }
                        "repeat_penalty" => {
                            if let Ok(v) = val.trim().parse::<f32>() {
                                session_repeat_penalty = Some(v);
                                println!("repeat_penalty = {v}\n");
                            } else {
                                println!("Invalid value for repeat_penalty\n");
                            }
                        }
                        "seed" => {
                            if let Ok(v) = val.trim().parse::<i64>() {
                                session_seed = Some(v);
                                println!("seed = {v}\n");
                            } else {
                                println!("Invalid value for seed\n");
                            }
                        }
                        other => {
                            println!("Unknown parameter: {other}\n");
                        }
                    }
                } else {
                    println!("Usage: /set parameter <key> <value>\n");
                }
            } else {
                println!("Unknown /set command. Available: verbose, nowordwrap, wordwrap, system, format, parameter\n");
            }
            continue;
        }

        // Handle multi-line input with """ delimiter
        let user_input = if trimmed == "\"\"\"" {
            let mut multiline = String::new();
            loop {
                print!("... ");
                io::stdout().flush().ok();
                let mut line = String::new();
                if stdin.lock().read_line(&mut line).is_err() || line.is_empty() {
                    break;
                }
                if line.trim() == "\"\"\"" {
                    break;
                }
                multiline.push_str(&line);
            }
            let result = multiline.trim().to_string();
            if result.is_empty() {
                continue;
            }
            result
        } else {
            trimmed.to_string()
        };

        messages.push(ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(user_input),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        });

        let request = ChatRequest {
            messages: messages.clone(),
            temperature: session_temperature,
            top_p: session_top_p,
            max_tokens: session_num_predict,
            stop: None,
            stream: true,
            top_k: session_top_k,
            min_p: None,
            repeat_penalty: session_repeat_penalty,
            frequency_penalty: None,
            presence_penalty: None,
            seed: session_seed,
            num_ctx: session_num_ctx,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tfs_z: None,
            typical_p: None,
            response_format: session_format.clone(),
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
        if session_verbose {
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

/// Print help for interactive mode commands.
fn print_help() {
    println!("Available commands:");
    println!("  /help                          Show this help message");
    println!("  /clear                         Clear conversation history");
    println!("  /show                          Show current model info");
    println!("  /show info                     Show current model info");
    println!("  /show system                   Show system prompt");
    println!("  /show template                 Show chat template");
    println!("  /show parameters               Show current parameters");
    println!("  /show license                  Show model license");
    println!("  /show modelfile                Show model Modelfile");
    println!("  /set system <prompt>           Set system prompt");
    println!("  /set parameter <key> <value>   Set a generation parameter");
    println!("  /set format <json|schema>      Set output format");
    println!("  /set verbose                   Toggle verbose mode");
    println!("  /set wordwrap                  Enable word wrap");
    println!("  /set nowordwrap                Disable word wrap");
    println!("  /exit                          Exit the chat");
    println!();
    println!("Use \"\"\" to begin and end a multi-line message.");
    println!();
}

/// Print model info for /show command.
fn print_show_info(
    model_name: &str,
    messages: &[ChatMessage],
    system: &Option<String>,
    format: &Option<serde_json::Value>,
) {
    println!("Model: {model_name}");
    let user_msgs = messages.iter().filter(|m| m.role == "user").count();
    let asst_msgs = messages.iter().filter(|m| m.role == "assistant").count();
    println!("Messages: {user_msgs} user, {asst_msgs} assistant");
    if let Some(ref sys) = system {
        println!("System: {sys}");
    }
    if let Some(ref fmt) = format {
        println!("Format: {fmt}");
    }
    println!();
}

fn fmt_opt(v: Option<f32>) -> String {
    v.map_or("(default)".to_string(), |v| format!("{v}"))
}

fn fmt_opt_i32(v: Option<i32>) -> String {
    v.map_or("(default)".to_string(), |v| format!("{v}"))
}

fn fmt_opt_u32(v: Option<u32>) -> String {
    v.map_or("(default)".to_string(), |v| format!("{v}"))
}

fn fmt_opt_i64(v: Option<i64>) -> String {
    v.map_or("(default)".to_string(), |v| format!("{v}"))
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

    #[test]
    fn test_fmt_opt_some() {
        assert_eq!(fmt_opt(Some(0.7)), "0.7");
        assert_eq!(fmt_opt_i32(Some(40)), "40");
        assert_eq!(fmt_opt_u32(Some(100)), "100");
        assert_eq!(fmt_opt_i64(Some(-1)), "-1");
    }

    #[test]
    fn test_fmt_opt_none() {
        assert_eq!(fmt_opt(None), "(default)");
        assert_eq!(fmt_opt_i32(None), "(default)");
        assert_eq!(fmt_opt_u32(None), "(default)");
        assert_eq!(fmt_opt_i64(None), "(default)");
    }
}
