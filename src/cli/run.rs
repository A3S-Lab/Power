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
    pub seed: Option<u32>,
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

    println!("Loading model '{}'...", manifest.name);
    if let Err(e) = backend.load(&manifest).await {
        println!("Failed to load model: {e}");
        return Ok(());
    }

    if let Some(prompt_text) = prompt {
        // Non-interactive: send a single prompt and print the streamed response
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(prompt_text.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        }];

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
            response_format: None,
            tools: None,
            tool_choice: None,
        };

        match backend.chat(&manifest.name, request).await {
            Ok(mut stream) => {
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            if !c.content.is_empty() {
                                print!("{}", c.content);
                                io::stdout().flush().ok();
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
    } else {
        // Interactive chat mode
        interactive_chat(&manifest.name, &backend, &options).await;
    }

    let _ = backend.unload(&manifest.name).await;
    Ok(())
}

async fn interactive_chat(
    model_name: &str,
    backend: &std::sync::Arc<dyn crate::backend::Backend>,
    options: &RunOptions,
) {
    let mut messages: Vec<ChatMessage> = Vec::new();
    let stdin = io::stdin();

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
            response_format: None,
            tools: None,
            tool_choice: None,
        };

        let mut assistant_response = String::new();

        match backend.chat(model_name, request).await {
            Ok(mut stream) => {
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            if !c.content.is_empty() {
                                print!("{}", c.content);
                                io::stdout().flush().ok();
                                assistant_response.push_str(&c.content);
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
