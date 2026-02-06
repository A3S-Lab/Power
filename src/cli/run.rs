use std::io::{self, BufRead, Write};

use futures::StreamExt;

use crate::backend::types::{ChatMessage, ChatRequest};
use crate::backend::BackendRegistry;
use crate::error::Result;
use crate::model::registry::ModelRegistry;

/// Execute the `run` command: load a model and start interactive chat.
pub async fn execute(
    model: &str,
    prompt: Option<&str>,
    registry: &ModelRegistry,
    backends: &BackendRegistry,
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
            content: prompt_text.to_string(),
        }];

        let request = ChatRequest {
            messages,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            stream: true,
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
        interactive_chat(&manifest.name, &backend).await;
    }

    let _ = backend.unload(&manifest.name).await;
    Ok(())
}

async fn interactive_chat(model_name: &str, backend: &std::sync::Arc<dyn crate::backend::Backend>) {
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
            content: input.to_string(),
        });

        let request = ChatRequest {
            messages: messages.clone(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            stream: true,
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
                content: assistant_response,
            });
        }
    }

    println!("Goodbye!");
}
