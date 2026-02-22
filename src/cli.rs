//! CLI argument parsing for a3s-power.
//!
//! Subcommands:
//!   serve  — Start the inference server (default if no subcommand given)
//!   models — Model management (list, pull, delete)
//!   chat   — Interactive chat with a loaded model
//!   ps     — Show loaded/running models

use clap::{Parser, Subcommand};

/// A3S Power — Privacy-preserving LLM inference for TEE environments.
#[derive(Parser, Debug)]
#[command(name = "a3s-power", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Start the inference server
    Serve(ServeArgs),

    /// Model management
    #[command(subcommand)]
    Models(ModelsCommand),

    /// Interactive chat with a model
    Chat(ChatArgs),

    /// Show loaded/running models
    Ps(PsArgs),
}

/// Arguments for the `serve` subcommand.
#[derive(Parser, Debug)]
pub struct ServeArgs {
    /// Bind address
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port
    #[arg(long, default_value = "11434")]
    pub port: u16,

    /// Config file path
    #[arg(long)]
    pub config: Option<String>,
}

/// Model management subcommands.
#[derive(Subcommand, Debug)]
pub enum ModelsCommand {
    /// List registered models
    List(ModelsListArgs),

    /// Pull a model from HuggingFace Hub
    Pull(ModelsPullArgs),

    /// Delete a model
    #[command(name = "rm")]
    Remove(ModelsRemoveArgs),

    /// Show model details
    Show(ModelsShowArgs),
}

#[derive(Parser, Debug)]
pub struct ModelsListArgs {
    /// Server URL
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    pub url: String,
}

#[derive(Parser, Debug)]
pub struct ModelsPullArgs {
    /// Model name (e.g., "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m")
    pub name: String,

    /// Server URL
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    pub url: String,

    /// Force re-download even if model exists
    #[arg(long)]
    pub force: bool,
}

#[derive(Parser, Debug)]
pub struct ModelsRemoveArgs {
    /// Model name to delete
    pub name: String,

    /// Server URL
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    pub url: String,
}

#[derive(Parser, Debug)]
pub struct ModelsShowArgs {
    /// Model name
    pub name: String,

    /// Server URL
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    pub url: String,
}

/// Arguments for the `chat` subcommand.
#[derive(Parser, Debug)]
pub struct ChatArgs {
    /// Model name to chat with
    pub model: String,

    /// Server URL
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    pub url: String,

    /// System prompt
    #[arg(long)]
    pub system: Option<String>,

    /// Temperature (0.0 = deterministic)
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,
}

/// Arguments for the `ps` subcommand.
#[derive(Parser, Debug)]
pub struct PsArgs {
    /// Server URL
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    pub url: String,
}
