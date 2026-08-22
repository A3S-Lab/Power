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
    #[arg(long)]
    pub host: Option<String>,

    /// Port
    #[arg(long)]
    pub port: Option<u16>,

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

    /// Show persisted pull progress for a model
    Status(ModelsStatusArgs),

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
pub struct ModelsStatusArgs {
    /// Model name whose persisted pull state should be shown
    pub name: String,

    /// Server URL
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    pub url: String,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_config_preserves_unspecified_bind_values() {
        let cli = Cli::try_parse_from(["a3s-power", "serve", "--config", "power.acl"])
            .expect("serve arguments should parse");
        let Some(Command::Serve(args)) = cli.command else {
            panic!("expected serve command");
        };

        assert_eq!(args.config.as_deref(), Some("power.acl"));
        assert_eq!(args.host, None);
        assert_eq!(args.port, None);
    }

    #[test]
    fn serve_cli_bind_values_remain_explicit_overrides() {
        let cli =
            Cli::try_parse_from(["a3s-power", "serve", "--host", "0.0.0.0", "--port", "18080"])
                .expect("serve arguments should parse");
        let Some(Command::Serve(args)) = cli.command else {
            panic!("expected serve command");
        };

        assert_eq!(args.host.as_deref(), Some("0.0.0.0"));
        assert_eq!(args.port, Some(18080));
    }
}
