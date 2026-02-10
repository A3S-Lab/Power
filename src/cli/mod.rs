pub mod delete;
pub mod list;
pub mod pull;
pub mod push;
pub mod run;
pub mod serve;
pub mod show;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// A3S Power - Local model management and serving
#[derive(Debug, Parser)]
#[command(name = "a3s-power", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Available CLI commands.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Pull (if needed), load, and start interactive chat with a model
    Run {
        /// Model name to run (e.g. "llama3.2:3b")
        model: String,

        /// Optional prompt to send directly instead of interactive mode
        #[arg(long)]
        prompt: Option<String>,

        /// Sampling temperature (0.0 - 2.0)
        #[arg(long)]
        temperature: Option<f32>,

        /// Top-p (nucleus) sampling threshold
        #[arg(long)]
        top_p: Option<f32>,

        /// Top-k sampling limit
        #[arg(long)]
        top_k: Option<i32>,

        /// Maximum number of tokens to generate
        #[arg(long)]
        num_predict: Option<u32>,

        /// Context window size in tokens
        #[arg(long)]
        num_ctx: Option<u32>,

        /// Repetition penalty (1.0 = disabled)
        #[arg(long)]
        repeat_penalty: Option<f32>,

        /// Random seed for reproducible output
        #[arg(long)]
        seed: Option<i64>,
    },

    /// Download a model
    Pull {
        /// Model name or URL to download
        model: String,
    },

    /// List locally available models
    List,

    /// Show details about a model
    Show {
        /// Model name to inspect
        model: String,
    },

    /// Delete a local model
    Delete {
        /// Model name to delete
        model: String,
    },

    /// Start the HTTP server
    Serve {
        /// Host address to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on
        #[arg(long, default_value_t = 11435)]
        port: u16,
    },

    /// Create a model from a Modelfile
    Create {
        /// Name for the new model
        name: String,

        /// Path to the Modelfile
        #[arg(short = 'f', long)]
        file: PathBuf,
    },

    /// Push a model to a remote registry
    Push {
        /// Model name to push (e.g. "llama3.2:3b")
        model: String,

        /// Destination registry URL
        #[arg(long)]
        destination: String,
    },

    /// Copy/alias a model to a new name
    Cp {
        /// Source model name
        source: String,

        /// Destination model name
        destination: String,
    },

    /// Update a3s-power to the latest version
    Update,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_parse_run_command() {
        let cli = Cli::parse_from(["a3s-power", "run", "llama3"]);
        match cli.command {
            Commands::Run { model, .. } => assert_eq!(model, "llama3"),
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_parse_run_with_options() {
        let cli = Cli::parse_from([
            "a3s-power",
            "run",
            "llama3",
            "--prompt",
            "hello",
            "--temperature",
            "0.7",
            "--top-k",
            "40",
            "--seed",
            "42",
        ]);
        match cli.command {
            Commands::Run {
                model,
                prompt,
                temperature,
                top_k,
                seed,
                ..
            } => {
                assert_eq!(model, "llama3");
                assert_eq!(prompt.as_deref(), Some("hello"));
                assert_eq!(temperature, Some(0.7));
                assert_eq!(top_k, Some(40));
                assert_eq!(seed, Some(42));
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_parse_pull_command() {
        let cli = Cli::parse_from(["a3s-power", "pull", "llama3:3b"]);
        match cli.command {
            Commands::Pull { model } => assert_eq!(model, "llama3:3b"),
            _ => panic!("Expected Pull command"),
        }
    }

    #[test]
    fn test_parse_push_command() {
        let cli = Cli::parse_from([
            "a3s-power",
            "push",
            "llama3",
            "--destination",
            "https://registry.example.com",
        ]);
        match cli.command {
            Commands::Push { model, destination } => {
                assert_eq!(model, "llama3");
                assert_eq!(destination, "https://registry.example.com");
            }
            _ => panic!("Expected Push command"),
        }
    }

    #[test]
    fn test_parse_cp_command() {
        let cli = Cli::parse_from(["a3s-power", "cp", "llama3", "my-llama"]);
        match cli.command {
            Commands::Cp {
                source,
                destination,
            } => {
                assert_eq!(source, "llama3");
                assert_eq!(destination, "my-llama");
            }
            _ => panic!("Expected Cp command"),
        }
    }

    #[test]
    fn test_parse_list_command() {
        let cli = Cli::parse_from(["a3s-power", "list"]);
        assert!(matches!(cli.command, Commands::List));
    }

    #[test]
    fn test_parse_show_command() {
        let cli = Cli::parse_from(["a3s-power", "show", "llama3"]);
        match cli.command {
            Commands::Show { model } => assert_eq!(model, "llama3"),
            _ => panic!("Expected Show command"),
        }
    }

    #[test]
    fn test_parse_delete_command() {
        let cli = Cli::parse_from(["a3s-power", "delete", "llama3"]);
        match cli.command {
            Commands::Delete { model } => assert_eq!(model, "llama3"),
            _ => panic!("Expected Delete command"),
        }
    }

    #[test]
    fn test_parse_serve_defaults() {
        let cli = Cli::parse_from(["a3s-power", "serve"]);
        match cli.command {
            Commands::Serve { host, port } => {
                assert_eq!(host, "127.0.0.1");
                assert_eq!(port, 11435);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_parse_serve_custom() {
        let cli = Cli::parse_from(["a3s-power", "serve", "--host", "0.0.0.0", "--port", "8080"]);
        match cli.command {
            Commands::Serve { host, port } => {
                assert_eq!(host, "0.0.0.0");
                assert_eq!(port, 8080);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_parse_create_command() {
        let cli = Cli::parse_from(["a3s-power", "create", "my-model", "-f", "Modelfile"]);
        match cli.command {
            Commands::Create { name, file } => {
                assert_eq!(name, "my-model");
                assert_eq!(file, PathBuf::from("Modelfile"));
            }
            _ => panic!("Expected Create command"),
        }
    }

    #[test]
    fn test_parse_update_command() {
        let cli = Cli::parse_from(["a3s-power", "update"]);
        assert!(matches!(cli.command, Commands::Update));
    }
}
