pub mod delete;
pub mod list;
pub mod ps;
pub mod pull;
pub mod push;
pub mod run;
pub mod serve;
pub mod show;
pub mod stop;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// A3S Power - Local model management and serving
#[derive(Debug, Parser)]
#[command(name = "a3s-power", version, about, disable_help_subcommand = true)]
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

        /// Response format: "json" for JSON output
        #[arg(long)]
        format: Option<String>,

        /// Override the system prompt
        #[arg(long)]
        system: Option<String>,

        /// Override the chat template
        #[arg(long)]
        template: Option<String>,

        /// How long to keep the model loaded (e.g. "5m", "1h", "0", "-1")
        #[arg(long, allow_hyphen_values = true)]
        keep_alive: Option<String>,

        /// Show timing and token statistics after generation
        #[arg(long)]
        verbose: bool,

        /// Skip TLS verification for registry operations
        #[arg(long)]
        insecure: bool,
    },

    /// Download a model
    Pull {
        /// Model name or URL to download
        model: String,

        /// Skip TLS verification for registry operations
        #[arg(long)]
        insecure: bool,
    },

    /// List locally available models
    List,

    /// Show details about a model
    Show {
        /// Model name to inspect
        model: String,

        /// Show verbose GGUF metadata and tensor information
        #[arg(long)]
        verbose: bool,
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
        #[arg(long, default_value_t = 11434)]
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

        /// Skip TLS verification for registry operations
        #[arg(long)]
        insecure: bool,
    },

    /// Copy/alias a model to a new name
    Cp {
        /// Source model name
        source: String,

        /// Destination model name
        destination: String,
    },

    /// List running (loaded) models on the server
    Ps,

    /// Stop (unload) a running model from the server
    Stop {
        /// Model name to stop/unload
        model: String,
    },

    /// Update a3s-power to the latest version
    Update,

    /// Show help for a command
    Help {
        /// Command name to get help for (e.g. "run", "pull")
        command: Option<String>,
    },
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
            Commands::Pull { model, insecure } => {
                assert_eq!(model, "llama3:3b");
                assert!(!insecure);
            }
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
            Commands::Push {
                model,
                destination,
                insecure,
            } => {
                assert_eq!(model, "llama3");
                assert_eq!(destination, "https://registry.example.com");
                assert!(!insecure);
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
            Commands::Show { model, verbose } => {
                assert_eq!(model, "llama3");
                assert!(!verbose);
            }
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
                assert_eq!(port, 11434);
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

    #[test]
    fn test_parse_ps_command() {
        let cli = Cli::parse_from(["a3s-power", "ps"]);
        assert!(matches!(cli.command, Commands::Ps));
    }

    #[test]
    fn test_parse_stop_command() {
        let cli = Cli::parse_from(["a3s-power", "stop", "llama3"]);
        match cli.command {
            Commands::Stop { model } => assert_eq!(model, "llama3"),
            _ => panic!("Expected Stop command"),
        }
    }

    #[test]
    fn test_parse_run_with_format() {
        let cli = Cli::parse_from(["a3s-power", "run", "llama3", "--format", "json"]);
        match cli.command {
            Commands::Run { model, format, .. } => {
                assert_eq!(model, "llama3");
                assert_eq!(format.as_deref(), Some("json"));
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_parse_run_with_system() {
        let cli = Cli::parse_from([
            "a3s-power",
            "run",
            "llama3",
            "--system",
            "You are a helpful assistant.",
        ]);
        match cli.command {
            Commands::Run { model, system, .. } => {
                assert_eq!(model, "llama3");
                assert_eq!(system.as_deref(), Some("You are a helpful assistant."));
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_parse_run_with_template() {
        let cli = Cli::parse_from([
            "a3s-power",
            "run",
            "llama3",
            "--template",
            "{{ .System }}\n{{ .Prompt }}",
        ]);
        match cli.command {
            Commands::Run {
                model, template, ..
            } => {
                assert_eq!(model, "llama3");
                assert_eq!(
                    template.as_deref(),
                    Some("{{ .System }}\n{{ .Prompt }}")
                );
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_parse_run_with_keep_alive() {
        let cli = Cli::parse_from(["a3s-power", "run", "llama3", "--keep-alive", "10m"]);
        match cli.command {
            Commands::Run {
                model, keep_alive, ..
            } => {
                assert_eq!(model, "llama3");
                assert_eq!(keep_alive.as_deref(), Some("10m"));
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_parse_run_with_verbose() {
        let cli = Cli::parse_from(["a3s-power", "run", "llama3", "--verbose"]);
        match cli.command {
            Commands::Run { model, verbose, .. } => {
                assert_eq!(model, "llama3");
                assert!(verbose);
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_parse_run_verbose_default_false() {
        let cli = Cli::parse_from(["a3s-power", "run", "llama3"]);
        match cli.command {
            Commands::Run { verbose, .. } => {
                assert!(!verbose);
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_parse_run_with_insecure() {
        let cli = Cli::parse_from(["a3s-power", "run", "llama3", "--insecure"]);
        match cli.command {
            Commands::Run {
                model, insecure, ..
            } => {
                assert_eq!(model, "llama3");
                assert!(insecure);
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_parse_run_insecure_default_false() {
        let cli = Cli::parse_from(["a3s-power", "run", "llama3"]);
        match cli.command {
            Commands::Run { insecure, .. } => {
                assert!(!insecure);
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_parse_run_all_new_options() {
        let cli = Cli::parse_from([
            "a3s-power",
            "run",
            "llama3",
            "--prompt",
            "hello",
            "--format",
            "json",
            "--system",
            "Be concise.",
            "--template",
            "custom",
            "--keep-alive",
            "-1",
            "--verbose",
            "--insecure",
            "--temperature",
            "0.5",
        ]);
        match cli.command {
            Commands::Run {
                model,
                prompt,
                format,
                system,
                template,
                keep_alive,
                verbose,
                insecure,
                temperature,
                ..
            } => {
                assert_eq!(model, "llama3");
                assert_eq!(prompt.as_deref(), Some("hello"));
                assert_eq!(format.as_deref(), Some("json"));
                assert_eq!(system.as_deref(), Some("Be concise."));
                assert_eq!(template.as_deref(), Some("custom"));
                assert_eq!(keep_alive.as_deref(), Some("-1"));
                assert!(verbose);
                assert!(insecure);
                assert_eq!(temperature, Some(0.5));
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_parse_pull_with_insecure() {
        let cli = Cli::parse_from(["a3s-power", "pull", "llama3", "--insecure"]);
        match cli.command {
            Commands::Pull { model, insecure } => {
                assert_eq!(model, "llama3");
                assert!(insecure);
            }
            _ => panic!("Expected Pull command"),
        }
    }

    #[test]
    fn test_parse_push_with_insecure() {
        let cli = Cli::parse_from([
            "a3s-power",
            "push",
            "llama3",
            "--destination",
            "https://example.com",
            "--insecure",
        ]);
        match cli.command {
            Commands::Push {
                model,
                destination,
                insecure,
            } => {
                assert_eq!(model, "llama3");
                assert_eq!(destination, "https://example.com");
                assert!(insecure);
            }
            _ => panic!("Expected Push command"),
        }
    }

    #[test]
    fn test_parse_show_with_verbose() {
        let cli = Cli::parse_from(["a3s-power", "show", "llama3", "--verbose"]);
        match cli.command {
            Commands::Show { model, verbose } => {
                assert_eq!(model, "llama3");
                assert!(verbose);
            }
            _ => panic!("Expected Show command"),
        }
    }

    #[test]
    fn test_parse_help_command_no_args() {
        let cli = Cli::parse_from(["a3s-power", "help"]);
        match cli.command {
            Commands::Help { command } => {
                assert!(command.is_none());
            }
            _ => panic!("Expected Help command"),
        }
    }

    #[test]
    fn test_parse_help_command_with_subcommand() {
        let cli = Cli::parse_from(["a3s-power", "help", "run"]);
        match cli.command {
            Commands::Help { command } => {
                assert_eq!(command.as_deref(), Some("run"));
            }
            _ => panic!("Expected Help command"),
        }
    }
}
