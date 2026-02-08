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
        seed: Option<u32>,
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
}
