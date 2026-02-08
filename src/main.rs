use clap::Parser;
use tracing_subscriber::EnvFilter;

use a3s_power::backend;
use a3s_power::cli::{Cli, Commands};
use a3s_power::config::PowerConfig;
use a3s_power::dirs;
use a3s_power::model::registry::ModelRegistry;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // Ensure storage directories exist
    dirs::ensure_dirs()?;

    let cli = Cli::parse();

    // Load configuration
    let config = std::sync::Arc::new(PowerConfig::load()?);

    // Initialize model registry
    let registry = ModelRegistry::new();
    registry.scan()?;

    // Initialize backends
    let backends = backend::default_backends(config.clone());

    match cli.command {
        Commands::Run {
            model,
            prompt,
            temperature,
            top_p,
            top_k,
            num_predict,
            num_ctx,
            repeat_penalty,
            seed,
        } => {
            let options = a3s_power::cli::run::RunOptions {
                temperature,
                top_p,
                top_k,
                num_predict,
                num_ctx,
                repeat_penalty,
                seed,
            };
            a3s_power::cli::run::execute_with_options(
                &model,
                prompt.as_deref(),
                &registry,
                &backends,
                options,
            )
            .await?;
        }
        Commands::Pull { model } => {
            a3s_power::cli::pull::execute(&model, &registry).await?;
        }
        Commands::List => {
            a3s_power::cli::list::execute(&registry)?;
        }
        Commands::Show { model } => {
            a3s_power::cli::show::execute(&model, &registry)?;
        }
        Commands::Delete { model } => {
            a3s_power::cli::delete::execute(&model, &registry)?;
        }
        Commands::Serve { host, port } => {
            a3s_power::cli::serve::execute(&host, port).await?;
        }
        Commands::Create { name, file } => {
            let content = std::fs::read_to_string(&file).map_err(|e| {
                anyhow::anyhow!("Failed to read Modelfile '{}': {e}", file.display())
            })?;
            let mf = a3s_power::model::modelfile::parse(&content)
                .map_err(|e| anyhow::anyhow!("Failed to parse Modelfile: {e}"))?;
            let base = registry.get(&mf.from).map_err(|_| {
                anyhow::anyhow!("Base model '{}' not found; pull it first", mf.from)
            })?;
            let default_params = a3s_power::model::modelfile::parameters_to_json(&mf);
            let modelfile_content = a3s_power::model::modelfile::to_string(&mf);
            let manifest = a3s_power::model::manifest::ModelManifest {
                name: name.clone(),
                format: base.format.clone(),
                size: base.size,
                sha256: base.sha256.clone(),
                parameters: base.parameters.clone(),
                created_at: chrono::Utc::now(),
                path: base.path.clone(),
                system_prompt: mf.system.clone(),
                template_override: mf.template.clone(),
                default_parameters: if default_params.is_empty() {
                    None
                } else {
                    Some(default_params)
                },
                modelfile_content: Some(modelfile_content),
            };
            registry.register(manifest)?;
            println!("Created model '{name}' from '{}'", mf.from);
        }
        Commands::Push {
            model,
            destination,
        } => {
            a3s_power::cli::push::execute(&model, &destination, &registry).await?;
        }
    }

    Ok(())
}
