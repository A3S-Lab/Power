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

    // Early exit for commands that don't need registry/backends
    match &cli.command {
        Commands::Help { command } => {
            use clap::CommandFactory;
            let mut cmd = Cli::command();
            if let Some(sub) = command {
                // Print help for a specific subcommand
                match cmd.find_subcommand_mut(sub) {
                    Some(subcmd) => {
                        subcmd.print_help().ok();
                        println!();
                    }
                    None => {
                        eprintln!("Unknown command: '{sub}'");
                        eprintln!("Run 'a3s-power help' for a list of commands.");
                    }
                }
            } else {
                cmd.print_help().ok();
                println!();
            }
            return Ok(());
        }
        Commands::Update => {
            return a3s_updater::run_update(&a3s_updater::UpdateConfig {
                binary_name: "a3s-power",
                crate_name: "a3s-power",
                current_version: env!("CARGO_PKG_VERSION"),
                github_owner: "A3S-Lab",
                github_repo: "Power",
            })
            .await;
        }
        Commands::Ps => {
            return a3s_power::cli::ps::execute().await.map_err(Into::into);
        }
        Commands::Stop { model } => {
            return a3s_power::cli::stop::execute(model)
                .await
                .map_err(Into::into);
        }
        _ => {}
    }

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
            prompt_args,
            temperature,
            top_p,
            top_k,
            num_predict,
            num_ctx,
            repeat_penalty,
            seed,
            format,
            system,
            template,
            keep_alive,
            verbose,
            insecure,
            think,
            hidethinking,
        } => {
            // Merge --prompt flag and trailing positional args.
            // Priority: --prompt flag wins; otherwise join trailing args.
            let effective_prompt = if prompt.is_some() {
                prompt
            } else if !prompt_args.is_empty() {
                Some(prompt_args.join(" "))
            } else {
                None
            };
            let options = a3s_power::cli::run::RunOptions {
                temperature,
                top_p,
                top_k,
                num_predict,
                num_ctx,
                repeat_penalty,
                seed,
                format,
                system,
                template,
                keep_alive,
                verbose,
                insecure,
                think,
                hidethinking,
            };
            a3s_power::cli::run::execute_with_options(
                &model,
                effective_prompt.as_deref(),
                &registry,
                &backends,
                options,
            )
            .await?;
        }
        Commands::Pull { model, insecure } => {
            if insecure {
                tracing::info!("TLS verification disabled (--insecure)");
            }
            a3s_power::cli::pull::execute(&model, &registry, insecure).await?;
        }
        Commands::List => {
            a3s_power::cli::list::execute(&registry)?;
        }
        Commands::Show { model, verbose } => {
            a3s_power::cli::show::execute(&model, &registry, verbose)?;
        }
        Commands::Delete { model } => {
            a3s_power::cli::delete::execute(&model, &registry)?;
        }
        Commands::Serve { host, port } => {
            a3s_power::cli::serve::execute(&host, port).await?;
        }
        Commands::Create {
            name,
            file,
            quantize,
        } => {
            if let Some(ref q) = quantize {
                tracing::warn!(
                    quantize = %q,
                    "Quantization requested but re-quantization is not yet supported; \
                     the model will use its original quantization level"
                );
            }
            let content = std::fs::read_to_string(&file).map_err(|e| {
                anyhow::anyhow!("Failed to read Modelfile '{}': {e}", file.display())
            })?;
            let mf = a3s_power::model::modelfile::parse(&content)
                .map_err(|e| anyhow::anyhow!("Failed to parse Modelfile: {e}"))?;

            // Determine if FROM references a local file or a registered model
            let from_path = std::path::Path::new(&mf.from);
            let is_local_file = from_path.extension().is_some_and(|ext| ext == "gguf")
                || mf.from.starts_with('/')
                || mf.from.starts_with("./")
                || mf.from.starts_with("../");

            let (
                base_format,
                base_size,
                base_sha256,
                base_params,
                base_path,
                base_family,
                base_families,
            ) = if is_local_file {
                // FROM /path/to/file.gguf — import local GGUF file
                let gguf_path = from_path.to_path_buf();
                if !gguf_path.exists() {
                    return Err(anyhow::anyhow!("GGUF file '{}' not found", mf.from));
                }
                let metadata = std::fs::metadata(&gguf_path)
                    .map_err(|e| anyhow::anyhow!("Failed to read GGUF file '{}': {e}", mf.from))?;
                let file_size = metadata.len();

                // Copy/link the file into blob storage
                let (blob_path, sha256) =
                    a3s_power::model::storage::store_blob_from_path(&gguf_path)?;

                (
                    a3s_power::model::manifest::ModelFormat::Gguf,
                    file_size,
                    sha256,
                    None, // no parameters yet
                    blob_path,
                    None,
                    None,
                )
            } else {
                // FROM model-name — look up registered model
                let base = registry.get(&mf.from).map_err(|_| {
                    anyhow::anyhow!("Base model '{}' not found; pull it first", mf.from)
                })?;
                (
                    base.format.clone(),
                    base.size,
                    base.sha256.clone(),
                    base.parameters.clone(),
                    base.path.clone(),
                    base.family.clone(),
                    base.families.clone(),
                )
            };

            let default_params = a3s_power::model::modelfile::parameters_to_json(&mf);
            let modelfile_content = a3s_power::model::modelfile::to_string(&mf);
            let manifest = a3s_power::model::manifest::ModelManifest {
                name: name.clone(),
                format: base_format,
                size: base_size,
                sha256: base_sha256,
                parameters: base_params,
                created_at: chrono::Utc::now(),
                path: base_path,
                system_prompt: mf.system.clone(),
                template_override: mf.template.clone(),
                default_parameters: if default_params.is_empty() {
                    None
                } else {
                    Some(default_params)
                },
                modelfile_content: Some(modelfile_content),
                license: mf.license.clone(),
                adapter_path: mf.adapter.clone(),
                projector_path: None,
                messages: mf
                    .messages
                    .iter()
                    .map(|m| a3s_power::model::manifest::ManifestMessage {
                        role: m.role.clone(),
                        content: m.content.clone(),
                    })
                    .collect(),
                family: base_family,
                families: base_families,
            };
            registry.register(manifest)?;
            println!("Created model '{name}' from '{}'", mf.from);
        }
        Commands::Push {
            model,
            destination,
            insecure,
        } => {
            if insecure {
                tracing::info!("TLS verification disabled (--insecure)");
            }
            a3s_power::cli::push::execute(&model, &destination, &registry, insecure).await?;
        }
        Commands::Cp {
            source,
            destination,
        } => {
            let manifest = registry
                .get(&source)
                .map_err(|_| anyhow::anyhow!("Source model '{}' not found", source))?;
            let mut new_manifest = manifest.clone();
            new_manifest.name = destination.clone();
            new_manifest.created_at = chrono::Utc::now();
            registry.register(new_manifest)?;
            println!("Copied '{}' to '{}'", source, destination);
        }
        Commands::Update => unreachable!(),
        Commands::Ps => unreachable!(),
        Commands::Stop { .. } => unreachable!(),
        Commands::Help { .. } => unreachable!(),
    }

    Ok(())
}
