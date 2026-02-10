use crate::error::Result;
use crate::model::registry::ModelRegistry;

/// Execute the `show` command: display detailed information about a model.
pub fn execute(model: &str, registry: &ModelRegistry, verbose: bool) -> Result<()> {
    let manifest = registry.get(model)?;

    println!("Model: {}", manifest.name);
    println!("Format: {}", manifest.format);
    println!("Size: {}", manifest.size_display());
    println!("SHA256: {}", manifest.sha256);
    println!("Path: {}", manifest.path.display());
    println!(
        "Created: {}",
        manifest.created_at.format("%Y-%m-%d %H:%M:%S UTC")
    );

    if let Some(params) = &manifest.parameters {
        println!("\nParameters:");
        if let Some(ctx) = params.context_length {
            println!("  Context Length: {ctx}");
        }
        if let Some(emb) = params.embedding_length {
            println!("  Embedding Length: {emb}");
        }
        if let Some(count) = params.parameter_count {
            let display = if count >= 1_000_000_000 {
                format!("{:.1}B", count as f64 / 1_000_000_000.0)
            } else if count >= 1_000_000 {
                format!("{:.1}M", count as f64 / 1_000_000.0)
            } else {
                format!("{count}")
            };
            println!("  Parameter Count: {display}");
        }
        if let Some(quant) = &params.quantization {
            println!("  Quantization: {quant}");
        }
    }

    if let Some(ref sys) = manifest.system_prompt {
        println!("\nSystem Prompt: {sys}");
    }

    if let Some(ref license) = manifest.license {
        println!("\nLicense: {}", &license[..license.len().min(200)]);
        if license.len() > 200 {
            println!("  ... (truncated, {} bytes total)", license.len());
        }
    }

    if verbose {
        print_verbose_info(&manifest)?;
    }

    Ok(())
}

/// Print verbose GGUF metadata and tensor information.
fn print_verbose_info(manifest: &crate::model::manifest::ModelManifest) -> Result<()> {
    if !manifest.path.exists() {
        println!("\n(Verbose: model file not found at {})", manifest.path.display());
        return Ok(());
    }

    match crate::model::gguf::read_metadata(&manifest.path) {
        Ok(meta) => {
            println!("\n--- GGUF Metadata ---");
            println!("GGUF Version: {}", meta.version);
            println!("Tensor Count: {}", meta.tensors.len());
            println!("\nMetadata Keys ({}):", meta.metadata.len());
            // Sort keys for consistent output
            let mut keys: Vec<_> = meta.metadata.keys().collect();
            keys.sort();
            for key in keys {
                let val = &meta.metadata[key];
                let display = format!("{}", val.to_json());
                // Truncate long values
                if display.len() > 120 {
                    println!("  {key}: {}...", &display[..120]);
                } else {
                    println!("  {key}: {display}");
                }
            }

            if !meta.tensors.is_empty() {
                println!("\nTensors ({}):", meta.tensors.len());
                for t in meta.tensors.iter().take(20) {
                    println!(
                        "  {} {:?} {:?} ({} elements)",
                        t.name,
                        t.dimensions,
                        t.tensor_type,
                        t.element_count()
                    );
                }
                if meta.tensors.len() > 20 {
                    println!("  ... and {} more tensors", meta.tensors.len() - 20);
                }
            }
        }
        Err(e) => {
            println!("\n(Verbose: failed to read GGUF metadata: {e})");
        }
    }

    Ok(())
}
