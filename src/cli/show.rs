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
        println!(
            "\n(Verbose: model file not found at {})",
            manifest.path.display()
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::manifest::{ModelFormat, ModelManifest, ModelParameters};
    use serial_test::serial;

    fn test_manifest(name: &str) -> ModelManifest {
        ModelManifest {
            name: name.to_string(),
            format: ModelFormat::Gguf,
            size: 1_500_000_000,
            sha256: "abc123".to_string(),
            parameters: Some(ModelParameters {
                context_length: Some(4096),
                embedding_length: Some(3200),
                parameter_count: Some(3_000_000_000),
                quantization: Some("Q4_K_M".to_string()),
            }),
            created_at: chrono::Utc::now(),
            path: std::path::PathBuf::from("/tmp/fake.gguf"),
            system_prompt: Some("You are helpful.".to_string()),
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: Some("MIT License".to_string()),
            adapter_path: None,
            projector_path: None,
            messages: vec![],
            family: None,
            families: None,
        }
    }

    #[test]
    #[serial]
    fn test_show_execute_success() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        registry.register(test_manifest("test-model")).unwrap();

        let result = execute("test-model", &registry, false);
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_show_execute_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let result = execute("nonexistent", &registry, false);
        assert!(result.is_err());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_show_execute_verbose_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        registry.register(test_manifest("test-model")).unwrap();

        // verbose=true but file doesn't exist — should still succeed
        let result = execute("test-model", &registry, true);
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_show_with_no_parameters() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let mut manifest = test_manifest("bare-model");
        manifest.parameters = None;
        manifest.system_prompt = None;
        manifest.license = None;
        registry.register(manifest).unwrap();

        let result = execute("bare-model", &registry, false);
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_show_with_long_license() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let mut manifest = test_manifest("licensed-model");
        manifest.license = Some("A".repeat(500));
        registry.register(manifest).unwrap();

        // Should truncate license to 200 chars
        let result = execute("licensed-model", &registry, false);
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_show_parameter_count_display_billions() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let manifest = test_manifest("big-model");
        registry.register(manifest).unwrap();

        // 3B params should display as "3.0B"
        let result = execute("big-model", &registry, false);
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_show_parameter_count_display_millions() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let mut manifest = test_manifest("small-model");
        manifest.parameters = Some(ModelParameters {
            context_length: None,
            embedding_length: None,
            parameter_count: Some(125_000_000),
            quantization: None,
        });
        registry.register(manifest).unwrap();

        let result = execute("small-model", &registry, false);
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_show_parameter_count_display_small() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let mut manifest = test_manifest("tiny-model");
        manifest.parameters = Some(ModelParameters {
            context_length: None,
            embedding_length: None,
            parameter_count: Some(500_000),
            quantization: None,
        });
        registry.register(manifest).unwrap();

        let result = execute("tiny-model", &registry, false);
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_show_with_all_parameters() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let mut manifest = test_manifest("full-model");
        manifest.parameters = Some(ModelParameters {
            context_length: Some(8192),
            embedding_length: Some(4096),
            parameter_count: Some(7_000_000_000),
            quantization: Some("Q8_0".to_string()),
        });
        registry.register(manifest).unwrap();

        let result = execute("full-model", &registry, false);
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_show_license_exactly_200_chars() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let mut manifest = test_manifest("exact-license");
        manifest.license = Some("A".repeat(200));
        registry.register(manifest).unwrap();

        let result = execute("exact-license", &registry, false);
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_show_license_short() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let mut manifest = test_manifest("short-license");
        manifest.license = Some("MIT".to_string());
        registry.register(manifest).unwrap();

        let result = execute("short-license", &registry, false);
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    fn test_print_verbose_info_nonexistent_file() {
        let manifest = test_manifest("missing");
        let result = print_verbose_info(&manifest);
        assert!(result.is_ok());
    }
}
