use indicatif::{ProgressBar, ProgressStyle};

use crate::error::Result;
use crate::model::pull::pull_model;
use crate::model::registry::ModelRegistry;
use crate::model::resolve;

/// Execute the `pull` command: download a model by name or URL.
pub async fn execute(model: &str, registry: &ModelRegistry, insecure: bool) -> Result<()> {
    // Check if model exists AND its blob file is present on disk
    if registry.exists(model) {
        let blob_ok = registry
            .get(model)
            .ok()
            .map(|m| m.path.exists())
            .unwrap_or(false);
        if blob_ok {
            println!("Model '{model}' already exists locally.");
            return Ok(());
        }
        // Manifest exists but blob is missing — re-pull
        tracing::warn!(
            model = model,
            "Manifest exists but blob file is missing, re-pulling"
        );
    }

    // Determine display name
    let display_name = if resolve::is_url(model) {
        crate::model::pull::extract_name_from_url(model)
    } else {
        model.to_string()
    };

    let pb = ProgressBar::new(0);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("=>-"),
    );

    let progress_bar = pb.clone();
    let progress = Box::new(move |downloaded: u64, total: u64| {
        if total > 0 {
            progress_bar.set_length(total);
        }
        progress_bar.set_position(downloaded);
    });

    println!("Pulling '{display_name}'...");

    let manifest = pull_model(model, None, Some(progress), insecure).await?;
    pb.finish_with_message("Download complete");

    let pulled_name = manifest.name.clone();
    registry.register(manifest)?;
    println!("Successfully pulled '{pulled_name}'");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::manifest::{ModelFormat, ModelManifest};
    use serial_test::serial;

    #[tokio::test]
    #[serial]
    async fn test_execute_returns_early_if_model_exists() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let manifest = ModelManifest {
            name: "existing-model".to_string(),
            format: ModelFormat::Gguf,
            size: 1000,
            sha256: "abc123".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: dir.path().join("model.gguf"),
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            projector_path: None,
            messages: vec![],
            family: None,
            families: None,
        };
        registry.register(manifest).unwrap();

        // Should return Ok without attempting to download
        let result = execute("existing-model", &registry, false).await;
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    fn test_display_name_from_url() {
        let url = "https://example.com/models/llama-7b.gguf";
        let display_name = if resolve::is_url(url) {
            crate::model::pull::extract_name_from_url(url)
        } else {
            url.to_string()
        };
        assert_eq!(display_name, "llama-7b");
    }

    #[test]
    fn test_display_name_from_model_name() {
        let model = "llama3.2:3b";
        let display_name = if resolve::is_url(model) {
            crate::model::pull::extract_name_from_url(model)
        } else {
            model.to_string()
        };
        assert_eq!(display_name, "llama3.2:3b");
    }
}
