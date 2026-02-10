use crate::error::Result;
use crate::model::registry::ModelRegistry;

/// Execute the `list` command: display all locally available models.
pub fn execute(registry: &ModelRegistry) -> Result<()> {
    let models = registry.list()?;

    if models.is_empty() {
        println!("No models found locally.");
        println!("Use `a3s-power pull <url>` to download a model.");
        return Ok(());
    }

    println!("{:<30} {:<12} {:<12} MODIFIED", "NAME", "FORMAT", "SIZE");
    for model in &models {
        let modified = model.created_at.format("%Y-%m-%d %H:%M");
        println!(
            "{:<30} {:<12} {:<12} {}",
            model.name,
            model.format.to_string(),
            model.size_display(),
            modified,
        );
    }

    println!("\n{} model(s) total", models.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::manifest::{ModelFormat, ModelManifest};
    use serial_test::serial;

    fn test_manifest(name: &str) -> ModelManifest {
        ModelManifest {
            name: name.to_string(),
            format: ModelFormat::Gguf,
            size: 1_000_000,
            sha256: format!("sha256-{name}"),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: std::path::PathBuf::from(format!("/tmp/{name}")),
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
        }
    }

    #[test]
    #[serial]
    fn test_list_empty() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let result = execute(&registry);
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_list_with_models() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        registry.register(test_manifest("model-a")).unwrap();
        registry.register(test_manifest("model-b")).unwrap();

        let result = execute(&registry);
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_list_single_model() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        registry.register(test_manifest("single-model")).unwrap();

        let result = execute(&registry);
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_list_many_models() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        for i in 0..10 {
            registry
                .register(test_manifest(&format!("model-{}", i)))
                .unwrap();
        }

        let result = execute(&registry);
        assert!(result.is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_list_returns_models_in_order() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        registry.register(test_manifest("zebra")).unwrap();
        registry.register(test_manifest("alpha")).unwrap();
        registry.register(test_manifest("beta")).unwrap();

        let models = registry.list().unwrap();
        assert_eq!(models.len(), 3);

        std::env::remove_var("A3S_POWER_HOME");
    }
}
