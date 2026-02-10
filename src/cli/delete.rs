use crate::error::Result;
use crate::model::registry::ModelRegistry;
use crate::model::storage;

/// Execute the `delete` command: remove a model from local storage.
pub fn execute(model: &str, registry: &ModelRegistry) -> Result<()> {
    let manifest = registry.remove(model)?;
    storage::delete_blob(&manifest)?;

    println!("Deleted model '{}'", manifest.name);
    println!("  Freed {}", manifest.size_display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::manifest::{ModelFormat, ModelManifest};
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_delete_success() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let manifest = ModelManifest {
            name: "to-delete".to_string(),
            format: ModelFormat::Gguf,
            size: 100,
            sha256: "abc".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: std::path::PathBuf::from("/tmp/nonexistent-blob"),
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

        let result = execute("to-delete", &registry);
        assert!(result.is_ok());
        assert!(!registry.exists("to-delete"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_delete_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let result = execute("nonexistent", &registry);
        assert!(result.is_err());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_delete_removes_from_registry() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let manifest = ModelManifest {
            name: "model-to-remove".to_string(),
            format: ModelFormat::Gguf,
            size: 1024,
            sha256: "def456".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: std::path::PathBuf::from("/tmp/fake-blob"),
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
        assert!(registry.exists("model-to-remove"));

        execute("model-to-remove", &registry).unwrap();
        assert!(!registry.exists("model-to-remove"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_delete_multiple_models_independently() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();

        // Register two models
        for name in ["model-a", "model-b"] {
            let manifest = ModelManifest {
                name: name.to_string(),
                format: ModelFormat::Gguf,
                size: 100,
                sha256: format!("{}-hash", name),
                parameters: None,
                created_at: chrono::Utc::now(),
                path: std::path::PathBuf::from(format!("/tmp/{}", name)),
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
        }

        // Delete model-a
        execute("model-a", &registry).unwrap();
        assert!(!registry.exists("model-a"));
        assert!(registry.exists("model-b"));

        // Delete model-b
        execute("model-b", &registry).unwrap();
        assert!(!registry.exists("model-b"));

        std::env::remove_var("A3S_POWER_HOME");
    }
}
