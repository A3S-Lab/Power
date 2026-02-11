use crate::error::Result;
use crate::model::registry::ModelRegistry;

/// Execute the `push` command: upload a model to a remote registry.
pub async fn execute(model: &str, destination: &str, registry: &ModelRegistry, insecure: bool) -> Result<()> {
    let manifest = registry.get(model)?;

    println!("Pushing '{}' to {}", model, destination);

    let progress = Box::new(|completed: u64, total: u64| {
        if total > 0 {
            let pct = upload_percentage(completed, total);
            println!("  Uploading: {}% ({}/{})", pct, completed, total);
        }
    });

    let digest = crate::model::push::push_model(&manifest, destination, Some(progress), insecure).await?;

    println!("Push complete: {}", digest);
    Ok(())
}

/// Calculate upload percentage for display.
fn upload_percentage(completed: u64, total: u64) -> u32 {
    if total == 0 {
        return 0;
    }
    (completed as f64 / total as f64 * 100.0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_upload_percentage_zero_total() {
        assert_eq!(upload_percentage(0, 0), 0);
    }

    #[test]
    fn test_upload_percentage_half() {
        assert_eq!(upload_percentage(50, 100), 50);
    }

    #[test]
    fn test_upload_percentage_complete() {
        assert_eq!(upload_percentage(100, 100), 100);
    }

    #[test]
    fn test_upload_percentage_large_values() {
        assert_eq!(upload_percentage(4_000_000_000, 8_000_000_000), 50);
    }

    #[test]
    fn test_upload_percentage_small_progress() {
        assert_eq!(upload_percentage(1, 1000), 0);
    }

    #[tokio::test]
    #[serial]
    async fn test_execute_model_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let registry = ModelRegistry::new();
        let result = execute("nonexistent", "http://localhost:9999", &registry, false).await;
        assert!(result.is_err());

        std::env::remove_var("A3S_POWER_HOME");
    }
}
