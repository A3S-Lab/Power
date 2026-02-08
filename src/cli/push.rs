use crate::error::Result;
use crate::model::registry::ModelRegistry;

/// Execute the `push` command: upload a model to a remote registry.
pub async fn execute(model: &str, destination: &str, registry: &ModelRegistry) -> Result<()> {
    let manifest = registry.get(model)?;

    println!("Pushing '{}' to {}", model, destination);

    let progress = Box::new(|completed: u64, total: u64| {
        if total > 0 {
            let pct = (completed as f64 / total as f64 * 100.0) as u32;
            println!("  Uploading: {}% ({}/{})", pct, completed, total);
        }
    });

    let digest =
        crate::model::push::push_model(&manifest, destination, Some(progress)).await?;

    println!("Push complete: {}", digest);
    Ok(())
}
