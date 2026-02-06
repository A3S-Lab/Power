use indicatif::{ProgressBar, ProgressStyle};

use crate::error::Result;
use crate::model::pull::pull_model;
use crate::model::registry::ModelRegistry;
use crate::model::resolve;

/// Execute the `pull` command: download a model by name or URL.
pub async fn execute(model: &str, registry: &ModelRegistry) -> Result<()> {
    if registry.exists(model) {
        println!("Model '{model}' already exists locally.");
        return Ok(());
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

    let manifest = pull_model(model, None, Some(progress)).await?;
    pb.finish_with_message("Download complete");

    let pulled_name = manifest.name.clone();
    registry.register(manifest)?;
    println!("Successfully pulled '{pulled_name}'");

    Ok(())
}
