use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use a3s_power::error::{PowerError, Result};

pub(super) fn write_json_output(output: &serde_json::Value, path: Option<&Path>) -> Result<()> {
    let mut encoded = serde_json::to_vec_pretty(output)?;
    encoded.push(b'\n');
    let Some(path) = path else {
        std::io::stdout().write_all(&encoded)?;
        return Ok(());
    };
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            PowerError::InvalidRequest(format!(
                "failed to create new benchmark output '{}': {error}",
                path.display()
            ))
        })?;
    if let Err(error) = file.write_all(&encoded).and_then(|()| file.sync_all()) {
        drop(file);
        let _ = std::fs::remove_file(path);
        return Err(error.into());
    }
    Ok(())
}
