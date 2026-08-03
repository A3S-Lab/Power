use std::fs::File;
use std::path::Path;

use crate::error::Result;

/// Synchronizes a directory entry update on platforms that expose directory
/// durability through file handles.
///
/// Windows file replacement APIs provide their own publication semantics and
/// Rust does not permit opening a directory as a normal file there.
#[cfg(unix)]
pub(super) fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(windows)]
pub(super) fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}
