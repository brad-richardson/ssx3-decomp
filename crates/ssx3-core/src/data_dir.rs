//! Where the program's own files go by default: the memory card, the
//! settings file, texture packs, screenshots and the disc image.
//!
//! On the desktop the defaults are relative to the working directory (the
//! disc image: the repository root). A platform without a usable working
//! directory (Android) sets a data directory at startup, and the same
//! defaults go there instead. The environment variables that name a file
//! (`SSX3_ISO`, `SSX3_MEMCARD`, ...) override either.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Set the data directory. Call once at startup, before any file is opened.
pub fn set(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// The data directory, if one was set.
pub fn get() -> Option<&'static Path> {
    DIR.get().map(PathBuf::as_path)
}

/// A default file location: `rel` in the data directory, or `rel` as it
/// is (relative to the working directory) when none was set.
pub fn path(rel: &str) -> PathBuf {
    match get() {
        Some(dir) => dir.join(rel),
        None => PathBuf::from(rel),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn relative_without_a_data_dir() {
        assert_eq!(super::path("saves/ssx3-mc0.ps2"), std::path::Path::new("saves/ssx3-mc0.ps2"));
    }
}
