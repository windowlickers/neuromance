//! XDG-compliant path helpers.
//!
//! Respects `XDG_CONFIG_HOME` and `XDG_DATA_HOME` environment
//! variables, falling back to `~/.config` and `~/.local/share`.

use std::path::PathBuf;

/// Returns the XDG config base directory.
///
/// Uses `XDG_CONFIG_HOME` if set, otherwise `~/.config`.
pub fn config_dir() -> Option<PathBuf> {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".config")))
}

/// Returns the XDG data base directory.
///
/// Uses `XDG_DATA_HOME` if set, otherwise `~/.local/share`.
pub fn data_dir() -> Option<PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".local/share")))
}
