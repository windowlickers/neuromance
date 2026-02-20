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

/// Returns the neuromance data directory.
///
/// Uses `NEUROMANCE_DATA_DIR` if set (returned as-is), otherwise
/// falls back to `data_dir().join("neuromance")`.
pub fn neuromance_data_dir() -> Option<PathBuf> {
    std::env::var_os("NEUROMANCE_DATA_DIR")
        .map(PathBuf::from)
        .or_else(|| data_dir().map(|d| d.join("neuromance")))
}

/// Returns the neuromance config directory.
///
/// Uses `NEUROMANCE_CONFIG_DIR` if set (returned as-is), otherwise
/// falls back to `config_dir().join("neuromance")`.
pub fn neuromance_config_dir() -> Option<PathBuf> {
    std::env::var_os("NEUROMANCE_CONFIG_DIR")
        .map(PathBuf::from)
        .or_else(|| config_dir().map(|d| d.join("neuromance")))
}
