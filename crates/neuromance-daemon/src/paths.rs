//! XDG-compliant path helpers.
//!
//! Respects `XDG_CONFIG_HOME` and `XDG_DATA_HOME` environment
//! variables, falling back to `~/.config` and `~/.local/share`.
//!
//! Uses XDG layout on all platforms (including macOS) rather than
//! platform-native directories.

use std::ffi::OsStr;
use std::path::PathBuf;

/// Env var: override for the neuromance data directory.
pub const ENV_DATA_DIR: &str = "NEUROMANCE_DATA_DIR";

/// Env var: override for the neuromance config directory.
pub const ENV_CONFIG_DIR: &str = "NEUROMANCE_CONFIG_DIR";

/// Env var: override for the daemon binary path.
pub const ENV_DAEMON_BIN: &str = "NEUROMANCE_DAEMON_BIN";

/// Returns the XDG config base directory.
///
/// Uses `XDG_CONFIG_HOME` if set, otherwise `~/.config`.
fn config_dir() -> Option<PathBuf> {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".config")))
}

/// Returns the XDG data base directory.
///
/// Uses `XDG_DATA_HOME` if set, otherwise `~/.local/share`.
fn data_dir() -> Option<PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".local/share")))
}

/// Returns the neuromance data directory.
///
/// Uses `NEUROMANCE_DATA_DIR` if set (returned as-is), otherwise
/// falls back to `data_dir().join("neuromance")`.
pub fn neuromance_data_dir() -> Option<PathBuf> {
    std::env::var_os(ENV_DATA_DIR)
        .map(PathBuf::from)
        .or_else(|| data_dir().map(|d| d.join("neuromance")))
}

/// Returns the neuromance config directory.
///
/// Uses `NEUROMANCE_CONFIG_DIR` if set (returned as-is), otherwise
/// falls back to `config_dir().join("neuromance")`.
pub fn neuromance_config_dir() -> Option<PathBuf> {
    std::env::var_os(ENV_CONFIG_DIR)
        .map(PathBuf::from)
        .or_else(|| config_dir().map(|d| d.join("neuromance")))
}

/// Returns a config file path under the neuromance config directory.
///
/// Combines `neuromance_config_dir()` with the given filename.
pub fn neuromance_config_file(
    filename: impl AsRef<OsStr>,
) -> Option<PathBuf> {
    neuromance_config_dir().map(|d| d.join(filename.as_ref()))
}

/// Returns the daemon binary path.
///
/// Uses `NEUROMANCE_DAEMON_BIN` if set, otherwise `"neuromance-daemon"`.
pub fn daemon_bin() -> PathBuf {
    std::env::var_os(ENV_DAEMON_BIN)
        .map_or_else(|| PathBuf::from("neuromance-daemon"), PathBuf::from)
}
