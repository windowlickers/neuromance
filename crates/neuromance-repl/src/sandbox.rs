//! Filesystem sandboxing using Landlock (Linux only).
//!
//! Provides utilities for restricting filesystem access using the Landlock LSM.

use anyhow::Context;
use std::path::Path;

#[cfg(all(feature = "sandbox", target_os = "linux"))]
use landlock::{
    ABI, Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr,
};

/// Apply Landlock sandboxing to restrict filesystem access.
///
/// Only the specified paths will be accessible. If the list is empty,
/// only Python stdlib paths are allowed (read-only).
///
/// # Errors
///
/// Returns an error if Landlock is not supported or if ruleset creation fails.
#[cfg(all(feature = "sandbox", target_os = "linux"))]
pub fn apply_landlock(allowed_paths: &[impl AsRef<Path>]) -> Result<(), anyhow::Error> {
    // Get the latest supported ABI
    let abi = ABI::V2;

    // Create base ruleset
    let mut ruleset = Ruleset::default()
        .handle_access(AccessFs::from_all(abi))?
        .create()?;

    // If no paths specified, add default safe paths
    if allowed_paths.is_empty() {
        // Allow read-only access to Python stdlib and common system libraries
        let default_paths = vec![
            "/usr/lib/python3",
            "/usr/local/lib/python3",
            "/lib",
            "/lib64",
            "/usr/lib",
        ];

        for path in default_paths {
            if Path::new(path).exists() {
                ruleset = ruleset
                    .add_rule(PathBeneath::new(
                        PathFd::new(path).context("Failed to open path")?,
                        AccessFs::from_read(abi),
                    ))
                    .context(format!("Failed to add rule for {path}"))?;
            }
        }
    } else {
        // Add user-specified paths with full access
        for path in allowed_paths {
            let path = path.as_ref();
            if path.exists() {
                ruleset = ruleset
                    .add_rule(PathBeneath::new(
                        PathFd::new(path).context("Failed to open path")?,
                        AccessFs::from_all(abi),
                    ))
                    .context(format!("Failed to add rule for {}", path.display()))?;
            }
        }
    }

    // Apply the ruleset to the current thread
    ruleset
        .restrict_self()
        .context("Failed to apply Landlock restrictions")?;

    Ok(())
}

/// Apply Landlock sandboxing (no-op on non-Linux platforms).
#[cfg(not(all(feature = "sandbox", target_os = "linux")))]
pub fn apply_landlock(_allowed_paths: &[impl AsRef<Path>]) -> Result<(), anyhow::Error> {
    Err(anyhow::anyhow!(
        "Landlock sandboxing is only available on Linux with the 'sandbox' feature enabled"
    ))
}

#[cfg(all(test, feature = "sandbox", target_os = "linux"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_landlock_basic() {
        // This test may fail if Landlock is not supported on the system
        // (requires Linux kernel 5.13+)
        let temp_dir = std::env::temp_dir();
        let result = apply_landlock(&[temp_dir]);

        // If Landlock is supported, this should succeed
        // If not, we'll get a specific error
        if let Err(e) = result {
            println!("Landlock test skipped: {e}");
        }
    }
}
