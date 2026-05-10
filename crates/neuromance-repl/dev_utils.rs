//! Shared dev-only helpers for benches and examples.
//!
//! Pulled in via `#[path = "../dev_utils.rs"] mod dev_utils;` from
//! `benches/` and `examples/`. Lives at the crate root rather than under
//! `src/` so Cargo does not auto-discover it as a target.

#![allow(dead_code)] // each importer uses a subset

/// Count open file descriptors held by the current process.
///
/// Linux-only: reads `/proc/self/fd`. Returns `0` on other platforms.
#[cfg(target_os = "linux")]
pub fn count_file_descriptors() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .map(|entries| entries.count())
        .unwrap_or(0)
}

#[cfg(not(target_os = "linux"))]
pub fn count_file_descriptors() -> usize {
    0
}
