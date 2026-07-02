//! Workspace archives: tar + zstd pack/unpack with exclusion globs.
//!
//! All functions are blocking — callers in async context wrap them in
//! `spawn_blocking`.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use globset::GlobSet;

/// Compression level for snapshots. Level 3 is zstd's default: fast enough
/// to fit inside a shutdown drain window, dense enough for source trees.
const ZSTD_LEVEL: i32 = 3;

#[derive(Debug, thiserror::Error)]
pub enum ArchiveError {
    #[error("packing {}: {source}", path.display())]
    Pack {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("unpacking archive into {}: {source}", path.display())]
    Unpack {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// An archive entry would resolve outside the destination directory.
    #[error("archive entry '{entry}' escapes the destination directory")]
    EntryEscapes { entry: String },
}

/// Pack `dir` into `dest` as a zstd-compressed tar, skipping paths (relative
/// to `dir`) that match `exclude`. Returns the number of entries written.
///
/// Directories whose contents are fully excluded by a `<dir>/**` pattern are
/// pruned without being walked. Symlinks are archived as links, never
/// followed.
///
/// # Errors
/// Returns [`ArchiveError::Pack`] on any I/O failure while walking `dir` or
/// writing `dest`.
pub fn pack(dir: &Path, dest: impl Write, exclude: &GlobSet) -> Result<usize, ArchiveError> {
    let pack_err = |source| ArchiveError::Pack {
        path: dir.to_path_buf(),
        source,
    };
    let encoder = zstd::Encoder::new(dest, ZSTD_LEVEL).map_err(pack_err)?;
    let mut builder = tar::Builder::new(encoder);
    builder.follow_symlinks(false);
    let count = append_dir(&mut builder, dir, dir, exclude)?;
    builder
        .into_inner()
        .and_then(zstd::Encoder::finish)
        .and_then(|mut w| w.flush())
        .map_err(pack_err)?;
    Ok(count)
}

/// Recursively append `current`'s entries to the tar, paths named relative
/// to `root`.
fn append_dir(
    builder: &mut tar::Builder<impl Write>,
    root: &Path,
    current: &Path,
    exclude: &GlobSet,
) -> Result<usize, ArchiveError> {
    let pack_err = |source| ArchiveError::Pack {
        path: current.to_path_buf(),
        source,
    };
    let mut count = 0;
    for entry in std::fs::read_dir(current).map_err(pack_err)? {
        let entry = entry.map_err(pack_err)?;
        let path = entry.path();
        let Ok(rel) = path.strip_prefix(root) else {
            continue;
        };
        if exclude.is_match(rel) {
            continue;
        }
        let file_type = entry.file_type().map_err(pack_err)?;
        if file_type.is_dir() {
            // Prune a directory outright when a `<dir>/**`-shaped pattern
            // covers arbitrary children — probing one synthetic child is how
            // we detect that without enumerating a possibly huge tree.
            if exclude.is_match(rel.join("__prune_probe__")) {
                continue;
            }
            builder
                .append_path_with_name(&path, rel)
                .map_err(pack_err)?;
            count += 1 + append_dir(builder, root, &path, exclude)?;
        } else {
            builder
                .append_path_with_name(&path, rel)
                .map_err(pack_err)?;
            count += 1;
        }
    }
    Ok(count)
}

/// Unpack a zstd-compressed tar from `src` into `dir`.
///
/// Entries that would resolve outside `dir` (absolute paths, `..`) are
/// rejected, failing the whole unpack.
///
/// # Errors
/// Returns [`ArchiveError::Unpack`] on I/O or decompression failure and
/// [`ArchiveError::EntryEscapes`] on a path-traversal entry.
pub fn unpack(src: impl Read, dir: &Path) -> Result<(), ArchiveError> {
    let unpack_err = |source| ArchiveError::Unpack {
        path: dir.to_path_buf(),
        source,
    };
    let decoder = zstd::Decoder::new(src).map_err(unpack_err)?;
    let mut archive = tar::Archive::new(decoder);
    for entry in archive.entries().map_err(unpack_err)? {
        let mut entry = entry.map_err(unpack_err)?;
        let escaped = !entry.unpack_in(dir).map_err(unpack_err)?;
        if escaped {
            let name = entry
                .path()
                .map_or_else(|_| "<non-utf8>".to_string(), |p| p.display().to_string());
            return Err(ArchiveError::EntryEscapes { entry: name });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use globset::{Glob, GlobSetBuilder};
    use tempfile::TempDir;

    fn globs(patterns: &[&str]) -> GlobSet {
        let mut builder = GlobSetBuilder::new();
        for p in patterns {
            builder.add(Glob::new(p).unwrap());
        }
        builder.build().unwrap()
    }

    fn write(dir: &Path, rel: &str, content: &str) {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    #[test]
    fn test_pack_unpack_round_trip_preserves_tree() {
        let src = TempDir::new().unwrap();
        write(src.path(), "a.txt", "alpha");
        write(src.path(), "sub/deep/b.txt", "beta");

        let mut buf = Vec::new();
        let count = pack(src.path(), &mut buf, &globs(&[])).unwrap();
        assert!(count >= 4, "files + dirs, got {count}");

        let out = TempDir::new().unwrap();
        unpack(buf.as_slice(), out.path()).unwrap();
        assert_eq!(
            std::fs::read_to_string(out.path().join("a.txt")).unwrap(),
            "alpha"
        );
        assert_eq!(
            std::fs::read_to_string(out.path().join("sub/deep/b.txt")).unwrap(),
            "beta"
        );
    }

    #[test]
    fn test_pack_applies_exclusion_globs() {
        let src = TempDir::new().unwrap();
        write(src.path(), "keep.txt", "k");
        write(src.path(), "target/debug/huge.o", "x");
        write(src.path(), "sub/node_modules/pkg/index.js", "x");

        let exclude = globs(&["target/**", "**/node_modules/**"]);
        let mut buf = Vec::new();
        pack(src.path(), &mut buf, &exclude).unwrap();

        let out = TempDir::new().unwrap();
        unpack(buf.as_slice(), out.path()).unwrap();
        assert!(out.path().join("keep.txt").exists());
        assert!(!out.path().join("target").exists(), "target pruned");
        assert!(
            !out.path().join("sub/node_modules").exists(),
            "nested node_modules pruned"
        );
        assert!(out.path().join("sub").exists());
    }

    #[test]
    fn test_pack_preserves_symlinks_without_following() {
        let src = TempDir::new().unwrap();
        write(src.path(), "real.txt", "content");
        std::os::unix::fs::symlink("real.txt", src.path().join("link.txt")).unwrap();

        let mut buf = Vec::new();
        pack(src.path(), &mut buf, &globs(&[])).unwrap();

        let out = TempDir::new().unwrap();
        unpack(buf.as_slice(), out.path()).unwrap();
        let link = out.path().join("link.txt");
        assert!(link.symlink_metadata().unwrap().file_type().is_symlink());
        assert_eq!(std::fs::read_to_string(&link).unwrap(), "content");
    }

    #[test]
    fn test_unpack_rejects_traversal_entry() {
        // Hand-build an archive whose entry climbs out of the destination.
        let mut tar_bytes = Vec::new();
        {
            let encoder = zstd::Encoder::new(&mut tar_bytes, 3).unwrap();
            let mut builder = tar::Builder::new(encoder);
            let mut header = tar::Header::new_gnu();
            // `set_path`/`append_data` refuse `..`, so write the raw name
            // field directly — exactly what a hostile archive would contain.
            let name = b"../evil.txt";
            header.as_old_mut().name[..name.len()].copy_from_slice(name);
            header.set_size(4);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append(&header, b"boom".as_slice()).unwrap();
            builder
                .into_inner()
                .and_then(zstd::Encoder::finish)
                .unwrap();
        }

        let out = TempDir::new().unwrap();
        let err = unpack(tar_bytes.as_slice(), out.path()).unwrap_err();
        assert!(
            matches!(err, ArchiveError::EntryEscapes { .. }),
            "got: {err}"
        );
        assert!(!out.path().parent().unwrap().join("evil.txt").exists());
    }

    #[test]
    fn test_unpack_garbage_input_errors() {
        let out = TempDir::new().unwrap();
        let err = unpack(b"not a zstd stream".as_slice(), out.path()).unwrap_err();
        assert!(matches!(err, ArchiveError::Unpack { .. }), "got: {err}");
    }
}
