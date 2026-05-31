use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::factory::ToolFactory;
use crate::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_head};
use crate::{ToolError, ToolImplementation, ToolRegistry, opt_u64, resolve_search_path};
use neuromance_common::tools::{Function, Parameters, Property, Tool};

/// Default cap on the number of entries returned.
const DEFAULT_LIMIT: u64 = 500;

/// Lists the immediate entries of a directory (non-recursive).
///
/// Directories are suffixed with `/`. Auto-approved: read-only.
pub struct LsTool;

#[async_trait]
impl ToolImplementation for LsTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            Property::string(
                "Optional absolute directory to list. Defaults to the current directory.",
            ),
        );
        properties.insert(
            "limit".to_string(),
            Property::number(format!(
                "Maximum number of entries to return. Defaults to {DEFAULT_LIMIT}."
            )),
        );

        Tool::builder()
            .function(Function {
                name: "ls".to_string(),
                description: "List the immediate entries of a directory (non-recursive). \
                              Directories are suffixed with '/'."
                    .to_string(),
                parameters: Parameters::new(properties, vec![]).into(),
            })
            .build()
    }

    async fn execute(&self, args: &Value) -> Result<String, ToolError> {
        let obj = args
            .as_object()
            .ok_or_else(|| ToolError::InvalidArguments("expected object arguments".into()))?;

        let root = resolve_search_path(obj.get("path").and_then(Value::as_str))?;
        if !root.is_dir() {
            return Err(ToolError::InvalidArguments(format!(
                "path is not a directory: {}",
                root.display()
            )));
        }
        let limit = usize::try_from(opt_u64(obj.get("limit"), "limit", DEFAULT_LIMIT)?.max(1))
            .unwrap_or(usize::MAX);

        let mut entries: Vec<String> = Vec::new();
        let mut reader = tokio::fs::read_dir(&root).await.map_err(|e| {
            ToolError::execution(format!(
                "failed to read directory '{}': {e}",
                root.display()
            ))
        })?;
        while let Some(entry) = reader.next_entry().await.map_err(|e| {
            ToolError::execution(format!(
                "failed to read directory '{}': {e}",
                root.display()
            ))
        })? {
            let name = entry.file_name().to_string_lossy().into_owned();
            let is_dir = entry.file_type().await.is_ok_and(|t| t.is_dir());
            entries.push(if is_dir { format!("{name}/") } else { name });
        }

        entries.sort_by_key(|e| e.to_lowercase());
        Ok(finalize(entries, limit))
    }

    fn is_auto_approved(&self) -> bool {
        true
    }
}

fn finalize(mut entries: Vec<String>, limit: usize) -> String {
    if entries.is_empty() {
        return "(empty directory)".to_string();
    }
    let total = entries.len();
    let limit_reached = total > limit;
    entries.truncate(limit);

    let body = entries.join("\n");
    let capped = truncate_head(&body, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    let truncated = capped.is_truncated();
    let (shown, capped_total) = (capped.shown_lines, capped.total_lines);
    let mut out = capped.content;
    if !out.ends_with('\n') {
        out.push('\n');
    }
    if truncated {
        let _ = writeln!(
            out,
            "[output truncated: showing {shown} of {capped_total} entries]"
        );
    }
    if limit_reached {
        let _ = writeln!(
            out,
            "[entry limit {limit} reached ({total} total); raise 'limit' for more]"
        );
    }
    out
}

/// Factory that registers [`LsTool`] under the name `ls`.
/// Takes no configuration.
pub struct LsToolFactory;

impl ToolFactory for LsToolFactory {
    fn name(&self) -> &'static str {
        "ls"
    }

    fn build(&self, _config: &Value, registry: &ToolRegistry) -> Result<(), ToolError> {
        registry.register(Arc::new(LsTool));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    async fn ls(args: Value) -> String {
        LsTool.execute(&args).await.unwrap()
    }

    #[tokio::test]
    async fn test_ls_lists_entries_sorted() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("banana.txt"), "").unwrap();
        std::fs::write(dir.path().join("apple.txt"), "").unwrap();
        std::fs::create_dir(dir.path().join("zeta")).unwrap();

        let out = ls(json!({ "path": dir.path().to_str().unwrap() })).await;
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines, vec!["apple.txt", "banana.txt", "zeta/"]);
    }

    #[tokio::test]
    async fn test_ls_marks_directories() {
        let dir = tempdir().unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let out = ls(json!({ "path": dir.path().to_str().unwrap() })).await;
        assert!(out.contains("subdir/"));
    }

    #[tokio::test]
    async fn test_ls_empty_directory() {
        let dir = tempdir().unwrap();
        let out = ls(json!({ "path": dir.path().to_str().unwrap() })).await;
        assert_eq!(out, "(empty directory)");
    }

    #[tokio::test]
    async fn test_ls_limit_footer() {
        let dir = tempdir().unwrap();
        for i in 0..10 {
            std::fs::write(dir.path().join(format!("f{i}.txt")), "").unwrap();
        }
        let out = ls(json!({ "path": dir.path().to_str().unwrap(), "limit": 3 })).await;
        assert!(
            out.contains("entry limit 3 reached (10 total)"),
            "got:\n{out}"
        );
    }

    #[tokio::test]
    async fn test_ls_rejects_file_path() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("f.txt");
        std::fs::write(&file, "").unwrap();
        let err = LsTool
            .execute(&json!({ "path": file.to_str().unwrap() }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not a directory"));
    }

    #[tokio::test]
    async fn test_ls_relative_path_rejected() {
        let err = LsTool
            .execute(&json!({ "path": "rel/dir" }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("must be absolute"));
    }

    #[test]
    fn test_is_auto_approved() {
        assert!(LsTool.is_auto_approved());
    }
}
