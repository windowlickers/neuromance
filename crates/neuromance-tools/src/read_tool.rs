use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use serde_json::Value;
use tokio::fs;

use crate::factory::ToolFactory;
use crate::{ToolImplementation, ToolRegistry};
use neuromance_common::tools::{Function, Parameters, Property, Tool};

/// Maximum bytes returned by a single Read invocation.
const MAX_OUTPUT_BYTES: usize = 256 * 1024;

/// Reads a UTF-8 file from disk with optional line-range slicing.
///
/// Output is prefixed with `cat -n`-style line numbers and capped at
/// [`MAX_OUTPUT_BYTES`].
pub struct ReadTool;

#[async_trait]
impl ToolImplementation for ReadTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            Property::string("Absolute path to the file to read."),
        );
        properties.insert(
            "offset".to_string(),
            Property::number(
                "Optional 1-indexed line number to start reading from. Defaults to 1.",
            ),
        );
        properties.insert(
            "limit".to_string(),
            Property::number(
                "Optional maximum number of lines to read. Defaults to the rest of the file.",
            ),
        );

        Tool::builder()
            .function(Function {
                name: "read".to_string(),
                description: "Read a UTF-8 text file from disk. Output is prefixed with \
                              cat -n style line numbers. Output is capped at 256 KiB; \
                              longer reads are truncated."
                    .to_string(),
                parameters: Parameters::new(properties, vec!["path".into()]).into(),
            })
            .build()
    }

    async fn execute(&self, args: &Value) -> Result<String> {
        let obj = args
            .as_object()
            .ok_or_else(|| anyhow!("expected object arguments"))?;

        let path_str = obj
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing 'path' parameter"))?;
        let path = PathBuf::from(path_str);
        if !path.is_absolute() {
            bail!("'path' must be absolute, got: {}", path.display());
        }

        let offset = parse_positive_u64(obj.get("offset"), "offset")?.unwrap_or(1);
        if offset == 0 {
            bail!("'offset' must be >= 1");
        }
        let limit = parse_positive_u64(obj.get("limit"), "limit")?;

        let raw = fs::read(&path)
            .await
            .with_context(|| format!("failed to read file '{}'", path.display()))?;
        let content = std::str::from_utf8(&raw)
            .with_context(|| format!("file '{}' is not valid UTF-8", path.display()))?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();
        let start = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .saturating_sub(1);
        if start >= total_lines && total_lines > 0 {
            bail!("'offset' {offset} is past EOF (file has {total_lines} lines)");
        }

        let take = limit.map_or_else(
            || total_lines.saturating_sub(start),
            |l| usize::try_from(l).unwrap_or(usize::MAX),
        );
        let end = (start + take).min(total_lines);

        Ok(format_with_line_numbers(
            &lines[start..end],
            start + 1,
            total_lines,
        ))
    }

    fn is_auto_approved(&self) -> bool {
        true
    }
}

fn parse_positive_u64(v: Option<&Value>, name: &str) -> Result<Option<u64>> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(val) => {
            let n = val
                .as_u64()
                .ok_or_else(|| anyhow!("'{name}' must be a positive integer"))?;
            Ok(Some(n))
        }
    }
}

fn format_with_line_numbers(lines: &[&str], first_line_no: usize, total_lines: usize) -> String {
    let mut out = String::new();
    let mut emitted = 0usize;

    for (i, line) in lines.iter().enumerate() {
        let line_no = first_line_no + i;
        let formatted = format!("{line_no:>6}\t{line}\n");
        if out.len() + formatted.len() > MAX_OUTPUT_BYTES {
            let remaining = lines.len() - i;
            let _ = writeln!(
                out,
                "[truncated, {remaining} more lines (output capped at {MAX_OUTPUT_BYTES} bytes)]"
            );
            return out;
        }
        out.push_str(&formatted);
        emitted += 1;
    }

    if emitted > 0 {
        let last_emitted = first_line_no + emitted - 1;
        if first_line_no > 1 || last_emitted < total_lines {
            let _ = writeln!(
                out,
                "[showing lines {first_line_no}-{last_emitted} of {total_lines}]"
            );
        }
    }

    out
}

/// Factory that registers [`ReadTool`] under the name `read`.
/// Takes no configuration.
pub struct ReadToolFactory;

impl ToolFactory for ReadToolFactory {
    fn name(&self) -> &'static str {
        "read"
    }

    fn build(&self, _config: &Value, registry: &ToolRegistry) -> Result<()> {
        registry.register(Arc::new(ReadTool));
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
    use tokio::fs as tokio_fs;

    #[tokio::test]
    async fn test_read_full_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("hello.txt");
        tokio_fs::write(&path, "alpha\nbeta\ngamma\n")
            .await
            .unwrap();

        let tool = ReadTool;
        let result = tool
            .execute(&json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();

        assert!(result.contains("     1\talpha"));
        assert!(result.contains("     2\tbeta"));
        assert!(result.contains("     3\tgamma"));
    }

    #[tokio::test]
    async fn test_read_with_offset_and_limit() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("multi.txt");
        tokio_fs::write(&path, "a\nb\nc\nd\ne\n").await.unwrap();

        let tool = ReadTool;
        let result = tool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "offset": 2,
                "limit": 2,
            }))
            .await
            .unwrap();

        assert!(result.contains("     2\tb"));
        assert!(result.contains("     3\tc"));
        assert!(!result.contains("     1\ta"));
        assert!(!result.contains("     4\td"));
    }

    #[tokio::test]
    async fn test_read_missing_file() {
        let tool = ReadTool;
        let err = tool
            .execute(&json!({ "path": "/does/not/exist/anywhere/x" }))
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("failed to read file"));
    }

    #[tokio::test]
    async fn test_read_rejects_relative_path() {
        let tool = ReadTool;
        let err = tool
            .execute(&json!({ "path": "relative.txt" }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("must be absolute"));
    }

    #[tokio::test]
    async fn test_read_offset_past_eof() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("short.txt");
        tokio_fs::write(&path, "only\n").await.unwrap();

        let tool = ReadTool;
        let err = tool
            .execute(&json!({ "path": path.to_str().unwrap(), "offset": 99 }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("past EOF"));
    }

    #[test]
    fn test_definition_has_required_path() {
        let def = ReadTool.get_definition();
        assert_eq!(def.function.name, "read");
        assert_eq!(def.function.parameters["required"], json!(["path"]));
    }

    #[test]
    fn test_is_auto_approved() {
        assert!(ReadTool.is_auto_approved());
    }
}
