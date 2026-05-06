use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use serde_json::Value;
use tokio::fs;

use crate::factory::ToolFactory;
use crate::{ToolImplementation, ToolRegistry};
use neuromance_common::tools::{Function, Parameters, Property, Tool};

/// Creates or overwrites a file with the given UTF-8 content.
///
/// Not auto-approved: this tool mutates the filesystem.
pub struct WriteTool;

#[async_trait]
impl ToolImplementation for WriteTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            Property::string("Absolute path to the file. Parent directory must already exist."),
        );
        properties.insert(
            "content".to_string(),
            Property::string("UTF-8 content to write. Replaces any existing file contents."),
        );

        Tool::builder()
            .function(Function {
                name: "write".to_string(),
                description: "Create or overwrite a file with the given content. \
                              The parent directory must already exist."
                    .to_string(),
                parameters: Parameters::new(properties, vec!["path".into(), "content".into()])
                    .into(),
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
        let content = obj
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing 'content' parameter"))?;

        let path = PathBuf::from(path_str);
        if !path.is_absolute() {
            bail!("'path' must be absolute, got: {}", path.display());
        }

        let parent = path
            .parent()
            .ok_or_else(|| anyhow!("'path' has no parent directory: {}", path.display()))?;
        if !parent.is_dir() {
            bail!(
                "parent directory does not exist: {} (parent of {})",
                parent.display(),
                path.display()
            );
        }

        fs::write(&path, content)
            .await
            .with_context(|| format!("failed to write file '{}'", path.display()))?;

        Ok(format!(
            "wrote {} bytes to {}",
            content.len(),
            path.display()
        ))
    }
}

/// Factory that registers [`WriteTool`] under the name `write`.
/// Takes no configuration.
pub struct WriteToolFactory;

impl ToolFactory for WriteToolFactory {
    fn name(&self) -> &'static str {
        "write"
    }

    fn build(&self, _config: &Value, registry: &ToolRegistry) -> Result<()> {
        registry.register(Arc::new(WriteTool));
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
    async fn test_write_creates_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("new.txt");

        let tool = WriteTool;
        let result = tool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "content": "hello world",
            }))
            .await
            .unwrap();

        assert!(result.contains("wrote 11 bytes"));
        let contents = tokio_fs::read_to_string(&path).await.unwrap();
        assert_eq!(contents, "hello world");
    }

    #[tokio::test]
    async fn test_write_overwrites_existing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("exists.txt");
        tokio_fs::write(&path, "old content").await.unwrap();

        let tool = WriteTool;
        tool.execute(&json!({
            "path": path.to_str().unwrap(),
            "content": "fresh",
        }))
        .await
        .unwrap();

        let contents = tokio_fs::read_to_string(&path).await.unwrap();
        assert_eq!(contents, "fresh");
    }

    #[tokio::test]
    async fn test_write_missing_parent_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested/missing/file.txt");

        let tool = WriteTool;
        let err = tool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "content": "x",
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("parent directory does not exist"));
    }

    #[tokio::test]
    async fn test_write_rejects_relative_path() {
        let tool = WriteTool;
        let err = tool
            .execute(&json!({ "path": "rel.txt", "content": "x" }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("must be absolute"));
    }

    #[test]
    fn test_definition_has_required_fields() {
        let def = WriteTool.get_definition();
        assert_eq!(def.function.name, "write");
        let req = &def.function.parameters["required"];
        assert!(req.as_array().unwrap().iter().any(|v| v == "path"));
        assert!(req.as_array().unwrap().iter().any(|v| v == "content"));
    }

    #[test]
    fn test_is_not_auto_approved() {
        assert!(!WriteTool.is_auto_approved());
    }
}
