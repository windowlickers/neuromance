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

/// Performs exact-string replacement on a file's contents.
///
/// By default `old_string` must occur exactly once; set `replace_all=true`
/// to replace every occurrence. Not auto-approved: this tool mutates the
/// filesystem.
pub struct EditTool;

#[async_trait]
impl ToolImplementation for EditTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            Property::string("Absolute path to the file to edit."),
        );
        properties.insert(
            "old_string".to_string(),
            Property::string(
                "Exact string to find. Must be unique in the file unless replace_all is true.",
            ),
        );
        properties.insert(
            "new_string".to_string(),
            Property::string("String to replace old_string with."),
        );
        properties.insert(
            "replace_all".to_string(),
            Property::boolean("If true, replace every occurrence. Defaults to false."),
        );

        Tool::builder()
            .function(Function {
                name: "edit".to_string(),
                description: "Perform an exact-string replacement on a file. By default, \
                              old_string must occur exactly once; pass replace_all=true to \
                              replace every occurrence."
                    .to_string(),
                parameters: Parameters::new(
                    properties,
                    vec!["path".into(), "old_string".into(), "new_string".into()],
                )
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
        let old_string = obj
            .get("old_string")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing 'old_string' parameter"))?;
        let new_string = obj
            .get("new_string")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing 'new_string' parameter"))?;
        let replace_all = obj
            .get("replace_all")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        if old_string.is_empty() {
            bail!("'old_string' must not be empty");
        }
        if old_string == new_string {
            bail!("'old_string' and 'new_string' are identical (no-op)");
        }

        let path = PathBuf::from(path_str);
        if !path.is_absolute() {
            bail!("'path' must be absolute, got: {}", path.display());
        }

        let original = fs::read_to_string(&path)
            .await
            .with_context(|| format!("failed to read file '{}'", path.display()))?;

        let count = original.matches(old_string).count();
        if count == 0 {
            bail!("'old_string' not found in {}", path.display());
        }

        let updated = if replace_all {
            original.replace(old_string, new_string)
        } else {
            if count > 1 {
                bail!(
                    "'old_string' found {count} times in {} but replace_all is false; \
                     either provide more context or set replace_all=true",
                    path.display()
                );
            }
            original.replacen(old_string, new_string, 1)
        };

        fs::write(&path, &updated)
            .await
            .with_context(|| format!("failed to write file '{}'", path.display()))?;

        let replaced = if replace_all { count } else { 1 };
        let plural = if replaced == 1 { "" } else { "s" };
        Ok(format!(
            "replaced {replaced} occurrence{plural} in {}",
            path.display()
        ))
    }
}

/// Factory that registers [`EditTool`] under the name `edit`.
/// Takes no configuration.
pub struct EditToolFactory;

impl ToolFactory for EditToolFactory {
    fn name(&self) -> &'static str {
        "edit"
    }

    fn build(&self, _config: &Value, registry: &ToolRegistry) -> Result<()> {
        registry.register(Arc::new(EditTool));
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
    async fn test_edit_single_match() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("f.txt");
        tokio_fs::write(&path, "hello world").await.unwrap();

        let tool = EditTool;
        let result = tool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "world",
                "new_string": "rust",
            }))
            .await
            .unwrap();
        assert!(result.contains("replaced 1 occurrence"));

        let contents = tokio_fs::read_to_string(&path).await.unwrap();
        assert_eq!(contents, "hello rust");
    }

    #[tokio::test]
    async fn test_edit_replace_all() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("f.txt");
        tokio_fs::write(&path, "foo bar foo baz foo").await.unwrap();

        let tool = EditTool;
        let result = tool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "foo",
                "new_string": "FOO",
                "replace_all": true,
            }))
            .await
            .unwrap();

        assert!(result.contains("replaced 3 occurrences"));
        let contents = tokio_fs::read_to_string(&path).await.unwrap();
        assert_eq!(contents, "FOO bar FOO baz FOO");
    }

    #[tokio::test]
    async fn test_edit_ambiguous_without_replace_all() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("f.txt");
        tokio_fs::write(&path, "x x x").await.unwrap();

        let tool = EditTool;
        let err = tool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "x",
                "new_string": "y",
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("found 3 times"));
    }

    #[tokio::test]
    async fn test_edit_not_found() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("f.txt");
        tokio_fs::write(&path, "hello").await.unwrap();

        let tool = EditTool;
        let err = tool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "absent",
                "new_string": "x",
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_edit_noop_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("f.txt");
        tokio_fs::write(&path, "abc").await.unwrap();

        let tool = EditTool;
        let err = tool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "same",
                "new_string": "same",
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("identical"));
    }

    #[tokio::test]
    async fn test_edit_empty_old_string_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("f.txt");
        tokio_fs::write(&path, "abc").await.unwrap();

        let tool = EditTool;
        let err = tool
            .execute(&json!({
                "path": path.to_str().unwrap(),
                "old_string": "",
                "new_string": "x",
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn test_is_not_auto_approved() {
        assert!(!EditTool.is_auto_approved());
    }
}
