use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;

use crate::ToolImplementation;
use common::{Function, Property, Tool};

pub struct ReadFileTool {
    root_directory: PathBuf,
}

impl ReadFileTool {
    pub fn new(root_directory: PathBuf) -> Self {
        Self {
            root_directory: root_directory.canonicalize().unwrap_or(root_directory),
        }
    }

    fn validate_path(&self, file_path: &Path) -> Result<PathBuf> {
        // If path is relative, resolve it from the root directory
        let resolved_path = if file_path.is_absolute() {
            file_path.to_path_buf()
        } else {
            self.root_directory.join(file_path)
        };

        let canonical_path = resolved_path
            .canonicalize()
            .with_context(|| format!("Failed to canonicalize path: {}", resolved_path.display()))?;

        let canonical_root = &self.root_directory;
        if !canonical_path.starts_with(canonical_root) {
            return Err(anyhow::anyhow!(
                "File path must be within the root directory ({}): {}",
                canonical_root.display(),
                canonical_path.display()
            ));
        }

        Ok(canonical_path)
    }

    async fn read_text_file(
        &self,
        path: &Path,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<String> {
        let file = fs::File::open(path)
            .with_context(|| format!("Failed to open file: {}", path.display()))?;
        let reader = BufReader::new(file);

        let mut lines: Vec<String> = Vec::new();
        let mut current_line = 0;

        let start_line = offset.unwrap_or(0);
        let max_lines = limit.unwrap_or(2000);

        for line_result in reader.lines() {
            if current_line >= start_line && lines.len() < max_lines {
                let line = line_result
                    .with_context(|| format!("Failed to read line {} from file", current_line))?;

                // Truncate lines longer than 2000 characters
                let truncated_line = if line.len() > 2000 {
                    format!("{}... [truncated]", &line[..2000])
                } else {
                    line
                };

                lines.push(format!("{:>6}\t{}", current_line + 1, truncated_line));
            }
            current_line += 1;

            if lines.len() >= max_lines {
                break;
            }
        }

        if lines.is_empty() && start_line > 0 {
            return Err(anyhow::anyhow!(
                "Offset {} is beyond the file's {} lines",
                start_line,
                current_line
            ));
        }

        Ok(lines.join("\n"))
    }

    async fn read_binary_file(&self, path: &Path) -> Result<String> {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase());

        match extension.as_deref() {
            Some("png") | Some("jpg") | Some("jpeg") | Some("gif") | Some("webp") | Some("svg")
            | Some("bmp") => Ok(format!(
                "[Image file: {}]\nNote: Image viewing is not yet implemented in the Rust version.",
                path.display()
            )),
            Some("pdf") => Ok(format!(
                "[PDF file: {}]\nNote: PDF reading is not yet implemented in the Rust version.",
                path.display()
            )),
            _ => {
                let metadata = fs::metadata(path)?;
                Ok(format!(
                    "[Binary file: {}, size: {} bytes]",
                    path.display(),
                    metadata.len()
                ))
            }
        }
    }
}

#[async_trait]
impl ToolImplementation for ReadFileTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();

        properties.insert(
            "path".to_string(),
            Property {
                prop_type: "string".to_string(),
                description: "The path to the file to read. Can be absolute (e.g., '/home/user/project/file.txt') or relative to the current working directory (e.g., 'file.txt', 'src/main.rs').".to_string(),
            },
        );

        properties.insert(
            "offset".to_string(),
            Property {
                prop_type: "number".to_string(),
                description: "Optional: For text files, the 0-based line number to start reading from. Requires 'limit' to be set. Use for paginating through large files.".to_string(),
            },
        );

        properties.insert(
            "limit".to_string(),
            Property {
                prop_type: "number".to_string(),
                description: "Optional: For text files, maximum number of lines to read. Use with 'offset' to paginate through large files. If omitted, reads the entire file (up to 2000 lines).".to_string(),
            },
        );

        Tool {
            r#type: "function".to_string(),
            function: Function {
                name: "read_file".to_string(),
                description: "Reads and returns the content of a specified file from the local filesystem. Handles text, images (PNG, JPG, GIF, WEBP, SVG, BMP), and PDF files. For text files, it can read specific line ranges.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": properties,
                    "required": vec!["path".to_string()],
                }),
            },
        }
    }

    async fn execute(&self, args: &Value) -> Result<String> {
        let obj = args
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Arguments must be an object"))?;

        let path_str = obj
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'path' parameter"))?;

        let offset = obj
            .get("offset")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let limit = obj
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        // Validate parameters
        if let Some(o) = offset {
            if o == 0 && limit.is_none() {
                return Err(anyhow::anyhow!(
                    "When offset is provided, limit must also be provided"
                ));
            }
        }

        if let Some(l) = limit {
            if l == 0 {
                return Err(anyhow::anyhow!("Limit must be a positive number"));
            }
        }

        let path = Path::new(path_str);
        let validated_path = self.validate_path(path)?;

        // Check if file exists
        if !validated_path.exists() {
            return Err(anyhow::anyhow!(
                "File not found: {}",
                validated_path.display()
            ));
        }

        // Check if it's a file (not a directory)
        if !validated_path.is_file() {
            return Err(anyhow::anyhow!(
                "Path is not a file: {}",
                validated_path.display()
            ));
        }

        // Determine if file is text or binary
        let is_text = match validated_path.extension().and_then(|ext| ext.to_str()) {
            Some(ext) => matches!(
                ext.to_lowercase().as_str(),
                "txt"
                    | "md"
                    | "rs"
                    | "toml"
                    | "json"
                    | "yaml"
                    | "yml"
                    | "xml"
                    | "html"
                    | "css"
                    | "js"
                    | "ts"
                    | "py"
                    | "sh"
                    | "c"
                    | "cpp"
                    | "h"
                    | "hpp"
                    | "java"
                    | "go"
                    | "rb"
                    | "php"
                    | "sql"
                    | "conf"
                    | "cfg"
                    | "ini"
                    | "log"
            ),
            None => {
                // Try to detect if it's text by reading first few bytes
                let mut file = fs::File::open(&validated_path)?;
                let mut buffer = [0; 512];
                use std::io::Read;
                let bytes_read = file.read(&mut buffer)?;
                buffer[..bytes_read]
                    .iter()
                    .all(|&b| b.is_ascii() || b >= 128)
            }
        };

        if is_text {
            self.read_text_file(&validated_path, offset, limit).await
        } else {
            self.read_binary_file(&validated_path).await
        }
    }

    fn is_auto_approved(&self) -> bool {
        false // sometimes it gets this wrong, approve it
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_read_text_file() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "Line 1\nLine 2\nLine 3\nLine 4\nLine 5").unwrap();

        let tool = ReadFileTool::new(temp_dir.path().to_path_buf());

        // Test reading entire file
        let args = json!({
            "path": test_file.to_str().unwrap()
        });
        let result = tool.execute(&args).await.unwrap();
        assert!(result.contains("Line 1"));
        assert!(result.contains("Line 5"));

        // Test reading with offset and limit
        let args = json!({
            "path": test_file.to_str().unwrap(),
            "offset": 1,
            "limit": 2
        });
        let result = tool.execute(&args).await.unwrap();
        assert!(result.contains("Line 2"));
        assert!(result.contains("Line 3"));
        assert!(!result.contains("Line 1"));
        assert!(!result.contains("Line 4"));
    }

    #[tokio::test]
    async fn test_validate_path() {
        let temp_dir = TempDir::new().unwrap();
        let tool = ReadFileTool::new(temp_dir.path().to_path_buf());

        // Test relative path (should now work)
        let relative_file = temp_dir.path().join("relative.txt");
        fs::write(&relative_file, "Relative content").unwrap();
        let args = json!({
            "path": "relative.txt"
        });
        let result = tool.execute(&args).await;
        assert!(result.is_ok());
        assert!(result.unwrap().contains("Relative content"));

        // Test path outside root
        let args = json!({
            "path": "/etc/passwd"
        });
        let result = tool.execute(&args).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("within the root directory")
        );
    }
}
