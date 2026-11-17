use async_trait::async_trait;
use rmcp::model::{Content, RawContent, ResourceContents, Tool as McpTool};
use serde_json::Value;
use std::sync::Arc;
use tracing::debug;

use crate::{ToolError, ToolImplementation};
use neuromance_common::tools::{Function, Tool};

use super::client::McpClientWrapper;

/// Separator between the server id and the tool name in an advertised tool name.
///
/// Provider APIs restrict function names to `^[a-zA-Z0-9_-]{1,64}$`, so the
/// separator cannot be `.` or `/`.
pub const NAME_SEPARATOR: &str = "__";

/// Adapter that wraps an MCP tool as a `ToolImplementation`
pub struct McpToolAdapter {
    pub server_id: String,
    pub tool_name: String,
    pub client: Arc<McpClientWrapper>,
    pub mcp_tool: McpTool,
    pub auto_approved: bool,
}

impl McpToolAdapter {
    #[must_use]
    pub fn new(server_id: String, client: Arc<McpClientWrapper>, mcp_tool: McpTool) -> Self {
        let tool_name = mcp_tool.name.to_string();
        let auto_approved = client.server_config.auto_approve;

        Self {
            server_id,
            tool_name,
            client,
            mcp_tool,
            auto_approved,
        }
    }

    /// The name advertised to the LLM: `<server_id>__<tool_name>`, with every
    /// character a provider rejects replaced by `_`.
    #[must_use]
    pub fn full_name(&self) -> String {
        qualified_name(&self.server_id, &self.tool_name)
    }
}

/// Build the provider-safe name for `tool` on `server`.
#[must_use]
pub fn qualified_name(server: &str, tool: &str) -> String {
    format!("{server}{NAME_SEPARATOR}{tool}")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Flatten an MCP tool result into the single string the LLM sees.
///
/// Text items are joined with newlines. Non-text items become a short
/// placeholder so the model knows something was returned.
fn render_content(content: &[Content]) -> String {
    if content.is_empty() {
        return "No content returned".to_string();
    }
    content
        .iter()
        .map(|item| match &item.raw {
            RawContent::Text(text) => text.text.clone(),
            RawContent::Image(image) => format!(
                "[Image: {} bytes, type: {}]",
                image.data.len(),
                image.mime_type
            ),
            RawContent::Audio(audio) => format!(
                "[Audio: {} bytes, type: {}]",
                audio.data.len(),
                audio.mime_type
            ),
            RawContent::Resource(resource) => match &resource.resource {
                ResourceContents::TextResourceContents { text, .. } => text.clone(),
                ResourceContents::BlobResourceContents { uri, .. } => format!("[Resource: {uri}]"),
            },
            RawContent::ResourceLink(link) => format!("[Resource: {}]", link.uri),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[async_trait]
impl ToolImplementation for McpToolAdapter {
    fn get_definition(&self) -> Tool {
        let description = self.mcp_tool.description.as_ref().map_or_else(
            || {
                format!(
                    "MCP tool '{}' from server '{}'",
                    self.tool_name, self.server_id
                )
            },
            ToString::to_string,
        );

        Tool::builder()
            .function(Function {
                name: self.full_name(),
                description,
                parameters: Value::Object(self.mcp_tool.input_schema.as_ref().clone()),
            })
            .build()
    }

    async fn execute(&self, args: &Value) -> Result<String, ToolError> {
        debug!(server = %self.server_id, tool = %self.tool_name, "executing MCP tool");

        let result = self
            .client
            .call_tool(&self.tool_name, args.clone())
            .await
            .map_err(|e| ToolError::Execution(e.into()))?;

        let rendered = render_content(&result.content);
        if result.is_error.unwrap_or(false) {
            return Err(ToolError::execution(format!(
                "MCP tool '{}' on server '{}' failed: {rendered}",
                self.tool_name, self.server_id
            )));
        }
        Ok(rendered)
    }

    fn is_auto_approved(&self) -> bool {
        self.auto_approved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qualified_name_is_provider_safe() {
        assert_eq!(qualified_name("fs", "read_file"), "fs__read_file");
        assert_eq!(
            qualified_name("my.server", "files/read"),
            "my_server__files_read"
        );
        assert_eq!(qualified_name("srv-1", "do-it"), "srv-1__do-it");
    }

    #[test]
    fn test_render_content_joins_every_text_item() {
        let content = vec![Content::text("first"), Content::text("second")];
        assert_eq!(render_content(&content), "first\nsecond");
    }

    #[test]
    fn test_render_content_reports_empty_result() {
        assert_eq!(render_content(&[]), "No content returned");
    }

    #[test]
    fn test_render_content_inlines_text_resources() {
        let content = vec![Content::embedded_text("file:///a.txt", "body")];
        assert_eq!(render_content(&content), "body");
    }
}
