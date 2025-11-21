use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

use crate::ToolImplementation;
use neuromance_common::{Function, Property, Tool};
use rmcp::model::Tool as McpTool;

use super::client::McpClientWrapper;

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

    /// Get the full tool name (`server_id.tool_name`)
    #[must_use]
    pub fn full_name(&self) -> String {
        format!("{}.{}", self.server_id, self.tool_name)
    }
}

#[async_trait]
impl ToolImplementation for McpToolAdapter {
    fn get_definition(&self) -> Tool {
        // Convert MCP tool definition to neuromancer Tool
        let mut properties = std::collections::HashMap::new();

        // Parse the MCP tool's input schema
        if let Some(props) = self
            .mcp_tool
            .input_schema
            .get("properties")
            .and_then(|p| p.as_object())
        {
            for (key, value) in props {
                let prop_type = value
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("string")
                    .to_string();

                let description = value
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string();

                properties.insert(
                    key.clone(),
                    Property {
                        prop_type,
                        description,
                    },
                );
            }
        }

        let required: Vec<String> = self
            .mcp_tool
            .input_schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Tool {
            r#type: "function".to_string(),
            function: Function {
                name: self.full_name(),
                description: self.mcp_tool.description.as_ref().map_or_else(
                    || {
                        format!(
                            "MCP tool '{}' from server '{}'",
                            self.tool_name, self.server_id
                        )
                    },
                    std::string::ToString::to_string,
                ),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }),
            },
        }
    }

    async fn execute(&self, args: &Value) -> Result<String> {
        log::info!(
            "Executing MCP tool '{}' on server '{}' with args: {}",
            self.tool_name,
            self.server_id,
            serde_json::to_string_pretty(args)?
        );

        // Call the tool through the MCP client
        let result = self.client.call_tool(&self.tool_name, args.clone()).await?;

        // Check if there was an error
        if result.is_error.unwrap_or(false) {
            return Err(anyhow::anyhow!(
                "MCP tool execution failed: {:?}",
                result.content
            ));
        }

        // Extract the content from the response
        let content = result.content.into_iter().next().map_or_else(
            || "No content returned".to_string(),
            |content| {
                content.as_text().map_or_else(
                    || {
                        content.as_image().map_or_else(
                            || {
                                content.as_resource().map_or_else(
                                    || "[Unknown content type]".to_string(),
                                    |resource| match &resource.resource {
                                        rmcp::model::ResourceContents::TextResourceContents {
                                            uri,
                                            ..
                                        }
                                        | rmcp::model::ResourceContents::BlobResourceContents {
                                            uri,
                                            ..
                                        } => {
                                            format!("[Resource: {uri}]")
                                        }
                                    },
                                )
                            },
                            |image_content| {
                                format!(
                                    "[Image: {} bytes, type: {}]",
                                    image_content.data.len(),
                                    image_content.mime_type
                                )
                            },
                        )
                    },
                    |text_content| text_content.text.clone(),
                )
            },
        );

        Ok(content)
    }

    fn is_auto_approved(&self) -> bool {
        self.auto_approved
    }
}
