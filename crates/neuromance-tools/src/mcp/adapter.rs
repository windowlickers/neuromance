use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::ToolImplementation;
use neuromance_common::{Function, ObjectSchema, Parameters, Property, Tool};
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

/// Extract a list of string-keyed `Property` values from a JSON schema object.
fn extract_properties(schema: &Value) -> HashMap<String, Property> {
    schema
        .get("properties")
        .and_then(|p| p.as_object())
        .map(|p| {
            p.iter()
                .map(|(k, v)| (k.clone(), convert_json_schema_property(v)))
                .collect()
        })
        .unwrap_or_default()
}

/// Extract the `required` array from a JSON schema object.
fn extract_required(schema: &Value) -> Vec<String> {
    schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Convert a single JSON schema property value into a `Property`.
fn convert_json_schema_property(value: &Value) -> Property {
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

    match prop_type.as_str() {
        "number" | "integer" => Property::number(description),
        "boolean" => Property::boolean(description),
        "array" => {
            let items_schema = value.get("items").map_or_else(
                || ObjectSchema::new(HashMap::new(), vec![]),
                |items| ObjectSchema::new(extract_properties(items), extract_required(items)),
            );
            Property::array(description, items_schema)
        }
        "object" => Property::object(
            description,
            extract_properties(value),
            extract_required(value),
        ),
        // Preserve the original type string for unknown types
        _ => Property {
            prop_type,
            description,
            enum_values: None,
            items: None,
            properties: None,
            required: None,
        },
    }
}

#[async_trait]
impl ToolImplementation for McpToolAdapter {
    fn get_definition(&self) -> Tool {
        let schema = serde_json::Value::Object(self.mcp_tool.input_schema.as_ref().clone());
        let properties = extract_properties(&schema);
        let required = extract_required(&schema);

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
                parameters: Parameters::new(properties, required).into(),
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
