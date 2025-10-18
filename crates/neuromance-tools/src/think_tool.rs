use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::ToolImplementation;
use neuromance_common::tools::{Function, Property, Tool};

/// A tool that allows the agent to record thoughts and reasoning
/// Used for making the agent's thinking process visible in the context
pub struct ThinkTool;

#[async_trait]
impl ToolImplementation for ThinkTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "thought".to_string(),
            Property {
                prop_type: "string".to_string(),
                description: "The agent's internal thought or reasoning process".to_string(),
            },
        );

        Tool {
            r#type: "function".to_string(),
            function: Function {
                name: "think".to_string(),
                description: "Record your internal thoughts and reasoning. Use this to make your thinking process visible and preserve it in the conversation context.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": properties,
                    "required": vec!["thought".to_string()],
                }),
            },
        }
    }

    async fn execute(&self, args: &Value) -> Result<String> {
        let obj = args
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Expected object arguments"))?;

        let thought = obj
            .get("thought")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing or invalid 'thought' parameter"))?;

        // Return the thought in a structured way that makes it clear this is internal reasoning
        Ok(format!("THOUGHT RECORDED: {thought}"))
    }

    fn is_auto_approved(&self) -> bool {
        true // Think tool is safe and can be auto-approved
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_think_tool() {
        let tool = ThinkTool;

        let args = json!({
            "thought": "I need to analyze the dependencies before proceeding with the refactoring"
        });

        let result = tool.execute(&args).await.unwrap();
        assert!(result.contains("THOUGHT RECORDED:"));
        assert!(
            result.contains(
                "I need to analyze the dependencies before proceeding with the refactoring"
            )
        );
    }

    #[tokio::test]
    async fn test_think_tool_missing_thought() {
        let tool = ThinkTool;

        let args = json!({});

        let result = tool.execute(&args).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Missing or invalid 'thought' parameter")
        );
    }
}
