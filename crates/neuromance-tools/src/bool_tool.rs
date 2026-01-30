use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::ToolImplementation;
use neuromance_common::tools::{Function, Parameters, Property, Tool};

/// A tool that returns a boolean result based on the agent's analysis
/// Used for binary decisions like goal verification
pub struct BooleanTool;

#[async_trait]
impl ToolImplementation for BooleanTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "result".to_string(),
            Property::boolean("The boolean result (true or false)"),
        );
        properties.insert(
            "reason".to_string(),
            Property::string("A brief explanation for the boolean result"),
        );

        Tool {
            r#type: "function".to_string(),
            function: Function {
                name: "return_bool".to_string(),
                description: "Return a boolean result (true/false) with an explanation. Use this to provide definitive yes/no answers.".to_string(),
                parameters: Parameters::new(properties, vec!["result".into(), "reason".into()]).into(),
            },
        }
    }

    async fn execute(&self, args: &Value) -> Result<String> {
        let obj = args
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Expected object arguments"))?;

        let result = obj
            .get("result")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| anyhow::anyhow!("Missing or invalid 'result' parameter"))?;

        let reason = obj
            .get("reason")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing or invalid 'reason' parameter"))?;

        // Format the result in a structured way
        Ok(format!(
            "RESULT: {}\nREASON: {}",
            if result { "TRUE" } else { "FALSE" },
            reason
        ))
    }

    fn is_auto_approved(&self) -> bool {
        true // Bool tool is safe and can be auto-approved
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_bool_tool_true() {
        let tool = BooleanTool;

        let args = json!({
            "result": true,
            "reason": "The goal was successfully achieved"
        });

        let result = tool.execute(&args).await.unwrap();
        assert!(result.contains("RESULT: TRUE"));
        assert!(result.contains("REASON: The goal was successfully achieved"));
    }

    #[tokio::test]
    async fn test_bool_tool_false() {
        let tool = BooleanTool;

        let args = json!({
            "result": false,
            "reason": "The task failed due to missing dependencies"
        });

        let result = tool.execute(&args).await.unwrap();
        assert!(result.contains("RESULT: FALSE"));
        assert!(result.contains("REASON: The task failed due to missing dependencies"));
    }

    #[tokio::test]
    async fn test_bool_tool_missing_result() {
        let tool = BooleanTool;

        let args = json!({
            "reason": "Some reason"
        });

        let result = tool.execute(&args).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Missing or invalid 'result' parameter")
        );
    }

    #[tokio::test]
    async fn test_bool_tool_missing_reason() {
        let tool = BooleanTool;

        let args = json!({
            "result": true
        });

        let result = tool.execute(&args).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Missing or invalid 'reason' parameter")
        );
    }
}
