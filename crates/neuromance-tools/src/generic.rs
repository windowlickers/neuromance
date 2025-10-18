use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::ToolImplementation;
use neuromance_common::{Function, Property, Tool};

pub struct CurrentTimeTool;
#[async_trait]
impl ToolImplementation for CurrentTimeTool {
    fn get_definition(&self) -> Tool {
        Tool {
            r#type: "function".to_string(),
            function: Function {
                name: "get_current_time".to_string(),
                description: "Get the current date and time in UTC format. Takes no parameters."
                    .to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": [],
                }),
            },
        }
    }

    async fn execute(&self, _args: &Value) -> Result<String> {
        let now: DateTime<Utc> = Utc::now();
        Ok(format!(
            "Current time: {}",
            now.format("%Y-%m-%d %H:%M:%S UTC")
        ))
    }

    fn is_auto_approved(&self) -> bool {
        true // Time tool is safe and can be auto-approved
    }
}

pub struct CalculatorTool;
#[async_trait]
impl ToolImplementation for CalculatorTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "expression".to_string(),
            Property {
                prop_type: "string".to_string(),
                description: "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                    .to_string(),
            },
        );

        Tool {
            r#type: "function".to_string(),
            function: Function {
                name: "calculate".to_string(),
                description: "Evaluate a mathematical expression and return the result".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": properties,
                    "required": vec!["expression".to_string()],
                }),
            },
        }
    }

    async fn execute(&self, args: &Value) -> Result<String> {
        let expression = args
            .as_object()
            .and_then(|obj| obj.get("expression"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'expression' parameter"))?;

        // For demo purposes, just handle simple cases
        let result = match expression {
            "2 + 2" => 4.0,
            "10 * 5" => 50.0,
            "100 / 4" => 25.0,
            _ => return Err(anyhow::anyhow!("Unsupported expression: {expression}")),
        };

        Ok(format!("{expression} = {result}"))
    }

    fn is_auto_approved(&self) -> bool {
        false // Calculator requires approval
    }
}
