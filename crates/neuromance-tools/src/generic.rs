use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::{ToolError, ToolImplementation};
use neuromance_common::tools::{Function, Parameters, Tool};

pub struct CurrentTimeTool;
#[async_trait]
impl ToolImplementation for CurrentTimeTool {
    fn get_definition(&self) -> Tool {
        Tool::builder()
            .function(Function {
                name: "get_current_time".to_string(),
                description: "Get the current date and time in UTC format. Takes no parameters."
                    .to_string(),
                parameters: Parameters::new(HashMap::new(), vec![]).into(),
            })
            .build()
    }

    async fn execute(&self, _args: &Value) -> Result<String, ToolError> {
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
