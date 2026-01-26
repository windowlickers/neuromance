//! # neuromance-tools
//!
//! Tool execution framework for Neuromance LLM library.
//!
//! This crate provides a flexible system for defining, registering, and executing tools
//! that can be called by Large Language Models (LLMs). It includes built-in tools,
//! support for custom tool implementations, and integration with the Model Context Protocol (MCP).
//!
//! ## Core Components
//!
//! - [`ToolImplementation`]: Trait for defining custom tools with execution logic
//! - [`ToolRegistry`]: Thread-safe registry for managing tool definitions
//! - [`ToolExecutor`]: High-level interface for tool execution with argument parsing
//! - [`mcp`]: Model Context Protocol client and server integration
//!
//! ## Built-in Tools
//!
//! - [`BooleanTool`]: Returns boolean values based on conditions
//! - [`ThinkTool`]: Enables chain-of-thought reasoning for LLMs
//! - [`TodoWriteTool`] / [`TodoReadTool`]: Task management tools for agents
//!
//! ## Example: Creating and Executing a Custom Tool
//!
//! ```rust
//! use neuromance_tools::{ToolImplementation, ToolExecutor};
//! use neuromance_common::tools::{Tool, Function};
//! use serde_json::{json, Value};
//! use async_trait::async_trait;
//! use anyhow::Result;
//!
//! // Define a custom tool
//! struct GreetingTool;
//!
//! #[async_trait]
//! impl ToolImplementation for GreetingTool {
//!     fn get_definition(&self) -> Tool {
//!         Tool {
//!             r#type: "function".to_string(),
//!             function: Function {
//!                 name: "greet".to_string(),
//!                 description: "Greet a person by name".to_string(),
//!                 parameters: json!({
//!                     "type": "object",
//!                     "properties": {
//!                         "name": {
//!                             "type": "string",
//!                             "description": "The person's name"
//!                         }
//!                     },
//!                     "required": ["name"]
//!                 }),
//!             },
//!         }
//!     }
//!
//!     async fn execute(&self, args: &Value) -> Result<String> {
//!         let name = args["name"].as_str().unwrap_or("stranger");
//!         Ok(format!("Hello, {}!", name))
//!     }
//!
//!     fn is_auto_approved(&self) -> bool {
//!         true // This tool can execute without user approval
//!     }
//! }
//!
//! # async fn example() -> Result<()> {
//! // Register and use the tool
//! let mut executor = ToolExecutor::new();
//! executor.add_tool(GreetingTool);
//!
//! // Get all tool definitions to send to the LLM
//! let tools = executor.get_all_tools();
//! # Ok(())
//! # }
//! ```
//!
//! ## MCP Integration
//!
//! The [`mcp`] module provides Model Context Protocol support for connecting
//! to external tool servers and exposing tools via MCP:
//!
//! ```rust,ignore
//! use neuromance_tools::mcp::{McpManager, McpClientConfig};
//!
//! # async fn example() -> Result<()> {
//! let mut manager = McpManager::new();
//!
//! // Connect to an MCP server
//! let config = McpClientConfig {
//!     name: "filesystem".to_string(),
//!     command: "npx".to_string(),
//!     args: vec!["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
//!         .into_iter()
//!         .map(String::from)
//!         .collect(),
//!     env: None,
//! };
//!
//! manager.add_server(config).await?;
//! let tools = manager.get_all_tools().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Tool Auto-Approval
//!
//! Tools can be marked as "auto-approved" via the [`ToolImplementation::is_auto_approved`]
//! method. Auto-approved tools execute without requiring user confirmation, which is
//! useful for safe, read-only operations.
//!
//! ## Thread Safety
//!
//! The [`ToolRegistry`] uses `DashMap` for concurrent access, making it safe to use
//! from multiple async tasks without additional synchronization.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use dashmap::DashMap;
use serde_json::Value;

use neuromance_common::tools::{FunctionCall, Tool, ToolCall};

mod bool_tool;
pub mod generic;
pub mod mcp;
pub mod proxy;
mod think_tool;
mod todo_tool;
pub use bool_tool::BooleanTool;
pub use think_tool::ThinkTool;
pub use todo_tool::{TodoReadTool, TodoWriteTool, create_todo_tools};

#[async_trait]
pub trait ToolImplementation: Send + Sync {
    fn get_definition(&self) -> Tool;

    async fn execute(&self, args: &Value) -> Result<String>;

    fn is_auto_approved(&self) -> bool {
        false
    }
}

pub struct ToolRegistry {
    tools: Arc<DashMap<String, Arc<dyn ToolImplementation>>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: Arc::new(DashMap::new()),
        }
    }

    pub fn register(&self, tool: Arc<dyn ToolImplementation>) {
        let name = tool.get_definition().function.name;
        self.tools.insert(name, tool);
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn ToolImplementation>> {
        self.tools.get(name).map(|r| r.value().clone())
    }

    #[must_use]
    pub fn get_all_definitions(&self) -> Vec<Tool> {
        self.tools.iter().map(|t| t.get_definition()).collect()
    }

    #[must_use]
    pub fn is_tool_auto_approved(&self, name: &str) -> bool {
        self.tools.get(name).is_some_and(|t| t.is_auto_approved())
    }

    pub fn remove(&mut self, name: &str) -> Option<Arc<dyn ToolImplementation>> {
        self.tools.remove(name).map(|(_, tool)| tool)
    }

    pub fn clear(&mut self) {
        self.tools.clear();
    }

    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    #[must_use]
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.key().clone()).collect()
    }
}

pub struct ToolExecutor {
    registry: ToolRegistry,
}

impl ToolExecutor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: ToolRegistry::new(),
        }
    }

    pub fn add_tool<T: ToolImplementation + 'static>(&mut self, tool: T) {
        self.registry.register(Arc::new(tool));
    }

    pub fn add_tool_arc(&mut self, tool: Arc<dyn ToolImplementation>) {
        self.registry.register(tool);
    }

    #[must_use]
    pub fn has_tool(&self, name: &str) -> bool {
        self.registry.contains(name)
    }

    #[must_use]
    pub fn get_all_tools(&self) -> Vec<Tool> {
        self.registry.get_all_definitions()
    }

    #[must_use]
    pub fn is_tool_auto_approved(&self, name: &str) -> bool {
        self.registry.is_tool_auto_approved(name)
    }

    pub fn remove_tool(&mut self, name: &str) -> Option<Arc<dyn ToolImplementation>> {
        self.registry.remove(name)
    }

    pub fn reset_tools(&mut self) {
        self.registry.clear();
    }

    /// Execute a tool call.
    ///
    /// # Errors
    /// Returns an error if the tool is not found or if execution fails.
    pub async fn execute_tool(&self, tool_call: &ToolCall) -> Result<String> {
        let function = &tool_call.function;

        let tool = self
            .registry
            .get(&function.name)
            .ok_or_else(|| anyhow::anyhow!("Unknown tool: '{}'", function.name))?;

        let args = Self::parse_arguments(function)?;

        // Execute the tool
        tool.execute(&args).await
    }

    fn parse_arguments(arguments: &FunctionCall) -> Result<Value> {
        let json = arguments.arguments_json();
        if json == "{}" {
            Ok(Value::Object(serde_json::Map::new()))
        } else {
            serde_json::from_str(&json).or_else(|_| Ok(Value::String(json)))
        }
    }
}

impl Default for ToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use serde_json::json;

    fn fc(args: Vec<String>) -> FunctionCall {
        FunctionCall {
            name: "test".to_string(),
            arguments: args,
        }
    }

    #[test]
    fn test_parse_arguments_empty() {
        let result = ToolExecutor::parse_arguments(&fc(vec![])).unwrap();
        assert_eq!(result, json!({}));
    }

    #[test]
    fn test_parse_arguments_single_json_object() {
        let f = fc(vec![r#"{"key": "value", "number": 42}"#.to_string()]);
        let result = ToolExecutor::parse_arguments(&f).unwrap();
        assert_eq!(result, json!({"key": "value", "number": 42}));
    }

    #[test]
    fn test_parse_arguments_single_json_array() {
        let f = fc(vec![r#"["item1", "item2", "item3"]"#.to_string()]);
        let result = ToolExecutor::parse_arguments(&f).unwrap();
        assert_eq!(result, json!(["item1", "item2", "item3"]));
    }

    #[test]
    fn test_parse_arguments_single_string_fallback() {
        let f = fc(vec!["plain text argument".to_string()]);
        let result = ToolExecutor::parse_arguments(&f).unwrap();
        assert_eq!(result, json!("plain text argument"));
    }

    #[test]
    fn test_parse_arguments_single_invalid_json_fallback() {
        let f = fc(vec![r#"{"incomplete json"#.to_string()]);
        let result = ToolExecutor::parse_arguments(&f).unwrap();
        assert_eq!(result, json!(r#"{"incomplete json"#));
    }

    #[test]
    fn test_parse_arguments_multiple_fragments_joined() {
        let f = fc(vec![r#"{"key":"#.to_string(), r#""value"}"#.to_string()]);
        let result = ToolExecutor::parse_arguments(&f).unwrap();
        assert_eq!(result, json!({"key": "value"}));
    }

    #[test]
    fn test_parse_arguments_single_number_string() {
        let f = fc(vec!["42".to_string()]);
        let result = ToolExecutor::parse_arguments(&f).unwrap();
        assert_eq!(result, json!(42));
    }

    #[test]
    fn test_parse_arguments_single_boolean_string() {
        let f = fc(vec!["true".to_string()]);
        let result = ToolExecutor::parse_arguments(&f).unwrap();
        assert_eq!(result, json!(true));
    }
}
