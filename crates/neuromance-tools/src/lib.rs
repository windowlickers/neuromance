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
//! - [`ReadTool`]: Read UTF-8 files with optional line offset/limit
//! - [`WriteTool`]: Create or overwrite a file at an absolute path
//! - [`EditTool`]: String replacement on a file, single or batched
//! - [`BashTool`]: Execute a shell command via `sh -c` with a timeout
//! - [`GrepTool`]: Search file contents by regex, respecting `.gitignore`
//! - [`FindTool`]: Find files by glob pattern, respecting `.gitignore`
//! - [`LsTool`]: List the entries of a directory
//!
//! ## Example: Creating and Executing a Custom Tool
//!
//! ```rust
//! use neuromance_tools::{ToolImplementation, ToolExecutor, ToolError};
//! use neuromance_common::tools::{Tool, Function};
//! use serde_json::{json, Value};
//! use async_trait::async_trait;
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
//!     async fn execute(&self, args: &Value) -> Result<String, ToolError> {
//!         let name = args["name"].as_str().unwrap_or("stranger");
//!         Ok(format!("Hello, {}!", name))
//!     }
//!
//!     fn is_auto_approved(&self) -> bool {
//!         true // This tool can execute without user approval
//!     }
//! }
//!
//! # async fn example() {
//! // Register and use the tool
//! let mut executor = ToolExecutor::new();
//! executor.add_tool(GreetingTool);
//!
//! // Get all tool definitions to send to the LLM
//! let tools = executor.get_all_tools();
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

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use serde_json::Value;

use neuromance_common::tools::{Tool, ToolCall};

mod bash_tool;
mod edit_tool;
mod error;
pub mod factory;
mod find_tool;
pub mod generic;
mod grep_tool;
mod ls_tool;
pub mod mcp;
pub mod proxy;
mod read_tool;
mod truncate;
mod write_tool;
pub use bash_tool::{BashTool, BashToolFactory};
pub use edit_tool::{EditTool, EditToolFactory};
pub use error::{ToolError, ToolExecutorError};
pub use factory::{ToolConfig, ToolFactory, ToolFactoryRegistry};
pub use find_tool::{FindTool, FindToolFactory};
pub use grep_tool::{GrepTool, GrepToolFactory};
pub use ls_tool::{LsTool, LsToolFactory};
pub use read_tool::{ReadTool, ReadToolFactory};
pub use write_tool::{WriteTool, WriteToolFactory};

/// Resolves an optional `path` argument for the search tools (`grep`, `find`,
/// `ls`) to an absolute directory or file.
///
/// When omitted, defaults to the current working directory. When provided it
/// must be absolute, matching the path discipline of the other built-in tools.
///
/// # Errors
/// Returns [`ToolError::InvalidArguments`] for a relative path, or
/// [`ToolError::Execution`] if the current directory cannot be determined.
pub(crate) fn resolve_search_path(raw: Option<&str>) -> Result<PathBuf, ToolError> {
    match raw {
        Some(p) => {
            let path = PathBuf::from(p);
            if !path.is_absolute() {
                return Err(ToolError::InvalidArguments(format!(
                    "'path' must be absolute, got: {}",
                    path.display()
                )));
            }
            Ok(path)
        }
        None => std::env::current_dir()
            .map_err(|e| ToolError::execution(format!("cannot determine current directory: {e}"))),
    }
}

/// Parses an optional non-negative integer argument, returning `default` when
/// absent and an error when present but not a non-negative integer.
///
/// # Errors
/// Returns [`ToolError::InvalidArguments`] if the value is the wrong type.
pub(crate) fn opt_u64(v: Option<&Value>, name: &str, default: u64) -> Result<u64, ToolError> {
    match v {
        None | Some(Value::Null) => Ok(default),
        Some(val) => val.as_u64().ok_or_else(|| {
            ToolError::InvalidArguments(format!("'{name}' must be a non-negative integer"))
        }),
    }
}

#[async_trait]
pub trait ToolImplementation: Send + Sync {
    fn get_definition(&self) -> Tool;

    async fn execute(&self, args: &Value) -> Result<String, ToolError>;

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

    #[must_use]
    pub fn remove(&self, name: &str) -> Option<Arc<dyn ToolImplementation>> {
        self.tools.remove(name).map(|(_, tool)| tool)
    }

    pub fn clear(&self) {
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

    #[must_use]
    pub fn remove_tool(&self, name: &str) -> Option<Arc<dyn ToolImplementation>> {
        self.registry.remove(name)
    }

    pub fn reset_tools(&self) {
        self.registry.clear();
    }

    /// Execute a tool call.
    ///
    /// Cancellation is the caller's responsibility — wrap this future in a
    /// `tokio::select!` against your `CancellationToken`. Dropping the future
    /// cancels the in-flight tool at its next `.await`.
    ///
    /// # Errors
    /// Returns [`ToolExecutorError::UnknownTool`] if the tool is not found,
    /// or [`ToolExecutorError::Tool`] if execution fails.
    pub async fn execute_tool(&self, tool_call: &ToolCall) -> Result<String, ToolExecutorError> {
        let function = &tool_call.function;
        self.execute_named(&function.name, function.arguments_json())
            .await
    }

    /// Execute a tool by name with raw JSON-encoded arguments.
    ///
    /// The shared dispatch path behind [`execute_tool`](Self::execute_tool):
    /// callers that don't hold a full [`ToolCall`] — e.g. a gRPC server
    /// reconstructing a request from name + arguments — reuse the same registry
    /// lookup and argument parsing.
    ///
    /// Cancellation is the caller's responsibility (see
    /// [`execute_tool`](Self::execute_tool)).
    ///
    /// # Errors
    /// Returns [`ToolExecutorError::UnknownTool`] if the tool is not found,
    /// or [`ToolExecutorError::Tool`] if execution fails.
    pub async fn execute_named(
        &self,
        name: &str,
        arguments_json: &str,
    ) -> Result<String, ToolExecutorError> {
        let tool = self
            .registry
            .get(name)
            .ok_or_else(|| ToolExecutorError::UnknownTool(name.to_owned()))?;

        let args = Self::parse_arguments(arguments_json);

        Ok(tool.execute(&args).await?)
    }

    fn parse_arguments(arguments_json: &str) -> Value {
        if arguments_json.is_empty() || arguments_json == "{}" {
            Value::Object(serde_json::Map::new())
        } else {
            serde_json::from_str(arguments_json)
                .unwrap_or_else(|_| Value::String(arguments_json.to_owned()))
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

    use async_trait::async_trait;
    use neuromance_common::tools::{Function, FunctionCall};
    use serde_json::json;

    use super::*;

    #[test]
    fn test_parse_arguments_empty() {
        assert_eq!(ToolExecutor::parse_arguments(""), json!({}));
    }

    #[test]
    fn test_parse_arguments_json_object() {
        let result = ToolExecutor::parse_arguments(r#"{"key": "value", "number": 42}"#);
        assert_eq!(result, json!({"key": "value", "number": 42}));
    }

    #[test]
    fn test_parse_arguments_json_array() {
        let result = ToolExecutor::parse_arguments(r#"["item1", "item2", "item3"]"#);
        assert_eq!(result, json!(["item1", "item2", "item3"]));
    }

    #[test]
    fn test_parse_arguments_string_fallback() {
        let result = ToolExecutor::parse_arguments("plain text argument");
        assert_eq!(result, json!("plain text argument"));
    }

    #[test]
    fn test_parse_arguments_invalid_json_fallback() {
        let result = ToolExecutor::parse_arguments(r#"{"incomplete json"#);
        assert_eq!(result, json!(r#"{"incomplete json"#));
    }

    #[test]
    fn test_parse_arguments_number_string() {
        assert_eq!(ToolExecutor::parse_arguments("42"), json!(42));
    }

    #[test]
    fn test_parse_arguments_boolean_string() {
        assert_eq!(ToolExecutor::parse_arguments("true"), json!(true));
    }

    /// A tool that echoes the `value` argument back so dispatch can be observed.
    struct EchoTool;

    #[async_trait]
    impl ToolImplementation for EchoTool {
        fn get_definition(&self) -> Tool {
            Tool::builder()
                .function(Function {
                    name: "echo".to_string(),
                    description: "echo".to_string(),
                    parameters: json!({}),
                })
                .build()
        }

        async fn execute(&self, args: &Value) -> Result<String, ToolError> {
            args.get("value")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .ok_or_else(|| ToolError::InvalidArguments("missing 'value'".into()))
        }
    }

    /// `execute_named` and `execute_tool` share one dispatch path: routing the
    /// same name + arguments through either reaches the same tool with the same
    /// parsed arguments.
    #[tokio::test]
    async fn test_execute_named_matches_execute_tool() {
        let mut executor = ToolExecutor::new();
        executor.add_tool(EchoTool);

        let args = r#"{"value": "hi"}"#;
        let via_named = executor.execute_named("echo", args).await.unwrap();

        let call = ToolCall {
            id: "1".to_string(),
            function: FunctionCall {
                name: "echo".to_string(),
                arguments: args.to_string(),
            },
            call_type: "function".to_string(),
            index: None,
        };
        let via_call = executor.execute_tool(&call).await.unwrap();

        assert_eq!(via_named, "hi");
        assert_eq!(via_named, via_call);
    }

    #[tokio::test]
    async fn test_execute_named_unknown_tool() {
        let executor = ToolExecutor::new();
        let err = executor.execute_named("missing", "{}").await.unwrap_err();
        assert!(matches!(err, ToolExecutorError::UnknownTool(name) if name == "missing"));
    }
}
