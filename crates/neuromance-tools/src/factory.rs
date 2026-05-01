//! Tool factory registry.
//!
//! [`ToolFactory`] turns a name + JSON config blob into one or more
//! registrations on a [`ToolRegistry`]. The runtime parses a list of
//! [`ToolConfig`] entries from its config file and dispatches each to the
//! matching factory in [`ToolFactoryRegistry`].
//!
//! A single factory entry may register multiple tools — `todos` registers
//! both `read_todos` and `write_todos` so they share storage.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ToolRegistry;
use crate::bool_tool::BoolToolFactory;
use crate::think_tool::ThinkToolFactory;
use crate::todo_tool::TodoToolsFactory;

/// Per-tool configuration entry.
///
/// Deserialized from the runtime's config file:
///
/// ```toml
/// [[tools]]
/// name = "think"
///
/// [[tools]]
/// name = "todos"
/// [tools.config]   # optional, factory-specific
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolConfig {
    pub name: String,
    #[serde(default)]
    pub config: Value,
}

/// Constructs and registers tool implementations from a name + config blob.
pub trait ToolFactory: Send + Sync {
    /// Stable factory name; matches [`ToolConfig::name`].
    fn name(&self) -> &'static str;

    /// Build tools from `config` and register them into `registry`.
    ///
    /// A single factory may register multiple tools.
    ///
    /// # Errors
    /// Returns an error if the config is invalid or construction fails.
    fn build(&self, config: &Value, registry: &ToolRegistry) -> Result<()>;
}

/// Registry of [`ToolFactory`] instances keyed by name.
pub struct ToolFactoryRegistry {
    factories: HashMap<String, Box<dyn ToolFactory>>,
}

impl ToolFactoryRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Returns a registry pre-populated with the built-in factories:
    /// `return_bool`, `think`, `todos`.
    #[must_use]
    pub fn with_builtin() -> Self {
        let mut r = Self::new();
        r.register(BoolToolFactory);
        r.register(ThinkToolFactory);
        r.register(TodoToolsFactory);
        r
    }

    pub fn register<F: ToolFactory + 'static>(&mut self, factory: F) {
        self.factories
            .insert(factory.name().to_string(), Box::new(factory));
    }

    /// Build a tool from a single config entry and register it into `registry`.
    ///
    /// # Errors
    /// Returns an error if no factory is registered for `config.name` or if
    /// the factory's `build` fails.
    pub fn build_one(&self, config: &ToolConfig, registry: &ToolRegistry) -> Result<()> {
        let factory = self.factories.get(&config.name).with_context(|| {
            format!(
                "no tool factory registered for '{}'; known factories: [{}]",
                config.name,
                self.factory_names().join(", ")
            )
        })?;
        factory
            .build(&config.config, registry)
            .with_context(|| format!("failed to build tool '{}'", config.name))
    }

    /// Build a fresh [`ToolRegistry`] populated from a slice of configs.
    ///
    /// # Errors
    /// Returns an error on the first failed config.
    pub fn build_all(&self, configs: &[ToolConfig]) -> Result<ToolRegistry> {
        let registry = ToolRegistry::new();
        for cfg in configs {
            self.build_one(cfg, &registry)?;
        }
        Ok(registry)
    }

    /// Sorted list of registered factory names. Useful for error messages.
    #[must_use]
    pub fn factory_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.factories.keys().cloned().collect();
        names.sort();
        names
    }
}

impl Default for ToolFactoryRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use std::sync::Arc;

    use super::*;
    use crate::ToolImplementation;
    use neuromance_common::tools::{Function, Parameters, Tool};

    struct DummyTool {
        name: String,
    }

    #[async_trait::async_trait]
    impl ToolImplementation for DummyTool {
        fn get_definition(&self) -> Tool {
            Tool::builder()
                .function(Function {
                    name: self.name.clone(),
                    description: "dummy".to_string(),
                    parameters: Parameters::new(HashMap::new(), vec![]).into(),
                })
                .build()
        }

        async fn execute(&self, _args: &Value) -> Result<String> {
            Ok("ok".to_string())
        }
    }

    struct DummyFactory;
    impl ToolFactory for DummyFactory {
        fn name(&self) -> &'static str {
            "dummy"
        }
        fn build(&self, _config: &Value, registry: &ToolRegistry) -> Result<()> {
            registry.register(Arc::new(DummyTool {
                name: "dummy_tool".to_string(),
            }));
            Ok(())
        }
    }

    #[test]
    fn test_with_builtin_registers_known_factories() {
        let r = ToolFactoryRegistry::with_builtin();
        let names = r.factory_names();
        assert!(names.contains(&"return_bool".to_string()));
        assert!(names.contains(&"think".to_string()));
        assert!(names.contains(&"todos".to_string()));
    }

    #[test]
    fn test_build_all_registers_all_tools() {
        let factory_registry = ToolFactoryRegistry::with_builtin();
        let configs = vec![
            ToolConfig {
                name: "think".to_string(),
                config: Value::Null,
            },
            ToolConfig {
                name: "todos".to_string(),
                config: Value::Null,
            },
        ];

        let tools = factory_registry.build_all(&configs).unwrap();
        let tool_names = tools.tool_names();

        // think + todos (which expands to read_todos and write_todos) = 3 tools
        assert!(tool_names.contains(&"think".to_string()));
        assert!(tool_names.contains(&"read_todos".to_string()));
        assert!(tool_names.contains(&"write_todos".to_string()));
        assert_eq!(tool_names.len(), 3);
    }

    #[test]
    fn test_build_unknown_factory_errors_with_known_names() {
        let factory_registry = ToolFactoryRegistry::with_builtin();
        let configs = vec![ToolConfig {
            name: "nonexistent".to_string(),
            config: Value::Null,
        }];

        let result = factory_registry.build_all(&configs);
        assert!(result.is_err());
        let err = result.err().unwrap();
        let msg = format!("{err:#}");
        assert!(msg.contains("nonexistent"));
        assert!(msg.contains("return_bool"));
        assert!(msg.contains("think"));
        assert!(msg.contains("todos"));
    }

    #[test]
    fn test_register_custom_factory() {
        let mut r = ToolFactoryRegistry::new();
        r.register(DummyFactory);
        let registry = ToolRegistry::new();
        r.build_one(
            &ToolConfig {
                name: "dummy".to_string(),
                config: Value::Null,
            },
            &registry,
        )
        .unwrap();
        assert!(registry.contains("dummy_tool"));
    }

    #[test]
    fn test_tool_config_deserializes_with_default_config() {
        let toml_str = r#"
            name = "think"
        "#;
        let cfg: ToolConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.name, "think");
        assert!(cfg.config.is_null());
    }
}
