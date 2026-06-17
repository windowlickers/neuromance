//! The `load_skill` tool: lets a model pull a skill's full `SKILL.md` body into
//! context on demand.
//!
//! The cheap menu of `name: description` is injected elsewhere (once per
//! conversation); this tool is how an autonomous agent loads the full
//! instructions for a skill it has decided to use. It is read-only and
//! auto-approved. Registration is programmatic — the runtime builds a
//! [`SkillCatalog`] from config and constructs the tool with it — so there is
//! no `ToolFactory`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use neuromance_common::tools::{Function, Parameters, Property, Tool};
use neuromance_context::skills::{SkillCatalog, SkillError};

use crate::{ToolError, ToolImplementation};

/// Tool that loads a named skill's body from a [`SkillCatalog`].
pub struct SkillTool {
    catalog: Arc<SkillCatalog>,
    body_budget: usize,
}

impl SkillTool {
    /// Create a `load_skill` tool over `catalog`, truncating bodies to
    /// `body_budget` bytes.
    #[must_use]
    pub const fn new(catalog: Arc<SkillCatalog>, body_budget: usize) -> Self {
        Self {
            catalog,
            body_budget,
        }
    }

    /// Comma-separated list of available skill names, for error messages.
    fn available_names(&self) -> String {
        self.catalog
            .metadata()
            .map(|m| m.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[async_trait]
impl ToolImplementation for SkillTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            Property::string("Name of the skill to load, as shown in the skills menu."),
        );

        Tool::builder()
            .function(Function {
                name: "load_skill".to_string(),
                description: "Load a skill's full instructions into context by name. Call this \
                              before performing a task when a skill in the skills menu matches it. \
                              Returns the skill's Markdown body."
                    .to_string(),
                parameters: Parameters::new(properties, vec!["name".into()]).into(),
            })
            .build()
    }

    async fn execute(&self, args: &Value) -> Result<String, ToolError> {
        let name = args
            .as_object()
            .and_then(|obj| obj.get("name"))
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidArguments("missing 'name' parameter".into()))?;

        match self.catalog.load(name, self.body_budget).await {
            Ok(body) => Ok(body),
            Err(SkillError::NotFound(_)) => Err(ToolError::InvalidArguments(format!(
                "unknown skill '{name}'. Available skills: [{}]",
                self.available_names()
            ))),
            Err(e) => Err(ToolError::execution(e)),
        }
    }

    fn is_auto_approved(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use async_trait::async_trait;
    use serde_json::json;

    use neuromance_context::skills::{
        DEFAULT_BUDGET_BYTES, SkillId, SkillLocator, SkillMetadata, SkillSource,
    };

    use super::*;

    struct OneSkill;

    #[async_trait]
    impl SkillSource for OneSkill {
        async fn list(&self) -> Result<Vec<SkillMetadata>, SkillError> {
            Ok(vec![SkillMetadata {
                id: SkillId::new("deploy"),
                name: "deploy".to_string(),
                description: "deploy the app".to_string(),
                locator: SkillLocator::Remote {
                    endpoint: "mem://skills".to_string(),
                    id: "deploy".to_string(),
                },
                extra: serde_yaml::Mapping::new(),
            }])
        }

        async fn load_body(&self, id: &SkillId) -> Result<String, SkillError> {
            if id.as_str() == "deploy" {
                Ok("run the deploy script".to_string())
            } else {
                Err(SkillError::NotFound(id.to_string()))
            }
        }
    }

    async fn tool() -> SkillTool {
        let catalog = SkillCatalog::build(vec![Box::new(OneSkill)]).await;
        SkillTool::new(Arc::new(catalog), DEFAULT_BUDGET_BYTES)
    }

    #[tokio::test]
    async fn test_loads_known_skill_body() {
        let tool = tool().await;
        let out = tool.execute(&json!({ "name": "deploy" })).await.unwrap();
        assert_eq!(out, "run the deploy script");
    }

    #[tokio::test]
    async fn test_unknown_skill_lists_available() {
        let tool = tool().await;
        let err = tool.execute(&json!({ "name": "ghost" })).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown skill 'ghost'"));
        assert!(msg.contains("deploy"));
    }

    #[tokio::test]
    async fn test_missing_name_argument_errors() {
        let tool = tool().await;
        let err = tool.execute(&json!({})).await.unwrap_err();
        assert!(err.to_string().contains("missing 'name'"));
    }

    #[tokio::test]
    async fn test_definition_and_auto_approval() {
        let catalog = SkillCatalog::build(vec![]).await;
        let tool = SkillTool::new(Arc::new(catalog), DEFAULT_BUDGET_BYTES);
        let def = tool.get_definition();
        assert_eq!(def.function.name, "load_skill");
        assert_eq!(def.function.parameters["required"], json!(["name"]));
        assert!(tool.is_auto_approved());
    }
}
