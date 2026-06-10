//! Exposes any [`Subagent`] as a [`ToolImplementation`], so a parent agent can
//! delegate by calling it like any other tool.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use neuromance_common::task::Task;
use neuromance_common::tools::{Function, Parameters, Property, Tool};
use neuromance_tools::{ToolError, ToolImplementation};

use super::Subagent;

/// A tool that runs a wrapped [`Subagent`] when called.
///
/// The wrapped subagent may be a leaf (e.g. [`LocalSubagent`](super::LocalSubagent))
/// or a combinator (e.g. [`FanoutVote`](super::FanoutVote)) — both present the same
/// `instructions`/`context` interface to the calling model.
pub struct SubagentTool {
    inner: Arc<dyn Subagent>,
    name: String,
    description: String,
    cancel: CancellationToken,
}

impl SubagentTool {
    /// Wrap `inner`, exposing it under `name` with `description` in the tool schema.
    ///
    /// `cancel` is passed to every delegated run, so cancelling it aborts
    /// in-flight delegations. [`ToolImplementation::execute`] provides no
    /// per-call token, so this construction-time token is the only way for a
    /// host to reach work the tool has delegated.
    #[must_use]
    pub fn new(
        inner: Arc<dyn Subagent>,
        name: impl Into<String>,
        description: impl Into<String>,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            inner,
            name: name.into(),
            description: description.into(),
            cancel,
        }
    }
}

#[async_trait]
impl ToolImplementation for SubagentTool {
    fn get_definition(&self) -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "instructions".to_string(),
            Property::string("What the subagent should do."),
        );
        properties.insert(
            "context".to_string(),
            Property::string("Optional supporting context for the subagent."),
        );

        Tool::builder()
            .function(Function {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: Parameters::new(properties, vec!["instructions".into()]).into(),
            })
            .build()
    }

    async fn execute(&self, args: &Value) -> Result<String, ToolError> {
        let obj = args
            .as_object()
            .ok_or_else(|| ToolError::InvalidArguments("expected object arguments".into()))?;

        let instructions = obj
            .get("instructions")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ToolError::InvalidArguments("missing 'instructions' parameter".into())
            })?;

        let mut task = Task::new(instructions);
        if let Some(context) = obj.get("context") {
            let context = context
                .as_str()
                .ok_or_else(|| ToolError::InvalidArguments("'context' must be a string".into()))?;
            task = task.with_context(context);
        }

        let outcome = self
            .inner
            .run(task, self.cancel.clone())
            .await
            .map_err(|e| ToolError::execution(format!("subagent '{}' failed: {e}", self.name)))?;
        Ok(outcome.content)
    }

    fn is_auto_approved(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use serde_json::json;

    use neuromance_common::task::Outcome;

    use super::super::SubagentError;
    use super::*;

    struct MockSubagent;

    #[async_trait]
    impl Subagent for MockSubagent {
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "mock"
        }

        async fn run(
            &self,
            task: Task,
            cancel: CancellationToken,
        ) -> Result<Outcome, SubagentError> {
            if cancel.is_cancelled() {
                return Err(SubagentError::execution("run cancelled"));
            }
            Ok(Outcome::new(task.id, format!("ran: {}", task.instructions)))
        }
    }

    fn delegate_tool(cancel: CancellationToken) -> SubagentTool {
        SubagentTool::new(Arc::new(MockSubagent), "delegate", "Delegate work.", cancel)
    }

    #[tokio::test]
    async fn test_execute_returns_outcome_content() {
        let tool = delegate_tool(CancellationToken::new());
        let result = tool
            .execute(&json!({ "instructions": "do it" }))
            .await
            .expect("execute succeeds");
        assert_eq!(result, "ran: do it");
    }

    #[tokio::test]
    async fn test_execute_rejects_missing_instructions() {
        let tool = delegate_tool(CancellationToken::new());
        let err = tool
            .execute(&json!({ "context": "no instructions here" }))
            .await
            .expect_err("missing instructions");
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[tokio::test]
    async fn test_cancellation_reaches_delegated_run() {
        let cancel = CancellationToken::new();
        let tool = delegate_tool(cancel.clone());
        cancel.cancel();

        let err = tool
            .execute(&json!({ "instructions": "do it" }))
            .await
            .expect_err("cancelled token must reach the subagent");
        assert!(
            err.to_string().contains("run cancelled"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn test_definition_requires_instructions() {
        let tool = delegate_tool(CancellationToken::new());
        let def = tool.get_definition();
        assert_eq!(def.function.name, "delegate");
        assert_eq!(def.function.parameters["required"], json!(["instructions"]));
    }

    #[test]
    fn test_not_auto_approved() {
        let tool = delegate_tool(CancellationToken::new());
        assert!(!tool.is_auto_approved());
    }
}
