//! Glob-triggered rule injection as a conversation [`Hook`].
//!
//! [`RulesHook`] injects `always_apply` rules once at conversation start, and
//! injects a glob-triggered rule the first time a file-path tool call touches a
//! matching path. Each rule is injected at most once per conversation.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, PoisonError};

use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use neuromance_common::chat::Message;
use neuromance_common::hook::{Hook, HookContext, HookOutcome};
use neuromance_common::tools::ToolCall;

use super::catalog::RuleCatalog;
use super::model::{RuleId, RuleMetadata};

/// Conversation hook that injects rule-file bodies — always-apply rules at the
/// start, and glob-matched rules after a tool touches a matching path.
#[derive(Debug)]
pub struct RulesHook {
    catalog: Arc<RuleCatalog>,
    body_budget: usize,
    /// `(conversation, rule)` pairs already injected, so each rule injects at
    /// most once per conversation.
    injected: Mutex<HashSet<(Uuid, RuleId)>>,
}

impl RulesHook {
    /// Build a rules hook over `catalog`, truncating each body to `body_budget`.
    #[must_use]
    pub fn new(catalog: Arc<RuleCatalog>, body_budget: usize) -> Self {
        Self {
            catalog,
            body_budget,
            injected: Mutex::new(HashSet::new()),
        }
    }

    /// Mark `(conversation, id)` injected; returns `false` if already present.
    fn mark(&self, conversation: Uuid, id: &RuleId) -> bool {
        self.injected
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert((conversation, id.clone()))
    }

    /// Load and wrap `metas` into messages, skipping rules already injected for
    /// `conversation`. `as_system` chooses the message role.
    async fn inject(
        &self,
        conversation: Uuid,
        metas: &[RuleMetadata],
        as_system: bool,
    ) -> HookOutcome {
        let mut messages = Vec::new();
        for meta in metas {
            if !self.mark(conversation, &meta.id) {
                continue;
            }
            match self.catalog.load(&meta.id, self.body_budget).await {
                Ok(body) => {
                    let content = format!(
                        "<rule>\n<id>{}</id>\n<source>{}</source>\n{body}\n</rule>",
                        meta.id,
                        meta.locator.render(),
                    );
                    messages.push(if as_system {
                        Message::system(conversation, content)
                    } else {
                        Message::user(conversation, content)
                    });
                }
                Err(e) => tracing::warn!(rule = %meta.id, "failed to load rule body: {e}"),
            }
        }
        HookOutcome::inject(messages)
    }
}

/// Extract the `path` string argument of a tool call, if present.
fn tool_path(call: &ToolCall) -> Option<String> {
    let args: Value = serde_json::from_str(call.function.arguments_json()).ok()?;
    args.get("path")?.as_str().map(ToString::to_string)
}

#[async_trait]
impl Hook for RulesHook {
    fn name(&self) -> &'static str {
        "rules"
    }

    async fn on_conversation_start(
        &self,
        ctx: &HookContext,
        _messages: &[Message],
    ) -> anyhow::Result<HookOutcome> {
        let always: Vec<RuleMetadata> = self.catalog.always_apply().cloned().collect();
        Ok(self.inject(ctx.conversation_id, &always, true).await)
    }

    async fn after_tool(
        &self,
        ctx: &HookContext,
        call: &ToolCall,
        _result: &str,
        _success: bool,
    ) -> anyhow::Result<HookOutcome> {
        let Some(path) = tool_path(call) else {
            return Ok(HookOutcome::none());
        };
        let matched: Vec<RuleMetadata> = self
            .catalog
            .match_path(&path)
            .into_iter()
            .cloned()
            .collect();
        Ok(self.inject(ctx.conversation_id, &matched, false).await)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::fs;

    use async_trait::async_trait;
    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::rules::error::RuleError;
    use crate::rules::model::RuleLocator;
    use crate::rules::source::{LocalRuleSource, RuleSource};

    /// A source that lists one always-apply rule but always fails to load its
    /// body, exercising the inject path that drops a rule on load failure.
    struct ListsButFailsToLoad;

    #[async_trait]
    impl RuleSource for ListsButFailsToLoad {
        async fn list(&self) -> Result<Vec<RuleMetadata>, RuleError> {
            Ok(vec![RuleMetadata {
                id: RuleId::new("ghost.md"),
                globs: Vec::new(),
                always_apply: true,
                description: None,
                locator: RuleLocator::Remote {
                    endpoint: "mem://rules".to_string(),
                    id: "ghost.md".to_string(),
                },
                extra: serde_yaml::Mapping::new(),
            }])
        }

        async fn load_body(&self, id: &RuleId) -> Result<String, RuleError> {
            Err(RuleError::NotFound(id.to_string()))
        }
    }

    async fn hook_over(files: &[(&str, &str)]) -> (RulesHook, TempDir) {
        let dir = tempdir().unwrap();
        for (name, content) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(path, content).unwrap();
        }
        let catalog = RuleCatalog::build(vec![Box::new(LocalRuleSource::new(vec![
            dir.path().into(),
        ]))])
        .await;
        (RulesHook::new(Arc::new(catalog), 8192), dir)
    }

    fn read_call(path: &str) -> ToolCall {
        ToolCall::new("read", format!(r#"{{"path":"{path}"}}"#))
    }

    #[tokio::test]
    async fn test_conversation_start_injects_always_apply() {
        let (hook, _dir) =
            hook_over(&[("global.md", "---\nalwaysApply: true\n---\nglobal rule")]).await;
        let ctx = HookContext::new(Uuid::new_v4(), 0);

        let outcome = hook.on_conversation_start(&ctx, &[]).await.unwrap();
        assert_eq!(outcome.messages.len(), 1);
        assert!(outcome.messages[0].content.contains("global rule"));
    }

    #[tokio::test]
    async fn test_after_tool_injects_matching_rule() {
        let (hook, _dir) = hook_over(&[("ts.md", "---\nglobs: \"*.ts\"\n---\nts rule")]).await;
        let ctx = HookContext::new(Uuid::new_v4(), 0);

        let outcome = hook
            .after_tool(&ctx, &read_call("/a/b/foo.ts"), "contents", true)
            .await
            .unwrap();
        assert_eq!(outcome.messages.len(), 1);
        assert!(outcome.messages[0].content.contains("ts rule"));
    }

    #[tokio::test]
    async fn test_after_tool_dedups_same_rule_per_conversation() {
        let (hook, _dir) = hook_over(&[("ts.md", "---\nglobs: \"*.ts\"\n---\nts rule")]).await;
        let ctx = HookContext::new(Uuid::new_v4(), 0);

        let first = hook
            .after_tool(&ctx, &read_call("/a/foo.ts"), "x", true)
            .await
            .unwrap();
        assert_eq!(first.messages.len(), 1);
        let second = hook
            .after_tool(&ctx, &read_call("/a/bar.ts"), "x", true)
            .await
            .unwrap();
        assert!(second.messages.is_empty());
    }

    #[tokio::test]
    async fn test_after_tool_no_path_arg_no_injection() {
        let (hook, _dir) = hook_over(&[("ts.md", "---\nglobs: \"*.ts\"\n---\nts rule")]).await;
        let ctx = HookContext::new(Uuid::new_v4(), 0);

        let call = ToolCall::new("bash", r#"{"command":"ls"}"#);
        let outcome = hook.after_tool(&ctx, &call, "x", true).await.unwrap();
        assert!(outcome.messages.is_empty());
    }

    #[tokio::test]
    async fn test_distinct_conversations_inject_independently() {
        let (hook, _dir) = hook_over(&[("ts.md", "---\nglobs: \"*.ts\"\n---\nts rule")]).await;

        let conv_a = HookContext::new(Uuid::new_v4(), 0);
        let conv_b = HookContext::new(Uuid::new_v4(), 0);
        let a = hook
            .after_tool(&conv_a, &read_call("/foo.ts"), "x", true)
            .await
            .unwrap();
        let b = hook
            .after_tool(&conv_b, &read_call("/foo.ts"), "x", true)
            .await
            .unwrap();
        assert_eq!(a.messages.len(), 1);
        assert_eq!(b.messages.len(), 1);
    }

    #[tokio::test]
    async fn test_inject_skips_rule_whose_body_fails_to_load() {
        let catalog = RuleCatalog::build(vec![Box::new(ListsButFailsToLoad)]).await;
        assert_eq!(
            catalog.len(),
            1,
            "rule is listed but its body load will fail"
        );
        let hook = RulesHook::new(Arc::new(catalog), 8192);
        let ctx = HookContext::new(Uuid::new_v4(), 0);

        let outcome = hook.on_conversation_start(&ctx, &[]).await.unwrap();
        assert!(outcome.messages.is_empty());
    }
}
