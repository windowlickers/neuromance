//! Skill injection as a conversation [`Hook`].
//!
//! [`SkillsHook`] injects the progressive-disclosure menu once at conversation
//! start and expands the bodies of any skills `$mention`ed in the latest user
//! message. The menu and each mentioned skill are injected at most once per
//! conversation, so re-running the hook on later turns (with the menu already
//! in the persisted history) does not duplicate them.
//!
//! Menu injection is optional (`inject_menu`): a host that folds the menu into
//! the system prompt itself disables it and keeps only the `$mention` path.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, PoisonError};

use async_trait::async_trait;
use uuid::Uuid;

use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::hook::{Hook, HookContext, HookOutcome};

use super::catalog::SkillCatalog;

/// Conversation hook that injects the skill menu and `$mention`ed skill bodies
/// from inside the conversation loop.
#[derive(Debug)]
pub struct SkillsHook {
    catalog: Arc<SkillCatalog>,
    menu_budget: usize,
    body_budget: usize,
    mention_enabled: bool,
    /// Whether to inject the menu; off when the host folds it into the system
    /// prompt and the hook only handles `$mention` expansion.
    inject_menu: bool,
    /// Conversations whose menu has been injected, so it is injected once.
    menu_injected: Mutex<HashSet<Uuid>>,
    /// `(conversation, skill name)` pairs already injected, so each mentioned
    /// skill body injects at most once per conversation.
    injected: Mutex<HashSet<(Uuid, String)>>,
}

impl SkillsHook {
    /// Build a skills hook over `catalog`, rendering the menu within
    /// `menu_budget` and each mentioned body within `body_budget`. When
    /// `mention_enabled` is false, `$mention` expansion is skipped; when
    /// `inject_menu` is false, the menu is never injected (the host is expected
    /// to place it in the system prompt instead).
    #[must_use]
    pub fn new(
        catalog: Arc<SkillCatalog>,
        menu_budget: usize,
        body_budget: usize,
        mention_enabled: bool,
        inject_menu: bool,
    ) -> Self {
        Self {
            catalog,
            menu_budget,
            body_budget,
            mention_enabled,
            inject_menu,
            menu_injected: Mutex::new(HashSet::new()),
            injected: Mutex::new(HashSet::new()),
        }
    }

    /// Mark `conversation`'s menu injected; returns `false` if already present.
    fn mark_menu(&self, conversation: Uuid) -> bool {
        self.menu_injected
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(conversation)
    }

    /// Mark `(conversation, name)` injected; returns `false` if already present.
    fn mark_mention(&self, conversation: Uuid, name: &str) -> bool {
        self.injected
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert((conversation, name.to_string()))
    }
}

/// The content of the most recent `User` message, if any.
fn latest_user_text(messages: &[Message]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == MessageRole::User)
        .map(|m| m.content.as_str())
}

#[async_trait]
impl Hook for SkillsHook {
    fn name(&self) -> &'static str {
        "skills"
    }

    async fn on_conversation_start(
        &self,
        ctx: &HookContext,
        messages: &[Message],
    ) -> anyhow::Result<HookOutcome> {
        let conversation = ctx.conversation_id;
        let mut out = Vec::new();

        if self.inject_menu
            && self.mark_menu(conversation)
            && let Some(menu) = self.catalog.menu(self.menu_budget)
        {
            out.push(Message::system(conversation, menu));
        }

        if self.mention_enabled
            && let Some(text) = latest_user_text(messages)
        {
            // Resolve to owned data before the async loads so no borrow of the
            // catalog is held across an await point.
            let mentions: Vec<(String, String)> = self
                .catalog
                .resolve_mentions(text)
                .into_iter()
                .map(|m| (m.name.clone(), m.locator.render()))
                .collect();
            for (name, locator) in mentions {
                if !self.mark_mention(conversation, &name) {
                    continue;
                }
                match self.catalog.load(&name, self.body_budget).await {
                    Ok(body) => out.push(Message::user(
                        conversation,
                        format!(
                            "<skill>\n<name>{name}</name>\n<source>{locator}</source>\n{body}\n</skill>"
                        ),
                    )),
                    Err(e) => {
                        tracing::warn!(skill = %name, "failed to load mentioned skill body: {e}");
                    }
                }
            }
        }

        Ok(HookOutcome::inject(out))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use async_trait::async_trait;

    use super::*;
    use crate::skills::error::SkillError;
    use crate::skills::model::{SkillId, SkillLocator, SkillMetadata};
    use crate::skills::source::SkillSource;

    /// In-memory source backed by `(name, body)` pairs for deterministic tests.
    struct MemSource {
        skills: Vec<(String, String)>,
    }

    impl MemSource {
        fn new(skills: &[(&str, &str)]) -> Self {
            Self {
                skills: skills
                    .iter()
                    .map(|(n, b)| ((*n).to_string(), (*b).to_string()))
                    .collect(),
            }
        }
    }

    #[async_trait]
    impl SkillSource for MemSource {
        async fn list(&self) -> Result<Vec<SkillMetadata>, SkillError> {
            Ok(self
                .skills
                .iter()
                .map(|(name, _)| SkillMetadata {
                    id: SkillId::new(name),
                    name: name.clone(),
                    description: format!("desc of {name}"),
                    locator: SkillLocator::Remote {
                        endpoint: "mem://skills".to_string(),
                        id: name.clone(),
                    },
                    extra: serde_yaml::Mapping::new(),
                })
                .collect())
        }

        async fn load_body(&self, id: &SkillId) -> Result<String, SkillError> {
            self.skills
                .iter()
                .find(|(name, _)| name == id.as_str())
                .map(|(_, body)| body.clone())
                .ok_or_else(|| SkillError::NotFound(id.to_string()))
        }
    }

    async fn hook_over(
        skills: &[(&str, &str)],
        mention_enabled: bool,
        inject_menu: bool,
    ) -> SkillsHook {
        let catalog = SkillCatalog::build(vec![Box::new(MemSource::new(skills))]).await;
        SkillsHook::new(Arc::new(catalog), 8192, 8192, mention_enabled, inject_menu)
    }

    #[tokio::test]
    async fn test_conversation_start_injects_menu_once() {
        let hook = hook_over(&[("alpha", "alpha body")], true, true).await;
        let ctx = HookContext::new(Uuid::new_v4(), 0);

        let first = hook.on_conversation_start(&ctx, &[]).await.unwrap();
        assert_eq!(first.messages.len(), 1);
        assert_eq!(first.messages[0].role, MessageRole::System);
        assert!(first.messages[0].content.contains("<skills_instructions>"));

        let second = hook.on_conversation_start(&ctx, &[]).await.unwrap();
        assert!(
            second.messages.is_empty(),
            "menu must not be injected twice per conversation"
        );
    }

    #[tokio::test]
    async fn test_distinct_conversations_inject_menu_independently() {
        let hook = hook_over(&[("alpha", "alpha body")], true, true).await;
        let conv_a = HookContext::new(Uuid::new_v4(), 0);
        let conv_b = HookContext::new(Uuid::new_v4(), 0);

        let a = hook.on_conversation_start(&conv_a, &[]).await.unwrap();
        let b = hook.on_conversation_start(&conv_b, &[]).await.unwrap();
        assert_eq!(a.messages.len(), 1);
        assert_eq!(b.messages.len(), 1);
    }

    #[tokio::test]
    async fn test_mention_injects_body_once() {
        let hook = hook_over(&[("alpha", "alpha body")], true, true).await;
        let conv = Uuid::new_v4();
        let ctx = HookContext::new(conv, 0);
        let seed = vec![
            Message::system(conv, "you are a test agent"),
            Message::user(conv, "please use $alpha now"),
        ];

        let first = hook.on_conversation_start(&ctx, &seed).await.unwrap();
        let bodies: Vec<&Message> = first
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::User)
            .collect();
        assert_eq!(bodies.len(), 1);
        assert!(bodies[0].content.contains("alpha body"));
        assert!(bodies[0].content.contains("<name>alpha</name>"));

        // A later turn that mentions the same skill must not re-inject its body.
        let next = vec![Message::user(conv, "and again $alpha")];
        let second = hook.on_conversation_start(&ctx, &next).await.unwrap();
        assert!(
            second.messages.iter().all(|m| m.role != MessageRole::User),
            "mentioned skill body must inject at most once per conversation"
        );
    }

    #[tokio::test]
    async fn test_mention_disabled_injects_no_body() {
        let hook = hook_over(&[("alpha", "alpha body")], false, true).await;
        let conv = Uuid::new_v4();
        let ctx = HookContext::new(conv, 0);
        let seed = vec![Message::user(conv, "please use $alpha")];

        let outcome = hook.on_conversation_start(&ctx, &seed).await.unwrap();
        assert!(
            outcome.messages.iter().all(|m| m.role != MessageRole::User),
            "mention expansion must be skipped when disabled"
        );
        // The menu still injects regardless of mention mode.
        assert!(
            outcome
                .messages
                .iter()
                .any(|m| m.role == MessageRole::System)
        );
    }

    #[tokio::test]
    async fn test_menu_disabled_still_expands_mentions() {
        // Host folds the menu into the system prompt itself: the hook injects no
        // System menu but still expands `$mention`ed bodies as User messages.
        let hook = hook_over(&[("alpha", "alpha body")], true, false).await;
        let conv = Uuid::new_v4();
        let ctx = HookContext::new(conv, 0);
        let seed = vec![
            Message::system(conv, "system prompt with the menu folded in"),
            Message::user(conv, "please use $alpha"),
        ];

        let outcome = hook.on_conversation_start(&ctx, &seed).await.unwrap();
        assert!(
            outcome
                .messages
                .iter()
                .all(|m| m.role != MessageRole::System),
            "menu must not be injected when inject_menu is false"
        );
        let bodies: Vec<&Message> = outcome
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::User)
            .collect();
        assert_eq!(bodies.len(), 1);
        assert!(bodies[0].content.contains("alpha body"));
    }

    #[tokio::test]
    async fn test_empty_catalog_injects_nothing() {
        let hook = hook_over(&[], true, true).await;
        let conv = Uuid::new_v4();
        let ctx = HookContext::new(conv, 0);
        let seed = vec![Message::user(conv, "use $alpha")];

        let outcome = hook.on_conversation_start(&ctx, &seed).await.unwrap();
        assert!(outcome.messages.is_empty());
    }
}
