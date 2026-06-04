//! In-process subagent backed by an [`Agent`].

use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use neuromance_client::LLMClient;
use neuromance_common::chat::Message;
use neuromance_common::task::{Outcome, Task};

use super::{Subagent, SubagentError};
use crate::Agent;

/// A [`Subagent`] that runs an in-process [`Agent`].
///
/// [`Subagent::run`] takes `&self`, but [`Agent::execute`] takes `&mut self`, so
/// the agent is held behind a [`Mutex`]. Each run builds a fresh `[system, user]`
/// message pair from the [`Task`] — the agent's accumulated stats persist across
/// runs, but the conversation passed to the model is always just the task.
pub struct LocalSubagent<C: LLMClient> {
    id: String,
    system_prompt: String,
    agent: Mutex<Agent<C>>,
}

impl<C: LLMClient> LocalSubagent<C> {
    /// Wrap `agent` as a subagent that prepends `system_prompt` to every task.
    ///
    /// The subagent id is taken from the agent's id.
    #[must_use]
    pub fn new(agent: Agent<C>, system_prompt: impl Into<String>) -> Self {
        Self {
            id: agent.id.clone(),
            system_prompt: system_prompt.into(),
            agent: Mutex::new(agent),
        }
    }
}

#[async_trait]
impl<C: LLMClient> Subagent for LocalSubagent<C> {
    fn id(&self) -> &str {
        &self.id
    }

    async fn run(&self, task: Task, cancel: CancellationToken) -> Result<Outcome, SubagentError> {
        let user_content = match &task.context {
            Some(ctx) => format!("{}\n\n{ctx}", task.instructions),
            None => task.instructions.clone(),
        };

        let response = {
            let mut agent = self.agent.lock().await;
            let conv_id = agent.conversation_id;
            let messages = vec![
                Message::system(conv_id, self.system_prompt.as_str()),
                Message::user(conv_id, user_content),
            ];
            agent
                .execute(Some(messages), cancel)
                .await
                .map_err(SubagentError::execution)?
        };

        Ok(Outcome {
            task_id: task.id,
            content: response.content.content,
            reasoning: response.reasoning,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use std::collections::HashMap;
    use std::pin::Pin;

    use async_trait::async_trait;
    use futures::Stream;
    use uuid::Uuid;

    use neuromance::Core;
    use neuromance_client::{ClientError, LLMClient};
    use neuromance_common::chat::{Message, MessageRole};
    use neuromance_common::client::{ChatChunk, ChatRequest, ChatResponse, Config, Usage};

    use super::*;

    struct EchoClient;

    #[async_trait]
    impl LLMClient for EchoClient {
        fn config(&self) -> &Config {
            // A leaked config keeps the `&Config` return without per-call allocation;
            // acceptable in a test-only mock.
            use std::sync::OnceLock;
            static CONFIG: OnceLock<Config> = OnceLock::new();
            CONFIG.get_or_init(|| Config::new("mock", "mock-model"))
        }

        async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ClientError> {
            let conv_id = request
                .messages
                .first()
                .map_or_else(Uuid::new_v4, |m| m.conversation_id);
            let last_user = request
                .messages
                .iter()
                .rev()
                .find(|m| m.role == MessageRole::User)
                .map_or_else(String::new, |m| m.content.clone());

            Ok(ChatResponse {
                message: Message::assistant(conv_id, format!("echo: {last_user}")),
                model: "mock-model".to_string(),
                usage: Some(Usage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                    cost: None,
                    input_tokens_details: None,
                    output_tokens_details: None,
                }),
                finish_reason: None,
                created_at: chrono::Utc::now(),
                response_id: None,
                metadata: HashMap::new(),
            })
        }

        async fn chat_stream(
            &self,
            _request: &ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, ClientError>> + Send>>, ClientError>
        {
            unreachable!("LocalSubagent does not stream")
        }

        fn supports_tools(&self) -> bool {
            true
        }

        fn supports_streaming(&self) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn test_run_maps_assistant_message_into_outcome() {
        let agent = Agent::new("echo".to_string(), Core::new(EchoClient));
        let subagent = LocalSubagent::new(agent, "You echo input.");

        let task = Task::new("ping");
        let outcome = subagent
            .run(task.clone(), CancellationToken::new())
            .await
            .expect("run succeeds");

        assert_eq!(outcome.task_id, task.id);
        assert_eq!(outcome.content, "echo: ping");
    }

    #[tokio::test]
    async fn test_run_appends_context_to_instructions() {
        let agent = Agent::new("echo".to_string(), Core::new(EchoClient));
        let subagent = LocalSubagent::new(agent, "You echo input.");

        let task = Task::new("ping").with_context("extra");
        let outcome = subagent
            .run(task, CancellationToken::new())
            .await
            .expect("run succeeds");

        assert_eq!(outcome.content, "echo: ping\n\nextra");
    }
}
