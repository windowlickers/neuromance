//! In-process subagent backed by an [`Agent`].

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use neuromance_client::LLMClient;
use neuromance_common::chat::Message;
use neuromance_common::task::{Outcome, Task};

use super::{Subagent, SubagentError};
use crate::Agent;

/// A [`Subagent`] that runs an in-process [`Agent`].
///
/// Each run builds a fresh [`Agent`] from the stored factory and hands it a
/// `[system, user]` message pair derived from the [`Task`]. Constructing per
/// run (rather than reusing one agent behind a lock) keeps concurrent runs of
/// the *same* subagent genuinely parallel — fan-outs like `spawn_agents` would
/// otherwise serialize on a shared agent. Nothing persists across runs; agents
/// are cheap to build because clients share their connection pools via `Arc`.
pub struct LocalSubagent<C: LLMClient> {
    id: String,
    system_prompt: String,
    build_agent: Box<dyn Fn() -> Agent<C> + Send + Sync>,
}

impl<C: LLMClient> LocalSubagent<C> {
    /// Create a subagent that prepends `system_prompt` to every task and runs
    /// each task on a fresh agent from `build_agent`.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        system_prompt: impl Into<String>,
        build_agent: impl Fn() -> Agent<C> + Send + Sync + 'static,
    ) -> Self {
        Self {
            id: id.into(),
            system_prompt: system_prompt.into(),
            build_agent: Box::new(build_agent),
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

        let mut agent = (self.build_agent)();
        let conv_id = agent.conversation_id;
        let messages = vec![
            Message::system(conv_id, self.system_prompt.as_str()),
            Message::user(conv_id, user_content),
        ];
        let response = agent
            .execute(Some(messages), cancel)
            .await
            .map_err(SubagentError::execution)?;

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
    use std::sync::{Arc, OnceLock};
    use std::time::Duration;

    use async_trait::async_trait;
    use futures::Stream;
    use tokio::sync::Barrier;
    use uuid::Uuid;

    use neuromance::Core;
    use neuromance_client::{ClientError, LLMClient};
    use neuromance_common::chat::{Message, MessageRole};
    use neuromance_common::client::{ChatChunk, ChatRequest, ChatResponse, Config, Usage};

    use super::*;

    /// Shared static config so mock clients can return `&Config` without
    /// per-call allocation.
    fn mock_config() -> &'static Config {
        static CONFIG: OnceLock<Config> = OnceLock::new();
        CONFIG.get_or_init(|| Config::new("mock", "mock-model"))
    }

    fn echo_response(request: &ChatRequest) -> ChatResponse {
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

        ChatResponse {
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
        }
    }

    struct EchoClient;

    #[async_trait]
    impl LLMClient for EchoClient {
        fn config(&self) -> &Config {
            mock_config()
        }

        async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ClientError> {
            Ok(echo_response(request))
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

    /// Blocks every chat call on a barrier, so a test can prove that two runs
    /// were in flight at the same time.
    struct BlockingClient {
        barrier: Arc<Barrier>,
    }

    #[async_trait]
    impl LLMClient for BlockingClient {
        fn config(&self) -> &Config {
            mock_config()
        }

        async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ClientError> {
            self.barrier.wait().await;
            Ok(echo_response(request))
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

    fn echo_subagent() -> LocalSubagent<EchoClient> {
        LocalSubagent::new("echo", "You echo input.", || {
            Agent::new("echo".to_string(), Core::new(EchoClient))
        })
    }

    #[tokio::test]
    async fn test_run_maps_assistant_message_into_outcome() {
        let subagent = echo_subagent();

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
        let subagent = echo_subagent();

        let task = Task::new("ping").with_context("extra");
        let outcome = subagent
            .run(task, CancellationToken::new())
            .await
            .expect("run succeeds");

        assert_eq!(outcome.content, "echo: ping\n\nextra");
    }

    #[tokio::test]
    async fn test_concurrent_runs_of_same_subagent_do_not_serialize() {
        // Each chat call blocks until both runs are in flight; an
        // implementation that serializes runs of the same subagent would
        // never release the first run and trip the timeout.
        let barrier = Arc::new(Barrier::new(2));
        let subagent = LocalSubagent::new("blocking", "sys", {
            let barrier = Arc::clone(&barrier);
            move || {
                Agent::new(
                    "blocking".to_string(),
                    Core::new(BlockingClient {
                        barrier: Arc::clone(&barrier),
                    }),
                )
            }
        });

        let (a, b) = tokio::time::timeout(
            Duration::from_secs(2),
            futures::future::join(
                subagent.run(Task::new("a"), CancellationToken::new()),
                subagent.run(Task::new("b"), CancellationToken::new()),
            ),
        )
        .await
        .expect("runs deadlocked — concurrent runs of one subagent serialized");

        assert_eq!(a.expect("run a succeeds").content, "echo: a");
        assert_eq!(b.expect("run b succeeds").content, "echo: b");
    }
}
