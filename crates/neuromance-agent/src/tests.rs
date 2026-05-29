#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::panic)]

use std::collections::HashMap;
use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use neuromance::Core;
use neuromance_client::{ClientError, LLMClient};
use neuromance_common::agents::{AgentMessage, AgentState, ContextUpdate};
use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::{ChatChunk, ChatRequest, ChatResponse, Config, ToolChoice, Usage};

use crate::Agent;

struct MockLLMClient {
    config: Config,
    usage: Usage,
}

impl MockLLMClient {
    fn new() -> Self {
        Self {
            config: Config::new("mock", "mock-model"),
            usage: Usage {
                prompt_tokens: 50,
                completion_tokens: 30,
                total_tokens: 80,
                cost: None,
                input_tokens_details: None,
                output_tokens_details: None,
            },
        }
    }
}

#[async_trait]
impl LLMClient for MockLLMClient {
    fn config(&self) -> &Config {
        &self.config
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ClientError> {
        let conv_id = request
            .messages
            .first()
            .map_or_else(Uuid::new_v4, |m| m.conversation_id);

        Ok(ChatResponse {
            message: Message::assistant(conv_id, "Mock response"),
            model: "mock-model".to_string(),
            usage: Some(self.usage.clone()),
            finish_reason: None,
            created_at: chrono::Utc::now(),
            response_id: Some("test-response".to_string()),
            metadata: HashMap::new(),
        })
    }

    async fn chat_stream(
        &self,
        _request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, ClientError>> + Send>>, ClientError>
    {
        use futures::stream;

        let chunk = ChatChunk {
            model: "mock-model".to_string(),
            delta_content: Some("Mock response".to_string()),
            delta_reasoning_content: None,
            delta_role: Some(MessageRole::Assistant),
            delta_tool_calls: None,
            finish_reason: None,
            usage: Some(self.usage.clone()),
            response_id: Some("test-response".to_string()),
            created_at: chrono::Utc::now(),
            metadata: HashMap::new(),
        };

        Ok(Box::pin(stream::iter(vec![Ok(chunk)])))
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

fn make_messages(conv_id: Uuid) -> Vec<Message> {
    vec![
        Message::system(conv_id, "You are a test agent."),
        Message::user(conv_id, "Hello"),
    ]
}

// -- Input validation tests --

#[tokio::test]
async fn execute_rejects_too_few_messages() {
    let client = MockLLMClient::new();
    let mut agent = Agent::new("test".into(), Core::new(client));

    let result = agent.execute(Some(vec![]), CancellationToken::new()).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("at least a system message"),
        "unexpected: {err}"
    );
}

#[tokio::test]
async fn execute_rejects_wrong_first_role() {
    let client = MockLLMClient::new();
    let mut agent = Agent::new("test".into(), Core::new(client));
    let conv_id = Uuid::new_v4();

    let result = agent
        .execute(
            Some(vec![
                Message::user(conv_id, "not system"),
                Message::user(conv_id, "hello"),
            ]),
            CancellationToken::new(),
        )
        .await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("First message must be a system message"),
        "unexpected: {err}"
    );
}

#[tokio::test]
async fn execute_rejects_wrong_second_role() {
    let client = MockLLMClient::new();
    let mut agent = Agent::new("test".into(), Core::new(client));
    let conv_id = Uuid::new_v4();

    let result = agent
        .execute(
            Some(vec![
                Message::system(conv_id, "system"),
                Message::system(conv_id, "not user"),
            ]),
            CancellationToken::new(),
        )
        .await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Second message must be a user message"),
        "unexpected: {err}"
    );
}

// -- reset() test --

#[tokio::test]
async fn reset_clears_all_state() {
    let client = MockLLMClient::new();
    let mut agent = Agent::new("test".into(), Core::new(client));
    let conv_id = agent.conversation_id;

    // Populate some state
    agent.state.stats.total_messages = 10;
    agent.state.stats.tokens_used = 500;
    agent.state.context.task = Some("task".into());
    agent.state.memory.short_term.push("observation".into());
    agent.messages.push(Message::user(conv_id, "hello"));

    agent.reset().await.unwrap();

    assert!(agent.state.conversation_history.is_empty());
    assert!(agent.state.memory.short_term.is_empty());
    assert!(agent.state.context.task.is_none());
    assert_eq!(agent.state.stats.total_messages, 0);
    assert_eq!(agent.state.stats.tokens_used, 0);
    assert!(agent.messages.is_empty());
    assert_ne!(agent.conversation_id, conv_id);
}

// -- AgentStats tracking tests --

#[tokio::test]
async fn execute_tracks_stats() {
    let client = MockLLMClient::new();
    let mut agent = Agent::new("test".into(), Core::new(client));
    let conv_id = agent.conversation_id;

    let result = agent
        .execute(Some(make_messages(conv_id)), CancellationToken::new())
        .await;
    assert!(result.is_ok());

    let stats = &agent.state.stats;
    // system + user + assistant = 3 messages returned from tool loop
    assert!(stats.total_messages >= 3);
    // 50 prompt + 30 completion = 80 tokens
    assert_eq!(stats.tokens_used, 80);
    assert_eq!(stats.successful_tool_calls, 0);
    assert_eq!(stats.failed_tool_calls, 0);
}

#[tokio::test]
async fn stats_accumulate_across_executions() {
    let client = MockLLMClient::new();
    let mut agent = Agent::new("test".into(), Core::new(client));
    let conv_id = agent.conversation_id;

    agent
        .execute(Some(make_messages(conv_id)), CancellationToken::new())
        .await
        .unwrap();
    let first_tokens = agent.state.stats.tokens_used;
    let first_messages = agent.state.stats.total_messages;

    agent
        .execute(Some(make_messages(conv_id)), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(agent.state.stats.tokens_used, first_tokens * 2);
    assert_eq!(agent.state.stats.total_messages, first_messages * 2);
}

// -- conversation_history tests --

#[tokio::test]
async fn execute_records_conversation_history() {
    let client = MockLLMClient::new();
    let mut agent = Agent::new("test".into(), Core::new(client));
    let conv_id = agent.conversation_id;

    agent
        .execute(Some(make_messages(conv_id)), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(agent.state.conversation_history.len(), 1);
    let (msg, resp) = &agent.state.conversation_history[0];
    match msg {
        AgentMessage::UserInput(content) => {
            assert_eq!(content, "Hello");
        }
        other => panic!("Expected UserInput, got {other:?}"),
    }
    assert_eq!(resp.content.role, MessageRole::Assistant);
}

// -- execute_with_history tests --

#[tokio::test]
async fn execute_with_history_returns_full_message_vec() {
    let client = MockLLMClient::new();
    let mut agent = Agent::new("test".into(), Core::new(client));
    let conv_id = agent.conversation_id;

    let (response, history) = agent
        .execute_with_history(Some(make_messages(conv_id)), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(response.content.role, MessageRole::Assistant);
    // History must include the input we sent and the assistant reply, in order,
    // so callers can replay it verbatim on the next turn.
    assert!(
        history.len() >= 3,
        "expected at least [system, user, assistant], got {}",
        history.len()
    );
    assert_eq!(history[0].role, MessageRole::System);
    assert_eq!(history[1].role, MessageRole::User);
    assert!(history.iter().any(|m| m.role == MessageRole::Assistant));
}

#[tokio::test]
async fn execute_with_history_round_trips_as_next_turn_input() {
    let client = MockLLMClient::new();
    let mut agent = Agent::new("test".into(), Core::new(client));
    let conv_id = agent.conversation_id;

    let (_first, mut history) = agent
        .execute_with_history(Some(make_messages(conv_id)), CancellationToken::new())
        .await
        .unwrap();

    history.push(Message::user(conv_id, "follow-up"));

    let (_second, history2) = agent
        .execute_with_history(Some(history.clone()), CancellationToken::new())
        .await
        .unwrap();

    assert!(history2.len() > history.len());
    assert_eq!(history2[0].role, MessageRole::System);
    assert_eq!(history2[1].role, MessageRole::User);
}

// -- context_prompt() tests --

#[test]
fn context_prompt_returns_none_when_empty() {
    let state = AgentState::default();
    assert!(state.context_prompt().is_none());
}

#[test]
fn context_prompt_includes_task() {
    let mut state = AgentState::default();
    state.context.task = Some("Do something".into());

    let prompt = state.context_prompt().unwrap();
    assert!(prompt.contains("Current task: Do something"));
}

#[test]
fn context_prompt_includes_goals() {
    let mut state = AgentState::default();
    state.context.goals = vec!["Goal A".into(), "Goal B".into()];

    let prompt = state.context_prompt().unwrap();
    assert!(prompt.contains("Goals:"));
    assert!(prompt.contains("- Goal A"));
    assert!(prompt.contains("- Goal B"));
}

#[test]
fn context_prompt_includes_constraints() {
    let mut state = AgentState::default();
    state.context.constraints = vec!["Be safe".into()];

    let prompt = state.context_prompt().unwrap();
    assert!(prompt.contains("Constraints:"));
    assert!(prompt.contains("- Be safe"));
}

#[test]
fn context_prompt_includes_short_term_memory() {
    let mut state = AgentState::default();
    state.memory.short_term.push("User likes cats".into());

    let prompt = state.context_prompt().unwrap();
    assert!(prompt.contains("Recent context:"));
    assert!(prompt.contains("- User likes cats"));
}

#[test]
fn context_prompt_combines_all_sections() {
    let mut state = AgentState::default();
    state.context.task = Some("Research".into());
    state.context.goals.push("Find data".into());
    state.context.constraints.push("Time limit".into());
    state.memory.short_term.push("Note 1".into());

    let prompt = state.context_prompt().unwrap();
    assert!(prompt.contains("Current task:"));
    assert!(prompt.contains("Goals:"));
    assert!(prompt.contains("Constraints:"));
    assert!(prompt.contains("Recent context:"));
}

// -- ContextUpdate::apply tests --

#[test]
fn apply_set_task() {
    let mut state = AgentState::default();
    state.apply_context_update(ContextUpdate::SetTask("my task".into()));
    assert_eq!(state.context.task.as_deref(), Some("my task"));
}

#[test]
fn apply_add_and_remove_goal() {
    let mut state = AgentState::default();
    state.apply_context_update(ContextUpdate::AddGoal("A".into()));
    state.apply_context_update(ContextUpdate::AddGoal("B".into()));
    assert_eq!(state.context.goals, vec!["A", "B"]);

    state.apply_context_update(ContextUpdate::RemoveGoal("A".into()));
    assert_eq!(state.context.goals, vec!["B"]);
}

#[test]
fn apply_add_and_remove_constraint() {
    let mut state = AgentState::default();
    state.apply_context_update(ContextUpdate::AddConstraint("fast".into()));
    state.apply_context_update(ContextUpdate::AddConstraint("safe".into()));
    assert_eq!(state.context.constraints, vec!["fast", "safe"]);

    state.apply_context_update(ContextUpdate::RemoveConstraint("fast".into()));
    assert_eq!(state.context.constraints, vec!["safe"]);
}

#[test]
fn apply_set_environment_variable() {
    let mut state = AgentState::default();
    state.apply_context_update(ContextUpdate::SetEnvironmentVariable(
        "key".into(),
        "val".into(),
    ));
    assert_eq!(
        state.context.environment.get("key").map(String::as_str),
        Some("val")
    );
}

#[test]
fn apply_clear_memory() {
    let mut state = AgentState::default();
    state.memory.short_term.push("note".into());
    state.memory.long_term.insert("k".into(), "v".into());

    state.apply_context_update(ContextUpdate::ClearMemory);

    assert!(state.memory.short_term.is_empty());
    assert!(state.memory.long_term.is_empty());
    assert!(state.memory.working_memory.is_empty());
}

// -- AgentBuilder tests --

#[test]
fn builder_sets_id_and_prompts() {
    let client = MockLLMClient::new();
    let agent = Agent::builder("my-agent", client)
        .system_prompt("You are helpful.")
        .user_prompt("Hi there")
        .build();

    assert_eq!(agent.id, "my-agent");
    assert_eq!(agent.system_prompt.as_deref(), Some("You are helpful."));
    assert_eq!(agent.messages.len(), 2);
    assert_eq!(agent.messages[0].role, MessageRole::System);
    assert_eq!(agent.messages[1].role, MessageRole::User);
    assert_eq!(agent.messages[1].content, "Hi there");
}

#[test]
fn builder_sets_max_turns() {
    let client = MockLLMClient::new();
    let agent = Agent::builder("agent", client).max_turns(5).build();

    assert_eq!(agent.core.max_turns, Some(5));
}

#[test]
fn builder_sets_auto_approve() {
    let client = MockLLMClient::new();
    let agent = Agent::builder("agent", client)
        .auto_approve_tools(true)
        .build();

    assert!(agent.core.auto_approve_tools);
}

#[test]
fn builder_sets_tool_approval_callback() {
    use neuromance_common::tools::ToolApproval;

    let client = MockLLMClient::new();
    let agent = Agent::builder("agent", client)
        .with_tool_approval_callback(|_tc| async { ToolApproval::Approved })
        .build();

    assert!(agent.core.tool_approval_callback.is_some());
}

#[test]
fn builder_sets_tool_choice() {
    let client = MockLLMClient::new();
    let agent = Agent::builder("agent", client)
        .tool_choice(ToolChoice::None)
        .build();

    assert!(matches!(agent.tool_choice, ToolChoice::None));
}

// -- Context injection into system prompt --

#[tokio::test]
async fn context_injected_into_system_prompt() {
    let client = MockLLMClient::new();
    let mut agent = Agent::new("test".into(), Core::new(client));
    let conv_id = agent.conversation_id;

    agent.state.context.task = Some("Find cats".into());

    let result = agent
        .execute(Some(make_messages(conv_id)), CancellationToken::new())
        .await;
    assert!(result.is_ok());
    // Stats should be populated (proves the execute path ran)
    assert!(agent.state.stats.total_messages > 0);
}

// -- AgentState default --

#[test]
fn agent_state_default_is_empty() {
    let state = AgentState::default();
    assert!(state.messages.is_empty());
    assert!(state.conversation_history.is_empty());
    assert!(state.memory.short_term.is_empty());
    assert!(state.memory.long_term.is_empty());
    assert!(state.context.task.is_none());
    assert!(state.context.goals.is_empty());
    assert_eq!(state.stats.total_messages, 0);
    assert_eq!(state.stats.tokens_used, 0);
    assert_eq!(state.stats.successful_tool_calls, 0);
    assert_eq!(state.stats.failed_tool_calls, 0);
}

// -- Agent trait accessors --

#[test]
fn agent_id_returns_correct_value() {
    let client = MockLLMClient::new();
    let agent = Agent::new("my-id".into(), Core::new(client));
    assert_eq!(agent.id(), "my-id");
}

#[test]
fn agent_state_accessors() {
    let client = MockLLMClient::new();
    let mut agent = Agent::new("test".into(), Core::new(client));
    agent.state_mut().stats.total_messages = 42;
    assert_eq!(agent.state().stats.total_messages, 42);
}

// -- CacheMetrics total_output_tokens --

#[test]
fn cache_metrics_records_output_tokens() {
    let mut m = neuromance_common::CacheMetrics::default();
    assert_eq!(m.total_output_tokens, 0);

    m.record(&Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
        cost: None,
        input_tokens_details: None,
        output_tokens_details: None,
    });

    assert_eq!(m.total_output_tokens, 50);
    assert_eq!(m.total_input_tokens, 100);
}
