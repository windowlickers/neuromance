#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::panic)]

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::Stream;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use neuromance::Core;
use neuromance_client::{ClientError, LLMClient};
use neuromance_common::agents::{AgentMessage, AgentState, ContextUpdate};
use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::{ChatChunk, ChatRequest, ChatResponse, Config, ToolChoice, Usage};
use neuromance_common::delegation::{self, DelegationContext};
use neuromance_common::tools::{Function, FunctionCall, Tool, ToolCall};
use neuromance_tools::{ToolError, ToolImplementation};
use tracing_subscriber::layer::SubscriberExt;

use crate::Agent;

struct MockLLMClient {
    config: Config,
    usage: Usage,
    reply: String,
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
            reply: "Mock response".to_string(),
        }
    }

    fn with_reply(reply: &str) -> Self {
        Self {
            reply: reply.to_string(),
            ..Self::new()
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
            message: Message::assistant(conv_id, &self.reply),
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
            delta_content: Some(self.reply.clone()),
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

    fn supports_structured_output(&self) -> bool {
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

#[tokio::test]
async fn builder_sets_tool_approval_callback() {
    use neuromance_common::hook::HookContext;
    use neuromance_common::tools::{ToolApproval, ToolCall};

    let client = MockLLMClient::new();
    let agent = Agent::builder("agent", client)
        .with_tool_approval_callback(|_tc| async { ToolApproval::Approved })
        .build();

    // The registered review hook actually decides a tool call, not merely
    // occupies a slot.
    let ctx = HookContext::new(uuid::Uuid::new_v4(), 0);
    let decision = agent.core.hooks[0]
        .review_tool(&ctx, &ToolCall::new("t", "{}"))
        .await
        .unwrap();
    assert_eq!(decision, Some(ToolApproval::Approved));
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

// -- Delegation context propagation --

/// A client that calls `ctx_probe` on its first turn, then finishes. Drives the
/// tool loop exactly once so a tool runs inside the agent's delegation scope.
struct ToolCallingMock {
    config: Config,
    calls: AtomicUsize,
}

impl ToolCallingMock {
    fn new() -> Self {
        Self {
            config: Config::new("mock", "mock-model"),
            calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl LLMClient for ToolCallingMock {
    fn config(&self) -> &Config {
        &self.config
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ClientError> {
        let conv_id = request
            .messages
            .first()
            .map_or_else(Uuid::new_v4, |m| m.conversation_id);
        let message = if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            Message::assistant(conv_id, "")
                .with_tool_calls(vec![ToolCall {
                    id: "call_1".to_string(),
                    function: FunctionCall {
                        name: "ctx_probe".to_string(),
                        arguments: "{}".to_string(),
                    },
                    call_type: "function".to_string(),
                    index: None,
                }])
                .expect("assistant message accepts tool calls")
        } else {
            Message::assistant(conv_id, "done")
        };
        Ok(ChatResponse {
            message,
            model: "mock-model".to_string(),
            usage: None,
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
        panic!("ToolCallingMock does not stream")
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_structured_output(&self) -> bool {
        true
    }

    fn supports_streaming(&self) -> bool {
        false
    }
}

/// Records the delegation parent it observes from the task-local context when
/// run. A subagent spawned during the parent's run would read the same value.
struct CtxProbe {
    // Outer `None`: the probe never ran. Inner `None`: it ran but observed no
    // delegation parent. The two cases must stay distinct for the assertion.
    #[allow(clippy::option_option)]
    seen: Arc<Mutex<Option<Option<Uuid>>>>,
}

#[async_trait]
impl ToolImplementation for CtxProbe {
    fn get_definition(&self) -> Tool {
        Tool::builder()
            .function(Function {
                name: "ctx_probe".to_string(),
                description: "records the observed delegation context".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            })
            .build()
    }

    async fn execute(&self, _args: &serde_json::Value) -> Result<String, ToolError> {
        let observed = neuromance_common::delegation::current().conversation_id;
        *self.seen.lock().expect("probe mutex") = Some(observed);
        Ok("ok".to_string())
    }

    fn is_auto_approved(&self) -> bool {
        true
    }
}

/// While an agent runs, its own conversation id is published to the delegation
/// context, so a tool (and any subagent it spawns) observes that id as its
/// parent. Without the scope wiring the probe would observe `None`.
#[tokio::test]
async fn execute_publishes_conversation_id_as_delegation_parent() {
    let seen = Arc::new(Mutex::new(None));
    let mut agent = Agent::new("parent".into(), Core::new(ToolCallingMock::new()));
    agent.core.auto_approve_tools = true;
    agent.core.tool_executor.add_tool(CtxProbe {
        seen: Arc::clone(&seen),
    });
    let conv_id = agent.conversation_id;

    agent
        .execute(Some(make_messages(conv_id)), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(
        *seen.lock().unwrap(),
        Some(Some(conv_id)),
        "probe should observe the running agent's conversation as its delegation parent"
    );
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

/// A skill source backing exactly one skill, for builder integration tests.
struct OneSkillSource;

#[async_trait]
impl neuromance_context::skills::SkillSource for OneSkillSource {
    async fn list(
        &self,
    ) -> Result<
        Vec<neuromance_context::skills::SkillMetadata>,
        neuromance_context::skills::SkillError,
    > {
        use neuromance_context::skills::{SkillId, SkillLocator, SkillMetadata};
        Ok(vec![SkillMetadata {
            id: SkillId::new("deploy"),
            name: "deploy".to_string(),
            description: "deploy the app".to_string(),
            locator: SkillLocator::Remote {
                endpoint: "mem://skills".to_string(),
                id: "deploy".to_string(),
            },
            extra: serde_yaml::Mapping::default(),
        }])
    }

    async fn load_body(
        &self,
        _id: &neuromance_context::skills::SkillId,
    ) -> Result<String, neuromance_context::skills::SkillError> {
        Ok("deploy instructions".to_string())
    }
}

#[tokio::test]
async fn test_builder_skills_registers_tool_and_keeps_clean_seed() {
    let catalog = Arc::new(
        neuromance_context::skills::SkillCatalog::build(vec![Box::new(OneSkillSource)]).await,
    );
    let agent = crate::AgentBuilder::new("a", MockLLMClient::new())
        .system_prompt("sys")
        .user_prompt("do it")
        .skills(catalog, 8192, 8192)
        .build();

    assert!(agent.core.tool_executor.has_tool("load_skill"));
    // The menu is injected by the SkillsHook inside the conversation loop, not
    // baked into the seed, so the seed stays a precondition-satisfying
    // [System, User] pair rather than [System, System(menu), User].
    assert_eq!(agent.messages.len(), 2);
    assert_eq!(agent.messages[0].role, MessageRole::System);
    assert_eq!(agent.messages[1].role, MessageRole::User);
    assert!(
        agent
            .messages
            .iter()
            .all(|m| !m.content.contains("<skills_instructions>")),
        "menu must not be baked into the seed"
    );
}

#[tokio::test]
async fn test_builder_without_skills_has_no_menu_or_tool() {
    let agent = crate::AgentBuilder::new("a", MockLLMClient::new())
        .system_prompt("sys")
        .build();

    assert!(!agent.core.tool_executor.has_tool("load_skill"));
    assert!(
        agent
            .messages
            .iter()
            .all(|m| !m.content.contains("<skills_instructions>"))
    );
}

/// A schema-carrying run parses the final message once, in Core, and hands the same value out on
/// the response — callers never re-parse the prose.
#[tokio::test]
async fn test_execute_populates_structured_output() {
    let client = MockLLMClient::with_reply(r#"{"answer": "42"}"#);
    let mut agent = Agent::builder("test", client)
        .output_schema(
            neuromance_common::client::OutputSchema::new(
                "answer",
                serde_json::json!({
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }),
            )
            .unwrap(),
        )
        .build();

    let response = agent
        .execute(
            Some(make_messages(Uuid::new_v4())),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.structured,
        Some(serde_json::json!({"answer": "42"}))
    );
}

/// Without a schema there is nothing to parse, so `structured` stays empty even when the model
/// happens to answer in JSON.
#[tokio::test]
async fn test_execute_leaves_structured_empty_without_a_schema() {
    let client = MockLLMClient::with_reply(r#"{"answer": "42"}"#);
    let mut agent = Agent::new("test".into(), Core::new(client));

    let response = agent
        .execute(
            Some(make_messages(Uuid::new_v4())),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_eq!(response.structured, None);
}

// -- Span tests --

/// A closed span's name and the stringified fields it carried.
type ClosedSpan = (String, HashMap<String, String>);

/// Collects the fields recorded on every closed span, keyed by span name.
#[derive(Clone, Default)]
struct SpanCapture(Arc<Mutex<Vec<ClosedSpan>>>);

impl SpanCapture {
    fn fields_for(&self, name: &str) -> Option<HashMap<String, String>> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .find(|(span, _)| span == name)
            .map(|(_, fields)| fields.clone())
    }
}

/// `tracing`'s visitor API hands values back one at a time; stringify each so a
/// test can assert on it without knowing the field's static type.
struct FieldVisitor<'a>(&'a mut HashMap<String, String>);

impl tracing::field::Visit for FieldVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().to_string(), format!("{value:?}"));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
}

impl<S> tracing_subscriber::Layer<S> for SpanCapture
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut fields = HashMap::new();
        attrs.record(&mut FieldVisitor(&mut fields));
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(fields);
        }
    }

    fn on_record(
        &self,
        id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if let Some(span) = ctx.span(id)
            && let Some(fields) = span.extensions_mut().get_mut::<HashMap<String, String>>()
        {
            values.record(&mut FieldVisitor(fields));
        }
    }

    fn on_close(&self, id: tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(span) = ctx.span(&id) {
            let fields = span
                .extensions()
                .get::<HashMap<String, String>>()
                .cloned()
                .unwrap_or_default();
            self.0
                .lock()
                .unwrap()
                .push((span.name().to_string(), fields));
        }
    }
}

/// The run span must wrap the whole execution, not a helper called at the end of
/// it. `execute_with_history` records `parent_conversation_id` onto the current
/// span, so a span attached to the wrong function silently drops the field.
#[test]
fn test_execute_span_records_the_delegation_parent() {
    let capture = SpanCapture::default();
    let subscriber = tracing_subscriber::registry().with(capture.clone());

    let parent_conversation = Uuid::new_v4();
    let task = Uuid::new_v4();
    let parent_ctx = DelegationContext {
        conversation_id: Some(parent_conversation),
        task_id: Some(task),
        parent_message_id: None,
        parent_tool_call_id: None,
        workspace_dir: None,
    };

    let mut agent = Agent::new("child".into(), Core::new(MockLLMClient::new()));
    let conversation_id = agent.conversation_id;

    // A current-thread runtime keeps the whole run on the thread that
    // `with_default` installed the subscriber on.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    tracing::subscriber::with_default(subscriber, || {
        runtime
            .block_on(delegation::scope(
                parent_ctx,
                agent.execute(
                    Some(make_messages(Uuid::new_v4())),
                    CancellationToken::new(),
                ),
            ))
            .unwrap();
    });

    let fields = capture
        .fields_for("invoke_agent")
        .expect("the run should produce an invoke_agent span");
    assert_eq!(
        fields.get("parent_conversation_id").map(String::as_str),
        Some(parent_conversation.to_string().as_str()),
    );
    assert_eq!(
        fields.get("task_id").map(String::as_str),
        Some(task.to_string().as_str()),
    );
    assert_eq!(
        fields.get("agent_id").map(String::as_str),
        Some("child"),
        "the span must belong to the agent that ran",
    );
    assert_eq!(
        fields.get("conversation_id").map(String::as_str),
        Some(conversation_id.to_string().as_str()),
    );
}

/// Agent dashboards key off `gen_ai.agent.*` and the span name. A span called
/// `agent.execute` with a field called `agent_id` is invisible to them.
#[test]
fn test_execute_span_carries_the_genai_agent_attributes() {
    use neuromance_common::telemetry::genai;

    let capture = SpanCapture::default();
    let subscriber = tracing_subscriber::registry().with(capture.clone());

    let mut agent = Agent::new("researcher".into(), Core::new(MockLLMClient::new()));
    let conversation_id = agent.conversation_id;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    tracing::subscriber::with_default(subscriber, || {
        runtime
            .block_on(agent.execute(
                Some(make_messages(Uuid::new_v4())),
                CancellationToken::new(),
            ))
            .unwrap();
    });

    let fields = capture
        .fields_for("invoke_agent")
        .expect("the run should produce an invoke_agent span");

    assert_eq!(
        fields.get("otel.name").map(String::as_str),
        Some("invoke_agent researcher")
    );
    assert_eq!(
        fields.get(genai::OPERATION_NAME).map(String::as_str),
        Some("invoke_agent")
    );
    assert_eq!(
        fields.get(genai::AGENT_ID).map(String::as_str),
        Some("researcher")
    );
    assert_eq!(
        fields.get(genai::CONVERSATION_ID).map(String::as_str),
        Some(conversation_id.to_string().as_str()),
    );
}

/// A failed run must be findable as a failure. `tracing-opentelemetry` derives an
/// error status only from an ERROR-level event, and a failed run returns its error
/// to the caller instead of logging one — so without the explicit record every
/// failure would export as a successful span.
#[test]
fn test_execute_span_marks_a_failed_run_as_an_error() {
    use neuromance_common::telemetry::genai;

    let capture = SpanCapture::default();
    let subscriber = tracing_subscriber::registry().with(capture.clone());

    let mut agent = Agent::new("researcher".into(), Core::new(MockLLMClient::new()));

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    tracing::subscriber::with_default(subscriber, || {
        // One message trips the leading system/user check, so the run fails
        // before it reaches the client.
        let result = runtime.block_on(agent.execute(
            Some(vec![Message::user(Uuid::new_v4(), "Hello")]),
            CancellationToken::new(),
        ));
        assert!(result.is_err(), "the run should fail");
    });

    let fields = capture
        .fields_for("invoke_agent")
        .expect("the run should produce an invoke_agent span");

    assert_eq!(
        fields.get("otel.status_code").map(String::as_str),
        Some("ERROR"),
    );
    assert_eq!(
        fields.get(genai::ERROR_TYPE).map(String::as_str),
        Some("invalid_input"),
    );
    assert!(
        fields.contains_key("otel.status_description"),
        "the failure needs a description, got {fields:?}",
    );
}

/// The success arm must set a status too: a span with no `otel.status_code` is
/// `Unset`, which a backend renders the same as an unfinished call.
#[test]
fn test_execute_span_marks_a_successful_run_as_ok() {
    let capture = SpanCapture::default();
    let subscriber = tracing_subscriber::registry().with(capture.clone());

    let mut agent = Agent::new("researcher".into(), Core::new(MockLLMClient::new()));

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    tracing::subscriber::with_default(subscriber, || {
        runtime
            .block_on(agent.execute(
                Some(make_messages(Uuid::new_v4())),
                CancellationToken::new(),
            ))
            .unwrap();
    });

    let fields = capture
        .fields_for("invoke_agent")
        .expect("the run should produce an invoke_agent span");

    assert_eq!(
        fields.get("otel.status_code").map(String::as_str),
        Some("OK"),
    );
}
