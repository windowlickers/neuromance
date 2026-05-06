//! # neuromance-agent
//!
//! Agent framework for autonomous multi-turn task execution with LLMs.
//!
//! Agents wrap [`neuromance::Core`] with state management, memory, and
//! sequential tool-using execution.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use neuromance::{ChatCompletionsClient, Config};
//! use neuromance_agent::{Agent, AgentBuilder};
//! use tokio_util::sync::CancellationToken;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = Config::new("openai", "gpt-4").with_api_key("sk-...");
//! let client = ChatCompletionsClient::new(config)?;
//!
//! let mut agent = AgentBuilder::new("research", client)
//!     .system_prompt("You are a research assistant.")
//!     .user_prompt("Find the population of Tokyo.")
//!     .max_turns(5)
//!     .auto_approve_tools(true)
//!     .build();
//!
//! let response = agent.execute(None, CancellationToken::new()).await?;
//! println!("{}", response.content.content);
//! # Ok(())
//! # }
//! ```

use log::info;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use neuromance::Core;
use neuromance::error::CoreError;
use neuromance_client::LLMClient;
use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::ToolChoice;

pub mod builder;

// --- Agent core ---
pub use builder::AgentBuilder;

// --- Agent state types (live in neuromance-common so they can be shared,
//     surfaced here so agent consumers only need this crate) ---
pub use neuromance_common::agents::{
    AgentContext, AgentMemory, AgentMessage, AgentResponse, AgentState, AgentStats, ContextUpdate,
};

/// Concrete agent: wraps [`Core`] with state, memory, and a tool-using execution loop.
pub struct Agent<C: LLMClient> {
    pub id: String,
    pub conversation_id: Uuid,
    pub core: Core<C>,
    pub state: AgentState,
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub tool_choice: ToolChoice,
}

impl<C: LLMClient> Agent<C> {
    pub fn new(id: String, core: Core<C>) -> Self {
        Self {
            id,
            conversation_id: Uuid::new_v4(),
            core,
            state: AgentState::default(),
            system_prompt: None,
            messages: Vec::<Message>::new(),
            tool_choice: ToolChoice::Auto,
        }
    }

    pub fn builder(id: impl Into<String>, client: C) -> AgentBuilder<C> {
        AgentBuilder::new(id, client)
    }
}

impl<C: LLMClient + Send + Sync> Agent<C> {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub const fn state(&self) -> &AgentState {
        &self.state
    }

    pub const fn state_mut(&mut self) -> &mut AgentState {
        &mut self.state
    }

    /// Reset conversation history, memory, context, stats, and message buffer.
    ///
    /// # Errors
    /// Returns [`CoreError`] for forward-compatibility with future
    /// implementations that may perform fallible I/O during reset.
    #[allow(clippy::unused_async)]
    pub async fn reset(&mut self) -> Result<(), CoreError> {
        self.state.conversation_history.clear();
        self.state.memory = AgentMemory::default();
        self.state.context = AgentContext::default();
        self.state.stats = AgentStats::default();
        self.conversation_id = Uuid::new_v4();
        self.messages = Vec::<Message>::new();
        Ok(())
    }

    /// Run the chat-with-tools loop and return the assistant's final response.
    ///
    /// # Errors
    /// Returns [`CoreError::InvalidInput`] if the message slice does not start
    /// with a system message followed by a user message; propagates any error
    /// from the underlying [`Core::chat_with_tool_loop`] (network, cancellation,
    /// tool failure); returns [`CoreError::NoResponse`] if the model produced
    /// no assistant message.
    pub async fn execute(
        &mut self,
        messages: Option<Vec<Message>>,
        cancel: CancellationToken,
    ) -> Result<AgentResponse, CoreError> {
        info!("Agent {} executing", self.id);
        self.core.tool_choice = self.tool_choice.clone();

        let mut messages = messages.unwrap_or_else(|| self.messages.clone());

        if messages.len() < 2 {
            return Err(CoreError::InvalidInput(
                "Agent requires at least a system message and \
                 user message to execute"
                    .to_string(),
            ));
        }

        if messages[0].role != MessageRole::System {
            return Err(CoreError::InvalidInput(format!(
                "First message must be a system message, found: {:?}",
                messages[0].role
            )));
        }

        if messages[1].role != MessageRole::User {
            return Err(CoreError::InvalidInput(format!(
                "Second message must be a user message, found: {:?}",
                messages[1].role
            )));
        }

        if let Some(ctx) = self.state.context_prompt() {
            messages[0].content.push_str("\n\n");
            messages[0].content.push_str(&ctx);
        }

        let (messages, run_stats) = self.core.chat_with_tool_loop(messages, cancel).await?;

        self.state.stats.total_messages += messages.len();
        #[allow(clippy::cast_possible_truncation)]
        {
            let tokens = run_stats.cache_metrics.total_input_tokens
                + run_stats.cache_metrics.total_output_tokens;
            self.state.stats.tokens_used += tokens as usize;
            self.state.stats.successful_tool_calls += run_stats.successful_tool_calls as usize;
            self.state.stats.failed_tool_calls += run_stats.failed_tool_calls as usize;
        }

        let content = messages
            .iter()
            .rfind(|m| m.role == MessageRole::Assistant)
            .cloned()
            .ok_or_else(|| {
                CoreError::NoResponse("LLM returned no assistant message".to_string())
            })?;

        let tool_responses: Vec<Message> = messages
            .iter()
            .filter(|m| m.role == MessageRole::Tool)
            .cloned()
            .collect();

        let response = AgentResponse {
            content,
            reasoning: None,
            tool_responses,
        };

        let user_content = messages
            .iter()
            .find(|m| m.role == MessageRole::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();
        self.state
            .conversation_history
            .push((AgentMessage::UserInput(user_content), response.clone()));

        Ok(response)
    }
}

#[cfg(test)]
mod tests;
