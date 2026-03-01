//! # neuromance-agent
//!
//! Agent framework for autonomous task execution with LLMs.
//!
//! This crate provides high-level abstractions for building autonomous agents that can
//! execute multi-step tasks, maintain state and memory, and use tools to accomplish goals.
//! Agents wrap the lower-level [`neuromance::Core`] functionality with task management,
//! state persistence, and sequential execution capabilities.
//!
//! ## Core Components
//!
//! - [`Agent`]: Trait defining the agent interface with state management and execution
//! - [`BaseAgent`]: Default implementation with conversation history and tool support
//! - [`AgentBuilder`]: Fluent builder for constructing agents with custom configuration
//! - [`AgentTask`]: Task abstraction for defining agent objectives and validation
//!
//! ## Agent State Management
//!
//! Agents maintain several types of state (from [`neuromance_common::agents`]):
//!
//! - **Conversation History**: Full message history and responses
//! - **Memory**: Short-term and long-term memory with working memory for active data
//! - **Context**: Task definition, goals, constraints, and environment variables
//! - **Statistics**: Execution metrics like token usage and tool call counts
//!
//! ## Example: Creating and Running an Agent
//!
//! ```rust,ignore
//! use neuromance_agent::{BaseAgent, Agent};
//! use neuromance::Core;
//! use neuromance_client::OpenAIClient;
//! use neuromance_common::{Config, Message};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create an LLM client
//! let config = Config::new("openai", "gpt-4")
//!     .with_api_key("sk-...");
//! let client = OpenAIClient::new(config)?;
//!
//! // Build an agent
//! let mut agent = BaseAgent::builder("research-agent", client)
//!     .system_prompt("You are a research assistant that finds information.")
//!     .user_prompt("Find the population of Tokyo.")
//!     .build();
//!
//! // Execute the agent
//! let response = agent.execute(None).await?;
//! println!("Agent response: {}", response.content.content);
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: Using the Agent Builder
//!
//! The [`AgentBuilder`] provides a fluent API for agent configuration:
//!
//! ```rust,ignore
//! use neuromance_agent::BaseAgent;
//! use neuromance_client::OpenAIClient;
//! use neuromance_common::Config;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = Config::new("openai", "gpt-4o-mini");
//! let client = OpenAIClient::new(config)?;
//!
//! let agent = BaseAgent::builder("task-agent", client)
//!     .system_prompt("You are a task completion agent.")
//!     .user_prompt("Complete the following task: organize these files.")
//!     .max_turns(5)
//!     .auto_approve_tools(true)
//!     .build();
//! # Ok(())
//! # }
//! ```
//!
//! ## Task-Based Execution
//!
//! The [`task`] module provides task abstractions for defining agent objectives:
//!
//! ```rust,ignore
//! use neuromance_agent::{AgentTask, BaseAgent};
//! use neuromance_common::Message;
//!
//! # async fn example() -> anyhow::Result<()> {
//! # let mut agent = unimplemented!();
//! // Define a task with validation
//! let task = AgentTask::new("research_task")
//!     .with_description("Research the history of Rust programming language")
//!     .with_validation(|response| {
//!         // Custom validation logic
//!         Ok(response.content.content.len() > 100)
//!     });
//!
//! // Execute the task
//! let response = task.execute(&mut agent).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Agent Lifecycle
//!
//! Agents follow a standard lifecycle:
//!
//! 1. **Creation**: Built with configuration and system/user prompts
//! 2. **Execution**: Process messages through LLM with tool support
//! 3. **State Updates**: Maintain conversation history and statistics
//! 4. **Reset**: Clear state for fresh execution (via [`Agent::reset`])
//!
//! ## Tool Integration
//!
//! Agents automatically integrate with [`neuromance_tools`] for tool execution.
//! Tools can be added to the agent's [`Core`] instance and will be available
//! during execution:
//!
//! ```rust,ignore
//! use neuromance_agent::BaseAgent;
//! use neuromance_tools::{ToolExecutor, ThinkTool};
//!
//! # async fn example() -> anyhow::Result<()> {
//! # let client = unimplemented!();
//! let mut agent = BaseAgent::new("agent-id".to_string(), Core::new(client));
//!
//! // Add tools to the agent's core
//! agent.core.tool_executor.add_tool(ThinkTool);
//!
//! // Tools are now available during execution
//! # Ok(())
//! # }
//! ```
//!
//! ## Memory and Context
//!
//! Agents maintain structured state via [`AgentState`]:
//!
//! - **Memory**: Stores short-term context and long-term knowledge
//! - **Context**: Task definition, goals, constraints, environment
//! - **Stats**: Execution metrics for monitoring and debugging
//!
//! This state can be serialized for persistence or debugging.

use async_trait::async_trait;
use log::info;
use uuid::Uuid;

use neuromance::Core;
use neuromance::error::CoreError;
use neuromance_client::LLMClient;
use neuromance_common::agents::{
    AgentContext, AgentMemory, AgentMessage, AgentResponse, AgentState, AgentStats,
};
use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::ToolChoice;

pub mod builder;
pub mod task;

pub use builder::AgentBuilder;
pub use task::{AgentTask, TaskResponse, TaskState};

/// Base agent implementation with common functionality
pub struct BaseAgent<C: LLMClient> {
    pub id: String,
    pub conversation_id: Uuid,
    pub core: Core<C>,
    pub state: AgentState,
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub tool_choice: ToolChoice,
}

impl<C: LLMClient> BaseAgent<C> {
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

#[async_trait]
impl<C: LLMClient + Send + Sync> Agent for BaseAgent<C> {
    fn id(&self) -> &str {
        &self.id
    }

    fn state(&self) -> &AgentState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut AgentState {
        &mut self.state
    }

    async fn reset(&mut self) -> Result<(), CoreError> {
        self.state.conversation_history.clear();
        self.state.memory = AgentMemory::default();
        self.state.context = AgentContext::default();
        self.state.stats = AgentStats::default();
        self.conversation_id = Uuid::new_v4();
        self.messages = Vec::<Message>::new();
        Ok(())
    }

    async fn execute(
        &mut self,
        messages: Option<Vec<Message>>,
    ) -> Result<AgentResponse, CoreError> {
        info!("Agent {} executing", self.id);
        self.core.tool_choice = self.tool_choice.clone();

        // Use provided messages or fall back to stored messages
        let mut messages = messages.unwrap_or_else(|| self.messages.clone());

        // Validate that we have at least system and user messages
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

        // Inject agent context into the system message
        if let Some(ctx) = self.state.context_prompt() {
            messages[0].content.push_str("\n\n");
            messages[0].content.push_str(&ctx);
        }

        // Snapshot Core counters before the tool loop
        let tokens_before = self.core.cache_metrics.total_input_tokens
            + self.core.cache_metrics.total_output_tokens;
        let success_before = self.core.successful_tool_calls;
        let fail_before = self.core.failed_tool_calls;

        let messages = self.core.chat_with_tool_loop(messages).await?;

        // Update stats with deltas from this execution
        self.state.stats.total_messages += messages.len();
        let tokens_after = self.core.cache_metrics.total_input_tokens
            + self.core.cache_metrics.total_output_tokens;
        #[allow(clippy::cast_possible_truncation)]
        {
            self.state.stats.tokens_used += (tokens_after - tokens_before) as usize;
        }
        self.state.stats.successful_tool_calls +=
            (self.core.successful_tool_calls - success_before) as usize;
        self.state.stats.failed_tool_calls += (self.core.failed_tool_calls - fail_before) as usize;

        // Extract the final assistant message and tool responses
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

        // Record conversation history
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

#[async_trait]
pub trait Agent: Send + Sync {
    /// Returns the unique identifier of the agent.
    fn id(&self) -> &str;

    /// Returns immutable reference to the agent's current state.
    fn state(&self) -> &AgentState;

    /// Returns mutable reference to the agent's current state.
    fn state_mut(&mut self) -> &mut AgentState;

    /// Resets the agent to its initial state.
    ///
    /// # Returns
    /// A result indicating success or failure of the reset operation
    async fn reset(&mut self) -> Result<(), CoreError>;

    /// Execute core chat with tools loop
    async fn execute(&mut self, messages: Option<Vec<Message>>)
    -> Result<AgentResponse, CoreError>;
}
