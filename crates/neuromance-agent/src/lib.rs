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
//!     .with_system_prompt("You are a research assistant that finds information.")
//!     .with_user_prompt("Find the population of Tokyo.")
//!     .build()?;
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
//!     .with_system_prompt("You are a task completion agent.")
//!     .with_user_prompt("Complete the following task: organize these files.")
//!     .with_max_turns(5)
//!     .with_auto_approve_tools(true)
//!     .build()?;
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

use anyhow::Result;
use async_trait::async_trait;
use log::info;
use uuid::Uuid;

use neuromance::Core;
use neuromance_client::LLMClient;
use neuromance_common::agents::{AgentContext, AgentMemory, AgentResponse, AgentState, AgentStats};
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
    pub user_prompt: Option<String>,
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
            user_prompt: None,
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

    async fn reset(&mut self) -> Result<()> {
        self.state.conversation_history.clear();
        self.state.memory = AgentMemory::default();
        self.state.context = AgentContext::default();
        self.state.stats = AgentStats::default();
        self.conversation_id = Uuid::new_v4();
        self.messages = Vec::<Message>::new();
        Ok(())
    }

    async fn execute(&mut self, messages: Option<Vec<Message>>) -> Result<AgentResponse> {
        info!("Agent {} executing", self.id);
        self.core.tool_choice = self.tool_choice.clone();

        // Use provided messages or fall back to stored messages
        let messages = messages.unwrap_or_else(|| self.messages.clone());

        // Validate that we have at least system and user messages
        if messages.len() < 2 {
            return Err(anyhow::anyhow!(
                "Agent requires at least a system message and user message to execute"
            ));
        }

        if messages[0].role != MessageRole::System {
            return Err(anyhow::anyhow!(
                "First message must be a system message, found: {:?}",
                messages[0].role
            ));
        }

        if messages[1].role != MessageRole::User {
            return Err(anyhow::anyhow!(
                "Second message must be a user message, found: {:?}",
                messages[1].role
            ));
        }

        let messages = self.core.chat_with_tool_loop(messages).await?;

        // Extract the final assistant message and tool responses
        let content = messages
            .iter()
            .filter(|m| m.role == MessageRole::Assistant)
            .next_back()
            .cloned()
            .unwrap_or_else(|| {
                Message::new(
                    self.conversation_id,
                    MessageRole::Assistant,
                    "No final response generated".to_string(),
                )
            });

        let tool_responses = messages
            .iter()
            .filter(|m| m.role == MessageRole::Tool)
            .cloned()
            .collect();

        Ok(AgentResponse {
            content,
            reasoning: None,
            tool_responses,
        })
    }
}

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
    async fn reset(&mut self) -> Result<()>;

    /// Execute core chat with tools loop
    async fn execute(&mut self, messages: Option<Vec<Message>>) -> Result<AgentResponse>;
}
