use uuid::Uuid;

use neuromance::Core;
use neuromance_client::LLMClient;
use neuromance_common::agents::AgentState;
use neuromance_common::chat::Message;
use neuromance_common::client::ToolChoice;
use neuromance_tools::ToolImplementation;

use crate::BaseAgent;

/// Simple builder for creating agents with common configuration
///
/// # Example
/// ```rust,no_run
/// # use neuromance_agent::AgentBuilder;
/// # use neuromance_client::openai::client::OpenAIClient;
/// # use neuromance_common::client::Config;
/// # use neuromance_tools::generic::CurrentTimeTool;
/// # fn example() -> anyhow::Result<()> {
/// let config = Config::new("openai", "gpt-4");
/// let client = OpenAIClient::new(config)?;
///
/// let agent = AgentBuilder::new("my-agent", client)
///     .system_prompt("You are a helpful assistant")
///     .add_tool(CurrentTimeTool)
///     .max_turns(5)
///     .auto_approve_tools(true)
///     .build();
/// # Ok(())
/// # }
/// ```
pub struct AgentBuilder<C: LLMClient> {
    id: String,
    core: Core<C>,
    system_prompt: Option<String>,
    user_prompt: Option<String>,
    tool_choice: ToolChoice,
}

impl<C: LLMClient> AgentBuilder<C> {
    /// Create a new agent builder
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the agent
    /// * `client` - LLM client implementation
    pub fn new(id: impl Into<String>, client: C) -> Self {
        Self {
            id: id.into(),
            core: Core::new(client),
            system_prompt: None,
            user_prompt: None,
            tool_choice: ToolChoice::Auto,
        }
    }

    /// Set the system prompt for the agent
    ///
    /// # Arguments
    /// * `prompt` - System prompt text
    #[must_use]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the user prompt for the agent
    ///
    /// # Arguments
    /// * `prompt` - User prompt text
    #[must_use]
    pub fn user_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.user_prompt = Some(prompt.into());
        self
    }

    /// Add a tool to the agent
    ///
    /// # Arguments
    /// * `tool` - Tool implementation to register
    #[must_use]
    pub fn add_tool<T: ToolImplementation + 'static>(mut self, tool: T) -> Self {
        self.core.tool_executor.add_tool(tool);
        self
    }

    /// Set maximum number of turns in the chat loop
    ///
    /// # Arguments
    /// * `max_turns` - Maximum number of turns before the loop exits
    #[must_use]
    pub const fn max_turns(mut self, max_turns: u32) -> Self {
        self.core.max_turns = Some(max_turns);
        self
    }

    /// Enable automatic approval of all tools
    ///
    /// # Arguments
    /// * `auto_approve` - Whether to auto-approve all tools
    #[must_use]
    pub const fn auto_approve_tools(mut self, auto_approve: bool) -> Self {
        self.core.auto_approve_tools = auto_approve;
        self
    }

    /// Set the tool choice strategy
    ///
    /// # Arguments
    /// * `tool_choice` - Tool choice strategy (Auto, None, Required, or specific function)
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Build the agent
    ///
    /// # Returns
    /// A fully configured `BaseAgent` instance
    pub fn build(self) -> BaseAgent<C> {
        let conversation_id = Uuid::new_v4();

        // Build messages from system and user prompts
        let mut messages = Vec::new();

        if let Some(ref prompt) = self.system_prompt {
            messages.push(Message::system(conversation_id, prompt));
        }

        if let Some(ref prompt) = self.user_prompt {
            messages.push(Message::user(conversation_id, prompt));
        }

        BaseAgent {
            id: self.id,
            conversation_id,
            core: self.core,
            state: AgentState::default(),
            system_prompt: self.system_prompt,
            user_prompt: self.user_prompt,
            messages,
            tool_choice: self.tool_choice,
        }
    }
}
