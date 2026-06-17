use std::future::Future;
use std::sync::Arc;

use uuid::Uuid;

use neuromance::Core;
use neuromance_client::LLMClient;
use neuromance_common::agents::AgentState;
use neuromance_common::chat::Message;
use neuromance_common::client::ToolChoice;
use neuromance_common::hook::{FnReviewHook, Hook};
use neuromance_common::tools::{ToolApproval, ToolCall};
use neuromance_context::skills::SkillCatalog;
use neuromance_tools::{SkillTool, ToolImplementation};

use crate::Agent;

/// Skills wiring captured by [`AgentBuilder::skills`], applied at build time.
struct BuilderSkills {
    catalog: Arc<SkillCatalog>,
    menu_budget: usize,
    body_budget: usize,
}

/// Simple builder for creating agents with common configuration
///
/// # Example
/// ```rust,no_run
/// # use neuromance_agent::AgentBuilder;
/// # use neuromance_client::chat_completions::client::ChatCompletionsClient;
/// # use neuromance_common::client::Config;
/// # use neuromance_tools::generic::CurrentTimeTool;
/// # fn example() -> anyhow::Result<()> {
/// let config = Config::new("openai", "gpt-4");
/// let client = ChatCompletionsClient::new(config)?;
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
    skills: Option<BuilderSkills>,
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
            skills: None,
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

    /// Register a lifecycle [`Hook`] on the underlying [`Core`].
    #[must_use]
    pub fn with_hook(mut self, hook: Arc<dyn Hook>) -> Self {
        self.core = self.core.with_hook(hook);
        self
    }

    /// Decide tool approval per call via a closure.
    ///
    /// Wraps `callback` in a [`Hook`] whose `review_tool` answers approvals
    /// internally instead of yielding `CoreEvent::ApprovalRequest`.
    /// `auto_approve_tools` still short-circuits it when enabled.
    ///
    /// # Arguments
    /// * `callback` - Async function called for each non-auto-approved tool call
    #[must_use]
    pub fn with_tool_approval_callback<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(&ToolCall) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolApproval> + Send + 'static,
    {
        self.core = self.core.with_hook(Arc::new(FnReviewHook::new(callback)));
        self
    }

    /// Set the tool choice strategy
    ///
    /// # Arguments
    /// * `tool_choice` - Tool choice strategy (Auto, None, Required, or specific function)
    #[must_use]
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Enable skills from `catalog`.
    ///
    /// At build time this registers the `load_skill` tool (so the model can
    /// pull a skill's body into context) and injects the skills menu as a
    /// system message. To also support `$mention` invocation, expand mentions
    /// in incoming user input with
    /// [`SkillCatalog::mention_messages`](neuromance_context::skills::SkillCatalog::mention_messages)
    /// and prepend the result to the conversation.
    ///
    /// # Arguments
    /// * `catalog` - The skill catalog to serve from
    /// * `menu_budget` - Byte budget for the injected menu
    /// * `body_budget` - Byte budget for each loaded skill body
    #[must_use]
    pub fn skills(
        mut self,
        catalog: Arc<SkillCatalog>,
        menu_budget: usize,
        body_budget: usize,
    ) -> Self {
        self.skills = Some(BuilderSkills {
            catalog,
            menu_budget,
            body_budget,
        });
        self
    }

    /// Build the agent
    ///
    /// # Returns
    /// A fully configured `Agent` instance
    pub fn build(mut self) -> Agent<C> {
        let conversation_id = Uuid::new_v4();

        // The load_skill tool is read-only and auto-approved, registered before
        // the conversation is seeded so it is available on the first turn.
        if let Some(skills) = &self.skills {
            self.core.tool_executor.add_tool(SkillTool::new(
                Arc::clone(&skills.catalog),
                skills.body_budget,
            ));
        }

        // Build messages from system and user prompts
        let mut messages = Vec::new();

        if let Some(ref prompt) = self.system_prompt {
            messages.push(Message::system(conversation_id, prompt));
        }

        // The menu is stable for the conversation's life, so it sits in a
        // system message without harming prompt caching.
        if let Some(skills) = &self.skills
            && let Some(menu) = skills.catalog.menu(skills.menu_budget)
        {
            messages.push(Message::system(conversation_id, menu));
        }

        if let Some(ref prompt) = self.user_prompt {
            messages.push(Message::user(conversation_id, prompt));
        }

        Agent {
            id: self.id,
            conversation_id,
            core: self.core,
            state: AgentState::default(),
            system_prompt: self.system_prompt,
            messages,
            tool_choice: self.tool_choice,
        }
    }
}
