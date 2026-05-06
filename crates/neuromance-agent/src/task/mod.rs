use std::fmt::Write;
use std::sync::Arc;

use neuromance::error::CoreError;
use neuromance_tools::{ToolImplementation, ToolRegistry};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use neuromance_client::LLMClient;
use neuromance_common::agents::AgentResponse;
use neuromance_common::tools::Tool;

mod agents;
use agents::{ActionAgent, ContextAgent, VerifierAgent};

/// Represents a complete agent task composed of three specialized agents:
/// - `ContextAgent`: Gathers and analyzes context
/// - `ActionAgent`: Takes actions based on context
/// - `VerifierAgent`: Verifies actions were successful
pub struct AgentTask<C: LLMClient> {
    pub id: String,
    pub conversation_id: Uuid,
    pub task_description: String,
    pub context_agent: ContextAgent<C>,
    pub action_agent: ActionAgent<C>,
    pub verifier_agent: VerifierAgent<C>,
    pub state: TaskState,
    pub tool_registry: ToolRegistry,
}

#[derive(Debug, Clone, Default)]
pub struct TaskState {
    pub verified: bool,
    pub context_response: Option<AgentResponse>,
    pub action_response: Option<AgentResponse>,
    pub verification_response: Option<AgentResponse>,
}

/// Response from executing the full agent task
#[derive(Debug, Clone)]
pub struct TaskResponse {
    pub context_response: AgentResponse,
    pub action_response: AgentResponse,
    pub verification_response: AgentResponse,
    pub success: bool,
}

impl<C: LLMClient + Send + Sync> AgentTask<C> {
    pub fn new(id: impl Into<String>, task_description: impl Into<String>, client: C) -> Self
    where
        C: Clone,
    {
        let conversation_id = Uuid::new_v4();
        let id = id.into();
        let task_description = task_description.into();

        Self {
            id: id.clone(),
            conversation_id,
            task_description,
            context_agent: ContextAgent::new(&id, client.clone()),
            action_agent: ActionAgent::new(&id, client.clone()),
            verifier_agent: VerifierAgent::new(&id, client),
            state: TaskState::default(),
            tool_registry: ToolRegistry::new(),
        }
    }

    /// Add tool to `AgentTask` `ToolRegistry` for `ActionAgent`
    pub fn add_tool<T: ToolImplementation + 'static>(&self, tool: T) {
        self.tool_registry.register(Arc::new(tool));
    }

    /// Add a pre-wrapped tool (`Arc<dyn ToolImplementation>`) to the registry
    pub fn add_tool_arc(&self, tool: Arc<dyn ToolImplementation>) {
        self.tool_registry.register(tool);
    }

    /// Remove Tool from `AgentTask` `ToolRegistry`
    pub fn remove_tool(&self, name: &str) -> Option<Arc<dyn ToolImplementation>> {
        self.tool_registry.remove(name)
    }

    /// Get all Tools from `AgentTask` `ToolRegistry`
    pub fn get_all_tools(&self) -> Vec<Tool> {
        self.tool_registry.get_all_definitions()
    }

    /// Format tools for inclusion in system prompt
    fn format_tools_for_prompt(tools: &[Tool]) -> String {
        if tools.is_empty() {
            return String::from("\n\nNo tools are currently available.");
        }

        let mut description = String::from("\n\nAvailable tools for the ActionAgent:\n");

        for tool in tools {
            let _ = write!(
                description,
                "\n- {}: {}\n",
                tool.function.name, tool.function.description
            );
        }

        description
    }

    /// Execute the context gathering phase
    ///
    /// # Errors
    /// Returns an error if context analysis fails
    pub async fn gather_context(
        &mut self,
        cancel: &CancellationToken,
    ) -> Result<AgentResponse, CoreError> {
        // Get all available tools from the registry
        let available_tools = self.get_all_tools();

        // Format tools for the context agent's system prompt
        let tools_description = Self::format_tools_for_prompt(&available_tools);

        // Pass tools information to context agent
        let response = self
            .context_agent
            .analyze_with_tools(&self.task_description, &tools_description, cancel)
            .await?;
        self.state.context_response = Some(response.clone());
        Ok(response)
    }

    /// Execute the action phase using context from previous phase
    ///
    /// # Errors
    /// Returns an error if context was not gathered first or action execution fails
    pub async fn take_action(
        &mut self,
        cancel: &CancellationToken,
    ) -> Result<AgentResponse, CoreError> {
        if self.state.context_response.is_none() {
            return Err(CoreError::InvalidInput(
                "Cannot take action before gathering context".to_string(),
            ));
        }

        // Add all tools from the registry to the action agent
        for tool_name in self.tool_registry.tool_names() {
            if let Some(tool) = self.tool_registry.get(&tool_name) {
                // Add tool to the action agent's tool executor
                // We need to clone the Arc to share the tool
                self.action_agent
                    .agent
                    .core
                    .tool_executor
                    .add_tool_arc(tool);
            }
        }

        let context = self
            .state
            .context_response
            .as_ref()
            .map(|r| r.content.content.clone())
            .unwrap_or_default();

        let response = self
            .action_agent
            .execute_task(&self.task_description, &context, cancel)
            .await?;
        self.state.action_response = Some(response.clone());
        Ok(response)
    }

    /// Execute the verification phase
    ///
    /// # Errors
    /// Returns an error if action was not taken first or verification fails
    pub async fn verify(&mut self, cancel: &CancellationToken) -> Result<AgentResponse, CoreError> {
        if self.state.action_response.is_none() {
            return Err(CoreError::InvalidInput(
                "Cannot verify before action is taken".to_string(),
            ));
        }

        let action_result = self
            .state
            .action_response
            .as_ref()
            .map(|r| r.content.content.clone())
            .unwrap_or_default();

        let (verified, response) = self
            .verifier_agent
            .verify(&self.task_description, &action_result, cancel)
            .await?;

        self.state.verified = verified;
        self.state.verification_response = Some(response.clone());
        Ok(response)
    }

    /// Execute the full task pipeline: context -> action -> verify
    ///
    /// # Errors
    /// Returns an error if any phase of the task pipeline fails
    pub async fn execute_full(
        &mut self,
        cancel: CancellationToken,
    ) -> Result<TaskResponse, CoreError> {
        // Phase 1: Gather context
        let context_response = self.gather_context(&cancel).await?;

        // Phase 2: Take action based on context
        let action_response = self.take_action(&cancel).await?;

        // Phase 3: Verify the action results
        let verification_response = self.verify(&cancel).await?;

        Ok(TaskResponse {
            context_response,
            action_response,
            verification_response,
            success: self.state.verified,
        })
    }

    /// Reset the task to initial state
    ///
    /// # Errors
    /// Returns an error if any agent reset fails
    pub async fn reset(&mut self) -> Result<(), CoreError> {
        self.context_agent.agent.reset().await?;
        self.action_agent.agent.reset().await?;
        self.verifier_agent.agent.reset().await?;
        self.state = TaskState::default();
        self.conversation_id = Uuid::new_v4();
        Ok(())
    }
}
