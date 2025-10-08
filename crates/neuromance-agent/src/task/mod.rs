use std::sync::Arc;

use anyhow::Result;
use neuromance_tools::{ToolImplementation, ToolRegistry};
use uuid::Uuid;

use neuromance_client::LLMClient;
use neuromance_common::agents::AgentResponse;
use neuromance_common::tools::Tool;

use crate::Agent;

mod agents;
use agents::{ActionAgent, ContextAgent, VerifierAgent};

/// /// Represents a complete agent task composed of three specialized agents:
/// - ContextAgent: Gathers and analyzes context
/// - ActionAgent: Takes actions based on context
/// - VerifierAgent: Verifies actions were successful
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
    pub context_gathered: bool,
    pub action_taken: bool,
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
            context_agent: ContextAgent::new(id.clone(), client.clone()),
            action_agent: ActionAgent::new(id.clone(), client.clone()),
            verifier_agent: VerifierAgent::new(id, client),
            state: TaskState::default(),
            tool_registry: ToolRegistry::new(),
        }
    }

    /// Add tool to AgentTask ToolRegistry for ActionAgent
    pub fn add_tool<T: ToolImplementation + 'static>(&self, tool: T) {
        self.tool_registry.register(Arc::new(tool));
    }

    /// Add a pre-wrapped tool (`Arc<dyn ToolImplementation>`) to the registry
    pub fn add_tool_arc(&self, tool: Arc<dyn ToolImplementation>) {
        self.tool_registry.register(tool);
    }

    /// Remove Tool from AgentTask ToolRegistry
    pub async fn remove_tool(&mut self, name: &str) -> Result<Option<Arc<dyn ToolImplementation>>> {
        let tool = self.tool_registry.remove(name);
        Ok(tool)
    }

    /// Get all Tools from AgentTask ToolRegistry
    pub fn get_all_tools(&self) -> Vec<Tool> {
        self.tool_registry.get_all_definitions()
    }

    /// Format tools for inclusion in system prompt
    fn format_tools_for_prompt(&self, tools: &[Tool]) -> String {
        if tools.is_empty() {
            return String::from("\n\nNo tools are currently available.");
        }

        let mut description = String::from("\n\nAvailable tools for the ActionAgent:\n");

        for tool in tools {
            description.push_str(&format!(
                "\n- {}: {}\n",
                tool.function.name, tool.function.description
            ));

            // NOTE parameters seem excessive and context wasting, revisit later
            //
            // Add parameter information if available
            // description.push_str(&format!("  Parameters: {}\n",
            //     serde_json::to_string_pretty(&tool.function.parameters).unwrap_or_else(|_| "N/A".to_string())
            // ));
        }

        description
    }

    /// Execute the context gathering phase
    pub async fn gather_context(&mut self) -> Result<AgentResponse> {
        // Get all available tools from the registry
        let available_tools = self.get_all_tools();

        // Format tools for the context agent's system prompt
        let tools_description = self.format_tools_for_prompt(&available_tools);

        // Pass tools information to context agent
        let response = self
            .context_agent
            .analyze_with_tools(self.task_description.clone(), tools_description)
            .await?;
        self.state.context_gathered = true;
        self.state.context_response = Some(response.clone());
        Ok(response)
    }

    /// Execute the action phase using context from previous phase
    pub async fn take_action(&mut self) -> Result<AgentResponse> {
        if !self.state.context_gathered {
            return Err(anyhow::anyhow!(
                "Cannot take action before gathering context"
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
            .execute_task(self.task_description.clone(), context)
            .await?;
        self.state.action_taken = true;
        self.state.action_response = Some(response.clone());
        Ok(response)
    }

    /// Execute the verification phase
    pub async fn verify(&mut self) -> Result<AgentResponse> {
        if !self.state.action_taken {
            return Err(anyhow::anyhow!("Cannot verify before action is taken"));
        }

        let action_result = self
            .state
            .action_response
            .as_ref()
            .map(|r| r.content.content.clone())
            .unwrap_or_default();

        let (verified, response) = self
            .verifier_agent
            .verify(self.task_description.clone(), action_result)
            .await?;

        self.state.verified = verified;
        self.state.verification_response = Some(response.clone());
        Ok(response)
    }

    /// Execute the full task pipeline: context -> action -> verify
    pub async fn execute_full(&mut self) -> Result<TaskResponse> {
        // Phase 1: Gather context
        let context_response = self.gather_context().await?;

        // Phase 2: Take action based on context
        let action_response = self.take_action().await?;

        // Phase 3: Verify the action results
        let verification_response = self.verify().await?;

        Ok(TaskResponse {
            context_response,
            action_response,
            verification_response,
            success: self.state.verified,
        })
    }

    /// Reset the task to initial state
    pub async fn reset(&mut self) -> Result<()> {
        self.context_agent.agent.reset().await?;
        self.action_agent.agent.reset().await?;
        self.verifier_agent.agent.reset().await?;
        self.state = TaskState::default();
        self.conversation_id = Uuid::new_v4();
        Ok(())
    }
}
