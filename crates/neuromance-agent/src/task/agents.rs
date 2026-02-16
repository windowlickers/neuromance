use anyhow::Result;
use log::debug;

use neuromance_client::LLMClient;
use neuromance_common::agents::AgentResponse;
use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::ChatRequest;
use neuromance_tools::{BooleanTool, ThinkTool, ToolImplementation, create_todo_tools};

use crate::{Agent, BaseAgent};

/// Context analysis agent that determines what's needed to complete a task
pub struct ContextAgent<C: LLMClient> {
    pub agent: BaseAgent<C>,
}

impl<C: LLMClient + Send + Sync> ContextAgent<C> {
    pub fn new(id: &str, client: C) -> Self {
        Self {
            agent: BaseAgent::builder(format!("{id}-context"), client)
                .system_prompt(
                    "You are the assistant. Your job is to analyze what tools, \
                    information, and approach would be needed to complete a task. Be concise, \
                    specific, and consider all requirements. You can use the think tool to think extra hard.",
                )
                .add_tool(ThinkTool)
                .max_turns(5)
                .auto_approve_tools(true)
                .build(),
        }
    }

    /// Add tool to `ContextAgent`
    pub fn add_tool<T: ToolImplementation + 'static>(&mut self, tool: T) {
        self.agent.core.tool_executor.add_tool(tool);
    }

    pub async fn analyze(&mut self, task_description: String) -> Result<AgentResponse> {
        let messages = vec![
            Message::new(
                self.agent.conversation_id,
                MessageRole::System,
                self.agent.system_prompt.clone().unwrap_or_default(),
            ),
            Message::new(
                self.agent.conversation_id,
                MessageRole::User,
                format!(
                    "Analyze what would be required to complete this task: {task_description}\n\n\
                    Consider: What tools are needed? What information must be gathered? \
                    What approach should be taken?",
                ),
            ),
        ];

        self.agent.execute(Some(messages)).await
    }

    pub async fn analyze_with_tools(
        &mut self,
        task_description: String,
        tools_description: String,
    ) -> Result<AgentResponse> {
        // Update system prompt to include available tools
        let enhanced_system_prompt = format!(
            "{}\n\n{}",
            self.agent.system_prompt.clone().unwrap_or_default(),
            tools_description
        );

        let messages = vec![
            Message::new(
                self.agent.conversation_id,
                MessageRole::System,
                enhanced_system_prompt,
            ),
            Message::new(
                self.agent.conversation_id,
                MessageRole::User,
                format!(
                    "Analyze what would be required to complete this task: {task_description}\n\n\
                    Based on the available tools listed in the system message, create a step by step plan:\n\
                    - Which tools need to be used?\n\
                    - What context is critical?\n\
                    - What steps should be taken?\n\n\
                    Please provide your concise analysis specifying what tools and instructions \
                    are critical to accomplishing the task.",
                ),
            ),
        ];

        self.agent.execute(Some(messages)).await
    }
}

/// Action agent that executes tasks based on context
pub struct ActionAgent<C: LLMClient> {
    pub agent: BaseAgent<C>,
}

impl<C: LLMClient + Send + Sync> ActionAgent<C> {
    pub fn new(id: &str, client: C) -> Self {
        let (todo_read, todo_write) = create_todo_tools();
        Self {
            agent: BaseAgent::builder(format!("{id}-action"), client)
                .system_prompt(
                    "You are an the assistant. Based on task analysis, you execute the \
                    required actions to complete tasks. Be precise and follow the recommended \
                    approach from the context analysis.",
                )
                .max_turns(5)
                .add_tool(todo_read)
                .add_tool(todo_write)
                .auto_approve_tools(true)
                .build(),
        }
    }

    pub async fn execute_task(
        &mut self,
        task_description: String,
        context: String,
    ) -> Result<AgentResponse> {
        let messages = vec![
            Message::new(
                self.agent.conversation_id,
                MessageRole::System,
                self.agent.system_prompt.clone().unwrap_or_default(),
            ),
            Message::new(
                self.agent.conversation_id,
                MessageRole::User,
                format!(
                    r"Task: {context}\n\n\Task Analysis:\n{task_description}\n\n
                    Now execute the task based on the context analysis.",
                ),
            ),
        ];

        self.agent.execute(Some(messages)).await
    }
}

/// Verifier agent that validates task completion
pub struct VerifierAgent<C: LLMClient> {
    pub agent: BaseAgent<C>,
}

impl<C: LLMClient + Send + Sync> VerifierAgent<C> {
    pub fn new(id: &str, client: C) -> Self {
        Self {
            agent: BaseAgent::builder(format!("{id}-verifier"), client)
                .system_prompt(
                    "You are a verification agent. You check if tasks were completed successfully \
                    and correctly. Use the return_bool tool to indicate success (true) or failure (false) \
                    with a clear explanation."
                )
                .add_tool(BooleanTool)
                .max_turns(2)
                .auto_approve_tools(true)
                .build(),
        }
    }

    pub async fn verify(
        &self,
        task_description: String,
        action_result: String,
    ) -> Result<(bool, AgentResponse)> {
        let messages = vec![
            Message::new(
                self.agent.conversation_id,
                MessageRole::System,
                self.agent.system_prompt.clone().unwrap_or_default(),
            ),
            Message::new(
                self.agent.conversation_id,
                MessageRole::User,
                format!(
                    "Original Task: {task_description}\n\nResult:\n{action_result}\n\n\
                    Did the agent successfully complete the task? Use the return_bool tool \
                    to provide your verification result.",
                ),
            ),
        ];

        // Get the boolean tool definition to set up tool choice
        let bool_tool = BooleanTool;
        let tool_def = bool_tool.get_definition();

        // Create chat request with required tool choice for the boolean tool
        let request = ChatRequest::from((self.agent.core.client.config(), messages))
            .with_tools(vec![tool_def])
            .with_tool_choice(neuromance_common::client::ToolChoice::Function {
                name: "return_bool".to_string(),
            });

        debug!(
            "Chat request:\n {}",
            serde_json::to_string_pretty(&request)?
        );

        let response = self.agent.core.client.chat(&request).await?;

        debug!("Received response from LLM");
        debug!(
            "Assistant Response:\n {}",
            serde_json::to_string_pretty(&response)?
        );

        // Check if we got the boolean tool call response
        if let Some(tool_call) = response.message.tool_calls.first()
            && tool_call.function.name == "return_bool"
        {
            // Parse the boolean result from the tool call arguments
            let args_str = tool_call.function.arguments_json();
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(args_str)
                && let Some(result) = args.get("result").and_then(serde_json::Value::as_bool)
            {
                let agent_response = AgentResponse {
                    content: response.message.clone(),
                    reasoning: args
                        .get("reason")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    tool_responses: vec![],
                };

                return Ok((result, agent_response));
            }
        }

        Err(anyhow::anyhow!(
            "Verifier agent did not return expected boolean tool call"
        ))
    }
}
