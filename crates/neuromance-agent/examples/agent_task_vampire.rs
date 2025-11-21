use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use log::info;

use neuromance_agent::AgentTask;
use neuromance_client::LLMClient;
use neuromance_client::openai::client::OpenAIClient;
use neuromance_common::client::Config;
use neuromance_tools::generic::CurrentTimeTool;
use neuromance_tools::mcp::{McpConfig, McpManager};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Base URL for the API endpoint
    #[arg(long, default_value = "http://localhost:8080/v1")]
    base_url: String,

    /// API key for authentication
    #[arg(long, default_value = "dummy")]
    api_key: String,

    /// Model to use for chat completion
    #[arg(long, default_value = "ggml-org/gpt-oss-120b-GGUF")]
    model: String,

    /// Path to MCP configuration file (TOML, JSON, or YAML)
    #[arg(long)]
    mcp_config: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    info!("Agent Task Demo: Vampire Time Teller");
    info!("=====================================");
    info!("Base URL: {}", args.base_url);
    info!("Model: {}", args.model);

    // Create client configuration
    let config = Config::new("openai", &args.model)
        .with_base_url(&args.base_url)
        .with_api_key(&args.api_key);

    info!("Config model: {}", config.model);

    let start_time = Instant::now();

    // Create the AgentTask with task description
    let client = OpenAIClient::new(config)?;
    info!("Client config model: {}", client.config().model);
    let mut task = AgentTask::new(
        "vampire-time-task",
        "Tell me the current time like a vampire from Transylvania would",
        client,
    );

    // Initialize MCP if config provided
    if let Some(mcp_config_path) = args.mcp_config {
        info!(
            "Loading MCP configuration from {}",
            mcp_config_path.display()
        );

        // Load the MCP configuration
        let mcp_config = McpConfig::from_file(&mcp_config_path)?;

        // Create and initialize the MCP manager (connects automatically)
        let mcp_manager = McpManager::new(mcp_config).await?;

        // Get all tools from connected servers
        let tools = mcp_manager.get_all_tools().await?;

        info!("Connected to MCP servers and loaded {} tools", tools.len());

        // Register each MCP tool with the core
        for tool in &tools {
            task.add_tool_arc(Arc::clone(tool));
        }

        println!(
            "{}",
            format!("✓ Loaded {} MCP tools", tools.len()).bright_green()
        );
    }

    // Add the time tool to the AgentTask ToolRegistry
    task.add_tool(CurrentTimeTool);

    // Set a different model for the verifier agent
    task.verifier_agent.agent.core.client = task
        .verifier_agent
        .agent
        .core
        .client
        .with_model("gpt-oss:120b".to_string());

    info!("\n=== Phase 1: Context Agent ===");
    let context_response = task.gather_context().await?;
    info!("Context Analysis:\n{}", context_response.content.content);

    info!("\n=== Phase 2: Action Agent ===");
    let action_response = task.take_action().await?;
    info!("Vampire says:\n{}", action_response.content.content);

    info!("\n=== Phase 3: Verifier Agent ===");
    let verify_response = task.verify().await?;

    info!("Task Result:\n\n {}", action_response.content.content);

    info!(
        "Verification Result: {}",
        if task.state.verified {
            "SUCCESS ✓"
        } else {
            "FAILED ✗"
        }
    );
    if let Some(reasoning) = &verify_response.reasoning {
        info!("Reasoning: {reasoning}");
    }

    let total_duration = start_time.elapsed();
    info!("\n=====================================");
    info!("Total execution time: {:.2}s", total_duration.as_secs_f64());
    info!("Task verified: {}", task.state.verified);

    Ok(())
}
