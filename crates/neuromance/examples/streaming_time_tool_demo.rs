use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use log::info;
use uuid::Uuid;

use neuromance::Core;
use neuromance_client::openai::client::OpenAIClient;
use neuromance_common::chat::{Message, MessageRole};
use neuromance_common::client::Config;
use neuromance_tools::ToolImplementation;
use neuromance_tools::generic::CurrentTimeTool;

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

    /// The user message to send
    #[arg(long, default_value = "What time is it?")]
    message: String,

    /// Enable auto-approval of all tools
    #[arg(long, default_value = "true")]
    auto_approve: bool,

    /// Maximum number of conversation turns
    #[arg(long, default_value = "10")]
    max_turns: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    // Parse command line arguments
    let args = Args::parse();

    info!("Streaming Time Tool Demo");
    info!("========================");
    info!("Base URL: {}", args.base_url);
    info!("Model: {}", args.model);
    info!("Message: {}", args.message);
    info!("Auto-approve: {}", args.auto_approve);
    info!("Max turns: {}", args.max_turns);
    info!("");

    // Create client configuration
    let config = Config::new("openai", &args.model)
        .with_base_url(&args.base_url)
        .with_api_key(&args.api_key);

    // Create OpenAI client
    let client = OpenAIClient::new(config.clone())?;

    // Create Core instance with streaming enabled
    let mut core = Core::new(client);
    core.auto_approve_tools = args.auto_approve;
    core.max_turns = Some(args.max_turns);

    // Enable streaming with a callback that prints chunks as they arrive
    core = core.with_streaming(|chunk| {
        print!("{}", chunk);
        std::io::stdout().flush().unwrap();
    });

    // Register the time tool using the new tools crate
    let time_tool: Arc<dyn ToolImplementation> = Arc::new(CurrentTimeTool);
    info!(
        "Registering tool: {}",
        time_tool.get_definition().function.name
    );

    core.tool_executor.add_tool_arc(time_tool);

    // Create the initial conversation
    let conversation_id = Uuid::new_v4();
    let mut messages: Vec<Message> = Vec::new();

    // Push system message so the assistant is a pirate
    messages.push(Message::system(
        conversation_id,
        "You always speak like a pirate.",
    ));

    // Push our question "What time is it?"
    messages.push(Message::user(conversation_id, &args.message));

    // Start timing the request
    let start_time = Instant::now();

    // Use the core's chat_with_tool_loop for automatic tool handling
    info!("Starting streaming conversation with automatic tool execution...");
    info!("");
    println!("Assistant: ");

    let original_message_count = messages.len();
    let final_messages = core.chat_with_tool_loop(messages.clone()).await?;

    // Calculate total time
    let total_duration = start_time.elapsed();

    println!("\n");
    info!("Conversation completed!");

    // Add new messages to original vector
    messages.extend_from_slice(&final_messages[original_message_count..]);

    // Get the final assistant message
    if let Some(final_msg) = messages
        .iter()
        .rev()
        .find(|m| m.role == MessageRole::Assistant)
    {
        info!("Full accumulated response: {}", final_msg.content);
    }

    // Print total time taken
    info!("Total request time: {:.2}ms", total_duration.as_millis());

    Ok(())
}
