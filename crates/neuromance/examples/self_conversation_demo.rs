use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use log::info;
use uuid::Uuid;

use neuromance_client::LLMClient;
use neuromance_client::openai::client::OpenAIClient;
use neuromance_common::chat::Message;
use neuromance_common::client::{ChatRequest, Config};

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

    /// The initial user message
    #[arg(long, default_value = "Hello")]
    message: String,

    /// Number of conversation turns
    #[arg(long, default_value = "20")]
    turns: u32,
}

#[allow(clippy::unwrap_used)]
fn strip_think_tags(content: &str) -> String {
    // Remove <think>...</think> tags and their content
    let re = regex::Regex::new(r"(?s)<think>.*?</think>").unwrap();
    let result = re.replace_all(content, "");
    // Also remove any orphaned closing tags
    let result = result.replace("</think>", "");
    result.trim().to_string()
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    info!("Self Conversation Demo");
    info!("======================");
    info!("Base URL: {}", args.base_url);
    info!("Model: {}", args.model);
    info!("Initial message: {}", args.message);
    info!("Turns: {}", args.turns);

    // Create client configuration
    let config = Config::new("openai", &args.model)
        .with_base_url(&args.base_url)
        .with_api_key(&args.api_key);

    // Create OpenAI client
    let client = OpenAIClient::new(config.clone())?;

    // Create the initial conversation
    let conversation_id = Uuid::new_v4();
    let mut messages: Vec<Message> = Vec::new();

    // System message - the agent doesn't know it's talking to itself
    messages.push(Message::system(
        conversation_id,
        "You are having a friendly conversation. Respond naturally and ask questions to keep the conversation going. Do not show your thinking process - only provide your final response.",
    ));

    // Initial user message
    messages.push(Message::user(conversation_id, &args.message));

    let start_time = Instant::now();

    // Run the conversation for the specified number of turns
    for turn in 1..=args.turns {
        info!(
            "{} {} {}",
            " --- Turn".red(),
            turn.to_string().red(),
            " ---".red()
        );

        // Get assistant response
        // Disable thinking to avoid conflicts with response prefill
        let request = ChatRequest::new(messages.clone()).with_thinking(false);

        let response = client.chat(&request).await?;

        // Extract the assistant's message and strip thinking tags
        let assistant_content = &response.message.content;
        let cleaned_content = strip_think_tags(assistant_content);

        info!("{}: {}", "Assistant".blue(), cleaned_content);

        // Add cleaned assistant response to messages (without think tags)
        messages.push(Message::assistant(conversation_id, &cleaned_content));

        // Get user response (feeding assistant's message as the latest context)
        let user_request = ChatRequest::new(messages.clone()).with_thinking(false);

        let user_response = client.chat(&user_request).await?;

        let user_content = &user_response.message.content;
        let cleaned_user_content = strip_think_tags(user_content);
        info!("{}: {}", "User".yellow(), cleaned_user_content);

        // Add cleaned user response to conversation (replacing the last assistant message)
        messages.pop(); // Remove the assistant message we just added
        messages.push(Message::user(conversation_id, &cleaned_user_content));
    }

    let total_duration = start_time.elapsed();

    info!("======================");
    info!("Conversation completed!");
    info!("Total turns: {}", args.turns);
    info!("Total time: {:.2}s", total_duration.as_secs_f64());
    info!("Total messages: {}", messages.len());

    Ok(())
}
