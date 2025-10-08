use anyhow::Result;
use clap::Parser;
use futures::StreamExt;
use log::info;
use uuid::Uuid;

use neuromance_client::{LLMClient, OpenAIClient};
use neuromance_common::{ChatRequest, Config, Message};

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
    #[arg(long, default_value = "gpt-oss:20b")]
    model: String,

    /// The user message to send
    #[arg(long, default_value = "Write a short poem about Rust programming.")]
    message: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    info!("Streaming Chat Demo");
    info!("===================");
    info!("Base URL: {}", args.base_url);
    info!("Model: {}", args.model);
    info!("Message: {}", args.message);
    info!("");

    // Create client configuration
    let config = Config::new("openai", &args.model)
        .with_base_url(&args.base_url)
        .with_api_key(&args.api_key);

    // Create OpenAI client
    let client = OpenAIClient::new(config.clone())?;

    // Create the chat request
    let conversation_id = Uuid::new_v4();
    let messages = vec![
        Message::system(
            conversation_id,
            "You are a helpful assistant that writes concise, creative content.",
        ),
        Message::user(conversation_id, &args.message),
    ];

    let request: ChatRequest = (config, messages).into();

    info!("Starting streaming chat...");
    info!("");

    // Get the streaming response
    let mut stream = client.chat_stream(&request).await?;

    // Print a header for the response
    println!("Assistant: ");

    // Process the stream chunk by chunk
    let mut full_content = String::new();
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;

        // Print the delta content if available
        if let Some(content) = chunk.delta_content {
            print!("{}", content);
            full_content.push_str(&content);
            // Flush stdout to show content immediately
            use std::io::Write;
            std::io::stdout().flush()?;
        }

        // Log finish reason when stream completes
        if let Some(reason) = chunk.finish_reason {
            info!("");
            info!("");
            info!("Stream finished: {:?}", reason);
        }

        // Log usage information if available
        if let Some(usage) = chunk.usage {
            info!("Usage:");
            info!("  Prompt tokens: {}", usage.prompt_tokens);
            info!("  Completion tokens: {}", usage.completion_tokens);
            info!("  Total tokens: {}", usage.total_tokens);
        }
    }

    info!("");
    info!("Total characters received: {}", full_content.len());

    Ok(())
}
