//! Demo for testing reasoning models with `reasoning_content` and `reasoning_effort`.
//!
//! This example demonstrates streaming chat completions with `OpenAI` reasoning models
//! (o1, o3, etc.) that support the `reasoning_content` field for chain-of-thought output.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example reasoning_demo
//! ```

use std::io::Write;

use anyhow::Result;
use clap::Parser;
use futures::StreamExt;
use uuid::Uuid;

use neuromance_client::LLMClient;
use neuromance_client::OpenAIClient;
use neuromance_common::{ChatRequest, Config, Message, ReasoningLevel};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Base URL for the API endpoint
    #[arg(long, default_value = "https://api.openai.com/v1")]
    base_url: String,

    /// API key for authentication (defaults to `OPENAI_API_KEY` env var)
    #[arg(long, env = "OPENAI_API_KEY")]
    api_key: String,

    /// Model to use for chat completion
    #[arg(long, default_value = "gpt-5-mini-2025-08-07")]
    model: String,

    /// Reasoning effort level (none, minimal, low, medium, high, xhigh)
    #[arg(long, default_value = "medium")]
    reasoning_effort: String,

    /// Maximum completion tokens (includes reasoning tokens)
    #[arg(long, default_value = "4096")]
    max_completion_tokens: u32,

    /// The user message to send
    #[arg(
        long,
        default_value = "What is the sum of the first 10 prime numbers? Show your reasoning."
    )]
    message: String,
}

fn parse_reasoning_level(s: &str) -> ReasoningLevel {
    match s.to_lowercase().as_str() {
        "default" | "none" => ReasoningLevel::Default,
        "minimal" => ReasoningLevel::Minimal,
        "low" => ReasoningLevel::Low,
        "medium" => ReasoningLevel::Medium,
        "high" => ReasoningLevel::High,
        "max" | "maximum" | "xhigh" => ReasoningLevel::Maximum,
        _ => ReasoningLevel::Default,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("Reasoning Model Demo");
    println!("====================");
    println!("Base URL: {}", args.base_url);
    println!("Model: {}", args.model);
    println!("Reasoning Effort: {}", args.reasoning_effort);
    println!("Max Completion Tokens: {}", args.max_completion_tokens);
    println!("Message: {}", args.message);
    println!();

    // Create client configuration
    let config = Config::new("openai", &args.model)
        .with_base_url(&args.base_url)
        .with_api_key(&args.api_key);

    // Create OpenAI client
    let client = OpenAIClient::new(config.clone())?;

    // Create the chat request with reasoning parameters
    let conversation_id = Uuid::new_v4();
    let messages = vec![Message::user(conversation_id, &args.message)];

    let reasoning_level = parse_reasoning_level(&args.reasoning_effort);

    let mut request = ChatRequest::new(messages)
        .with_model(&args.model)
        .with_max_completion_tokens(args.max_completion_tokens)
        .with_reasoning_level(reasoning_level);

    // Enable streaming
    request.stream = true;

    println!("Starting streaming chat with reasoning model...\n");

    // Get the streaming response
    let mut stream = client.chat_stream(&request).await?;

    // Track accumulated content
    let mut full_content = String::new();
    let mut full_reasoning = String::new();
    let mut in_reasoning = false;

    // Process the stream chunk by chunk
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;

        // Handle reasoning content (chain of thought)
        if let Some(ref reasoning) = chunk.delta_reasoning_content {
            if !in_reasoning {
                println!("\n--- Reasoning ---");
                in_reasoning = true;
            }
            print!("{reasoning}");
            full_reasoning.push_str(reasoning);
            std::io::stdout().flush()?;
        }

        // Handle regular content
        if let Some(ref content) = chunk.delta_content {
            if in_reasoning {
                println!("\n--- Response ---");
                in_reasoning = false;
            }
            print!("{content}");
            full_content.push_str(content);
            std::io::stdout().flush()?;
        }

        // Log finish reason when stream completes
        if let Some(ref reason) = chunk.finish_reason {
            println!("\n");
            println!("Stream finished: {reason:?}");
        }

        // Log usage information if available
        if let Some(ref usage) = chunk.usage {
            println!("Usage:");
            println!("  Prompt tokens: {}", usage.prompt_tokens);
            println!("  Completion tokens: {}", usage.completion_tokens);
            println!("  Total tokens: {}", usage.total_tokens);
        }
    }

    println!();
    println!("=== Summary ===");
    println!(
        "Reasoning characters: {}",
        if full_reasoning.is_empty() {
            "none (model may not support reasoning_content)".to_string()
        } else {
            full_reasoning.len().to_string()
        }
    );
    println!("Response characters: {}", full_content.len());

    Ok(())
}
