//! Anthropic Claude API Demo
//!
//! This example demonstrates using the Anthropic client to interact with Claude models.
//! It shows both non-streaming and streaming chat completions.
//!
//! # Usage
//!
//! ```bash
//! # Set your API key
//! export ANTHROPIC_API_KEY="sk-ant-..."
//!
//! # Run with defaults (streaming)
//! cargo run --example anthropic_demo
//!
//! # Run with custom message
//! cargo run --example anthropic_demo -- --message "Explain Rust's ownership in 3 sentences"
//!
//! # Run non-streaming
//! cargo run --example anthropic_demo -- --no-stream
//!
//! # Use a specific model
//! cargo run --example anthropic_demo -- --model claude-haiku-4-5-20251001
//! ```

use std::io::Write;

use anyhow::Result;
use clap::Parser;
use futures::StreamExt;
use log::info;
use uuid::Uuid;

use neuromance_client::{AnthropicClient, LLMClient};
use neuromance_common::{ChatRequest, Config, Message};

#[derive(Parser, Debug)]
#[command(author, version, about = "Anthropic Claude API Demo")]
struct Args {
    /// API key for authentication (or set `ANTHROPIC_API_KEY` env var)
    #[arg(long, env = "ANTHROPIC_API_KEY")]
    api_key: String,

    /// Model to use for chat completion
    #[arg(long, default_value = "claude-sonnet-4-5-20250929")]
    model: String,

    /// Maximum tokens to generate
    #[arg(long, default_value = "1024")]
    max_tokens: u32,

    /// The user message to send
    #[arg(long, default_value = "Write a haiku about Rust programming.")]
    message: String,

    /// Disable streaming (use non-streaming API)
    #[arg(long)]
    no_stream: bool,

    /// Temperature for sampling (0.0-1.0)
    #[arg(long)]
    temperature: Option<f32>,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("Anthropic Claude Demo");
    println!("=====================");
    println!("Model: {}", args.model);
    println!("Max tokens: {}", args.max_tokens);
    println!("Streaming: {}", !args.no_stream);
    println!();

    // Create client configuration
    let mut config = Config::new("anthropic", &args.model)
        .with_api_key(&args.api_key)
        .with_max_tokens(args.max_tokens);

    if let Some(temp) = args.temperature {
        config = config.with_temperature(temp);
    }

    // Create Anthropic client
    let client = AnthropicClient::new(config)?;

    // Create the chat request
    let conversation_id = Uuid::new_v4();
    let messages = vec![
        Message::system(
            conversation_id,
            "You are a helpful assistant that writes concise, creative content.",
        ),
        Message::user(conversation_id, &args.message),
    ];

    let request = ChatRequest::new(messages)
        .with_model(&args.model)
        .with_max_tokens(args.max_tokens);

    if args.no_stream {
        // Non-streaming request
        println!("Sending non-streaming request...");
        println!();

        let response = client.chat(&request).await?;

        println!("Assistant: {}", response.message.content);
        println!();

        if let Some(reason) = response.finish_reason {
            info!("Finish reason: {reason:?}");
        }

        if let Some(usage) = response.usage {
            println!("Usage:");
            println!("  Input tokens: {}", usage.prompt_tokens);
            println!("  Output tokens: {}", usage.completion_tokens);
            println!("  Total tokens: {}", usage.total_tokens);
        }
    } else {
        // Streaming request
        println!("Starting streaming chat...");
        println!();
        print!("Assistant: ");
        std::io::stdout().flush()?;

        let mut stream = client.chat_stream(&request).await?;

        let mut full_content = String::new();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;

            // Print the delta content if available
            if let Some(content) = chunk.delta_content {
                print!("{content}");
                full_content.push_str(&content);
                std::io::stdout().flush()?;
            }

            // Print reasoning content if available (for thinking models)
            if let Some(reasoning) = chunk.delta_reasoning_content {
                // Thinking content could be shown differently
                info!("[Thinking: {reasoning}]");
            }

            // Log finish reason when stream completes
            if let Some(reason) = chunk.finish_reason {
                println!();
                println!();
                info!("Stream finished: {reason:?}");
            }

            // Log usage information if available
            if let Some(usage) = chunk.usage {
                println!("Usage:");
                println!("  Input tokens: {}", usage.prompt_tokens);
                println!("  Output tokens: {}", usage.completion_tokens);
                println!("  Total tokens: {}", usage.total_tokens);
            }
        }

        println!();
        info!("Total characters received: {}", full_content.len());
    }

    Ok(())
}
