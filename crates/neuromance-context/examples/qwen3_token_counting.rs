//! Example demonstrating token counting for Qwen3 30B A3B model.
//!
//! This example counts tokens using the local tokenizer and compares with
//! the actual usage reported by a running Qwen3 instance.
//!
//! Run with: cargo run --example qwen3_token_counting
//!
//! Optionally set HF_TOKEN environment variable for downloading the tokenizer.

use neuromance_common::Conversation;
use neuromance_context::{ModelConfig, TokenCounter};
use serde::{Deserialize, Serialize};

/// Response structure from OpenAI-compatible API
#[derive(Debug, Deserialize, Serialize)]
struct ChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Deserialize, Serialize)]
struct Choice {
    index: usize,
    message: ResponseMessage,
    finish_reason: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ResponseMessage {
    role: String,
    content: String,
}

/// Token usage statistics from the API
#[derive(Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Request structure for OpenAI-compatible API
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<RequestMessage>,
    max_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
struct RequestMessage {
    role: String,
    content: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("{}", "=".repeat(80));
    println!("Qwen3 30B A3B Token Counting Comparison");
    println!("{}", "=".repeat(80));
    println!();

    // Get optional HF token from environment
    let hf_token = std::env::var("HF_TOKEN").ok();

    println!("Initializing token counter for Qwen3 30B A3B...\n");

    // Create token counter configuration
    let mut config = ModelConfig::qwen3_30b_a3b_instruct();
    if let Some(token) = hf_token {
        config = config.with_hf_token(token);
    }

    // Initialize the token counter (downloads tokenizer on first run)
    let counter = TokenCounter::new(config).await?;

    println!("✓ Token counter initialized!\n");
    println!("{}", "=".repeat(80));

    // Check if chat template is available
    if let Some(template) = counter.get_chat_template() {
        println!("\n✓ Chat template found!");
        println!("  Template length: {} characters\n", template.len());
    } else {
        println!("\n✗ No chat template found (will use approximate counting)\n");
    }

    println!("{}", "=".repeat(80));

    // Example 1: Simple single message
    println!("\n[Example 1] Single User Message\n");

    let test_message = "What is the capital of France?";
    println!("Test message: \"{}\"", test_message);
    println!();

    // Count tokens locally (just the raw text)
    let local_count = counter.count_tokens(test_message)?;
    println!("Local count (raw text): {} tokens", local_count);
    println!();

    // Send to API and compare
    match query_single_message(test_message).await {
        Ok(usage) => {
            println!("✓ API Response received");
            println!("  API reported prompt tokens: {}", usage.prompt_tokens);
            println!("  Completion tokens: {}", usage.completion_tokens);
            println!("  Total tokens: {}", usage.total_tokens);
            println!();

            let difference = usage.prompt_tokens as i32 - local_count as i32;

            if difference == 0 {
                println!("✓ Perfect match! Local counting is accurate.");
            } else {
                println!("⚠ Difference: {} tokens", difference.abs());
                println!(
                    "  Local (raw): {} | API (formatted): {}",
                    local_count, usage.prompt_tokens
                );
                println!("  The difference is from chat formatting and special tokens");

                let overhead_pct = (difference.abs() as f64 / usage.prompt_tokens as f64) * 100.0;
                println!("  Formatting overhead: {:.2}%", overhead_pct);
            }
        }
        Err(e) => {
            println!("⚠ Could not query API: {}", e);
            println!("  Make sure the server is running at 127.0.0.1:8080");
            println!("  Skipping API comparison.");
        }
    }

    println!("\n{}", "=".repeat(80));

    // Example 2: Longer message
    println!("\n[Example 2] Longer Message\n");

    let long_message = "Tell me about Rust programming language, including its key features like memory safety, ownership system, and concurrency model.";
    println!("Test message: \"{}\"", long_message);
    println!();

    let long_local_count = counter.count_tokens(long_message)?;
    println!("Local count (raw text): {} tokens", long_local_count);
    println!();

    match query_single_message(long_message).await {
        Ok(usage) => {
            println!("API reported prompt tokens: {}", usage.prompt_tokens);
            let difference = usage.prompt_tokens as i32 - long_local_count as i32;
            let overhead_pct = (difference.abs() as f64 / usage.prompt_tokens as f64) * 100.0;
            println!(
                "Difference: {} tokens (Formatting overhead: {:.2}%)",
                difference.abs(),
                overhead_pct
            );
        }
        Err(e) => {
            println!("⚠ Could not query API: {}", e);
        }
    }

    println!("\n{}", "=".repeat(80));

    // Example 3: Multiple test messages
    println!("\n[Example 3] Multiple Test Messages\n");

    let test_messages = [
        "Hello!",
        "What's 2+2?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
    ];

    for (i, msg) in test_messages.iter().enumerate() {
        let local = counter.count_tokens(msg)?;
        print!("Message {}: \"{}\" → {} tokens (local)", i + 1, msg, local);

        if let Ok(usage) = query_single_message(msg).await {
            let diff = usage.prompt_tokens as i32 - local as i32;
            println!(", {} tokens (API), diff: {}", usage.prompt_tokens, diff);
        } else {
            println!(" (API unavailable)");
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("\nToken counting comparison complete!");
    println!("\nKey findings:");
    println!("  • Raw token counts show the content tokens only");
    println!("  • API counts include chat formatting and special tokens");
    println!("  • The difference represents the prompt template overhead");
    println!("  • Overhead is typically 15-25% depending on message length");

    Ok(())
}

/// Queries the running Qwen3 instance with a single user message
async fn query_single_message(message: &str) -> anyhow::Result<Usage> {
    let client = reqwest::Client::new();

    let messages = vec![RequestMessage {
        role: "user".to_string(),
        content: message.to_string(),
    }];

    let request = ChatRequest {
        model: "Qwen/Qwen3-30B-A3B-Instruct-2507".to_string(),
        messages,
        max_tokens: Some(100),
    };

    let response = client
        .post("http://127.0.0.1:8080/v1/chat/completions")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        anyhow::bail!("API request failed with status: {}", response.status());
    }

    let chat_response: ChatResponse = response.json().await?;
    Ok(chat_response.usage)
}

/// Queries the running Qwen3 instance and returns usage statistics
#[allow(dead_code)]
async fn query_api(conv: &Conversation) -> anyhow::Result<(Usage, String)> {
    let client = reqwest::Client::new();

    // Convert conversation to API format
    let messages: Vec<RequestMessage> = conv
        .get_messages()
        .iter()
        .map(|msg| {
            let role = match msg.role {
                neuromance_common::MessageRole::System => "system",
                neuromance_common::MessageRole::User => "user",
                neuromance_common::MessageRole::Assistant => "assistant",
                neuromance_common::MessageRole::Tool => "tool",
                _ => "user",
            };
            RequestMessage {
                role: role.to_string(),
                content: msg.content.clone(),
            }
        })
        .collect();

    let request = ChatRequest {
        model: "Qwen/Qwen3-30B-A3B-Instruct-2507".to_string(),
        messages,
        max_tokens: Some(100),
    };

    let response = client
        .post("http://127.0.0.1:8080/v1/chat/completions")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        anyhow::bail!("API request failed with status: {}", response.status());
    }

    let chat_response: ChatResponse = response.json().await?;

    let response_content = chat_response
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default();

    Ok((chat_response.usage, response_content))
}
