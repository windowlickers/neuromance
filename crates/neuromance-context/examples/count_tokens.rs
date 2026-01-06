//! Example demonstrating token counting for strings and conversations.
//!
//! Run with: cargo run --example count_tokens
//!
//! Requires HF_TOKEN environment variable to be set.

use neuromance_common::Conversation;
use neuromance_context::{ModelConfig, TokenCounter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Get Hugging Face token from environment
    let hf_token = std::env::var("HF_TOKEN").expect("HF_TOKEN environment variable must be set");

    println!("Initializing token counter for GPT-OSS-20B...");

    // Create token counter configuration
    let config = ModelConfig::gpt_oss_20b().with_hf_token(hf_token);

    // Initialize the token counter (downloads tokenizer on first run)
    let counter = TokenCounter::new(config).await?;

    println!("Token counter initialized!\n");

    // Example 1: Count tokens in a simple string
    let simple_text = "Hello, how many tokens is this?";
    let simple_count = counter.count_tokens(simple_text)?;
    println!("Example 1: Simple string");
    println!("Text: \"{}\"", simple_text);
    println!("Token count: {}\n", simple_count);

    // Example 2: Count tokens in a longer text
    let long_text = "The quick brown fox jumps over the lazy dog. \
                     This is a longer piece of text that will have more tokens. \
                     We're testing the token counting functionality of the neuromance-context crate.";
    let long_count = counter.count_tokens(long_text)?;
    println!("Example 2: Longer text");
    println!("Text: \"{}\"", long_text);
    println!("Token count: {}\n", long_count);

    // Example 3: Count tokens in a conversation
    println!("Example 3: Conversation token counting");
    let mut conv = Conversation::new()
        .with_title("Weather Assistant")
        .with_description("A conversation about the weather");

    // Add messages to the conversation
    let system_msg = conv.system_message("You are a helpful weather assistant.");
    conv.add_message(system_msg)?;

    let user_msg = conv.user_message("What's the weather like in Tokyo today?");
    conv.add_message(user_msg)?;

    let assistant_msg = conv.assistant_message("Let me check the current weather in Tokyo for you. Tokyo is currently experiencing partly cloudy skies with a temperature of 18°C (64°F).");
    conv.add_message(assistant_msg)?;

    let user_followup = conv.user_message("Will it rain later?");
    conv.add_message(user_followup)?;

    // Count total conversation tokens
    let conv_count = counter.count_conversation_tokens(&conv)?;
    println!(
        "Conversation: \"{}\"",
        conv.title.as_deref().unwrap_or("Untitled")
    );
    println!("Number of messages: {}", conv.get_messages().len());
    println!("Total token count: {}\n", conv_count);

    // Example 4: Count individual message tokens
    println!("Example 4: Individual message token counts");
    for (i, message) in conv.get_messages().iter().enumerate() {
        let msg_count = counter.count_message_tokens(message)?;
        let preview = if message.content.len() > 50 {
            format!("{}...", &message.content[..47])
        } else {
            message.content.clone()
        };
        println!(
            "Message {}: {:?} - \"{}\" = {} tokens",
            i + 1,
            message.role,
            preview,
            msg_count
        );
    }

    Ok(())
}
