//! Example demonstrating chat template formatting and accurate token counting.
//!
//! Run with: cargo run --example chat_template
//!
//! Requires HF_TOKEN environment variable to be set.

use neuromance_common::{Conversation, ToolCall};
use neuromance_context::{ModelConfig, TokenCounter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Get Hugging Face token from environment
    let hf_token = std::env::var("HF_TOKEN").expect("HF_TOKEN environment variable must be set");

    println!("Initializing token counter for GPT-OSS-20B...\n");

    // Create token counter configuration
    let config = ModelConfig::gpt_oss_20b().with_hf_token(hf_token);

    // Initialize the token counter
    let counter = TokenCounter::new(config).await?;

    println!("Token counter initialized!\n");
    println!("{}", "=".repeat(80));

    // Example 1: Check if tokenizer has a chat template
    println!("\n[Example 1] Chat Template Detection\n");

    if let Some(template) = counter.get_chat_template() {
        println!("✓ Chat template found!");
        println!("\nTemplate preview (first 200 chars):");
        let preview = if template.len() > 200 {
            format!("{}...", &template[..200])
        } else {
            template.to_string()
        };
        println!("{}", preview);
    } else {
        println!("✗ No chat template found in tokenizer");
        println!("  (Will use approximate token counting)");
    }

    println!("\n{}", "=".repeat(80));

    // Example 2: Compare token counts with and without template
    println!("\n[Example 2] Token Count Comparison\n");

    let mut conv = Conversation::new()
        .with_title("Weather Assistant")
        .with_description("A simple weather conversation");

    // Add system message
    let system_msg = conv.system_message("You are a helpful weather assistant.");
    conv.add_message(system_msg)?;

    // Add user message
    let user_msg = conv.user_message("What's the weather like in Tokyo today?");
    conv.add_message(user_msg)?;

    // Add assistant response
    let assistant_msg = conv.assistant_message(
        "Let me check the current weather in Tokyo for you. Tokyo is currently experiencing partly cloudy skies with a temperature of 18°C (64°F)."
    );
    conv.add_message(assistant_msg)?;

    println!("Conversation:");
    for (i, msg) in conv.get_messages().iter().enumerate() {
        println!("  Message {}: {:?} - {}", i + 1, msg.role, msg.content);
    }
    println!();

    // Count tokens without template (approximate)
    let count_without_template = counter.count_conversation_tokens(&conv)?;
    println!("Tokens (approximate method): {}", count_without_template);

    // Count tokens with template (accurate)
    match counter.count_conversation_tokens_with_template(&conv) {
        Ok(count_with_template) => {
            println!("Tokens (chat template method): {}", count_with_template);
            let difference = count_with_template as i32 - count_without_template as i32;
            println!("Difference: {} tokens", difference);

            if difference != 0 {
                println!(
                    "\n⚠ The approximate method differs by {} tokens!",
                    difference.abs()
                );
                println!("  Use count_conversation_tokens_with_template() for accuracy.");
            } else {
                println!("\n✓ Both methods agree!");
            }
        }
        Err(e) => {
            println!("Chat template method not available: {}", e);
            println!("(This is normal if the model doesn't have a chat template)");
        }
    }

    println!("\n{}", "=".repeat(80));

    // Example 3: View formatted conversation
    println!("\n[Example 3] Formatted Conversation Output\n");

    match counter.format_conversation_with_template(&conv) {
        Ok(formatted) => {
            println!("Formatted conversation (as the model sees it):\n");
            println!("{}", "─".repeat(80));
            println!("{}", formatted);
            println!("{}", "─".repeat(80));
            println!("\nThis is the exact text that gets tokenized and sent to the model.");
        }
        Err(e) => {
            println!("Could not format conversation: {}", e);
        }
    }

    println!("\n{}", "=".repeat(80));

    // Example 4: Conversation with tool calls
    println!("\n[Example 4] Token Counting with Tool Calls\n");

    let mut tool_conv = Conversation::new().with_title("Weather with Tools");

    // System message
    let system =
        tool_conv.system_message("You are a weather assistant with access to weather APIs.");
    tool_conv.add_message(system)?;

    // User message
    let user = tool_conv.user_message("What's the weather in New York?");
    tool_conv.add_message(user)?;

    // Assistant message with tool call
    let tool_call = ToolCall::new("get_weather", r#"{"location": "New York, NY"}"#);
    let assistant = tool_conv
        .assistant_message("Let me check the weather for you.")
        .with_tool_calls(vec![tool_call.clone()])?;
    tool_conv.add_message(assistant)?;

    // Tool response
    let tool_response = tool_conv.tool_message(
        r#"{"temperature": 72, "condition": "sunny", "humidity": 45}"#,
        tool_call.id.clone(),
        "get_weather".to_string(),
    )?;
    tool_conv.add_message(tool_response)?;

    // Final assistant response
    let final_response = tool_conv.assistant_message(
        "The weather in New York is currently sunny with a temperature of 72°F and 45% humidity.",
    );
    tool_conv.add_message(final_response)?;

    println!(
        "Conversation with {} messages (including tool calls)",
        tool_conv.get_messages().len()
    );

    let approx_tokens = counter.count_conversation_tokens(&tool_conv)?;
    println!("Approximate token count: {}", approx_tokens);

    match counter.count_conversation_tokens_with_template(&tool_conv) {
        Ok(accurate_tokens) => {
            println!("Accurate token count (with template): {}", accurate_tokens);

            if let Ok(formatted) = counter.format_conversation_with_template(&tool_conv) {
                println!("\nFormatted length: {} characters", formatted.len());
            }
        }
        Err(e) => {
            println!("Template-based counting not available: {}", e);
        }
    }

    println!("\n{}", "=".repeat(80));

    // Example 5: Context budget planning with templates
    println!("\n[Example 5] Context Budget Planning\n");

    let budget: usize = 100;
    println!("Context budget: {} tokens", budget);

    // Build a conversation
    let mut budget_conv = Conversation::new();
    budget_conv.add_message(budget_conv.system_message("You are a helpful assistant."))?;
    budget_conv.add_message(budget_conv.user_message("Tell me about Rust programming."))?;
    budget_conv.add_message(budget_conv.assistant_message(
        "Rust is a systems programming language that focuses on safety, concurrency, and performance. \
        It provides memory safety without garbage collection and prevents data races at compile time."
    ))?;
    budget_conv.add_message(budget_conv.user_message("What are its main features?"))?;

    let current_tokens = match counter.count_conversation_tokens_with_template(&budget_conv) {
        Ok(count) => count,
        Err(_) => counter.count_conversation_tokens(&budget_conv)?,
    };

    println!("Current conversation: {} tokens", current_tokens);
    println!(
        "Remaining budget: {} tokens",
        budget.saturating_sub(current_tokens)
    );

    if current_tokens > budget {
        println!(
            "\n⚠ Conversation exceeds budget by {} tokens!",
            current_tokens - budget
        );
        println!("  Consider trimming earlier messages or reducing response length.");
    } else {
        println!("\n✓ Conversation fits within budget!");
    }

    println!("\n{}", "=".repeat(80));
    println!("\nChat template example complete!");
    println!("\nKey takeaways:");
    println!("  • Chat templates provide the most accurate token counts");
    println!("  • Templates format conversations exactly as the model sees them");
    println!("  • Use count_conversation_tokens_with_template() for production");
    println!("  • Fall back to count_conversation_tokens() if no template exists");

    Ok(())
}
