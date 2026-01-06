//! Example demonstrating tokenizer extraction from GGUF files.
//!
//! This example shows how to:
//! 1. Extract a tokenizer from a GGUF file
//! 2. Use it to count tokens in text
//! 3. Use it with a Conversation
//!
//! Usage:
//!   cargo run --example gguf_tokenizer -- path/to/model.gguf

use neuromance_common::Conversation;
use neuromance_context::{ModelConfig, TokenCounter};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // Get GGUF file path from command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path-to-gguf-file>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  cargo run --example gguf_tokenizer -- /path/to/model.gguf");
        std::process::exit(1);
    }

    let gguf_path = &args[1];
    println!("Loading tokenizer from GGUF file: {}\n", gguf_path);

    // Create a ModelConfig from the GGUF file
    let config = ModelConfig::from_gguf(gguf_path)?;
    println!("✓ Created ModelConfig from GGUF");
    println!("  Model repo: {}", config.model_repo);

    // Create TokenCounter (this will extract the tokenizer from GGUF)
    let counter = TokenCounter::new(config).await?;
    println!("✓ Loaded TokenCounter from GGUF\n");

    // Test 1: Count tokens in simple text
    println!("Test 1: Simple token counting");
    println!("==============================");
    let test_texts = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning was the Word, and the Word was with God, and the Word was God.",
    ];

    for text in &test_texts {
        let count = counter.count_tokens(text)?;
        println!("  \"{}\"", text);
        println!("    → {} tokens\n", count);
    }

    // Test 2: Tokenize with positions
    println!("\nTest 2: Detailed tokenization");
    println!("==============================");
    let text = "Artificial intelligence is transforming the world.";
    let tokenized = counter.tokenize_with_positions(text)?;

    println!("Text: \"{}\"", text);
    println!("Token count: {}\n", tokenized.token_count());
    println!("Tokens:");
    for token_info in &tokenized.tokens {
        println!(
            "  [{}] '{}' (id: {}, pos: {}-{})",
            token_info.index,
            token_info.token,
            token_info.token_id,
            token_info.char_start,
            token_info.char_end
        );
    }

    // Test 3: Count conversation tokens
    println!("\n\nTest 3: Conversation token counting");
    println!("====================================");
    let mut conv = Conversation::new();
    conv.add_message(conv.user_message("What is the capital of France?"))?;
    conv.add_message(conv.assistant_message("The capital of France is Paris."))?;
    conv.add_message(conv.user_message("What is its population?"))?;

    let conv_tokens = counter.count_conversation_tokens(&conv)?;
    println!("Conversation with {} messages:", conv.get_messages().len());
    for (i, msg) in conv.get_messages().iter().enumerate() {
        let msg_tokens = counter.count_message_tokens(msg)?;
        println!(
            "  Message {}: {:?} - {} tokens",
            i + 1,
            msg.role,
            msg_tokens
        );
        println!("    Content: \"{}\"", msg.content);
    }
    println!("\nTotal conversation tokens: {}", conv_tokens);

    // Test 4: Chat template formatting (if available)
    if let Some(template) = counter.get_chat_template() {
        println!("\n\nTest 4: Chat template formatting");
        println!("=================================");
        println!("✓ Chat template found in GGUF metadata");
        println!("Template length: {} characters\n", template.len());

        match counter.format_conversation_with_template(&conv) {
            Ok(formatted) => {
                println!("Formatted conversation:");
                println!("{}", formatted);
                println!();

                let template_tokens = counter.count_tokens(&formatted)?;
                println!("Tokens with template formatting: {}", template_tokens);
                println!(
                    "Overhead from formatting: {} tokens",
                    template_tokens as i64 - conv_tokens as i64
                );
            }
            Err(e) => {
                println!("Note: Could not format with template: {}", e);
            }
        }
    } else {
        println!("\n\nTest 4: Chat template");
        println!("=====================");
        println!("✗ No chat template found in GGUF metadata");
    }

    // Test 5: Search with token positions
    println!("\n\nTest 5: Search with token positions");
    println!("====================================");
    let search_text = "The cat sat on the mat. The dog ran in the park.";
    let pattern = r"the";
    let matches = counter.search_with_token_positions(search_text, pattern)?;

    println!("Searching for pattern '{}' in:", pattern);
    println!("  \"{}\"", search_text);
    println!("\nFound {} matches:", matches.len());
    for (i, m) in matches.iter().enumerate() {
        println!(
            "  Match {}: \"{}\" at chars {}-{}, tokens {:?}-{:?}",
            i + 1,
            m.matched_text,
            m.char_start,
            m.char_end,
            m.token_start,
            m.token_end
        );
    }

    println!("\n✓ All tests completed successfully!");

    Ok(())
}
