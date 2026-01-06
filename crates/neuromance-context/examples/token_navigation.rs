//! Example demonstrating token navigation, search, and position mapping.
//!
//! Run with: cargo run --example token_navigation
//!
//! Requires HF_TOKEN environment variable to be set.

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

    // Initialize the token counter (downloads tokenizer on first run)
    let counter = TokenCounter::new(config).await?;

    println!("Token counter initialized!\n");
    println!("{}", "=".repeat(80));

    // Example 1: Tokenize text with position information
    println!("\n[Example 1] Tokenization with Position Mapping\n");

    let text = "The quick brown fox jumps over the lazy dog. This is a test sentence.";
    println!("Text: \"{}\"\n", text);

    let tokenized = counter.tokenize_with_positions(text)?;
    println!("Total tokens: {}\n", tokenized.token_count());

    println!("First 10 tokens:");
    for token_info in tokenized.tokens.iter().take(10) {
        println!(
            "  Token {}: \"{}\" (chars {}..{}, id: {})",
            token_info.index,
            token_info.token,
            token_info.char_start,
            token_info.char_end,
            token_info.token_id
        );
    }

    println!("\n{}", "=".repeat(80));

    // Example 2: Find token at specific character position
    println!("\n[Example 2] Find Token at Character Position\n");

    let char_pos = 20; // Position of 'f' in 'fox'
    if let Some(token) = tokenized.token_at_char_position(char_pos) {
        println!(
            "Character position {}: '{}'",
            char_pos,
            &text[char_pos..char_pos + 1]
        );
        println!(
            "  Token {}: \"{}\" (chars {}..{})",
            token.index, token.token, token.char_start, token.char_end
        );
        println!(
            "  Full token text: \"{}\"",
            &text[token.char_start..token.char_end]
        );
    }

    println!("\n{}", "=".repeat(80));

    // Example 3: Search with token positions
    println!("\n[Example 3] Search with Token Position Information\n");

    let search_text = "In neuroscience, neural networks process information. \
                      In computer science, artificial neural networks learn patterns.";
    println!("Text: \"{}\"\n", search_text);

    let pattern = r"neural";
    println!("Searching for pattern: '{}'\n", pattern);

    let matches = counter.search_with_token_positions(search_text, pattern)?;
    println!("Found {} matches:\n", matches.len());

    for (i, match_result) in matches.iter().enumerate() {
        println!("Match {}:", i + 1);
        println!("  Text: \"{}\"", match_result.matched_text);
        println!(
            "  Character range: {}..{} (length: {})",
            match_result.char_start,
            match_result.char_end,
            match_result.char_length()
        );
        if let Some((start, end)) = match_result.token_range() {
            println!("  Token range: {}..{}", start, end);
        }
        println!();
    }

    println!("{}", "=".repeat(80));

    // Example 4: Extract token ranges
    println!("\n[Example 4] Extract Text by Token Range\n");

    let range_text = "Token extraction allows you to get specific portions of text \
                     based on token indices rather than character positions.";
    println!("Text: \"{}\"\n", range_text);

    let tokenized_range = counter.tokenize_with_positions(range_text)?;
    println!("Total tokens: {}\n", tokenized_range.token_count());

    // Extract tokens 0-5
    let start = 0;
    let end = 5;
    let extracted = counter.extract_token_range(range_text, start, end)?;
    println!("Tokens {}..{}: \"{}\"", start, end, extracted);

    // Extract tokens 5-10
    let start = 5;
    let end = 10;
    let extracted = counter.extract_token_range(range_text, start, end)?;
    println!("Tokens {}..{}: \"{}\"", start, end, extracted);

    // Extract last 5 tokens
    let start = tokenized_range.token_count() - 5;
    let end = tokenized_range.token_count();
    let extracted = counter.extract_token_range(range_text, start, end)?;
    println!("Tokens {}..{} (last 5): \"{}\"", start, end, extracted);

    println!("\n{}", "=".repeat(80));

    // Example 5: Context window planning
    println!("\n[Example 5] Context Window Planning\n");

    let long_text = "This is a longer piece of text that we might want to fit into a context window. \
                    We can use token navigation to determine exactly how much text fits. \
                    For example, if we have a 50-token budget, we can extract exactly that many tokens. \
                    This is much more precise than character-based truncation.";

    let tokenized_long = counter.tokenize_with_positions(long_text)?;
    println!("Full text tokens: {}", tokenized_long.token_count());
    println!("Full text: \"{}\"\n", long_text);

    let budget = 20;
    println!("Context budget: {} tokens\n", budget);

    let fitted_text = counter.extract_token_range(long_text, 0, budget)?;
    println!("Text that fits in budget:");
    println!("\"{}\"", fitted_text);
    println!(
        "\nActual token count: {}",
        counter.count_tokens(&fitted_text)?
    );

    println!("\n{}", "=".repeat(80));

    // Example 6: Search in code with token positions
    println!("\n[Example 6] Search in Code\n");

    let code = r#"
fn calculate_tokens(text: &str) -> usize {
    let tokenizer = load_tokenizer();
    tokenizer.encode(text).len()
}

fn process_with_tokens(input: &str) -> Result<String> {
    let tokens = calculate_tokens(input);
    println!("Processing {} tokens", tokens);
    Ok(input.to_string())
}
"#;

    println!("Code:\n{}", code);

    // Search for function definitions
    let fn_pattern = r"fn \w+";
    println!("Searching for pattern: '{}'\n", fn_pattern);

    let fn_matches = counter.search_with_token_positions(code, fn_pattern)?;
    println!("Found {} function definitions:\n", fn_matches.len());

    for (i, match_result) in fn_matches.iter().enumerate() {
        println!("Function {}:", i + 1);
        println!("  Name: \"{}\"", match_result.matched_text);
        println!("  Character position: {}", match_result.char_start);
        if let Some((start, end)) = match_result.token_range() {
            println!("  Token range: {}..{} ({} tokens)", start, end, end - start);
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("\nToken navigation example complete!");

    Ok(())
}
