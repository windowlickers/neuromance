//! Example demonstrating GGUF metadata extraction.
//!
//! This example shows how to extract model metadata from a GGUF file
//! without loading the full model tensors.
//!
//! Usage:
//!   cargo run --example gguf_metadata -- path/to/model.gguf

use neuromance_context::tokens::gguf::GGUFModelInfo;
use std::env;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // Get GGUF file path from command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path-to-gguf-file>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  cargo run --example gguf_metadata -- /path/to/model.gguf");
        std::process::exit(1);
    }

    let gguf_path = &args[1];
    println!("Reading GGUF file: {}\n", gguf_path);

    // Extract metadata (fast, no tensor loading)
    let info = GGUFModelInfo::from_file(gguf_path)?;

    // Print formatted summary
    info.print_summary();

    // Access specific fields
    println!("\n\nDetailed Field Access:");
    println!("======================");

    if let Some(ctx_len) = info.context_length {
        println!("✓ Context window: {} tokens", ctx_len);
    } else {
        println!("✗ Context length not found in metadata");
    }

    if let Some(vocab) = info.vocab_size {
        println!("✓ Vocabulary contains {} tokens", vocab);
    } else {
        println!("✗ Vocabulary size not found in metadata");
    }

    if let Some(template) = &info.chat_template {
        println!("✓ Chat template found ({} chars)", template.len());
        println!("\nChat template preview:");
        println!("{}", &template.chars().take(200).collect::<String>());
        if template.len() > 200 {
            println!("... (truncated)");
        }
    } else {
        println!("✗ No chat template in GGUF metadata");
    }

    // Show raw metadata keys
    println!(
        "\n\nAvailable Metadata Keys ({} total):",
        info.raw_metadata.len()
    );
    println!("====================================");
    let mut keys: Vec<_> = info.raw_metadata.keys().collect();
    keys.sort();

    for key in keys.iter().take(20) {
        if let Some(value) = info.raw_metadata.get(*key) {
            match value {
                neuromance_context::tokens::gguf::MetadataValue::String(s) => {
                    if s.len() > 50 {
                        println!(
                            "  {} = \"{}...\"",
                            key,
                            &s.chars().take(47).collect::<String>()
                        );
                    } else {
                        println!("  {} = \"{}\"", key, s);
                    }
                }
                neuromance_context::tokens::gguf::MetadataValue::U32(n) => {
                    println!("  {} = {}", key, n)
                }
                neuromance_context::tokens::gguf::MetadataValue::U64(n) => {
                    println!("  {} = {}", key, n)
                }
                neuromance_context::tokens::gguf::MetadataValue::I32(n) => {
                    println!("  {} = {}", key, n)
                }
                neuromance_context::tokens::gguf::MetadataValue::I64(n) => {
                    println!("  {} = {}", key, n)
                }
                neuromance_context::tokens::gguf::MetadataValue::F32(f) => {
                    println!("  {} = {}", key, f)
                }
                neuromance_context::tokens::gguf::MetadataValue::F64(f) => {
                    println!("  {} = {}", key, f)
                }
                neuromance_context::tokens::gguf::MetadataValue::Bool(b) => {
                    println!("  {} = {}", key, b)
                }
                neuromance_context::tokens::gguf::MetadataValue::Array(_) => {
                    println!("  {} = [array]", key)
                }
            }
        }
    }

    if keys.len() > 20 {
        println!("  ... and {} more", keys.len() - 20);
    }

    Ok(())
}
