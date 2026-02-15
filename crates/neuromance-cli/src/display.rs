//! Display utilities for CLI output formatting
//!
//! Provides formatted output for messages, tool calls, and tool results

use std::io::Write;

use colored::Colorize;
use neuromance_common::ToolCall;

/// Display a tool call being requested by the assistant
pub fn display_tool_call_request(tool_call: &ToolCall) {
    println!(
        "\n○ {} {}",
        "Tool call:".bright_yellow(),
        tool_call.function.name.bright_green().bold()
    );

    // Parse and display arguments
    let args_str = tool_call.function.arguments_json();
    if let Ok(args) = serde_json::from_str::<serde_json::Value>(&args_str)
        && let Some(obj) = args.as_object()
    {
        for (key, value) in obj {
            println!("  {}: {}", key.cyan(), value);
        }
    }
}

/// Display tool execution result
pub fn display_tool_result(tool_name: &str, result: &str, success: bool) {
    // Truncate very long results
    let display_result = if result.len() > 200 {
        let truncate_idx = result
            .char_indices()
            .take(200)
            .last()
            .map_or(0, |(idx, ch)| idx + ch.len_utf8());
        format!("{}... ({} chars)", &result[..truncate_idx], result.len())
    } else {
        result.to_string()
    };

    if success {
        println!("○ {} {}", "Result:".bright_green(), tool_name.bright_cyan());
        println!("  {display_result}");
    } else {
        println!("○ {} {}", "Failed:".bright_red(), tool_name.bright_cyan());
        println!("  {}", display_result.bright_red());
    }
}

/// Display assistant response header
pub fn display_assistant_header() {
    println!("\n╭─● {}", "Assistant".bright_magenta().bold());
    println!("╰───────────○");
    let _ = std::io::stdout().flush();
}

/// Display end of assistant flow
pub fn display_assistant_end() {
    println!();
}
