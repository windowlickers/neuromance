//! Display utilities for CLI output formatting
//!
//! Provides formatted output for messages, tool calls, and tool results

use std::io::Write;

use neuromance_common::ToolCall;

use crate::theme::Theme;

/// Display a tool call being requested by the assistant
pub fn display_tool_call_request(tool_call: &ToolCall, theme: &Theme) {
    println!(
        "{}",
        theme
            .tool_call
            .render(&[("name", &tool_call.function.name)])
    );

    let args_str = tool_call.function.arguments_json();
    if let Ok(args) = serde_json::from_str::<serde_json::Value>(&args_str)
        && let Some(obj) = args.as_object()
    {
        for (key, value) in obj {
            let val_str = value.to_string();
            println!(
                "{}",
                theme
                    .tool_arg
                    .render(&[("key", key.as_str()), ("value", &val_str),])
            );
        }
    }
}

/// Display tool execution result
pub fn display_tool_result(tool_name: &str, result: &str, success: bool, theme: &Theme) {
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

    let label = if success {
        &theme.tool_result_ok
    } else {
        &theme.tool_result_err
    };
    println!("{}", label.render(&[("name", tool_name)]));
    println!("  {display_result}");
}

/// Display assistant response header
pub fn display_assistant_header(theme: &Theme) {
    println!("{}", theme.assistant_header.render(&[]));
    let _ = std::io::stdout().flush();
}

/// Display end of assistant flow
pub fn display_assistant_end(theme: &Theme) {
    let footer = theme.assistant_footer.render(&[]);
    if footer.is_empty() {
        println!();
    } else {
        println!("{footer}");
    }
}
