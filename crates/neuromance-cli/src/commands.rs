//! Command implementations for the CLI.

use std::io::Write;

use anyhow::Result;
use chrono::{DateTime, Utc};
use colored::Colorize;
use neuromance_common::ToolApproval;
use neuromance_proto::{conversation_summary_from_proto, proto};
use rustyline::DefaultEditor;
use serde_json::json;

use crate::client::DaemonClient;
use crate::display::{
    display_assistant_end, display_assistant_header, display_tool_call_request, display_tool_result,
};
use crate::theme::Theme;

/// Prompts the user to approve, deny, or quit a tool call.
fn prompt_tool_approval(
    tool_call: &neuromance_common::ToolCall,
    theme: &Theme,
) -> Result<ToolApproval> {
    display_tool_call_request(tool_call, theme);

    let args_display = tool_call.function.arguments_json();
    println!(
        "\n{} Execute tool '{}' with arguments:",
        "Tool Approval Required:".bright_yellow().bold(),
        tool_call.function.name.bright_green(),
    );
    println!("  {}", args_display.bright_cyan());
    println!(
        "{} /yes or /y to approve, /no or /n to deny, \
         /quit or /q to abort",
        "→".bright_yellow()
    );

    let mut editor = DefaultEditor::new()?;

    loop {
        let prompt = format!("{} ", "Approve?".bright_yellow().bold());
        match editor.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();
                editor.add_history_entry(line).ok();

                match line {
                    "/yes" | "/y" => {
                        println!("{} Tool approved\n", "✓".bright_green().bold());
                        return Ok(ToolApproval::Approved);
                    }
                    "/no" | "/n" => {
                        println!("{} Tool denied\n", "✗".bright_red().bold());
                        return Ok(ToolApproval::Denied(
                            "User denied tool execution".to_string(),
                        ));
                    }
                    "/quit" | "/q" => {
                        println!("{} Aborting conversation\n", "!".bright_red().bold());
                        return Ok(ToolApproval::Quit);
                    }
                    _ => {
                        println!(
                            "{} Invalid response. \
                             Use /yes, /no, or /quit",
                            "!".bright_yellow().bold()
                        );
                    }
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                return Ok(ToolApproval::Denied(
                    "User interrupted with CTRL-C".to_string(),
                ));
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                return Ok(ToolApproval::Quit);
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Error reading input: {e}"));
            }
        }
    }
}

/// Sends a message to a conversation.
///
/// Returns the conversation ID from the completed message,
/// so callers can pin subsequent messages to the same conversation.
pub async fn send_message(
    client: &mut DaemonClient,
    conversation_id: Option<String>,
    message: String,
    theme: &Theme,
) -> Result<Option<String>> {
    let mut session = client.chat(conversation_id, message).await?;

    display_assistant_header(theme);

    let mut resolved_id = None;
    loop {
        let Some(event) = session.next_event().await? else {
            break;
        };

        let cid = event.conversation_id;
        match event.event {
            Some(proto::chat_event::Event::StreamChunk(chunk)) => {
                print!("{}", chunk.content);
                let _ = std::io::stdout().flush();
            }
            Some(proto::chat_event::Event::ToolApprovalRequest(req)) => {
                if let Some(tc_proto) = req.tool_call {
                    let tc = neuromance_common::ToolCall::from(tc_proto);
                    let approval = prompt_tool_approval(&tc, theme)?;

                    session
                        .send_tool_approval(cid, tc.id, approval)
                        .await?;
                }
            }
            Some(proto::chat_event::Event::ToolResult(tr)) => {
                display_tool_result(&tr.tool_name, &tr.result, tr.success, theme);
            }
            Some(proto::chat_event::Event::Usage(u)) => {
                let total = u.total_tokens.to_string();
                let input = u.prompt_tokens.to_string();
                let output = u.completion_tokens.to_string();
                println!(
                    "{}",
                    theme.usage_tokens.render(&[
                        ("total", total.as_str()),
                        ("input", input.as_str()),
                        ("output", output.as_str()),
                    ])
                );
            }
            Some(proto::chat_event::Event::MessageCompleted(_mc)) => {
                resolved_id = Some(cid);
                display_assistant_end(theme);
                break;
            }
            Some(proto::chat_event::Event::Error(e)) => {
                eprintln!("\n{} {}", "Error:".bright_red(), e.message);
                break;
            }
            None => {}
        }
    }

    Ok(resolved_id)
}

/// Creates a new conversation.
pub async fn new_conversation(
    client: &mut DaemonClient,
    model: Option<String>,
    system_message: Option<String>,
) -> Result<()> {
    let resp = client.new_conversation(model, system_message).await?;

    if let Some(conv) = resp.conversation {
        println!(
            "{} Created conversation {}",
            "✓".bright_green(),
            conv.short_id.bright_cyan()
        );
        if let Some(title) = conv.title {
            println!("  Title: {title}");
        }
        println!("  Model: {}", conv.model.bright_yellow());
    }

    Ok(())
}

/// Lists messages from a conversation.
pub async fn list_messages(
    client: &mut DaemonClient,
    conversation_id: Option<String>,
    limit: Option<usize>,
) -> Result<()> {
    let resp = client.list_messages(conversation_id, limit).await?;

    if resp.messages.is_empty() {
        println!("No messages in this conversation");
        return Ok(());
    }

    println!(
        "Showing {} of {} messages:",
        resp.messages.len(),
        resp.total_count
    );
    println!();

    for msg in resp.messages {
        let role =
            proto::MessageRole::try_from(msg.role).unwrap_or(proto::MessageRole::Unspecified);
        let role_str = match role {
            proto::MessageRole::System => "System".bright_blue(),
            proto::MessageRole::User => "User".bright_green(),
            proto::MessageRole::Assistant => "Assistant".bright_magenta(),
            proto::MessageRole::Tool => "Tool".bright_yellow(),
            proto::MessageRole::Unspecified => "Unknown".bright_white(),
        };

        println!("{} {}", "●".bright_white(), role_str.bold());

        if !msg.content.is_empty() {
            println!("  {}", truncate_chars(&msg.content, 200));
        }

        if !msg.tool_calls.is_empty() {
            println!("  {} tool calls", msg.tool_calls.len());
        }

        println!();
    }

    Ok(())
}

/// Lists all conversations.
pub async fn list_conversations(client: &mut DaemonClient, limit: Option<usize>) -> Result<()> {
    let resp = client.list_conversations(limit).await?;

    if resp.conversations.is_empty() {
        println!("No conversations found");
        return Ok(());
    }

    println!("Conversations:");
    println!();

    for conv in resp.conversations {
        let id_display = conv.short_id.bright_cyan();
        let model_display = conv.model.bright_yellow();

        print!("{} {} ({})", "●".bright_white(), id_display, model_display);

        if let Some(title) = conv.title {
            print!(" - {title}");
        }

        println!();

        if !conv.bookmarks.is_empty() {
            println!("  Bookmarks: {}", conv.bookmarks.join(", "));
        }

        println!("  {} messages", conv.message_count);
        println!();
    }

    Ok(())
}

/// Sets a bookmark for a conversation.
pub async fn set_bookmark(
    client: &mut DaemonClient,
    conversation_id: String,
    name: String,
) -> Result<()> {
    let resp = client.set_bookmark(conversation_id, name).await?;
    println!("{} {}", "✓".bright_green(), resp.message);
    Ok(())
}

/// Removes a bookmark.
pub async fn remove_bookmark(client: &mut DaemonClient, name: String) -> Result<()> {
    let resp = client.remove_bookmark(name).await?;
    println!("{} {}", "✓".bright_green(), resp.message);
    Ok(())
}

/// Deletes a conversation after optional confirmation.
pub async fn delete_conversation(
    client: &mut DaemonClient,
    conversation_id: String,
    force: bool,
) -> Result<()> {
    if !force {
        print!(
            "Delete conversation '{}'? [y/N] ",
            conversation_id.bright_cyan()
        );
        let _ = std::io::stdout().flush();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        if !matches!(input.trim(), "y" | "Y" | "yes" | "YES") {
            println!("Cancelled");
            return Ok(());
        }
    }

    let resp = client.delete_conversation(conversation_id).await?;
    println!("{} {}", "✓".bright_green(), resp.message);
    Ok(())
}

/// Gets daemon status.
pub async fn daemon_status(client: &mut DaemonClient) -> Result<()> {
    let resp = client.get_status().await?;
    println!("{} Daemon is running", "✓".bright_green());
    println!("  Uptime: {}s", resp.uptime_seconds);
    println!("  Active conversations: {}", resp.active_conversations);
    Ok(())
}

/// Checks daemon health and version compatibility.
pub async fn daemon_health(client: &mut DaemonClient) -> Result<()> {
    let resp = client.health_check().await?;

    if resp.compatible {
        println!("{} Daemon is healthy", "✓".bright_green());
    } else {
        println!("{} Version compatibility issue", "⚠".bright_yellow());
    }

    println!("  Daemon version: {}", resp.daemon_version.bright_cyan());
    println!(
        "  Client version: {}",
        env!("CARGO_PKG_VERSION").bright_cyan()
    );
    println!(
        "  Compatible: {}",
        if resp.compatible {
            "yes".bright_green()
        } else {
            "no".bright_red()
        }
    );
    println!("  Uptime: {}s", resp.uptime_seconds);

    if let Some(warning_msg) = resp.warning {
        println!();
        println!("{} {}", "Warning:".bright_yellow(), warning_msg);
    }

    Ok(())
}

/// Shuts down the daemon.
pub async fn shutdown_daemon(client: &mut DaemonClient) -> Result<()> {
    let _ = client.shutdown().await?;
    println!("{} Daemon shutdown requested", "✓".bright_green());
    Ok(())
}

/// Lists available models.
pub async fn list_models(client: &mut DaemonClient) -> Result<()> {
    let resp = client.list_models().await?;

    println!("Available models:");
    println!();

    for model in resp.models {
        let marker = if model.nickname == resp.active {
            "●".bright_green()
        } else {
            "○".bright_white()
        };

        println!(
            "{} {} ({} / {})",
            marker,
            model.nickname.bright_cyan(),
            model.provider.bright_yellow(),
            model.model
        );
    }

    Ok(())
}

/// Switches the model for a conversation.
pub async fn switch_model(
    client: &mut DaemonClient,
    conversation_id: Option<String>,
    model_nickname: String,
) -> Result<()> {
    let resp = client.switch_model(conversation_id, model_nickname).await?;

    if let Some(conv) = resp.conversation {
        println!(
            "{} Switched to model {}",
            "✓".bright_green(),
            conv.model.bright_yellow()
        );
        println!(
            "  {} {}",
            "Conversation:".dimmed(),
            conv.short_id.bright_cyan()
        );
    }

    Ok(())
}

/// Shows comprehensive status.
#[allow(clippy::cast_possible_truncation)]
pub async fn status(client: &mut DaemonClient, json: bool) -> Result<()> {
    let resp = client.get_detailed_status().await?;
    let active = resp.active_conversations as usize;

    if json {
        let current = resp
            .current_conversation
            .and_then(|cs| conversation_summary_from_proto(cs).ok());

        print_status_json(true, Some(resp.uptime_seconds), active, current.as_ref())?;
    } else {
        let current = resp
            .current_conversation
            .and_then(|cs| conversation_summary_from_proto(cs).ok());

        print_status_human(resp.uptime_seconds, active, current);
    }

    Ok(())
}

/// Handles status when daemon is not running.
pub fn status_daemon_not_running(json: bool) -> Result<()> {
    if json {
        print_status_json(false, None, 0, None)?;
    } else {
        println!("{} Daemon is not running", "○".dimmed());
        println!();
        println!("Start a conversation with:");
        println!("  {}", "nx \"your message\"".bright_cyan());
    }
    Ok(())
}

/// Formats human-readable status output.
fn print_status_human(
    uptime_seconds: u64,
    active_conversations: usize,
    current_conversation: Option<neuromance_common::protocol::ConversationSummary>,
) {
    println!("{} Daemon is running", "✓".bright_green());
    println!("  Uptime: {}", format_duration(uptime_seconds));
    println!("  Active conversations: {active_conversations}");
    println!();

    if let Some(conv) = current_conversation {
        println!("{} Current conversation", "●".bright_cyan());
        println!("  ID: {}", conv.short_id.bright_cyan());

        if let Some(title) = conv.title {
            println!("  Title: {title}");
        }

        println!("  Model: {}", conv.model.bright_yellow());
        println!("  Messages: {}", conv.message_count);

        if !conv.bookmarks.is_empty() {
            println!("  Bookmarks: {}", conv.bookmarks.join(", ").bright_green());
        }

        println!("  Updated: {}", format_relative_time(conv.updated_at));
    } else {
        println!("{} No current conversation", "○".dimmed());
    }
}

/// Formats JSON status output.
fn print_status_json(
    daemon_running: bool,
    uptime_seconds: Option<u64>,
    active_conversations: usize,
    current_conversation: Option<&neuromance_common::protocol::ConversationSummary>,
) -> Result<()> {
    let output = if daemon_running {
        json!({
            "daemon": {
                "running": true,
                "uptime_seconds": uptime_seconds,
                "active_conversations": active_conversations
            },
            "current_conversation": current_conversation
        })
    } else {
        json!({
            "daemon": {
                "running": false
            },
            "current_conversation": null
        })
    };

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

/// Formats duration as human-readable string.
fn format_duration(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;

    if hours > 0 {
        format!("{hours}h {minutes}m {secs}s")
    } else if minutes > 0 {
        format!("{minutes}m {secs}s")
    } else {
        format!("{secs}s")
    }
}

/// Formats relative time.
fn format_relative_time(timestamp: DateTime<Utc>) -> String {
    let now = Utc::now();
    let diff = now.signed_duration_since(timestamp);

    if diff.num_days() > 0 {
        format!("{} days ago", diff.num_days())
    } else if diff.num_hours() > 0 {
        format!("{} hours ago", diff.num_hours())
    } else if diff.num_minutes() > 0 {
        format!("{} minutes ago", diff.num_minutes())
    } else {
        "just now".to_string()
    }
}

/// Truncates a string to at most `max_chars` characters.
fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    let truncate_idx = s
        .char_indices()
        .take(max_chars)
        .last()
        .map_or(0, |(idx, ch)| idx + ch.len_utf8());
    format!("{}...", &s[..truncate_idx])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use chrono::TimeDelta;

    #[test]
    fn format_duration_zero() {
        assert_eq!(format_duration(0), "0s");
    }

    #[test]
    fn format_duration_seconds_only() {
        assert_eq!(format_duration(45), "45s");
    }

    #[test]
    fn format_duration_minutes_and_seconds() {
        assert_eq!(format_duration(125), "2m 5s");
    }

    #[test]
    fn format_duration_hours_minutes_seconds() {
        assert_eq!(format_duration(3661), "1h 1m 1s");
    }

    #[test]
    fn format_duration_exact_hour() {
        assert_eq!(format_duration(3600), "1h 0m 0s");
    }

    #[test]
    fn format_duration_exact_minute() {
        assert_eq!(format_duration(60), "1m 0s");
    }

    #[test]
    fn format_relative_time_just_now() {
        let now = Utc::now();
        assert_eq!(format_relative_time(now), "just now");
    }

    #[test]
    fn format_relative_time_minutes_ago() {
        let timestamp = Utc::now() - TimeDelta::minutes(5);
        assert_eq!(format_relative_time(timestamp), "5 minutes ago");
    }

    #[test]
    fn format_relative_time_hours_ago() {
        let timestamp = Utc::now() - TimeDelta::hours(3);
        assert_eq!(format_relative_time(timestamp), "3 hours ago");
    }

    #[test]
    fn format_relative_time_days_ago() {
        let timestamp = Utc::now() - TimeDelta::days(2);
        assert_eq!(format_relative_time(timestamp), "2 days ago");
    }

    #[test]
    fn format_relative_time_one_day() {
        let timestamp = Utc::now() - TimeDelta::days(1);
        assert_eq!(format_relative_time(timestamp), "1 days ago");
    }

    #[test]
    fn truncate_chars_short_string() {
        assert_eq!(truncate_chars("hello", 200), "hello");
    }

    #[test]
    fn truncate_chars_ascii() {
        let long = "a".repeat(300);
        let result = truncate_chars(&long, 200);
        assert_eq!(result.len(), 203);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn truncate_chars_multibyte_utf8() {
        let emojis = "\u{1F600}".repeat(50);
        let result = truncate_chars(&emojis, 10);
        assert!(result.ends_with("..."));
        let char_count = result.chars().count();
        assert_eq!(char_count, 13);
    }

    #[test]
    fn truncate_chars_empty_string() {
        assert_eq!(truncate_chars("", 200), "");
    }

    #[test]
    fn truncate_chars_mixed_utf8() {
        let mixed = "aé中\u{1F600}".repeat(60);
        let result = truncate_chars(&mixed, 200);
        assert!(result.ends_with("..."));
    }
}
