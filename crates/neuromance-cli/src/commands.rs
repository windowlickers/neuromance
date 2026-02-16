//! Command implementations for the CLI.

use std::io::Write;

use anyhow::Result;
use chrono::{DateTime, Utc};
use colored::Colorize;
use neuromance_common::protocol::{DaemonRequest, DaemonResponse};
use neuromance_common::{ToolApproval, ToolCall};
use rustyline::DefaultEditor;
use serde_json::json;

use crate::client::DaemonClient;
use crate::display::{
    display_assistant_end, display_assistant_header, display_tool_call_request, display_tool_result,
};

/// Prompts the user to approve, deny, or quit a tool call.
///
/// Returns the user's approval decision.
fn prompt_tool_approval(tool_call: &ToolCall) -> Result<ToolApproval> {
    // Display the tool request
    display_tool_call_request(tool_call);

    let args_display = tool_call.function.arguments_json();
    println!(
        "\n{} Execute tool '{}' with arguments:",
        "Tool Approval Required:".bright_yellow().bold(),
        tool_call.function.name.bright_green(),
    );
    println!("  {}", args_display.bright_cyan());
    println!(
        "{} /yes or /y to approve, /no or /n to deny, /quit or /q to abort",
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
                            "{} Invalid response. Use /yes, /no, or /quit",
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
pub async fn send_message(
    client: &mut DaemonClient,
    conversation_id: Option<String>,
    message: String,
) -> Result<()> {
    let request = DaemonRequest::SendMessage {
        conversation_id,
        content: message,
    };

    client.send_request(&request).await?;

    display_assistant_header();

    // Read streaming responses with manual loop to handle async tool approval
    loop {
        let response = client.read_response().await?;

        match response {
            DaemonResponse::StreamChunk { content, .. } => {
                print!("{content}");
                let _ = std::io::stdout().flush();
            }
            DaemonResponse::ToolApprovalRequest {
                conversation_id,
                tool_call,
            } => {
                // Prompt user for approval
                let approval = prompt_tool_approval(&tool_call)?;

                // Send approval back to daemon
                let approval_request = DaemonRequest::ToolApproval {
                    conversation_id,
                    tool_call_id: tool_call.id.clone(),
                    approval,
                };
                client.send_request(&approval_request).await?;

                // Read and ignore the Success response
                let _ = client.read_response().await?;
            }
            DaemonResponse::ToolResult {
                tool_name,
                result,
                success,
                ..
            } => {
                display_tool_result(&tool_name, &result, success);
            }
            DaemonResponse::Usage { usage, .. } => {
                println!(
                    "\n{} {} tokens (in: {}, out: {})",
                    "○".bright_blue(),
                    usage.total_tokens,
                    usage.prompt_tokens,
                    usage.completion_tokens
                );
            }
            DaemonResponse::MessageCompleted { .. } => {
                display_assistant_end();
                break; // Done
            }
            DaemonResponse::Error { message } => {
                eprintln!("\n{} {message}", "Error:".bright_red());
                break; // Done
            }
            _ => {
                // Unexpected response, continue
            }
        }
    }

    Ok(())
}

/// Creates a new conversation.
pub async fn new_conversation(
    client: &mut DaemonClient,
    model: Option<String>,
    system_message: Option<String>,
) -> Result<()> {
    let request = DaemonRequest::NewConversation {
        model,
        system_message,
    };

    client.send_request(&request).await?;

    let response = client.read_response().await?;

    match response {
        DaemonResponse::ConversationCreated { conversation } => {
            println!(
                "{} Created conversation {}",
                "✓".bright_green(),
                conversation.short_id.bright_cyan()
            );
            if let Some(title) = conversation.title {
                println!("  Title: {title}");
            }
            println!("  Model: {}", conversation.model.bright_yellow());
        }
        DaemonResponse::Error { message } => {
            eprintln!("{} {message}", "Error:".bright_red());
        }
        _ => {
            eprintln!("{} Unexpected response", "Error:".bright_red());
        }
    }

    Ok(())
}

/// Lists messages from a conversation.
pub async fn list_messages(
    client: &mut DaemonClient,
    conversation_id: Option<String>,
    limit: Option<usize>,
) -> Result<()> {
    let request = DaemonRequest::ListMessages {
        conversation_id,
        limit,
    };

    client.send_request(&request).await?;

    let response = client.read_response().await?;

    match response {
        DaemonResponse::Messages {
            messages,
            total_count,
            ..
        } => {
            if messages.is_empty() {
                println!("No messages in this conversation");
                return Ok(());
            }

            println!("Showing {} of {} messages:", messages.len(), total_count);
            println!();

            for msg in messages {
                let role_str = match msg.role {
                    neuromance_common::MessageRole::System => "System".bright_blue(),
                    neuromance_common::MessageRole::User => "User".bright_green(),
                    neuromance_common::MessageRole::Assistant => "Assistant".bright_magenta(),
                    neuromance_common::MessageRole::Tool => "Tool".bright_yellow(),
                    _ => "Unknown".bright_white(),
                };

                println!("{} {}", "●".bright_white(), role_str.bold());

                if !msg.content.is_empty() {
                    let preview = if msg.content.len() > 200 {
                        format!("{}...", &msg.content[..200])
                    } else {
                        msg.content.clone()
                    };
                    println!("  {preview}");
                }

                if !msg.tool_calls.is_empty() {
                    println!("  {} tool calls", msg.tool_calls.len());
                }

                println!();
            }
        }
        DaemonResponse::Error { message } => {
            eprintln!("{} {message}", "Error:".bright_red());
        }
        _ => {
            eprintln!("{} Unexpected response", "Error:".bright_red());
        }
    }

    Ok(())
}

/// Lists all conversations.
pub async fn list_conversations(client: &mut DaemonClient, limit: Option<usize>) -> Result<()> {
    let request = DaemonRequest::ListConversations { limit };

    client.send_request(&request).await?;

    let response = client.read_response().await?;

    match response {
        DaemonResponse::Conversations { conversations } => {
            if conversations.is_empty() {
                println!("No conversations found");
                return Ok(());
            }

            println!("Conversations:");
            println!();

            for conv in conversations {
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
        }
        DaemonResponse::Error { message } => {
            eprintln!("{} {message}", "Error:".bright_red());
        }
        _ => {
            eprintln!("{} Unexpected response", "Error:".bright_red());
        }
    }

    Ok(())
}

/// Sets a bookmark for a conversation.
pub async fn set_bookmark(
    client: &mut DaemonClient,
    conversation_id: String,
    name: String,
) -> Result<()> {
    let request = DaemonRequest::SetBookmark {
        conversation_id: conversation_id.clone(),
        name: name.clone(),
    };

    client.send_request(&request).await?;

    let response = client.read_response().await?;

    match response {
        DaemonResponse::Success { message } => {
            println!("{} {message}", "✓".bright_green());
        }
        DaemonResponse::Error { message } => {
            eprintln!("{} {message}", "Error:".bright_red());
        }
        _ => {
            eprintln!("{} Unexpected response", "Error:".bright_red());
        }
    }

    Ok(())
}

/// Removes a bookmark.
pub async fn remove_bookmark(client: &mut DaemonClient, name: String) -> Result<()> {
    let request = DaemonRequest::RemoveBookmark { name: name.clone() };

    client.send_request(&request).await?;

    let response = client.read_response().await?;

    match response {
        DaemonResponse::Success { message } => {
            println!("{} {message}", "✓".bright_green());
        }
        DaemonResponse::Error { message } => {
            eprintln!("{} {message}", "Error:".bright_red());
        }
        _ => {
            eprintln!("{} Unexpected response", "Error:".bright_red());
        }
    }

    Ok(())
}

/// Gets daemon status.
pub async fn daemon_status(client: &mut DaemonClient) -> Result<()> {
    let request = DaemonRequest::Status;

    client.send_request(&request).await?;

    let response = client.read_response().await?;

    match response {
        DaemonResponse::Status {
            uptime_seconds,
            active_conversations,
            ..
        } => {
            println!("{} Daemon is running", "✓".bright_green());
            println!("  Uptime: {uptime_seconds}s");
            println!("  Active conversations: {active_conversations}");
        }
        DaemonResponse::Error { message } => {
            eprintln!("{} {message}", "Error:".bright_red());
        }
        _ => {
            eprintln!("{} Unexpected response", "Error:".bright_red());
        }
    }

    Ok(())
}

/// Checks daemon health and version compatibility.
pub async fn daemon_health(client: &mut DaemonClient) -> Result<()> {
    let client_version = env!("CARGO_PKG_VERSION").to_string();
    let request = DaemonRequest::Health { client_version };

    client.send_request(&request).await?;

    let response = client.read_response().await?;

    match response {
        DaemonResponse::Health {
            daemon_version,
            compatible,
            warning,
            uptime_seconds,
        } => {
            if compatible {
                println!("{} Daemon is healthy", "✓".bright_green());
            } else {
                println!("{} Version compatibility issue", "⚠".bright_yellow());
            }

            println!("  Daemon version: {}", daemon_version.bright_cyan());
            println!("  Client version: {}", env!("CARGO_PKG_VERSION").bright_cyan());
            println!("  Compatible: {}", if compatible { "yes".bright_green() } else { "no".bright_red() });
            println!("  Uptime: {uptime_seconds}s");

            if let Some(warning_msg) = warning {
                println!();
                println!("{} {}", "Warning:".bright_yellow(), warning_msg);
            }
        }
        DaemonResponse::Error { message } => {
            eprintln!("{} {message}", "Error:".bright_red());
        }
        _ => {
            eprintln!("{} Unexpected response", "Error:".bright_red());
        }
    }

    Ok(())
}

/// Shuts down the daemon.
pub async fn shutdown_daemon(client: &mut DaemonClient) -> Result<()> {
    let request = DaemonRequest::Shutdown;

    client.send_request(&request).await?;

    println!("{} Daemon shutdown requested", "✓".bright_green());

    Ok(())
}

/// Lists available models.
pub async fn list_models(client: &mut DaemonClient) -> Result<()> {
    let request = DaemonRequest::ListModels;

    client.send_request(&request).await?;

    let response = client.read_response().await?;

    match response {
        DaemonResponse::Models { models, active } => {
            println!("Available models:");
            println!();

            for model in models {
                let marker = if model.nickname == active {
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
        }
        DaemonResponse::Error { message } => {
            eprintln!("{} {message}", "Error:".bright_red());
        }
        _ => {
            eprintln!("{} Unexpected response", "Error:".bright_red());
        }
    }

    Ok(())
}

/// Switches the model for a conversation.
pub async fn switch_model(
    client: &mut DaemonClient,
    conversation_id: Option<String>,
    model_nickname: String,
) -> Result<()> {
    let request = DaemonRequest::SwitchModel {
        conversation_id,
        model_nickname: model_nickname.clone(),
    };

    client.send_request(&request).await?;

    let response = client.read_response().await?;

    match response {
        DaemonResponse::ConversationCreated { conversation } => {
            println!(
                "{} Switched to model {}",
                "✓".bright_green(),
                model_nickname.bright_yellow()
            );
            println!(
                "  {} {}",
                "Conversation:".dimmed(),
                conversation.id.bright_cyan()
            );
        }
        DaemonResponse::Error { message } => {
            eprintln!("{} {message}", "Error:".bright_red());
        }
        _ => {
            eprintln!("{} Unexpected response", "Error:".bright_red());
        }
    }

    Ok(())
}

/// Shows comprehensive status (daemon + current conversation).
pub async fn status(client: &mut DaemonClient, json: bool) -> Result<()> {
    let request = DaemonRequest::DetailedStatus;
    client.send_request(&request).await?;
    let response = client.read_response().await?;

    match response {
        DaemonResponse::Status {
            uptime_seconds,
            active_conversations,
            current_conversation,
        } => {
            if json {
                print_status_json(true, Some(uptime_seconds), active_conversations, current_conversation.as_ref())?;
            } else {
                print_status_human(uptime_seconds, active_conversations, current_conversation);
            }
        }
        DaemonResponse::Error { message } => {
            if json {
                eprintln!(r#"{{"error":"{}"}}"#, message.replace('"', r#"\""#));
            } else {
                eprintln!("{} {message}", "Error:".bright_red());
            }
        }
        _ => {
            if json {
                eprintln!(r#"{{"error":"Unexpected response"}}"#);
            } else {
                eprintln!("{} Unexpected response", "Error:".bright_red());
            }
        }
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

/// Formats duration as human-readable string (e.g., "1h 5m 23s").
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

/// Formats relative time (e.g., "5 minutes ago").
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
