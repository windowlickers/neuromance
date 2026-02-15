//! Command implementations for the CLI.

use std::io::Write;

use anyhow::Result;
use colored::Colorize;
use neuromance_common::protocol::{DaemonRequest, DaemonResponse};

use crate::client::DaemonClient;
use crate::display::{display_assistant_end, display_assistant_header, display_tool_result};

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

    // Read streaming responses
    client
        .read_until_complete(|response| {
            match response {
                DaemonResponse::StreamChunk { content, .. } => {
                    print!("{content}");
                    let _ = std::io::stdout().flush();
                    true // Continue
                }
                DaemonResponse::ToolResult {
                    tool_name,
                    result,
                    success,
                    ..
                } => {
                    display_tool_result(tool_name, result, *success);
                    true // Continue
                }
                DaemonResponse::Usage { usage, .. } => {
                    println!(
                        "\n{} {} tokens (in: {}, out: {})",
                        "○".bright_blue(),
                        usage.total_tokens,
                        usage.prompt_tokens,
                        usage.completion_tokens
                    );
                    true // Continue
                }
                DaemonResponse::MessageCompleted { .. } => {
                    display_assistant_end();
                    false // Done
                }
                DaemonResponse::Error { message } => {
                    eprintln!("\n{} {message}", "Error:".bright_red());
                    false // Done
                }
                _ => true, // Continue for other responses
            }
        })
        .await?;

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

            println!(
                "Showing {} of {} messages:",
                messages.len(),
                total_count
            );
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
pub async fn list_conversations(
    client: &mut DaemonClient,
    limit: Option<usize>,
) -> Result<()> {
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
        } => {
            println!("{} Daemon is running", "✓".bright_green());
            println!("  Uptime: {}s", uptime_seconds);
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
