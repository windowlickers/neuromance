//! Neuromance CLI - Lightweight client for the Neuromance daemon.

mod client;
mod commands;
mod display;
mod repl;
mod theme;

use anyhow::Result;
use clap::{Parser, Subcommand};

use client::DaemonClient;

#[derive(Parser)]
#[command(name = "nm")]
#[command(about = "Neuromance CLI - Lightweight LLM conversation client", long_about = None)]
struct Cli {
    /// Conversation ID (bookmark, full UUID, or short hash)
    #[arg(short, long, global = true)]
    conversation: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,

    /// Message to send (shorthand for `nm send "message"`)
    message: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Send a message to a conversation
    Send {
        /// The message content
        message: String,
    },

    /// Enter REPL mode for interactive conversation
    #[command(alias = "r")]
    Repl,

    /// List messages from a conversation
    #[command(alias = "ls")]
    Messages {
        /// Maximum number of messages to show
        #[arg(short = 'n', long)]
        limit: Option<usize>,
    },

    /// Create a new conversation
    #[command(alias = "n")]
    New {
        /// Model nickname to use
        #[arg(short, long)]
        model: Option<String>,

        /// System message to initialize with
        #[arg(short, long)]
        system: Option<String>,
    },

    /// Delete a conversation
    #[command(alias = "rm")]
    Delete {
        /// Conversation ID (bookmark, full UUID, or short hash)
        conversation: String,

        /// Skip confirmation prompt
        #[arg(short, long)]
        force: bool,
    },

    /// Delete old or empty conversations in bulk
    Prune {
        /// Delete ALL conversations, not just empty ones
        #[arg(long)]
        all: bool,

        /// Skip confirmation prompt
        #[arg(short, long)]
        force: bool,
    },

    /// List all conversations
    #[command(alias = "convs", alias = "c")]
    Conversations {
        /// Maximum number of conversations to show
        #[arg(short = 'n', long)]
        limit: Option<usize>,
    },

    /// Manage bookmarks
    #[command(alias = "b")]
    Bookmark {
        #[command(subcommand)]
        action: BookmarkAction,
    },

    /// Manage models
    #[command(alias = "m")]
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },

    /// Show current status (daemon + active conversation)
    #[command(alias = "st")]
    Status {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Daemon management
    #[command(alias = "d")]
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },
}

#[derive(Subcommand)]
enum BookmarkAction {
    /// Set a bookmark for a conversation
    Set {
        /// Conversation ID
        conversation_id: String,
        /// Bookmark name
        name: String,
    },

    /// Remove a bookmark
    Remove {
        /// Bookmark name
        name: String,
    },
}

#[derive(Subcommand)]
enum ModelAction {
    /// List available models
    #[command(alias = "ls")]
    List,

    /// Switch to a different model for the current conversation
    Switch {
        /// Model nickname to switch to
        model_nickname: String,
    },
}

#[derive(Subcommand)]
enum DaemonAction {
    /// Show daemon status
    Status,

    /// Check daemon health and version compatibility
    Health,

    /// Shutdown the daemon
    Stop,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();
    let theme = theme::Theme::load();

    // Handle shorthand: `nm "message"`
    if let Some(message) = cli.message {
        let mut client = DaemonClient::connect().await?;
        commands::send_message(&mut client, cli.conversation, message, &theme).await?;
        return Ok(());
    }

    // Handle commands
    match cli.command {
        Some(Command::Send { message }) => {
            let mut client = DaemonClient::connect().await?;
            commands::send_message(&mut client, cli.conversation, message, &theme).await?;
        }

        Some(Command::Repl) => {
            repl::run_repl(cli.conversation, &theme).await?;
        }

        Some(Command::Messages { limit }) => {
            let mut client = DaemonClient::connect().await?;
            commands::list_messages(&mut client, cli.conversation, limit).await?;
        }

        Some(Command::New { model, system }) => {
            let mut client = DaemonClient::connect().await?;
            commands::new_conversation(&mut client, model, system).await?;
        }

        Some(Command::Delete {
            conversation,
            force,
        }) => {
            let mut client = DaemonClient::connect().await?;
            commands::delete_conversation(&mut client, conversation, force).await?;
        }

        Some(Command::Prune { all, force }) => {
            let mut client = DaemonClient::connect().await?;
            commands::prune_conversations(&mut client, all, force).await?;
        }

        Some(Command::Conversations { limit }) => {
            let mut client = DaemonClient::connect().await?;
            commands::list_conversations(&mut client, limit).await?;
        }

        Some(Command::Bookmark { action }) => {
            let mut client = DaemonClient::connect().await?;
            match action {
                BookmarkAction::Set {
                    conversation_id,
                    name,
                } => {
                    commands::set_bookmark(&mut client, conversation_id, name).await?;
                }
                BookmarkAction::Remove { name } => {
                    commands::remove_bookmark(&mut client, name).await?;
                }
            }
        }

        Some(Command::Model { action }) => {
            let mut client = DaemonClient::connect().await?;
            match action {
                ModelAction::List => {
                    commands::list_models(&mut client).await?;
                }
                ModelAction::Switch { model_nickname } => {
                    commands::switch_model(&mut client, cli.conversation, model_nickname).await?;
                }
            }
        }

        Some(Command::Status { json }) => {
            match DaemonClient::connect().await {
                Ok(mut client) => {
                    commands::status(&mut client, json).await?;
                }
                Err(_) => {
                    // Daemon not running
                    commands::status_daemon_not_running(json)?;
                }
            }
        }

        Some(Command::Daemon { action }) => {
            let mut client = DaemonClient::connect().await?;
            match action {
                DaemonAction::Status => {
                    commands::daemon_status(&mut client).await?;
                }
                DaemonAction::Health => {
                    commands::daemon_health(&mut client).await?;
                }
                DaemonAction::Stop => {
                    commands::shutdown_daemon(&mut client).await?;
                }
            }
        }

        None => {
            // No command specified, enter REPL
            repl::run_repl(cli.conversation, &theme).await?;
        }
    }

    Ok(())
}
