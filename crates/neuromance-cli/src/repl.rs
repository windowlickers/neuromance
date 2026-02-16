//! REPL mode for interactive conversations.

use anyhow::Result;
use colored::Colorize;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use crate::client::DaemonClient;
use crate::commands::send_message;
use crate::theme::Theme;

/// Runs the REPL loop.
///
/// # Errors
///
/// Returns an error if the REPL initialization or message sending fails.
#[allow(clippy::significant_drop_tightening, clippy::future_not_send)]
pub async fn run_repl(initial_conversation_id: Option<String>, theme: &Theme) -> Result<()> {
    println!("{}", theme.repl_title.render(&[]));
    println!("{}", theme.repl_subtitle.render(&[]));
    println!();

    let mut rl = DefaultEditor::new()?;
    let mut client = DaemonClient::connect().await?;
    let mut conversation_id = initial_conversation_id;

    let prompt = theme.prompt_user.render(&[]);

    loop {
        let readline = rl.readline(&prompt);

        match readline {
            Ok(line) => {
                let line = line.trim();

                if line.is_empty() {
                    continue;
                }

                // Handle REPL commands
                if let Some(cmd) = line.strip_prefix(':') {
                    let parts: Vec<&str> = cmd.split_whitespace().collect();
                    match parts.as_slice() {
                        ["switch", model_nickname] => {
                            if let Err(e) = crate::commands::switch_model(
                                &mut client,
                                conversation_id.clone(),
                                model_nickname.to_string(),
                            )
                            .await
                            {
                                eprintln!("{} {e}", "Error:".bright_red());
                            }
                        }
                        ["help"] => {
                            println!("{}", "REPL Commands:".bright_cyan().bold());
                            println!(
                                "  {} - Switch to a different model",
                                ":switch <model>".bright_yellow()
                            );
                            println!("  {} - Show this help message", ":help".bright_yellow());
                            println!("  {} - Exit the REPL", "Ctrl-D".bright_yellow());
                            println!();
                        }
                        _ => {
                            eprintln!(
                                "{} Unknown command: :{}\nType :help for available commands",
                                "Error:".bright_red(),
                                parts.join(" ")
                            );
                        }
                    }
                    continue;
                }

                // Add to history
                let _ = rl.add_history_entry(line);

                // Send message, attempting reconnection on failure
                match send_message(
                    &mut client,
                    conversation_id.clone(),
                    line.to_string(),
                    theme,
                )
                .await
                {
                    Ok(resolved_id) => {
                        // Pin to the resolved conversation for all
                        // subsequent messages in this REPL session
                        if resolved_id.is_some() {
                            conversation_id = resolved_id;
                        }
                    }
                    Err(e) => {
                        eprintln!("{} {e}", "Error:".bright_red());
                        eprintln!("{}", "Attempting to reconnect...".dimmed());
                        match DaemonClient::connect().await {
                            Ok(new_client) => {
                                client = new_client;
                                eprintln!("{} Reconnected", "✓".bright_green());
                            }
                            Err(reconnect_err) => {
                                eprintln!(
                                    "{} Failed to reconnect: {reconnect_err}",
                                    "Error:".bright_red()
                                );
                            }
                        }
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("Interrupted");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("Exiting");
                break;
            }
            Err(err) => {
                eprintln!("Error: {err}");
                break;
            }
        }
    }

    Ok(())
}
