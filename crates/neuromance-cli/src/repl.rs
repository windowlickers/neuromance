//! REPL mode for interactive conversations.

use anyhow::Result;
use colored::Colorize;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use crate::client::DaemonClient;
use crate::commands::send_message;

/// Runs the REPL loop.
///
/// # Errors
///
/// Returns an error if the REPL initialization or message sending fails.
pub async fn run_repl(conversation_id: Option<String>) -> Result<()> {
    println!("{}", "Neuromance REPL".bright_magenta().bold());
    println!("{}", "Ctrl-D to exit".dimmed());
    println!();

    let mut rl = DefaultEditor::new()?;
    let mut client = DaemonClient::connect().await?;

    loop {
        let readline = rl.readline(&format!("{} ", ">".bright_green()));

        match readline {
            Ok(line) => {
                let line = line.trim();

                if line.is_empty() {
                    continue;
                }

                // Add to history
                let _ = rl.add_history_entry(line);

                // Send message
                if let Err(e) = send_message(&mut client, conversation_id.clone(), line.to_string()).await {
                    eprintln!("{} {e}", "Error:".bright_red());
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
