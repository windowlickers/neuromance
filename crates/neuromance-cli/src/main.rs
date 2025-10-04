//! Neuromance CLI - Interactive REPL for LLM conversations
//!
//! This CLI provides a rustyline-based interface for managing conversations with LLMs.
//! Users can edit messages in the buffer and resubmit them to continue the conversation.
use std::io::Write;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use colored::Colorize;
use futures::StreamExt;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use secrecy::SecretString;

use neuromance::{ChatRequest, ToolChoice};
use neuromance::{
    Config, Conversation, Core, LLMClient, Message, MessageRole, OpenAIClient, ToolCall, Usage,
};
use neuromance_tools::{ThinkTool, ToolImplementation, create_todo_tools};

mod display;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Base URL for the API endpoint
    #[arg(long, default_value = "https://api.openai.com/v1")]
    base_url: String,

    /// API key for authentication (or set OPENAI_API_KEY env var)
    #[arg(long)]
    api_key: Option<String>,

    /// Model to use for chat completion
    #[arg(long, default_value = "gpt-4")]
    model: String,

    /// Path to MCP (Model Context Protocol) configuration file
    /// Supports .toml, .yaml, .yml, and .json formats
    /// See mcp_config.toml.example for configuration examples
    #[arg(long)]
    mcp_config: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize the LLM client
    let api_key = args
        .api_key
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .context("API key must be provided via --api-key or OPENAI_API_KEY env var")?;
    let api_key = SecretString::new(api_key.into());

    let mut config = Config::new("openai", &args.model);
    config.api_key = Some(api_key);
    config.base_url = Some(args.base_url.clone());

    let client = OpenAIClient::new(config)?;

    // Initialize Core with tools and streaming
    let mut core = Core::new(client);
    core.auto_approve_tools = true;
    core.streaming = true;

    // Register built-in tools
    let think_tool: Arc<dyn ToolImplementation> = Arc::new(ThinkTool);
    core.tool_executor.add_tool_arc(Arc::clone(&think_tool));

    let (todo_read, todo_write) = create_todo_tools();
    let todo_read_tool: Arc<dyn ToolImplementation> = Arc::new(todo_read);
    let todo_write_tool: Arc<dyn ToolImplementation> = Arc::new(todo_write);
    core.tool_executor.add_tool_arc(Arc::clone(&todo_read_tool));
    core.tool_executor
        .add_tool_arc(Arc::clone(&todo_write_tool));

    let registered_tools = ["think", "read_todos", "write_todos"];

    // Load MCP configuration if provided
    if let Some(mcp_config_path) = &args.mcp_config {
        use neuromance_tools::mcp::McpManager;
        use std::path::Path;

        println!("Loading MCP configuration from: {}", mcp_config_path);
        match McpManager::from_config_file(Path::new(mcp_config_path)).await {
            Ok(mcp_manager) => {
                // Get all tools from MCP servers
                match mcp_manager.get_all_tools().await {
                    Ok(mcp_tools) => {
                        let mcp_tool_count = mcp_tools.len();

                        for tool in mcp_tools {
                            core.tool_executor.add_tool_arc(tool);
                        }

                        // Get server status to count active servers
                        let status = mcp_manager.get_status().await;
                        let server_count = status
                            .values()
                            .filter(|s| {
                                matches!(s, neuromance_tools::mcp::ServerStatus::Connected { .. })
                            })
                            .count();

                        println!(
                            "Successfully loaded {} MCP tool(s) from {} server(s)",
                            mcp_tool_count, server_count
                        );
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load MCP tools: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Failed to load MCP configuration: {}", e);
                eprintln!("Continuing without MCP tools...");
            }
        }
    }

    println!("Registered tools: {}\n", registered_tools.join(", "));

    // Initialize conversation
    let mut conversation = Conversation::new()
        .with_title("CLI Chat Session")
        .with_description("Interactive conversation with LLM");

    // Add system message
    let system_msg =
        conversation.system_message("You are a helpful assistant. Be concise and informative.");
    conversation.add_message(system_msg)?;

    // Initialize rustyline editor
    let mut rl = DefaultEditor::new()?;
    let history_file = ".neuromance_history";

    // Load history if it exists
    if rl.load_history(history_file).is_err() {
        println!("No previous history found.");
    }

    println!("Neuromance CLI");
    println!("Commands:");
    println!("  /edit <index> - Edit message at index and resubmit from that point");
    println!("  /list         - List all messages in the conversation");
    println!("  /clear        - Clear conversation (keeps system message)");
    println!("  /quit         - Exit the CLI");
    println!();

    // Track usage stats
    let mut total_usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        cost: None,
        input_tokens_details: None,
        output_tokens_details: None,
    };

    loop {
        let prompt = format!(
            "\n╭─● {} [{}↑{} {}↓{} {}Σ{}]\n╰─○ ",
            "You".bright_cyan().bold(),
            "".cyan(),
            total_usage.prompt_tokens.to_string().cyan(),
            "".green(),
            total_usage.completion_tokens.to_string().green(),
            "".yellow(),
            total_usage.total_tokens.to_string().yellow()
        );
        let readline = rl.readline(&prompt);
        match readline {
            Ok(line) => {
                let line = line.trim();

                // Add to history
                rl.add_history_entry(line)?;

                // Handle commands
                if line.starts_with('/') {
                    if handle_command(
                        line,
                        &mut conversation,
                        &mut rl,
                        &mut core,
                        &mut total_usage,
                    )
                    .await?
                    {
                        break; // Exit if command returns true
                    }
                    continue;
                }

                // Skip empty input
                if line.is_empty() {
                    continue;
                }

                // Add user message
                let user_msg = conversation.user_message(line);
                conversation.add_message(user_msg)?;

                // Send to LLM with tool execution support
                match execute_with_tools(&mut core, &mut conversation).await {
                    Ok(usage) => {
                        if let Some(u) = usage {
                            total_usage.prompt_tokens += u.prompt_tokens;
                            total_usage.completion_tokens += u.completion_tokens;
                            total_usage.total_tokens += u.total_tokens;
                        }
                    }
                    Err(e) => {
                        eprintln!("Error: {}", e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }

    // Save history
    rl.save_history(history_file)?;
    println!("Goodbye!");

    Ok(())
}

/// Handle CLI commands
async fn handle_command(
    command: &str,
    conversation: &mut Conversation,
    rl: &mut DefaultEditor,
    core: &mut Core<OpenAIClient>,
    total_usage: &mut Usage,
) -> Result<bool> {
    let parts: Vec<&str> = command.split_whitespace().collect();

    match parts.first().copied() {
        Some("/quit") | Some("/exit") => {
            return Ok(true);
        }
        Some("/list") => {
            list_messages(conversation);
        }
        Some("/clear") => {
            clear_conversation(conversation)?;
            // Reset usage stats
            total_usage.prompt_tokens = 0;
            total_usage.completion_tokens = 0;
            total_usage.total_tokens = 0;
        }
        Some("/edit") => {
            if let Some(index_str) = parts.get(1) {
                let index: usize = index_str.parse().context("Index must be a number")?;
                edit_message(index, conversation, rl, core).await?;
            } else {
                println!("Usage: /edit <index>");
            }
        }
        _ => {
            println!("Unknown command. Available commands: /list, /edit, /clear, /quit");
        }
    }

    Ok(false)
}

/// List all messages in the conversation
fn list_messages(conversation: &Conversation) {
    println!("\n=== Conversation Messages ===\n");
    for (idx, msg) in conversation.get_messages().iter().enumerate() {
        let (role, color_fn): (&str, fn(&str) -> colored::ColoredString) = match msg.role {
            MessageRole::System => ("System", |s: &str| s.bright_black()),
            MessageRole::User => ("User", |s: &str| s.bright_cyan()),
            MessageRole::Assistant => ("Assistant", |s: &str| s.bright_magenta()),
            MessageRole::Tool => ("Tool", |s: &str| s.bright_yellow()),
            _ => ("Unknown", |s: &str| s.white()),
        };

        let content = &msg.content;
        println!("[{}] {}: {}", idx, color_fn(role), content);

        // Display tool calls for assistant messages
        if !msg.tool_calls.is_empty() {
            for tc in &msg.tool_calls {
                println!("      ├─○ Tool Call: {}", tc.function.name.bright_green());
            }
        }

        // Display tool metadata for tool messages
        if msg.role == MessageRole::Tool
            && let Some(tool_name) = &msg.name
        {
            println!("      ├─✓ Tool Result: {}", tool_name.bright_green());
        }
    }
    println!();
}

/// Clear conversation but keep system message
fn clear_conversation(conversation: &mut Conversation) -> Result<()> {
    let system_msg_content = conversation
        .get_messages()
        .iter()
        .find(|m| matches!(m.role, MessageRole::System))
        .map(|m| m.content.clone());

    *conversation = Conversation::new()
        .with_title("CLI Chat Session")
        .with_description("Interactive conversation with LLM");

    if let Some(content) = system_msg_content {
        let new_system_msg = conversation.system_message(&content);
        conversation.add_message(new_system_msg)?;
    }

    println!("Conversation cleared.");
    Ok(())
}

/// Edit a message and resubmit from that point
async fn edit_message(
    index: usize,
    conversation: &mut Conversation,
    rl: &mut DefaultEditor,
    core: &mut Core<OpenAIClient>,
) -> Result<()> {
    let messages = conversation.get_messages();

    if index >= messages.len() {
        println!("Invalid index. Use /list to see all messages.");
        return Ok(());
    }

    let msg = &messages[index];

    // Only allow editing user and assistant messages
    if !matches!(msg.role, MessageRole::User | MessageRole::Assistant) {
        println!("Can only edit User or Assistant messages.");
        return Ok(());
    }

    let current_content = &msg.content;

    // Pre-fill the editor with current content
    println!(
        "Editing message [{}]. Press Enter to accept or modify:",
        index
    );
    let edited = rl.readline_with_initial(">> ", (current_content, ""))?;
    let edited = edited.trim();

    if edited.is_empty() {
        println!("Edit cancelled (empty input).");
        return Ok(());
    }

    // Truncate conversation at the edit point
    let mut new_messages: Vec<Message> = messages.iter().take(index).cloned().collect();

    // Add the edited message
    let edited_msg = match msg.role {
        MessageRole::User => conversation.user_message(edited),
        MessageRole::Assistant => conversation.assistant_message(edited),
        _ => unreachable!(),
    };

    new_messages.push(edited_msg);

    // Rebuild conversation
    *conversation = Conversation::new()
        .with_title("CLI Chat Session")
        .with_description("Interactive conversation with LLM");

    for msg in new_messages {
        conversation.add_message(msg)?;
    }

    println!("\nMessage edited. Resubmitting...\n");

    // Resubmit to LLM with tool execution
    match execute_with_tools(core, conversation).await {
        Ok(_usage) => {
            // Messages already displayed by execute_with_tools
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}

/// Execute conversation with Core's tool loop, displaying results with streaming
async fn execute_with_tools(
    core: &mut Core<OpenAIClient>,
    conversation: &mut Conversation,
) -> Result<Option<Usage>> {
    let mut total_usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        cost: None,
        input_tokens_details: None,
        output_tokens_details: None,
    };

    let max_turns = 10;
    let mut turn = 0;

    loop {
        turn += 1;
        if turn > max_turns {
            display::display_warning(&format!(
                "Maximum tool execution turns ({}) reached. Response may be incomplete.",
                max_turns
            ));
            break;
        }

        // Create request
        let request =
            ChatRequest::from((core.client.config(), conversation.get_messages().to_vec()))
                .with_tools(core.tool_executor.get_all_tools())
                .with_tool_choice(ToolChoice::Auto);

        // Stream the response
        let mut stream = core.client.chat_stream(&request).await?;
        // Pre-allocate capacity for typical streaming responses
        // Average LLM response is ~200-500 chars, allocate for 1KB to reduce reallocations
        let mut accumulated_content = String::with_capacity(1024);
        // Most responses have 0-3 tool calls, pre-allocate for 4 to avoid most reallocations
        let mut tool_calls = Vec::with_capacity(4);
        let mut usage_info = None;
        let mut has_shown_header = false;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;

            // Display content as it streams
            if let Some(content) = &chunk.delta_content {
                if !has_shown_header {
                    display::display_assistant_header();
                    has_shown_header = true;
                }
                print!("{}", content);
                std::io::stdout().flush()?;
                accumulated_content.push_str(content);
            }

            // Accumulate tool calls
            if let Some(delta_calls) = &chunk.delta_tool_calls {
                tool_calls = ToolCall::merge_deltas(tool_calls, delta_calls);
            }

            // Capture usage
            if let Some(usage) = chunk.usage {
                usage_info = Some(usage);
            }
        }

        // Update total usage
        if let Some(usage) = usage_info {
            total_usage.prompt_tokens += usage.prompt_tokens;
            total_usage.completion_tokens += usage.completion_tokens;
            total_usage.total_tokens += usage.total_tokens;
        }

        // Create assistant message
        let assistant_msg = conversation.assistant_message(&accumulated_content);
        let assistant_msg_with_tools = if !tool_calls.is_empty() {
            assistant_msg.with_tool_calls(tool_calls.clone())?
        } else {
            assistant_msg
        };
        conversation.add_message(assistant_msg_with_tools)?;

        // If no tool calls, end the branch and we're done
        if tool_calls.is_empty() {
            if has_shown_header {
                display::display_assistant_end();
            }
            break;
        }

        // Display and execute tool calls (continues the branch)
        for tool_call in &tool_calls {
            display::display_tool_call_request(tool_call);

            // Execute the tool (auto-approved in this CLI)
            match core.tool_executor.execute_tool(tool_call).await {
                Ok(result) => {
                    display::display_tool_result(&tool_call.function.name, &result, true);

                    let tool_msg = Message::tool(
                        conversation.id,
                        result,
                        tool_call.id.clone(),
                        tool_call.function.name.clone(),
                    )?;
                    conversation.add_message(tool_msg)?;
                }
                Err(e) => {
                    let error_msg = format!("Tool execution failed: {}", e);
                    display::display_tool_result(&tool_call.function.name, &error_msg, false);

                    let tool_msg = Message::tool(
                        conversation.id,
                        error_msg,
                        tool_call.id.clone(),
                        tool_call.function.name.clone(),
                    )?;
                    conversation.add_message(tool_msg)?;
                }
            }
        }
    }

    Ok(Some(total_usage))
}
