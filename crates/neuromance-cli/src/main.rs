//! Neuromance CLI - Interactive REPL for LLM conversations
//!
//! This CLI provides a rustyline-based interface for managing conversations with LLMs.
use std::io::Write;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use colored::Colorize;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use secrecy::SecretString;
use tokio::sync::Mutex;

use neuromance::{
    AnthropicClient, Config, Conversation, Core, CoreEvent, LLMClient, Message, MessageRole,
    OpenAIClient, ToolApproval, ToolCall, Usage,
};
use neuromance_tools::{ThinkTool, ToolImplementation, create_todo_tools};

mod display;

/// Supported LLM providers
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum Provider {
    /// `OpenAI` API
    #[default]
    Openai,
    /// Anthropic Messages API
    Anthropic,
}

impl Provider {
    const fn default_base_url(self) -> &'static str {
        match self {
            Self::Openai => "https://api.openai.com/v1",
            Self::Anthropic => "https://api.anthropic.com/v1",
        }
    }

    const fn default_model(self) -> &'static str {
        match self {
            Self::Openai => "gpt-4o",
            Self::Anthropic => "claude-sonnet-4-5-20250929",
        }
    }

    const fn api_key_env_var(self) -> &'static str {
        match self {
            Self::Openai => "OPENAI_API_KEY",
            Self::Anthropic => "ANTHROPIC_API_KEY",
        }
    }

    const fn config_name(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// LLM provider to use
    #[arg(long, short, value_enum, default_value_t = Provider::Openai)]
    provider: Provider,

    /// Base URL for the API endpoint
    #[arg(long)]
    base_url: Option<String>,

    /// API key for authentication (or set via provider-specific env var)
    #[arg(long)]
    api_key: Option<String>,

    /// Model to use for chat completion
    #[arg(long)]
    model: Option<String>,

    /// Extended thinking budget in tokens (Anthropic Claude models only)
    /// Enables Claude's extended thinking capability with the specified token budget
    #[arg(long)]
    thinking_budget: Option<u32>,

    /// Enable interleaved thinking between tool calls (Anthropic Claude 4+ models)
    /// Allows Claude to reason after receiving tool results
    #[arg(long)]
    interleaved_thinking: bool,

    /// Path to MCP (Model Context Protocol) configuration file
    /// Supports .toml, .yaml, .yml, and .json formats
    /// See `mcp_config.toml.example` for configuration examples
    #[arg(long)]
    mcp_config: Option<String>,
}

#[tokio::main]
#[allow(clippy::too_many_lines)]
async fn main() -> Result<()> {
    let args = Args::parse();
    let provider = args.provider;

    // Build configuration from provider defaults and CLI overrides
    let api_key = args
        .api_key
        .clone()
        .or_else(|| std::env::var(provider.api_key_env_var()).ok())
        .with_context(|| {
            format!(
                "API key must be provided via --api-key or {} env var",
                provider.api_key_env_var()
            )
        })?;
    let api_key = SecretString::new(api_key.into());

    let model = args
        .model
        .clone()
        .unwrap_or_else(|| provider.default_model().to_string());
    let base_url = args
        .base_url
        .clone()
        .unwrap_or_else(|| provider.default_base_url().to_string());

    let mut config = Config::new(provider.config_name(), &model);
    config.api_key = Some(api_key);
    config.base_url = Some(base_url);

    match provider {
        Provider::Anthropic => {
            let client = AnthropicClient::new(config)?;
            run_cli(client, &args).await
        }
        Provider::Openai => {
            let client = OpenAIClient::new(config)?;
            run_cli(client, &args).await
        }
    }
}

/// Run the CLI with the given LLM client
#[allow(clippy::too_many_lines)]
async fn run_cli<C: LLMClient>(client: C, args: &Args) -> Result<()> {
    // tool approval editor, locks when prompting user to approve/deny tools
    let approval_editor = Arc::new(Mutex::new(DefaultEditor::new()?));

    // usage tracker, updates via event callback
    let total_usage = Arc::new(Mutex::new(Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        cost: None,
        input_tokens_details: None,
        output_tokens_details: None,
    }));

    // initialize neuromance core
    //
    // NOTE: callbacks use async closure pattern for shared state
    // `move |arg| { let captured = Arc::clone(&captured); async move { ... } }`
    let mut core = Core::new(client)
        .with_streaming()
        .with_tool_approval_callback({
            let editor = Arc::clone(&approval_editor);
            move |tool_call: &ToolCall| {
                // clone ToolCall and move into async block to avoid lifetime issues
                let tool_call = tool_call.clone();
                let editor = Arc::clone(&editor);
                async move { prompt_for_tool_approval(&tool_call, &editor).await }
            }
        })
        .with_event_callback({
            let usage = Arc::clone(&total_usage);
            move |event: CoreEvent| {
                // Arc::clone inside closure for each invocation
                let usage = Arc::clone(&usage);
                async move {
                    match event {
                        CoreEvent::Streaming(chunk) => {
                            print!("{chunk}");
                            std::io::stdout().flush().ok();
                        }
                        CoreEvent::ToolResult {
                            name,
                            result,
                            success,
                        } => {
                            display::display_tool_result(&name, &result, success);
                        }
                        CoreEvent::Usage(u) => {
                            // await async lock so we can update usage
                            let mut total = usage.lock().await;
                            total.prompt_tokens += u.prompt_tokens;
                            total.completion_tokens += u.completion_tokens;
                            total.total_tokens += u.total_tokens;
                        }
                    }
                }
            }
        });

    // Apply thinking configuration from CLI args
    if let Some(budget) = args.thinking_budget {
        core = core.with_thinking_budget(budget);
    }
    if args.interleaved_thinking {
        core = core.with_interleaved_thinking();
    }

    core.auto_approve_tools = false;
    core.max_turns = Some(10);

    // tool registration
    let think_tool: Arc<dyn ToolImplementation> = Arc::new(ThinkTool);
    core.tool_executor.add_tool_arc(Arc::clone(&think_tool));

    let (todo_read, todo_write) = create_todo_tools();
    let todo_read_tool: Arc<dyn ToolImplementation> = Arc::new(todo_read);
    let todo_write_tool: Arc<dyn ToolImplementation> = Arc::new(todo_write);
    core.tool_executor.add_tool_arc(Arc::clone(&todo_read_tool));
    core.tool_executor
        .add_tool_arc(Arc::clone(&todo_write_tool));

    let registered_tools = ["think", "read_todos", "write_todos"];

    // load MCP conf if provided
    if let Some(mcp_config_path) = &args.mcp_config {
        use neuromance_tools::mcp::McpManager;
        use std::path::Path;

        println!("Loading MCP configuration from: {mcp_config_path}");
        match McpManager::from_config_file(Path::new(mcp_config_path)).await {
            Ok(mcp_manager) => match mcp_manager.get_all_tools().await {
                Ok(mcp_tools) => {
                    let mcp_tool_count = mcp_tools.len();

                    for tool in mcp_tools {
                        core.tool_executor.add_tool_arc(tool);
                    }

                    let status = mcp_manager.get_status().await;
                    let server_count = status
                        .values()
                        .filter(|s| {
                            matches!(s, neuromance_tools::mcp::ServerStatus::Connected { .. })
                        })
                        .count();

                    println!(
                        "Successfully loaded {mcp_tool_count} MCP tool(s) from {server_count} server(s)"
                    );
                }
                Err(e) => {
                    eprintln!("Warning: Failed to load MCP tools: {e}");
                }
            },
            Err(e) => {
                eprintln!("Warning: Failed to load MCP configuration: {e}");
                eprintln!("Continuing without MCP tools...");
            }
        }
    }

    println!("Registered tools: {}\n", registered_tools.join(", "));

    // initialize conversation
    let mut conversation = Conversation::new()
        .with_title("CLI Chat Session")
        .with_description("Interactive conversation with LLM");

    // set system message
    let system_msg =
        conversation.system_message("You are a helpful assistant. Be concise and informative.");
    conversation.add_message(system_msg)?;

    // initialize rustyline editor
    let mut rl = DefaultEditor::new()?;
    let history_file = ".neuromance_history";

    // load history if it exists
    if rl.load_history(history_file).is_err() {
        println!("No previous history found.");
    }

    // print usage menu
    // TODO: put this in display.rs
    println!("Neuromance CLI");
    println!("Commands:");
    println!("  /edit <index> - Edit message at index and resubmit from that point");
    println!("  /list         - List all messages in the conversation");
    println!("  /clear        - Clear conversation (keeps system message)");
    println!("  /quit         - Exit the CLI");
    println!();
    println!("Tool Approval:");
    println!("  When a tool is requested, you'll be prompted to approve:");
    println!("  /yes or /y    - Approve the tool execution");
    println!("  /no or /n     - Deny the tool execution");
    println!("  /quit or /q   - Abort the entire conversation");
    println!();

    // core cli loop
    loop {
        let (prompt_tokens, completion_tokens, total_tokens) = {
            let usage = total_usage.lock().await;
            (
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
        };
        // PS1
        // ╭─● You [↑0 ↓0 Σ0]
        // ╰─○
        // TODO: put this in display.rs
        let prompt = format!(
            "\n╭─● {} [{}↑{} {}↓{} {}Σ{}]\n╰─○ ",
            "You".bright_cyan().bold(),
            "".cyan(),
            prompt_tokens.to_string().cyan(),
            "".green(),
            completion_tokens.to_string().green(),
            "".yellow(),
            total_tokens.to_string().yellow()
        );
        let readline = rl.readline(&prompt);
        match readline {
            Ok(line) => {
                let line = line.trim();

                // add to history
                rl.add_history_entry(line)?;

                // handle commands
                if line.starts_with('/') {
                    if handle_command(line, &mut conversation, &mut rl, &core, &total_usage).await?
                    {
                        break; // exit if command returns true
                    }
                    continue;
                }

                // empty input skip
                if line.is_empty() {
                    continue;
                }

                // add user message
                let user_msg = conversation.user_message(line);
                conversation.add_message(user_msg)?;

                // send conversation using neuromance core
                match execute_conversation(&core, &mut conversation).await {
                    Ok(()) => {
                        // Successfully completed - messages already in conversation
                    }
                    Err(e) => {
                        eprintln!("\n{} {}", "Error:".red().bold(), e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
            }
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                break;
            }
            Err(err) => {
                eprintln!("Error: {err:?}");
                break;
            }
        }
    }

    // save history
    rl.save_history(history_file)?;
    println!("Goodbye!");

    Ok(())
}

/// Handle CLI commands
async fn handle_command<C: LLMClient>(
    command: &str,
    conversation: &mut Conversation,
    rl: &mut DefaultEditor,
    core: &Core<C>,
    total_usage: &Arc<Mutex<Usage>>,
) -> Result<bool> {
    let parts: Vec<&str> = command.split_whitespace().collect();

    match parts.first().copied() {
        Some("/quit" | "/exit") => {
            return Ok(true);
        }
        Some("/list") => {
            list_messages(conversation);
        }
        Some("/clear") => {
            clear_conversation(conversation)?;
            // Reset usage stats
            let mut usage = total_usage.lock().await;
            usage.prompt_tokens = 0;
            usage.completion_tokens = 0;
            usage.total_tokens = 0;
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
async fn edit_message<C: LLMClient>(
    index: usize,
    conversation: &mut Conversation,
    rl: &mut DefaultEditor,
    core: &Core<C>,
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
    println!("Editing message [{index}]. Press Enter to accept or modify:");
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
    match execute_conversation(core, conversation).await {
        Ok(()) => {
            // Successfully completed
        }
        Err(e) => {
            eprintln!("\n{} {}", "Error:".red().bold(), e);
        }
    }

    Ok(())
}

/// Prompt user for tool approval with `/yes`, `/no`, or `/quit` commands
///
/// Mutex is safe here: only held during user I/O, Core doesn't re-enter while waiting.
/// Timeout provides failsafe against unexpected deadlocks.
async fn prompt_for_tool_approval(
    tool_call: &ToolCall,
    editor: &Arc<Mutex<DefaultEditor>>,
) -> ToolApproval {
    // Display the tool request
    display::display_tool_call_request(tool_call);

    // Parse arguments for display
    let args_display = if tool_call.function.arguments.is_empty() {
        "{}".to_string()
    } else if tool_call.function.arguments.len() == 1 {
        tool_call.function.arguments[0].clone()
    } else {
        format!("[{}]", tool_call.function.arguments.join(", "))
    };

    println!(
        "\n{} Execute tool '{}' with arguments: {}",
        "Tool Approval Required:".yellow().bold(),
        tool_call.function.name.bright_green(),
        args_display.bright_cyan()
    );
    println!(
        "{} /yes or /y to approve, /no or /n to deny, /quit or /q to abort",
        "→".yellow()
    );

    // Acquire lock with timeout as defensive measure
    let Ok(mut ed) = tokio::time::timeout(std::time::Duration::from_secs(30), editor.lock()).await
    else {
        eprintln!("{} Editor lock timeout", "!".red().bold());
        return ToolApproval::Denied("Editor lock timeout".to_string());
    };

    loop {
        let prompt = format!("{} ", "Approve?".yellow().bold());
        match ed.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();
                ed.add_history_entry(line).ok();

                match line {
                    "/yes" | "/y" => {
                        println!("{} Tool approved\n", "✓".green().bold());
                        return ToolApproval::Approved;
                    }
                    "/no" | "/n" => {
                        println!("{} Tool denied\n", "✗".red().bold());
                        return ToolApproval::Denied("User denied tool execution".to_string());
                    }
                    "/quit" | "/q" => {
                        println!("{} Aborting conversation\n", "!".red().bold());
                        return ToolApproval::Quit;
                    }
                    _ => {
                        println!(
                            "{} Invalid response. Use /yes, /no, or /quit",
                            "!".yellow().bold()
                        );
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("{} Tool denied (CTRL-C)\n", "✗".red().bold());
                return ToolApproval::Denied("User interrupted with CTRL-C".to_string());
            }
            Err(ReadlineError::Eof) => {
                println!("{} Aborting conversation (CTRL-D)\n", "!".red().bold());
                return ToolApproval::Quit;
            }
            Err(e) => {
                eprintln!("Error reading input: {e}");
                return ToolApproval::Denied(format!("Error reading input: {e}"));
            }
        }
    }
}

/// Execute conversation using Core's built-in tool loop with streaming
async fn execute_conversation<C: LLMClient>(
    core: &Core<C>,
    conversation: &mut Conversation,
) -> Result<()> {
    // Display assistant header before streaming starts
    display::display_assistant_header();

    // Use Core's chat_with_tool_loop which handles:
    // - Streaming with callback
    // - Tool approval via callback
    // - Tool execution
    // - Max turns checking
    let messages = core
        .chat_with_tool_loop(conversation.get_messages().to_vec())
        .await?;

    // Update conversation with all messages from the loop
    // The tool loop returns all messages including assistant responses and tool results
    // We need to add only the new ones (skip messages already in conversation)
    let existing_count = conversation.get_messages().len();
    for msg in messages.into_iter().skip(existing_count) {
        conversation.add_message(msg)?;
    }

    display::display_assistant_end();

    Ok(())
}
