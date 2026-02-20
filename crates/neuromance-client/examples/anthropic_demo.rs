//! Anthropic Claude API Demo
//!
//! This example demonstrates using the Anthropic client with tool calling
//! and a multi-turn conversation loop, plus optional streaming.
//!
//! # Usage
//!
//! ```bash
//! # Set your API key
//! export ANTHROPIC_API_KEY="sk-ant-..."
//!
//! # Run with defaults (streaming, triggers tool use)
//! cargo run --example anthropic_demo
//!
//! # Run with custom message
//! cargo run --example anthropic_demo -- \
//!     --message "Add a todo to water the plants"
//!
//! # Run non-streaming
//! cargo run --example anthropic_demo -- --no-stream
//!
//! # Use a specific model
//! cargo run --example anthropic_demo -- \
//!     --model claude-haiku-4-5-20251001
//! ```

use std::collections::HashMap;
use std::io::Write;

use anyhow::Result;
use clap::Parser;
use futures::StreamExt;
use log::info;
use serde::Deserialize;
use uuid::Uuid;

use neuromance_client::{AnthropicClient, LLMClient};
use neuromance_common::{
    ChatRequest, Config, Function, Message, Parameters, Property, Tool, ToolCall, ToolChoice,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Anthropic Claude API Demo")]
struct Args {
    /// API key (or set `ANTHROPIC_API_KEY` env var)
    #[arg(long, env = "ANTHROPIC_API_KEY")]
    api_key: String,

    /// Model to use for chat completion
    #[arg(long, default_value = "claude-sonnet-4-5-20250929")]
    model: String,

    /// Maximum tokens to generate
    #[arg(long, default_value = "1024")]
    max_tokens: u32,

    /// The user message to send
    #[arg(
        long,
        default_value = "Add three todos: buy groceries (high priority), \
            call the dentist, and read a book (low priority). \
            Then list them all and mark the second one as done."
    )]
    message: String,

    /// Disable streaming (use non-streaming API)
    #[arg(long)]
    no_stream: bool,

    /// Temperature for sampling (0.0-1.0)
    #[arg(long)]
    temperature: Option<f32>,
}

// -- Todo state -----------------------------------------------------------

struct TodoItem {
    title: String,
    priority: String,
    done: bool,
}

// -- Arg structs ----------------------------------------------------------

#[derive(Deserialize)]
struct AddTodoArgs {
    title: String,
    #[serde(default = "default_priority")]
    priority: String,
}

fn default_priority() -> String {
    "medium".to_string()
}

#[derive(Deserialize)]
struct ListTodosArgs {
    #[serde(default)]
    include_completed: bool,
}

#[derive(Deserialize)]
struct CompleteTodoArgs {
    index: usize,
}

// -- Tool definitions -----------------------------------------------------

fn create_add_todo_tool() -> Tool {
    let mut props = HashMap::new();
    props.insert(
        "title".to_string(),
        Property::string("The title of the todo item"),
    );
    props.insert(
        "priority".to_string(),
        Property::string_enum(
            "Priority level (defaults to medium)",
            vec!["low", "medium", "high"],
        ),
    );

    Tool {
        r#type: "function".to_string(),
        function: Function {
            name: "add_todo".to_string(),
            description: "Add a new todo item to the list.".to_string(),
            parameters: Parameters::new(props, vec!["title".to_string()]).into(),
        },
    }
}

fn create_list_todos_tool() -> Tool {
    Tool {
        r#type: "function".to_string(),
        function: Function {
            name: "list_todos".to_string(),
            description: "List all todo items.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "include_completed": {
                        "type": "boolean",
                        "description":
                            "Include completed items (default false)"
                    }
                },
                "required": []
            }),
        },
    }
}

fn create_complete_todo_tool() -> Tool {
    let mut props = HashMap::new();
    props.insert(
        "index".to_string(),
        Property::number("1-based index of the todo to complete"),
    );

    Tool {
        r#type: "function".to_string(),
        function: Function {
            name: "complete_todo".to_string(),
            description: "Mark a todo item as completed.".to_string(),
            parameters: Parameters::new(props, vec!["index".to_string()]).into(),
        },
    }
}

// -- Tool execution -------------------------------------------------------

fn execute_tool(name: &str, arguments: &str, todos: &mut Vec<TodoItem>) -> Result<String> {
    match name {
        "add_todo" => {
            let args: AddTodoArgs = serde_json::from_str(arguments)?;
            todos.push(TodoItem {
                title: args.title.clone(),
                priority: args.priority.clone(),
                done: false,
            });
            Ok(format!(
                "Added todo #{}: \"{}\" (priority: {})",
                todos.len(),
                args.title,
                args.priority,
            ))
        }
        "list_todos" => {
            let args: ListTodosArgs = serde_json::from_str(arguments)?;
            if todos.is_empty() {
                return Ok("No todos yet.".to_string());
            }
            let mut lines = Vec::new();
            for (i, item) in todos.iter().enumerate() {
                if !args.include_completed && item.done {
                    continue;
                }
                let status = if item.done { "done" } else { "pending" };
                lines.push(format!(
                    "{}. [{}] {} (priority: {})",
                    i + 1,
                    status,
                    item.title,
                    item.priority,
                ));
            }
            if lines.is_empty() {
                return Ok("All todos are completed (use \
                     include_completed=true to see them)."
                    .to_string());
            }
            Ok(lines.join("\n"))
        }
        "complete_todo" => {
            let args: CompleteTodoArgs = serde_json::from_str(arguments)?;
            let idx = args.index;
            if idx == 0 || idx > todos.len() {
                return Ok(format!(
                    "Invalid index {idx}. \
                     Valid range: 1-{}",
                    todos.len(),
                ));
            }
            let item = &mut todos[idx - 1];
            if item.done {
                return Ok(format!("Todo #{idx} \"{}\" is already done.", item.title,));
            }
            item.done = true;
            Ok(format!("Completed todo #{idx}: \"{}\"", item.title,))
        }
        _ => anyhow::bail!("Unknown tool: {name}"),
    }
}

// -- Conversation turn handling -------------------------------------------

enum TurnResult {
    Done,
    ToolCalls {
        content: String,
        tool_calls: Vec<ToolCall>,
    },
}

fn print_usage(usage: &neuromance_common::Usage) {
    println!("Usage:");
    println!("  Input tokens: {}", usage.prompt_tokens);
    println!("  Output tokens: {}", usage.completion_tokens);
    println!("  Total tokens: {}", usage.total_tokens);
}

async fn send_non_streaming(client: &AnthropicClient, request: &ChatRequest) -> Result<TurnResult> {
    println!("Sending request...");
    let response = client.chat(request).await?;

    if response.message.tool_calls.is_empty() {
        println!();
        println!("Assistant: {}", response.message.content);
        println!();
        if let Some(reason) = response.finish_reason {
            info!("Finish reason: {reason:?}");
        }
        if let Some(ref usage) = response.usage {
            print_usage(usage);
        }
        return Ok(TurnResult::Done);
    }

    let tool_calls = response.message.tool_calls.to_vec();
    let content = response.message.content;
    Ok(TurnResult::ToolCalls {
        content,
        tool_calls,
    })
}

async fn send_streaming(client: &AnthropicClient, request: &ChatRequest) -> Result<TurnResult> {
    println!("Sending streaming request...");
    println!();

    let mut stream = client.chat_stream(request).await?;

    let mut full_content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    print!("Assistant: ");
    std::io::stdout().flush()?;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;

        if let Some(content) = chunk.delta_content {
            print!("{content}");
            full_content.push_str(&content);
            std::io::stdout().flush()?;
        }

        if let Some(reasoning) = chunk.delta_reasoning_content {
            info!("[Thinking: {reasoning}]");
        }

        if let Some(tcs) = chunk.delta_tool_calls {
            tool_calls.extend(tcs);
        }

        if let Some(reason) = &chunk.finish_reason {
            info!("Stream finished: {reason:?}");
        }

        if let Some(ref usage) = chunk.usage {
            println!();
            print_usage(usage);
        }
    }
    println!();

    if tool_calls.is_empty() {
        return Ok(TurnResult::Done);
    }

    Ok(TurnResult::ToolCalls {
        content: full_content,
        tool_calls,
    })
}

fn process_tool_calls(
    content: &str,
    tool_calls: &[ToolCall],
    conversation_id: Uuid,
    messages: &mut Vec<Message>,
    todos: &mut Vec<TodoItem>,
) -> Result<()> {
    let mut assistant_msg = Message::assistant(conversation_id, content);
    assistant_msg.tool_calls.extend(tool_calls.iter().cloned());
    messages.push(assistant_msg);

    for tc in tool_calls {
        let args_str = tc.function.arguments_json();
        println!("  Tool call: {}({})", tc.function.name, args_str);
        let result = execute_tool(&tc.function.name, args_str, todos)?;
        println!("  Result:    {result}");

        let tool_msg = Message::tool(
            conversation_id,
            result,
            tc.id.clone(),
            tc.function.name.clone(),
        )?;
        messages.push(tool_msg);
    }
    println!();
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("Anthropic Claude Demo");
    println!("=====================");
    println!("Model: {}", args.model);
    println!("Max tokens: {}", args.max_tokens);
    println!("Streaming: {}", !args.no_stream);
    println!();

    let mut config = Config::new("anthropic", &args.model)
        .with_api_key(&args.api_key)
        .with_max_tokens(args.max_tokens);

    if let Some(temp) = args.temperature {
        config = config.with_temperature(temp);
    }

    let client = AnthropicClient::new(config)?;

    let conversation_id = Uuid::new_v4();
    let mut messages = vec![
        Message::system(
            conversation_id,
            "You are a helpful assistant with access to a \
             todo list. Use the provided tools to manage it.",
        ),
        Message::user(conversation_id, &args.message),
    ];

    let tools = vec![
        create_add_todo_tool(),
        create_list_todos_tool(),
        create_complete_todo_tool(),
    ];

    let mut todos: Vec<TodoItem> = Vec::new();

    loop {
        let mut request = ChatRequest::new(messages.clone())
            .with_model(&args.model)
            .with_max_tokens(args.max_tokens)
            .with_tools(tools.clone())
            .with_tool_choice(ToolChoice::Auto);

        if let Some(temp) = args.temperature {
            request = request.with_temperature(temp);
        }

        let result = if args.no_stream {
            send_non_streaming(&client, &request).await?
        } else {
            send_streaming(&client, &request).await?
        };

        match result {
            TurnResult::Done => break,
            TurnResult::ToolCalls {
                content,
                tool_calls,
            } => {
                process_tool_calls(
                    &content,
                    &tool_calls,
                    conversation_id,
                    &mut messages,
                    &mut todos,
                )?;
            }
        }
    }

    Ok(())
}
