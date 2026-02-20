//! `OpenAI` Chat Completions API Demo
//!
//! This example demonstrates using the `OpenAI` client (Chat Completions API)
//! with tool calling and a multi-turn conversation loop. Compatible with
//! `OpenAI` and any provider exposing a `/chat/completions` endpoint.
//!
//! # Usage
//!
//! ```bash
//! # With OpenAI
//! cargo run --example openai_demo -- \
//!     --api-key sk-... --model gpt-5-mini-2025-08-07
//!
//! # With a local server
//! cargo run --example openai_demo -- \
//!     --base-url http://localhost:8080/v1 \
//!     --model my-model
//!
//! # With a custom prompt
//! cargo run --example openai_demo -- \
//!     --message "Add a todo to water the plants"
//! ```

use std::collections::HashMap;

use anyhow::Result;
use clap::Parser;
use serde::Deserialize;
use uuid::Uuid;

use neuromance_client::{LLMClient, OpenAIClient};
use neuromance_common::{
    ChatRequest, Config, Function, Message, Parameters, Property, Tool, ToolChoice,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "OpenAI Chat Completions API Demo")]
struct Args {
    /// Base URL for the API endpoint
    #[arg(long, default_value = "http://localhost:8080/v1")]
    base_url: String,

    /// API key for authentication
    #[arg(long, default_value = "dummy")]
    api_key: String,

    /// Model to use for chat completion
    #[arg(long, default_value = "gpt-5-mini-2025-08-07")]
    model: String,

    /// The user message to send
    #[arg(
        long,
        default_value = "Add three todos: buy groceries (high priority), \
            call the dentist, and read a book (low priority). \
            Then list them all and mark the second one as done."
    )]
    message: String,

    /// Temperature for sampling (0.0-2.0)
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

// -- Main -----------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("OpenAI Chat Completions Demo");
    println!("============================");
    println!("Base URL: {}", args.base_url);
    println!("Model: {}", args.model);
    println!();

    let mut config = Config::new("openai", &args.model)
        .with_base_url(&args.base_url)
        .with_api_key(&args.api_key);

    if let Some(temp) = args.temperature {
        config = config.with_temperature(temp);
    }

    let client = OpenAIClient::new(config)?;

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
            .with_tools(tools.clone())
            .with_tool_choice(ToolChoice::Auto);

        if let Some(temp) = args.temperature {
            request = request.with_temperature(temp);
        }

        println!("Sending request...");
        let response = client.chat(&request).await?;

        if response.message.tool_calls.is_empty() {
            println!();
            println!("Assistant: {}", response.message.content);
            println!();
            if let Some(reason) = response.finish_reason {
                println!("Finish reason: {reason:?}");
            }
            if let Some(usage) = response.usage {
                println!("Usage:");
                println!("  Input tokens: {}", usage.prompt_tokens);
                println!("  Output tokens: {}", usage.completion_tokens);
                println!("  Total tokens: {}", usage.total_tokens);
            }
            break;
        }

        let tool_calls = response.message.tool_calls.clone();
        messages.push(response.message);

        for tc in &tool_calls {
            let args_str = tc.function.arguments_json();
            println!("  Tool call: {}({})", tc.function.name, args_str);
            let result = execute_tool(&tc.function.name, args_str, &mut todos)?;
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
    }

    Ok(())
}
