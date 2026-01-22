//! `OpenAI` Responses API Demo
//!
//! This example demonstrates using the Responses API client with tool calling
//! and a multi-turn conversation loop. Compatible with `OpenAI` and any
//! provider implementing the Responses API (e.g., Ollama).
//!
//! # Usage
//!
//! ```bash
//! # With OpenAI
//! cargo run --example responses_demo -- --api-key sk-... --model gpt-4o
//!
//! # With Ollama (openresponses)
//! cargo run --example responses_demo -- \
//!     --base-url http://localhost:11434/v1 \
//!     --model llama3.2
//!
//! # With a tool-calling prompt
//! cargo run --example responses_demo -- \
//!     --base-url http://192.168.1.31:11434/v1 \
//!     --model gpt-oss:20b \
//!     --message "What time is it right now?"
//! ```

use anyhow::Result;
use clap::Parser;
use uuid::Uuid;

use neuromance_client::{LLMClient, ResponsesClient};
use neuromance_common::{ChatRequest, Config, Function, Message, Tool, ToolChoice};

#[derive(Parser, Debug)]
#[command(author, version, about = "OpenAI Responses API Demo")]
struct Args {
    /// Base URL for the API endpoint
    #[arg(long, default_value = "https://api.openai.com/v1")]
    base_url: String,

    /// API key for authentication (or set `OPENAI_API_KEY` env var)
    #[arg(long, env = "OPENAI_API_KEY", default_value = "ollama")]
    api_key: String,

    /// Model to use
    #[arg(long, default_value = "gpt-4o")]
    model: String,

    /// The user message to send
    #[arg(long, default_value = "What time is it right now?")]
    message: String,

    /// Temperature for sampling (0.0-2.0)
    #[arg(long)]
    temperature: Option<f32>,
}

fn create_time_tool() -> Tool {
    Tool {
        r#type: "function".to_string(),
        function: Function {
            name: "get_current_time".to_string(),
            description: "Get the current date and time in UTC.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
    }
}

fn execute_tool(name: &str, _arguments: &str) -> Result<String> {
    match name {
        "get_current_time" => {
            let now = chrono::Utc::now();
            Ok(format!(
                "Current time: {}",
                now.format("%Y-%m-%d %H:%M:%S UTC")
            ))
        }
        _ => anyhow::bail!("Unknown tool: {name}"),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("Responses API Demo");
    println!("==================");
    println!("Base URL: {}", args.base_url);
    println!("Model: {}", args.model);
    println!();

    let config = Config::new("responses", &args.model)
        .with_base_url(&args.base_url)
        .with_api_key(&args.api_key);

    let client = ResponsesClient::new(config)?;

    let conversation_id = Uuid::new_v4();
    let mut messages = vec![
        Message::system(conversation_id, "You are a helpful assistant."),
        Message::user(conversation_id, &args.message),
    ];

    let time_tool = create_time_tool();

    // Conversation loop: keep going until we get a final text response
    loop {
        let mut request = ChatRequest::new(messages.clone())
            .with_model(&args.model)
            .with_tools(vec![time_tool.clone()])
            .with_tool_choice(ToolChoice::Auto);

        if let Some(temp) = args.temperature {
            request = request.with_temperature(temp);
        }

        println!("Sending request...");
        let response = client.chat(&request).await?;

        if response.message.tool_calls.is_empty() {
            // Final text response — no more tool calls
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

        // Add the assistant response (with tool calls) to history first,
        // then append tool results — order matters for the model.
        let tool_calls = response.message.tool_calls.clone();
        messages.push(response.message);

        for tc in &tool_calls {
            let args_str = tc.function.arguments_json();
            println!("  Tool call: {}({})", tc.function.name, args_str);
            let result = execute_tool(&tc.function.name, &args_str)?;
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
