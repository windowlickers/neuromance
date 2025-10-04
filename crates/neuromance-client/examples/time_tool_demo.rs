use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono_tz::Tz;
use clap::Parser;
use log::{debug, info};
use once_cell::sync::Lazy;
use serde::Deserialize;
use uuid::Uuid;

use neuromance_client::{LLMClient, OpenAIClient};
use neuromance_common::{
    ChatRequest, Config, Function, Message, Property, Tool, ToolChoice, Usage,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Base URL for the API endpoint
    #[arg(long, default_value = "http://localhost:8080/v1")]
    base_url: String,

    /// API key for authentication
    #[arg(long, default_value = "dummy")]
    api_key: String,

    /// Model to use for chat completion
    #[arg(long, default_value = "gpt-oss:20b")]
    model: String,

    /// The user message to send
    #[arg(long, default_value = "What time is it?")]
    message: String,
}

/// Arguments for the get_current_time tool.
#[derive(Debug, Deserialize)]
struct GetCurrentTimeArgs {
    #[serde(default = "default_timezone")]
    timezone: String,
}

fn default_timezone() -> String {
    "UTC".to_string()
}

/// Parse tool arguments with proper error handling.
fn parse_tool_args(arguments: &str) -> Result<GetCurrentTimeArgs> {
    serde_json::from_str(arguments).context("Failed to parse tool arguments")
}

fn create_get_current_time_tool() -> Tool {
    let mut properties = HashMap::new();

    // The get_current_time tool doesn't need any parameters
    // but we'll include an optional timezone parameter
    properties.insert(
        "timezone".to_string(),
        Property {
            prop_type: "string".to_string(),
            description: "Optional timezone (e.g., 'UTC', 'America/New_York'). Defaults to UTC."
                .to_string(),
        },
    );

    Tool {
        r#type: "function".to_string(),
        function: Function {
            name: "get_current_time".to_string(),
            description: "Get the current date and time. Optionally specify a timezone."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": properties,
                "required": []
            }),
        },
    }
}

/// Lazily initialized tool definition.
static TIME_TOOL: Lazy<Tool> = Lazy::new(create_get_current_time_tool);

fn tool_execution(tool_name: &str, arguments: &str) -> Result<String> {
    match tool_name {
        "get_current_time" => {
            // Parse timezone from arguments with proper error handling
            let args = parse_tool_args(arguments).unwrap_or_else(|_| GetCurrentTimeArgs {
                timezone: default_timezone(),
            });

            let now = chrono::Utc::now();

            // Try to parse the timezone string
            match args.timezone.parse::<Tz>() {
                Ok(tz) => {
                    let local_time = now.with_timezone(&tz);
                    Ok(format!(
                        "Current time in {}: {}",
                        args.timezone,
                        local_time.format("%Y-%m-%d %H:%M:%S %Z")
                    ))
                }
                Err(_) => {
                    // If parsing fails, fall back to UTC
                    Ok(format!(
                        "Unknown timezone '{}'. Current time in UTC: {}",
                        args.timezone,
                        now.format("%Y-%m-%d %H:%M:%S UTC")
                    ))
                }
            }
        }
        _ => Ok(format!("Unknown tool: {}", tool_name)),
    }
}

fn print_usage(u: Usage) -> Result<()> {
    info!("Usage:");
    info!("Prompt tokens: {}", u.prompt_tokens);
    info!("Completion tokens: {}", u.completion_tokens);
    info!("Total tokens: {}", u.total_tokens);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    // Parse command line arguments
    let args = Args::parse();

    info!("Time Tool Demo");
    info!("==============");
    info!("Base URL: {}", args.base_url);
    info!("Model: {}", args.model);
    info!("Message: {}", args.message);

    // Create client configuration
    let config = Config::new("openai", &args.model)
        .with_base_url(&args.base_url)
        .with_api_key(&args.api_key);
    // .with_temperature(0.7);

    // Create OpenAI client
    let client = OpenAIClient::new(config.clone())?;

    // Use the lazily initialized tool
    info!("Using tool: {}", TIME_TOOL.function.name);
    info!("Tool description: {}", TIME_TOOL.function.description);

    // Create the chat request with the tool
    let conversation_id = Uuid::new_v4();
    let mut messages: Vec<Message> = Vec::new();

    messages.push(Message::system(
        conversation_id,
        "You always speak like a pirate.",
    ));
    messages.push(Message::user(conversation_id, &args.message));

    // turbo-fish style
    let request: ChatRequest = Into::<ChatRequest>::into((config.clone(), messages.clone()))
        .with_tools(vec![TIME_TOOL.clone()])
        .with_tool_choice(ToolChoice::Auto);

    // Simple style
    // let mut request: ChatRequest = (config.clone(), messages.clone()).into();
    // request = request
    //     .with_tools(vec![time_tool])
    //     .with_tool_choice(ToolChoice::Auto);

    info!("Sending chat request...");
    let pretty_request = serde_json::to_string_pretty(&request.clone())?;
    debug!("\n{}", pretty_request);

    // Start timing the request
    let start_time = Instant::now();

    // Send the chat request
    let response = client.chat(&request).await?;

    info!("Response received!");
    info!("Model: {}", response.model);
    info!("Content: {}", response.message.content);
    if let Some(usage) = response.usage.clone() {
        print_usage(usage)?;
    }

    // Check if the model made any tool calls
    let mut tool_responses: Vec<Message> = Vec::new();
    if !response.message.tool_calls.is_empty() {
        info!("Tool calls made:");
        for tool_call in &response.message.tool_calls {
            info!("Tool: {}", tool_call.function.name);
            info!("Arguments: {}", tool_call.function.arguments.join(", "));

            // executing the tool
            let tool_result = tool_execution(
                &tool_call.function.name,
                &tool_call.function.arguments.join(""),
            )?;
            info!("Result: {}", tool_result);

            let tool_response = Message::tool(
                conversation_id,
                tool_result,
                tool_call.id.clone(),
                tool_call.function.name.clone(),
            )?;
            tool_responses.push(tool_response);
        }

        // Send tool responses back to the model for a final response
        info!("Sending tool responses back to the model...");

        // Add the assistant's response with tool calls to the message history first
        messages.push(response.message.clone());
        // Then add the tool responses
        messages.extend(tool_responses);

        // NOTE We are not passing the tool on the follow up request
        // We added the tool response to messages and it's no longer required.
        // Best to not waste tokens on follow up.
        //
        // turbo-fish style
        let follow_up_request: ChatRequest =
            Into::<ChatRequest>::into((config.clone(), messages.clone()));
        // simple style
        // let follow_up_request: ChatRequest = (config.clone(), messages.clone()).into();

        info!("Sending chat request...");
        let pretty_request = serde_json::to_string_pretty(&follow_up_request.clone())?;
        debug!("\n{}", pretty_request);

        let final_response = client.chat(&follow_up_request).await?;
        info!("Final response:\n\n {}", final_response.message.content);

        if let Some(final_usage) = final_response.usage {
            print_usage(final_usage)?;
        }
    } else {
        info!("No tool calls were made by the model.");
    }

    // Calculate total time
    let total_duration = start_time.elapsed();

    // Print total time taken
    info!("Total request time: {:.2}ms", total_duration.as_millis());

    Ok(())
}
