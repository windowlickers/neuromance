use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use log::info;
use uuid::Uuid;

use neuromance_agent::{Agent, BaseAgent};
use neuromance_client::openai::client::OpenAIClient;
use neuromance_common::chat::Message;
use neuromance_common::client::Config;
use neuromance_tools::generic::CurrentTimeTool;

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
    #[arg(long, default_value = "ggml-org/gpt-oss-120b-GGUF")]
    model: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    info!("Sequential Agents Demo (Builder Pattern)");
    info!("=========================================");
    info!("Base URL: {}", args.base_url);
    info!("Model: {}", args.model);

    // Create client configuration
    let config = Config::new("openai", &args.model)
        .with_base_url(&args.base_url)
        .with_api_key(&args.api_key);

    let start_time = Instant::now();
    let conversation_id = Uuid::new_v4();

    // Agent 1: Pirate (using builder)
    info!(" --- Agent 1: Pirate ---");
    let client1 = OpenAIClient::new(config.clone())?;

    let mut pirate_agent = BaseAgent::builder("pirate", client1)
        .add_tool(CurrentTimeTool)
        .max_turns(3)
        .auto_approve_tools(true)
        .build();

    let pirate_messages = vec![
        Message::system(conversation_id, "You always speak like a pirate."),
        Message::user(conversation_id, "What time is it?"),
    ];

    let pirate_response = pirate_agent.execute(Some(pirate_messages)).await?;
    info!("Pirate says: {}", pirate_response.content.content);

    // Agent 2: Vampire (using builder)
    info!(" --- Agent 2: Vampire ---");
    let client2 = OpenAIClient::new(config.clone())?;

    let mut vampire_agent = BaseAgent::builder("vampire", client2)
        .add_tool(CurrentTimeTool)
        .max_turns(3)
        .auto_approve_tools(true)
        .build();

    let vampire_messages = vec![
        Message::system(
            conversation_id,
            "You always speak like a vampire from Transylvania.",
        ),
        Message::user(conversation_id, "What time is it?"),
    ];

    let vampire_response = vampire_agent.execute(Some(vampire_messages)).await?;
    info!("Vampire says: {}", vampire_response.content.content);

    // Agent 3: Summarizer (using builder)
    info!(" --- Agent 3: Summarizer ---");
    let client3 = OpenAIClient::new(config.clone())?;

    let mut summarizer_agent = BaseAgent::builder("summarizer", client3)
        .max_turns(3)
        .auto_approve_tools(true)
        .build();

    let summary_prompt = format!(
        "Summarize what these two characters said about the time:\n\nPirate: {}\n\nVampire: {}",
        pirate_response.content.content, vampire_response.content.content
    );

    let summarizer_messages = vec![
        Message::system(
            conversation_id,
            "You are a helpful assistant that summarizes conversations concisely.",
        ),
        Message::user(conversation_id, &summary_prompt),
    ];

    let summary_response = summarizer_agent.execute(Some(summarizer_messages)).await?;
    info!("Summary:\n\n {}", summary_response.content.content);

    let total_duration = start_time.elapsed();
    info!("Total execution time: {:.2}s", total_duration.as_secs_f64());

    Ok(())
}
