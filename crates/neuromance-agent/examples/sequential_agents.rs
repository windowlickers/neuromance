use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use log::info;
use uuid::Uuid;

use neuromance::Core;
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

    info!("Sequential Agents Demo");
    info!("======================");
    info!("Base URL: {}", args.base_url);
    info!("Model: {}", args.model);

    // Create client configuration
    let config = Config::new("openai", &args.model)
        .with_base_url(&args.base_url)
        .with_api_key(&args.api_key);

    let start_time = Instant::now();

    // Agent 1: Pirate
    info!(" --- Agent 1: Pirate ---");
    let client1 = OpenAIClient::new(config.clone())?;
    let mut core1 = Core::new(client1);
    core1.tool_executor.add_tool(CurrentTimeTool);
    let mut pirate_agent = BaseAgent::new("pirate".to_string(), core1);

    let conversation_id = Uuid::new_v4();
    let pirate_messages = vec![
        Message::system(conversation_id, "You always speak like a pirate."),
        Message::user(conversation_id, "What time is it?"),
    ];

    let pirate_response = pirate_agent.execute(Some(pirate_messages)).await?;
    info!("Pirate says: {}", pirate_response.content.content);

    // Agent 2: Vampire
    info!(" --- Agent 2: Vampire ---");
    let client2 = OpenAIClient::new(config.clone())?;
    let mut core2 = Core::new(client2);
    core2.tool_executor.add_tool(CurrentTimeTool);
    let mut vampire_agent = BaseAgent::new("vampire".to_string(), core2);

    let vampire_messages = vec![
        Message::system(
            conversation_id,
            "You always speak like a vampire from Transylvania.",
        ),
        Message::user(conversation_id, "What time is it?"),
    ];

    let vampire_response = vampire_agent.execute(Some(vampire_messages)).await?;
    info!("Vampire says: {}", vampire_response.content.content);

    // Agent 3: Summarizer
    info!(" --- Agent 3: Summarizer ---");
    let client3 = OpenAIClient::new(config.clone())?;
    let core3 = Core::new(client3);
    let mut summarizer_agent = BaseAgent::new("summarizer".to_string(), core3);

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
