//! Fanout + vote: run a task across several agents in parallel, then have a judge
//! agent pick or synthesize the best answer. The combinator is itself a subagent,
//! so the same `FanoutVote` is also wrapped as a tool a parent agent could call.

use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tokio_util::sync::CancellationToken;
use tracing::info;

use neuromance_agent::{Agent, FanoutVote, LocalSubagent, Subagent, SubagentTool};
use neuromance_client::chat_completions::client::ChatCompletionsClient;
use neuromance_common::client::Config;
use neuromance_common::task::Task;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Base URL for the API endpoint.
    #[arg(long, default_value = "http://localhost:8080/v1")]
    base_url: String,

    /// API key for authentication.
    #[arg(long, default_value = "dummy")]
    api_key: String,

    /// Model to use for chat completion.
    #[arg(long, default_value = "ggml-org/gpt-oss-120b-GGUF")]
    model: String,

    /// Number of member agents to fan out to.
    #[arg(long, default_value_t = 3)]
    members: usize,
}

fn build_local_subagent(
    id: &str,
    system_prompt: &str,
    config: &Config,
) -> Result<Arc<dyn Subagent>> {
    let client = ChatCompletionsClient::new(config.clone())?;
    let agent_id = id.to_string();
    Ok(Arc::new(LocalSubagent::new(id, system_prompt, move || {
        Agent::builder(agent_id.clone(), client.clone())
            .max_turns(3)
            .auto_approve_tools(true)
            .build()
    })))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let config = Config::new("openai", &args.model)
        .with_base_url(&args.base_url)
        .with_api_key(&args.api_key);

    let members = (0..args.members)
        .map(|i| {
            build_local_subagent(
                &format!("member-{i}"),
                "You are a careful problem solver. Answer concisely.",
                &config,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let judge = build_local_subagent(
        "judge",
        "You are a judge. Pick or synthesize the best answer from the candidates.",
        &config,
    )?;

    let fanout = Arc::new(FanoutVote::new("fanout-vote", members, judge)?);

    info!("Running fanout+vote across {} members", args.members);
    let task = Task::new("What is the capital of France, and why is it significant?");
    let outcome = fanout.run(task, CancellationToken::new()).await?;
    info!("Winning answer:\n\n{}", outcome.content);

    // The same combinator can be handed to a parent agent as a tool.
    let _delegate = SubagentTool::new(
        fanout as Arc<dyn Subagent>,
        "delegate",
        "Delegate a task to a panel of agents that vote on the best answer.",
        CancellationToken::new(),
    );
    info!("FanoutVote is also usable as a tool via SubagentTool::new");

    Ok(())
}
