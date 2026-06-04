//! Drive subagents from inside the Python REPL.
//!
//! Registers a `worker` and a `judge` subagent, exposes them to Python as
//! `run_subagent(...)`, and lets the *Python code* define the orchestration
//! technique (here: fan out to the worker, then have the judge pick the best).
//!
//! Run with a local OpenAI-compatible server (e.g. llama.cpp on :8080):
//! ```bash
//! cargo run -p neuromance-repl --example python_subagent --features subagent -- \
//!     --base-url http://127.0.0.1:8080/v1 --model local
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tokio_util::sync::CancellationToken;

use neuromance_agent::{Agent, LocalSubagent};
use neuromance_client::chat_completions::client::ChatCompletionsClient;
use neuromance_common::client::Config;
use neuromance_common::subagent::Subagent;
use neuromance_repl::SubagentRepl;
use neuromance_repl::python::PythonRepl;

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
    #[arg(long, default_value = "local")]
    model: String,
}

fn build_subagent(id: &str, system_prompt: &str, config: &Config) -> Result<Arc<dyn Subagent>> {
    let client = ChatCompletionsClient::new(config.clone())?;
    let agent = Agent::builder(id, client)
        .max_turns(3)
        .auto_approve_tools(true)
        .build();
    Ok(Arc::new(LocalSubagent::new(agent, system_prompt)))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let config = Config::new("openai", &args.model)
        .with_base_url(&args.base_url)
        .with_api_key(&args.api_key);

    let mut subagents: HashMap<String, Arc<dyn Subagent>> = HashMap::new();
    subagents.insert(
        "worker".to_string(),
        build_subagent(
            "worker",
            "You are a careful problem solver. Answer concisely.",
            &config,
        )?,
    );
    subagents.insert(
        "judge".to_string(),
        build_subagent(
            "judge",
            "You are a judge. Pick or synthesize the best answer from the candidates.",
            &config,
        )?,
    );

    let repl = Arc::new(PythonRepl::new()?);
    let bridge = SubagentRepl::new(repl, subagents, CancellationToken::new())?;

    // The technique lives in Python, not Rust: fan out, then vote.
    let code = r#"
question = "What is the capital of France, and why is it significant?"
answers = [run_subagent('worker', question) for _ in range(3)]
verdict = run_subagent('judge', 'Pick the best answer:\n\n' + '\n\n'.join(answers))
print(verdict)
"#;

    let result = bridge.repl().execute(code).await?;
    if result.success {
        println!("--- stdout ---\n{}", result.stdout);
    } else {
        eprintln!("--- python error ---\n{}", result.stderr);
    }

    Ok(())
}
