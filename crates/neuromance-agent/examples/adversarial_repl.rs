//! Adversarial subagent orchestration, driven by the LLM through the Python REPL.
//!
//! Unlike `fanout_vote` (where the combinator is fixed Rust), here an *orchestrator*
//! agent is handed the `execute_python` tool and a registry of subagents
//! (`solver`, `critic`, `judge`) reachable from Python via `run_subagent(...)`. The
//! system prompt teaches it one adversarial technique (draft → critique → revise →
//! judge) and lets it write the orchestration code itself. This is the live harness
//! we tweak against a real model.
//!
//! Run against a local OpenAI-compatible server (e.g. llama.cpp on :8080):
//! ```bash
//! cargo run -p neuromance-agent --example adversarial_repl -- \
//!     --base-url http://127.0.0.1:8080/v1 --model local
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tokio_util::sync::CancellationToken;
use tracing::info;

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

    /// The question the orchestrator must answer via its subagents.
    #[arg(
        long,
        default_value = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more \
                         than the ball. How much does the ball cost? Show your reasoning."
    )]
    question: String,
}

/// System prompt: teaches the orchestrator the adversarial technique and the
/// `run_subagent` primitive available inside `execute_python`.
const ORCHESTRATOR_SYSTEM_PROMPT: &str = r#"You are an orchestrator. You do NOT answer the user directly from your own knowledge.
Instead, you delegate to specialist subagents by running Python, then synthesize their work.

The ONLY way to run Python is to make an execute_python tool call. Never write Python in your
message text — code in your reply does nothing. You cannot see any subagent output until you
actually call the execute_python tool and read its result.

Inside that Python environment one function is available:

    run_subagent(name, instructions, context=None) -> str

It runs the named subagent and returns its answer as a string. Available subagents:
  - 'solver': proposes an answer to a problem.
  - 'critic': adversarially attacks an answer, listing concrete flaws, gaps, and counterexamples.
  - 'judge': given the full record, returns the single best final answer.

Use an adversarial loop: draft -> critique -> revise -> judge. For example:

    question = "<the user's question>"
    draft = run_subagent('solver', question)
    critique = run_subagent('critic',
        'Find every flaw, gap, or error in this answer. Be specific.', context=draft)
    revised = run_subagent('solver',
        'Revise your answer to fix these problems.',
        context='Question: ' + question + '\n\nDraft: ' + draft + '\n\nCritique: ' + critique)
    final = run_subagent('judge',
        'Give the single best, correct final answer to the question.',
        context='Question: ' + question + '\n\nRevised answer: ' + revised + '\n\nCritique: ' + critique)
    print(final)

Rules:
  - You MUST delegate the actual reasoning to the subagents via run_subagent. Do NOT solve the
    problem yourself and do NOT compute the answer in Python — Python is only glue.
  - You MUST print() the value you want to read back; expression results are not shown.
  - Call execute_python with one complete script that runs the whole loop.
  - The `code` argument must be raw Python with real newlines. Do NOT wrap it in markdown
    fences (no ```), and do NOT write the two characters backslash-n for newlines.
  - After you see the printed result, reply to the user with that final answer and nothing else."#;

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

    // The specialists the orchestrator can call from Python.
    let mut subagents: HashMap<String, Arc<dyn Subagent>> = HashMap::new();
    subagents.insert(
        "solver".to_string(),
        build_subagent(
            "solver",
            "You are a careful problem solver. Think step by step and answer concisely.",
            &config,
        )?,
    );
    subagents.insert(
        "critic".to_string(),
        build_subagent(
            "critic",
            "You are a relentless adversarial critic. Given an answer, find every flaw, \
             hidden assumption, and counterexample. Do not be agreeable.",
            &config,
        )?,
    );
    subagents.insert(
        "judge".to_string(),
        build_subagent(
            "judge",
            "You are a judge. Weigh the draft, critique, and revision, then state the single \
             best final answer plainly.",
            &config,
        )?,
    );

    // Inject the subagents into a REPL and expose it as the execute_python tool.
    let repl = Arc::new(PythonRepl::new()?);
    let python_tool = SubagentRepl::new(repl, subagents, CancellationToken::new())?.into_tool();

    let mut orchestrator = Agent::builder("orchestrator", ChatCompletionsClient::new(config)?)
        .system_prompt(ORCHESTRATOR_SYSTEM_PROMPT)
        .user_prompt(&args.question)
        .add_tool(python_tool)
        .max_turns(6)
        .auto_approve_tools(true)
        .build();

    info!(question = %args.question, "running adversarial orchestration");
    let response = orchestrator.execute(None, CancellationToken::new()).await?;

    // Show what the orchestrator ran (the python transcript) for live tweaking.
    for (i, tool_msg) in response.tool_responses.iter().enumerate() {
        println!("--- execute_python result #{} ---\n{}\n", i + 1, tool_msg.content);
    }
    println!("=== orchestrator final answer ===\n{}", response.content.content);

    Ok(())
}
