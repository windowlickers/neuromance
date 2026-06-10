//! Adversarial subagent orchestration, driven by the LLM through the Python REPL.
//!
//! Unlike `fanout_vote` (where the combinator is fixed Rust), here an *orchestrator*
//! agent is handed the `execute_python` tool and a registry of subagents
//! (`solver`, `critic`, `judge`) reachable from Python via `run_subagent(...)`. The
//! system prompt teaches it one adversarial technique (draft → critique → judge) and
//! lets it write the orchestration code itself. This is the live harness we tweak
//! against a real model.
//!
//! Run against a local OpenAI-compatible server (e.g. llama.cpp on :8080):
//! ```bash
//! cargo run -p neuromance-agent --example adversarial_repl -- \
//!     --base-url http://127.0.0.1:8080/v1 --model local
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use tokio_util::sync::CancellationToken;
use tracing::info;

use neuromance_agent::{Agent, LocalSubagent};
use neuromance_client::chat_completions::client::ChatCompletionsClient;
use neuromance_common::client::Config;
use neuromance_common::subagent::Subagent;
use neuromance_repl::SubagentRepl;
use neuromance_repl::python::{PythonRepl, PythonReplConfig};

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

    /// Per-execution REPL timeout, in seconds. Must exceed the total time of all
    /// nested `run_subagent` calls in one script (each can take minutes).
    #[arg(long, default_value_t = 300)]
    repl_timeout_secs: u64,

    /// Append `/no_think` to every prompt to disable reasoning traces on models
    /// that support it (e.g. Qwen3) — much faster for a live demo.
    #[arg(long, default_value_t = false)]
    no_think: bool,
}

/// Optionally append the Qwen-style `/no_think` directive to a prompt.
fn with_think(base: &str, no_think: bool) -> String {
    if no_think {
        format!("{base}\n\n/no_think")
    } else {
        base.to_string()
    }
}

/// System prompt: teaches the orchestrator the adversarial technique and the
/// `run_subagent` / `spawn_agents` primitives available inside `execute_python`.
const ORCHESTRATOR_SYSTEM_PROMPT: &str = r#"You are an orchestrator. You do NOT answer the user directly from your own knowledge.
Instead, you delegate to specialist subagents by running Python, then synthesize their work.

The ONLY way to run Python is to make an execute_python tool call. Never write Python in your
message text — code in your reply does nothing. You cannot see any subagent output until you
actually call the execute_python tool and read its result.

Inside that Python environment these functions are ALREADY DEFINED as globals:

    run_subagent(name, instructions, context=None) -> str
    spawn_agents([Agent(name, instructions, context=None), ...]) -> list[str]

run_subagent runs one subagent and returns its answer — use it for dependent steps where each
call needs the previous answer. spawn_agents runs a batch concurrently and returns their answers
in order — use it for independent fan-out instead of calling run_subagent in a loop. They are
builtins — never import them, and never `import` anything else. Available subagents:
  - 'solver': proposes an answer to a problem.
  - 'critic': adversarially attacks an answer, listing concrete flaws, gaps, and counterexamples.
  - 'judge': given the full record, returns the single best final answer.

Use an adversarial loop: draft -> critique -> judge. Pass the WHOLE script as the `code`
argument on a SINGLE LINE, separating statements with semicolons. Do not use newlines.
For example (one line):

    question = "<the user's question>"; draft = run_subagent('solver', question); critique = run_subagent('critic', 'Find every flaw, gap, or error in this answer. Be specific.', context=draft); final = run_subagent('judge', 'Give the single best, correct final answer to the question.', context='Question: ' + question + ' || Draft: ' + draft + ' || Critique: ' + critique); print(final)

Rules:
  - You MUST delegate the actual reasoning to the subagents via run_subagent. Do NOT solve the
    problem yourself and do NOT compute the answer in Python — Python is only glue.
  - You MUST print() the value you want to read back; expression results are not shown.
  - The `code` argument must be ONE LINE of raw Python with semicolons between statements.
    No newlines, no backslashes, and no markdown fences (no ```).
  - After you see the printed result, reply to the user with that final answer and nothing else."#;

fn build_subagent(id: &str, system_prompt: &str, config: &Config) -> Result<Arc<dyn Subagent>> {
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

    // The specialists the orchestrator can call from Python.
    let mut subagents: HashMap<String, Arc<dyn Subagent>> = HashMap::new();
    subagents.insert(
        "solver".to_string(),
        build_subagent(
            "solver",
            &with_think(
                "You are a careful problem solver. Think step by step and answer concisely.",
                args.no_think,
            ),
            &config,
        )?,
    );
    subagents.insert(
        "critic".to_string(),
        build_subagent(
            "critic",
            &with_think(
                "You are a relentless adversarial critic. Given an answer, find every flaw, \
                 hidden assumption, and counterexample. Do not be agreeable.",
                args.no_think,
            ),
            &config,
        )?,
    );
    subagents.insert(
        "judge".to_string(),
        build_subagent(
            "judge",
            &with_think(
                "You are a judge. Weigh the draft and the critique, then state the single \
                 best final answer plainly.",
                args.no_think,
            ),
            &config,
        )?,
    );

    // Inject the subagents into a REPL and expose it as the execute_python tool.
    // The timeout must cover all nested run_subagent calls in a single script.
    let repl_config =
        PythonReplConfig::default().with_timeout(Duration::from_secs(args.repl_timeout_secs));
    let repl = Arc::new(PythonRepl::with_config(repl_config)?);
    let python_tool = SubagentRepl::new(repl, subagents, CancellationToken::new())?.into_tool();

    // Tool choice stays Auto: the model calls execute_python, then replies with text,
    // which lets the loop finish. (Forcing ToolChoice::Required would make every turn a
    // tool call, so the loop could only ever end by erroring at max_turns.)
    let mut orchestrator = Agent::builder("orchestrator", ChatCompletionsClient::new(config)?)
        .system_prompt(with_think(ORCHESTRATOR_SYSTEM_PROMPT, args.no_think))
        .user_prompt(&args.question)
        .add_tool(python_tool)
        .max_turns(4)
        .auto_approve_tools(true)
        .build();

    info!(question = %args.question, "running adversarial orchestration");
    let response = orchestrator.execute(None, CancellationToken::new()).await?;

    // Show what the orchestrator ran (the python transcript) for live tweaking.
    for (i, tool_msg) in response.tool_responses.iter().enumerate() {
        println!(
            "--- execute_python result #{} ---\n{}\n",
            i + 1,
            tool_msg.content
        );
    }
    println!(
        "=== orchestrator final answer ===\n{}",
        response.content.content
    );

    Ok(())
}
