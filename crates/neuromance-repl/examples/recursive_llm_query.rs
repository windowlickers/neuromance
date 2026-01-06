//! Demonstrates recursive REPL environments with depth-limited llm_query.
//!
//! Each layer spawns a new Python REPL with:
//! - Context set as a variable
//! - The same llm_query function (with depth tracking)
//!
//! Run with: cargo run --example recursive_llm_query --features python

use neuromance_repl::python::PythonRepl;
use neuromance_repl::{PythonCallback, ReplEnvironment};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// Maximum recursion depth for llm_query calls
const MAX_DEPTH: u32 = 6;

/// Global counter to track total recursive calls made
static TOTAL_CALLS: AtomicU32 = AtomicU32::new(0);

/// Creates a recursive llm_query callback that:
/// 1. Creates a new Python REPL at each level
/// 2. Injects context as a variable
/// 3. Injects itself with incremented depth
/// 4. Stops at MAX_DEPTH
fn make_recursive_callback(depth: u32, max_depth: u32) -> PythonCallback {
    Box::new(move |args: Vec<String>, kwargs: HashMap<String, String>| {
        Box::pin(async move {
            let call_num = TOTAL_CALLS.fetch_add(1, Ordering::SeqCst) + 1;
            let indent = "  ".repeat(depth as usize);

            println!("{indent}[Call #{call_num}] llm_query at depth {depth}");

            // Check depth limit
            if depth >= max_depth {
                println!(
                    "{indent}  -> MAX_DEPTH ({max_depth}) reached, returning truncated response"
                );
                return Ok(format!(
                    "[DEPTH_LIMIT] Cannot recurse further (depth={depth}, max={max_depth})"
                ));
            }

            // Extract query and context
            let query = args
                .first()
                .cloned()
                .unwrap_or_else(|| "no query".to_string());
            let context = kwargs
                .get("context")
                .cloned()
                .or_else(|| args.get(1).cloned());

            println!(
                "{indent}  Query: '{}'",
                if query.len() > 50 {
                    format!("{}...", &query[..50])
                } else {
                    query.clone()
                }
            );
            println!(
                "{indent}  Context: {} chars",
                context.as_ref().map(|c| c.len()).unwrap_or(0)
            );

            // Create a new REPL for this level
            let repl = PythonRepl::new().map_err(|e| format!("Failed to create REPL: {e}"))?;

            // Inject context as a variable if provided
            if let Some(ctx) = &context {
                repl.set_variable("context", ctx)
                    .await
                    .map_err(|e| format!("Failed to set context: {e}"))?;
                println!("{indent}  -> Injected 'context' variable into child REPL");
            }

            // Inject llm_query with incremented depth into the child REPL
            repl.inject_function("llm_query", make_recursive_callback(depth + 1, max_depth))
                .await
                .map_err(|e| format!("Failed to inject llm_query: {e}"))?;
            println!(
                "{indent}  -> Injected 'llm_query' (depth={}) into child REPL",
                depth + 1
            );

            // Simulate "processing" by executing some Python code
            // In a real scenario, this would involve an actual LLM call
            let python_code = format!(
                r#"
# Simulating LLM processing at depth {depth}
result = "Processed: {query}"

# If we have context, reference it
if 'context' in dir():
    result += f" (context length: {{len(context)}})"
"#,
                depth = depth,
                query = query.replace('"', r#"\""#)
            );

            let exec_result = repl
                .execute(&python_code)
                .await
                .map_err(|e| format!("Execution failed: {e}"))?;

            if !exec_result.success {
                return Err(format!("Python error: {}", exec_result.stderr));
            }

            // Get the result
            let result = repl
                .get_variable("result")
                .await
                .map_err(|e| format!("Failed to get result: {e}"))?
                .unwrap_or_else(|| "No result".to_string());

            println!("{indent}  -> Result: {result}");

            Ok(result)
        })
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Recursive REPL Callback Demo ===\n");
    println!("This example demonstrates nested Python REPL environments.");
    println!("Each llm_query call creates a NEW REPL with its own context.\n");
    println!("Max depth: {MAX_DEPTH}\n");

    // Create the root REPL
    let root_repl = PythonRepl::new()?;

    // Inject the recursive llm_query starting at depth 0
    root_repl
        .inject_function("llm_query", make_recursive_callback(0, MAX_DEPTH))
        .await?;

    println!("--- Test 1: Simple query (no recursion) ---\n");
    let result = root_repl
        .execute("response = llm_query('What is 2+2?')")
        .await?;
    println!("\nExecution success: {}", result.success);
    if !result.stdout.is_empty() {
        println!("Stdout: {}", result.stdout);
    }

    let response = root_repl.get_variable("response").await?;
    println!("Response: {:?}\n", response);

    println!("--- Test 2: Query with context ---\n");
    let result = root_repl
        .execute(
            r#"
context_text = "The Divine Comedy is an epic poem by Dante Alighieri."
response2 = llm_query('Summarize this text', context=context_text)
"#,
        )
        .await?;
    println!("\nExecution success: {}", result.success);

    let response2 = root_repl.get_variable("response2").await?;
    println!("Response: {:?}\n", response2);

    println!("--- Test 3: Multiple sequential calls ---\n");
    println!("Demonstrating that each call gets its own REPL environment...\n");

    // Reset call counter
    TOTAL_CALLS.store(0, Ordering::SeqCst);

    let result = root_repl
        .execute(
            r#"
# Multiple calls to llm_query - each creates a NEW isolated REPL
results = []
for topic in ['Introduction', 'Chapter 1', 'Conclusion']:
    r = llm_query(f'Summarize {topic}', context=f'Content of {topic}...')
    results.append(r)
"#,
        )
        .await?;

    println!("\nExecution success: {}", result.success);
    if !result.stderr.is_empty() {
        println!("Stderr: {}", result.stderr);
    }

    let results = root_repl.get_variable("results").await?;
    println!("Results list: {:?}", results);

    println!("\n--- Test 4: Demonstrating depth limit ---\n");
    println!("Creating a deeply nested scenario to hit MAX_DEPTH...\n");

    // Reset call counter
    TOTAL_CALLS.store(0, Ordering::SeqCst);

    // Create a fresh REPL that will execute code which calls llm_query,
    // and that llm_query's child REPL also calls llm_query, etc.
    let deep_repl = PythonRepl::new()?;

    // Inject a callback that ALSO executes code calling llm_query in its child REPL
    fn make_deeply_recursive_callback(depth: u32, max_depth: u32) -> PythonCallback {
        Box::new(move |args: Vec<String>, kwargs: HashMap<String, String>| {
            Box::pin(async move {
                let indent = "  ".repeat(depth as usize);
                println!("{indent}>>> Entering depth {depth}");

                if depth >= max_depth {
                    println!("{indent}<<< MAX_DEPTH reached at {depth}");
                    return Ok(format!("BOTTOM (depth={})", depth));
                }

                let query = args.first().cloned().unwrap_or_default();
                let context = kwargs.get("context").cloned();

                // Create child REPL
                let child_repl =
                    PythonRepl::new().map_err(|e| format!("REPL creation failed: {e}"))?;

                // Inject context
                if let Some(ctx) = &context {
                    child_repl
                        .set_variable("context", ctx)
                        .await
                        .map_err(|e| format!("Set context failed: {e}"))?;
                }

                // Inject llm_query that will recurse further
                child_repl
                    .inject_function(
                        "llm_query",
                        make_deeply_recursive_callback(depth + 1, max_depth),
                    )
                    .await
                    .map_err(|e| format!("Inject failed: {e}"))?;

                // Execute Python code that calls llm_query AGAIN
                println!("{indent}    Executing Python that calls llm_query...");
                let result = child_repl
                    .execute(&format!(
                        r#"
# This Python code calls llm_query, triggering recursion!
deeper_result = llm_query('Going deeper from depth {}', context='nested context')
result = f"Depth {}: processed '{}' -> {{deeper_result}}"
"#,
                        depth, depth, query
                    ))
                    .await
                    .map_err(|e| format!("Exec failed: {e}"))?;

                if !result.success {
                    return Err(format!("Python error: {}", result.stderr));
                }

                let final_result = child_repl
                    .get_variable("result")
                    .await
                    .map_err(|e| format!("Get result failed: {e}"))?
                    .unwrap_or_else(|| "No result".to_string());

                println!("{indent}<<< Returning from depth {depth}");
                Ok(final_result)
            })
        })
    }

    deep_repl
        .inject_function("llm_query", make_deeply_recursive_callback(0, MAX_DEPTH))
        .await?;

    let result = deep_repl
        .execute("final = llm_query('Start the recursion', context='initial context')")
        .await?;

    println!("\nExecution success: {}", result.success);
    if !result.stderr.is_empty() {
        println!("Stderr: {}", result.stderr);
    }

    let final_result = deep_repl.get_variable("final").await?;
    println!("\nFinal nested result: {:?}", final_result);

    println!("\n--- Summary ---");
    println!(
        "Total llm_query calls made: {}",
        TOTAL_CALLS.load(Ordering::SeqCst)
    );
    println!("Each call created its own isolated Python REPL environment.");

    Ok(())
}
