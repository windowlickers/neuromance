//! Stress test for nested REPL depth to find resource limits.
//!
//! This example progressively creates more nested REPLs until failure,
//! tracking memory, file descriptors, and timing at each step.
//!
//! Run with: cargo run --example depth_stress_test --features python --release
//!
//! Optional env vars:
//!   MAX_DEPTH=500       - Maximum depth to test (default: 500)
//!   STEP_SIZE=10        - Increment between measurements (default: 10)
//!   WITH_CALLBACKS=1    - Inject callbacks at each level (default: off)
//!   WITH_EXECUTION=1    - Execute code at each level (default: off)
//!
//! Optional flags:
//!   --context-file <path>  - Load a text file and inject as 'context' variable
//!
//! Example with large context:
//!   cargo run --example depth_stress_test --features python --release -- \
//!     --context-file /path/to/large_document.txt

use neuromance_repl::python::PythonRepl;
use neuromance_repl::ReplEnvironment;
use std::collections::HashMap;
use std::time::Instant;

#[cfg(target_os = "linux")]
fn count_file_descriptors() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .map(|entries| entries.count())
        .unwrap_or(0)
}

#[cfg(not(target_os = "linux"))]
fn count_file_descriptors() -> usize {
    0
}

fn get_memory_kb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // Read from /proc/self/statm for accuracy
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(rss_pages) = statm.split_whitespace().nth(1) {
                if let Ok(pages) = rss_pages.parse::<u64>() {
                    // Page size is typically 4KB
                    return pages * 4;
                }
            }
        }
    }
    0
}

#[allow(dead_code)]
struct ResourceSnapshot {
    depth: usize,
    memory_kb: u64,
    file_descriptors: usize,
    creation_time_ms: u128,
    total_time_ms: u128,
}

fn parse_context_file() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--context-file" && i + 1 < args.len() {
            let path = &args[i + 1];
            match std::fs::read_to_string(path) {
                Ok(content) => return Some(content),
                Err(e) => {
                    eprintln!("Error reading context file '{}': {}", path, e);
                    std::process::exit(1);
                }
            }
        }
        i += 1;
    }
    None
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let max_depth: usize = std::env::var("MAX_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    let step_size: usize = std::env::var("STEP_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let with_callbacks = std::env::var("WITH_CALLBACKS").is_ok();
    let with_execution = std::env::var("WITH_EXECUTION").is_ok();
    let context_content = parse_context_file();

    println!("=== REPL Depth Stress Test ===\n");
    println!("Configuration:");
    println!("  MAX_DEPTH: {max_depth}");
    println!("  STEP_SIZE: {step_size}");
    println!("  WITH_CALLBACKS: {with_callbacks}");
    println!("  WITH_EXECUTION: {with_execution}");
    if let Some(ref content) = context_content {
        println!("  CONTEXT_FILE: {} bytes", content.len());
    }
    println!();

    let baseline_mem = get_memory_kb();
    let baseline_fds = count_file_descriptors();

    println!("Baseline:");
    println!("  Memory: {} KB", baseline_mem);
    println!("  File descriptors: {}", baseline_fds);
    println!();

    println!(
        "{:>6} | {:>12} | {:>10} | {:>12} | {:>12} | {:>10} | {:>10}",
        "Depth", "Memory (KB)", "Mem Delta", "Per-REPL KB", "FDs", "FD Delta", "Time (ms)"
    );
    println!("{}", "-".repeat(90));

    let mut snapshots: Vec<ResourceSnapshot> = Vec::new();
    let mut last_success_depth = 0;

    for target_depth in (step_size..=max_depth).step_by(step_size) {
        let total_start = Instant::now();
        let creation_start = Instant::now();

        // Create REPLs up to target_depth
        let mut repls: Vec<PythonRepl> = Vec::with_capacity(target_depth);

        let mut failed = false;
        for i in 0..target_depth {
            match PythonRepl::new() {
                Ok(repl) => {
                    // Inject context variable if provided
                    if let Some(ref content) = context_content {
                        if let Err(e) = repl.set_variable("context", content).await {
                            eprintln!("\nContext injection failed at depth {i}: {e}");
                            failed = true;
                            break;
                        }
                    }

                    if with_callbacks {
                        let depth = i;
                        if let Err(e) = repl
                            .inject_function(
                                "llm_query",
                                Box::new(move |args: Vec<String>, _kwargs: HashMap<String, String>| {
                                    let d = depth;
                                    Box::pin(async move {
                                        Ok(format!("Response from depth {d}: {:?}", args))
                                    })
                                }),
                            )
                            .await
                        {
                            eprintln!("\nCallback injection failed at depth {i}: {e}");
                            failed = true;
                            break;
                        }
                    }

                    if with_execution {
                        // If context is provided, also verify it's accessible
                        let code = if context_content.is_some() {
                            format!("depth = {i}; context_len = len(context)")
                        } else {
                            format!("depth = {i}")
                        };
                        if let Err(e) = repl.execute(&code).await {
                            eprintln!("\nExecution failed at depth {i}: {e}");
                            failed = true;
                            break;
                        }
                    }

                    repls.push(repl);
                }
                Err(e) => {
                    eprintln!("\nREPL creation failed at depth {i}: {e}");
                    failed = true;
                    break;
                }
            }
        }

        if failed {
            println!("\n=== FAILURE at depth {} ===", repls.len());
            break;
        }

        let creation_time = creation_start.elapsed().as_millis();
        let actual_depth = repls.len();
        last_success_depth = actual_depth;

        let current_mem = get_memory_kb();
        let current_fds = count_file_descriptors();

        let mem_delta = current_mem.saturating_sub(baseline_mem);
        let fds_delta = current_fds.saturating_sub(baseline_fds);
        let per_repl_kb = if actual_depth > 0 {
            mem_delta / actual_depth as u64
        } else {
            0
        };

        let total_time = total_start.elapsed().as_millis();

        println!(
            "{:>6} | {:>12} | {:>+10} | {:>12} | {:>12} | {:>+10} | {:>10}",
            actual_depth, current_mem, mem_delta, per_repl_kb, current_fds, fds_delta, total_time
        );

        snapshots.push(ResourceSnapshot {
            depth: actual_depth,
            memory_kb: current_mem,
            file_descriptors: current_fds,
            creation_time_ms: creation_time,
            total_time_ms: total_time,
        });

        // Drop REPLs before next iteration
        drop(repls);

        // Small pause to let OS reclaim resources
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    println!();
    println!("=== Summary ===");
    println!("Maximum successful depth: {last_success_depth}");

    if let Some(last) = snapshots.last() {
        let mem_growth = last.memory_kb.saturating_sub(baseline_mem);
        println!(
            "Total memory growth at max depth: {} KB ({:.2} MB)",
            mem_growth,
            mem_growth as f64 / 1024.0
        );
        println!(
            "Average per-REPL memory: {} KB",
            mem_growth / last.depth as u64
        );
        println!(
            "File descriptor growth: {}",
            last.file_descriptors.saturating_sub(baseline_fds)
        );
    }

    // Check for concerning patterns
    println!();
    println!("=== Analysis ===");

    if last_success_depth >= max_depth {
        println!("All depths completed successfully. Consider increasing MAX_DEPTH.");
    } else {
        println!(
            "Hit limit at depth {}. This may indicate:",
            last_success_depth
        );
        println!("  - Memory exhaustion");
        println!("  - File descriptor limits (check `ulimit -n`)");
        println!("  - Python GIL contention");
    }

    // Print resource growth rate
    if snapshots.len() >= 2 {
        let first = &snapshots[0];
        let last = snapshots.last().unwrap();

        let depth_delta = (last.depth - first.depth) as u64;
        let mem_per_repl = last
            .memory_kb
            .saturating_sub(first.memory_kb)
            .checked_div(depth_delta)
            .unwrap_or(0);

        // Calculate time per REPL more carefully to avoid overflow
        let time_delta_ms = last.total_time_ms.saturating_sub(first.total_time_ms);
        let time_per_repl_us = if depth_delta > 0 {
            (time_delta_ms * 1000) / depth_delta as u128
        } else {
            0
        };

        println!();
        println!("Growth rates:");
        println!("  Memory: ~{mem_per_repl} KB per REPL");
        println!(
            "  Time: ~{:.2} ms per REPL",
            time_per_repl_us as f64 / 1000.0
        );

        // Estimate safe limits
        let available_mem_kb = 1024 * 1024; // Assume 1GB available
        let safe_depth = available_mem_kb / mem_per_repl.max(1);
        println!();
        println!(
            "Estimated safe depth (1GB memory budget): {} REPLs",
            safe_depth
        );
    }

    Ok(())
}
