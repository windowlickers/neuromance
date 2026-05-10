#![allow(clippy::unwrap_used, clippy::expect_used, clippy::pedantic)]
//! Resource usage benchmarks for PythonRepl.
//!
//! Tracks memory, file descriptors, and timing for REPL operations.
//!
//! # Running Benchmarks
//!
//! ```sh
//! # Run all benchmarks
//! cargo bench --features python -p neuromance-repl
//!
//! # Run specific benchmark group
//! cargo bench --features python -p neuromance-repl -- nested_repl_depth
//! cargo bench --features python -p neuromance-repl -- resource_usage
//! cargo bench --features python -p neuromance-repl -- execution_throughput
//!
//! # View HTML reports (after running)
//! open target/criterion/report/index.html
//! ```
//!
//! # Flamegraph Profiling
//!
//! Requires: `cargo install flamegraph` and perf (Linux) or DTrace (macOS)
//!
//! ```sh
//! # Profile all benchmarks
//! cargo flamegraph --bench repl_resources --features python -p neuromance-repl -- --bench
//!
//! # Profile specific benchmark (faster iteration)
//! cargo flamegraph --bench repl_resources --features python -p neuromance-repl -- \
//!     --bench nested_repl_depth
//!
//! # With sudo for perf access (Linux)
//! sudo cargo flamegraph --bench repl_resources --features python -p neuromance-repl -- --bench
//!
//! # Output: flamegraph.svg in current directory
//! ```
//!
//! # Stress Test (for finding resource limits)
//!
//! ```sh
//! # Basic stress test
//! cargo run --example depth_stress_test --features python --release
//!
//! # With callbacks and execution (more realistic)
//! WITH_CALLBACKS=1 WITH_EXECUTION=1 cargo run --example depth_stress_test --features python --release
//!
//! # Custom depth limits
//! MAX_DEPTH=1000 STEP_SIZE=50 cargo run --example depth_stress_test --features python --release
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use neuromance_repl::PythonRepl;
use std::collections::HashMap;
use sysinfo::{Pid, System};

/// Get current process memory usage in bytes
fn get_memory_usage() -> u64 {
    let mut sys = System::new();
    let pid = Pid::from_u32(std::process::id());
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);

    sys.process(pid).map_or(0, |p| p.memory())
}

/// Count open file descriptors (Linux only)
#[cfg(target_os = "linux")]
fn count_file_descriptors() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .map(|entries| entries.count())
        .unwrap_or(0)
}

#[cfg(not(target_os = "linux"))]
fn count_file_descriptors() -> usize {
    0 // Not supported on non-Linux
}

/// Benchmark single REPL creation
fn bench_repl_creation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");

    c.bench_function("repl_creation", |b| {
        b.iter(|| {
            let repl = PythonRepl::new().expect("Failed to create REPL");
            black_box(repl)
        })
    });

    // Also measure with immediate execution (more realistic)
    c.bench_function("repl_creation_with_exec", |b| {
        b.iter(|| {
            let repl = PythonRepl::new().expect("Failed to create REPL");
            let _ = rt.block_on(async {
                let result = repl.execute("x = 1 + 1").await.expect("Execution failed");
                black_box(result)
            });
            black_box(repl)
        })
    });
}

/// Benchmark nested REPL creation at various depths
fn bench_nested_repl_depth(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");

    let mut group = c.benchmark_group("nested_repl_depth");

    for depth in [1, 5, 10, 25, 50, 100].iter() {
        group.throughput(Throughput::Elements(*depth as u64));
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {
            b.iter(|| {
                // Create `depth` REPLs and store them (simulating recursion)
                let repls: Vec<PythonRepl> = (0..depth)
                    .map(|_| PythonRepl::new().expect("Failed to create REPL"))
                    .collect();

                // Execute simple code in each to initialize them
                rt.block_on(async {
                    for (i, repl) in repls.iter().enumerate() {
                        let _ = repl
                            .execute(&format!("depth = {i}"))
                            .await
                            .expect("Execution failed");
                    }
                });

                black_box(repls)
            })
        });
    }

    group.finish();
}

/// Stress test: measure resource usage at various depths
fn bench_resource_usage_by_depth(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");

    let mut group = c.benchmark_group("resource_usage");
    group.sample_size(10); // Fewer samples since this is expensive

    for depth in [10, 25, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("memory_and_fds", depth),
            depth,
            |b, &depth| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let mem_before = get_memory_usage();
                        let fds_before = count_file_descriptors();

                        let start = std::time::Instant::now();

                        // Create nested REPLs
                        let repls: Vec<PythonRepl> = (0..depth)
                            .map(|_| PythonRepl::new().expect("Failed to create REPL"))
                            .collect();

                        // Initialize each with a callback and execution
                        rt.block_on(async {
                            for (i, repl) in repls.iter().enumerate() {
                                repl.inject_function(
                                    "callback",
                                    move |_args, _kwargs: HashMap<String, String>| {
                                        Box::pin(async move { Ok(format!("depth_{i}")) })
                                    },
                                )
                                .expect("Inject failed");

                                let _ = repl
                                    .execute("result = callback()")
                                    .await
                                    .expect("Execution failed");
                            }
                        });

                        total_duration += start.elapsed();

                        let mem_after = get_memory_usage();
                        let fds_after = count_file_descriptors();

                        // Print resource delta (only first iteration to avoid spam)
                        if iters == 1 {
                            let mem_delta_kb = (mem_after.saturating_sub(mem_before)) / 1024;
                            let fds_delta = fds_after.saturating_sub(fds_before);
                            eprintln!(
                                "\n[depth={}] Memory delta: {} KB, FD delta: {}",
                                depth, mem_delta_kb, fds_delta
                            );
                            eprintln!(
                                "  Per-REPL: ~{} KB, ~{:.2} FDs",
                                mem_delta_kb / depth as u64,
                                fds_delta as f64 / depth as f64
                            );
                        }

                        // Let REPLs drop
                        drop(repls);
                    }

                    total_duration
                })
            },
        );
    }

    group.finish();
}

/// Benchmark callback injection overhead
fn bench_callback_injection(c: &mut Criterion) {
    let mut group = c.benchmark_group("callback_injection");

    // Single callback
    group.bench_function("single_callback", |b| {
        let repl = PythonRepl::new().expect("Failed to create REPL");
        b.iter(|| {
            repl.inject_function("test_fn", |_args, _kwargs: HashMap<String, String>| {
                Box::pin(async move { Ok("result".to_string()) })
            })
            .expect("Inject failed");
        })
    });

    // Multiple callbacks (10)
    group.bench_function("ten_callbacks", |b| {
        b.iter(|| {
            let repl = PythonRepl::new().expect("Failed to create REPL");
            for i in 0..10 {
                repl.inject_function(
                    &format!("fn_{i}"),
                    move |_args, _kwargs: HashMap<String, String>| {
                        Box::pin(async move { Ok(format!("result_{i}")) })
                    },
                )
                .expect("Inject failed");
            }
            black_box(repl)
        })
    });

    group.finish();
}

/// Benchmark execution throughput
fn bench_execution_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");

    let mut group = c.benchmark_group("execution_throughput");

    // Simple arithmetic
    group.bench_function("simple_arithmetic", |b| {
        let repl = PythonRepl::new().expect("Failed to create REPL");
        b.iter(|| {
            rt.block_on(async {
                let result = repl.execute("x = 1 + 1").await.expect("Exec failed");
                black_box(result)
            })
        })
    });

    // With callback invocation
    group.bench_function("with_callback_call", |b| {
        let repl = PythonRepl::new().expect("Failed to create REPL");
        repl.inject_function("get_value", |_args, _kwargs: HashMap<String, String>| {
            Box::pin(async move { Ok("42".to_string()) })
        })
        .expect("Inject failed");

        b.iter(|| {
            rt.block_on(async {
                let result = repl
                    .execute("result = get_value()")
                    .await
                    .expect("Exec failed");
                black_box(result)
            })
        })
    });

    // Variable get/set
    group.bench_function("variable_roundtrip", |b| {
        let repl = PythonRepl::new().expect("Failed to create REPL");
        b.iter(|| {
            rt.block_on(async {
                repl.set_variable("input", "hello")
                    .await
                    .expect("Set failed");
                let val = repl.get_variable("input").await.expect("Get failed");
                black_box(val)
            })
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_repl_creation,
    bench_nested_repl_depth,
    bench_resource_usage_by_depth,
    bench_callback_injection,
    bench_execution_throughput,
);

criterion_main!(benches);
