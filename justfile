# Run all CI checks locally (format, clippy, tests, build)
ci:
    @echo "Checking formatting..."
    cargo fmt -- --check
    @echo "Running clippy..."
    cargo clippy --all-targets --all-features -- -D warnings
    @echo "Running tests..."
    cargo test --all-features
    @echo "Building release..."
    cargo build --release --all-features
    @echo "Checking benchmarks compile..."
    cargo bench --no-run
    @echo "✓ All CI checks passed!"

