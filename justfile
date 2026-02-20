# Run all CI checks locally (format, clippy, tests, build)
ci:
    nix flake check

# Run the CLI against a locally-built daemon (isolated from installed version)
dev *args:
    cargo build --bin neuromance-daemon
    NEUROMANCE_DATA_DIR=~/.local/share/neuromance-dev \
    NEUROMANCE_DAEMON_BIN=./target/debug/neuromance-daemon \
    cargo run --bin neuromance -- {{args}}

# Run the daemon in foreground for debugging (isolated data dir)
dev-daemon:
    NEUROMANCE_DATA_DIR=~/.local/share/neuromance-dev \
    cargo run --bin neuromance-daemon

# Stop the dev daemon
dev-stop:
    NEUROMANCE_DATA_DIR=~/.local/share/neuromance-dev \
    cargo run --bin neuromance -- daemon stop
