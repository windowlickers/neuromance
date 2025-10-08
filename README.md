# Neuromance

A Rust library for controlling and orchestrating LLM interactions.

[![Crates.io](https://img.shields.io/crates/v/neuromance.svg)](https://crates.io/crates/neuromance)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.90%2B-orange.svg)](https://www.rust-lang.org)

## Overview

Neuromance provides high-level abstractions for building LLM-powered applications in Rust.

- **`neuromance`** - Main library providing unified interface for LLM orchestration
- **`neuromance-common`** - Common types and data structures for conversations, messages, and tools
- **`neuromance-client`** - Client implementations for various LLM providers
- **`neuromance-agent`** - Agent framework for autonomous task execution with LLMs
- **`neuromance-tools`** - Tool execution framework with MCP support
- **`neuromance-cli`** - Interactive command-line interface for LLM interactions

## Development

### Prerequisites

- Rust 1.90 or higher
- Cargo

### Building

```bash
cargo build
```

### Testing

```bash
cargo test
```

### Linting

```bash
cargo clippy --all-targets --all-features
```

### Formatting

```bash
cargo fmt
```

## Workspace Structure

```
neuromance/
├── crates/
│   ├── neuromance/          # Main library
│   ├── neuromance-common/   # Common types and data structures
│   ├── neuromance-client/   # Client implementations
│   ├── neuromance-agent/    # Agent framework
│   ├── neuromance-tools/    # Tool execution framework
│   └── neuromance-cli/      # Command-line interface
├── Cargo.toml               # Workspace configuration
├── mcp_config.toml.example  # Example MCP configuration
└── README.md
```

## Model Context Protocol (MCP) Support

Neuromance supports the [Model Context Protocol](https://modelcontextprotocol.io/) for connecting to external tool servers. MCP allows LLMs to access tools like filesystem operations, database queries, web APIs, and more.

### Quick Start with MCP

1. Copy the example configuration:
   ```bash
   cp mcp_config.toml.example mcp_config.toml
   ```

2. Edit `mcp_config.toml` to configure your MCP servers

3. Use with the CLI:
   ```bash
   cargo run --bin neuromance-cli -- --mcp-config mcp_config.toml
   ```

See `mcp_config.toml.example` for detailed configuration examples.

## Contributing

Contributions are welcome.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- Evan Dobry ([@ecdobry](https://github.com/ecdobry))

## Acknowledgments

Named after William Gibson's novel Neuromancer.
