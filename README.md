# Neuromance

A Rust library for controlling and orchestrating LLM interactions.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

## Overview

Neuromance provides high-level abstractions for building LLM-powered applications in Rust.

- **`neuromance`** - Main library providing unified interface for LLM orchestration
- **`neuromance-common`** - Common types and data structures for conversations, messages, and tools
- **`neuromance-client`** - Client implementations for various LLM providers

## Development

### Prerequisites

- Rust 1.70 or higher
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
│   └── neuromance-client/   # Client implementations
├── Cargo.toml               # Workspace configuration
└── README.md
```

## Contributing

Contributions are welcome.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- Evan Dobry ([@ecdobry](https://github.com/ecdobry))

## Acknowledgments

Named after William Gibson's novel Neuromancer. 
