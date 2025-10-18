{
  description = "Neuromance - A Rust library for controlling and orchestrating LLM interactions";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Extract version from workspace Cargo.toml
        workspaceCargoToml = builtins.fromTOML (builtins.readFile ./Cargo.toml);
        version = workspaceCargoToml.workspace.package.version;

        # Use Rust version from rust-toolchain.toml
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        # Common build inputs for all packages (runtime dependencies)
        commonBuildInputs = with pkgs; [
          openssl
        ];

        # Common native build inputs (build-time tools)
        commonNativeBuildInputs = with pkgs; [
          rustToolchain
          pkg-config
        ] ++ lib.optionals stdenv.isDarwin [
          darwin.apple_sdk.frameworks.Security
          darwin.apple_sdk.frameworks.SystemConfiguration
        ];

      in
      {
        # Development shell
        devShells.default = pkgs.mkShell {
          buildInputs = commonBuildInputs ++ [
            pkgs.cargo-watch
            pkgs.cargo-edit
            pkgs.cargo-outdated
            pkgs.cargo-audit
            pkgs.cargo-expand
          ];

          nativeBuildInputs = commonNativeBuildInputs;

          # Environment variables
          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          RUST_BACKTRACE = "1";
        };

        # Build the neuromance-cli binary
        packages.neuromance-cli = pkgs.rustPlatform.buildRustPackage {
          pname = "neuromance-cli";
          inherit version;

          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          buildInputs = commonBuildInputs;
          nativeBuildInputs = commonNativeBuildInputs;

          # Build only the CLI package
          cargoBuildFlags = [ "--package" "neuromance-cli" ];
          cargoTestFlags = [ "--package" "neuromance-cli" ];

          # Run tests during build
          doCheck = true;

          meta = with pkgs.lib; {
            description = "CLI for Neuromance - A Rust library for controlling and orchestrating LLM interactions";
            homepage = "https://github.com/windowlickers/neuromance";
            license = licenses.asl20;
            maintainers = [
              {
                name = "Evan Dobry";
                email = "evandobry@gmail.com";
                github = "ecdobry";
                githubId = 16653165;
              }
            ];
          };
        };

        # Build the main neuromance library (default)
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "neuromance";
          inherit version;

          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          buildInputs = commonBuildInputs;
          nativeBuildInputs = commonNativeBuildInputs;

          # Run tests during build
          doCheck = true;

          meta = with pkgs.lib; {
            description = "A Rust library for controlling and orchestrating LLM interactions";
            homepage = "https://github.com/windowlickers/neuromance";
            license = licenses.asl20;
            maintainers = [
              {
                name = "Evan Dobry";
                email = "evandobry@gmail.com";
                github = "ecdobry";
                githubId = 16653165;
              }
            ];
          };
        };

        # Formatter
        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
