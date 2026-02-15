{
  description = "Neuromance - A Rust library for controlling and orchestrating LLM interactions";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, crane, flake-utils }:
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
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        # Common source filtering
        src = craneLib.cleanCargoSource ./.;

        # Common build inputs (runtime dependencies)
        commonBuildInputs = with pkgs; [ openssl ]
          ++ lib.optionals stdenv.isDarwin [
            darwin.apple_sdk.frameworks.Security
            darwin.apple_sdk.frameworks.SystemConfiguration
          ];

        # Common native build inputs (build-time tools)
        commonNativeBuildInputs = with pkgs; [ pkg-config ];

        # Common arguments for all builds
        commonArgs = {
          inherit src;
          pname = "neuromance";
          inherit version;
          buildInputs = commonBuildInputs;
          nativeBuildInputs = commonNativeBuildInputs;
        };

        # Build dependencies separately for caching
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        # The main crate build
        neuromance = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
          cargoExtraArgs = "--all-features";
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
        });

        # The CLI binary
        neuromance-cli = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
          cargoExtraArgs = "--package neuromance-cli";
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
        });

        # The daemon binary
        neuromance-daemon = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
          cargoExtraArgs = "--package neuromance-daemon";
          meta = with pkgs.lib; {
            description = "Long-running daemon for managing Neuromance conversations";
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
        });

      in
      {
        # `nix flake check` runs all of these
        checks = {
          inherit neuromance neuromance-cli neuromance-daemon;

          fmt = craneLib.cargoFmt { inherit src; };

          clippy = craneLib.cargoClippy (commonArgs // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--all-targets --all-features -- -D warnings";
          });

          tests = craneLib.cargoTest (commonArgs // {
            inherit cargoArtifacts;
            cargoTestExtraArgs = "--all-features";
          });
        };

        packages = {
          inherit neuromance neuromance-cli neuromance-daemon;
          default = neuromance;
          nm = neuromance-cli;  # Alias for the nm binary
        };

        devShells.default = craneLib.devShell {
          checks = self.checks.${system};
          packages = with pkgs; [
            cargo-watch
            cargo-edit
            cargo-outdated
            cargo-audit
            cargo-expand
          ];

          # Environment variables
          RUST_BACKTRACE = "1";
        };

        # Formatter
        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
