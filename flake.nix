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

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      crane,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
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

        src = craneLib.cleanCargoSource ./.;

        commonBuildInputs = with pkgs; [ openssl ];
        commonNativeBuildInputs = with pkgs; [ pkg-config ];

        commonArgs = {
          inherit src;
          pname = "neuromance";
          inherit version;
          buildInputs = commonBuildInputs;
          nativeBuildInputs = commonNativeBuildInputs;
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        neuromance = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            cargoExtraArgs = "--all-features";
            meta = with pkgs.lib; {
              description = "A Rust library for controlling and orchestrating LLM interactions";
              homepage = "https://github.com/windowlickers/neuromance";
              license = licenses.asl20;
              maintainers = [
                {
                  name = "Evan Dobry";
                  email = "ecdobry@windowlicke.rs";
                  github = "ecdobry";
                  githubId = 16653165;
                }
              ];
            };
          }
        );

        neuromance-runtime = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            pname = "neuromance-runtime";
            cargoExtraArgs = "--package neuromance-runtime";
            meta = with pkgs.lib; {
              description = "Runtime that executes a Neuromance agent from config (oneshot or serve mode)";
              homepage = "https://github.com/windowlickers/neuromance";
              license = licenses.asl20;
              maintainers = [
                {
                  name = "Evan Dobry";
                  email = "ecdobry@windowlicke.rs";
                  github = "ecdobry";
                  githubId = 16653165;
                }
              ];
            };
          }
        );

        neuromanceImage = import ./image.nix {
          inherit pkgs neuromance-runtime version;
          variant = "minimal";
        };

        toolkitTools = with pkgs; [
          busybox
          coreutils
          git
          curl
          jq
          nodejs
          python3
        ];

        neuromanceImageToolkit = import ./image.nix {
          inherit pkgs neuromance-runtime version;
          variant = "toolkit";
          extraTools = toolkitTools;
          includeShell = true;
        };

        mkLoad = image: variant: pkgs.writeShellScriptBin "load-${variant}" ''
          set -euo pipefail
          echo "Loading neuromance-runtime:${version}-${variant} into Docker..."
          ${pkgs.docker}/bin/docker load < ${image}
          echo "Loaded neuromance-runtime:${version}-${variant}"
        '';

        defaultRegistry = "ghcr.io/windowlickers";
        mkPush = image: variant: pkgs.writeShellScriptBin "push-${variant}" ''
          set -euo pipefail
          registry="''${1:-''${REGISTRY:-${defaultRegistry}}}"
          echo "Pushing to $registry/neuromance-runtime:${version}-${variant}..."
          ${pkgs.skopeo}/bin/skopeo --insecure-policy copy \
            docker-archive:${image} \
            "docker://$registry/neuromance-runtime:${version}-${variant}"
          echo "Pushing to $registry/neuromance-runtime:${variant}..."
          ${pkgs.skopeo}/bin/skopeo --insecure-policy copy \
            docker-archive:${image} \
            "docker://$registry/neuromance-runtime:${variant}"
          echo "Pushed ${version}-${variant}"
        '';

        loadMinimal = mkLoad neuromanceImage "minimal";
        loadToolkit = mkLoad neuromanceImageToolkit "toolkit";
        pushMinimal = mkPush neuromanceImage "minimal";
        pushToolkit = mkPush neuromanceImageToolkit "toolkit";

      in
      {
        checks = {
          inherit neuromance neuromance-runtime;

          fmt = craneLib.cargoFmt { inherit src; };

          clippy = craneLib.cargoClippy (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets --all-features -- -D warnings";
            }
          );

          tests = craneLib.cargoTest (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoTestExtraArgs = "--all-features";
            }
          );
        };

        packages = {
          inherit neuromance neuromance-runtime;
          neuromance-image = neuromanceImage;
          neuromance-image-toolkit = neuromanceImageToolkit;
          default = neuromance;
        };

        apps = {
          load-minimal = {
            type = "app";
            program = "${loadMinimal}/bin/load-minimal";
          };
          load-toolkit = {
            type = "app";
            program = "${loadToolkit}/bin/load-toolkit";
          };
          push-minimal = {
            type = "app";
            program = "${pushMinimal}/bin/push-minimal";
          };
          push-toolkit = {
            type = "app";
            program = "${pushToolkit}/bin/push-toolkit";
          };
        };

        devShells.default = craneLib.devShell {
          checks = self.checks.${system};
          packages = with pkgs; [
            cargo-watch
            cargo-edit
            cargo-outdated
            cargo-audit
            cargo-expand
            just
            skopeo
            dive
          ];

          RUST_BACKTRACE = "1";
        };

        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
