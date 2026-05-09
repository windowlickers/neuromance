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

        python = pkgs.python3;

        commonBuildInputs = with pkgs; [ openssl python ];
        commonNativeBuildInputs = with pkgs; [ pkg-config ];

        commonArgs = {
          inherit src;
          pname = "neuromance";
          inherit version;
          buildInputs = commonBuildInputs;
          nativeBuildInputs = commonNativeBuildInputs;
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        fmtCheck = craneLib.cargoFmt { inherit src; };

        clippyCheck = craneLib.cargoClippy (
          commonArgs
          // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--all-targets --all-features -- -D warnings";
          }
        );

        testCheck = craneLib.cargoTest (
          commonArgs
          // {
            inherit cargoArtifacts;
            cargoTestExtraArgs = "--all-features";
          }
        );

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

        runtimeMeta = with pkgs.lib; {
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

        neuromance-runtime = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            pname = "neuromance-runtime";
            cargoExtraArgs = "--package neuromance-runtime";
            meta = runtimeMeta;
          }
        );

        neuromance-runtime-toolkit = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            pname = "neuromance-runtime-toolkit";
            cargoExtraArgs = "--package neuromance-runtime --features python-repl";
            meta = runtimeMeta;
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
          inherit pkgs version;
          neuromance-runtime = neuromance-runtime-toolkit;
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

        mkAppMeta = description: with pkgs.lib; {
          inherit description;
          license = licenses.asl20;
          platforms = platforms.linux;
        };

        mkCraneApp = name: drv: pkgs.writeShellScriptBin name ''
          echo "${name} ok: ${drv}"
        '';

      in
      {
        checks = {
          inherit neuromance neuromance-runtime neuromance-runtime-toolkit;
          fmt = fmtCheck;
          clippy = clippyCheck;
          tests = testCheck;
        };

        packages = {
          inherit neuromance neuromance-runtime neuromance-runtime-toolkit;
          neuromance-image = neuromanceImage;
          neuromance-image-toolkit = neuromanceImageToolkit;
          default = neuromance;
        };

        apps = {
          fmt = {
            type = "app";
            program = "${mkCraneApp "fmt" fmtCheck}/bin/fmt";
            meta = mkAppMeta "Run cargo fmt via crane (cached)";
          };
          clippy = {
            type = "app";
            program = "${mkCraneApp "clippy" clippyCheck}/bin/clippy";
            meta = mkAppMeta "Run cargo clippy via crane (cached)";
          };
          test = {
            type = "app";
            program = "${mkCraneApp "test" testCheck}/bin/test";
            meta = mkAppMeta "Run cargo test via crane (cached)";
          };
          build = {
            type = "app";
            program = "${mkCraneApp "build" neuromance}/bin/build";
            meta = mkAppMeta "Build neuromance via crane (cached)";
          };
          load-minimal = {
            type = "app";
            program = "${loadMinimal}/bin/load-minimal";
            meta = mkAppMeta "Load the minimal neuromance-runtime image into Docker";
          };
          load-toolkit = {
            type = "app";
            program = "${loadToolkit}/bin/load-toolkit";
            meta = mkAppMeta "Load the toolkit neuromance-runtime image into Docker";
          };
          push-minimal = {
            type = "app";
            program = "${pushMinimal}/bin/push-minimal";
            meta = mkAppMeta "Push the minimal neuromance-runtime image to a registry";
          };
          push-toolkit = {
            type = "app";
            program = "${pushToolkit}/bin/push-toolkit";
            meta = mkAppMeta "Push the toolkit neuromance-runtime image to a registry";
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
            python
          ];

          RUST_BACKTRACE = "1";
          PYO3_PYTHON = "${python}/bin/python3";
        };

        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
