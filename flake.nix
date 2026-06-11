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

        # Git revision of the flake input. `self.rev` is set for clean git
        # trees; `self.dirtyRev` is set when uncommitted changes are present.
        # Falls back to "unknown" for non-git sources (e.g. tarball builds).
        revision = self.rev or self.dirtyRev or "unknown";

        # Use Rust version from rust-toolchain.toml
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        # Like `cleanCargoSource`, but also keeps sqlx's embedded migrations
        # and the committed `.sqlx/` offline query metadata, which the default
        # filter would strip (silently breaking the query macros in CI only).
        src = pkgs.lib.cleanSourceWith {
          src = ./.;
          name = "source";
          filter =
            path: type:
            (craneLib.filterCargoSources path type)
            || (builtins.match ".*/migrations/.*\\.sql$" path != null)
            || (builtins.match ".*/\\.sqlx/query-.*\\.json$" path != null);
        };

        # Default Python package set for the embedded REPL — empty by design.
        # Downstream flakes pass a custom selector to `lib.mkRuntimeImage` to
        # bundle additional packages (e.g. `ps: with ps; [ requests numpy ]`).
        defaultPythonPackages = _ps: [ ];

        defaultPythonEnv = pkgs.python3.withPackages defaultPythonPackages;

        mkCommonArgs = pythonEnv: {
          inherit src;
          pname = "neuromance";
          inherit version;
          buildInputs = [ pkgs.openssl pythonEnv ];
          nativeBuildInputs = [ pkgs.pkg-config ];
          PYO3_PYTHON = "${pythonEnv}/bin/python3";
          # Compile sqlx query macros from the committed .sqlx/ metadata —
          # build sandboxes have no database.
          SQLX_OFFLINE = "true";
        };

        commonArgs = mkCommonArgs defaultPythonEnv;

        # Build deps with `--all-features` so every downstream package and
        # check reuses the same artifact. Without this, feature-gated deps
        # (pyo3 from runtime's `python-repl`) get recompiled in each package
        # build.
        #
        # Parameterized by `pythonEnv` because PYO3_PYTHON keys pyo3's build
        # script; runtimes built against a custom pythonEnv need their own
        # deps artifact. Nix dedupes identical inputs, so callers using
        # `defaultPythonEnv` share the cached `cargoArtifacts` below.
        mkCargoArtifacts =
          pythonEnv:
          craneLib.buildDepsOnly (
            (mkCommonArgs pythonEnv)
            // {
              cargoExtraArgs = "--all-features";
            }
          );

        cargoArtifacts = mkCargoArtifacts defaultPythonEnv;

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

        # Build a runtime binary linked against `pythonEnv`. PyO3 picks up the
        # interpreter via `PYO3_PYTHON`; the embedded REPL loads its
        # `libpython3.so` from the same store path at runtime.
        mkRuntime =
          {
            pname,
            pythonEnv,
          }:
          craneLib.buildPackage (
            (mkCommonArgs pythonEnv)
            // {
              cargoArtifacts = mkCargoArtifacts pythonEnv;
              inherit pname;
              cargoExtraArgs = "--package neuromance-runtime --features python-repl";
              meta = runtimeMeta;
            }
          );

        # Bundle a pre-built `neuromance-runtime` binary with its matching
        # `pythonEnv` into a container image. The runtime and pythonEnv must
        # have been built together so the dlopened libpython matches.
        mkRuntimeImage =
          {
            runtime,
            pythonEnv,
            variant ? "minimal",
            extraTools ? [ ],
            includeShell ? false,
          }:
          import ./image.nix {
            inherit
              pkgs
              version
              revision
              variant
              extraTools
              includeShell
              pythonEnv
              ;
            package = runtime;
            binName = "neuromance-runtime";
            title = "Neuromance Runtime";
            description = "Container runtime that executes a Neuromance agent from config (oneshot or serve mode)";
            exposedPorts = {
              "8080/tcp" = { };
              "8081/tcp" = { };
            };
            extraEnv = [ "NEUROMANCE_CONFIG=/etc/neuromance/config.toml" ];
            configDir = "etc/neuromance";
          };

        # Convenience: build runtime + image together from a `pythonPackages`
        # selector. Returns `{ runtime, image }`.
        mkRuntimeBundle =
          {
            variant ? "minimal",
            pythonPackages ? defaultPythonPackages,
            extraTools ? [ ],
            includeShell ? false,
          }:
          let
            pythonEnv = pkgs.python3.withPackages pythonPackages;
            runtime = mkRuntime {
              pname = "neuromance-runtime-${variant}";
              inherit pythonEnv;
            };
          in
          {
            inherit runtime;
            image = mkRuntimeImage {
              inherit
                runtime
                pythonEnv
                variant
                extraTools
                includeShell
                ;
            };
          };

        minimalBundle = mkRuntimeBundle { variant = "minimal"; };

        toolkitTools = with pkgs; [
          busybox
          coreutils
          findutils
          gnused
          gnugrep
          gawk
          gnutar
          gzip
          less
          procps
          which
          curl
          git
          openssh
          jq
          nodejs
          python3
          nix
        ];

        toolkitBundle = mkRuntimeBundle {
          variant = "toolkit";
          extraTools = toolkitTools;
          includeShell = true;
        };

        neuromance-runtime = minimalBundle.runtime;
        neuromance-runtime-toolkit = toolkitBundle.runtime;
        # `floatingTag` is the moving alias `release` publishes alongside the
        # immutable `imageTag` (e.g. `:minimal` always points at the latest
        # minimal build). `info` and `release` read it back via `nix eval`.
        neuromanceImage = minimalBundle.image // { floatingTag = "minimal"; };
        neuromanceImageToolkit = toolkitBundle.image // { floatingTag = "toolkit"; };

        # Generic image apps: each takes the image as an argument and looks up
        # `.#<image>-image` for metadata, replacing one app per (verb × image)
        # combo.
        # Single source of truth for the verb apps. `list` reads name/tag/
        # floating off each image at eval time and bakes them into its script
        # — no `nix eval` calls at runtime.
        imagePackages = {
          neuromance = neuromanceImage;
          neuromance-toolkit = neuromanceImageToolkit;
        };
        imageKeys = builtins.attrNames imagePackages;

        buildHelper = ''
          build_image() {
            local img="$1"
            if ! nix eval --raw ".#$img-image.imageName" >/dev/null 2>&1; then
              echo "Error: package .#$img-image not found in this flake" >&2
              exit 1
            fi
            nix build ".#$img-image"
          }
        '';

        load = pkgs.writeShellApplication {
          name = "load";
          runtimeInputs = with pkgs; [ docker nix ];
          text = ''
            ${buildHelper}
            if [ $# -ne 1 ]; then
              echo "usage: nix run .#load -- <image>" >&2
              exit 2
            fi
            build_image "$1"
            docker load < result
          '';
        };

        push = pkgs.writeShellApplication {
          name = "push";
          runtimeInputs = with pkgs; [ skopeo nix ];
          text = ''
            ${buildHelper}
            if [ $# -lt 3 ] || [ $# -gt 4 ]; then
              echo "usage: nix run .#push -- <image> <registry> <namespace> [tag]" >&2
              exit 2
            fi
            image="$1"
            registry="$2"
            namespace="$3"
            name=$(nix eval --raw ".#$image-image.imageName")
            tag="''${4:-$(nix eval --raw ".#$image-image.imageTag")}"
            build_image "$image"
            skopeo --insecure-policy copy "docker-archive:result" \
              "docker://$registry/$namespace/$name:$tag"
          '';
        };

        release = pkgs.writeShellApplication {
          name = "release";
          runtimeInputs = with pkgs; [ skopeo nix ];
          text = ''
            ${buildHelper}
            if [ $# -ne 3 ]; then
              echo "usage: nix run .#release -- <image> <registry> <namespace>" >&2
              exit 2
            fi
            image="$1"
            registry="$2"
            namespace="$3"
            build_image "$image"
            name=$(nix eval --raw ".#$image-image.imageName")
            version=$(nix eval --raw ".#$image-image.imageTag")
            floating=$(nix eval --raw ".#$image-image.floatingTag")
            echo "Publishing $name:$version (and :$floating)"
            skopeo --insecure-policy copy "docker-archive:result" \
              "docker://$registry/$namespace/$name:$version"
            skopeo --insecure-policy copy "docker-archive:result" \
              "docker://$registry/$namespace/$name:$floating"
          '';
        };

        inspect = pkgs.writeShellApplication {
          name = "inspect";
          runtimeInputs = with pkgs; [ docker dive nix ];
          text = ''
            ${buildHelper}
            if [ $# -ne 1 ]; then
              echo "usage: nix run .#inspect -- <image>" >&2
              exit 2
            fi
            image="$1"
            build_image "$image"
            docker load < result
            name=$(nix eval --raw ".#$image-image.imageName")
            tag=$(nix eval --raw ".#$image-image.imageTag")
            dive "$name:$tag"
          '';
        };

        list = pkgs.writeShellApplication {
          name = "list";
          runtimeInputs = [ ];
          text = ''
            if [ $# -ne 0 ]; then
              echo "usage: nix run .#list" >&2
              exit 2
            fi
            printf '%-22s %-22s %-15s %s\n' IMAGE NAME TAG FLOATING
            ${pkgs.lib.concatMapStringsSep "\n" (img:
              let m = imagePackages.${img}; in
              "printf '%-22s %-22s %-15s %s\\n' "
              + pkgs.lib.escapeShellArg img + " "
              + pkgs.lib.escapeShellArg m.imageName + " "
              + pkgs.lib.escapeShellArg m.imageTag + " "
              + pkgs.lib.escapeShellArg m.floatingTag
            ) imageKeys}
          '';
        };

        info = pkgs.writeShellApplication {
          name = "info";
          runtimeInputs = with pkgs; [ nix coreutils ];
          text = ''
            if [ $# -ne 1 ]; then
              echo "usage: nix run .#info -- <image>" >&2
              exit 2
            fi
            image="$1"
            if ! nix eval --raw ".#$image-image.imageName" >/dev/null 2>&1; then
              echo "Error: package .#$image-image not found in this flake" >&2
              exit 1
            fi
            name=$(nix eval --raw ".#$image-image.imageName")
            tag=$(nix eval --raw ".#$image-image.imageTag")
            floating=$(nix eval --raw ".#$image-image.floatingTag")
            store=$(nix eval --raw ".#$image-image")
            echo "name:     $name"
            echo "tag:      $tag"
            echo "floating: $floating"
            echo "store:    $store"
            if [ -e "$store" ]; then
              size=$(du -h "$store" | cut -f1)
              echo "size:     $size"
            fi
          '';
        };

        mkAppMeta =
          description: with pkgs.lib; {
            inherit description;
            license = licenses.asl20;
            platforms = platforms.linux;
          };

        mkCraneApp =
          name: drv:
          pkgs.writeShellScriptBin name ''
            echo "${name} ok: ${drv}"
          '';

      in
      {
        checks = {
          inherit
            neuromance
            neuromance-runtime
            neuromance-runtime-toolkit
            ;
          fmt = fmtCheck;
          clippy = clippyCheck;
          tests = testCheck;
        };

        packages = {
          inherit
            neuromance
            neuromance-runtime
            neuromance-runtime-toolkit
            ;
          neuromance-image = neuromanceImage;
          neuromance-toolkit-image = neuromanceImageToolkit;
          default = neuromance;
        };

        # Exposed for downstream flakes that want to bundle additional Python
        # packages into the runtime image:
        #
        #   neuromance.lib.${system}.mkRuntimeBundle {
        #     variant = "data-science";
        #     pythonPackages = ps: with ps; [ requests numpy pandas ];
        #   }
        #
        # Returns `{ runtime, image }`. Use `mkRuntime` / `mkRuntimeImage`
        # directly for finer-grained control.
        lib = {
          inherit mkRuntime mkRuntimeImage mkRuntimeBundle;
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
          list = {
            type = "app";
            program = "${list}/bin/list";
            meta = mkAppMeta "List available container images with their tags";
          };
          load = {
            type = "app";
            program = "${load}/bin/load";
            meta = mkAppMeta "Build a container image and docker load it";
          };
          push = {
            type = "app";
            program = "${push}/bin/push";
            meta = mkAppMeta "Build and push <image> to <registry>/<namespace>/<image>:[tag]";
          };
          release = {
            type = "app";
            program = "${release}/bin/release";
            meta = mkAppMeta "Build and push :<version> and :<floating> tags to the registry";
          };
          inspect = {
            type = "app";
            program = "${inspect}/bin/inspect";
            meta = mkAppMeta "Build, load, and open the image in dive";
          };
          info = {
            type = "app";
            program = "${info}/bin/info";
            meta = mkAppMeta "Show image name, tag, floating tag, store path, and tarball size";
          };
        };

        devShells.default = craneLib.devShell {
          checks = self.checks.${system};
          packages = [
            defaultPythonEnv
          ] ++ (with pkgs; [
            cargo-watch
            cargo-edit
            cargo-outdated
            cargo-audit
            cargo-expand
            cargo-release
            just
            skopeo
            dive
            sqlx-cli
            postgresql
          ]);

          RUST_BACKTRACE = "1";
          PYO3_PYTHON = "${defaultPythonEnv}/bin/python3";
        };

        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
