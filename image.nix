{
  pkgs,
  neuromance-runtime,
  version,
  variant ? "minimal",
  extraTools ? [ ],
  includeShell ? false,
}:

let
  shellPackages = pkgs.lib.optionals includeShell [ pkgs.bashInteractive ];

  shellExtraCommands = pkgs.lib.optionalString includeShell ''
    ln -sf bash bin/sh
  '';
in
pkgs.dockerTools.buildLayeredImage {
  name = "neuromance-runtime";
  tag = "${version}-${variant}";

  contents = [
    neuromance-runtime
    pkgs.cacert
  ] ++ shellPackages ++ extraTools;

  extraCommands = ''
    mkdir -p tmp var/tmp etc/neuromance bin
    chmod 1777 tmp var/tmp
  '' + shellExtraCommands;

  config = {
    Entrypoint = [ "${neuromance-runtime}/bin/neuromance-runtime" ];
    WorkingDir = "/";
    User = "65532:65532";

    ExposedPorts = {
      "8080/tcp" = { };
      "8081/tcp" = { };
    };

    Env = [
      "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      "SSL_CERT_DIR=${pkgs.cacert}/etc/ssl/certs"
      "NEUROMANCE_CONFIG=/etc/neuromance/config.toml"
      "RUST_BACKTRACE=1"
    ];

    Labels = {
      "org.opencontainers.image.title" = "Neuromance Runtime (${variant})";
      "org.opencontainers.image.description" = "Container runtime that executes a Neuromance agent from config (oneshot or serve mode)";
      "org.opencontainers.image.version" = version;
      "org.opencontainers.image.source" = "https://github.com/windowlickers/neuromance";
      "org.opencontainers.image.licenses" = "Apache-2.0";
    };
  };
}
