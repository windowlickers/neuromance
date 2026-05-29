{
  pkgs,
  package,
  binName,
  pythonEnv ? null,
  version,
  revision,
  title,
  description,
  variant ? null,
  imageName ? binName,
  exposedPorts ? { },
  extraEnv ? [ ],
  configDir ? null,
  extraTools ? [ ],
  includeShell ? false,
}:

let
  shellPackages = pkgs.lib.optionals includeShell [ pkgs.bashInteractive ];

  shellExtraCommands = pkgs.lib.optionalString includeShell ''
    ln -sf bash bin/sh
  '';

  configDirCommands = pkgs.lib.optionalString (configDir != null) ''
    mkdir -p ${configDir}
  '';

  # PyO3 dlopens libpython at startup, so when a pythonEnv is provided the
  # interpreter store path must be in the image and PYTHONPATH must point at
  # the bundled site-packages.
  pythonContents = pkgs.lib.optional (pythonEnv != null) pythonEnv;
  pythonEnvVars = pkgs.lib.optional (
    pythonEnv != null
  ) "PYTHONPATH=${pythonEnv}/${pythonEnv.sitePackages}";

  variantSuffix = pkgs.lib.optionalString (variant != null) "-${variant}";
  variantTitleSuffix = pkgs.lib.optionalString (variant != null) " (${variant})";
in
pkgs.dockerTools.buildLayeredImage {
  name = imageName;
  tag = "${version}${variantSuffix}";

  contents = [
    package
    pkgs.cacert
  ] ++ pythonContents ++ shellPackages ++ extraTools;

  extraCommands = ''
    mkdir -p tmp var/tmp bin etc etc/nix home/nonroot etc/ssl/certs
    chmod 1777 tmp var/tmp
    ln -sf ${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt etc/ssl/cert.pem
    ln -sf ${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt etc/ssl/certs/ca-certificates.crt
    cat > etc/passwd <<EOF
    root:x:0:0:root:/:/sbin/nologin
    nonroot:x:65532:65532:nonroot:/home/nonroot:/sbin/nologin
    EOF
    cat > etc/group <<EOF
    root:x:0:
    nonroot:x:65532:
    EOF
    cat > etc/nix/nix.conf <<EOF
    experimental-features = nix-command flakes pipe-operators
    sandbox = false
    build-users-group =
    store = /tmp/nix
    EOF
  '' + configDirCommands + shellExtraCommands;

  config = {
    Entrypoint = [ "${package}/bin/${binName}" ];
    WorkingDir = "/";
    User = "65532:65532";

    ExposedPorts = exposedPorts;

    Env = [
      "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      "SSL_CERT_DIR=${pkgs.cacert}/etc/ssl/certs"
      "GIT_SSL_CAINFO=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      "CURL_CA_BUNDLE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      "NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      "HOME=/home/nonroot"
      "XDG_CACHE_HOME=/home/nonroot/.cache"
      "XDG_CONFIG_HOME=/home/nonroot/.config"
      "XDG_STATE_HOME=/home/nonroot/.local/state"
      "RUST_BACKTRACE=1"
    ] ++ pythonEnvVars ++ extraEnv;

    Labels = {
      "org.opencontainers.image.title" = "${title}${variantTitleSuffix}";
      "org.opencontainers.image.description" = description;
      "org.opencontainers.image.version" = version;
      "org.opencontainers.image.revision" = revision;
      "org.opencontainers.image.source" = "https://github.com/windowlickers/neuromance";
      "org.opencontainers.image.licenses" = "Apache-2.0";
    };
  };
}
