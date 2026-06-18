//! Compile the sandbox gRPC service definition into client and server stubs.
//!
//! Output lands in `OUT_DIR` and is pulled in via `tonic::include_proto!` from
//! `src/sandbox/proto.rs`. `protoc` must be on `PATH` (provided by
//! `nativeBuildInputs` in the Nix build).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/sandbox.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/sandbox.proto");
    Ok(())
}
