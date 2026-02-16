//! gRPC protocol definitions for neuromance daemon/CLI communication.
//!
//! This crate contains the protobuf-generated types and service definitions,
//! plus conversion traits between proto types and domain types from
//! `neuromance-common`.

mod convert;

#[allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    clippy::unwrap_used,
    clippy::expect_used
)]
pub mod proto {
    tonic::include_proto!("neuromance.v1");
}

pub use convert::*;
pub use proto::neuromance_client::NeuromanceClient;
pub use proto::neuromance_server::{Neuromance, NeuromanceServer};
