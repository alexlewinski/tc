//! gRPC image generation service: protocol types, server and client.

pub mod client;
pub mod server;

pub mod proto {
    #![allow(clippy::all)]
    tonic::include_proto!("_");
}

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use proto::image_generation_service_server::ImageGenerationServiceServer;
use tonic::transport::{Identity, Server, ServerTlsConfig};

pub use server::{ImageService, ServiceOptions};

/// Maximum gRPC message size (images and uploads can be large).
pub const MAX_MESSAGE_SIZE: usize = 1 << 30;

/// A self-signed certificate for TLS (clients of this protocol do not verify it).
pub fn self_signed_identity() -> Result<Identity> {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).context("generating certificate")?;
    Ok(Identity::from_pem(cert.cert.pem(), cert.key_pair.serialize_pem()))
}

/// Serve until `shutdown` resolves.
pub async fn serve(
    addr: SocketAddr,
    worker: Arc<tc_service::Worker>,
    options: ServiceOptions,
    tls: bool,
    shutdown: impl std::future::Future<Output = ()>,
) -> Result<()> {
    let svc = ImageGenerationServiceServer::new(ImageService::new(worker, options))
        .max_decoding_message_size(MAX_MESSAGE_SIZE)
        .max_encoding_message_size(MAX_MESSAGE_SIZE);
    let mut builder = Server::builder();
    if tls {
        builder = builder.tls_config(ServerTlsConfig::new().identity(self_signed_identity()?))?;
    }
    builder.add_service(svc).serve_with_shutdown(addr, shutdown).await?;
    Ok(())
}
