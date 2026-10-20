//! Minimal client used by `draw-things-cli generate --remote`.

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};

use crate::proto::image_generation_service_client::ImageGenerationServiceClient;
use crate::proto::*;
use crate::MAX_MESSAGE_SIZE;

pub struct RemoteClient {
    inner: ImageGenerationServiceClient<Channel>,
    shared_secret: Option<String>,
}

/// Result of a remote generation.
pub struct RemoteResult {
    pub images: Vec<tc_wire::WireTensor>,
}

impl RemoteClient {
    /// Connect to `host:port`. With `tls`, the server certificate must chain to a trusted root
    /// unless `TONIC_CLAP_TLS_CA` names a PEM file to trust.
    pub async fn connect(host: &str, port: u16, tls: bool, shared_secret: Option<String>) -> Result<RemoteClient> {
        let scheme = if tls { "https" } else { "http" };
        let host = host.trim_start_matches("http://").trim_start_matches("https://");
        let mut ep = Endpoint::from_shared(format!("{scheme}://{host}:{port}"))?;
        if tls {
            let mut cfg = ClientTlsConfig::new().with_enabled_roots();
            if let Ok(ca) = std::env::var("TONIC_CLAP_TLS_CA") {
                let pem = std::fs::read(&ca).with_context(|| format!("reading {ca}"))?;
                cfg = cfg.ca_certificate(tonic::transport::Certificate::from_pem(pem));
            }
            ep = ep.tls_config(cfg)?;
        }
        let channel = ep.connect().await.with_context(|| format!("connecting to {host}:{port}"))?;
        let inner = ImageGenerationServiceClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE);
        Ok(RemoteClient { inner, shared_secret })
    }

    pub async fn echo(&mut self, name: &str) -> Result<EchoReply> {
        let r = self
            .inner
            .echo(EchoRequest { name: name.into(), shared_secret: self.shared_secret.clone() })
            .await?
            .into_inner();
        if r.shared_secret_missing {
            bail!("server requires a (different) shared secret");
        }
        Ok(r)
    }

    /// Run a generation. `image` and `references` are wire tensors (see `tc-wire`).
    pub async fn generate(
        &mut self,
        configuration: Vec<u8>,
        prompt: &str,
        negative_prompt: &str,
        image: Option<Vec<u8>>,
        references: Vec<Vec<u8>>,
        mut on_progress: impl FnMut(&ImageGenerationSignpostProto),
    ) -> Result<RemoteResult> {
        let mut contents = vec![];
        let mut by_hash = |data: Vec<u8>| {
            let h = Sha256::digest(&data).to_vec();
            contents.push(data);
            h
        };
        let image = image.map(&mut by_hash);
        let hints = if references.is_empty() {
            vec![]
        } else {
            vec![HintProto {
                hint_type: "shuffle".into(),
                tensors: references.into_iter().map(|r| TensorAndWeight { tensor: by_hash(r), weight: 1.0 }).collect(),
            }]
        };
        let req = ImageGenerationRequest {
            image,
            scale_factor: 1,
            mask: None,
            hints,
            prompt: prompt.into(),
            negative_prompt: negative_prompt.into(),
            configuration,
            r#override: Some(MetadataOverride::default()),
            keywords: vec![],
            user: "draw-things-cli".into(),
            device: DeviceType::Laptop as i32,
            contents,
            shared_secret: self.shared_secret.clone(),
            chunked: true,
        };
        let mut stream = self.inner.generate_image(req).await?.into_inner();
        let mut images = vec![];
        let mut pending: Vec<u8> = vec![];
        while let Some(msg) = stream.message().await? {
            if let Some(s) = &msg.current_signpost {
                on_progress(s);
            }
            for data in msg.generated_images {
                pending.extend_from_slice(&data);
                if msg.chunk_state == ChunkState::LastChunk as i32 {
                    images.push(tc_wire::decode(&pending)?);
                    pending.clear();
                }
            }
        }
        Ok(RemoteResult { images })
    }
}
