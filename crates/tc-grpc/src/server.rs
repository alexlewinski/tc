//! `ImageGenerationService` implementation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use log::{error, info, warn};
use sha2::{Digest, Sha256};
use tc_core::pipeline::{Cancellation, Progress};
use tc_service::{Event, GenerationInput, ModelStore, Worker};
use tc_wire::{Compression, WireTensor, WireType};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

use crate::proto::image_generation_service_server::ImageGenerationService;
use crate::proto::*;

/// Server behaviour switches (mirroring the `gRPCServerCLI` flags).
#[derive(Clone, Debug, Default)]
pub struct ServiceOptions {
    pub shared_secret: Option<String>,
    pub response_compression: bool,
    pub model_browser: bool,
}

pub struct ImageService {
    worker: Arc<Worker>,
    store: ModelStore,
    options: ServiceOptions,
    server_id: u64,
}

const CHUNK: usize = 4 * 1024 * 1024;

impl ImageService {
    pub fn new(worker: Arc<Worker>, options: ServiceOptions) -> ImageService {
        let store = worker.store.clone();
        ImageService { worker, store, options, server_id: rand::random() }
    }

    fn check_secret(&self, provided: Option<&str>) -> Result<(), Status> {
        match self.options.shared_secret.as_deref() {
            Some(s) if !s.is_empty() && provided != Some(s) => Err(Status::unauthenticated("shared secret mismatch")),
            _ => Ok(()),
        }
    }
}

fn signpost(p: &Progress) -> ImageGenerationSignpostProto {
    use image_generation_signpost_proto::Signpost;
    let s = match p {
        Progress::TextEncoded => Signpost::TextEncoded(Default::default()),
        Progress::ImageEncoded => Signpost::ImageEncoded(Default::default()),
        Progress::Sampling { step, .. } => Signpost::Sampling(image_generation_signpost_proto::Sampling { step: *step as i32 }),
        Progress::ImageDecoded => Signpost::ImageDecoded(Default::default()),
    };
    ImageGenerationSignpostProto { signpost: Some(s) }
}

/// Decode an image tensor (`[1, H, W, C]` or `[H, W, C]`, float in [-1, 1]) to RGB.
pub fn tensor_to_rgb(t: &WireTensor) -> anyhow::Result<image::RgbImage> {
    let s = &t.shape;
    let (h, w, c) = match s.len() {
        4 => (s[1], s[2], s[3]),
        3 => (s[0], s[1], s[2]),
        _ => anyhow::bail!("unexpected image tensor shape {s:?}"),
    };
    let v = t.to_f32();
    let scale_u8 = t.dtype == WireType::U8;
    let mut out = Vec::with_capacity(h * w * 3);
    for p in 0..h * w {
        for ch in 0..3 {
            let x = v[p * c + ch.min(c - 1)];
            let y = if scale_u8 { x } else { (x.clamp(-1.0, 1.0) + 1.0) * 127.5 };
            out.push(y.round().clamp(0.0, 255.0) as u8);
        }
    }
    Ok(image::RgbImage::from_raw(w as u32, h as u32, out).unwrap())
}

/// Encode an output image as an f16 NHWC tensor in [-1, 1].
pub fn image_tensor_bytes(img: &nnc::HostTensor, compression: Compression) -> Vec<u8> {
    let t = WireTensor::from_f32(&img.to_f32_vec(), img.shape(), WireType::F16);
    tc_wire::encode(&t, compression)
}

fn sanitize_filename(name: &str) -> Option<String> {
    let base = Path::new(name).file_name()?.to_str()?.to_string();
    if base.is_empty() || base.starts_with('.') {
        return None;
    }
    Some(base)
}

type ResponseStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl ImageGenerationService for ImageService {
    type GenerateImageStream = ResponseStream<ImageGenerationResponse>;
    type UploadFileStream = ResponseStream<UploadResponse>;

    async fn generate_image(
        &self,
        request: Request<ImageGenerationRequest>,
    ) -> Result<Response<Self::GenerateImageStream>, Status> {
        let req = request.into_inner();
        self.check_secret(req.shared_secret.as_deref())?;
        let config = tc_config::GenerationConfiguration::from_flatbuffer(&req.configuration)
            .map_err(|e| Status::invalid_argument(format!("configuration: {e}")))?;
        let contents: HashMap<Vec<u8>, &Vec<u8>> =
            req.contents.iter().map(|c| (Sha256::digest(c).to_vec(), c)).collect();
        let unwrap = |d: &Vec<u8>| -> Vec<u8> {
            if d.len() == 32 {
                contents.get(d).map(|c| (*c).clone()).unwrap_or_default()
            } else {
                d.clone()
            }
        };
        let image = match &req.image {
            Some(d) => {
                let t = tc_wire::decode(&unwrap(d)).map_err(|e| Status::invalid_argument(format!("image: {e}")))?;
                Some(tensor_to_rgb(&t).map_err(|e| Status::invalid_argument(format!("image: {e}")))?)
            }
            None => None,
        };
        if req.mask.is_some() {
            warn!("mask provided; inpainting masks are not supported yet and will be ignored");
        }
        let mut references = vec![];
        for hint in &req.hints {
            if hint.hint_type == "shuffle" {
                for t in &hint.tensors {
                    let w = tc_wire::decode(&unwrap(&t.tensor)).map_err(|e| Status::invalid_argument(format!("hint: {e}")))?;
                    references.push(tensor_to_rgb(&w).map_err(|e| Status::invalid_argument(format!("hint: {e}")))?);
                }
            } else {
                warn!("ignoring unsupported hint type {:?}", hint.hint_type);
            }
        }
        info!(
            "GenerateImage from {:?}: model={:?} seed={} steps={} size={}x{}",
            req.user,
            config.model,
            config.seed,
            config.steps,
            config.width_px(),
            config.height_px()
        );
        let steps = config.steps.max(1) as i32;
        let has_images = image.is_some() || !references.is_empty();
        let input = GenerationInput {
            config,
            prompt: req.prompt.clone(),
            negative_prompt: req.negative_prompt.clone(),
            image,
            references,
        };
        let cancel = Cancellation::default();
        let events = self.worker.submit(input, cancel.clone());
        let (tx, rx) = mpsc::channel::<Result<ImageGenerationResponse, Status>>(16);
        let compression = if self.options.response_compression { Compression::FpzipOrZip } else { Compression::None };
        let chunked = req.chunked;
        let mut expected = vec![signpost(&Progress::TextEncoded)];
        if has_images {
            expected.push(signpost(&Progress::ImageEncoded));
        }
        for s in 0..steps {
            expected.push(signpost(&Progress::Sampling { step: s as usize, total: steps as usize }));
        }
        expected.push(signpost(&Progress::ImageDecoded));
        tokio::task::spawn_blocking(move || {
            let send = |m: ImageGenerationResponse| tx.blocking_send(Ok(m)).is_ok();
            for ev in events {
                match ev {
                    Event::Progress(p) => {
                        let msg = ImageGenerationResponse {
                            current_signpost: Some(signpost(&p)),
                            signposts: expected.clone(),
                            ..Default::default()
                        };
                        if !send(msg) {
                            info!("client went away; cancelling generation");
                            cancel.cancel();
                        }
                    }
                    Event::Finished(Ok(images)) => {
                        let datas: Vec<Vec<u8>> = images.iter().map(|i| image_tensor_bytes(i, compression)).collect();
                        let total: usize = datas.iter().map(|d| d.len()).sum();
                        if total > 0 {
                            send(ImageGenerationResponse { download_size: Some(total as i64), ..Default::default() });
                        }
                        let chunk_it = chunked && total > CHUNK;
                        for d in datas {
                            if chunk_it && d.len() > CHUNK {
                                let mut off = 0;
                                while off < d.len() {
                                    let end = (off + CHUNK).min(d.len());
                                    let state = if end == d.len() { ChunkState::LastChunk } else { ChunkState::MoreChunks };
                                    send(ImageGenerationResponse {
                                        generated_images: vec![d[off..end].to_vec()],
                                        scale_factor: Some(1),
                                        chunk_state: state as i32,
                                        ..Default::default()
                                    });
                                    off = end;
                                }
                            } else {
                                send(ImageGenerationResponse {
                                    generated_images: vec![d],
                                    scale_factor: Some(1),
                                    chunk_state: ChunkState::LastChunk as i32,
                                    ..Default::default()
                                });
                            }
                        }
                        return;
                    }
                    Event::Finished(Err(e)) => {
                        error!("generation failed: {e:#}");
                        let _ = tx.blocking_send(Err(Status::internal(format!("{e:#}"))));
                        return;
                    }
                }
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn files_exist(&self, request: Request<FileListRequest>) -> Result<Response<FileExistenceResponse>, Status> {
        let req = request.into_inner();
        self.check_secret(req.shared_secret.as_deref())?;
        let store = self.store.clone();
        let res = tokio::task::spawn_blocking(move || {
            let mut out = FileExistenceResponse::default();
            for f in &req.files {
                let path = store.local_path(f);
                out.files.push(f.clone());
                out.existences.push(path.is_some());
                let hash = match (&path, req.files_with_hash.contains(f)) {
                    (Some(p), true) if p.is_file() => sha256_file(p).unwrap_or_default(),
                    _ => vec![],
                };
                out.hashes.push(hash);
            }
            out
        })
        .await
        .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(res))
    }

    async fn upload_file(
        &self,
        request: Request<Streaming<FileUploadRequest>>,
    ) -> Result<Response<Self::UploadFileStream>, Status> {
        let mut stream = request.into_inner();
        let dir = self.store.primary_dir().to_path_buf();
        let secret = self.options.shared_secret.clone().filter(|s| !s.is_empty());
        let (tx, rx) = mpsc::channel::<Result<UploadResponse, Status>>(16);
        tokio::spawn(async move {
            let result = handle_upload(&mut stream, &dir, secret.as_deref(), &tx).await;
            if let Err(status) = result {
                let _ = tx.send(Err(status)).await;
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn echo(&self, request: Request<EchoRequest>) -> Result<Response<EchoReply>, Status> {
        let req = request.into_inner();
        if let Some(s) = self.options.shared_secret.as_deref().filter(|s| !s.is_empty()) {
            if req.shared_secret.as_deref() != Some(s) {
                return Ok(Response::new(EchoReply { shared_secret_missing: true, ..Default::default() }));
            }
        }
        let mut reply = EchoReply {
            message: format!("HELLO {}", req.name),
            server_identifier: self.server_id,
            shared_secret_missing: false,
            ..Default::default()
        };
        if self.options.model_browser {
            let (files, models) = browse(&self.store);
            reply.files = files;
            reply.r#override = Some(MetadataOverride { models: models.into_bytes(), ..Default::default() });
        }
        Ok(Response::new(reply))
    }

    async fn pubkey(&self, _request: Request<PubkeyRequest>) -> Result<Response<PubkeyResponse>, Status> {
        Ok(Response::new(PubkeyResponse::default()))
    }

    async fn hours(&self, _request: Request<HoursRequest>) -> Result<Response<HoursResponse>, Status> {
        Ok(Response::new(HoursResponse::default()))
    }
}

fn sha256_file(p: &Path) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h)?;
    Ok(h.finalize().to_vec())
}

/// Locally available models and LoRA files, plus a JSON description of the models.
fn browse(store: &ModelStore) -> (Vec<String>, String) {
    let mut files = vec![];
    let mut models = vec![];
    for spec in &store.models {
        if store.snapshot_dir(&spec.repo).is_some() {
            files.push(spec.id.clone());
            models.push(serde_json::json!({
                "name": spec.name,
                "file": spec.id,
                "prefix": "",
                "version": if spec.id.contains("4b") { "flux2_4b" } else { "flux2_9b" },
                "default_scale": spec.default_size / 64,
                "huggingface_link": spec.repo,
            }));
        }
    }
    for d in &store.dirs {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.ends_with(".safetensors") && e.metadata().map(|m| m.len() > 0).unwrap_or(false) {
                    files.push(name);
                }
            }
        }
    }
    (files, serde_json::Value::Array(models).to_string())
}

async fn handle_upload(
    stream: &mut Streaming<FileUploadRequest>,
    dir: &Path,
    secret: Option<&str>,
    tx: &mpsc::Sender<Result<UploadResponse, Status>>,
) -> Result<(), Status> {
    use file_upload_request::Request as R;
    let mut state: Option<(String, i64, Vec<u8>, PathBuf, tokio::fs::File)> = None;
    let mut received: i64 = 0;
    while let Some(msg) = stream.message().await? {
        if let Some(s) = secret {
            if msg.shared_secret.as_deref() != Some(s) {
                return Err(Status::unauthenticated("shared secret mismatch"));
            }
        }
        match msg.request {
            Some(R::InitRequest(init)) => {
                let name = sanitize_filename(&init.filename).ok_or_else(|| Status::invalid_argument("invalid filename"))?;
                let tmp = dir.join(format!("{name}.part"));
                let file = tokio::fs::File::create(&tmp).await.map_err(|e| Status::internal(e.to_string()))?;
                state = Some((name.clone(), init.total_size, init.sha256.clone(), tmp, file));
                received = 0;
                let _ = tx
                    .send(Ok(UploadResponse {
                        chunk_upload_success: true,
                        filename: name,
                        message: "upload initialized".into(),
                        ..Default::default()
                    }))
                    .await;
            }
            Some(R::Chunk(chunk)) => {
                let Some((name, _, _, _, file)) = state.as_mut() else {
                    return Err(Status::failed_precondition("chunk before init"));
                };
                if chunk.offset != received {
                    return Err(Status::data_loss("unexpected chunk offset"));
                }
                file.write_all(&chunk.content).await.map_err(|e| Status::internal(e.to_string()))?;
                received += chunk.content.len() as i64;
                let _ = tx
                    .send(Ok(UploadResponse {
                        chunk_upload_success: true,
                        filename: name.clone(),
                        received_offset: received,
                        message: "chunk received".into(),
                    }))
                    .await;
            }
            None => {}
        }
    }
    let Some((name, total, expected_hash, tmp, mut file)) = state else {
        return Err(Status::failed_precondition("missing upload metadata"));
    };
    file.flush().await.map_err(|e| Status::internal(e.to_string()))?;
    drop(file);
    if received != total {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(Status::invalid_argument(format!("received {received} bytes, expected {total}")));
    }
    let tmp2 = tmp.clone();
    let hash = tokio::task::spawn_blocking(move || sha256_file(&tmp2))
        .await
        .map_err(|e| Status::internal(e.to_string()))?
        .map_err(|e| Status::internal(e.to_string()))?;
    if !expected_hash.is_empty() && hash != expected_hash {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(Status::data_loss("file validation failed (sha256 mismatch)"));
    }
    tokio::fs::rename(&tmp, dir.join(&name)).await.map_err(|e| Status::internal(e.to_string()))?;
    info!("uploaded {name} ({received} bytes)");
    Ok(())
}
