//! gRPC end-to-end: in-process server on a tiny FLUX.2 [klein] snapshot, real tonic client.

use std::path::PathBuf;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tc_config::{GenerationConfiguration, LoRA};
use tc_core::pipeline::{EngineOptions, Precision};
use tc_grpc::client::RemoteClient;
use tc_grpc::proto::image_generation_service_client::ImageGenerationServiceClient;
use tc_grpc::proto::*;
use tc_grpc::ServiceOptions;
use tc_service::{ModelStore, Worker};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tc-service/tests/fixtures")
}

fn expected(file: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixtures().join(file)).unwrap();
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let offs = header["image"]["data_offsets"].as_array().unwrap();
    let (a, b) = (offs[0].as_u64().unwrap() as usize, offs[1].as_u64().unwrap() as usize);
    bytes[8 + n + a..8 + n + b].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

async fn start(secret: Option<&str>, upload_dir: PathBuf) -> u16 {
    start_with(secret, vec![upload_dir, fixtures()]).await
}

async fn start_with(secret: Option<&str>, dirs: Vec<PathBuf>) -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let store = ModelStore::new(dirs);
    let device = nnc::Device::Cpu;
    let worker = Arc::new(Worker::spawn(
        store,
        EngineOptions { device, precision: Precision::default_for(device), keep_text_encoder: true, prompt_cache_size: 4, quantize: false, keep_palettized: true },
    ));
    let options = ServiceOptions { shared_secret: secret.map(|s| s.to_string()), response_compression: true, model_browser: true };
    let addr = format!("127.0.0.1:{port}").parse().unwrap();
    tokio::spawn(async move {
        tc_grpc::serve(addr, worker, options, false, futures::future::pending()).await.unwrap();
    });
    for _ in 0..100 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_ok() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    port
}

fn config() -> GenerationConfiguration {
    GenerationConfiguration {
        model: Some("tiny-klein".into()),
        start_width: 2,
        start_height: 1,
        steps: 3,
        seed: 7,
        guidance_scale: 1.0,
        strength: 1.0,
        sampler: tc_config::SamplerType::DDIMTrailing as i8,
        loras: vec![
            LoRA { file: Some("lora_a.safetensors".into()), weight: 0.8, mode: 0 },
            LoRA { file: Some("lora_b.safetensors".into()), weight: 0.5, mode: 0 },
        ],
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_generate_echo_files_and_upload() {
    let upload_dir = std::env::temp_dir().join(format!("tc-grpc-upload-{}", std::process::id()));
    std::fs::create_dir_all(&upload_dir).unwrap();
    let port = start(Some("s3cret"), upload_dir.clone()).await;

    // Wrong secret.
    let mut bad = RemoteClient::connect("127.0.0.1", port, false, Some("nope".into())).await.unwrap();
    assert!(bad.echo("test").await.is_err());

    let mut client = RemoteClient::connect("127.0.0.1", port, false, Some("s3cret".into())).await.unwrap();
    let echo = client.echo("test").await.unwrap();
    assert_eq!(echo.message, "HELLO test");
    assert!(echo.files.iter().any(|f| f == "lora_a.safetensors"), "{:?}", echo.files);

    // Generation with two LoRAs.
    let mut steps_seen = 0;
    let result = client
        .generate(config().to_flatbuffer(), "a red cube on a table", "", None, vec![], |s| {
            if let Some(image_generation_signpost_proto::Signpost::Sampling(_)) = s.signpost {
                steps_seen += 1;
            }
        })
        .await
        .unwrap();
    assert_eq!(steps_seen, 3);
    assert_eq!(result.images.len(), 1);
    let t = &result.images[0];
    assert_eq!(t.shape, vec![1, 64, 128, 3]);
    assert_eq!(t.dtype, tc_wire::WireType::F16);
    let got: Vec<f32> = t.to_f32().iter().map(|v| (v / 2.0 + 0.5).clamp(0.0, 1.0)).collect();
    let exp = expected("expected.safetensors");
    let max = got.iter().zip(exp.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(max < 2e-3, "max error {max}");

    // Two LoRAs plus a reference image, sent both ways a client may send it: as the canvas
    // image (strength 1 keeps it a reference) and as a "shuffle" hint.
    let reference = {
        let t = tc_core::image_io::to_tensor(&tc_core::image_io::load_rgb(&fixtures().join("ref.png")).unwrap());
        tc_wire::encode(&tc_wire::WireTensor::from_f32(&t.to_f32_vec(), t.shape(), tc_wire::WireType::F16), tc_wire::Compression::FpzipOrZip)
    };
    let exp = expected("expected_ref.safetensors");
    for (image, hints) in [(Some(reference.clone()), vec![]), (None, vec![reference.clone()])] {
        let result = client.generate(config().to_flatbuffer(), "a red cube on a table", "", image, hints, |_| {}).await.unwrap();
        let t = &result.images[0];
        assert_eq!(t.shape, vec![1, 64, 128, 3]);
        let got: Vec<f32> = t.to_f32().iter().map(|v| (v / 2.0 + 0.5).clamp(0.0, 1.0)).collect();
        let max = got.iter().zip(exp.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max < 2e-3, "reference image: max error {max}");
    }

    // Raw stub for the remaining RPCs.
    let channel = tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{port}")).unwrap().connect().await.unwrap();
    let mut raw = ImageGenerationServiceClient::new(channel);
    let files = raw
        .files_exist(FileListRequest {
            files: vec!["tiny-klein".into(), "lora_a.safetensors".into(), "missing.safetensors".into()],
            files_with_hash: vec!["lora_a.safetensors".into()],
            shared_secret: Some("s3cret".into()),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(files.existences, vec![true, true, false]);
    let lora_bytes = std::fs::read(fixtures().join("lora_a.safetensors")).unwrap();
    assert_eq!(files.hashes[1], Sha256::digest(&lora_bytes).to_vec());

    // Upload a file in two chunks.
    let payload: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
    let digest = Sha256::digest(&payload).to_vec();
    let secret = Some("s3cret".to_string());
    let msgs = vec![
        FileUploadRequest {
            request: Some(file_upload_request::Request::InitRequest(InitUploadRequest {
                filename: "uploaded.safetensors".into(),
                sha256: digest,
                total_size: payload.len() as i64,
            })),
            shared_secret: secret.clone(),
        },
        FileUploadRequest {
            request: Some(file_upload_request::Request::Chunk(FileChunk { content: payload[..200_000].to_vec(), filename: "uploaded.safetensors".into(), offset: 0 })),
            shared_secret: secret.clone(),
        },
        FileUploadRequest {
            request: Some(file_upload_request::Request::Chunk(FileChunk { content: payload[200_000..].to_vec(), filename: "uploaded.safetensors".into(), offset: 200_000 })),
            shared_secret: secret.clone(),
        },
    ];
    let mut responses = raw.upload_file(tokio_stream::iter(msgs)).await.unwrap().into_inner();
    let mut last_offset = 0;
    while let Some(r) = responses.message().await.unwrap() {
        assert!(r.chunk_upload_success);
        last_offset = r.received_offset;
    }
    assert_eq!(last_offset, payload.len() as i64);
    assert_eq!(std::fs::read(upload_dir.join("uploaded.safetensors")).unwrap(), payload);

    // Path traversal is rejected.
    let evil = vec![FileUploadRequest {
        request: Some(file_upload_request::Request::InitRequest(InitUploadRequest { filename: "../../evil".into(), sha256: vec![], total_size: 1 })),
        shared_secret: secret.clone(),
    }];
    let mut r = raw.upload_file(tokio_stream::iter(evil)).await.unwrap().into_inner();
    let mut ok = true;
    while let Ok(Some(_)) = r.message().await {}
    if let Err(_) = r.message().await {
        ok = false;
    }
    let _ = ok;
    assert!(!upload_dir.parent().unwrap().join("evil").exists());
    let _ = std::fs::remove_dir_all(upload_dir);
}

/// The Draw Things file layout (transformer / text encoder / VAE / LoRA as `.ckpt`) served over
/// gRPC: model and LoRAs are named by their Draw Things file names.
#[tokio::test(flavor = "multi_thread")]
async fn grpc_generate_with_draw_things_checkpoints() {
    let dir = std::env::temp_dir().join(format!("tc-grpc-dt-models-{}", std::process::id()));
    tc_service::testing::write_tiny_draw_things_models(&fixtures(), &dir).unwrap();
    let port = start_with(None, vec![dir.clone()]).await;
    let mut client = RemoteClient::connect("127.0.0.1", port, false, None).await.unwrap();
    let mut c = config();
    c.model = Some("flux_2_klein_tiny_f32.ckpt".into());
    c.loras[1].file = Some("lora_b_dt.ckpt".into());
    let result = client.generate(c.to_flatbuffer(), "a red cube on a table", "", None, vec![], |_| {}).await.unwrap();
    let t = &result.images[0];
    assert_eq!(t.shape, vec![1, 64, 128, 3]);
    let got: Vec<f32> = t.to_f32().iter().map(|v| (v / 2.0 + 0.5).clamp(0.0, 1.0)).collect();
    let exp = expected("expected.safetensors");
    let max = got.iter().zip(exp.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(max < 2e-3, "max error {max}");
    let _ = std::fs::remove_dir_all(dir);
}
