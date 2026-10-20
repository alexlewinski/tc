//! `gRPCServerCLI` — serve image generation over the Draw Things compatible gRPC protocol.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::Parser;
use log::{info, warn};
use tc_core::pipeline::{EngineOptions, Precision};
use tc_grpc::ServiceOptions;
use tc_service::{model_dirs, ModelStore, Worker};

#[derive(Parser, Debug)]
#[command(name = "gRPCServerCLI", version, about = "Serve FLUX.2 image generation over gRPC.")]
struct Args {
    /// The directory path to load models.
    models_directory: PathBuf,

    /// The name you expose in the local network.
    #[arg(short, long)]
    name: Option<String>,

    /// The address in your local network that you want to expose.
    #[arg(short, long, default_value = "0.0.0.0")]
    address: String,

    /// The port in your local network that you want to expose.
    #[arg(short, long, default_value_t = 7859)]
    port: u16,

    /// The GPU to use for image generation.
    #[arg(short, long, default_value_t = 0)]
    gpu: u32,

    /// Use Datadog as the logging backend (not supported; accepted for compatibility).
    #[arg(short, long, default_value = "")]
    datadog_api_key: String,

    /// The shared secret that can help to secure the server.
    #[arg(short, long, default_value = "")]
    shared_secret: String,

    /// Disable TLS for the connection.
    #[arg(long)]
    no_tls: bool,

    /// Disable response compression.
    #[arg(long)]
    no_response_compression: bool,

    /// Enable model browsing.
    #[arg(long)]
    model_browser: bool,

    /// Disable FlashAttention.
    #[arg(long)]
    no_flash_attention: bool,

    /// The weights cache size in GiB (accepted for compatibility; weights stay resident).
    #[arg(short, long, default_value_t = 0)]
    weights_cache: u32,

    /// Debug flag for verbose model inference logging.
    #[arg(long)]
    debug: bool,

    /// Path to the directory for secondary models.
    #[arg(long)]
    secondary_models_directory: Option<PathBuf>,

    /// Blob store API access key (not supported; accepted for compatibility).
    #[arg(long)]
    blob_store_access_key: Option<String>,
    /// Blob store API secret (not supported).
    #[arg(long)]
    blob_store_secret: Option<String>,
    /// Endpoint URL for the blob store (not supported).
    #[arg(long)]
    blob_store_endpoint: Option<String>,
    /// The bucket for the blob store (not supported).
    #[arg(long)]
    blob_store_bucket: Option<String>,

    /// Offload weights to reduce memory: the text encoder is unloaded after each prompt.
    #[arg(long)]
    cpu_offload: bool,

    /// Prefer fread over mmap (accepted for compatibility).
    #[arg(long)]
    fread_preferred: bool,

    /// The directory where cached files can be stored.
    #[arg(long)]
    cache_uri: Option<PathBuf>,

    /// Supervise the server so it restarts upon an internal crash.
    #[arg(long)]
    supervised: bool,
    /// The maximum number of crashes within the time window before giving up.
    #[arg(long)]
    max_crashes_within_time_window: Option<u32>,
    /// The time window for counting crashes, in seconds.
    #[arg(long, default_value_t = 60.0)]
    crash_time_window: f64,

    /// Join a proxy (not supported; accepted for compatibility).
    #[arg(long)]
    join: Option<String>,

    /// Warn if a cancelled generation has not returned after this many seconds.
    #[arg(long)]
    cancellation_warning_timeout: Option<f64>,
    /// Exit if a cancelled generation has not returned this many seconds after the warning.
    #[arg(long)]
    cancellation_crash_timeout: Option<f64>,

    /// Answer echo on the generation queue (health check). Accepted for compatibility.
    #[arg(long)]
    echo_on_queue: bool,

    /// Transformer precision: f16, bf16 or f32 (default: f16 on GPU, f32 on CPU).
    #[arg(long)]
    precision: Option<String>,
    /// Text encoder precision: f16, bf16 or f32 (default: bf16 on GPU, f32 on CPU).
    #[arg(long)]
    text_encoder_precision: Option<String>,
    /// Keep the text encoder loaded between requests (faster, more memory).
    #[arg(long)]
    keep_text_encoder: bool,
    /// Run on CPU even if a GPU is available.
    #[arg(long)]
    cpu: bool,
    /// Store transformer / text encoder weights 8-bit palettized (about half the memory).
    #[arg(long)]
    quantize: bool,
}

pub fn parse_dtype(s: &str) -> Result<nnc::DataType> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "f16" | "fp16" | "float16" | "half" => nnc::DataType::F16,
        "bf16" | "bfloat16" => nnc::DataType::BF16,
        "f32" | "fp32" | "float32" | "float" => nnc::DataType::F32,
        other => bail!("unknown precision {other:?} (use f16, bf16 or f32)"),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(if args.debug { "debug" } else { "info" })).init();
    if args.debug {
        nnc::set_log_level(2);
    }
    if !args.models_directory.is_dir() {
        bail!("invalid models directory: {}", args.models_directory.display());
    }
    if let Some(dir) = &args.secondary_models_directory {
        if !dir.is_dir() {
            bail!("invalid secondary models directory: {}", dir.display());
        }
    }
    for (flag, set) in [
        ("--datadog-api-key", !args.datadog_api_key.is_empty()),
        ("--blob-store-*", args.blob_store_access_key.is_some() || args.blob_store_bucket.is_some()),
        ("--join", args.join.is_some()),
    ] {
        if set {
            warn!("{flag} is not supported by this implementation and is ignored");
        }
    }
    if args.supervised && std::env::var_os("TONIC_CLAP_SUPERVISED_CHILD").is_none() {
        return supervise(args.max_crashes_within_time_window, args.crash_time_window);
    }
    run(args)
}

/// Restart the server (as a child process) when it crashes.
fn supervise(max_crashes: Option<u32>, window: f64) -> Result<()> {
    let exe = std::env::current_exe()?;
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut crashes: Vec<Instant> = vec![];
    loop {
        let status = std::process::Command::new(&exe).args(&argv).env("TONIC_CLAP_SUPERVISED_CHILD", "1").status()?;
        if status.success() {
            return Ok(());
        }
        let now = Instant::now();
        crashes.push(now);
        crashes.retain(|t| now.duration_since(*t).as_secs_f64() <= window);
        warn!("server exited with {status}; restarting");
        if let Some(max) = max_crashes {
            if crashes.len() as u32 > max {
                bail!("too many crashes within {window}s, giving up");
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn run(args: Args) -> Result<()> {
    nnc::set_flash_attention(!args.no_flash_attention);
    let device = if args.cpu || nnc::gpu_count() == 0 { nnc::Device::Cpu } else { nnc::Device::Gpu(args.gpu) };
    let mut precision = Precision::default_for(device);
    if let Some(p) = &args.precision {
        precision.transformer = parse_dtype(p)?;
    }
    if let Some(p) = &args.text_encoder_precision {
        precision.text_encoder = parse_dtype(p)?;
    }
    let options = EngineOptions { device, precision, keep_text_encoder: args.keep_text_encoder && !args.cpu_offload, prompt_cache_size: 16, quantize: args.quantize };
    info!("device: {device:?}, precision: {precision:?}");
    let store = ModelStore::new(model_dirs(args.models_directory.clone(), args.secondary_models_directory.clone()));
    let worker = Arc::new(Worker::spawn(store, options));
    let addr: SocketAddr = format!("{}:{}", args.address, args.port)
        .parse()
        .or_else(|_| {
            use std::net::ToSocketAddrs;
            format!("{}:{}", args.address, args.port).to_socket_addrs()?.next().context("cannot resolve address")
        })
        .context("invalid address")?;
    let name = args.name.clone().unwrap_or_else(|| hostname::get().ok().and_then(|h| h.into_string().ok()).unwrap_or_else(|| "Unknown Computer".into()));
    let options = ServiceOptions {
        shared_secret: Some(args.shared_secret.clone()).filter(|s| !s.is_empty()),
        response_compression: !args.no_response_compression,
        model_browser: args.model_browser,
    };
    let tls = !args.no_tls;
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let advertiser = advertise(&name, args.port);
        println!("Server started on local address: {addr}, port {}", args.port);
        let shutdown = async {
            let _ = tokio::signal::ctrl_c().await;
        };
        let r = tc_grpc::serve(addr, worker, options, tls, shutdown).await;
        if let Some(d) = advertiser {
            let _ = d.shutdown();
        }
        r
    })
}

/// Advertise `_dt-grpc._tcp.local.` via mDNS so clients on the LAN can discover the server.
fn advertise(name: &str, port: u16) -> Option<mdns_sd::ServiceDaemon> {
    let daemon = mdns_sd::ServiceDaemon::new().ok()?;
    let host = format!("{}.local.", hostname::get().ok()?.to_string_lossy().split('.').next().unwrap_or("localhost"));
    let info = mdns_sd::ServiceInfo::new("_dt-grpc._tcp.local.", name, &host, "", port, None::<std::collections::HashMap<String, String>>)
        .ok()?
        .enable_addr_auto();
    if let Err(e) = daemon.register(info) {
        warn!("mDNS advertisement failed: {e}");
        return None;
    }
    Some(daemon)
}
