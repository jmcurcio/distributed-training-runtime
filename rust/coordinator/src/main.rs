//! Distributed Training Runtime Coordinator Service
//!
//! This binary provides a standalone coordinator service for multi-worker
//! distributed training coordination.
//!
//! # Usage
//!
//! ```bash
//! # Start coordinator with default settings
//! dtr-coordinator
//!
//! # Start with custom port
//! dtr-coordinator --port 50052
//!
//! # Start with configuration file
//! dtr-coordinator --config coordinator.toml
//! ```

mod proto {
    include!("proto/dtr.coordinator.rs");
}

mod service;
mod state;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use tonic::transport::Server;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use proto::coordinator_service_server::CoordinatorServiceServer;
use runtime_core::config::ShardStrategy;
use service::CoordinatorServiceImpl;
use state::CoordinatorState;

/// Distributed Training Runtime Coordinator
#[derive(Parser, Debug)]
#[command(name = "dtr-coordinator")]
#[command(about = "Coordinator service for distributed training runtime")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "50051")]
    port: u16,

    /// Address to bind to
    #[arg(short, long, default_value = "0.0.0.0")]
    address: String,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<String>,

    /// Worker timeout in seconds
    #[arg(long, default_value = "30")]
    worker_timeout: u64,

    /// Heartbeat interval in milliseconds
    #[arg(long, default_value = "5000")]
    heartbeat_interval: u64,

    /// Checkpoint timeout in seconds
    #[arg(long, default_value = "300")]
    checkpoint_timeout: u64,

    /// Shard assignment strategy (static, dynamic, locality_aware)
    #[arg(long, default_value = "static")]
    shard_strategy: String,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize logging
    let filter = tracing_subscriber::filter::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::filter::EnvFilter::new(&args.log_level));

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse shard strategy
    let shard_strategy = match args.shard_strategy.to_lowercase().as_str() {
        "static" => ShardStrategy::Static,
        "dynamic" => ShardStrategy::Dynamic,
        "locality_aware" | "locality-aware" => ShardStrategy::LocalityAware,
        _ => {
            tracing::warn!(
                "Unknown shard strategy '{}', defaulting to static",
                args.shard_strategy
            );
            ShardStrategy::Static
        }
    };

    tracing::info!("Starting DTR Coordinator");
    tracing::info!("  Worker timeout: {}s", args.worker_timeout);
    tracing::info!("  Heartbeat interval: {}ms", args.heartbeat_interval);
    tracing::info!("  Checkpoint timeout: {}s", args.checkpoint_timeout);
    tracing::info!("  Shard strategy: {:?}", shard_strategy);

    // Create coordinator state
    let state = Arc::new(CoordinatorState::new(
        Duration::from_secs(args.worker_timeout),
        shard_strategy,
        (args.checkpoint_timeout * 1000) as i64,
        args.heartbeat_interval,
    ));

    // Create service
    let service = CoordinatorServiceImpl::new(state.clone());

    // Build address
    let addr: SocketAddr = format!("{}:{}", args.address, args.port).parse()?;

    tracing::info!("Listening on {}", addr);

    // Start background tasks
    let bg_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        loop {
            interval.tick().await;

            // Check for worker timeouts
            let timed_out = bg_state.check_worker_timeouts().await;
            for worker_id in timed_out {
                tracing::warn!("Worker {} timed out", worker_id);
            }

            // Check for checkpoint timeouts
            let timed_out_checkpoints = bg_state.check_checkpoint_timeouts().await;
            for checkpoint_id in timed_out_checkpoints {
                tracing::warn!("Checkpoint {} timed out", checkpoint_id);
            }
        }
    });

    // Start gRPC server
    Server::builder()
        .add_service(CoordinatorServiceServer::new(service))
        .serve_with_shutdown(addr, async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to install CTRL+C handler");
            tracing::info!("Shutting down coordinator...");
        })
        .await?;

    Ok(())
}
