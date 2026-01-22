//! Coordinator client implementation.
//!
//! This module provides the gRPC client for workers to communicate with
//! the coordinator service.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::RwLock;
use tonic::transport::{Channel, Endpoint};
use tonic::Request;

use super::proto::coordinator_service_client::CoordinatorServiceClient;
use super::proto::{
    GetShardAssignmentRequest, GetWorkerStatusRequest, HeartbeatRequest,
    InitiateCheckpointRequest, RegisterWorkerRequest, UnregisterWorkerRequest,
};
use super::protocol::{
    CheckpointPlan, CheckpointShardAck, CheckpointShardConfirmation, HeartbeatResponse,
    ProgressReport, ShardAssignment, WorkerCapabilities, WorkerConfig, WorkerInfo, WorkerStatus,
};
use crate::config::CoordinatorConfig;
use crate::error::{Result, RuntimeError};

/// Trait for coordinator client implementations.
#[async_trait]
pub trait CoordinatorClient: Send + Sync {
    /// Register this worker with the coordinator.
    async fn register(&mut self, capabilities: WorkerCapabilities) -> Result<WorkerConfig>;

    /// Get shard assignment for a dataset.
    async fn get_shard_assignment(
        &self,
        dataset_id: &str,
        total_shards: u32,
    ) -> Result<ShardAssignment>;

    /// Report training progress.
    async fn report_progress(&self, progress: ProgressReport) -> Result<()>;

    /// Send heartbeat to coordinator.
    async fn heartbeat(&self, status: WorkerStatus, step: u64) -> Result<HeartbeatResponse>;

    /// Initiate a distributed checkpoint.
    async fn initiate_checkpoint(
        &self,
        checkpoint_name: &str,
        step: u64,
        metadata: std::collections::HashMap<String, String>,
    ) -> Result<CheckpointPlan>;

    /// Confirm that a checkpoint shard has been written.
    async fn confirm_checkpoint_shard(
        &self,
        confirmation: CheckpointShardConfirmation,
    ) -> Result<CheckpointShardAck>;

    /// Get information about workers in the job.
    async fn get_worker_status(&self, worker_id: Option<&str>) -> Result<Vec<WorkerInfo>>;

    /// Unregister this worker (graceful shutdown).
    async fn unregister(&self, reason: &str) -> Result<()>;

    /// Get the worker configuration (available after registration).
    fn worker_config(&self) -> Option<&WorkerConfig>;

    /// Check if the client is connected.
    fn is_connected(&self) -> bool;
}

/// gRPC-based coordinator client.
pub struct GrpcCoordinatorClient {
    config: CoordinatorConfig,
    client: Option<CoordinatorServiceClient<Channel>>,
    worker_config: Option<WorkerConfig>,
    memory_usage: Arc<RwLock<u64>>,
}

impl GrpcCoordinatorClient {
    /// Create a new gRPC coordinator client.
    pub fn new(config: CoordinatorConfig) -> Self {
        Self {
            config,
            client: None,
            worker_config: None,
            memory_usage: Arc::new(RwLock::new(0)),
        }
    }

    /// Connect to the coordinator service.
    pub async fn connect(&mut self) -> Result<()> {
        let endpoint = Endpoint::from_shared(format!("http://{}", self.config.address))
            .map_err(|e| RuntimeError::coordinator_with_source("invalid endpoint address", e))?
            .connect_timeout(Duration::from_millis(self.config.connect_timeout_ms))
            .timeout(Duration::from_millis(self.config.request_timeout_ms));

        let channel = endpoint.connect().await.map_err(|e| {
            RuntimeError::coordinator_with_source(
                format!("failed to connect to coordinator at {}", self.config.address),
                e,
            )
        })?;

        self.client = Some(CoordinatorServiceClient::new(channel));
        Ok(())
    }

    /// Connect with retry logic.
    pub async fn connect_with_retry(&mut self) -> Result<()> {
        let mut attempts = 0;
        let mut delay = Duration::from_millis(self.config.reconnect_delay_ms);

        loop {
            match self.connect().await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.config.max_reconnect_attempts {
                        return Err(RuntimeError::coordinator(format!(
                            "failed to connect after {} attempts: {}",
                            attempts, e
                        )));
                    }

                    tokio::time::sleep(delay).await;
                    delay = std::cmp::min(
                        delay * 2,
                        Duration::from_millis(self.config.request_timeout_ms),
                    );
                }
            }
        }
    }

    fn get_client(&self) -> Result<&CoordinatorServiceClient<Channel>> {
        self.client
            .as_ref()
            .ok_or_else(|| RuntimeError::coordinator("not connected to coordinator"))
    }

    fn get_session_token(&self) -> String {
        self.worker_config
            .as_ref()
            .map(|c| c.session_token.clone())
            .unwrap_or_default()
    }

    fn get_worker_id(&self) -> String {
        self.worker_config
            .as_ref()
            .map(|c| c.worker_id.clone())
            .unwrap_or_else(|| self.config.worker_id.clone())
    }

    /// Update tracked memory usage.
    pub async fn set_memory_usage(&self, bytes: u64) {
        *self.memory_usage.write().await = bytes;
    }
}

#[async_trait]
impl CoordinatorClient for GrpcCoordinatorClient {
    async fn register(&mut self, capabilities: WorkerCapabilities) -> Result<WorkerConfig> {
        let mut client = self.get_client()?.clone();

        let request = Request::new(RegisterWorkerRequest {
            worker_id: self.config.worker_id.clone(),
            job_id: self.config.job_id.clone(),
            capabilities: Some(capabilities.into()),
            hostname: hostname::get()
                .map(|h| h.to_string_lossy().to_string())
                .unwrap_or_default(),
            tags: std::collections::HashMap::new(),
        });

        let response = client
            .register_worker(request)
            .await
            .map_err(|e| RuntimeError::coordinator_with_source("worker registration failed", e))?;

        let proto_config = response
            .into_inner()
            .config
            .ok_or_else(|| RuntimeError::coordinator("missing worker config in response"))?;

        let config = WorkerConfig::from(proto_config);
        self.worker_config = Some(config.clone());
        Ok(config)
    }

    async fn get_shard_assignment(
        &self,
        dataset_id: &str,
        total_shards: u32,
    ) -> Result<ShardAssignment> {
        let mut client = self.get_client()?.clone();

        let request = Request::new(GetShardAssignmentRequest {
            worker_id: self.get_worker_id(),
            session_token: self.get_session_token(),
            dataset_id: dataset_id.to_string(),
            total_shards,
        });

        let response = client.get_shard_assignment(request).await.map_err(|e| {
            RuntimeError::coordinator_with_source("failed to get shard assignment", e)
        })?;

        Ok(ShardAssignment::from(response.into_inner()))
    }

    async fn report_progress(&self, progress: ProgressReport) -> Result<()> {
        let mut client = self.get_client()?.clone();

        // For now, use unary call. Streaming will be added later.
        let mut proto_progress: super::proto::ProgressReport = progress.into();
        proto_progress.worker_id = self.get_worker_id();
        proto_progress.session_token = self.get_session_token();

        // Create a single-item stream
        let stream = tokio_stream::once(proto_progress);
        let request = Request::new(stream);

        let mut response_stream = client
            .report_progress(request)
            .await
            .map_err(|e| RuntimeError::coordinator_with_source("failed to report progress", e))?
            .into_inner();

        // Consume first response
        use tokio_stream::StreamExt;
        if let Some(ack) = response_stream.next().await {
            let _ack = ack.map_err(|e| {
                RuntimeError::coordinator_with_source("progress stream error", e)
            })?;
        }

        Ok(())
    }

    async fn heartbeat(&self, status: WorkerStatus, step: u64) -> Result<HeartbeatResponse> {
        let mut client = self.get_client()?.clone();

        let memory_used = *self.memory_usage.read().await;

        let request = Request::new(HeartbeatRequest {
            worker_id: self.get_worker_id(),
            session_token: self.get_session_token(),
            status: super::proto::WorkerStatus::from(status) as i32,
            step,
            memory_used_bytes: memory_used,
            gpu_utilization: None,
        });

        let response = client
            .heartbeat(request)
            .await
            .map_err(|e| RuntimeError::coordinator_with_source("heartbeat failed", e))?;

        Ok(HeartbeatResponse::from(response.into_inner()))
    }

    async fn initiate_checkpoint(
        &self,
        checkpoint_name: &str,
        step: u64,
        metadata: std::collections::HashMap<String, String>,
    ) -> Result<CheckpointPlan> {
        let mut client = self.get_client()?.clone();

        let request = Request::new(InitiateCheckpointRequest {
            worker_id: self.get_worker_id(),
            session_token: self.get_session_token(),
            checkpoint_name: checkpoint_name.to_string(),
            step,
            metadata,
        });

        let response = client.initiate_checkpoint(request).await.map_err(|e| {
            RuntimeError::coordinator_with_source("failed to initiate checkpoint", e)
        })?;

        Ok(CheckpointPlan::from(response.into_inner()))
    }

    async fn confirm_checkpoint_shard(
        &self,
        mut confirmation: CheckpointShardConfirmation,
    ) -> Result<CheckpointShardAck> {
        let mut client = self.get_client()?.clone();

        confirmation.worker_id = self.get_worker_id();
        confirmation.session_token = self.get_session_token();

        let request = Request::new(confirmation.into());

        let response = client
            .confirm_checkpoint_shard(request)
            .await
            .map_err(|e| {
                RuntimeError::coordinator_with_source("failed to confirm checkpoint shard", e)
            })?;

        Ok(CheckpointShardAck::from(response.into_inner()))
    }

    async fn get_worker_status(&self, worker_id: Option<&str>) -> Result<Vec<WorkerInfo>> {
        let mut client = self.get_client()?.clone();

        let request = Request::new(GetWorkerStatusRequest {
            worker_id: worker_id.unwrap_or("").to_string(),
            job_id: self
                .worker_config
                .as_ref()
                .map(|c| c.job_id.clone())
                .unwrap_or_else(|| self.config.job_id.clone()),
        });

        let response = client
            .get_worker_status(request)
            .await
            .map_err(|e| RuntimeError::coordinator_with_source("failed to get worker status", e))?;

        Ok(response
            .into_inner()
            .workers
            .into_iter()
            .map(WorkerInfo::from)
            .collect())
    }

    async fn unregister(&self, reason: &str) -> Result<()> {
        let mut client = self.get_client()?.clone();

        let request = Request::new(UnregisterWorkerRequest {
            worker_id: self.get_worker_id(),
            session_token: self.get_session_token(),
            reason: reason.to_string(),
        });

        client
            .unregister_worker(request)
            .await
            .map_err(|e| RuntimeError::coordinator_with_source("failed to unregister worker", e))?;

        Ok(())
    }

    fn worker_config(&self) -> Option<&WorkerConfig> {
        self.worker_config.as_ref()
    }

    fn is_connected(&self) -> bool {
        self.client.is_some() && self.worker_config.is_some()
    }
}

/// Background heartbeat task that runs while the worker is active.
pub struct HeartbeatTask {
    client: Arc<RwLock<Box<dyn CoordinatorClient>>>,
    interval: Duration,
    shutdown: tokio::sync::watch::Receiver<bool>,
}

impl HeartbeatTask {
    /// Create a new heartbeat task.
    pub fn new(
        client: Arc<RwLock<Box<dyn CoordinatorClient>>>,
        interval_ms: u64,
        shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> Self {
        Self {
            client,
            interval: Duration::from_millis(interval_ms),
            shutdown,
        }
    }

    /// Run the heartbeat task.
    pub async fn run(mut self, status_provider: impl Fn() -> (WorkerStatus, u64) + Send + 'static) {
        let mut interval = tokio::time::interval(self.interval);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let (status, step) = status_provider();
                    let client = self.client.read().await;
                    if let Err(e) = client.heartbeat(status, step).await {
                        tracing::warn!("Heartbeat failed: {}", e);
                    }
                }
                _ = self.shutdown.changed() => {
                    if *self.shutdown.borrow() {
                        break;
                    }
                }
            }
        }
    }
}

// Hostname utility
mod hostname {
    use std::ffi::OsString;

    pub fn get() -> std::io::Result<OsString> {
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let mut buf = vec![0u8; 256];
            // SAFETY: gethostname writes a null-terminated string into buf
            let ret = unsafe {
                ::libc::gethostname(buf.as_mut_ptr() as *mut ::libc::c_char, buf.len())
            };
            if ret != 0 {
                return Err(std::io::Error::last_os_error());
            }
            let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            buf.truncate(len);
            Ok(OsString::from_vec(buf))
        }

        #[cfg(not(unix))]
        {
            Ok(OsString::from("unknown"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let config = CoordinatorConfig {
            enabled: true,
            address: "localhost:50051".to_string(),
            worker_id: "test-worker".to_string(),
            job_id: "test-job".to_string(),
            ..Default::default()
        };

        let client = GrpcCoordinatorClient::new(config);
        assert!(!client.is_connected());
        assert!(client.worker_config().is_none());
    }
}
