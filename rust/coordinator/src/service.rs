//! gRPC service implementation for the coordinator.

use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use crate::proto::coordinator_service_server::CoordinatorService;
use crate::proto::{
    CheckpointPlan as ProtoCheckpointPlan, CheckpointShardAck, CheckpointShardConfirmation,
    CheckpointStatus, GetCheckpointStatusRequest, GetShardAssignmentRequest,
    GetShardAssignmentResponse, GetWorkerStatusRequest, HeartbeatRequest, HeartbeatResponse,
    InitiateCheckpointRequest, ProgressAck, ProgressReport, RegisterWorkerRequest,
    RegisterWorkerResponse, ShardRange as ProtoShardRange, UnregisterWorkerRequest,
    UnregisterWorkerResponse, WorkerCheckpointAssignment, WorkerConfig as ProtoWorkerConfig,
    WorkerInfo as ProtoWorkerInfo, WorkerStatusResponse,
};
use crate::state::{CheckpointOperationState, CoordinatorState};

use runtime_core::coordinator::protocol::{WorkerCapabilities, WorkerStatus};

pub struct CoordinatorServiceImpl {
    state: Arc<CoordinatorState>,
}

impl CoordinatorServiceImpl {
    pub fn new(state: Arc<CoordinatorState>) -> Self {
        Self { state }
    }

    fn validate_session(&self, worker_id: &str, session_token: &str) -> Result<(), Status> {
        // Session validation is done in state methods
        // This is a placeholder for additional checks
        if worker_id.is_empty() {
            return Err(Status::invalid_argument("worker_id is required"));
        }
        Ok(())
    }
}

#[tonic::async_trait]
impl CoordinatorService for CoordinatorServiceImpl {
    async fn register_worker(
        &self,
        request: Request<RegisterWorkerRequest>,
    ) -> Result<Response<RegisterWorkerResponse>, Status> {
        let req = request.into_inner();

        let capabilities = req
            .capabilities
            .map(|c| WorkerCapabilities {
                gpu_count: c.gpu_count,
                memory_bytes: c.memory_bytes,
                has_local_storage: c.has_local_storage,
                accessible_paths: c.accessible_paths,
                custom: c.custom,
            })
            .unwrap_or_default();

        let worker_id = if req.worker_id.is_empty() {
            None
        } else {
            Some(req.worker_id)
        };

        let (worker, reconnected) = self
            .state
            .register_worker(worker_id, req.job_id, capabilities, req.hostname)
            .await;

        let total_workers = self.state.worker_count().await;

        let response = RegisterWorkerResponse {
            config: Some(ProtoWorkerConfig {
                worker_id: worker.worker_id,
                worker_index: worker.worker_index,
                total_workers,
                job_id: worker.job_id,
                heartbeat_interval_ms: self.state.heartbeat_interval_ms,
                session_token: worker.session_token,
                settings: std::collections::HashMap::new(),
            }),
            reconnected,
        };

        tracing::info!(
            "Worker {} registered (index={}, reconnected={})",
            response.config.as_ref().unwrap().worker_id,
            response.config.as_ref().unwrap().worker_index,
            reconnected
        );

        Ok(Response::new(response))
    }

    async fn get_shard_assignment(
        &self,
        request: Request<GetShardAssignmentRequest>,
    ) -> Result<Response<GetShardAssignmentResponse>, Status> {
        let req = request.into_inner();
        self.validate_session(&req.worker_id, &req.session_token)?;

        if !self
            .state
            .validate_session(&req.worker_id, &req.session_token)
            .await
        {
            return Err(Status::unauthenticated("invalid session"));
        }

        let (ranges, version) = self
            .state
            .get_worker_assignment(&req.worker_id, &req.dataset_id, req.total_shards)
            .await
            .ok_or_else(|| Status::not_found("worker not found"))?;

        let proto_ranges: Vec<ProtoShardRange> = ranges
            .into_iter()
            .map(|r| ProtoShardRange {
                start_shard: r.start_shard,
                end_shard: r.end_shard,
                priority: r.priority,
            })
            .collect();

        let response = GetShardAssignmentResponse {
            shard_ranges: proto_ranges,
            assignment_version: version,
            strategy: crate::proto::ShardStrategy::Static as i32,
        };

        Ok(Response::new(response))
    }

    type ReportProgressStream =
        Pin<Box<dyn Stream<Item = Result<ProgressAck, Status>> + Send + 'static>>;

    async fn report_progress(
        &self,
        request: Request<Streaming<ProgressReport>>,
    ) -> Result<Response<Self::ReportProgressStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = mpsc::channel(32);
        let state = self.state.clone();

        tokio::spawn(async move {
            while let Some(result) = stream.next().await {
                match result {
                    Ok(report) => {
                        // Validate session
                        if !state
                            .validate_session(&report.worker_id, &report.session_token)
                            .await
                        {
                            let _ = tx
                                .send(Err(Status::unauthenticated("invalid session")))
                                .await;
                            break;
                        }

                        // Update worker heartbeat
                        let status = WorkerStatus::from(report.step as i32);
                        state
                            .update_heartbeat(
                                &report.worker_id,
                                &report.session_token,
                                status,
                                report.step,
                            )
                            .await;

                        // Send acknowledgment
                        let ack = ProgressAck {
                            continue_training: true,
                            command: crate::proto::CoordinatorCommand::Continue as i32,
                            server_timestamp_ms: chrono::Utc::now().timestamp_millis(),
                        };

                        if tx.send(Ok(ack)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Progress stream error: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let req = request.into_inner();
        self.validate_session(&req.worker_id, &req.session_token)?;

        if !self
            .state
            .validate_session(&req.worker_id, &req.session_token)
            .await
        {
            return Err(Status::unauthenticated("invalid session"));
        }

        let status = WorkerStatus::from(req.status);

        let worker = self
            .state
            .update_heartbeat(&req.worker_id, &req.session_token, status, req.step)
            .await
            .ok_or_else(|| Status::not_found("worker not found"))?;

        let response = HeartbeatResponse {
            registered: true,
            command: crate::proto::CoordinatorCommand::Continue as i32,
            checkpoint_plan: None,
            leader_worker_id: String::new(), // TODO: Implement leader election
        };

        Ok(Response::new(response))
    }

    async fn initiate_checkpoint(
        &self,
        request: Request<InitiateCheckpointRequest>,
    ) -> Result<Response<ProtoCheckpointPlan>, Status> {
        let req = request.into_inner();
        self.validate_session(&req.worker_id, &req.session_token)?;

        if !self
            .state
            .validate_session(&req.worker_id, &req.session_token)
            .await
        {
            return Err(Status::unauthenticated("invalid session"));
        }

        let checkpoint = self
            .state
            .initiate_checkpoint(
                req.checkpoint_name.clone(),
                req.step,
                format!("/checkpoints/{}", req.checkpoint_name),
                "lz4".to_string(),
            )
            .await;

        // Create assignments for each worker
        let workers = self.state.get_all_workers().await;
        let assignments: Vec<WorkerCheckpointAssignment> = workers
            .iter()
            .enumerate()
            .map(|(i, w)| WorkerCheckpointAssignment {
                worker_id: w.worker_id.clone(),
                shard_path: format!(
                    "{}/shard_{}",
                    checkpoint.base_path,
                    i
                ),
                shard_index: i as u32,
                assigned_keys: vec![], // All keys (empty means all)
            })
            .collect();

        let response = ProtoCheckpointPlan {
            checkpoint_id: checkpoint.checkpoint_id,
            checkpoint_name: checkpoint.checkpoint_name,
            step: checkpoint.step,
            base_path: checkpoint.base_path,
            assignments,
            deadline_ms: checkpoint.deadline.timestamp_millis(),
            compression: checkpoint.compression,
        };

        Ok(Response::new(response))
    }

    async fn confirm_checkpoint_shard(
        &self,
        request: Request<CheckpointShardConfirmation>,
    ) -> Result<Response<CheckpointShardAck>, Status> {
        let req = request.into_inner();
        self.validate_session(&req.worker_id, &req.session_token)?;

        if !self
            .state
            .validate_session(&req.worker_id, &req.session_token)
            .await
        {
            return Err(Status::unauthenticated("invalid session"));
        }

        let confirmation =
            runtime_core::coordinator::protocol::CheckpointShardConfirmation {
                worker_id: req.worker_id,
                session_token: req.session_token,
                checkpoint_id: req.checkpoint_id.clone(),
                shard_path: req.shard_path,
                shard_index: req.shard_index,
                size_bytes: req.size_bytes,
                checksum: req.checksum,
                success: req.success,
                error_message: req.error_message,
            };

        let checkpoint = self
            .state
            .confirm_checkpoint_shard(&req.checkpoint_id, confirmation)
            .await
            .ok_or_else(|| Status::not_found("checkpoint not found"))?;

        let response = CheckpointShardAck {
            accepted: true,
            confirmed_count: checkpoint.confirmed_count(),
            total_workers: checkpoint.total_workers(),
            checkpoint_complete: checkpoint.state == CheckpointOperationState::Finalizing,
        };

        Ok(Response::new(response))
    }

    async fn get_checkpoint_status(
        &self,
        request: Request<GetCheckpointStatusRequest>,
    ) -> Result<Response<CheckpointStatus>, Status> {
        let req = request.into_inner();

        let checkpoint = self
            .state
            .get_checkpoint(&req.checkpoint_id)
            .await
            .ok_or_else(|| Status::not_found("checkpoint not found"))?;

        let state = match checkpoint.state {
            CheckpointOperationState::InProgress => crate::proto::CheckpointState::InProgress,
            CheckpointOperationState::Finalizing => crate::proto::CheckpointState::Finalizing,
            CheckpointOperationState::Complete => crate::proto::CheckpointState::Complete,
            CheckpointOperationState::Failed => crate::proto::CheckpointState::Failed,
            CheckpointOperationState::Timeout => crate::proto::CheckpointState::Timeout,
        };

        let confirmed_workers: Vec<_> = checkpoint.confirmations.keys().cloned().collect();
        let pending_workers: Vec<_> = checkpoint
            .expected_workers
            .iter()
            .filter(|w| !checkpoint.confirmations.contains_key(*w))
            .cloned()
            .collect();

        let response = CheckpointStatus {
            checkpoint_id: checkpoint.checkpoint_id,
            checkpoint_name: checkpoint.checkpoint_name,
            state: state as i32,
            confirmed_workers,
            pending_workers,
            manifest: checkpoint.manifest.map(|m| crate::proto::CheckpointManifest {
                checkpoint_id: m.checkpoint_id,
                checkpoint_name: m.checkpoint_name,
                step: m.step,
                created_at_ms: m.created_at_ms,
                shards: m
                    .shards
                    .into_iter()
                    .map(|s| crate::proto::CheckpointShardInfo {
                        shard_index: s.shard_index,
                        worker_id: s.worker_id,
                        path: s.path,
                        size_bytes: s.size_bytes,
                        checksum: s.checksum,
                        keys: s.keys,
                    })
                    .collect(),
                total_size_bytes: m.total_size_bytes,
                total_checksum: m.total_checksum,
                metadata: m.metadata,
                worker_count: m.worker_count,
                format_version: m.format_version,
            }),
            error_message: checkpoint.error_message.unwrap_or_default(),
        };

        Ok(Response::new(response))
    }

    async fn get_worker_status(
        &self,
        request: Request<GetWorkerStatusRequest>,
    ) -> Result<Response<WorkerStatusResponse>, Status> {
        let req = request.into_inner();

        let workers = if !req.worker_id.is_empty() {
            self.state
                .get_worker(&req.worker_id)
                .await
                .into_iter()
                .collect()
        } else if !req.job_id.is_empty() {
            self.state.get_workers_for_job(&req.job_id).await
        } else {
            self.state.get_all_workers().await
        };

        let proto_workers: Vec<ProtoWorkerInfo> = workers
            .into_iter()
            .map(|w| {
                let info = w.to_worker_info();
                // Convert runtime_core WorkerStatus to proto WorkerStatus
                let proto_status = match info.status {
                    WorkerStatus::Idle => crate::proto::WorkerStatus::Idle,
                    WorkerStatus::Training => crate::proto::WorkerStatus::Training,
                    WorkerStatus::Checkpointing => crate::proto::WorkerStatus::Checkpointing,
                    WorkerStatus::Loading => crate::proto::WorkerStatus::Loading,
                    WorkerStatus::Error => crate::proto::WorkerStatus::Error,
                    WorkerStatus::ShuttingDown => crate::proto::WorkerStatus::ShuttingDown,
                };
                ProtoWorkerInfo {
                    worker_id: info.worker_id,
                    worker_index: info.worker_index,
                    status: proto_status as i32,
                    last_heartbeat_ms: info.last_heartbeat_ms,
                    step: info.step,
                    assignments: std::collections::HashMap::new(),
                    capabilities: Some(crate::proto::WorkerCapabilities {
                        gpu_count: info.capabilities.gpu_count,
                        memory_bytes: info.capabilities.memory_bytes,
                        has_local_storage: info.capabilities.has_local_storage,
                        accessible_paths: info.capabilities.accessible_paths,
                        custom: info.capabilities.custom,
                    }),
                    healthy: info.healthy,
                    hostname: info.hostname,
                }
            })
            .collect();

        Ok(Response::new(WorkerStatusResponse {
            workers: proto_workers,
        }))
    }

    async fn unregister_worker(
        &self,
        request: Request<UnregisterWorkerRequest>,
    ) -> Result<Response<UnregisterWorkerResponse>, Status> {
        let req = request.into_inner();
        self.validate_session(&req.worker_id, &req.session_token)?;

        if !self
            .state
            .validate_session(&req.worker_id, &req.session_token)
            .await
        {
            return Err(Status::unauthenticated("invalid session"));
        }

        let success = self.state.unregister_worker(&req.worker_id).await;

        tracing::info!(
            "Worker {} unregistered (reason: {})",
            req.worker_id,
            req.reason
        );

        Ok(Response::new(UnregisterWorkerResponse { success }))
    }
}
