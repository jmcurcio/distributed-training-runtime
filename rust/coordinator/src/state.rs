//! In-memory state management for the coordinator service.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use uuid::Uuid;

use runtime_core::config::ShardStrategy;
use runtime_core::coordinator::{
    ShardAssigner, StaticAssigner, DynamicAssigner, LocalityAwareAssigner,
    WorkerAssignment, WorkerInfo,
    protocol::{
        CheckpointManifest, CheckpointShardConfirmation, ShardRange, WorkerCapabilities,
        WorkerStatus,
    },
};

/// Worker state tracked by the coordinator.
#[derive(Debug, Clone)]
pub struct WorkerState {
    pub worker_id: String,
    pub worker_index: u32,
    pub job_id: String,
    pub session_token: String,
    pub status: WorkerStatus,
    pub capabilities: WorkerCapabilities,
    pub hostname: String,
    pub registered_at: DateTime<Utc>,
    pub last_heartbeat: Instant,
    pub last_heartbeat_time: DateTime<Utc>,
    pub current_step: u64,
    pub healthy: bool,
}

impl WorkerState {
    pub fn new(
        worker_id: String,
        worker_index: u32,
        job_id: String,
        capabilities: WorkerCapabilities,
        hostname: String,
    ) -> Self {
        let session_token = Uuid::new_v4().to_string();
        let now = Utc::now();

        Self {
            worker_id,
            worker_index,
            job_id,
            session_token,
            status: WorkerStatus::Idle,
            capabilities,
            hostname,
            registered_at: now,
            last_heartbeat: Instant::now(),
            last_heartbeat_time: now,
            current_step: 0,
            healthy: true,
        }
    }

    pub fn update_heartbeat(&mut self, status: WorkerStatus, step: u64) {
        self.last_heartbeat = Instant::now();
        self.last_heartbeat_time = Utc::now();
        self.status = status;
        self.current_step = step;
        self.healthy = true;
    }

    pub fn is_timed_out(&self, timeout: Duration) -> bool {
        self.last_heartbeat.elapsed() > timeout
    }

    pub fn to_worker_info(&self) -> WorkerInfo {
        WorkerInfo {
            worker_id: self.worker_id.clone(),
            worker_index: self.worker_index,
            status: self.status,
            last_heartbeat_ms: self.last_heartbeat_time.timestamp_millis(),
            step: self.current_step,
            capabilities: self.capabilities.clone(),
            healthy: self.healthy,
            hostname: self.hostname.clone(),
        }
    }
}

/// Dataset assignment state.
#[derive(Debug, Clone)]
pub struct DatasetAssignment {
    pub dataset_id: String,
    pub total_shards: u32,
    pub strategy: ShardStrategy,
    pub assignments: HashMap<String, Vec<ShardRange>>,
    pub version: u64,
}

/// Checkpoint operation state.
#[derive(Debug, Clone)]
pub struct CheckpointState {
    pub checkpoint_id: String,
    pub checkpoint_name: String,
    pub step: u64,
    pub base_path: String,
    pub compression: String,
    pub initiated_at: DateTime<Utc>,
    pub deadline: DateTime<Utc>,
    pub expected_workers: Vec<String>,
    pub confirmations: HashMap<String, CheckpointShardConfirmation>,
    pub state: CheckpointOperationState,
    pub manifest: Option<CheckpointManifest>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointOperationState {
    InProgress,
    Finalizing,
    Complete,
    Failed,
    Timeout,
}

impl CheckpointState {
    pub fn new(
        checkpoint_name: String,
        step: u64,
        base_path: String,
        compression: String,
        expected_workers: Vec<String>,
        timeout_ms: i64,
    ) -> Self {
        let now = Utc::now();
        Self {
            checkpoint_id: Uuid::new_v4().to_string(),
            checkpoint_name,
            step,
            base_path,
            compression,
            initiated_at: now,
            deadline: now + chrono::Duration::milliseconds(timeout_ms),
            expected_workers,
            confirmations: HashMap::new(),
            state: CheckpointOperationState::InProgress,
            manifest: None,
            error_message: None,
        }
    }

    pub fn add_confirmation(&mut self, confirmation: CheckpointShardConfirmation) -> bool {
        if !confirmation.success {
            self.state = CheckpointOperationState::Failed;
            self.error_message = Some(confirmation.error_message.clone());
            return false;
        }

        self.confirmations
            .insert(confirmation.worker_id.clone(), confirmation);

        // Check if all workers have confirmed
        if self.confirmations.len() == self.expected_workers.len() {
            self.state = CheckpointOperationState::Finalizing;
            return true;
        }

        false
    }

    pub fn is_timed_out(&self) -> bool {
        Utc::now() > self.deadline
    }

    pub fn confirmed_count(&self) -> u32 {
        self.confirmations.len() as u32
    }

    pub fn total_workers(&self) -> u32 {
        self.expected_workers.len() as u32
    }
}

/// Main coordinator state container.
pub struct CoordinatorState {
    /// Registered workers by ID.
    pub workers: RwLock<HashMap<String, WorkerState>>,
    /// Dataset assignments.
    pub assignments: RwLock<HashMap<String, DatasetAssignment>>,
    /// Active checkpoint operations.
    pub checkpoints: RwLock<HashMap<String, CheckpointState>>,
    /// Worker timeout duration.
    pub worker_timeout: Duration,
    /// Default shard strategy.
    pub default_strategy: ShardStrategy,
    /// Checkpoint timeout in milliseconds.
    pub checkpoint_timeout_ms: i64,
    /// Heartbeat interval recommendation.
    pub heartbeat_interval_ms: u64,
    /// Next worker index counter.
    worker_index_counter: RwLock<u32>,
}

impl CoordinatorState {
    pub fn new(
        worker_timeout: Duration,
        default_strategy: ShardStrategy,
        checkpoint_timeout_ms: i64,
        heartbeat_interval_ms: u64,
    ) -> Self {
        Self {
            workers: RwLock::new(HashMap::new()),
            assignments: RwLock::new(HashMap::new()),
            checkpoints: RwLock::new(HashMap::new()),
            worker_timeout,
            default_strategy,
            checkpoint_timeout_ms,
            heartbeat_interval_ms,
            worker_index_counter: RwLock::new(0),
        }
    }

    /// Register a new worker.
    pub async fn register_worker(
        &self,
        worker_id: Option<String>,
        job_id: String,
        capabilities: WorkerCapabilities,
        hostname: String,
    ) -> (WorkerState, bool) {
        let mut workers = self.workers.write().await;

        // Check for existing worker (reconnection)
        let provided_id = worker_id.clone();
        let worker_id = worker_id.unwrap_or_else(|| Uuid::new_v4().to_string());

        if let Some(existing) = workers.get_mut(&worker_id) {
            // Reconnection - update state
            existing.session_token = Uuid::new_v4().to_string();
            existing.last_heartbeat = Instant::now();
            existing.last_heartbeat_time = Utc::now();
            existing.healthy = true;
            existing.capabilities = capabilities;
            return (existing.clone(), true);
        }

        // New worker registration
        let worker_index = {
            let mut counter = self.worker_index_counter.write().await;
            let idx = *counter;
            *counter += 1;
            idx
        };

        let worker = WorkerState::new(worker_id.clone(), worker_index, job_id, capabilities, hostname);
        workers.insert(worker_id, worker.clone());

        (worker, false)
    }

    /// Unregister a worker.
    pub async fn unregister_worker(&self, worker_id: &str) -> bool {
        let mut workers = self.workers.write().await;
        workers.remove(worker_id).is_some()
    }

    /// Get worker by ID.
    pub async fn get_worker(&self, worker_id: &str) -> Option<WorkerState> {
        let workers = self.workers.read().await;
        workers.get(worker_id).cloned()
    }

    /// Get all workers.
    pub async fn get_all_workers(&self) -> Vec<WorkerState> {
        let workers = self.workers.read().await;
        workers.values().cloned().collect()
    }

    /// Get workers for a job.
    pub async fn get_workers_for_job(&self, job_id: &str) -> Vec<WorkerState> {
        let workers = self.workers.read().await;
        workers
            .values()
            .filter(|w| w.job_id == job_id)
            .cloned()
            .collect()
    }

    /// Update worker heartbeat.
    pub async fn update_heartbeat(
        &self,
        worker_id: &str,
        session_token: &str,
        status: WorkerStatus,
        step: u64,
    ) -> Option<WorkerState> {
        let mut workers = self.workers.write().await;

        if let Some(worker) = workers.get_mut(worker_id) {
            if worker.session_token == session_token {
                worker.update_heartbeat(status, step);
                return Some(worker.clone());
            }
        }

        None
    }

    /// Check for timed out workers and mark them unhealthy.
    pub async fn check_worker_timeouts(&self) -> Vec<String> {
        let mut workers = self.workers.write().await;
        let mut timed_out = Vec::new();

        for worker in workers.values_mut() {
            if worker.is_timed_out(self.worker_timeout) {
                worker.healthy = false;
                timed_out.push(worker.worker_id.clone());
            }
        }

        timed_out
    }

    /// Get total worker count.
    pub async fn worker_count(&self) -> u32 {
        let workers = self.workers.read().await;
        workers.len() as u32
    }

    /// Assign shards to workers for a dataset.
    pub async fn assign_shards(
        &self,
        dataset_id: &str,
        total_shards: u32,
        strategy: Option<ShardStrategy>,
    ) -> DatasetAssignment {
        let strategy = strategy.unwrap_or(self.default_strategy.clone());
        let workers = self.get_all_workers().await;
        let worker_infos: Vec<_> = workers.iter().map(|w| w.to_worker_info()).collect();

        let assigner: Box<dyn ShardAssigner> = match strategy {
            ShardStrategy::Static => Box::new(StaticAssigner),
            ShardStrategy::Dynamic => Box::new(DynamicAssigner),
            ShardStrategy::LocalityAware => Box::new(LocalityAwareAssigner::new()),
        };

        let assignments = assigner.assign_shards(total_shards, &worker_infos);

        let assignment_map: HashMap<String, Vec<ShardRange>> = assignments
            .into_iter()
            .map(|a| {
                (a.worker_id, a.ranges)
            })
            .collect();

        let mut assignments_lock = self.assignments.write().await;

        let version = assignments_lock
            .get(dataset_id)
            .map(|a| a.version + 1)
            .unwrap_or(1);

        let dataset_assignment = DatasetAssignment {
            dataset_id: dataset_id.to_string(),
            total_shards,
            strategy,
            assignments: assignment_map,
            version,
        };

        assignments_lock.insert(dataset_id.to_string(), dataset_assignment.clone());

        dataset_assignment
    }

    /// Get shard assignment for a worker.
    pub async fn get_worker_assignment(
        &self,
        worker_id: &str,
        dataset_id: &str,
        total_shards: u32,
    ) -> Option<(Vec<ShardRange>, u64)> {
        // Check if assignment exists
        {
            let assignments = self.assignments.read().await;
            if let Some(assignment) = assignments.get(dataset_id) {
                if let Some(ranges) = assignment.assignments.get(worker_id) {
                    return Some((ranges.clone(), assignment.version));
                }
            }
        }

        // Create new assignment
        let assignment = self.assign_shards(dataset_id, total_shards, None).await;
        assignment
            .assignments
            .get(worker_id)
            .map(|ranges| (ranges.clone(), assignment.version))
    }

    /// Initiate a checkpoint operation.
    pub async fn initiate_checkpoint(
        &self,
        checkpoint_name: String,
        step: u64,
        base_path: String,
        compression: String,
    ) -> CheckpointState {
        let workers = self.get_all_workers().await;
        let expected_workers: Vec<_> = workers.iter().map(|w| w.worker_id.clone()).collect();

        let checkpoint = CheckpointState::new(
            checkpoint_name,
            step,
            base_path,
            compression,
            expected_workers,
            self.checkpoint_timeout_ms,
        );

        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.insert(checkpoint.checkpoint_id.clone(), checkpoint.clone());

        checkpoint
    }

    /// Add checkpoint shard confirmation.
    pub async fn confirm_checkpoint_shard(
        &self,
        checkpoint_id: &str,
        confirmation: CheckpointShardConfirmation,
    ) -> Option<CheckpointState> {
        let mut checkpoints = self.checkpoints.write().await;

        if let Some(checkpoint) = checkpoints.get_mut(checkpoint_id) {
            checkpoint.add_confirmation(confirmation);
            return Some(checkpoint.clone());
        }

        None
    }

    /// Get checkpoint state.
    pub async fn get_checkpoint(&self, checkpoint_id: &str) -> Option<CheckpointState> {
        let checkpoints = self.checkpoints.read().await;
        checkpoints.get(checkpoint_id).cloned()
    }

    /// Mark checkpoint as complete with manifest.
    pub async fn complete_checkpoint(
        &self,
        checkpoint_id: &str,
        manifest: CheckpointManifest,
    ) -> bool {
        let mut checkpoints = self.checkpoints.write().await;

        if let Some(checkpoint) = checkpoints.get_mut(checkpoint_id) {
            checkpoint.state = CheckpointOperationState::Complete;
            checkpoint.manifest = Some(manifest);
            return true;
        }

        false
    }

    /// Check for timed out checkpoints.
    pub async fn check_checkpoint_timeouts(&self) -> Vec<String> {
        let mut checkpoints = self.checkpoints.write().await;
        let mut timed_out = Vec::new();

        for checkpoint in checkpoints.values_mut() {
            if checkpoint.state == CheckpointOperationState::InProgress
                && checkpoint.is_timed_out()
            {
                checkpoint.state = CheckpointOperationState::Timeout;
                timed_out.push(checkpoint.checkpoint_id.clone());
            }
        }

        timed_out
    }

    /// Validate session token for a worker.
    pub async fn validate_session(&self, worker_id: &str, session_token: &str) -> bool {
        let workers = self.workers.read().await;
        workers
            .get(worker_id)
            .map(|w| w.session_token == session_token)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_worker_registration() {
        let state = CoordinatorState::new(
            Duration::from_secs(30),
            ShardStrategy::Static,
            60000,
            5000,
        );

        let (worker, reconnected) = state
            .register_worker(
                Some("worker-0".to_string()),
                "job-1".to_string(),
                WorkerCapabilities::default(),
                "localhost".to_string(),
            )
            .await;

        assert!(!reconnected);
        assert_eq!(worker.worker_id, "worker-0");
        assert_eq!(worker.worker_index, 0);

        // Second registration should get index 1
        let (worker2, _) = state
            .register_worker(
                Some("worker-1".to_string()),
                "job-1".to_string(),
                WorkerCapabilities::default(),
                "localhost".to_string(),
            )
            .await;

        assert_eq!(worker2.worker_index, 1);
    }

    #[tokio::test]
    async fn test_worker_reconnection() {
        let state = CoordinatorState::new(
            Duration::from_secs(30),
            ShardStrategy::Static,
            60000,
            5000,
        );

        let (worker1, _) = state
            .register_worker(
                Some("worker-0".to_string()),
                "job-1".to_string(),
                WorkerCapabilities::default(),
                "localhost".to_string(),
            )
            .await;

        let original_token = worker1.session_token.clone();

        // Reconnect same worker
        let (worker2, reconnected) = state
            .register_worker(
                Some("worker-0".to_string()),
                "job-1".to_string(),
                WorkerCapabilities::default(),
                "localhost".to_string(),
            )
            .await;

        assert!(reconnected);
        assert_eq!(worker2.worker_index, 0);
        assert_ne!(worker2.session_token, original_token);
    }

    #[tokio::test]
    async fn test_shard_assignment() {
        let state = CoordinatorState::new(
            Duration::from_secs(30),
            ShardStrategy::Static,
            60000,
            5000,
        );

        // Register workers
        for i in 0..4 {
            state
                .register_worker(
                    Some(format!("worker-{}", i)),
                    "job-1".to_string(),
                    WorkerCapabilities::default(),
                    "localhost".to_string(),
                )
                .await;
        }

        // Assign shards
        let assignment = state.assign_shards("dataset-1", 8, None).await;

        assert_eq!(assignment.assignments.len(), 4);
        assert_eq!(assignment.version, 1);

        // Each worker should have 2 shards
        for ranges in assignment.assignments.values() {
            let total: u32 = ranges.iter().map(|r| r.count()).sum();
            assert_eq!(total, 2);
        }
    }

    #[tokio::test]
    async fn test_checkpoint_flow() {
        let state = CoordinatorState::new(
            Duration::from_secs(30),
            ShardStrategy::Static,
            60000,
            5000,
        );

        // Register workers
        state
            .register_worker(
                Some("worker-0".to_string()),
                "job-1".to_string(),
                WorkerCapabilities::default(),
                "localhost".to_string(),
            )
            .await;
        state
            .register_worker(
                Some("worker-1".to_string()),
                "job-1".to_string(),
                WorkerCapabilities::default(),
                "localhost".to_string(),
            )
            .await;

        // Initiate checkpoint
        let checkpoint = state
            .initiate_checkpoint(
                "epoch_10".to_string(),
                1000,
                "/checkpoints".to_string(),
                "lz4".to_string(),
            )
            .await;

        assert_eq!(checkpoint.expected_workers.len(), 2);
        assert_eq!(checkpoint.state, CheckpointOperationState::InProgress);

        // Confirm first shard
        let conf1 = CheckpointShardConfirmation {
            worker_id: "worker-0".to_string(),
            session_token: String::new(),
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            shard_path: "/checkpoints/shard_0".to_string(),
            shard_index: 0,
            size_bytes: 1024,
            checksum: 12345,
            success: true,
            error_message: String::new(),
        };

        let updated = state
            .confirm_checkpoint_shard(&checkpoint.checkpoint_id, conf1)
            .await
            .unwrap();

        assert_eq!(updated.confirmed_count(), 1);
        assert_eq!(updated.state, CheckpointOperationState::InProgress);

        // Confirm second shard
        let conf2 = CheckpointShardConfirmation {
            worker_id: "worker-1".to_string(),
            session_token: String::new(),
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            shard_path: "/checkpoints/shard_1".to_string(),
            shard_index: 1,
            size_bytes: 2048,
            checksum: 67890,
            success: true,
            error_message: String::new(),
        };

        let updated = state
            .confirm_checkpoint_shard(&checkpoint.checkpoint_id, conf2)
            .await
            .unwrap();

        assert_eq!(updated.confirmed_count(), 2);
        assert_eq!(updated.state, CheckpointOperationState::Finalizing);
    }
}
