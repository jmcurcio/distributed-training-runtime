//! Protocol wrapper types for coordinator messages.
//!
//! This module provides Rust-friendly wrapper types around the generated
//! protobuf messages, with conversion traits for ergonomic usage.

use std::collections::HashMap;

use crate::error::{Result, RuntimeError};

// Re-export proto types for internal use
pub(crate) use super::proto;

/// Worker status enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkerStatus {
    #[default]
    Idle,
    Training,
    Checkpointing,
    Loading,
    Error,
    ShuttingDown,
}

impl From<proto::WorkerStatus> for WorkerStatus {
    fn from(status: proto::WorkerStatus) -> Self {
        match status {
            proto::WorkerStatus::Unspecified => Self::Idle,
            proto::WorkerStatus::Idle => Self::Idle,
            proto::WorkerStatus::Training => Self::Training,
            proto::WorkerStatus::Checkpointing => Self::Checkpointing,
            proto::WorkerStatus::Loading => Self::Loading,
            proto::WorkerStatus::Error => Self::Error,
            proto::WorkerStatus::ShuttingDown => Self::ShuttingDown,
        }
    }
}

impl From<WorkerStatus> for proto::WorkerStatus {
    fn from(status: WorkerStatus) -> Self {
        match status {
            WorkerStatus::Idle => Self::Idle,
            WorkerStatus::Training => Self::Training,
            WorkerStatus::Checkpointing => Self::Checkpointing,
            WorkerStatus::Loading => Self::Loading,
            WorkerStatus::Error => Self::Error,
            WorkerStatus::ShuttingDown => Self::ShuttingDown,
        }
    }
}

impl From<i32> for WorkerStatus {
    fn from(value: i32) -> Self {
        proto::WorkerStatus::try_from(value)
            .unwrap_or(proto::WorkerStatus::Unspecified)
            .into()
    }
}

/// Worker capabilities for assignment decisions.
#[derive(Debug, Clone, Default)]
pub struct WorkerCapabilities {
    /// Number of GPUs available.
    pub gpu_count: u32,
    /// Total memory in bytes.
    pub memory_bytes: u64,
    /// Whether worker can access local storage.
    pub has_local_storage: bool,
    /// Storage paths accessible by this worker.
    pub accessible_paths: Vec<String>,
    /// Custom capability flags.
    pub custom: HashMap<String, String>,
}

impl From<proto::WorkerCapabilities> for WorkerCapabilities {
    fn from(caps: proto::WorkerCapabilities) -> Self {
        Self {
            gpu_count: caps.gpu_count,
            memory_bytes: caps.memory_bytes,
            has_local_storage: caps.has_local_storage,
            accessible_paths: caps.accessible_paths,
            custom: caps.custom,
        }
    }
}

impl From<WorkerCapabilities> for proto::WorkerCapabilities {
    fn from(caps: WorkerCapabilities) -> Self {
        Self {
            gpu_count: caps.gpu_count,
            memory_bytes: caps.memory_bytes,
            has_local_storage: caps.has_local_storage,
            accessible_paths: caps.accessible_paths,
            custom: caps.custom,
        }
    }
}

/// Configuration assigned to a worker by the coordinator.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Confirmed worker ID.
    pub worker_id: String,
    /// Worker index in the job (0-based).
    pub worker_index: u32,
    /// Total number of workers in the job.
    pub total_workers: u32,
    /// Job identifier.
    pub job_id: String,
    /// Recommended heartbeat interval in milliseconds.
    pub heartbeat_interval_ms: u64,
    /// Session token for authentication.
    pub session_token: String,
    /// Coordinator-assigned settings.
    pub settings: HashMap<String, String>,
}

impl From<proto::WorkerConfig> for WorkerConfig {
    fn from(config: proto::WorkerConfig) -> Self {
        Self {
            worker_id: config.worker_id,
            worker_index: config.worker_index,
            total_workers: config.total_workers,
            job_id: config.job_id,
            heartbeat_interval_ms: config.heartbeat_interval_ms,
            session_token: config.session_token,
            settings: config.settings,
        }
    }
}

/// Shard range for assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardRange {
    /// Start shard ID (inclusive).
    pub start_shard: u32,
    /// End shard ID (exclusive).
    pub end_shard: u32,
    /// Optional priority for processing order.
    pub priority: u32,
}

impl ShardRange {
    /// Create a new shard range.
    pub fn new(start: u32, end: u32) -> Self {
        Self {
            start_shard: start,
            end_shard: end,
            priority: 0,
        }
    }

    /// Create a new shard range with priority.
    pub fn with_priority(start: u32, end: u32, priority: u32) -> Self {
        Self {
            start_shard: start,
            end_shard: end,
            priority,
        }
    }

    /// Get the number of shards in this range.
    pub fn count(&self) -> u32 {
        self.end_shard.saturating_sub(self.start_shard)
    }

    /// Check if a shard ID is in this range.
    pub fn contains(&self, shard_id: u32) -> bool {
        shard_id >= self.start_shard && shard_id < self.end_shard
    }

    /// Iterate over shard IDs in this range.
    pub fn iter(&self) -> impl Iterator<Item = u32> {
        self.start_shard..self.end_shard
    }
}

impl From<proto::ShardRange> for ShardRange {
    fn from(range: proto::ShardRange) -> Self {
        Self {
            start_shard: range.start_shard,
            end_shard: range.end_shard,
            priority: range.priority,
        }
    }
}

impl From<ShardRange> for proto::ShardRange {
    fn from(range: ShardRange) -> Self {
        Self {
            start_shard: range.start_shard,
            end_shard: range.end_shard,
            priority: range.priority,
        }
    }
}

/// Progress report from a worker.
#[derive(Debug, Clone, Default)]
pub struct ProgressReport {
    /// Worker ID.
    pub worker_id: String,
    /// Session token.
    pub session_token: String,
    /// Training step/iteration.
    pub step: u64,
    /// Samples processed in this report period.
    pub samples_processed: u64,
    /// Current loss value (optional).
    pub loss: Option<f64>,
    /// Custom metrics.
    pub metrics: HashMap<String, f64>,
    /// Timestamp (Unix milliseconds).
    pub timestamp_ms: i64,
    /// Current dataset being processed.
    pub current_dataset: String,
    /// Current shard being processed.
    pub current_shard: u32,
}

impl From<ProgressReport> for proto::ProgressReport {
    fn from(report: ProgressReport) -> Self {
        Self {
            worker_id: report.worker_id,
            session_token: report.session_token,
            step: report.step,
            samples_processed: report.samples_processed,
            loss: report.loss,
            metrics: report.metrics,
            timestamp_ms: report.timestamp_ms,
            current_dataset: report.current_dataset,
            current_shard: report.current_shard,
        }
    }
}

/// Checkpoint plan distributed to workers.
#[derive(Debug, Clone)]
pub struct CheckpointPlan {
    /// Unique checkpoint operation ID.
    pub checkpoint_id: String,
    /// Checkpoint name.
    pub checkpoint_name: String,
    /// Training step.
    pub step: u64,
    /// Base path for checkpoint shards.
    pub base_path: String,
    /// Worker assignments.
    pub assignments: Vec<WorkerCheckpointAssignment>,
    /// Deadline for completing checkpoint (Unix milliseconds).
    pub deadline_ms: i64,
    /// Compression algorithm.
    pub compression: String,
}

impl From<proto::CheckpointPlan> for CheckpointPlan {
    fn from(plan: proto::CheckpointPlan) -> Self {
        Self {
            checkpoint_id: plan.checkpoint_id,
            checkpoint_name: plan.checkpoint_name,
            step: plan.step,
            base_path: plan.base_path,
            assignments: plan
                .assignments
                .into_iter()
                .map(WorkerCheckpointAssignment::from)
                .collect(),
            deadline_ms: plan.deadline_ms,
            compression: plan.compression,
        }
    }
}

/// Worker checkpoint assignment.
#[derive(Debug, Clone)]
pub struct WorkerCheckpointAssignment {
    pub worker_id: String,
    pub shard_path: String,
    pub shard_index: u32,
    pub assigned_keys: Vec<String>,
}

impl From<proto::WorkerCheckpointAssignment> for WorkerCheckpointAssignment {
    fn from(assignment: proto::WorkerCheckpointAssignment) -> Self {
        Self {
            worker_id: assignment.worker_id,
            shard_path: assignment.shard_path,
            shard_index: assignment.shard_index,
            assigned_keys: assignment.assigned_keys,
        }
    }
}

impl From<WorkerCheckpointAssignment> for proto::WorkerCheckpointAssignment {
    fn from(assignment: WorkerCheckpointAssignment) -> Self {
        Self {
            worker_id: assignment.worker_id,
            shard_path: assignment.shard_path,
            shard_index: assignment.shard_index,
            assigned_keys: assignment.assigned_keys,
        }
    }
}

/// Checkpoint shard confirmation from a worker.
#[derive(Debug, Clone)]
pub struct CheckpointShardConfirmation {
    pub worker_id: String,
    pub session_token: String,
    pub checkpoint_id: String,
    pub shard_path: String,
    pub shard_index: u32,
    pub size_bytes: u64,
    pub checksum: u64,
    pub success: bool,
    pub error_message: String,
}

impl From<CheckpointShardConfirmation> for proto::CheckpointShardConfirmation {
    fn from(conf: CheckpointShardConfirmation) -> Self {
        Self {
            worker_id: conf.worker_id,
            session_token: conf.session_token,
            checkpoint_id: conf.checkpoint_id,
            shard_path: conf.shard_path,
            shard_index: conf.shard_index,
            size_bytes: conf.size_bytes,
            checksum: conf.checksum,
            success: conf.success,
            error_message: conf.error_message,
        }
    }
}

/// Worker information.
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub worker_id: String,
    pub worker_index: u32,
    pub status: WorkerStatus,
    pub last_heartbeat_ms: i64,
    pub step: u64,
    pub capabilities: WorkerCapabilities,
    pub healthy: bool,
    pub hostname: String,
}

impl From<proto::WorkerInfo> for WorkerInfo {
    fn from(info: proto::WorkerInfo) -> Self {
        Self {
            worker_id: info.worker_id,
            worker_index: info.worker_index,
            status: info.status.into(),
            last_heartbeat_ms: info.last_heartbeat_ms,
            step: info.step,
            capabilities: info
                .capabilities
                .map(WorkerCapabilities::from)
                .unwrap_or_default(),
            healthy: info.healthy,
            hostname: info.hostname,
        }
    }
}

/// Coordinator command enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoordinatorCommand {
    #[default]
    Continue,
    Pause,
    Checkpoint,
    Stop,
    RefreshAssignments,
}

impl From<proto::CoordinatorCommand> for CoordinatorCommand {
    fn from(cmd: proto::CoordinatorCommand) -> Self {
        match cmd {
            proto::CoordinatorCommand::Unspecified => Self::Continue,
            proto::CoordinatorCommand::Continue => Self::Continue,
            proto::CoordinatorCommand::Pause => Self::Pause,
            proto::CoordinatorCommand::Checkpoint => Self::Checkpoint,
            proto::CoordinatorCommand::Stop => Self::Stop,
            proto::CoordinatorCommand::RefreshAssignments => Self::RefreshAssignments,
        }
    }
}

impl From<i32> for CoordinatorCommand {
    fn from(value: i32) -> Self {
        proto::CoordinatorCommand::try_from(value)
            .unwrap_or(proto::CoordinatorCommand::Unspecified)
            .into()
    }
}

/// Heartbeat response from coordinator.
#[derive(Debug, Clone)]
pub struct HeartbeatResponse {
    pub registered: bool,
    pub command: CoordinatorCommand,
    pub checkpoint_plan: Option<CheckpointPlan>,
    pub leader_worker_id: String,
}

impl From<proto::HeartbeatResponse> for HeartbeatResponse {
    fn from(resp: proto::HeartbeatResponse) -> Self {
        Self {
            registered: resp.registered,
            command: resp.command.into(),
            checkpoint_plan: resp.checkpoint_plan.map(CheckpointPlan::from),
            leader_worker_id: resp.leader_worker_id,
        }
    }
}

/// Shard assignment response.
#[derive(Debug, Clone)]
pub struct ShardAssignment {
    pub ranges: Vec<ShardRange>,
    pub version: u64,
    pub strategy: crate::config::ShardStrategy,
}

impl From<proto::GetShardAssignmentResponse> for ShardAssignment {
    fn from(resp: proto::GetShardAssignmentResponse) -> Self {
        let strategy = match resp.strategy {
            x if x == proto::ShardStrategy::Static as i32 => crate::config::ShardStrategy::Static,
            x if x == proto::ShardStrategy::Dynamic as i32 => crate::config::ShardStrategy::Dynamic,
            x if x == proto::ShardStrategy::LocalityAware as i32 => {
                crate::config::ShardStrategy::LocalityAware
            }
            _ => crate::config::ShardStrategy::Static,
        };
        Self {
            ranges: resp.shard_ranges.into_iter().map(ShardRange::from).collect(),
            version: resp.assignment_version,
            strategy,
        }
    }
}

/// Progress acknowledgment from coordinator.
#[derive(Debug, Clone)]
pub struct ProgressAck {
    pub continue_training: bool,
    pub command: CoordinatorCommand,
    pub server_timestamp_ms: i64,
}

impl From<proto::ProgressAck> for ProgressAck {
    fn from(ack: proto::ProgressAck) -> Self {
        Self {
            continue_training: ack.continue_training,
            command: ack.command.into(),
            server_timestamp_ms: ack.server_timestamp_ms,
        }
    }
}

/// Checkpoint shard acknowledgment.
#[derive(Debug, Clone)]
pub struct CheckpointShardAck {
    pub accepted: bool,
    pub confirmed_count: u32,
    pub total_workers: u32,
    pub checkpoint_complete: bool,
}

impl From<proto::CheckpointShardAck> for CheckpointShardAck {
    fn from(ack: proto::CheckpointShardAck) -> Self {
        Self {
            accepted: ack.accepted,
            confirmed_count: ack.confirmed_count,
            total_workers: ack.total_workers,
            checkpoint_complete: ack.checkpoint_complete,
        }
    }
}

/// Checkpoint manifest containing all shard information.
#[derive(Debug, Clone)]
pub struct CheckpointManifest {
    pub checkpoint_id: String,
    pub checkpoint_name: String,
    pub step: u64,
    pub created_at_ms: i64,
    pub shards: Vec<CheckpointShardInfo>,
    pub total_size_bytes: u64,
    pub total_checksum: u64,
    pub metadata: HashMap<String, String>,
    pub worker_count: u32,
    pub format_version: u32,
}

impl From<proto::CheckpointManifest> for CheckpointManifest {
    fn from(manifest: proto::CheckpointManifest) -> Self {
        Self {
            checkpoint_id: manifest.checkpoint_id,
            checkpoint_name: manifest.checkpoint_name,
            step: manifest.step,
            created_at_ms: manifest.created_at_ms,
            shards: manifest
                .shards
                .into_iter()
                .map(CheckpointShardInfo::from)
                .collect(),
            total_size_bytes: manifest.total_size_bytes,
            total_checksum: manifest.total_checksum,
            metadata: manifest.metadata,
            worker_count: manifest.worker_count,
            format_version: manifest.format_version,
        }
    }
}

impl From<CheckpointManifest> for proto::CheckpointManifest {
    fn from(manifest: CheckpointManifest) -> Self {
        Self {
            checkpoint_id: manifest.checkpoint_id,
            checkpoint_name: manifest.checkpoint_name,
            step: manifest.step,
            created_at_ms: manifest.created_at_ms,
            shards: manifest.shards.into_iter().map(proto::CheckpointShardInfo::from).collect(),
            total_size_bytes: manifest.total_size_bytes,
            total_checksum: manifest.total_checksum,
            metadata: manifest.metadata,
            worker_count: manifest.worker_count,
            format_version: manifest.format_version,
        }
    }
}

/// Checkpoint shard information.
#[derive(Debug, Clone)]
pub struct CheckpointShardInfo {
    pub shard_index: u32,
    pub worker_id: String,
    pub path: String,
    pub size_bytes: u64,
    pub checksum: u64,
    pub keys: Vec<String>,
}

impl From<proto::CheckpointShardInfo> for CheckpointShardInfo {
    fn from(info: proto::CheckpointShardInfo) -> Self {
        Self {
            shard_index: info.shard_index,
            worker_id: info.worker_id,
            path: info.path,
            size_bytes: info.size_bytes,
            checksum: info.checksum,
            keys: info.keys,
        }
    }
}

impl From<CheckpointShardInfo> for proto::CheckpointShardInfo {
    fn from(info: CheckpointShardInfo) -> Self {
        Self {
            shard_index: info.shard_index,
            worker_id: info.worker_id,
            path: info.path,
            size_bytes: info.size_bytes,
            checksum: info.checksum,
            keys: info.keys,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_range_basic() {
        let range = ShardRange::new(0, 10);
        assert_eq!(range.count(), 10);
        assert!(range.contains(0));
        assert!(range.contains(5));
        assert!(range.contains(9));
        assert!(!range.contains(10));
    }

    #[test]
    fn test_shard_range_iter() {
        let range = ShardRange::new(5, 8);
        let shards: Vec<_> = range.iter().collect();
        assert_eq!(shards, vec![5, 6, 7]);
    }

    #[test]
    fn test_worker_status_conversion() {
        assert_eq!(WorkerStatus::from(proto::WorkerStatus::Idle), WorkerStatus::Idle);
        assert_eq!(WorkerStatus::from(proto::WorkerStatus::Training), WorkerStatus::Training);
        assert_eq!(
            proto::WorkerStatus::from(WorkerStatus::Checkpointing),
            proto::WorkerStatus::Checkpointing
        );
    }

    #[test]
    fn test_coordinator_command_conversion() {
        assert_eq!(
            CoordinatorCommand::from(proto::CoordinatorCommand::Stop),
            CoordinatorCommand::Stop
        );
        assert_eq!(
            CoordinatorCommand::from(proto::CoordinatorCommand::Checkpoint),
            CoordinatorCommand::Checkpoint
        );
    }
}
