//! Multi-worker coordination module.
//!
//! This module provides gRPC-based coordination between workers for distributed
//! training, including:
//!
//! - Worker registration and discovery
//! - Shard assignment and rebalancing
//! - Progress reporting and health monitoring
//! - Distributed checkpoint coordination
//!
//! # Feature
//!
//! This module requires the `coordinator` feature to be enabled.

mod client;
pub mod protocol;
mod shard_assignment;

pub mod checkpoint;

// Include generated protobuf code
pub mod proto {
    include!("proto/dtr.coordinator.rs");
}

// Re-exports
pub use client::{CoordinatorClient, GrpcCoordinatorClient};
pub use protocol::{
    CheckpointPlan, CheckpointShardConfirmation, ProgressReport, ShardRange, WorkerCapabilities,
    WorkerConfig, WorkerInfo, WorkerStatus,
};
pub use shard_assignment::{
    DynamicAssigner, LocalityAwareAssigner, ShardAssigner, StaticAssigner, WorkerAssignment,
};
