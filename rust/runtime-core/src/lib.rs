// rust/runtime-core/src/lib.rs

//! Distributed Training Runtime - Core Library
//!
//! This crate provides core functionality for distributed training workloads,
//! including error handling, storage abstractions, dataset management, and
//! checkpoint operations.
//!
//! # Features
//!
//! - `local` (default): Local filesystem storage backend
//! - `s3`: S3-compatible object storage backend (requires `object_store` crate)
//! - `all-backends`: Enable all storage backends

pub mod config;
pub mod error;
pub mod storage;

// Re-export commonly used types for convenience
pub use config::RuntimeConfig;
pub use error::{Result, RuntimeError};

// Sync storage exports
pub use storage::{LocalStorage, ObjectMeta, StorageBackend, StorageReader, StorageWriter};

// Async storage exports
pub use storage::{
    AsyncLocalStorage, AsyncObjectMeta, AsyncStorageBackend, AsyncStorageReader,
    AsyncStorageWriter, ByteStream,
};

// S3 storage exports (feature-gated)
#[cfg(feature = "s3")]
pub use config::S3Config;
#[cfg(feature = "s3")]
pub use storage::{RetryConfig, S3Storage};

pub mod checkpoint;
pub use checkpoint::{CheckpointHeader, CheckpointReader, CheckpointWriter};

pub mod dataset;
pub use dataset::{
    calculate_shards, Batch, FixedSizeFormat, IteratorConfig, LengthPrefixedFormat,
    NewlineDelimitedFormat, RecordFormat, ShardIterator, ShardSpec,
};

pub mod runtime;
pub use runtime::{Dataset, Runtime};

// Async runtime module
pub mod async_runtime;
pub use async_runtime::AsyncRuntime;

// Coordinator module (feature-gated)
#[cfg(feature = "coordinator")]
pub mod coordinator;
#[cfg(feature = "coordinator")]
pub use config::{CoordinatorConfig, CoordinatorMode, ShardStrategy};
#[cfg(feature = "coordinator")]
pub use coordinator::{
    CoordinatorClient, GrpcCoordinatorClient, ShardAssigner, WorkerAssignment, WorkerInfo,
};
