// rust/runtime-core/src/lib.rs

//! Distributed Training Runtime - Core Library
//!
//! This crate provides core functionality for distributed training workloads,
//! including error handling, storage abstractions, dataset management, and
//! checkpoint operations.

pub mod config;
pub mod error;
pub mod storage;

// Re-export commonly used types for convenience
pub use config::RuntimeConfig;
pub use error::{Result, RuntimeError};
pub use storage::{LocalStorage, ObjectMeta, StorageBackend, StorageReader, StorageWriter};

pub mod checkpoint;
pub use checkpoint::{CheckpointHeader, CheckpointReader, CheckpointWriter};

pub mod dataset;
pub use dataset::{
    calculate_shards, Batch, FixedSizeFormat, IteratorConfig, LengthPrefixedFormat,
    NewlineDelimitedFormat, RecordFormat, ShardIterator, ShardSpec,
};

pub mod runtime;
pub use runtime::{Dataset, Runtime};
