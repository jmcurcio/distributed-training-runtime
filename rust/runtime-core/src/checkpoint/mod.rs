// rust/runtime-core/src/checkpoint/mod.rs

//! Checkpoint system for saving and loading training state.
//!
//! This module provides functionality for:
//! - Writing checkpoints with compression (none, lz4, zstd)
//! - Reading and verifying checkpoints
//! - Integrity verification via XXHash64 checksums
//! - Atomic writes to prevent partial checkpoints
//! - Automatic cleanup of old checkpoints
//!
//! # File Format
//!
//! Checkpoints use a binary format:
//! ```text
//! +------------------------+
//! | Header Length (4 bytes)|  <- u32 little-endian
//! +------------------------+
//! | Header (bincode)       |  <- CheckpointHeader serialized
//! +------------------------+
//! | Compressed Data        |  <- Payload compressed per header
//! +------------------------+
//! ```
//!
//! # Example
//!
//! ```no_run
//! use runtime_core::checkpoint::{CheckpointWriter, CheckpointReader};
//! use runtime_core::config::{CheckpointConfig, StorageConfig};
//! use runtime_core::storage::LocalStorage;
//! use std::sync::Arc;
//!
//! // Create storage and writer
//! let storage_config = StorageConfig::default();
//! let storage = Arc::new(LocalStorage::new(&storage_config).unwrap());
//!
//! let checkpoint_config = CheckpointConfig::default();
//! let writer = CheckpointWriter::new(storage.clone(), checkpoint_config);
//!
//! // Write a checkpoint
//! let data = b"training state data";
//! let path = writer.write("model", data).unwrap();
//!
//! // Read it back
//! let reader = CheckpointReader::new(storage);
//! let loaded = reader.read(&path).unwrap();
//! assert_eq!(loaded, data);
//! ```

mod format;
mod reader;
mod writer;

pub use format::CheckpointHeader;
pub use reader::CheckpointReader;
pub use writer::CheckpointWriter;
