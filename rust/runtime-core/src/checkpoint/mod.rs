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
//! # Format Versions
//!
//! ## V1 Format (Sync)
//!
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
//! ## V2 Format (Async/Streaming)
//!
//! ```text
//! +------------------------+
//! | Magic "DTR2" (4 bytes) |
//! +------------------------+
//! | Flags (4 bytes)        |
//! +------------------------+
//! | Reserved (24 bytes)    |
//! +------------------------+
//! | Compressed Data        |  <- Variable length
//! +------------------------+
//! | Trailer (bincode)      |  <- CheckpointTrailer
//! +------------------------+
//! | Trailer Length (4 bytes)|
//! +------------------------+
//! ```
//!
//! The V2 format places the checksum and metadata in a trailer at the end,
//! allowing streaming writes without knowing the final size upfront.
//!
//! # Example (Sync)
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
//!
//! # Example (Async)
//!
//! ```no_run
//! use runtime_core::checkpoint::{AsyncCheckpointWriter, AsyncCheckpointReader};
//! use runtime_core::config::{CheckpointConfig, StorageConfig};
//! use runtime_core::storage::AsyncLocalStorage;
//! use std::sync::Arc;
//!
//! # async fn example() -> runtime_core::Result<()> {
//! // Create async storage and writer
//! let storage_config = StorageConfig::default();
//! let storage: Arc<dyn runtime_core::AsyncStorageBackend> =
//!     Arc::new(AsyncLocalStorage::new(&storage_config).await?);
//!
//! let checkpoint_config = CheckpointConfig::default();
//! let writer = AsyncCheckpointWriter::new(storage.clone(), checkpoint_config);
//!
//! // Write a checkpoint (V2 format)
//! let data = b"training state data";
//! let path = writer.write("model", data).await?;
//!
//! // Read it back (supports both V1 and V2)
//! let reader = AsyncCheckpointReader::new(storage);
//! let loaded = reader.read(&path).await?;
//! assert_eq!(loaded, data);
//! # Ok(())
//! # }
//! ```

// Sync checkpoint (V1 format)
mod format;
mod reader;
mod writer;

pub use format::CheckpointHeader;
pub use reader::CheckpointReader;
pub use writer::CheckpointWriter;

// Async checkpoint (V2 format)
mod async_reader;
mod async_writer;
mod format_v2;

pub use async_reader::{AsyncCheckpointReader, CheckpointMetadata};
pub use async_writer::{AsyncCheckpointWriter, StreamingCheckpointWriter};
pub use format_v2::{
    CheckpointHeaderV2, CheckpointTrailer, CheckpointVersion, ChunkInfo, CompressionType,
};
