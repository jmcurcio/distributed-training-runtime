// rust/runtime-core/src/storage/mod.rs

//! Storage abstraction for the distributed training runtime.
//!
//! This module provides traits and implementations for storage backends,
//! allowing the runtime to work with different storage systems (local
//! filesystem, S3, etc.) through a unified interface.
//!
//! # Sync vs Async
//!
//! The module provides both synchronous and asynchronous storage traits:
//!
//! - **Sync traits** (`StorageBackend`, `StorageReader`, `StorageWriter`): For
//!   simple, blocking I/O operations. Use these when async is not needed.
//!
//! - **Async traits** (`AsyncStorageBackend`, `AsyncStorageReader`,
//!   `AsyncStorageWriter`): For non-blocking I/O with tokio. Use these for
//!   high-performance scenarios, especially with remote storage like S3.
//!
//! # Example (Sync)
//!
//! ```no_run
//! use runtime_core::config::StorageConfig;
//! use runtime_core::storage::{LocalStorage, StorageBackend};
//! use std::io::{Read, Write};
//! use std::path::Path;
//!
//! let config = StorageConfig::default();
//! let storage = LocalStorage::new(&config).unwrap();
//!
//! // Write a file
//! let mut writer = storage.open_write(Path::new("example.txt")).unwrap();
//! writer.write_all(b"Hello, world!").unwrap();
//! writer.finish().unwrap();
//!
//! // Read it back
//! let mut reader = storage.open_read(Path::new("example.txt")).unwrap();
//! let mut content = String::new();
//! reader.read_to_string(&mut content).unwrap();
//! ```
//!
//! # Example (Async)
//!
//! ```no_run
//! use runtime_core::config::StorageConfig;
//! use runtime_core::storage::{AsyncLocalStorage, AsyncStorageBackend};
//! use std::path::Path;
//!
//! # async fn example() -> runtime_core::Result<()> {
//! let config = StorageConfig::default();
//! let storage = AsyncLocalStorage::new(&config).await?;
//!
//! // Write a file
//! let mut writer = storage.open_write(Path::new("example.txt")).await?;
//! writer.write_all_bytes(b"Hello, world!").await?;
//! writer.finish().await?;
//!
//! // Read it back
//! let mut reader = storage.open_read(Path::new("example.txt")).await?;
//! let content = reader.read_all().await?;
//! # Ok(())
//! # }
//! ```

// Sync storage
mod local;
mod traits;

pub use local::LocalStorage;
pub use traits::{ObjectMeta, StorageBackend, StorageReader, StorageWriter};

// Async storage
mod async_local;
mod async_traits;

pub use async_local::{AsyncLocalStorage, AsyncLocalReader, AsyncLocalWriter};
pub use async_traits::{
    AsyncObjectMeta, AsyncStorageBackend, AsyncStorageReader, AsyncStorageWriter,
    ByteStream, MultipartUploadBackend,
};

// S3 storage (requires feature flag)
#[cfg(feature = "s3")]
mod s3;
#[cfg(feature = "s3")]
mod retry;

#[cfg(feature = "s3")]
pub use s3::S3Storage;
#[cfg(feature = "s3")]
pub use retry::RetryConfig;
