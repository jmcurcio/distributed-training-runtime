// rust/runtime-core/src/storage/mod.rs

//! Storage abstraction for the distributed training runtime.
//!
//! This module provides traits and implementations for storage backends,
//! allowing the runtime to work with different storage systems (local
//! filesystem, S3, etc.) through a unified interface.
//!
//! # Example
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

mod local;
mod traits;

pub use local::LocalStorage;
pub use traits::{ObjectMeta, StorageBackend, StorageReader, StorageWriter};
