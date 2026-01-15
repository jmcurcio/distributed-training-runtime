// rust/runtime-core/src/storage/traits.rs

//! Storage abstraction traits for the distributed training runtime.
//!
//! This module defines the core traits for storage backends, allowing
//! different implementations (local filesystem, S3, etc.) to be used
//! interchangeably.

use std::io::{Read, Seek, Write};
use std::path::Path;

use crate::error::Result;

/// Metadata about a stored object.
#[derive(Debug, Clone)]
pub struct ObjectMeta {
    /// Size of the object in bytes.
    pub size: u64,
    /// Last modification time, if available.
    pub modified: Option<std::time::SystemTime>,
    /// Whether this object is a directory.
    pub is_dir: bool,
}

/// A handle for reading from storage.
///
/// This trait extends `Read` and `Seek` with additional methods for
/// efficient random access.
pub trait StorageReader: Read + Seek + Send {
    /// Returns the total size of the object in bytes.
    fn size(&self) -> u64;

    /// Reads a range of bytes from the object.
    ///
    /// # Arguments
    ///
    /// * `start` - The byte offset to start reading from.
    /// * `length` - The number of bytes to read.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails or the range is out of bounds.
    fn read_range(&mut self, start: u64, length: usize) -> Result<Vec<u8>>;
}

/// A handle for writing to storage.
///
/// This trait extends `Write` with a method to finalize the write operation.
pub trait StorageWriter: Write + Send {
    /// Finishes the write operation, ensuring all data is persisted.
    ///
    /// This method must be called to complete the write. After calling
    /// `finish`, the writer should not be used again.
    ///
    /// # Errors
    ///
    /// Returns an error if the finalization fails (e.g., flush fails,
    /// rename fails for atomic writes).
    fn finish(self: Box<Self>) -> Result<()>;
}

/// The core storage backend trait.
///
/// This trait defines the operations that any storage backend must support.
/// Implementations can use local filesystem, S3, GCS, or any other storage
/// system.
///
/// # Object Safety
///
/// This trait is object-safe and can be used with `Box<dyn StorageBackend>`.
pub trait StorageBackend: Send + Sync {
    /// Checks if an object exists at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the existence check fails (e.g., permission denied).
    fn exists(&self, path: &Path) -> Result<bool>;

    /// Retrieves metadata for an object.
    ///
    /// # Errors
    ///
    /// Returns an error if the object doesn't exist or metadata cannot be read.
    fn metadata(&self, path: &Path) -> Result<ObjectMeta>;

    /// Opens an object for reading.
    ///
    /// # Errors
    ///
    /// Returns an error if the object doesn't exist or cannot be opened.
    fn open_read(&self, path: &Path) -> Result<Box<dyn StorageReader>>;

    /// Opens an object for writing.
    ///
    /// If the object already exists, it will be overwritten.
    /// Parent directories will be created if they don't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the object cannot be created or opened.
    fn open_write(&self, path: &Path) -> Result<Box<dyn StorageWriter>>;

    /// Deletes an object.
    ///
    /// # Errors
    ///
    /// Returns an error if the object doesn't exist or cannot be deleted.
    fn delete(&self, path: &Path) -> Result<()>;

    /// Lists objects with the given prefix.
    ///
    /// Returns a list of relative paths from the prefix directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the listing fails.
    fn list(&self, prefix: &Path) -> Result<Vec<String>>;

    /// Renames an object from one path to another.
    ///
    /// # Errors
    ///
    /// Returns an error if the source doesn't exist or the rename fails.
    fn rename(&self, from: &Path, to: &Path) -> Result<()>;

    /// Creates a directory and all parent directories.
    ///
    /// # Errors
    ///
    /// Returns an error if directory creation fails.
    fn create_dir_all(&self, path: &Path) -> Result<()>;
}
