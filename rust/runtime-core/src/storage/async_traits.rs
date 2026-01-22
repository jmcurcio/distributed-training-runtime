// rust/runtime-core/src/storage/async_traits.rs

//! Async storage abstraction traits for the distributed training runtime.
//!
//! This module defines async versions of the storage traits, allowing
//! for non-blocking I/O operations with different storage backends
//! (local filesystem, S3, etc.).

use std::path::Path;
use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;
use tokio::io::{AsyncRead, AsyncSeek, AsyncWrite};

use crate::error::Result;

/// Metadata about a stored object.
#[derive(Debug, Clone)]
pub struct AsyncObjectMeta {
    /// Size of the object in bytes.
    pub size: u64,
    /// Last modification time as Unix timestamp (seconds since epoch).
    pub modified: Option<i64>,
    /// ETag or content hash, if available.
    pub etag: Option<String>,
    /// Whether this object is a directory/prefix.
    pub is_dir: bool,
}

/// A handle for async reading from storage.
///
/// This trait extends `AsyncRead` and `AsyncSeek` with additional methods
/// for efficient random access.
#[async_trait]
pub trait AsyncStorageReader: AsyncRead + AsyncSeek + Send + Sync + Unpin {
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
    async fn read_range(&mut self, start: u64, length: usize) -> Result<Bytes>;

    /// Reads the entire object into memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    async fn read_all(&mut self) -> Result<Bytes>;
}

/// A handle for async writing to storage.
///
/// This trait extends `AsyncWrite` with methods to finalize the write operation
/// and handle streaming writes.
#[async_trait]
pub trait AsyncStorageWriter: AsyncWrite + Send + Sync + Unpin {
    /// Finishes the write operation, ensuring all data is persisted.
    ///
    /// This method must be called to complete the write. After calling
    /// `finish`, the writer should not be used again.
    ///
    /// # Errors
    ///
    /// Returns an error if the finalization fails (e.g., flush fails,
    /// rename fails for atomic writes, multipart upload completion fails).
    async fn finish(self: Box<Self>) -> Result<()>;

    /// Writes all data from a bytes slice.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    async fn write_all_bytes(&mut self, data: &[u8]) -> Result<()>;

    /// Returns the number of bytes written so far.
    fn bytes_written(&self) -> u64;
}

/// A streaming reader for large objects.
///
/// This allows processing objects chunk-by-chunk without loading
/// the entire content into memory.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>;

/// The core async storage backend trait.
///
/// This trait defines the async operations that any storage backend must support.
/// Implementations can use local filesystem, S3, GCS, or any other storage system.
#[async_trait]
pub trait AsyncStorageBackend: Send + Sync {
    /// Checks if an object exists at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the existence check fails (e.g., permission denied).
    async fn exists(&self, path: &Path) -> Result<bool>;

    /// Retrieves metadata for an object.
    ///
    /// # Errors
    ///
    /// Returns an error if the object doesn't exist or metadata cannot be read.
    async fn metadata(&self, path: &Path) -> Result<AsyncObjectMeta>;

    /// Opens an object for async reading.
    ///
    /// # Errors
    ///
    /// Returns an error if the object doesn't exist or cannot be opened.
    async fn open_read(&self, path: &Path) -> Result<Box<dyn AsyncStorageReader>>;

    /// Opens an object for async writing.
    ///
    /// If the object already exists, it will be overwritten.
    /// Parent directories will be created if they don't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the object cannot be created or opened.
    async fn open_write(&self, path: &Path) -> Result<Box<dyn AsyncStorageWriter>>;

    /// Gets a streaming reader for large objects.
    ///
    /// This is more efficient than `open_read` for processing large objects
    /// as it doesn't require random access support.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream cannot be created.
    async fn get_stream(&self, path: &Path) -> Result<ByteStream>;

    /// Gets a streaming reader for a range of bytes.
    ///
    /// # Arguments
    ///
    /// * `path` - The object path.
    /// * `start` - The starting byte offset.
    /// * `end` - The ending byte offset (exclusive).
    ///
    /// # Errors
    ///
    /// Returns an error if the stream cannot be created.
    async fn get_range_stream(&self, path: &Path, start: u64, end: u64) -> Result<ByteStream>;

    /// Deletes an object.
    ///
    /// # Errors
    ///
    /// Returns an error if the object doesn't exist or cannot be deleted.
    async fn delete(&self, path: &Path) -> Result<()>;

    /// Lists objects with the given prefix.
    ///
    /// Returns a list of relative paths from the prefix directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the listing fails.
    async fn list(&self, prefix: &Path) -> Result<Vec<String>>;

    /// Lists objects with pagination support.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix to list under.
    /// * `max_results` - Maximum number of results to return.
    /// * `continuation_token` - Token from previous list call for pagination.
    ///
    /// # Returns
    ///
    /// A tuple of (paths, optional continuation token for next page).
    ///
    /// # Errors
    ///
    /// Returns an error if the listing fails.
    async fn list_paginated(
        &self,
        prefix: &Path,
        max_results: usize,
        continuation_token: Option<&str>,
    ) -> Result<(Vec<String>, Option<String>)>;

    /// Renames/moves an object from one path to another.
    ///
    /// Note: For object stores like S3, this is typically implemented
    /// as a copy followed by delete.
    ///
    /// # Errors
    ///
    /// Returns an error if the source doesn't exist or the rename fails.
    async fn rename(&self, from: &Path, to: &Path) -> Result<()>;

    /// Copies an object from one path to another.
    ///
    /// # Errors
    ///
    /// Returns an error if the source doesn't exist or the copy fails.
    async fn copy(&self, from: &Path, to: &Path) -> Result<()>;

    /// Creates a directory and all parent directories.
    ///
    /// For object stores, this may be a no-op since directories are virtual.
    ///
    /// # Errors
    ///
    /// Returns an error if directory creation fails.
    async fn create_dir_all(&self, path: &Path) -> Result<()>;

    /// Returns the backend type name (e.g., "local", "s3").
    fn backend_type(&self) -> &'static str;

    /// Returns true if this backend supports atomic renames.
    ///
    /// Local filesystem typically does, S3 does not.
    fn supports_atomic_rename(&self) -> bool;

    /// Returns true if this backend supports multipart uploads.
    fn supports_multipart(&self) -> bool {
        false
    }
}

/// Extension trait for async storage backends with multipart upload support.
#[async_trait]
pub trait MultipartUploadBackend: AsyncStorageBackend {
    /// Initiates a multipart upload.
    ///
    /// # Returns
    ///
    /// An upload ID to use for subsequent part uploads.
    ///
    /// # Errors
    ///
    /// Returns an error if the multipart upload cannot be initiated.
    async fn initiate_multipart(&self, path: &Path) -> Result<String>;

    /// Uploads a part of a multipart upload.
    ///
    /// # Arguments
    ///
    /// * `path` - The object path.
    /// * `upload_id` - The upload ID from `initiate_multipart`.
    /// * `part_number` - The part number (1-indexed).
    /// * `data` - The part data.
    ///
    /// # Returns
    ///
    /// An ETag for the uploaded part.
    ///
    /// # Errors
    ///
    /// Returns an error if the part upload fails.
    async fn upload_part(
        &self,
        path: &Path,
        upload_id: &str,
        part_number: u32,
        data: Bytes,
    ) -> Result<String>;

    /// Completes a multipart upload.
    ///
    /// # Arguments
    ///
    /// * `path` - The object path.
    /// * `upload_id` - The upload ID from `initiate_multipart`.
    /// * `parts` - A list of (part_number, etag) tuples.
    ///
    /// # Errors
    ///
    /// Returns an error if the completion fails.
    async fn complete_multipart(
        &self,
        path: &Path,
        upload_id: &str,
        parts: Vec<(u32, String)>,
    ) -> Result<()>;

    /// Aborts a multipart upload.
    ///
    /// # Errors
    ///
    /// Returns an error if the abort fails.
    async fn abort_multipart(&self, path: &Path, upload_id: &str) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_object_meta_debug() {
        let meta = AsyncObjectMeta {
            size: 1024,
            modified: Some(1234567890),
            etag: Some("abc123".to_string()),
            is_dir: false,
        };
        let debug_str = format!("{:?}", meta);
        assert!(debug_str.contains("1024"));
        assert!(debug_str.contains("abc123"));
    }

    #[test]
    fn test_async_object_meta_clone() {
        let meta = AsyncObjectMeta {
            size: 2048,
            modified: None,
            etag: None,
            is_dir: true,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.size, 2048);
        assert!(cloned.is_dir);
    }
}
