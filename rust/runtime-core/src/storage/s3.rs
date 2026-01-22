// rust/runtime-core/src/storage/s3.rs

//! S3-compatible storage backend using the object_store crate.
//!
//! This module provides an async storage backend for S3-compatible object stores
//! including AWS S3, MinIO, LocalStack, and other compatible services.

use std::io::SeekFrom;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{Stream, StreamExt, TryStreamExt};
use object_store::aws::{AmazonS3, AmazonS3Builder};
use object_store::{ObjectStore, PutPayload, WriteMultipart};
use tokio::io::{AsyncRead, AsyncSeek, AsyncWrite, ReadBuf};

use super::async_traits::{
    AsyncObjectMeta, AsyncStorageBackend, AsyncStorageReader, AsyncStorageWriter,
    ByteStream, MultipartUploadBackend,
};
use super::retry::{RetryConfig, RetryResult, retry_async};
use crate::config::S3Config;
use crate::error::{Result, RuntimeError};

/// S3-compatible storage backend.
///
/// This backend uses the `object_store` crate to provide a unified interface
/// for S3-compatible object stores.
pub struct S3Storage {
    /// The underlying object store client.
    store: Arc<AmazonS3>,
    /// Base prefix for all keys.
    base_prefix: String,
    /// Retry configuration.
    retry_config: RetryConfig,
    /// Multipart upload threshold.
    multipart_threshold: u64,
    /// Multipart chunk size.
    multipart_chunk_size: usize,
}

impl S3Storage {
    /// Creates a new S3Storage from configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the S3 client cannot be configured.
    pub fn new(config: &S3Config, base_prefix: impl Into<String>) -> Result<Self> {
        let mut builder = AmazonS3Builder::new()
            .with_bucket_name(&config.bucket)
            .with_region(&config.region);

        // Set endpoint if provided (for MinIO, LocalStack, etc.)
        if let Some(endpoint) = &config.endpoint {
            builder = builder.with_endpoint(endpoint);
        }

        // Set credentials if provided
        if let Some(access_key) = &config.access_key_id {
            builder = builder.with_access_key_id(access_key);
        }
        if let Some(secret_key) = &config.secret_access_key {
            builder = builder.with_secret_access_key(secret_key);
        }
        if let Some(token) = &config.session_token {
            builder = builder.with_token(token);
        }

        // Configure path style (required for MinIO)
        if config.force_path_style {
            builder = builder.with_virtual_hosted_style_request(false);
        }

        // Allow HTTP if configured
        if config.allow_http {
            builder = builder.with_allow_http(true);
        }

        // Note: Timeout configuration is handled by the underlying HTTP client.
        // object_store 0.11+ uses default timeouts; custom timeouts require
        // building a custom client with reqwest::ClientBuilder.
        let _ = config.connect_timeout_ms;  // Acknowledge config (future use)
        let _ = config.request_timeout_ms;  // Acknowledge config (future use)

        let store = builder.build().map_err(|e| {
            RuntimeError::config_with_source("failed to build S3 client", e)
        })?;

        Ok(Self {
            store: Arc::new(store),
            base_prefix: base_prefix.into(),
            retry_config: RetryConfig::from(config),
            multipart_threshold: config.multipart_threshold,
            multipart_chunk_size: config.multipart_chunk_size,
        })
    }

    /// Resolves a path to an object_store path with the base prefix.
    fn resolve_path(&self, path: &Path) -> object_store::path::Path {
        let path_str = path.to_string_lossy();
        let full_path = if self.base_prefix.is_empty() {
            path_str.to_string()
        } else {
            format!("{}/{}", self.base_prefix.trim_end_matches('/'), path_str.trim_start_matches('/'))
        };
        object_store::path::Path::from(full_path)
    }

    /// Converts an object_store error to a RuntimeError.
    fn convert_error(path: &Path, message: &str, error: object_store::Error) -> RuntimeError {
        RuntimeError::Storage {
            path: path.to_path_buf(),
            message: format!("{}: {}", message, error),
            source: None,
        }
    }

    /// Determines if an error is retryable.
    fn is_retryable_error(error: &object_store::Error) -> bool {
        matches!(
            error,
            object_store::Error::Generic { .. }
            | object_store::Error::NotSupported { .. }
        ) || error.to_string().contains("timeout")
           || error.to_string().contains("connection")
           || error.to_string().contains("503")
           || error.to_string().contains("500")
    }
}

#[async_trait]
impl AsyncStorageBackend for S3Storage {
    async fn exists(&self, path: &Path) -> Result<bool> {
        let object_path = self.resolve_path(path);
        let path_clone = path.to_path_buf();

        let result = retry_async(&self.retry_config, || {
            let store = self.store.clone();
            let obj_path = object_path.clone();
            async move {
                match store.head(&obj_path).await {
                    Ok(_) => RetryResult::Ok(true),
                    Err(object_store::Error::NotFound { .. }) => RetryResult::Ok(false),
                    Err(e) if Self::is_retryable_error(&e) => RetryResult::Retry(e),
                    Err(e) => RetryResult::Fail(e),
                }
            }
        }).await;

        result.map_err(|e| Self::convert_error(&path_clone, "failed to check existence", e))
    }

    async fn metadata(&self, path: &Path) -> Result<AsyncObjectMeta> {
        let object_path = self.resolve_path(path);
        let path_clone = path.to_path_buf();

        let result = retry_async(&self.retry_config, || {
            let store = self.store.clone();
            let obj_path = object_path.clone();
            async move {
                match store.head(&obj_path).await {
                    Ok(meta) => RetryResult::Ok(meta),
                    Err(e) if Self::is_retryable_error(&e) => RetryResult::Retry(e),
                    Err(e) => RetryResult::Fail(e),
                }
            }
        }).await;

        let meta = result.map_err(|e| Self::convert_error(&path_clone, "failed to get metadata", e))?;

        Ok(AsyncObjectMeta {
            size: meta.size as u64,
            modified: Some(meta.last_modified.timestamp()),
            etag: meta.e_tag,
            is_dir: false, // S3 doesn't have real directories
        })
    }

    async fn open_read(&self, path: &Path) -> Result<Box<dyn AsyncStorageReader>> {
        let object_path = self.resolve_path(path);
        let path_clone = path.to_path_buf();

        // First get metadata to know the size
        let meta = retry_async(&self.retry_config, || {
            let store = self.store.clone();
            let obj_path = object_path.clone();
            async move {
                match store.head(&obj_path).await {
                    Ok(meta) => RetryResult::Ok(meta),
                    Err(e) if Self::is_retryable_error(&e) => RetryResult::Retry(e),
                    Err(e) => RetryResult::Fail(e),
                }
            }
        }).await.map_err(|e| Self::convert_error(&path_clone, "failed to get metadata", e))?;

        // Get the full object content for now (can be optimized for streaming later)
        let data = retry_async(&self.retry_config, || {
            let store = self.store.clone();
            let obj_path = object_path.clone();
            async move {
                match store.get(&obj_path).await {
                    Ok(result) => match result.bytes().await {
                        Ok(bytes) => RetryResult::Ok(bytes),
                        Err(e) if Self::is_retryable_error(&e) => RetryResult::Retry(e),
                        Err(e) => RetryResult::Fail(e),
                    },
                    Err(e) if Self::is_retryable_error(&e) => RetryResult::Retry(e),
                    Err(e) => RetryResult::Fail(e),
                }
            }
        }).await.map_err(|e| Self::convert_error(&path_clone, "failed to read object", e))?;

        Ok(Box::new(S3Reader::new(data, meta.size as u64)))
    }

    async fn open_write(&self, path: &Path) -> Result<Box<dyn AsyncStorageWriter>> {
        let object_path = self.resolve_path(path);

        Ok(Box::new(S3Writer::new(
            self.store.clone(),
            object_path,
            self.multipart_threshold,
            self.multipart_chunk_size,
        )))
    }

    async fn get_stream(&self, path: &Path) -> Result<ByteStream> {
        let object_path = self.resolve_path(path);
        let path_clone = path.to_path_buf();

        let result = retry_async(&self.retry_config, || {
            let store = self.store.clone();
            let obj_path = object_path.clone();
            async move {
                match store.get(&obj_path).await {
                    Ok(result) => RetryResult::Ok(result),
                    Err(e) if Self::is_retryable_error(&e) => RetryResult::Retry(e),
                    Err(e) => RetryResult::Fail(e),
                }
            }
        }).await.map_err(|e| Self::convert_error(&path_clone, "failed to get stream", e))?;

        let stream = result.into_stream().map(|chunk| {
            chunk.map_err(|e| RuntimeError::Storage {
                path: PathBuf::new(),
                message: format!("stream error: {}", e),
                source: None,
            })
        });

        Ok(Box::pin(stream))
    }

    async fn get_range_stream(&self, path: &Path, start: u64, end: u64) -> Result<ByteStream> {
        let object_path = self.resolve_path(path);
        let path_clone = path.to_path_buf();
        let range = std::ops::Range {
            start: start as usize,
            end: end as usize,
        };

        let result = retry_async(&self.retry_config, || {
            let store = self.store.clone();
            let obj_path = object_path.clone();
            let r = range.clone();
            async move {
                match store.get_range(&obj_path, r).await {
                    Ok(bytes) => RetryResult::Ok(bytes),
                    Err(e) if Self::is_retryable_error(&e) => RetryResult::Retry(e),
                    Err(e) => RetryResult::Fail(e),
                }
            }
        }).await.map_err(|e| Self::convert_error(&path_clone, "failed to get range", e))?;

        // Convert the bytes to a single-item stream
        let stream = futures::stream::once(async move { Ok(result) });
        Ok(Box::pin(stream))
    }

    async fn delete(&self, path: &Path) -> Result<()> {
        let object_path = self.resolve_path(path);
        let path_clone = path.to_path_buf();

        retry_async(&self.retry_config, || {
            let store = self.store.clone();
            let obj_path = object_path.clone();
            async move {
                match store.delete(&obj_path).await {
                    Ok(()) => RetryResult::Ok(()),
                    Err(e) if Self::is_retryable_error(&e) => RetryResult::Retry(e),
                    Err(e) => RetryResult::Fail(e),
                }
            }
        }).await.map_err(|e| Self::convert_error(&path_clone, "failed to delete object", e))
    }

    async fn list(&self, prefix: &Path) -> Result<Vec<String>> {
        let (entries, _) = self.list_paginated(prefix, 10000, None).await?;
        Ok(entries)
    }

    async fn list_paginated(
        &self,
        prefix: &Path,
        max_results: usize,
        continuation_token: Option<&str>,
    ) -> Result<(Vec<String>, Option<String>)> {
        let object_prefix = self.resolve_path(prefix);
        let prefix_clone = prefix.to_path_buf();

        // object_store doesn't directly support pagination tokens, so we'll
        // collect results and handle pagination manually
        let prefix_str = object_prefix.to_string();
        let prefix_len = if prefix_str.is_empty() { 0 } else { prefix_str.len() + 1 };

        let result = retry_async(&self.retry_config, || {
            let store = self.store.clone();
            let obj_prefix = object_prefix.clone();
            async move {
                let stream = store.list(Some(&obj_prefix));
                match stream.try_collect::<Vec<_>>().await {
                    Ok(entries) => RetryResult::Ok(entries),
                    Err(e) if Self::is_retryable_error(&e) => RetryResult::Retry(e),
                    Err(e) => RetryResult::Fail(e),
                }
            }
        }).await.map_err(|e| Self::convert_error(&prefix_clone, "failed to list objects", e))?;

        // Parse continuation token as offset
        let offset: usize = continuation_token
            .and_then(|t| t.parse().ok())
            .unwrap_or(0);

        // Extract names and paginate
        let names: Vec<String> = result
            .into_iter()
            .map(|m| {
                let path = m.location.to_string();
                if path.len() > prefix_len {
                    path[prefix_len..].to_string()
                } else {
                    path
                }
            })
            .collect();

        let total = names.len();
        let end = (offset + max_results).min(total);
        let page = names[offset..end].to_vec();

        let next_token = if end < total {
            Some(end.to_string())
        } else {
            None
        };

        Ok((page, next_token))
    }

    async fn rename(&self, from: &Path, to: &Path) -> Result<()> {
        // S3 doesn't support rename, so we copy then delete
        self.copy(from, to).await?;
        self.delete(from).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> Result<()> {
        let from_path = self.resolve_path(from);
        let to_path = self.resolve_path(to);
        let from_clone = from.to_path_buf();

        retry_async(&self.retry_config, || {
            let store = self.store.clone();
            let src = from_path.clone();
            let dst = to_path.clone();
            async move {
                match store.copy(&src, &dst).await {
                    Ok(()) => RetryResult::Ok(()),
                    Err(e) if Self::is_retryable_error(&e) => RetryResult::Retry(e),
                    Err(e) => RetryResult::Fail(e),
                }
            }
        }).await.map_err(|e| Self::convert_error(&from_clone, "failed to copy object", e))
    }

    async fn create_dir_all(&self, _path: &Path) -> Result<()> {
        // S3 doesn't have real directories, this is a no-op
        Ok(())
    }

    fn backend_type(&self) -> &'static str {
        "s3"
    }

    fn supports_atomic_rename(&self) -> bool {
        false // S3 uses copy+delete
    }

    fn supports_multipart(&self) -> bool {
        true
    }
}

#[async_trait]
impl MultipartUploadBackend for S3Storage {
    async fn initiate_multipart(&self, path: &Path) -> Result<String> {
        let object_path = self.resolve_path(path);
        let path_clone = path.to_path_buf();

        let result = retry_async(&self.retry_config, || {
            let store = self.store.clone();
            let obj_path = object_path.clone();
            async move {
                match store.put_multipart(&obj_path).await {
                    Ok(upload) => RetryResult::Ok(upload),
                    Err(e) if Self::is_retryable_error(&e) => RetryResult::Retry(e),
                    Err(e) => RetryResult::Fail(e),
                }
            }
        }).await.map_err(|e| Self::convert_error(&path_clone, "failed to initiate multipart upload", e))?;

        // object_store doesn't expose upload IDs directly, so we use the path as identifier
        // The actual upload is tracked internally by object_store
        Ok(object_path.to_string())
    }

    async fn upload_part(
        &self,
        _path: &Path,
        _upload_id: &str,
        _part_number: u32,
        _data: Bytes,
    ) -> Result<String> {
        // object_store handles multipart uploads internally through WriteMultipart
        // This method is provided for compatibility but the actual implementation
        // uses S3Writer which handles multipart internally
        Err(RuntimeError::config(
            "Direct part upload not supported; use open_write() instead which handles multipart automatically"
        ))
    }

    async fn complete_multipart(
        &self,
        _path: &Path,
        _upload_id: &str,
        _parts: Vec<(u32, String)>,
    ) -> Result<()> {
        // object_store handles completion internally through WriteMultipart::finish()
        Ok(())
    }

    async fn abort_multipart(&self, path: &Path, _upload_id: &str) -> Result<()> {
        let object_path = self.resolve_path(path);
        let path_clone = path.to_path_buf();

        // Try to delete any partial upload artifacts
        retry_async(&self.retry_config, || {
            let store = self.store.clone();
            let obj_path = object_path.clone();
            async move {
                match store.delete(&obj_path).await {
                    Ok(()) => RetryResult::Ok(()),
                    Err(object_store::Error::NotFound { .. }) => RetryResult::Ok(()),
                    Err(e) if Self::is_retryable_error(&e) => RetryResult::Retry(e),
                    Err(e) => RetryResult::Fail(e),
                }
            }
        }).await.map_err(|e| Self::convert_error(&path_clone, "failed to abort multipart upload", e))
    }
}

/// S3 reader that wraps downloaded content.
pub struct S3Reader {
    data: Bytes,
    position: u64,
    size: u64,
}

impl S3Reader {
    fn new(data: Bytes, size: u64) -> Self {
        Self {
            data,
            position: 0,
            size,
        }
    }
}

impl AsyncRead for S3Reader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let pos = self.position as usize;
        let remaining = &self.data[pos..];
        let to_read = buf.remaining().min(remaining.len());

        if to_read > 0 {
            buf.put_slice(&remaining[..to_read]);
            self.position += to_read as u64;
        }

        Poll::Ready(Ok(()))
    }
}

impl AsyncSeek for S3Reader {
    fn start_seek(mut self: Pin<&mut Self>, position: SeekFrom) -> std::io::Result<()> {
        let new_pos = match position {
            SeekFrom::Start(pos) => pos as i64,
            SeekFrom::End(offset) => self.size as i64 + offset,
            SeekFrom::Current(offset) => self.position as i64 + offset,
        };

        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek to negative position",
            ));
        }

        self.position = new_pos as u64;
        Ok(())
    }

    fn poll_complete(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<u64>> {
        Poll::Ready(Ok(self.position))
    }
}

#[async_trait]
impl AsyncStorageReader for S3Reader {
    fn size(&self) -> u64 {
        self.size
    }

    async fn read_range(&mut self, start: u64, length: usize) -> Result<Bytes> {
        let end = (start as usize + length).min(self.data.len());
        Ok(self.data.slice(start as usize..end))
    }

    async fn read_all(&mut self) -> Result<Bytes> {
        self.position = 0;
        Ok(self.data.clone())
    }
}

/// S3 writer that buffers data and uploads on finish.
pub struct S3Writer {
    store: Arc<AmazonS3>,
    path: object_store::path::Path,
    buffer: Vec<u8>,
    bytes_written: AtomicU64,
    multipart_threshold: u64,
    multipart_chunk_size: usize,
}

impl S3Writer {
    fn new(
        store: Arc<AmazonS3>,
        path: object_store::path::Path,
        multipart_threshold: u64,
        multipart_chunk_size: usize,
    ) -> Self {
        Self {
            store,
            path,
            buffer: Vec::new(),
            bytes_written: AtomicU64::new(0),
            multipart_threshold,
            multipart_chunk_size,
        }
    }
}

impl AsyncWrite for S3Writer {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        self.buffer.extend_from_slice(buf);
        self.bytes_written.fetch_add(buf.len() as u64, Ordering::Relaxed);
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

#[async_trait]
impl AsyncStorageWriter for S3Writer {
    async fn finish(self: Box<Self>) -> Result<()> {
        let data = Bytes::from(self.buffer);
        let size = data.len() as u64;

        if size >= self.multipart_threshold {
            // Use multipart upload for large files
            let mut upload = self.store.put_multipart(&self.path).await.map_err(|e| {
                RuntimeError::Storage {
                    path: PathBuf::from(self.path.to_string()),
                    message: format!("failed to initiate multipart upload: {}", e),
                    source: None,
                }
            })?;

            let mut offset = 0;
            while offset < data.len() {
                let end = (offset + self.multipart_chunk_size).min(data.len());
                let chunk = data.slice(offset..end);
                upload.put_part(PutPayload::from_bytes(chunk)).await.map_err(|e| {
                    RuntimeError::Storage {
                        path: PathBuf::from(self.path.to_string()),
                        message: format!("failed to upload part: {}", e),
                        source: None,
                    }
                })?;
                offset = end;
            }

            upload.complete().await.map_err(|e| {
                RuntimeError::Storage {
                    path: PathBuf::from(self.path.to_string()),
                    message: format!("failed to complete multipart upload: {}", e),
                    source: None,
                }
            })?;
        } else {
            // Use single PUT for small files
            self.store.put(&self.path, PutPayload::from_bytes(data)).await.map_err(|e| {
                RuntimeError::Storage {
                    path: PathBuf::from(self.path.to_string()),
                    message: format!("failed to put object: {}", e),
                    source: None,
                }
            })?;
        }

        Ok(())
    }

    async fn write_all_bytes(&mut self, data: &[u8]) -> Result<()> {
        self.buffer.extend_from_slice(data);
        self.bytes_written.fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn bytes_written(&self) -> u64 {
        self.bytes_written.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_path() {
        let config = S3Config {
            bucket: "test-bucket".to_string(),
            allow_http: true,
            ..Default::default()
        };
        let storage = S3Storage::new(&config, "prefix").unwrap();

        let resolved = storage.resolve_path(Path::new("file.txt"));
        assert_eq!(resolved.to_string(), "prefix/file.txt");

        let resolved = storage.resolve_path(Path::new("/file.txt"));
        assert_eq!(resolved.to_string(), "prefix/file.txt");
    }

    #[test]
    fn test_resolve_path_empty_prefix() {
        let config = S3Config {
            bucket: "test-bucket".to_string(),
            allow_http: true,
            ..Default::default()
        };
        let storage = S3Storage::new(&config, "").unwrap();

        let resolved = storage.resolve_path(Path::new("file.txt"));
        assert_eq!(resolved.to_string(), "file.txt");
    }

    #[test]
    fn test_backend_type() {
        let config = S3Config {
            bucket: "test-bucket".to_string(),
            allow_http: true,
            ..Default::default()
        };
        let storage = S3Storage::new(&config, "").unwrap();
        assert_eq!(storage.backend_type(), "s3");
    }

    #[test]
    fn test_supports_atomic_rename() {
        let config = S3Config {
            bucket: "test-bucket".to_string(),
            allow_http: true,
            ..Default::default()
        };
        let storage = S3Storage::new(&config, "").unwrap();
        assert!(!storage.supports_atomic_rename());
    }

    #[test]
    fn test_supports_multipart() {
        let config = S3Config {
            bucket: "test-bucket".to_string(),
            allow_http: true,
            ..Default::default()
        };
        let storage = S3Storage::new(&config, "").unwrap();
        assert!(storage.supports_multipart());
    }

    #[test]
    fn test_s3_reader() {
        let data = Bytes::from("hello world");
        let reader = S3Reader::new(data.clone(), 11);

        assert_eq!(reader.size(), 11);
    }

    // Note: Full integration tests require a running S3-compatible server (MinIO)
    // and should be marked with #[ignore] to skip in regular test runs.
    //
    // To run integration tests:
    // cargo test -p runtime-core --features s3 -- --ignored s3
}
