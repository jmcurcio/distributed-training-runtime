// rust/runtime-core/src/storage/async_local.rs

//! Async local filesystem storage backend implementation.
//!
//! This module provides an async storage backend that uses the local filesystem
//! via tokio's async filesystem operations.

use std::io::SeekFrom;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};

use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;
use tokio::fs::{self, File, OpenOptions};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt, AsyncWrite, AsyncWriteExt, BufReader, BufWriter, ReadBuf};

use super::async_traits::{AsyncObjectMeta, AsyncStorageBackend, AsyncStorageReader, AsyncStorageWriter, ByteStream};
use crate::config::StorageConfig;
use crate::error::{Result, RuntimeError};

/// Async local filesystem storage backend.
///
/// This backend stores objects as files on the local filesystem using
/// tokio's async filesystem operations.
pub struct AsyncLocalStorage {
    /// Base path for all storage operations.
    base_path: PathBuf,
    /// Buffer size for buffered I/O operations.
    buffer_size: usize,
}

impl AsyncLocalStorage {
    /// Creates a new `AsyncLocalStorage` instance from configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the base path cannot be created.
    pub async fn new(config: &StorageConfig) -> Result<Self> {
        let base_path = config.base_path.clone();

        // Create base directory if it doesn't exist
        if !base_path.exists() {
            fs::create_dir_all(&base_path).await.map_err(|e| {
                RuntimeError::storage_with_source(&base_path, "failed to create base directory", e)
            })?;
        }

        Ok(Self {
            base_path,
            buffer_size: config.buffer_size,
        })
    }

    /// Creates a new `AsyncLocalStorage` instance synchronously.
    ///
    /// This is useful when you need to create storage outside an async context.
    ///
    /// # Errors
    ///
    /// Returns an error if the base path cannot be created.
    pub fn new_sync(config: &StorageConfig) -> Result<Self> {
        let base_path = config.base_path.clone();

        // Create base directory if it doesn't exist
        if !base_path.exists() {
            std::fs::create_dir_all(&base_path).map_err(|e| {
                RuntimeError::storage_with_source(&base_path, "failed to create base directory", e)
            })?;
        }

        Ok(Self {
            base_path,
            buffer_size: config.buffer_size,
        })
    }

    /// Resolves a path relative to the base path.
    fn resolve_path(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.base_path.join(path)
        }
    }
}

#[async_trait]
impl AsyncStorageBackend for AsyncLocalStorage {
    async fn exists(&self, path: &Path) -> Result<bool> {
        let full_path = self.resolve_path(path);
        Ok(fs::try_exists(&full_path).await.unwrap_or(false))
    }

    async fn metadata(&self, path: &Path) -> Result<AsyncObjectMeta> {
        let full_path = self.resolve_path(path);
        let meta = fs::metadata(&full_path).await.map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to read metadata", e)
        })?;

        let modified = meta.modified().ok().map(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0)
        });

        Ok(AsyncObjectMeta {
            size: meta.len(),
            modified,
            etag: None,
            is_dir: meta.is_dir(),
        })
    }

    async fn open_read(&self, path: &Path) -> Result<Box<dyn AsyncStorageReader>> {
        let full_path = self.resolve_path(path);
        let file = File::open(&full_path).await.map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to open file", e)
        })?;

        let meta = file.metadata().await.map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to read file metadata", e)
        })?;
        let size = meta.len();

        Ok(Box::new(AsyncLocalReader::new(file, size, self.buffer_size)))
    }

    async fn open_write(&self, path: &Path) -> Result<Box<dyn AsyncStorageWriter>> {
        let full_path = self.resolve_path(path);

        // Create parent directories if needed
        if let Some(parent) = full_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).await.map_err(|e| {
                    RuntimeError::storage_with_source(
                        parent,
                        "failed to create parent directories",
                        e,
                    )
                })?;
            }
        }

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&full_path)
            .await
            .map_err(|e| {
                RuntimeError::storage_with_source(&full_path, "failed to create file", e)
            })?;

        Ok(Box::new(AsyncLocalWriter::new(file, self.buffer_size)))
    }

    async fn get_stream(&self, path: &Path) -> Result<ByteStream> {
        let full_path = self.resolve_path(path);
        let file = File::open(&full_path).await.map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to open file", e)
        })?;

        let reader = BufReader::with_capacity(self.buffer_size, file);
        Ok(Box::pin(FileByteStream::new(reader, self.buffer_size)))
    }

    async fn get_range_stream(&self, path: &Path, start: u64, end: u64) -> Result<ByteStream> {
        let full_path = self.resolve_path(path);
        let mut file = File::open(&full_path).await.map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to open file", e)
        })?;

        // Seek to start position
        file.seek(SeekFrom::Start(start)).await.map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to seek to start position", e)
        })?;

        let reader = BufReader::with_capacity(self.buffer_size, file);
        let remaining = end.saturating_sub(start);
        Ok(Box::pin(FileByteStream::with_limit(reader, self.buffer_size, remaining)))
    }

    async fn delete(&self, path: &Path) -> Result<()> {
        let full_path = self.resolve_path(path);

        let meta = fs::metadata(&full_path).await.map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to read metadata", e)
        })?;

        if meta.is_dir() {
            fs::remove_dir_all(&full_path).await.map_err(|e| {
                RuntimeError::storage_with_source(&full_path, "failed to delete directory", e)
            })
        } else {
            fs::remove_file(&full_path).await.map_err(|e| {
                RuntimeError::storage_with_source(&full_path, "failed to delete file", e)
            })
        }
    }

    async fn list(&self, prefix: &Path) -> Result<Vec<String>> {
        let full_path = self.resolve_path(prefix);

        if !full_path.exists() {
            return Ok(Vec::new());
        }

        let meta = fs::metadata(&full_path).await.map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to read metadata", e)
        })?;

        if !meta.is_dir() {
            return Err(RuntimeError::storage(&full_path, "path is not a directory"));
        }

        let mut entries = Vec::new();
        let mut dir = fs::read_dir(&full_path).await.map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to read directory", e)
        })?;

        while let Some(entry) = dir.next_entry().await.map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to read directory entry", e)
        })? {
            if let Some(name) = entry.file_name().to_str() {
                entries.push(name.to_string());
            }
        }

        entries.sort();
        Ok(entries)
    }

    async fn list_paginated(
        &self,
        prefix: &Path,
        max_results: usize,
        continuation_token: Option<&str>,
    ) -> Result<(Vec<String>, Option<String>)> {
        // For local filesystem, we do simple pagination based on offset
        let all_entries = self.list(prefix).await?;

        let start_offset: usize = continuation_token
            .and_then(|t| t.parse().ok())
            .unwrap_or(0);

        let end_offset = (start_offset + max_results).min(all_entries.len());
        let page: Vec<String> = all_entries[start_offset..end_offset].to_vec();

        let next_token = if end_offset < all_entries.len() {
            Some(end_offset.to_string())
        } else {
            None
        };

        Ok((page, next_token))
    }

    async fn rename(&self, from: &Path, to: &Path) -> Result<()> {
        let from_path = self.resolve_path(from);
        let to_path = self.resolve_path(to);

        // Create parent directories for destination if needed
        if let Some(parent) = to_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).await.map_err(|e| {
                    RuntimeError::storage_with_source(
                        parent,
                        "failed to create parent directories",
                        e,
                    )
                })?;
            }
        }

        fs::rename(&from_path, &to_path).await.map_err(|e| {
            RuntimeError::storage_with_source(
                &from_path,
                format!("failed to rename to {}", to_path.display()),
                e,
            )
        })
    }

    async fn copy(&self, from: &Path, to: &Path) -> Result<()> {
        let from_path = self.resolve_path(from);
        let to_path = self.resolve_path(to);

        // Create parent directories for destination if needed
        if let Some(parent) = to_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).await.map_err(|e| {
                    RuntimeError::storage_with_source(
                        parent,
                        "failed to create parent directories",
                        e,
                    )
                })?;
            }
        }

        fs::copy(&from_path, &to_path).await.map_err(|e| {
            RuntimeError::storage_with_source(
                &from_path,
                format!("failed to copy to {}", to_path.display()),
                e,
            )
        })?;

        Ok(())
    }

    async fn create_dir_all(&self, path: &Path) -> Result<()> {
        let full_path = self.resolve_path(path);
        fs::create_dir_all(&full_path).await.map_err(|e| {
            RuntimeError::storage_with_source(&full_path, "failed to create directories", e)
        })
    }

    fn backend_type(&self) -> &'static str {
        "local"
    }

    fn supports_atomic_rename(&self) -> bool {
        true
    }
}

/// Async buffered file reader for local storage.
pub struct AsyncLocalReader {
    reader: BufReader<File>,
    size: u64,
}

impl AsyncLocalReader {
    fn new(file: File, size: u64, buffer_size: usize) -> Self {
        Self {
            reader: BufReader::with_capacity(buffer_size, file),
            size,
        }
    }
}

impl AsyncRead for AsyncLocalReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.reader).poll_read(cx, buf)
    }
}

impl AsyncSeek for AsyncLocalReader {
    fn start_seek(mut self: Pin<&mut Self>, position: SeekFrom) -> std::io::Result<()> {
        Pin::new(&mut self.reader).start_seek(position)
    }

    fn poll_complete(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<u64>> {
        Pin::new(&mut self.reader).poll_complete(cx)
    }
}

#[async_trait]
impl AsyncStorageReader for AsyncLocalReader {
    fn size(&self) -> u64 {
        self.size
    }

    async fn read_range(&mut self, start: u64, length: usize) -> Result<Bytes> {
        self.reader.seek(SeekFrom::Start(start)).await.map_err(|e| {
            RuntimeError::Storage {
                path: PathBuf::from("<file>"),
                message: format!("failed to seek to position {start}"),
                source: Some(e),
            }
        })?;

        let mut buf = vec![0u8; length];
        self.reader.read_exact(&mut buf).await.map_err(|e| {
            RuntimeError::Storage {
                path: PathBuf::from("<file>"),
                message: format!("failed to read {length} bytes at position {start}"),
                source: Some(e),
            }
        })?;

        Ok(Bytes::from(buf))
    }

    async fn read_all(&mut self) -> Result<Bytes> {
        self.reader.seek(SeekFrom::Start(0)).await.map_err(|e| {
            RuntimeError::Storage {
                path: PathBuf::from("<file>"),
                message: "failed to seek to start".to_string(),
                source: Some(e),
            }
        })?;

        let mut buf = Vec::with_capacity(self.size as usize);
        self.reader.read_to_end(&mut buf).await.map_err(|e| {
            RuntimeError::Storage {
                path: PathBuf::from("<file>"),
                message: "failed to read file".to_string(),
                source: Some(e),
            }
        })?;

        Ok(Bytes::from(buf))
    }
}

/// Async buffered file writer for local storage.
pub struct AsyncLocalWriter {
    writer: BufWriter<File>,
    bytes_written: AtomicU64,
}

impl AsyncLocalWriter {
    fn new(file: File, buffer_size: usize) -> Self {
        Self {
            writer: BufWriter::with_capacity(buffer_size, file),
            bytes_written: AtomicU64::new(0),
        }
    }
}

impl AsyncWrite for AsyncLocalWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        let result = Pin::new(&mut self.writer).poll_write(cx, buf);
        if let Poll::Ready(Ok(n)) = &result {
            self.bytes_written.fetch_add(*n as u64, Ordering::Relaxed);
        }
        result
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.writer).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.writer).poll_shutdown(cx)
    }
}

#[async_trait]
impl AsyncStorageWriter for AsyncLocalWriter {
    async fn finish(mut self: Box<Self>) -> Result<()> {
        self.writer.flush().await.map_err(|e| RuntimeError::Storage {
            path: PathBuf::from("<file>"),
            message: "failed to flush writer".to_string(),
            source: Some(e),
        })?;

        // Sync to disk
        self.writer.get_ref().sync_all().await.map_err(|e| RuntimeError::Storage {
            path: PathBuf::from("<file>"),
            message: "failed to sync file to disk".to_string(),
            source: Some(e),
        })?;

        Ok(())
    }

    async fn write_all_bytes(&mut self, data: &[u8]) -> Result<()> {
        self.writer.write_all(data).await.map_err(|e| RuntimeError::Storage {
            path: PathBuf::from("<file>"),
            message: "failed to write data".to_string(),
            source: Some(e),
        })?;
        self.bytes_written.fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn bytes_written(&self) -> u64 {
        self.bytes_written.load(Ordering::Relaxed)
    }
}

/// A stream that reads bytes from a file in chunks.
struct FileByteStream {
    reader: BufReader<File>,
    buffer_size: usize,
    remaining: Option<u64>,
    exhausted: bool,
}

impl FileByteStream {
    fn new(reader: BufReader<File>, buffer_size: usize) -> Self {
        Self {
            reader,
            buffer_size,
            remaining: None,
            exhausted: false,
        }
    }

    fn with_limit(reader: BufReader<File>, buffer_size: usize, limit: u64) -> Self {
        Self {
            reader,
            buffer_size,
            remaining: Some(limit),
            exhausted: false,
        }
    }
}

impl Stream for FileByteStream {
    type Item = Result<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.exhausted {
            return Poll::Ready(None);
        }

        let chunk_size = match self.remaining {
            Some(0) => {
                self.exhausted = true;
                return Poll::Ready(None);
            }
            Some(remaining) => self.buffer_size.min(remaining as usize),
            None => self.buffer_size,
        };

        let mut buf = vec![0u8; chunk_size];
        let mut read_buf = ReadBuf::new(&mut buf);

        match Pin::new(&mut self.reader).poll_read(cx, &mut read_buf) {
            Poll::Ready(Ok(())) => {
                let filled = read_buf.filled().len();
                if filled == 0 {
                    self.exhausted = true;
                    Poll::Ready(None)
                } else {
                    buf.truncate(filled);
                    if let Some(ref mut remaining) = self.remaining {
                        *remaining = remaining.saturating_sub(filled as u64);
                    }
                    Poll::Ready(Some(Ok(Bytes::from(buf))))
                }
            }
            Poll::Ready(Err(e)) => {
                self.exhausted = true;
                Poll::Ready(Some(Err(RuntimeError::Storage {
                    path: PathBuf::from("<file>"),
                    message: "failed to read from file".to_string(),
                    source: Some(e),
                })))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use tempfile::TempDir;

    async fn create_test_storage() -> (AsyncLocalStorage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            buffer_size: 4096,
            use_mmap: false,
            mmap_threshold: 1024,
            ..Default::default()
        };
        let storage = AsyncLocalStorage::new(&config).await.unwrap();
        (storage, temp_dir)
    }

    #[tokio::test]
    async fn test_new_creates_base_dir() {
        let temp_dir = TempDir::new().unwrap();
        let new_base = temp_dir.path().join("new_subdir");

        let config = StorageConfig {
            base_path: new_base.clone(),
            ..Default::default()
        };

        let _storage = AsyncLocalStorage::new(&config).await.unwrap();
        assert!(new_base.exists());
    }

    #[tokio::test]
    async fn test_exists_file() {
        let (storage, _temp) = create_test_storage().await;

        // File doesn't exist yet
        assert!(!storage.exists(Path::new("test.txt")).await.unwrap());

        // Create a file
        let mut writer = storage.open_write(Path::new("test.txt")).await.unwrap();
        writer.write_all_bytes(b"hello").await.unwrap();
        writer.finish().await.unwrap();

        // Now it exists
        assert!(storage.exists(Path::new("test.txt")).await.unwrap());
    }

    #[tokio::test]
    async fn test_metadata_file() {
        let (storage, _temp) = create_test_storage().await;

        // Write a file
        let data = b"hello world";
        let mut writer = storage.open_write(Path::new("test.txt")).await.unwrap();
        writer.write_all_bytes(data).await.unwrap();
        writer.finish().await.unwrap();

        // Check metadata
        let meta = storage.metadata(Path::new("test.txt")).await.unwrap();
        assert_eq!(meta.size, data.len() as u64);
        assert!(!meta.is_dir);
        assert!(meta.modified.is_some());
    }

    #[tokio::test]
    async fn test_write_and_read() {
        let (storage, _temp) = create_test_storage().await;

        // Write a file
        let data = b"hello world";
        let mut writer = storage.open_write(Path::new("test.txt")).await.unwrap();
        writer.write_all_bytes(data).await.unwrap();
        writer.finish().await.unwrap();

        // Read it back
        let mut reader = storage.open_read(Path::new("test.txt")).await.unwrap();
        let content = reader.read_all().await.unwrap();

        assert_eq!(content.as_ref(), data);
        assert_eq!(reader.size(), data.len() as u64);
    }

    #[tokio::test]
    async fn test_read_range() {
        let (storage, _temp) = create_test_storage().await;

        // Write a file
        let data = b"hello world";
        let mut writer = storage.open_write(Path::new("test.txt")).await.unwrap();
        writer.write_all_bytes(data).await.unwrap();
        writer.finish().await.unwrap();

        // Read a range
        let mut reader = storage.open_read(Path::new("test.txt")).await.unwrap();
        let range = reader.read_range(6, 5).await.unwrap();

        assert_eq!(range.as_ref(), b"world");
    }

    #[tokio::test]
    async fn test_get_stream() {
        let (storage, _temp) = create_test_storage().await;

        // Write a file
        let data = b"hello world from stream";
        let mut writer = storage.open_write(Path::new("test.txt")).await.unwrap();
        writer.write_all_bytes(data).await.unwrap();
        writer.finish().await.unwrap();

        // Get stream and collect
        let mut stream = storage.get_stream(Path::new("test.txt")).await.unwrap();
        let mut collected = Vec::new();
        while let Some(chunk) = stream.next().await {
            collected.extend_from_slice(&chunk.unwrap());
        }

        assert_eq!(collected, data);
    }

    #[tokio::test]
    async fn test_get_range_stream() {
        let (storage, _temp) = create_test_storage().await;

        // Write a file
        let data = b"hello world from stream";
        let mut writer = storage.open_write(Path::new("test.txt")).await.unwrap();
        writer.write_all_bytes(data).await.unwrap();
        writer.finish().await.unwrap();

        // Get range stream
        let mut stream = storage.get_range_stream(Path::new("test.txt"), 6, 11).await.unwrap();
        let mut collected = Vec::new();
        while let Some(chunk) = stream.next().await {
            collected.extend_from_slice(&chunk.unwrap());
        }

        assert_eq!(collected, b"world");
    }

    #[tokio::test]
    async fn test_delete_file() {
        let (storage, _temp) = create_test_storage().await;

        // Create a file
        let mut writer = storage.open_write(Path::new("test.txt")).await.unwrap();
        writer.write_all_bytes(b"hello").await.unwrap();
        writer.finish().await.unwrap();

        assert!(storage.exists(Path::new("test.txt")).await.unwrap());

        // Delete it
        storage.delete(Path::new("test.txt")).await.unwrap();

        assert!(!storage.exists(Path::new("test.txt")).await.unwrap());
    }

    #[tokio::test]
    async fn test_list() {
        let (storage, _temp) = create_test_storage().await;

        // Create some files
        storage.create_dir_all(Path::new("dir")).await.unwrap();

        for name in &["c.txt", "a.txt", "b.txt"] {
            let mut writer = storage.open_write(&Path::new("dir").join(name)).await.unwrap();
            writer.write_all_bytes(b"data").await.unwrap();
            writer.finish().await.unwrap();
        }

        // List should be sorted
        let entries = storage.list(Path::new("dir")).await.unwrap();
        assert_eq!(entries, vec!["a.txt", "b.txt", "c.txt"]);
    }

    #[tokio::test]
    async fn test_list_paginated() {
        let (storage, _temp) = create_test_storage().await;

        // Create some files
        storage.create_dir_all(Path::new("dir")).await.unwrap();

        for i in 0..10 {
            let name = format!("file{:02}.txt", i);
            let mut writer = storage.open_write(&Path::new("dir").join(&name)).await.unwrap();
            writer.write_all_bytes(b"data").await.unwrap();
            writer.finish().await.unwrap();
        }

        // First page
        let (page1, token1) = storage.list_paginated(Path::new("dir"), 3, None).await.unwrap();
        assert_eq!(page1.len(), 3);
        assert!(token1.is_some());

        // Second page
        let (page2, token2) = storage.list_paginated(Path::new("dir"), 3, token1.as_deref()).await.unwrap();
        assert_eq!(page2.len(), 3);
        assert!(token2.is_some());

        // Third page
        let (page3, token3) = storage.list_paginated(Path::new("dir"), 3, token2.as_deref()).await.unwrap();
        assert_eq!(page3.len(), 3);
        assert!(token3.is_some());

        // Fourth page (partial)
        let (page4, token4) = storage.list_paginated(Path::new("dir"), 3, token3.as_deref()).await.unwrap();
        assert_eq!(page4.len(), 1);
        assert!(token4.is_none());
    }

    #[tokio::test]
    async fn test_rename() {
        let (storage, _temp) = create_test_storage().await;

        // Create a file
        let mut writer = storage.open_write(Path::new("old.txt")).await.unwrap();
        writer.write_all_bytes(b"hello").await.unwrap();
        writer.finish().await.unwrap();

        // Rename it
        storage.rename(Path::new("old.txt"), Path::new("new.txt")).await.unwrap();

        assert!(!storage.exists(Path::new("old.txt")).await.unwrap());
        assert!(storage.exists(Path::new("new.txt")).await.unwrap());

        // Verify content
        let mut reader = storage.open_read(Path::new("new.txt")).await.unwrap();
        let content = reader.read_all().await.unwrap();
        assert_eq!(content.as_ref(), b"hello");
    }

    #[tokio::test]
    async fn test_copy() {
        let (storage, _temp) = create_test_storage().await;

        // Create a file
        let mut writer = storage.open_write(Path::new("source.txt")).await.unwrap();
        writer.write_all_bytes(b"hello").await.unwrap();
        writer.finish().await.unwrap();

        // Copy it
        storage.copy(Path::new("source.txt"), Path::new("dest.txt")).await.unwrap();

        // Both should exist
        assert!(storage.exists(Path::new("source.txt")).await.unwrap());
        assert!(storage.exists(Path::new("dest.txt")).await.unwrap());

        // Verify content
        let mut reader = storage.open_read(Path::new("dest.txt")).await.unwrap();
        let content = reader.read_all().await.unwrap();
        assert_eq!(content.as_ref(), b"hello");
    }

    #[tokio::test]
    async fn test_create_dir_all() {
        let (storage, _temp) = create_test_storage().await;

        storage.create_dir_all(Path::new("a/b/c/d")).await.unwrap();

        assert!(storage.exists(Path::new("a/b/c/d")).await.unwrap());
        let meta = storage.metadata(Path::new("a/b/c/d")).await.unwrap();
        assert!(meta.is_dir);
    }

    #[tokio::test]
    async fn test_backend_type() {
        let (storage, _temp) = create_test_storage().await;
        assert_eq!(storage.backend_type(), "local");
    }

    #[tokio::test]
    async fn test_supports_atomic_rename() {
        let (storage, _temp) = create_test_storage().await;
        assert!(storage.supports_atomic_rename());
    }

    #[tokio::test]
    async fn test_bytes_written() {
        let (storage, _temp) = create_test_storage().await;

        let mut writer = storage.open_write(Path::new("test.txt")).await.unwrap();
        assert_eq!(writer.bytes_written(), 0);

        writer.write_all_bytes(b"hello").await.unwrap();
        assert_eq!(writer.bytes_written(), 5);

        writer.write_all_bytes(b" world").await.unwrap();
        assert_eq!(writer.bytes_written(), 11);

        writer.finish().await.unwrap();
    }
}
