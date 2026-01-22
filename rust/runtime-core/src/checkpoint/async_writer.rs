// rust/runtime-core/src/checkpoint/async_writer.rs

//! Async checkpoint writer with streaming support.
//!
//! This module provides an async checkpoint writer that supports streaming
//! writes with incremental checksum computation and compression.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use twox_hash::XxHash64;

use super::format_v2::{CheckpointHeaderV2, CheckpointTrailer, CompressionType};
use crate::config::CheckpointConfig;
use crate::error::{Result, RuntimeError};
use crate::storage::{AsyncStorageBackend, AsyncStorageWriter};

/// Async checkpoint writer for streaming writes.
///
/// This writer uses the V2 format with trailer at end, allowing streaming
/// writes without knowing the final size upfront.
pub struct AsyncCheckpointWriter {
    /// The storage backend.
    storage: Arc<dyn AsyncStorageBackend>,
    /// Checkpoint configuration.
    config: CheckpointConfig,
}

impl AsyncCheckpointWriter {
    /// Creates a new async checkpoint writer.
    pub fn new(storage: Arc<dyn AsyncStorageBackend>, config: CheckpointConfig) -> Self {
        Self { storage, config }
    }

    /// Writes checkpoint data with the specified name.
    ///
    /// Returns the path where the checkpoint was written.
    pub async fn write(&self, name: &str, data: &[u8]) -> Result<PathBuf> {
        self.write_with_metadata(name, data, HashMap::new()).await
    }

    /// Writes checkpoint data with metadata.
    ///
    /// Returns the path where the checkpoint was written.
    pub async fn write_with_metadata(
        &self,
        name: &str,
        data: &[u8],
        metadata: HashMap<String, String>,
    ) -> Result<PathBuf> {
        // Generate timestamped filename
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let filename = format!("{}_{}.ckpt", name, timestamp);

        // Determine paths
        let final_path = self.config.checkpoint_dir.join(&filename);
        let temp_path = if self.config.atomic_writes {
            self.config.checkpoint_dir.join(format!(".{}.tmp", filename))
        } else {
            final_path.clone()
        };

        // Ensure directory exists
        self.storage
            .create_dir_all(&self.config.checkpoint_dir)
            .await?;

        // Compute checksum
        let checksum = compute_xxhash64(data);

        // Compress data
        let compression = CompressionType::parse(&self.config.compression);
        let compressed = compress_data(data, compression, self.config.compression_level)?;

        // Create header
        let header = CheckpointHeaderV2::new(compression);

        // Create trailer
        let trailer = CheckpointTrailer::with_metadata(
            checksum,
            data.len() as u64,
            compressed.len() as u64,
            metadata,
        );

        // Serialize trailer
        let trailer_bytes = trailer.to_bytes();
        let trailer_len = trailer_bytes.len() as u32;

        // Write to storage
        let mut writer = self.storage.open_write(&temp_path).await?;

        // Write header
        writer.write_all_bytes(&header.to_bytes()).await?;

        // Write compressed data
        writer.write_all_bytes(&compressed).await?;

        // Write trailer
        writer.write_all_bytes(&trailer_bytes).await?;

        // Write trailer length
        writer.write_all_bytes(&trailer_len.to_le_bytes()).await?;

        // Finish write
        writer.finish().await?;

        // Atomic rename if enabled
        if self.config.atomic_writes && temp_path != final_path {
            self.storage.rename(&temp_path, &final_path).await?;
        }

        // Clean up old checkpoints
        self.cleanup_old_checkpoints(name).await?;

        Ok(final_path)
    }

    /// Writes checkpoint data using streaming compression.
    ///
    /// This is more memory-efficient for very large checkpoints as it
    /// compresses in chunks rather than all at once.
    pub async fn write_streaming(
        &self,
        name: &str,
        data: &[u8],
        chunk_size: usize,
    ) -> Result<PathBuf> {
        self.write_streaming_with_metadata(name, data, chunk_size, HashMap::new())
            .await
    }

    /// Writes checkpoint data using streaming compression with metadata.
    pub async fn write_streaming_with_metadata(
        &self,
        name: &str,
        data: &[u8],
        _chunk_size: usize,
        metadata: HashMap<String, String>,
    ) -> Result<PathBuf> {
        // For now, fall back to regular write
        // A true streaming implementation would compress in chunks
        // and write incrementally to storage
        self.write_with_metadata(name, data, metadata).await
    }

    /// Removes old checkpoints, keeping only the most recent ones.
    async fn cleanup_old_checkpoints(&self, name: &str) -> Result<()> {
        let prefix = format!("{}_", name);
        let entries = self.storage.list(&self.config.checkpoint_dir).await?;

        // Filter to matching checkpoints and sort by timestamp (descending)
        let mut checkpoints: Vec<_> = entries
            .into_iter()
            .filter(|e| e.starts_with(&prefix) && e.ends_with(".ckpt"))
            .collect();

        checkpoints.sort_by(|a, b| b.cmp(a)); // Reverse sort (newest first)

        // Delete old checkpoints beyond keep_last_n
        if checkpoints.len() > self.config.keep_last_n {
            for old_checkpoint in &checkpoints[self.config.keep_last_n..] {
                let path = self.config.checkpoint_dir.join(old_checkpoint);
                if let Err(e) = self.storage.delete(&path).await {
                    // Log but don't fail if cleanup fails
                    eprintln!("Warning: failed to delete old checkpoint {:?}: {}", path, e);
                }
            }
        }

        Ok(())
    }
}

/// Streaming checkpoint writer that allows incremental writes.
///
/// This writer is useful when checkpoint data is generated incrementally
/// and you don't want to buffer the entire checkpoint in memory.
pub struct StreamingCheckpointWriter {
    writer: Box<dyn AsyncStorageWriter>,
    hasher: XxHash64,
    compression: CompressionType,
    compression_level: i32,
    uncompressed_size: u64,
    compressed_size: u64,
    buffer: Vec<u8>,
    chunk_size: usize,
    header_written: bool,
}

impl StreamingCheckpointWriter {
    /// Creates a new streaming checkpoint writer.
    pub async fn new(
        storage: &dyn AsyncStorageBackend,
        path: &Path,
        compression: CompressionType,
        compression_level: i32,
        chunk_size: usize,
    ) -> Result<Self> {
        let writer = storage.open_write(path).await?;

        Ok(Self {
            writer,
            hasher: XxHash64::with_seed(0),
            compression,
            compression_level,
            uncompressed_size: 0,
            compressed_size: 0,
            buffer: Vec::with_capacity(chunk_size),
            chunk_size,
            header_written: false,
        })
    }

    /// Writes the header if not already written.
    async fn ensure_header_written(&mut self) -> Result<()> {
        if !self.header_written {
            let header = CheckpointHeaderV2::new(self.compression);
            self.writer.write_all_bytes(&header.to_bytes()).await?;
            self.header_written = true;
        }
        Ok(())
    }

    /// Writes data to the checkpoint.
    pub async fn write(&mut self, data: &[u8]) -> Result<()> {
        self.ensure_header_written().await?;

        // Update hasher with uncompressed data
        std::hash::Hasher::write(&mut self.hasher, data);
        self.uncompressed_size += data.len() as u64;

        // Buffer data
        self.buffer.extend_from_slice(data);

        // Flush buffer if it exceeds chunk size
        while self.buffer.len() >= self.chunk_size {
            self.flush_chunk().await?;
        }

        Ok(())
    }

    /// Flushes a chunk of buffered data.
    async fn flush_chunk(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let chunk_data: Vec<u8> = self.buffer.drain(..self.chunk_size.min(self.buffer.len())).collect();
        let compressed = compress_data(&chunk_data, self.compression, self.compression_level)?;

        self.writer.write_all_bytes(&compressed).await?;
        self.compressed_size += compressed.len() as u64;

        Ok(())
    }

    /// Finishes the checkpoint write, writing any remaining data and the trailer.
    pub async fn finish(mut self, metadata: HashMap<String, String>) -> Result<()> {
        self.ensure_header_written().await?;

        // Flush any remaining buffered data
        while !self.buffer.is_empty() {
            self.flush_chunk().await?;
        }

        // Compute final checksum
        let checksum = std::hash::Hasher::finish(&self.hasher);

        // Create and write trailer
        let trailer = CheckpointTrailer::with_metadata(
            checksum,
            self.uncompressed_size,
            self.compressed_size,
            metadata,
        );
        let trailer_bytes = trailer.to_bytes();
        let trailer_len = trailer_bytes.len() as u32;

        self.writer.write_all_bytes(&trailer_bytes).await?;
        self.writer.write_all_bytes(&trailer_len.to_le_bytes()).await?;

        self.writer.finish().await
    }
}

/// Computes XXHash64 checksum of data.
fn compute_xxhash64(data: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut hasher = XxHash64::with_seed(0);
    hasher.write(data);
    hasher.finish()
}

/// Compresses data using the specified algorithm.
fn compress_data(data: &[u8], compression: CompressionType, level: i32) -> Result<Vec<u8>> {
    match compression {
        CompressionType::None => Ok(data.to_vec()),
        CompressionType::Lz4 => {
            let compressed = lz4_flex::compress_prepend_size(data);
            Ok(compressed)
        }
        CompressionType::Zstd => {
            let level = level.clamp(1, 22);
            let compressed = zstd::encode_all(data, level).map_err(|e| {
                RuntimeError::checkpoint_with_source("zstd compression failed", e)
            })?;
            Ok(compressed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::format_v2::{HEADER_SIZE, TRAILER_LEN_SIZE};
    use crate::storage::AsyncLocalStorage;
    use crate::config::StorageConfig;
    use tempfile::TempDir;

    async fn create_test_storage() -> (Arc<dyn AsyncStorageBackend>, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let storage = AsyncLocalStorage::new(&config).await.unwrap();
        (Arc::new(storage), temp_dir)
    }

    #[tokio::test]
    async fn test_write_checkpoint() {
        let (storage, temp_dir) = create_test_storage().await;
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().join("checkpoints"),
            compression: "lz4".to_string(),
            ..Default::default()
        };

        let writer = AsyncCheckpointWriter::new(storage.clone(), config);
        let data = b"test checkpoint data";

        let path = writer.write("test", data).await.unwrap();
        assert!(path.exists());

        // Verify file structure (header + compressed + trailer + trailer_len)
        let content = std::fs::read(&path).unwrap();
        assert!(content.len() > HEADER_SIZE + TRAILER_LEN_SIZE);

        // Verify magic bytes
        assert_eq!(&content[0..4], b"DTR2");
    }

    #[tokio::test]
    async fn test_write_with_metadata() {
        let (storage, temp_dir) = create_test_storage().await;
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().join("checkpoints"),
            compression: "none".to_string(),
            ..Default::default()
        };

        let writer = AsyncCheckpointWriter::new(storage, config);

        let mut metadata = HashMap::new();
        metadata.insert("epoch".to_string(), "10".to_string());
        metadata.insert("step".to_string(), "5000".to_string());

        let path = writer
            .write_with_metadata("test", b"data", metadata)
            .await
            .unwrap();
        assert!(path.exists());
    }

    #[tokio::test]
    async fn test_cleanup_old_checkpoints() {
        let (storage, temp_dir) = create_test_storage().await;
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().join("checkpoints"),
            compression: "none".to_string(),
            keep_last_n: 2,
            ..Default::default()
        };

        let writer = AsyncCheckpointWriter::new(storage.clone(), config.clone());

        // Write 4 checkpoints
        for i in 0..4 {
            writer.write("test", format!("data{}", i).as_bytes()).await.unwrap();
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        // Should only have 2 checkpoints remaining
        let entries = storage.list(&config.checkpoint_dir).await.unwrap();
        let checkpoints: Vec<_> = entries
            .iter()
            .filter(|e| e.starts_with("test_") && e.ends_with(".ckpt"))
            .collect();
        assert_eq!(checkpoints.len(), 2);
    }

    #[test]
    fn test_compute_xxhash64() {
        let data = b"hello world";
        let hash1 = compute_xxhash64(data);
        let hash2 = compute_xxhash64(data);
        assert_eq!(hash1, hash2);

        let different_data = b"different data";
        let hash3 = compute_xxhash64(different_data);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_compress_none() {
        let data = b"test data";
        let compressed = compress_data(data, CompressionType::None, 0).unwrap();
        assert_eq!(compressed, data);
    }

    #[test]
    fn test_compress_lz4() {
        let data = b"test data that should be compressed somewhat";
        let compressed = compress_data(data, CompressionType::Lz4, 0).unwrap();
        // LZ4 prepends size, so compressed should be different
        assert_ne!(compressed.as_slice(), data.as_slice());
    }

    #[test]
    fn test_compress_zstd() {
        let data = b"test data that should be compressed somewhat";
        let compressed = compress_data(data, CompressionType::Zstd, 3).unwrap();
        // Verify we can decompress
        let decompressed = zstd::decode_all(compressed.as_slice()).unwrap();
        assert_eq!(decompressed, data);
    }
}
