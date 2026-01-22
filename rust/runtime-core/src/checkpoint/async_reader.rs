// rust/runtime-core/src/checkpoint/async_reader.rs

//! Async checkpoint reader supporting both V1 and V2 formats.
//!
//! This module provides an async checkpoint reader that can read checkpoints
//! written with either the V1 (header-first) or V2 (trailer-at-end) format.

use std::collections::HashMap;
use std::hash::Hasher;
use std::path::Path;
use std::sync::Arc;

use twox_hash::XxHash64;

use super::format::CheckpointHeader;
use super::format_v2::{
    CheckpointHeaderV2, CheckpointTrailer, CheckpointVersion, HEADER_SIZE,
    TRAILER_LEN_SIZE,
};
use crate::error::{Result, RuntimeError};
use crate::storage::AsyncStorageBackend;

/// Async checkpoint reader supporting V1 and V2 formats.
pub struct AsyncCheckpointReader {
    /// The storage backend.
    storage: Arc<dyn AsyncStorageBackend>,
}

impl AsyncCheckpointReader {
    /// Creates a new async checkpoint reader.
    pub fn new(storage: Arc<dyn AsyncStorageBackend>) -> Self {
        Self { storage }
    }

    /// Reads a checkpoint from the specified path.
    ///
    /// Automatically detects the format version and reads accordingly.
    pub async fn read(&self, path: &Path) -> Result<Vec<u8>> {
        // Read the first 4 bytes to detect format
        let mut reader = self.storage.open_read(path).await?;
        let magic_bytes = reader.read_range(0, 4).await?;

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&magic_bytes);

        match CheckpointVersion::detect(&magic) {
            Some(CheckpointVersion::V1) => self.read_v1(path).await,
            Some(CheckpointVersion::V2) => self.read_v2(path).await,
            None => Err(RuntimeError::checkpoint(format!(
                "unknown checkpoint format: magic bytes {:?}",
                magic
            ))),
        }
    }

    /// Reads the header/metadata from a checkpoint without reading the full data.
    pub async fn read_metadata(&self, path: &Path) -> Result<CheckpointMetadata> {
        let mut reader = self.storage.open_read(path).await?;
        let magic_bytes = reader.read_range(0, 4).await?;

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&magic_bytes);

        match CheckpointVersion::detect(&magic) {
            Some(CheckpointVersion::V1) => self.read_metadata_v1(path).await,
            Some(CheckpointVersion::V2) => self.read_metadata_v2(path).await,
            None => Err(RuntimeError::checkpoint(format!(
                "unknown checkpoint format: magic bytes {:?}",
                magic
            ))),
        }
    }

    /// Reads a V1 format checkpoint.
    async fn read_v1(&self, path: &Path) -> Result<Vec<u8>> {
        let mut reader = self.storage.open_read(path).await?;
        let file_size = reader.size();

        // Read header length (first 4 bytes)
        let header_len_bytes = reader.read_range(0, 4).await?;
        let header_len = u32::from_le_bytes([
            header_len_bytes[0],
            header_len_bytes[1],
            header_len_bytes[2],
            header_len_bytes[3],
        ]) as usize;

        // Read header
        let header_bytes = reader.read_range(4, header_len).await?;
        let header: CheckpointHeader = bincode::deserialize(&header_bytes).map_err(|e| {
            RuntimeError::checkpoint_with_source("failed to deserialize V1 header", e)
        })?;

        // Validate magic and version
        if !header.validate_magic() {
            return Err(RuntimeError::checkpoint("invalid V1 checkpoint magic bytes"));
        }
        if !header.validate_version() {
            return Err(RuntimeError::checkpoint(format!(
                "unsupported V1 checkpoint version: {}",
                header.version
            )));
        }

        // Read compressed data
        let data_offset = 4 + header_len;
        let data_len = file_size as usize - data_offset;
        let compressed = reader.read_range(data_offset as u64, data_len).await?;

        // Decompress
        let decompressed = decompress_data(&compressed, &header.compression)?;

        // Verify checksum
        let checksum = compute_xxhash64(&decompressed);
        if checksum != header.checksum {
            return Err(RuntimeError::checkpoint(format!(
                "checksum mismatch: expected {}, got {}",
                header.checksum, checksum
            )));
        }

        // Verify size
        if decompressed.len() as u64 != header.uncompressed_size {
            return Err(RuntimeError::checkpoint(format!(
                "size mismatch: expected {}, got {}",
                header.uncompressed_size,
                decompressed.len()
            )));
        }

        Ok(decompressed)
    }

    /// Reads metadata from a V1 format checkpoint.
    async fn read_metadata_v1(&self, path: &Path) -> Result<CheckpointMetadata> {
        let mut reader = self.storage.open_read(path).await?;

        // Read header length
        let header_len_bytes = reader.read_range(0, 4).await?;
        let header_len = u32::from_le_bytes([
            header_len_bytes[0],
            header_len_bytes[1],
            header_len_bytes[2],
            header_len_bytes[3],
        ]) as usize;

        // Read header
        let header_bytes = reader.read_range(4, header_len).await?;
        let header: CheckpointHeader = bincode::deserialize(&header_bytes).map_err(|e| {
            RuntimeError::checkpoint_with_source("failed to deserialize V1 header", e)
        })?;

        Ok(CheckpointMetadata {
            version: CheckpointVersion::V1,
            compression: header.compression.clone(),
            uncompressed_size: header.uncompressed_size,
            checksum: header.checksum,
            metadata: header.metadata.clone(),
        })
    }

    /// Reads a V2 format checkpoint.
    async fn read_v2(&self, path: &Path) -> Result<Vec<u8>> {
        let mut reader = self.storage.open_read(path).await?;
        let file_size = reader.size();

        // Read header
        let header_bytes = reader.read_range(0, HEADER_SIZE).await?;
        let header = CheckpointHeaderV2::from_bytes(&header_bytes).ok_or_else(|| {
            RuntimeError::checkpoint("failed to parse V2 header")
        })?;

        if !header.validate_magic() {
            return Err(RuntimeError::checkpoint("invalid V2 checkpoint magic bytes"));
        }

        // Read trailer length (last 4 bytes)
        let trailer_len_offset = file_size - TRAILER_LEN_SIZE as u64;
        let trailer_len_bytes = reader.read_range(trailer_len_offset, TRAILER_LEN_SIZE).await?;
        let trailer_len = u32::from_le_bytes([
            trailer_len_bytes[0],
            trailer_len_bytes[1],
            trailer_len_bytes[2],
            trailer_len_bytes[3],
        ]) as usize;

        // Read trailer
        let trailer_offset = trailer_len_offset - trailer_len as u64;
        let trailer_bytes = reader.read_range(trailer_offset, trailer_len).await?;
        let trailer = CheckpointTrailer::from_bytes(&trailer_bytes).ok_or_else(|| {
            RuntimeError::checkpoint("failed to parse V2 trailer")
        })?;

        // Read compressed data
        let data_offset = HEADER_SIZE;
        let data_len = trailer_offset as usize - data_offset;
        let compressed = reader.read_range(data_offset as u64, data_len).await?;

        // Verify compressed size
        if compressed.len() as u64 != trailer.compressed_size {
            return Err(RuntimeError::checkpoint(format!(
                "compressed size mismatch: expected {}, got {}",
                trailer.compressed_size,
                compressed.len()
            )));
        }

        // Decompress
        let compression = header.compression();
        let decompressed = decompress_data(&compressed, compression.as_str())?;

        // Verify checksum
        let checksum = compute_xxhash64(&decompressed);
        if checksum != trailer.checksum {
            return Err(RuntimeError::checkpoint(format!(
                "checksum mismatch: expected {}, got {}",
                trailer.checksum, checksum
            )));
        }

        // Verify uncompressed size
        if decompressed.len() as u64 != trailer.uncompressed_size {
            return Err(RuntimeError::checkpoint(format!(
                "uncompressed size mismatch: expected {}, got {}",
                trailer.uncompressed_size,
                decompressed.len()
            )));
        }

        Ok(decompressed)
    }

    /// Reads metadata from a V2 format checkpoint.
    async fn read_metadata_v2(&self, path: &Path) -> Result<CheckpointMetadata> {
        let mut reader = self.storage.open_read(path).await?;
        let file_size = reader.size();

        // Read header
        let header_bytes = reader.read_range(0, HEADER_SIZE).await?;
        let header = CheckpointHeaderV2::from_bytes(&header_bytes).ok_or_else(|| {
            RuntimeError::checkpoint("failed to parse V2 header")
        })?;

        // Read trailer length
        let trailer_len_offset = file_size - TRAILER_LEN_SIZE as u64;
        let trailer_len_bytes = reader.read_range(trailer_len_offset, TRAILER_LEN_SIZE).await?;
        let trailer_len = u32::from_le_bytes([
            trailer_len_bytes[0],
            trailer_len_bytes[1],
            trailer_len_bytes[2],
            trailer_len_bytes[3],
        ]) as usize;

        // Read trailer
        let trailer_offset = trailer_len_offset - trailer_len as u64;
        let trailer_bytes = reader.read_range(trailer_offset, trailer_len).await?;
        let trailer = CheckpointTrailer::from_bytes(&trailer_bytes).ok_or_else(|| {
            RuntimeError::checkpoint("failed to parse V2 trailer")
        })?;

        Ok(CheckpointMetadata {
            version: CheckpointVersion::V2,
            compression: header.compression().as_str().to_string(),
            uncompressed_size: trailer.uncompressed_size,
            checksum: trailer.checksum,
            metadata: trailer.metadata.clone(),
        })
    }
}

/// Metadata about a checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    /// Format version.
    pub version: CheckpointVersion,
    /// Compression algorithm.
    pub compression: String,
    /// Uncompressed data size.
    pub uncompressed_size: u64,
    /// Checksum of uncompressed data.
    pub checksum: u64,
    /// User-defined metadata.
    pub metadata: HashMap<String, String>,
}

/// Decompresses data using the specified algorithm.
fn decompress_data(data: &[u8], compression: &str) -> Result<Vec<u8>> {
    match compression.to_lowercase().as_str() {
        "none" => Ok(data.to_vec()),
        "lz4" => {
            lz4_flex::decompress_size_prepended(data).map_err(|e| {
                RuntimeError::checkpoint(format!("lz4 decompression failed: {}", e))
            })
        }
        "zstd" => {
            zstd::decode_all(data).map_err(|e| {
                RuntimeError::checkpoint_with_source("zstd decompression failed", e)
            })
        }
        _ => Err(RuntimeError::checkpoint(format!(
            "unsupported compression algorithm: {}",
            compression
        ))),
    }
}

/// Computes XXHash64 checksum of data.
fn compute_xxhash64(data: &[u8]) -> u64 {
    let mut hasher = XxHash64::with_seed(0);
    hasher.write(data);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::async_writer::AsyncCheckpointWriter;
    use crate::config::{CheckpointConfig, StorageConfig};
    use crate::storage::AsyncLocalStorage;
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
    async fn test_read_v2_checkpoint() {
        let (storage, temp_dir) = create_test_storage().await;
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().join("checkpoints"),
            compression: "lz4".to_string(),
            ..Default::default()
        };

        // Write a checkpoint
        let writer = AsyncCheckpointWriter::new(storage.clone(), config);
        let original_data = b"test checkpoint data for reading";
        let path = writer.write("test", original_data).await.unwrap();

        // Read it back
        let reader = AsyncCheckpointReader::new(storage);
        let data = reader.read(&path).await.unwrap();

        assert_eq!(data, original_data);
    }

    #[tokio::test]
    async fn test_read_v2_with_metadata() {
        let (storage, temp_dir) = create_test_storage().await;
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().join("checkpoints"),
            compression: "none".to_string(),
            ..Default::default()
        };

        let writer = AsyncCheckpointWriter::new(storage.clone(), config);

        let mut metadata = HashMap::new();
        metadata.insert("epoch".to_string(), "10".to_string());
        metadata.insert("step".to_string(), "5000".to_string());

        let original_data = b"data with metadata";
        let path = writer
            .write_with_metadata("test", original_data, metadata.clone())
            .await
            .unwrap();

        let reader = AsyncCheckpointReader::new(storage);

        // Read metadata
        let ckpt_metadata = reader.read_metadata(&path).await.unwrap();
        assert_eq!(ckpt_metadata.version, CheckpointVersion::V2);
        assert_eq!(ckpt_metadata.compression, "none");
        assert_eq!(ckpt_metadata.uncompressed_size, original_data.len() as u64);
        assert_eq!(ckpt_metadata.metadata.get("epoch"), Some(&"10".to_string()));
        assert_eq!(ckpt_metadata.metadata.get("step"), Some(&"5000".to_string()));

        // Read full data
        let data = reader.read(&path).await.unwrap();
        assert_eq!(data, original_data);
    }

    #[tokio::test]
    async fn test_read_v2_zstd() {
        let (storage, temp_dir) = create_test_storage().await;
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().join("checkpoints"),
            compression: "zstd".to_string(),
            compression_level: 3,
            ..Default::default()
        };

        let writer = AsyncCheckpointWriter::new(storage.clone(), config);
        let original_data = b"test checkpoint data compressed with zstd";
        let path = writer.write("test", original_data).await.unwrap();

        let reader = AsyncCheckpointReader::new(storage);
        let data = reader.read(&path).await.unwrap();

        assert_eq!(data, original_data);
    }

    #[tokio::test]
    async fn test_read_invalid_magic() {
        let (storage, temp_dir) = create_test_storage().await;

        // Write invalid data
        let path = temp_dir.path().join("invalid.ckpt");
        let mut writer = storage.open_write(&path).await.unwrap();
        writer.write_all_bytes(b"XXXX invalid checkpoint").await.unwrap();
        writer.finish().await.unwrap();

        let reader = AsyncCheckpointReader::new(storage);
        let result = reader.read(&path).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unknown checkpoint format"));
    }

    #[test]
    fn test_decompress_none() {
        let data = b"test data";
        let result = decompress_data(data, "none").unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_decompress_lz4() {
        let original = b"test data to compress";
        let compressed = lz4_flex::compress_prepend_size(original);
        let result = decompress_data(&compressed, "lz4").unwrap();
        assert_eq!(result, original);
    }

    #[test]
    fn test_decompress_zstd() {
        let original = b"test data to compress";
        let compressed = zstd::encode_all(original.as_slice(), 3).unwrap();
        let result = decompress_data(&compressed, "zstd").unwrap();
        assert_eq!(result, original);
    }

    #[test]
    fn test_decompress_invalid_algorithm() {
        let result = decompress_data(b"data", "gzip");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unsupported compression"));
    }
}
