// rust/runtime-core/src/checkpoint/reader.rs

//! Checkpoint reader implementation.

use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use twox_hash::XxHash64;

use crate::error::{Result, RuntimeError};
use crate::storage::StorageBackend;

use super::format::CheckpointHeader;

/// Reads and verifies checkpoints.
///
/// The `CheckpointReader` handles:
/// - Reading checkpoint files
/// - Decompressing data with the appropriate algorithm
/// - Verifying checksums for integrity
pub struct CheckpointReader {
    storage: Arc<dyn StorageBackend>,
}

impl CheckpointReader {
    /// Creates a new checkpoint reader.
    pub fn new(storage: Arc<dyn StorageBackend>) -> Self {
        Self { storage }
    }

    /// Reads and decompresses a checkpoint, verifying its integrity.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the checkpoint file
    ///
    /// # Returns
    ///
    /// The decompressed checkpoint data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - The magic bytes are invalid
    /// - The checksum doesn't match
    /// - Decompression fails
    pub fn read(&self, path: &Path) -> Result<Vec<u8>> {
        // Read the entire checkpoint file
        let mut reader = self.storage.open_read(path)?;
        let mut checkpoint_data = Vec::new();
        reader.read_to_end(&mut checkpoint_data).map_err(|e| {
            RuntimeError::checkpoint(format!("failed to read checkpoint file: {e}"))
        })?;

        // Parse header length (first 4 bytes)
        if checkpoint_data.len() < 4 {
            return Err(RuntimeError::checkpoint("checkpoint file too small"));
        }

        let header_len = u32::from_le_bytes(checkpoint_data[..4].try_into().unwrap()) as usize;

        if checkpoint_data.len() < 4 + header_len {
            return Err(RuntimeError::checkpoint(
                "checkpoint file truncated: header incomplete",
            ));
        }

        // Deserialize header
        let header: CheckpointHeader = bincode::deserialize(&checkpoint_data[4..4 + header_len])
            .map_err(|e| RuntimeError::checkpoint(format!("failed to deserialize header: {e}")))?;

        // Validate magic bytes
        if !header.validate_magic() {
            return Err(RuntimeError::checkpoint(format!(
                "invalid magic bytes: expected {:?}, got {:?}",
                CheckpointHeader::MAGIC,
                header.magic
            )));
        }

        // Validate version
        if !header.validate_version() {
            return Err(RuntimeError::checkpoint(format!(
                "unsupported version: expected {}, got {}",
                CheckpointHeader::VERSION,
                header.version
            )));
        }

        // Extract compressed data
        let compressed_data = &checkpoint_data[4 + header_len..];

        // Decompress
        let decompressed = self.decompress(compressed_data, &header.compression)?;

        // Verify checksum
        let computed_checksum = self.calculate_checksum(&decompressed);
        if computed_checksum != header.checksum {
            return Err(RuntimeError::checkpoint(format!(
                "checksum mismatch: expected {}, got {}",
                header.checksum, computed_checksum
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

    /// Reads the header from a checkpoint without decompressing the data.
    ///
    /// This is useful for inspecting checkpoint metadata without loading
    /// the full data.
    pub fn read_header(&self, path: &Path) -> Result<CheckpointHeader> {
        let mut reader = self.storage.open_read(path)?;

        // Read header length
        let mut len_bytes = [0u8; 4];
        reader
            .read_exact(&mut len_bytes)
            .map_err(|e| RuntimeError::checkpoint(format!("failed to read header length: {e}")))?;
        let header_len = u32::from_le_bytes(len_bytes) as usize;

        // Read header bytes
        let mut header_bytes = vec![0u8; header_len];
        reader
            .read_exact(&mut header_bytes)
            .map_err(|e| RuntimeError::checkpoint(format!("failed to read header: {e}")))?;

        // Deserialize
        let header: CheckpointHeader = bincode::deserialize(&header_bytes)
            .map_err(|e| RuntimeError::checkpoint(format!("failed to deserialize header: {e}")))?;

        // Validate magic
        if !header.validate_magic() {
            return Err(RuntimeError::checkpoint("invalid magic bytes"));
        }

        Ok(header)
    }

    /// Decompresses data according to the specified algorithm.
    fn decompress(&self, data: &[u8], compression: &str) -> Result<Vec<u8>> {
        match compression {
            "none" => Ok(data.to_vec()),
            "lz4" => lz4_flex::decompress_size_prepended(data)
                .map_err(|e| RuntimeError::checkpoint(format!("lz4 decompression failed: {e}"))),
            "zstd" => zstd::decode_all(data)
                .map_err(|e| RuntimeError::checkpoint(format!("zstd decompression failed: {e}"))),
            _ => Err(RuntimeError::checkpoint(format!(
                "unknown compression algorithm: {compression}"
            ))),
        }
    }

    /// Calculates XXHash64 checksum of data.
    fn calculate_checksum(&self, data: &[u8]) -> u64 {
        use std::hash::Hasher;
        let mut hasher = XxHash64::with_seed(0);
        hasher.write(data);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::CheckpointWriter;
    use crate::config::{CheckpointConfig, StorageConfig};
    use crate::storage::LocalStorage;
    use std::io::Write as IoWrite;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_setup() -> (
        CheckpointWriter,
        CheckpointReader,
        Arc<dyn StorageBackend>,
        TempDir,
    ) {
        let temp_dir = TempDir::new().unwrap();

        let storage_config = StorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let storage: Arc<dyn StorageBackend> =
            Arc::new(LocalStorage::new(&storage_config).unwrap());

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: PathBuf::from("checkpoints"),
            compression: "none".to_string(),
            compression_level: 1,
            keep_last_n: 10,
            atomic_writes: true,
        };

        let writer = CheckpointWriter::new(storage.clone(), checkpoint_config);
        let reader = CheckpointReader::new(storage.clone());

        (writer, reader, storage, temp_dir)
    }

    #[test]
    fn test_write_read_roundtrip() {
        let (writer, reader, _, _temp) = create_test_setup();

        let original_data = b"This is test checkpoint data for roundtrip verification";
        let path = writer.write("test", original_data).unwrap();

        let read_data = reader.read(&path).unwrap();
        assert_eq!(read_data, original_data);
    }

    #[test]
    fn test_compression_none() {
        let (writer, reader, _, _temp) = create_test_setup();

        let data = b"uncompressed data test";
        let path = writer.write("none_test", data).unwrap();

        let read_data = reader.read(&path).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_compression_lz4() {
        let temp_dir = TempDir::new().unwrap();
        let storage_config = StorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let storage: Arc<dyn StorageBackend> =
            Arc::new(LocalStorage::new(&storage_config).unwrap());

        let config = CheckpointConfig {
            checkpoint_dir: PathBuf::from("checkpoints"),
            compression: "lz4".to_string(),
            compression_level: 1,
            keep_last_n: 10,
            atomic_writes: true,
        };

        let writer = CheckpointWriter::new(storage.clone(), config);
        let reader = CheckpointReader::new(storage);

        let data = b"lz4 compressed data test with some repeated content content content";
        let path = writer.write("lz4_test", data).unwrap();

        let read_data = reader.read(&path).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_compression_zstd() {
        let temp_dir = TempDir::new().unwrap();
        let storage_config = StorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let storage: Arc<dyn StorageBackend> =
            Arc::new(LocalStorage::new(&storage_config).unwrap());

        let config = CheckpointConfig {
            checkpoint_dir: PathBuf::from("checkpoints"),
            compression: "zstd".to_string(),
            compression_level: 3,
            keep_last_n: 10,
            atomic_writes: true,
        };

        let writer = CheckpointWriter::new(storage.clone(), config);
        let reader = CheckpointReader::new(storage);

        let data = b"zstd compressed data test with repeated content content content";
        let path = writer.write("zstd_test", data).unwrap();

        let read_data = reader.read(&path).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_compression_ratio() {
        let temp_dir = TempDir::new().unwrap();
        let storage_config = StorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let storage: Arc<dyn StorageBackend> =
            Arc::new(LocalStorage::new(&storage_config).unwrap());

        let config = CheckpointConfig {
            checkpoint_dir: PathBuf::from("checkpoints"),
            compression: "lz4".to_string(),
            compression_level: 1,
            keep_last_n: 10,
            atomic_writes: true,
        };

        let writer = CheckpointWriter::new(storage.clone(), config);

        // Highly compressible data (lots of repetition)
        let data: Vec<u8> = (0..10000).map(|_| b'a').collect();
        let path = writer.write("compressible", &data).unwrap();

        // Check that compressed file is smaller
        let meta = storage.metadata(&path).unwrap();
        assert!(
            meta.size < data.len() as u64,
            "Compressed size {} should be less than uncompressed {}",
            meta.size,
            data.len()
        );
    }

    #[test]
    fn test_checksum_verification() {
        let (writer, reader, _, _temp) = create_test_setup();

        let data = b"data with valid checksum";
        let path = writer.write("checksum_test", data).unwrap();

        // Should succeed with valid checksum
        let result = reader.read(&path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_checksum_mismatch() {
        let (writer, _, storage, _temp) = create_test_setup();

        let data = b"original data";
        let path = writer.write("corrupt_test", data).unwrap();

        // Corrupt the file by modifying the compressed data section
        let mut checkpoint_data = Vec::new();
        {
            let mut r = storage.open_read(&path).unwrap();
            r.read_to_end(&mut checkpoint_data).unwrap();
        }

        // Modify a byte in the data section (after header)
        if checkpoint_data.len() > 50 {
            checkpoint_data[50] ^= 0xFF; // Flip bits
        }

        // Write corrupted data back
        {
            let mut w = storage.open_write(&path).unwrap();
            w.write_all(&checkpoint_data).unwrap();
            w.finish().unwrap();
        }

        // Create a new reader and try to read
        let reader = CheckpointReader::new(storage);
        let result = reader.read(&path);

        // Should fail with checksum mismatch or decompression error
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_magic() {
        let (_, _, storage, temp_dir) = create_test_setup();

        // Create a file with invalid magic bytes
        let path = temp_dir.path().join("checkpoints/invalid_magic.ckpt");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        // Create a header with wrong magic
        let mut header = CheckpointHeader::new("none".to_string(), 4, 0);
        header.magic = *b"XXXX";

        let header_bytes = bincode::serialize(&header).unwrap();
        let header_len = header_bytes.len() as u32;

        let mut file_data = Vec::new();
        file_data.extend_from_slice(&header_len.to_le_bytes());
        file_data.extend_from_slice(&header_bytes);
        file_data.extend_from_slice(b"data");

        std::fs::write(&path, &file_data).unwrap();

        let reader = CheckpointReader::new(storage);
        let result = reader.read(&path);

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("magic"));
    }

    #[test]
    fn test_read_header() {
        let (writer, reader, _, _temp) = create_test_setup();

        let data = b"test data for header reading";
        let path = writer.write("header_test", data).unwrap();

        let header = reader.read_header(&path).unwrap();

        assert_eq!(header.magic, CheckpointHeader::MAGIC);
        assert_eq!(header.version, CheckpointHeader::VERSION);
        assert_eq!(header.compression, "none");
        assert_eq!(header.uncompressed_size, data.len() as u64);
    }

    #[test]
    fn test_large_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let storage_config = StorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let storage: Arc<dyn StorageBackend> =
            Arc::new(LocalStorage::new(&storage_config).unwrap());

        let config = CheckpointConfig {
            checkpoint_dir: PathBuf::from("checkpoints"),
            compression: "lz4".to_string(),
            compression_level: 1,
            keep_last_n: 3,
            atomic_writes: true,
        };

        let writer = CheckpointWriter::new(storage.clone(), config);
        let reader = CheckpointReader::new(storage);

        // Create 100MB+ of data (use pattern for compressibility and verification)
        let size = 100 * 1024 * 1024 + 1234; // Just over 100MB
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        let path = writer.write("large_checkpoint", &data).unwrap();
        let read_data = reader.read(&path).unwrap();

        assert_eq!(read_data.len(), data.len());
        assert_eq!(read_data, data);
    }
}
