// rust/runtime-core/src/checkpoint/writer.rs

//! Checkpoint writer implementation.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use twox_hash::XxHash64;

use crate::config::CheckpointConfig;
use crate::error::{Result, RuntimeError};
use crate::storage::StorageBackend;

use super::format::CheckpointHeader;

/// Writes checkpoints with compression and integrity verification.
///
/// The `CheckpointWriter` handles:
/// - Compressing data with configurable algorithms (none, lz4, zstd)
/// - Computing checksums for integrity verification
/// - Atomic writes (write to temp file then rename)
/// - Cleanup of old checkpoints
pub struct CheckpointWriter {
    storage: Arc<dyn StorageBackend>,
    config: CheckpointConfig,
}

impl CheckpointWriter {
    /// Creates a new checkpoint writer.
    pub fn new(storage: Arc<dyn StorageBackend>, config: CheckpointConfig) -> Self {
        Self { storage, config }
    }

    /// Writes a checkpoint and returns the path to the saved file.
    ///
    /// The checkpoint is written atomically (if configured) and old checkpoints
    /// are cleaned up according to the `keep_last_n` configuration.
    ///
    /// # Arguments
    ///
    /// * `name` - Base name for the checkpoint (e.g., "model")
    /// * `data` - Raw data to checkpoint
    ///
    /// # Returns
    ///
    /// Path to the saved checkpoint file.
    pub fn write(&self, name: &str, data: &[u8]) -> Result<PathBuf> {
        // Ensure checkpoint directory exists
        self.storage.create_dir_all(&self.config.checkpoint_dir)?;

        // Calculate checksum of uncompressed data
        let checksum = self.calculate_checksum(data);

        // Compress the data
        let (compressed_data, compression) = self.compress(data)?;

        // Create header
        let header = CheckpointHeader::new(compression, data.len() as u64, checksum);

        // Serialize header
        let header_bytes = bincode::serialize(&header)
            .map_err(|e| RuntimeError::checkpoint(format!("failed to serialize header: {e}")))?;

        // Create the full checkpoint data: header length (4 bytes) + header + compressed data
        let header_len = header_bytes.len() as u32;
        let mut checkpoint_data =
            Vec::with_capacity(4 + header_bytes.len() + compressed_data.len());
        checkpoint_data.extend_from_slice(&header_len.to_le_bytes());
        checkpoint_data.extend_from_slice(&header_bytes);
        checkpoint_data.extend_from_slice(&compressed_data);

        // Generate timestamped filename
        let filename = self.generate_filename(name);
        let final_path = self.config.checkpoint_dir.join(&filename);

        if self.config.atomic_writes {
            // Write to temporary file first, then rename
            let temp_filename = format!(".{filename}.tmp");
            let temp_path = self.config.checkpoint_dir.join(&temp_filename);

            // Write to temp file
            self.write_to_path(&temp_path, &checkpoint_data)?;

            // Atomic rename
            self.storage.rename(&temp_path, &final_path)?;
        } else {
            // Direct write
            self.write_to_path(&final_path, &checkpoint_data)?;
        }

        // Cleanup old checkpoints
        self.cleanup_old_checkpoints(name)?;

        Ok(final_path)
    }

    /// Compresses data according to the configured algorithm.
    ///
    /// Returns the compressed data and the compression algorithm name.
    fn compress(&self, data: &[u8]) -> Result<(Vec<u8>, String)> {
        let compression = self.config.compression.as_str();

        let compressed = match compression {
            "none" => data.to_vec(),
            "lz4" => lz4_flex::compress_prepend_size(data),
            "zstd" => {
                let level = self.config.compression_level;
                zstd::encode_all(data, level).map_err(|e| {
                    RuntimeError::checkpoint(format!("zstd compression failed: {e}"))
                })?
            }
            _ => {
                return Err(RuntimeError::checkpoint(format!(
                    "unknown compression algorithm: {compression}"
                )));
            }
        };

        Ok((compressed, compression.to_string()))
    }

    /// Calculates XXHash64 checksum of data.
    fn calculate_checksum(&self, data: &[u8]) -> u64 {
        use std::hash::Hasher;
        let mut hasher = XxHash64::with_seed(0);
        hasher.write(data);
        hasher.finish()
    }

    /// Generates a timestamped filename for a checkpoint.
    fn generate_filename(&self, name: &str) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        format!("{name}_{timestamp}.ckpt")
    }

    /// Writes data to a path using the storage backend.
    fn write_to_path(&self, path: &Path, data: &[u8]) -> Result<()> {
        let mut writer = self.storage.open_write(path)?;
        writer.write_all(data).map_err(|e| {
            RuntimeError::checkpoint(format!("failed to write checkpoint data: {e}"))
        })?;
        writer.finish()?;
        Ok(())
    }

    /// Cleans up old checkpoints, keeping only the most recent `keep_last_n`.
    fn cleanup_old_checkpoints(&self, name: &str) -> Result<()> {
        let entries = self.storage.list(&self.config.checkpoint_dir)?;

        // Filter to only checkpoints matching this name
        let prefix = format!("{name}_");
        let suffix = ".ckpt";
        let mut matching: Vec<_> = entries
            .into_iter()
            .filter(|e| e.starts_with(&prefix) && e.ends_with(suffix) && !e.starts_with('.'))
            .collect();

        // Sort by name (which includes timestamp, so this gives chronological order)
        matching.sort();

        // If we have more than keep_last_n, delete the oldest
        if matching.len() > self.config.keep_last_n {
            let to_delete = matching.len() - self.config.keep_last_n;
            for filename in matching.iter().take(to_delete) {
                let path = self.config.checkpoint_dir.join(filename);
                self.storage.delete(&path)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StorageConfig;
    use crate::storage::LocalStorage;
    use std::path::Path;
    use tempfile::TempDir;

    fn create_test_writer() -> (CheckpointWriter, TempDir) {
        let temp_dir = TempDir::new().unwrap();

        let storage_config = StorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let storage = Arc::new(LocalStorage::new(&storage_config).unwrap());

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: PathBuf::from("checkpoints"),
            compression: "none".to_string(),
            compression_level: 1,
            keep_last_n: 3,
            atomic_writes: true,
        };

        let writer = CheckpointWriter::new(storage, checkpoint_config);
        (writer, temp_dir)
    }

    #[test]
    fn test_write_creates_file() {
        let (writer, temp_dir) = create_test_writer();
        let data = b"test checkpoint data";

        let path = writer.write("test", data).unwrap();

        assert!(path.exists() || temp_dir.path().join(&path).exists());
    }

    #[test]
    fn test_timestamped_filename() {
        let (writer, _temp) = create_test_writer();

        let filename = writer.generate_filename("model");

        assert!(filename.starts_with("model_"));
        assert!(filename.ends_with(".ckpt"));
    }

    #[test]
    fn test_checksum_calculation() {
        let (writer, _temp) = create_test_writer();

        let data = b"hello world";
        let checksum1 = writer.calculate_checksum(data);
        let checksum2 = writer.calculate_checksum(data);

        // Same data should produce same checksum
        assert_eq!(checksum1, checksum2);

        // Different data should produce different checksum
        let checksum3 = writer.calculate_checksum(b"different data");
        assert_ne!(checksum1, checksum3);
    }

    #[test]
    fn test_compression_none() {
        let (writer, _temp) = create_test_writer();

        let data = b"test data that won't be compressed";
        let (compressed, algo) = writer.compress(data).unwrap();

        assert_eq!(algo, "none");
        assert_eq!(compressed, data);
    }

    #[test]
    fn test_compression_lz4() {
        let temp_dir = TempDir::new().unwrap();
        let storage_config = StorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let storage = Arc::new(LocalStorage::new(&storage_config).unwrap());

        let config = CheckpointConfig {
            checkpoint_dir: PathBuf::from("checkpoints"),
            compression: "lz4".to_string(),
            compression_level: 1,
            keep_last_n: 3,
            atomic_writes: true,
        };

        let writer = CheckpointWriter::new(storage, config);

        let data = b"test data for compression";
        let (compressed, algo) = writer.compress(data).unwrap();

        assert_eq!(algo, "lz4");
        // LZ4 with prepended size will be different from original
        assert_ne!(compressed.as_slice(), data.as_slice());
    }

    #[test]
    fn test_compression_zstd() {
        let temp_dir = TempDir::new().unwrap();
        let storage_config = StorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let storage = Arc::new(LocalStorage::new(&storage_config).unwrap());

        let config = CheckpointConfig {
            checkpoint_dir: PathBuf::from("checkpoints"),
            compression: "zstd".to_string(),
            compression_level: 3,
            keep_last_n: 3,
            atomic_writes: true,
        };

        let writer = CheckpointWriter::new(storage, config);

        let data = b"test data for zstd compression";
        let (compressed, algo) = writer.compress(data).unwrap();

        assert_eq!(algo, "zstd");
        // Compressed data should be different from original
        assert_ne!(compressed.as_slice(), data.as_slice());
    }

    #[test]
    fn test_cleanup_old_checkpoints() {
        let temp_dir = TempDir::new().unwrap();
        let storage_config = StorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let storage = Arc::new(LocalStorage::new(&storage_config).unwrap());

        let config = CheckpointConfig {
            checkpoint_dir: PathBuf::from("checkpoints"),
            compression: "none".to_string(),
            compression_level: 1,
            keep_last_n: 2,
            atomic_writes: false,
        };

        let writer = CheckpointWriter::new(storage.clone(), config.clone());
        let data = b"checkpoint data";

        // Write 4 checkpoints
        for _ in 0..4 {
            writer.write("model", data).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(10)); // Ensure different timestamps
        }

        // Should only have 2 left
        let entries = storage.list(Path::new("checkpoints")).unwrap();
        let checkpoint_count = entries
            .iter()
            .filter(|e| e.starts_with("model_") && e.ends_with(".ckpt"))
            .count();

        assert_eq!(checkpoint_count, 2);
    }

    #[test]
    fn test_atomic_write() {
        let temp_dir = TempDir::new().unwrap();
        let storage_config = StorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let storage = Arc::new(LocalStorage::new(&storage_config).unwrap());

        let config = CheckpointConfig {
            checkpoint_dir: PathBuf::from("checkpoints"),
            compression: "none".to_string(),
            compression_level: 1,
            keep_last_n: 3,
            atomic_writes: true,
        };

        let writer = CheckpointWriter::new(storage.clone(), config);
        let data = b"atomic write test";

        let path = writer.write("atomic", data).unwrap();

        // File should exist at final path
        assert!(storage.exists(&path).unwrap());

        // Temp file should not exist
        let temp_path = path.with_file_name(format!(
            ".{}.tmp",
            path.file_name().unwrap().to_str().unwrap()
        ));
        assert!(!storage.exists(&temp_path).unwrap());
    }
}
