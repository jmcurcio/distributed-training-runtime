// rust/runtime-core/src/runtime.rs

//! Main runtime orchestration.
//!
//! This module provides the `Runtime` struct that ties together all components
//! of the distributed training runtime: storage, datasets, and checkpoints.
//!
//! # Example
//!
//! ```no_run
//! use runtime_core::Runtime;
//! use std::path::Path;
//!
//! // Create runtime with default configuration
//! let runtime = Runtime::new().unwrap();
//!
//! // Register a dataset
//! let dataset = runtime.register_dataset("data.jsonl", 4, "newline").unwrap();
//! println!("Dataset has {} shards", dataset.num_shards());
//!
//! // Iterate over a shard
//! let mut iter = dataset.iter_shard(0, 64 * 1024).unwrap();
//! for batch in iter {
//!     let batch = batch.unwrap();
//!     // Process batch.data
//! }
//!
//! // Save a checkpoint
//! let path = runtime.save_checkpoint("model", b"training state").unwrap();
//!
//! // Load it back
//! let data = runtime.load_checkpoint(&path).unwrap();
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::checkpoint::{CheckpointReader, CheckpointWriter};
use crate::config::RuntimeConfig;
use crate::dataset::{
    calculate_shards, FixedSizeFormat, IteratorConfig, LengthPrefixedFormat,
    NewlineDelimitedFormat, RecordFormat, ShardIterator, ShardSpec,
};
use crate::error::{Result, RuntimeError};
use crate::storage::{LocalStorage, StorageBackend};

/// The main runtime that orchestrates all components.
///
/// The `Runtime` owns the storage backend and provides methods for:
/// - Registering and managing datasets
/// - Saving and loading checkpoints
/// - Accessing configuration
pub struct Runtime {
    config: RuntimeConfig,
    storage: Arc<dyn StorageBackend>,
    checkpoint_writer: CheckpointWriter,
    checkpoint_reader: CheckpointReader,
}

impl Runtime {
    /// Creates a runtime with default configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage backend cannot be initialized.
    pub fn new() -> Result<Self> {
        Self::from_config(RuntimeConfig::default())
    }

    /// Creates a runtime from a configuration file.
    ///
    /// The configuration file should be in TOML format. Environment variable
    /// overrides are applied after loading the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, parsed, or is invalid.
    pub fn from_config_file(path: impl AsRef<Path>) -> Result<Self> {
        let config = RuntimeConfig::from_file(path)?.with_env_overrides();
        Self::from_config(config)
    }

    /// Creates a runtime from a configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage backend cannot be initialized.
    pub fn from_config(config: RuntimeConfig) -> Result<Self> {
        config.validate()?;

        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(&config.storage)?);

        let checkpoint_writer = CheckpointWriter::new(storage.clone(), config.checkpoint.clone());
        let checkpoint_reader = CheckpointReader::new(storage.clone());

        Ok(Self {
            config,
            storage,
            checkpoint_writer,
            checkpoint_reader,
        })
    }

    /// Registers a dataset and calculates shard boundaries.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the dataset file (relative to storage base path)
    /// * `shard_count` - Number of shards to divide the dataset into
    /// * `format` - Record format: "fixed:N" (N-byte records), "newline", or "length-prefixed"
    ///
    /// # Returns
    ///
    /// A `Dataset` that can be used to iterate over shards.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file does not exist
    /// - The format string is invalid
    /// - Shard calculation fails
    pub fn register_dataset(
        &self,
        path: impl AsRef<Path>,
        shard_count: u32,
        format: &str,
    ) -> Result<Dataset> {
        let path = path.as_ref();
        let format_impl = parse_format(format)?;

        // Open the file to calculate shards
        let mut reader = self.storage.open_read(path)?;
        let shards = calculate_shards(&mut *reader, shard_count, format_impl.as_ref())?;

        Ok(Dataset {
            storage: self.storage.clone(),
            path: path.to_path_buf(),
            shards,
            format: format_impl,
        })
    }

    /// Saves a checkpoint.
    ///
    /// The checkpoint is compressed according to the configuration and written
    /// atomically. Old checkpoints are cleaned up based on `keep_last_n`.
    ///
    /// # Arguments
    ///
    /// * `name` - Base name for the checkpoint (e.g., "model", "optimizer")
    /// * `data` - Raw data to checkpoint
    ///
    /// # Returns
    ///
    /// The path to the saved checkpoint file.
    pub fn save_checkpoint(&self, name: &str, data: &[u8]) -> Result<PathBuf> {
        self.checkpoint_writer.write(name, data)
    }

    /// Loads a checkpoint.
    ///
    /// The checkpoint is decompressed and its integrity is verified.
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
    /// Returns an error if the checkpoint is corrupted or cannot be read.
    pub fn load_checkpoint(&self, path: impl AsRef<Path>) -> Result<Vec<u8>> {
        self.checkpoint_reader.read(path.as_ref())
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Returns a reference to the storage backend.
    pub fn storage(&self) -> &Arc<dyn StorageBackend> {
        &self.storage
    }
}

/// A registered dataset with pre-calculated shard boundaries.
///
/// Datasets are created through `Runtime::register_dataset` and provide
/// methods for inspecting shard information and creating iterators.
pub struct Dataset {
    storage: Arc<dyn StorageBackend>,
    path: PathBuf,
    shards: Vec<ShardSpec>,
    format: Arc<dyn RecordFormat>,
}

impl Dataset {
    /// Creates a new Dataset from its components.
    ///
    /// This is primarily used internally for converting from AsyncDataset.
    pub fn from_parts(
        storage: Arc<dyn StorageBackend>,
        path: PathBuf,
        shards: Vec<ShardSpec>,
        format: Arc<dyn RecordFormat>,
    ) -> Self {
        Self {
            storage,
            path,
            shards,
            format,
        }
    }
}

impl std::fmt::Debug for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dataset")
            .field("path", &self.path)
            .field("num_shards", &self.shards.len())
            .field("format", &self.format.name())
            .finish()
    }
}

impl Dataset {
    /// Returns the number of shards in this dataset.
    pub fn num_shards(&self) -> u32 {
        self.shards.len() as u32
    }

    /// Returns the total size of the dataset in bytes.
    pub fn total_bytes(&self) -> u64 {
        self.shards.last().map(|s| s.byte_end).unwrap_or(0)
    }

    /// Returns information about a specific shard.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard ID is out of range.
    pub fn shard_info(&self, shard_id: u32) -> Result<&ShardSpec> {
        self.shards
            .get(shard_id as usize)
            .ok_or_else(|| RuntimeError::invalid_shard(shard_id, self.num_shards()))
    }

    /// Creates an iterator for a specific shard.
    ///
    /// # Arguments
    ///
    /// * `shard_id` - The shard to iterate over
    /// * `batch_size` - Number of bytes per batch
    ///
    /// # Returns
    ///
    /// A `ShardIterator` that yields batches of data.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard ID is out of range.
    pub fn iter_shard(&self, shard_id: u32, batch_size: usize) -> Result<ShardIterator> {
        let shard = self.shard_info(shard_id)?.clone();

        let config = IteratorConfig {
            batch_size,
            shard_id,
        };

        Ok(ShardIterator::new(
            self.storage.clone(),
            self.path.clone(),
            shard,
            self.format.clone(),
            config,
        ))
    }

    /// Returns the path to the dataset file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns all shard specifications.
    pub fn shards(&self) -> &[ShardSpec] {
        &self.shards
    }

    /// Returns a reference to the storage backend.
    pub fn storage(&self) -> Arc<dyn StorageBackend> {
        self.storage.clone()
    }

    /// Returns a reference to the record format.
    pub fn format(&self) -> Arc<dyn RecordFormat> {
        self.format.clone()
    }
}

/// Parses a format string into a RecordFormat implementation.
///
/// Supported formats:
/// - "fixed:N" - Fixed-size records of N bytes
/// - "newline" - Newline-delimited records (JSONL, CSV, etc.)
/// - "length-prefixed" - 4-byte big-endian length prefix + data
fn parse_format(format: &str) -> Result<Arc<dyn RecordFormat>> {
    if let Some(size_str) = format.strip_prefix("fixed:") {
        let size: usize = size_str.parse().map_err(|_| {
            RuntimeError::config(format!("invalid fixed record size: '{size_str}'"))
        })?;
        if size == 0 {
            return Err(RuntimeError::config("fixed record size must be > 0"));
        }
        Ok(Arc::new(FixedSizeFormat::new(size)))
    } else {
        match format {
            "newline" => Ok(Arc::new(NewlineDelimitedFormat::new())),
            "length-prefixed" => Ok(Arc::new(LengthPrefixedFormat::new())),
            _ => Err(RuntimeError::config(format!(
                "unknown record format: '{}'. Expected 'fixed:N', 'newline', or 'length-prefixed'",
                format
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_runtime() -> (Runtime, TempDir) {
        let temp_dir = TempDir::new().unwrap();

        let mut config = RuntimeConfig::default();
        config.storage.base_path = temp_dir.path().to_path_buf();
        config.checkpoint.checkpoint_dir = PathBuf::from("checkpoints");

        let runtime = Runtime::from_config(config).unwrap();
        (runtime, temp_dir)
    }

    fn create_test_file(temp_dir: &TempDir, name: &str, content: &[u8]) -> PathBuf {
        let path = temp_dir.path().join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, content).unwrap();
        PathBuf::from(name)
    }

    #[test]
    fn test_runtime_new_default() {
        // This may fail if ./data doesn't exist, but the construction itself should work
        let result = Runtime::new();
        // We don't assert success because it depends on filesystem state
        // Just verify it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_runtime_from_config() {
        let (runtime, _temp) = create_test_runtime();

        // Verify config is accessible
        assert!(runtime.config().storage.buffer_size > 0);
    }

    #[test]
    fn test_register_dataset() {
        let (runtime, temp_dir) = create_test_runtime();

        // Create a test file with newline-delimited data
        let content = b"line1\nline2\nline3\nline4\n";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let dataset = runtime.register_dataset(&path, 2, "newline").unwrap();

        assert!(dataset.num_shards() > 0);
        assert_eq!(dataset.total_bytes(), content.len() as u64);
    }

    #[test]
    fn test_dataset_shard_count() {
        let (runtime, temp_dir) = create_test_runtime();

        // Create a file with fixed-size records (10 bytes each, 10 records = 100 bytes)
        let content = vec![0u8; 100];
        let path = create_test_file(&temp_dir, "data.bin", &content);

        let dataset = runtime.register_dataset(&path, 4, "fixed:10").unwrap();

        // With 100 bytes and 4 requested shards, we should get close to 4
        assert!(dataset.num_shards() >= 1);
        assert!(dataset.num_shards() <= 4);
    }

    #[test]
    fn test_dataset_total_bytes() {
        let (runtime, temp_dir) = create_test_runtime();

        let content = b"test data content\n";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let dataset = runtime.register_dataset(&path, 1, "newline").unwrap();

        assert_eq!(dataset.total_bytes(), content.len() as u64);
    }

    #[test]
    fn test_dataset_iter_shard() {
        let (runtime, temp_dir) = create_test_runtime();

        let content = b"line1\nline2\nline3\n";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let dataset = runtime.register_dataset(&path, 1, "newline").unwrap();

        let mut iter = dataset.iter_shard(0, 1024).unwrap();

        let mut all_data = Vec::new();
        while let Some(batch) = iter.next() {
            all_data.extend_from_slice(&batch.unwrap().data);
        }

        assert_eq!(all_data, content);
    }

    #[test]
    fn test_save_load_checkpoint() {
        let (runtime, _temp) = create_test_runtime();

        let original_data = b"checkpoint data for testing";
        let path = runtime.save_checkpoint("test", original_data).unwrap();

        let loaded_data = runtime.load_checkpoint(&path).unwrap();

        assert_eq!(loaded_data, original_data);
    }

    #[test]
    fn test_invalid_shard_id() {
        let (runtime, temp_dir) = create_test_runtime();

        let content = b"line1\nline2\n";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let dataset = runtime.register_dataset(&path, 2, "newline").unwrap();

        // Try to access a shard that doesn't exist
        let result = dataset.shard_info(100);
        assert!(result.is_err());

        let result = dataset.iter_shard(100, 1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_format() {
        let (runtime, temp_dir) = create_test_runtime();

        let content = b"some data";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let result = runtime.register_dataset(&path, 1, "unknown-format");
        assert!(result.is_err());

        let err = result.unwrap_err().to_string();
        assert!(err.contains("unknown record format"));
    }

    #[test]
    fn test_fixed_format_parsing() {
        let (runtime, temp_dir) = create_test_runtime();

        let content = vec![0u8; 50];
        let path = create_test_file(&temp_dir, "data.bin", &content);

        // Valid fixed format
        let dataset = runtime.register_dataset(&path, 2, "fixed:10").unwrap();
        assert!(dataset.num_shards() > 0);

        // Invalid: not a number
        let result = runtime.register_dataset(&path, 2, "fixed:abc");
        assert!(result.is_err());

        // Invalid: zero size
        let result = runtime.register_dataset(&path, 2, "fixed:0");
        assert!(result.is_err());
    }

    #[test]
    fn test_length_prefixed_format() {
        let (runtime, temp_dir) = create_test_runtime();

        // Create length-prefixed data
        let mut content = Vec::new();
        // Record 1: length=5, data="hello"
        content.extend_from_slice(&5u32.to_be_bytes());
        content.extend_from_slice(b"hello");
        // Record 2: length=5, data="world"
        content.extend_from_slice(&5u32.to_be_bytes());
        content.extend_from_slice(b"world");

        let path = create_test_file(&temp_dir, "data.bin", &content);

        let dataset = runtime
            .register_dataset(&path, 1, "length-prefixed")
            .unwrap();

        let mut iter = dataset.iter_shard(0, 1024).unwrap();
        let mut all_data = Vec::new();
        while let Some(batch) = iter.next() {
            all_data.extend_from_slice(&batch.unwrap().data);
        }

        assert_eq!(all_data, content);
    }

    #[test]
    fn test_config_file_loading() {
        let temp_dir = TempDir::new().unwrap();

        // Create a config file
        let config_content = format!(
            r#"
            [storage]
            base_path = "{}"
            buffer_size = 32768

            [checkpoint]
            compression = "none"
            "#,
            temp_dir.path().display()
        );

        let config_path = temp_dir.path().join("config.toml");
        let mut file = std::fs::File::create(&config_path).unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let runtime = Runtime::from_config_file(&config_path).unwrap();

        assert_eq!(runtime.config().storage.buffer_size, 32768);
        assert_eq!(runtime.config().checkpoint.compression, "none");
    }

    #[test]
    fn test_dataset_path() {
        let (runtime, temp_dir) = create_test_runtime();

        let content = b"data";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let dataset = runtime.register_dataset(&path, 1, "newline").unwrap();

        assert_eq!(dataset.path(), Path::new("data.txt"));
    }

    #[test]
    fn test_dataset_shards() {
        let (runtime, temp_dir) = create_test_runtime();

        let content = b"line1\nline2\nline3\nline4\n";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let dataset = runtime.register_dataset(&path, 2, "newline").unwrap();

        let shards = dataset.shards();
        assert!(!shards.is_empty());

        // Verify shards cover the full file
        assert_eq!(shards[0].byte_start, 0);
        assert_eq!(shards.last().unwrap().byte_end, content.len() as u64);
    }

    #[test]
    fn test_runtime_storage_access() {
        let (runtime, temp_dir) = create_test_runtime();

        // Create a file
        let content = b"test content";
        create_test_file(&temp_dir, "test.txt", content);

        // Access storage directly
        let storage = runtime.storage();
        assert!(storage.exists(Path::new("test.txt")).unwrap());
    }
}
