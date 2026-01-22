// rust/runtime-core/src/async_runtime.rs

//! Async runtime orchestration for distributed training.
//!
//! This module provides the `AsyncRuntime` struct that ties together all async
//! components: async storage backends, async checkpoints, and prefetching iterators.
//!
//! # Example
//!
//! ```no_run
//! use runtime_core::async_runtime::AsyncRuntime;
//! use runtime_core::config::RuntimeConfig;
//!
//! # async fn example() -> runtime_core::Result<()> {
//! // Create async runtime with default configuration
//! let runtime = AsyncRuntime::new().await?;
//!
//! // Register a dataset
//! let dataset = runtime.register_dataset("data.jsonl", 4, "newline").await?;
//! println!("Dataset has {} shards", dataset.num_shards());
//!
//! // Save a checkpoint (using V2 streaming format)
//! let path = runtime.save_checkpoint("model", b"training state").await?;
//!
//! // Load it back
//! let data = runtime.load_checkpoint(&path).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # S3 Backend (feature-gated)
//!
//! ```no_run,ignore
//! use runtime_core::async_runtime::AsyncRuntime;
//! use runtime_core::config::{RuntimeConfig, StorageBackendType};
//!
//! # async fn example() -> runtime_core::Result<()> {
//! // Create runtime with S3 storage
//! let mut config = RuntimeConfig::default();
//! config.storage.backend = StorageBackendType::S3;
//! config.storage.s3 = Some(Default::default());
//!
//! let runtime = AsyncRuntime::from_config(config).await?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::checkpoint::{
    AsyncCheckpointReader, AsyncCheckpointWriter, CheckpointMetadata,
};
use crate::config::{RuntimeConfig, StorageBackendType};
use crate::dataset::{
    calculate_shards, FixedSizeFormat, IteratorConfig, LengthPrefixedFormat,
    NewlineDelimitedFormat, PrefetchConfig, PrefetchingIterator, RecordFormat, ShardIterator,
    ShardSpec,
};
use crate::error::{Result, RuntimeError};
use crate::storage::{AsyncLocalStorage, AsyncStorageBackend, LocalStorage, StorageBackend};

#[cfg(feature = "s3")]
use crate::storage::S3Storage;

#[cfg(feature = "coordinator")]
use crate::coordinator::{
    CoordinatorClient, GrpcCoordinatorClient, WorkerCapabilities,
    protocol::{ShardRange, WorkerConfig, WorkerStatus},
};
#[cfg(feature = "coordinator")]
use tokio::sync::RwLock;

/// The async runtime that orchestrates all async components.
///
/// The `AsyncRuntime` owns both sync and async storage backends and provides
/// async methods for:
/// - Registering and managing datasets
/// - Saving and loading checkpoints (using V2 streaming format)
/// - Accessing configuration
///
/// # Storage Backend Selection
///
/// The runtime automatically selects the appropriate storage backend based on
/// configuration:
/// - `Local`: Uses filesystem storage (default)
/// - `S3`: Uses S3-compatible object storage (requires `s3` feature)
pub struct AsyncRuntime {
    config: RuntimeConfig,
    /// Sync storage backend (for shard iteration)
    sync_storage: Arc<dyn StorageBackend>,
    /// Async storage backend (for checkpoints and async operations)
    async_storage: Arc<dyn AsyncStorageBackend>,
    /// Async checkpoint writer (V2 format)
    checkpoint_writer: AsyncCheckpointWriter,
    /// Async checkpoint reader (supports V1 and V2)
    checkpoint_reader: AsyncCheckpointReader,
    /// Coordinator client for multi-worker coordination (feature-gated)
    #[cfg(feature = "coordinator")]
    coordinator_client: Option<Arc<RwLock<GrpcCoordinatorClient>>>,
    /// Cached shard assignments from coordinator
    #[cfg(feature = "coordinator")]
    shard_assignments: Arc<RwLock<HashMap<String, Vec<ShardRange>>>>,
}

impl AsyncRuntime {
    /// Creates an async runtime with default configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage backend cannot be initialized.
    pub async fn new() -> Result<Self> {
        Self::from_config(RuntimeConfig::default()).await
    }

    /// Creates an async runtime from a configuration file.
    ///
    /// The configuration file should be in TOML format. Environment variable
    /// overrides are applied after loading the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, parsed, or is invalid.
    pub async fn from_config_file(path: impl AsRef<Path>) -> Result<Self> {
        let config = RuntimeConfig::from_file(path)?.with_env_overrides();
        Self::from_config(config).await
    }

    /// Creates an async runtime from a configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage backend cannot be initialized.
    pub async fn from_config(config: RuntimeConfig) -> Result<Self> {
        config.validate()?;

        // Create storage backends based on configuration
        let (sync_storage, async_storage): (Arc<dyn StorageBackend>, Arc<dyn AsyncStorageBackend>) =
            match config.storage.backend {
                StorageBackendType::Local => {
                    let sync = Arc::new(LocalStorage::new(&config.storage)?);
                    let async_local = AsyncLocalStorage::new(&config.storage).await?;
                    (sync, Arc::new(async_local))
                }
                #[cfg(feature = "s3")]
                StorageBackendType::S3 => {
                    let s3_config = config.storage.s3.as_ref().ok_or_else(|| {
                        RuntimeError::config("S3 backend selected but no S3 configuration provided")
                    })?;

                    // S3 storage is async-only; create a wrapper for sync operations
                    // Don't use "." as S3 prefix - it would be URL-encoded as %2E
                    let base_prefix = config.storage.base_path.to_string_lossy().to_string();
                    let base_prefix = if base_prefix == "." || base_prefix.is_empty() {
                        String::new()
                    } else {
                        base_prefix
                    };
                    let s3_storage = S3Storage::new(s3_config, base_prefix)?;
                    let s3_arc: Arc<dyn AsyncStorageBackend> = Arc::new(s3_storage);

                    // For sync operations with S3, we need a local cache or error
                    // For now, create a local storage for sync operations (dataset iteration)
                    // In a full implementation, this would be an S3-backed sync wrapper
                    let sync = Arc::new(LocalStorage::new(&config.storage)?);

                    (sync, s3_arc)
                }
                #[cfg(not(feature = "s3"))]
                StorageBackendType::S3 => {
                    return Err(RuntimeError::config(
                        "S3 backend requested but 's3' feature is not enabled. \
                        Compile with --features s3 to enable S3 support.",
                    ));
                }
            };

        let checkpoint_writer =
            AsyncCheckpointWriter::new(async_storage.clone(), config.checkpoint.clone());
        let checkpoint_reader = AsyncCheckpointReader::new(async_storage.clone());

        Ok(Self {
            config,
            sync_storage,
            async_storage,
            checkpoint_writer,
            checkpoint_reader,
            #[cfg(feature = "coordinator")]
            coordinator_client: None,
            #[cfg(feature = "coordinator")]
            shard_assignments: Arc::new(RwLock::new(HashMap::new())),
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
    /// An `AsyncDataset` that can be used to iterate over shards.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file does not exist
    /// - The format string is invalid
    /// - Shard calculation fails
    pub async fn register_dataset(
        &self,
        path: impl AsRef<Path>,
        shard_count: u32,
        format: &str,
    ) -> Result<AsyncDataset> {
        let path = path.as_ref();
        let format_impl = parse_format(format)?;

        // Open the file to calculate shards (uses sync storage for now)
        let mut reader = self.sync_storage.open_read(path)?;
        let shards = calculate_shards(&mut *reader, shard_count, format_impl.as_ref())?;

        Ok(AsyncDataset {
            sync_storage: self.sync_storage.clone(),
            path: path.to_path_buf(),
            shards,
            format: format_impl,
            default_prefetch_config: PrefetchConfig::default(),
        })
    }

    /// Saves a checkpoint using the V2 streaming format.
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
    pub async fn save_checkpoint(&self, name: &str, data: &[u8]) -> Result<PathBuf> {
        self.checkpoint_writer.write(name, data).await
    }

    /// Saves a checkpoint with custom metadata.
    ///
    /// # Arguments
    ///
    /// * `name` - Base name for the checkpoint
    /// * `data` - Raw data to checkpoint
    /// * `metadata` - Custom metadata to include in the checkpoint
    ///
    /// # Returns
    ///
    /// The path to the saved checkpoint file.
    pub async fn save_checkpoint_with_metadata(
        &self,
        name: &str,
        data: &[u8],
        metadata: HashMap<String, String>,
    ) -> Result<PathBuf> {
        self.checkpoint_writer
            .write_with_metadata(name, data, metadata)
            .await
    }

    /// Loads a checkpoint.
    ///
    /// The checkpoint is decompressed and its integrity is verified.
    /// Supports both V1 and V2 checkpoint formats.
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
    pub async fn load_checkpoint(&self, path: impl AsRef<Path>) -> Result<Vec<u8>> {
        self.checkpoint_reader.read(path.as_ref()).await
    }

    /// Reads checkpoint metadata without loading the full data.
    ///
    /// This is useful for inspecting checkpoint information (format version,
    /// compression, size, user metadata) before deciding whether to load it.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the checkpoint file
    ///
    /// # Returns
    ///
    /// Metadata about the checkpoint.
    pub async fn read_checkpoint_metadata(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<CheckpointMetadata> {
        self.checkpoint_reader.read_metadata(path.as_ref()).await
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Returns a reference to the sync storage backend.
    pub fn sync_storage(&self) -> &Arc<dyn StorageBackend> {
        &self.sync_storage
    }

    /// Returns a reference to the async storage backend.
    pub fn async_storage(&self) -> &Arc<dyn AsyncStorageBackend> {
        &self.async_storage
    }

    /// Connect to the coordinator service.
    ///
    /// This method initializes the gRPC client and registers this worker with
    /// the coordinator. After successful connection, the runtime will use
    /// coordinator-assigned shards for dataset iteration.
    ///
    /// # Arguments
    ///
    /// * `capabilities` - Worker capabilities for assignment decisions
    ///
    /// # Returns
    ///
    /// The worker configuration assigned by the coordinator.
    ///
    /// # Errors
    ///
    /// Returns an error if coordination is not enabled or connection fails.
    #[cfg(feature = "coordinator")]
    pub async fn connect_coordinator(
        &mut self,
        capabilities: WorkerCapabilities,
    ) -> Result<WorkerConfig> {
        let coord_config = self
            .config
            .coordinator
            .as_ref()
            .ok_or_else(|| RuntimeError::coordinator("coordinator not configured"))?;

        if !coord_config.enabled {
            return Err(RuntimeError::coordinator("coordination is not enabled"));
        }

        let mut client = GrpcCoordinatorClient::new(coord_config.clone());
        client.connect_with_retry().await?;
        let worker_config = client.register(capabilities).await?;

        self.coordinator_client = Some(Arc::new(RwLock::new(client)));

        Ok(worker_config)
    }

    /// Check if connected to coordinator.
    #[cfg(feature = "coordinator")]
    pub fn is_coordinated(&self) -> bool {
        self.coordinator_client.is_some()
    }

    /// Get the worker configuration from coordinator.
    ///
    /// Returns None if not connected to coordinator.
    #[cfg(feature = "coordinator")]
    pub async fn worker_config(&self) -> Option<WorkerConfig> {
        if let Some(client) = &self.coordinator_client {
            let client = client.read().await;
            client.worker_config().cloned()
        } else {
            None
        }
    }

    /// Get the worker index (0-based) from coordinator.
    ///
    /// Returns None if not connected to coordinator.
    #[cfg(feature = "coordinator")]
    pub async fn worker_index(&self) -> Option<u32> {
        self.worker_config().await.map(|c| c.worker_index)
    }

    /// Get the total number of workers from coordinator.
    ///
    /// Returns None if not connected to coordinator.
    #[cfg(feature = "coordinator")]
    pub async fn total_workers(&self) -> Option<u32> {
        self.worker_config().await.map(|c| c.total_workers)
    }

    /// Get assigned shards for a dataset from the coordinator.
    ///
    /// # Arguments
    ///
    /// * `dataset_id` - Dataset identifier
    /// * `total_shards` - Total number of shards in the dataset
    ///
    /// # Returns
    ///
    /// Vector of shard IDs assigned to this worker.
    ///
    /// # Errors
    ///
    /// Returns an error if not connected to coordinator.
    #[cfg(feature = "coordinator")]
    pub async fn assigned_shards(&self, dataset_id: &str, total_shards: u32) -> Result<Vec<u32>> {
        // Check cache first
        {
            let cache = self.shard_assignments.read().await;
            if let Some(ranges) = cache.get(dataset_id) {
                return Ok(ranges.iter().flat_map(|r| r.iter()).collect());
            }
        }

        // Fetch from coordinator
        let client = self
            .coordinator_client
            .as_ref()
            .ok_or_else(|| RuntimeError::coordinator("not connected to coordinator"))?;

        let assignment = {
            let client = client.read().await;
            client.get_shard_assignment(dataset_id, total_shards).await?
        };

        // Cache the assignment
        {
            let mut cache = self.shard_assignments.write().await;
            cache.insert(dataset_id.to_string(), assignment.ranges.clone());
        }

        Ok(assignment.ranges.iter().flat_map(|r| r.iter()).collect())
    }

    /// Clear cached shard assignments.
    ///
    /// Call this when the coordinator signals to refresh assignments.
    #[cfg(feature = "coordinator")]
    pub async fn clear_shard_assignment_cache(&self) {
        let mut cache = self.shard_assignments.write().await;
        cache.clear();
    }

    /// Send heartbeat to coordinator.
    ///
    /// # Arguments
    ///
    /// * `status` - Current worker status
    /// * `step` - Current training step
    ///
    /// # Returns
    ///
    /// The heartbeat response containing any commands from the coordinator.
    #[cfg(feature = "coordinator")]
    pub async fn heartbeat(
        &self,
        status: WorkerStatus,
        step: u64,
    ) -> Result<crate::coordinator::protocol::HeartbeatResponse> {
        let client = self
            .coordinator_client
            .as_ref()
            .ok_or_else(|| RuntimeError::coordinator("not connected to coordinator"))?;

        let client = client.read().await;
        client.heartbeat(status, step).await
    }

    /// Unregister from coordinator (graceful shutdown).
    ///
    /// # Arguments
    ///
    /// * `reason` - Reason for unregistration
    #[cfg(feature = "coordinator")]
    pub async fn unregister(&self, reason: &str) -> Result<()> {
        let client = self
            .coordinator_client
            .as_ref()
            .ok_or_else(|| RuntimeError::coordinator("not connected to coordinator"))?;

        let client = client.read().await;
        client.unregister(reason).await
    }

    /// Get access to the coordinator client for advanced operations.
    #[cfg(feature = "coordinator")]
    pub fn coordinator_client(&self) -> Option<&Arc<RwLock<GrpcCoordinatorClient>>> {
        self.coordinator_client.as_ref()
    }
}

/// A registered dataset with pre-calculated shard boundaries and async support.
///
/// Datasets are created through `AsyncRuntime::register_dataset` and provide
/// methods for inspecting shard information and creating iterators with
/// optional prefetching.
pub struct AsyncDataset {
    sync_storage: Arc<dyn StorageBackend>,
    path: PathBuf,
    shards: Vec<ShardSpec>,
    format: Arc<dyn RecordFormat>,
    default_prefetch_config: PrefetchConfig,
}

impl std::fmt::Debug for AsyncDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncDataset")
            .field("path", &self.path)
            .field("num_shards", &self.shards.len())
            .field("format", &self.format.name())
            .finish()
    }
}

impl AsyncDataset {
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

    /// Creates an iterator for a specific shard without prefetching.
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
            self.sync_storage.clone(),
            self.path.clone(),
            shard,
            self.format.clone(),
            config,
        ))
    }

    /// Creates a prefetching iterator for a specific shard.
    ///
    /// The prefetching iterator loads batches in a background thread to reduce
    /// I/O stalls during training.
    ///
    /// # Arguments
    ///
    /// * `shard_id` - The shard to iterate over
    /// * `batch_size` - Number of bytes per batch
    /// * `prefetch_config` - Configuration for prefetching behavior
    ///
    /// # Returns
    ///
    /// A `PrefetchingIterator` that yields batches with background I/O.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard ID is out of range.
    pub fn iter_shard_prefetch(
        &self,
        shard_id: u32,
        batch_size: usize,
        prefetch_config: PrefetchConfig,
    ) -> Result<PrefetchingIterator> {
        let shard = self.shard_info(shard_id)?.clone();

        let iter_config = IteratorConfig {
            batch_size,
            shard_id,
        };

        Ok(PrefetchingIterator::new(
            self.sync_storage.clone(),
            self.path.clone(),
            shard,
            self.format.clone(),
            iter_config,
            prefetch_config,
        ))
    }

    /// Creates a prefetching iterator with default prefetch configuration.
    ///
    /// Uses the default prefetch settings: buffer_size=4, enabled=true.
    ///
    /// # Arguments
    ///
    /// * `shard_id` - The shard to iterate over
    /// * `batch_size` - Number of bytes per batch
    ///
    /// # Returns
    ///
    /// A `PrefetchingIterator` that yields batches with background I/O.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard ID is out of range.
    pub fn iter_shard_with_prefetch(
        &self,
        shard_id: u32,
        batch_size: usize,
    ) -> Result<PrefetchingIterator> {
        self.iter_shard_prefetch(shard_id, batch_size, self.default_prefetch_config.clone())
    }

    /// Sets the default prefetch configuration for this dataset.
    pub fn set_default_prefetch_config(&mut self, config: PrefetchConfig) {
        self.default_prefetch_config = config;
    }

    /// Returns the path to the dataset file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns all shard specifications.
    pub fn shards(&self) -> &[ShardSpec] {
        &self.shards
    }
}

/// Conversion from AsyncDataset to sync Dataset.
///
/// This allows using AsyncDataset with code that expects a sync Dataset.
/// The conversion preserves the sync storage backend used for iteration.
impl From<AsyncDataset> for crate::runtime::Dataset {
    fn from(async_dataset: AsyncDataset) -> Self {
        crate::runtime::Dataset::from_parts(
            async_dataset.sync_storage,
            async_dataset.path,
            async_dataset.shards,
            async_dataset.format,
        )
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
    use tempfile::TempDir;

    async fn create_test_runtime() -> (AsyncRuntime, TempDir) {
        let temp_dir = TempDir::new().unwrap();

        let mut config = RuntimeConfig::default();
        config.storage.base_path = temp_dir.path().to_path_buf();
        config.checkpoint.checkpoint_dir = PathBuf::from("checkpoints");

        let runtime = AsyncRuntime::from_config(config).await.unwrap();
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

    #[tokio::test]
    async fn test_async_runtime_new() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = RuntimeConfig::default();
        config.storage.base_path = temp_dir.path().to_path_buf();

        let result = AsyncRuntime::from_config(config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_register_dataset() {
        let (runtime, temp_dir) = create_test_runtime().await;

        let content = b"line1\nline2\nline3\nline4\n";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let dataset = runtime.register_dataset(&path, 2, "newline").await.unwrap();

        assert!(dataset.num_shards() > 0);
        assert_eq!(dataset.total_bytes(), content.len() as u64);
    }

    #[tokio::test]
    async fn test_dataset_iter_shard() {
        let (runtime, temp_dir) = create_test_runtime().await;

        let content = b"line1\nline2\nline3\n";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let dataset = runtime.register_dataset(&path, 1, "newline").await.unwrap();

        let mut iter = dataset.iter_shard(0, 1024).unwrap();

        let mut all_data = Vec::new();
        while let Some(batch) = iter.next() {
            all_data.extend_from_slice(&batch.unwrap().data);
        }

        assert_eq!(all_data, content);
    }

    #[tokio::test]
    async fn test_dataset_iter_with_prefetch() {
        let (runtime, temp_dir) = create_test_runtime().await;

        let content = b"line1\nline2\nline3\nline4\nline5\n";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let dataset = runtime.register_dataset(&path, 1, "newline").await.unwrap();

        let mut iter = dataset.iter_shard_with_prefetch(0, 1024).unwrap();

        let mut all_data = Vec::new();
        while let Some(batch) = iter.next() {
            all_data.extend_from_slice(&batch.unwrap().data);
        }

        assert_eq!(all_data, content);
    }

    #[tokio::test]
    async fn test_save_load_checkpoint() {
        let (runtime, _temp) = create_test_runtime().await;

        let original_data = b"checkpoint data for testing";
        let path = runtime.save_checkpoint("test", original_data).await.unwrap();

        let loaded_data = runtime.load_checkpoint(&path).await.unwrap();

        assert_eq!(loaded_data, original_data);
    }

    #[tokio::test]
    async fn test_checkpoint_with_metadata() {
        let (runtime, _temp) = create_test_runtime().await;

        let mut metadata = HashMap::new();
        metadata.insert("epoch".to_string(), "42".to_string());
        metadata.insert("step".to_string(), "1000".to_string());

        let original_data = b"checkpoint with metadata";
        let path = runtime
            .save_checkpoint_with_metadata("test", original_data, metadata.clone())
            .await
            .unwrap();

        // Read metadata
        let ckpt_metadata = runtime.read_checkpoint_metadata(&path).await.unwrap();
        assert_eq!(ckpt_metadata.metadata.get("epoch"), Some(&"42".to_string()));
        assert_eq!(ckpt_metadata.metadata.get("step"), Some(&"1000".to_string()));

        // Read data
        let loaded_data = runtime.load_checkpoint(&path).await.unwrap();
        assert_eq!(loaded_data, original_data);
    }

    #[tokio::test]
    async fn test_invalid_shard_id() {
        let (runtime, temp_dir) = create_test_runtime().await;

        let content = b"line1\nline2\n";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let dataset = runtime.register_dataset(&path, 2, "newline").await.unwrap();

        let result = dataset.shard_info(100);
        assert!(result.is_err());

        let result = dataset.iter_shard(100, 1024);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_invalid_format() {
        let (runtime, temp_dir) = create_test_runtime().await;

        let content = b"some data";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let result = runtime
            .register_dataset(&path, 1, "unknown-format")
            .await;
        assert!(result.is_err());

        let err = result.unwrap_err().to_string();
        assert!(err.contains("unknown record format"));
    }

    #[tokio::test]
    async fn test_config_access() {
        let (runtime, _temp) = create_test_runtime().await;

        // Verify config is accessible
        assert!(runtime.config().storage.buffer_size > 0);
    }

    #[tokio::test]
    async fn test_storage_access() {
        let (runtime, temp_dir) = create_test_runtime().await;

        // Create a file
        let content = b"test content";
        create_test_file(&temp_dir, "test.txt", content);

        // Access sync storage directly
        let storage = runtime.sync_storage();
        assert!(storage.exists(Path::new("test.txt")).unwrap());

        // Access async storage
        let async_storage = runtime.async_storage();
        assert!(async_storage.exists(Path::new("test.txt")).await.unwrap());
    }

    #[tokio::test]
    async fn test_prefetch_config_custom() {
        let (runtime, temp_dir) = create_test_runtime().await;

        let content = b"line1\nline2\nline3\n";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let dataset = runtime.register_dataset(&path, 1, "newline").await.unwrap();

        let prefetch_config = PrefetchConfig {
            buffer_size: 8,
            enabled: true,
        };

        let mut iter = dataset
            .iter_shard_prefetch(0, 1024, prefetch_config)
            .unwrap();

        let mut all_data = Vec::new();
        while let Some(batch) = iter.next() {
            all_data.extend_from_slice(&batch.unwrap().data);
        }

        assert_eq!(all_data, content);
    }

    #[tokio::test]
    async fn test_prefetch_disabled() {
        let (runtime, temp_dir) = create_test_runtime().await;

        let content = b"line1\nline2\n";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let dataset = runtime.register_dataset(&path, 1, "newline").await.unwrap();

        let prefetch_config = PrefetchConfig {
            buffer_size: 4,
            enabled: false,
        };

        let iter = dataset
            .iter_shard_prefetch(0, 1024, prefetch_config)
            .unwrap();

        assert!(!iter.is_prefetching_enabled());
    }

    #[tokio::test]
    async fn test_dataset_debug() {
        let (runtime, temp_dir) = create_test_runtime().await;

        let content = b"data";
        let path = create_test_file(&temp_dir, "data.txt", content);

        let dataset = runtime.register_dataset(&path, 1, "newline").await.unwrap();

        let debug_str = format!("{:?}", dataset);
        assert!(debug_str.contains("AsyncDataset"));
        assert!(debug_str.contains("data.txt"));
    }

    #[test]
    fn test_parse_format_fixed() {
        let format = parse_format("fixed:100").unwrap();
        assert_eq!(format.name(), "fixed-size");
    }

    #[test]
    fn test_parse_format_newline() {
        let format = parse_format("newline").unwrap();
        assert_eq!(format.name(), "newline-delimited");
    }

    #[test]
    fn test_parse_format_length_prefixed() {
        let format = parse_format("length-prefixed").unwrap();
        assert_eq!(format.name(), "length-prefixed");
    }

    #[test]
    fn test_parse_format_invalid() {
        assert!(parse_format("invalid").is_err());
        assert!(parse_format("fixed:0").is_err());
        assert!(parse_format("fixed:abc").is_err());
    }
}
