// rust/runtime-core/src/dataset/parallel.rs

//! Parallel shard loading for multi-worker training.
//!
//! This module provides utilities for loading multiple shards concurrently,
//! which is useful for data-parallel training where each worker processes
//! different shards simultaneously.

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::mpsc;

use super::traits::{Batch, RecordFormat, ShardSpec};
use super::iterator::{IteratorConfig, ShardIterator};
use crate::error::{Result, RuntimeError};
use crate::storage::StorageBackend;

/// Configuration for parallel shard loading.
#[derive(Debug, Clone)]
pub struct ParallelLoadConfig {
    /// Number of shards to load in parallel.
    pub parallelism: usize,
    /// Channel buffer size for each shard.
    pub channel_buffer: usize,
    /// Whether to shuffle the order of batches across shards.
    pub interleave: bool,
}

impl Default for ParallelLoadConfig {
    fn default() -> Self {
        Self {
            parallelism: 4,
            channel_buffer: 4,
            interleave: true,
        }
    }
}

/// A parallel shard loader that loads multiple shards concurrently.
///
/// This loader spawns multiple tasks to load shards in parallel and
/// combines their output into a single stream of batches.
pub struct ParallelShardLoader {
    /// Configuration for parallel loading.
    config: ParallelLoadConfig,
    /// Storage backend.
    storage: Arc<dyn StorageBackend>,
    /// Path to the data file.
    path: PathBuf,
    /// Record format.
    format: Arc<dyn RecordFormat>,
    /// Iterator configuration.
    iter_config: IteratorConfig,
}

impl ParallelShardLoader {
    /// Creates a new parallel shard loader.
    pub fn new(
        storage: Arc<dyn StorageBackend>,
        path: PathBuf,
        format: Arc<dyn RecordFormat>,
        iter_config: IteratorConfig,
        config: ParallelLoadConfig,
    ) -> Self {
        Self {
            config,
            storage,
            path,
            format,
            iter_config,
        }
    }

    /// Loads batches from multiple shards in parallel.
    ///
    /// Returns a receiver that yields batches from all shards.
    /// Batches may be interleaved from different shards depending on config.
    pub async fn load_shards(
        &self,
        shards: Vec<ShardSpec>,
    ) -> Result<mpsc::Receiver<Result<BatchWithShard>>> {
        let num_shards = shards.len();
        let parallelism = self.config.parallelism.min(num_shards);

        // Create output channel
        let (tx, rx) = mpsc::channel(self.config.channel_buffer * parallelism);

        // Group shards for parallel loading
        let shards_per_worker = num_shards.div_ceil(parallelism);

        for (worker_id, chunk) in shards.chunks(shards_per_worker).enumerate() {
            let tx = tx.clone();
            let storage = self.storage.clone();
            let path = self.path.clone();
            let format = self.format.clone();
            let iter_config = self.iter_config.clone();
            let worker_shards: Vec<_> = chunk.to_vec();

            // Spawn worker task
            tokio::task::spawn_blocking(move || {
                for shard in worker_shards {
                    let mut iter = ShardIterator::new(
                        storage.clone(),
                        path.clone(),
                        shard.clone(),
                        format.clone(),
                        iter_config.clone(),
                    );

                    loop {
                        match iter.next_batch() {
                            Ok(Some(batch)) => {
                                let batch_with_shard = BatchWithShard {
                                    batch,
                                    shard_id: shard.shard_id,
                                    worker_id: worker_id as u32,
                                };
                                if tx.blocking_send(Ok(batch_with_shard)).is_err() {
                                    return; // Receiver dropped
                                }
                            }
                            Ok(None) => break, // Shard exhausted
                            Err(e) => {
                                let _ = tx.blocking_send(Err(e));
                                return;
                            }
                        }
                    }
                }
            });
        }

        Ok(rx)
    }

    /// Loads all batches from the specified shards and collects them.
    ///
    /// This is a convenience method that collects all batches into a vector.
    /// Use `load_shards` for streaming access to avoid memory issues with large datasets.
    pub async fn load_all(&self, shards: Vec<ShardSpec>) -> Result<Vec<BatchWithShard>> {
        let mut rx = self.load_shards(shards).await?;
        let mut batches = Vec::new();

        while let Some(result) = rx.recv().await {
            batches.push(result?);
        }

        Ok(batches)
    }
}

/// A batch with its associated shard information.
#[derive(Debug, Clone)]
pub struct BatchWithShard {
    /// The batch data.
    pub batch: Batch,
    /// The shard ID this batch came from.
    pub shard_id: u32,
    /// The worker ID that loaded this batch.
    pub worker_id: u32,
}

/// A round-robin shard iterator that alternates between multiple shards.
///
/// This is useful for interleaving batches from different shards to ensure
/// diverse training data within each training step.
pub struct RoundRobinShardIterator {
    /// Iterators for each shard.
    iterators: Vec<ShardIterator>,
    /// Current iterator index.
    current_index: usize,
    /// Number of exhausted iterators.
    exhausted_count: usize,
}

impl RoundRobinShardIterator {
    /// Creates a new round-robin iterator over multiple shards.
    pub fn new(
        storage: Arc<dyn StorageBackend>,
        path: PathBuf,
        shards: Vec<ShardSpec>,
        format: Arc<dyn RecordFormat>,
        iter_config: IteratorConfig,
    ) -> Self {
        let iterators: Vec<_> = shards
            .into_iter()
            .enumerate()
            .map(|(i, shard)| {
                let config = IteratorConfig {
                    shard_id: i as u32,
                    ..iter_config.clone()
                };
                ShardIterator::new(
                    storage.clone(),
                    path.clone(),
                    shard,
                    format.clone(),
                    config,
                )
            })
            .collect();

        Self {
            iterators,
            current_index: 0,
            exhausted_count: 0,
        }
    }

    /// Gets the next batch, rotating through shards.
    pub fn next_batch(&mut self) -> Result<Option<BatchWithShard>> {
        if self.exhausted_count >= self.iterators.len() {
            return Ok(None);
        }

        let num_iters = self.iterators.len();
        let start_index = self.current_index;

        loop {
            let iter = &mut self.iterators[self.current_index];
            self.current_index = (self.current_index + 1) % num_iters;

            match iter.next_batch()? {
                Some(batch) => {
                    return Ok(Some(BatchWithShard {
                        shard_id: batch.shard_id,
                        worker_id: 0,
                        batch,
                    }));
                }
                None => {
                    self.exhausted_count += 1;
                    if self.exhausted_count >= num_iters {
                        return Ok(None);
                    }
                    // Skip this iterator and try the next one
                    if self.current_index == start_index {
                        // We've gone full circle without finding a non-exhausted iterator
                        return Ok(None);
                    }
                }
            }
        }
    }

    /// Resets all iterators to their starting positions.
    pub fn reset(&mut self) {
        for iter in &mut self.iterators {
            iter.reset();
        }
        self.current_index = 0;
        self.exhausted_count = 0;
    }

    /// Returns the total number of shards.
    pub fn num_shards(&self) -> usize {
        self.iterators.len()
    }
}

impl Iterator for RoundRobinShardIterator {
    type Item = Result<BatchWithShard>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Builder for creating parallel shard loaders with a fluent API.
pub struct ParallelShardLoaderBuilder {
    storage: Option<Arc<dyn StorageBackend>>,
    path: Option<PathBuf>,
    format: Option<Arc<dyn RecordFormat>>,
    iter_config: IteratorConfig,
    parallel_config: ParallelLoadConfig,
}

impl ParallelShardLoaderBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self {
            storage: None,
            path: None,
            format: None,
            iter_config: IteratorConfig::default(),
            parallel_config: ParallelLoadConfig::default(),
        }
    }

    /// Sets the storage backend.
    pub fn storage(mut self, storage: Arc<dyn StorageBackend>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Sets the data file path.
    pub fn path(mut self, path: PathBuf) -> Self {
        self.path = Some(path);
        self
    }

    /// Sets the record format.
    pub fn format(mut self, format: Arc<dyn RecordFormat>) -> Self {
        self.format = Some(format);
        self
    }

    /// Sets the batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.iter_config.batch_size = size;
        self
    }

    /// Sets the parallelism (number of concurrent shard loaders).
    pub fn parallelism(mut self, n: usize) -> Self {
        self.parallel_config.parallelism = n;
        self
    }

    /// Sets the channel buffer size.
    pub fn channel_buffer(mut self, size: usize) -> Self {
        self.parallel_config.channel_buffer = size;
        self
    }

    /// Enables or disables interleaving.
    pub fn interleave(mut self, enable: bool) -> Self {
        self.parallel_config.interleave = enable;
        self
    }

    /// Builds the parallel shard loader.
    pub fn build(self) -> Result<ParallelShardLoader> {
        let storage = self.storage.ok_or_else(|| {
            RuntimeError::config("storage backend is required")
        })?;
        let path = self.path.ok_or_else(|| {
            RuntimeError::config("path is required")
        })?;
        let format = self.format.ok_or_else(|| {
            RuntimeError::config("format is required")
        })?;

        Ok(ParallelShardLoader::new(
            storage,
            path,
            format,
            self.iter_config,
            self.parallel_config,
        ))
    }
}

impl Default for ParallelShardLoaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::{Cursor, Read, Seek, SeekFrom, Write};
    use std::path::Path;
    use std::sync::Mutex;

    use crate::storage::{ObjectMeta, StorageReader, StorageWriter};
    use super::super::traits::NewlineDelimitedFormat;

    struct MockReader {
        data: Cursor<Vec<u8>>,
        size: u64,
    }

    impl Read for MockReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.data.read(buf)
        }
    }

    impl Seek for MockReader {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            self.data.seek(pos)
        }
    }

    impl StorageReader for MockReader {
        fn size(&self) -> u64 {
            self.size
        }

        fn read_range(&mut self, start: u64, length: usize) -> Result<Vec<u8>> {
            self.data.seek(SeekFrom::Start(start)).unwrap();
            let mut buf = vec![0u8; length];
            let bytes_read = self.data.read(&mut buf).unwrap();
            buf.truncate(bytes_read);
            Ok(buf)
        }
    }

    struct MockWriter;

    impl Write for MockWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl StorageWriter for MockWriter {
        fn finish(self: Box<Self>) -> Result<()> {
            Ok(())
        }
    }

    struct MockStorage {
        files: Mutex<HashMap<PathBuf, Vec<u8>>>,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                files: Mutex::new(HashMap::new()),
            }
        }

        fn add_file(&self, path: impl Into<PathBuf>, data: Vec<u8>) {
            self.files.lock().unwrap().insert(path.into(), data);
        }
    }

    impl StorageBackend for MockStorage {
        fn exists(&self, path: &Path) -> Result<bool> {
            Ok(self.files.lock().unwrap().contains_key(path))
        }

        fn metadata(&self, path: &Path) -> Result<ObjectMeta> {
            let files = self.files.lock().unwrap();
            let data = files.get(path).ok_or_else(|| RuntimeError::storage(path, "not found"))?;
            Ok(ObjectMeta {
                size: data.len() as u64,
                modified: None,
                is_dir: false,
            })
        }

        fn open_read(&self, path: &Path) -> Result<Box<dyn StorageReader>> {
            let files = self.files.lock().unwrap();
            let data = files.get(path).ok_or_else(|| RuntimeError::storage(path, "not found"))?.clone();
            let size = data.len() as u64;
            Ok(Box::new(MockReader {
                data: Cursor::new(data),
                size,
            }))
        }

        fn open_write(&self, _path: &Path) -> Result<Box<dyn StorageWriter>> {
            Ok(Box::new(MockWriter))
        }

        fn delete(&self, _path: &Path) -> Result<()> {
            Ok(())
        }

        fn list(&self, _prefix: &Path) -> Result<Vec<String>> {
            Ok(vec![])
        }

        fn rename(&self, _from: &Path, _to: &Path) -> Result<()> {
            Ok(())
        }

        fn create_dir_all(&self, _path: &Path) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_round_robin_iterator() {
        let storage = Arc::new(MockStorage::new());
        let data = b"line1\nline2\nline3\nline4\n".to_vec();
        storage.add_file("test.txt", data.clone());

        let shards = vec![
            ShardSpec {
                shard_id: 0,
                total_shards: 2,
                byte_start: 0,
                byte_end: 12, // "line1\nline2\n"
            },
            ShardSpec {
                shard_id: 1,
                total_shards: 2,
                byte_start: 12,
                byte_end: 24, // "line3\nline4\n"
            },
        ];

        let format = Arc::new(NewlineDelimitedFormat::new());
        let config = IteratorConfig {
            batch_size: 6, // One line per batch
            shard_id: 0,
        };

        let mut iter = RoundRobinShardIterator::new(
            storage,
            PathBuf::from("test.txt"),
            shards,
            format,
            config,
        );

        let batches: Vec<_> = iter.by_ref().map(|r| r.unwrap()).collect();
        assert!(!batches.is_empty());

        // Verify we got batches from both shards
        let shard_0_count = batches.iter().filter(|b| b.shard_id == 0).count();
        let shard_1_count = batches.iter().filter(|b| b.shard_id == 1).count();
        assert!(shard_0_count > 0);
        assert!(shard_1_count > 0);
    }

    #[test]
    fn test_round_robin_reset() {
        let storage = Arc::new(MockStorage::new());
        let data = b"line1\nline2\n".to_vec();
        storage.add_file("test.txt", data.clone());

        let shards = vec![ShardSpec {
            shard_id: 0,
            total_shards: 1,
            byte_start: 0,
            byte_end: data.len() as u64,
        }];

        let format = Arc::new(NewlineDelimitedFormat::new());
        let config = IteratorConfig::default();

        let mut iter = RoundRobinShardIterator::new(
            storage,
            PathBuf::from("test.txt"),
            shards,
            format,
            config,
        );

        let first_pass: Vec<_> = iter.by_ref().map(|r| r.unwrap()).collect();
        iter.reset();
        let second_pass: Vec<_> = iter.by_ref().map(|r| r.unwrap()).collect();

        assert_eq!(first_pass.len(), second_pass.len());
    }

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelLoadConfig::default();
        assert_eq!(config.parallelism, 4);
        assert_eq!(config.channel_buffer, 4);
        assert!(config.interleave);
    }

    #[test]
    fn test_builder_missing_storage() {
        let result = ParallelShardLoaderBuilder::new()
            .path(PathBuf::from("test.txt"))
            .format(Arc::new(NewlineDelimitedFormat::new()))
            .build();

        assert!(result.is_err());
    }
}
