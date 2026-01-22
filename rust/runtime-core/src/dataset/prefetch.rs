// rust/runtime-core/src/dataset/prefetch.rs

//! Prefetching iterator for reducing I/O stalls during training.
//!
//! This module provides a prefetching wrapper around the `ShardIterator` that
//! loads batches in the background using a separate thread, allowing the main
//! training loop to continue without waiting for I/O.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam::queue::ArrayQueue;

use super::traits::{Batch, RecordFormat, ShardSpec};
use super::iterator::{IteratorConfig, ShardIterator};
use crate::error::{Result, RuntimeError};
use crate::storage::StorageBackend;

/// Configuration for prefetching behavior.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Number of batches to prefetch.
    pub buffer_size: usize,
    /// Whether prefetching is enabled.
    pub enabled: bool,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            buffer_size: 4,
            enabled: true,
        }
    }
}

/// Result type for prefetched items.
type PrefetchItem = Result<Batch>;

/// A prefetching iterator that loads batches in a background thread.
///
/// This iterator wraps a `ShardIterator` and uses a background thread to
/// prefetch batches into a bounded queue, reducing I/O stalls during training.
pub struct PrefetchingIterator {
    /// Queue for receiving prefetched batches.
    queue: Arc<ArrayQueue<PrefetchItem>>,
    /// Flag to signal the prefetch thread to stop.
    stop_flag: Arc<AtomicBool>,
    /// Handle to the prefetch thread.
    prefetch_thread: Option<JoinHandle<()>>,
    /// Whether prefetching is enabled.
    enabled: bool,
    /// Fallback iterator when prefetching is disabled.
    fallback_iter: Option<ShardIterator>,
    /// Track if we've seen the end.
    exhausted: bool,
}

impl PrefetchingIterator {
    /// Creates a new prefetching iterator.
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend to read from.
    /// * `path` - Path to the data file.
    /// * `shard` - The shard specification.
    /// * `format` - The record format.
    /// * `iter_config` - Iterator configuration.
    /// * `prefetch_config` - Prefetching configuration.
    pub fn new(
        storage: Arc<dyn StorageBackend>,
        path: PathBuf,
        shard: ShardSpec,
        format: Arc<dyn RecordFormat>,
        iter_config: IteratorConfig,
        prefetch_config: PrefetchConfig,
    ) -> Self {
        if !prefetch_config.enabled || prefetch_config.buffer_size == 0 {
            // Prefetching disabled, use direct iteration
            let iter = ShardIterator::new(storage, path, shard, format, iter_config);
            return Self {
                queue: Arc::new(ArrayQueue::new(1)),
                stop_flag: Arc::new(AtomicBool::new(false)),
                prefetch_thread: None,
                enabled: false,
                fallback_iter: Some(iter),
                exhausted: false,
            };
        }

        let queue = Arc::new(ArrayQueue::new(prefetch_config.buffer_size));
        let stop_flag = Arc::new(AtomicBool::new(false));

        // Clone references for the background thread
        let queue_clone = queue.clone();
        let stop_flag_clone = stop_flag.clone();

        // Spawn prefetch thread
        let prefetch_thread = thread::spawn(move || {
            let mut iter = ShardIterator::new(storage, path, shard, format, iter_config);

            loop {
                // Check if we should stop
                if stop_flag_clone.load(Ordering::Relaxed) {
                    break;
                }

                // Try to fetch the next batch
                match iter.next_batch() {
                    Ok(Some(batch)) => {
                        // Try to push to queue, blocking until space is available
                        loop {
                            if stop_flag_clone.load(Ordering::Relaxed) {
                                return;
                            }

                            match queue_clone.push(Ok(batch.clone())) {
                                Ok(()) => break,
                                Err(_) => {
                                    // Queue is full, wait a bit
                                    thread::sleep(std::time::Duration::from_micros(100));
                                }
                            }
                        }
                    }
                    Ok(None) => {
                        // End of iterator, push a sentinel (we use a special error)
                        // Actually, let's just break - the consumer will know it's done
                        // when it can't get more items
                        break;
                    }
                    Err(e) => {
                        // Push error to queue
                        let _ = queue_clone.push(Err(e));
                        break;
                    }
                }
            }
        });

        Self {
            queue,
            stop_flag,
            prefetch_thread: Some(prefetch_thread),
            enabled: true,
            fallback_iter: None,
            exhausted: false,
        }
    }

    /// Gets the next batch from the iterator.
    pub fn next_batch(&mut self) -> Result<Option<Batch>> {
        if self.exhausted {
            return Ok(None);
        }

        if !self.enabled {
            // Use fallback iterator
            if let Some(ref mut iter) = self.fallback_iter {
                let result = iter.next_batch();
                if matches!(result, Ok(None)) {
                    self.exhausted = true;
                }
                return result;
            }
            return Ok(None);
        }

        // Try to pop from the queue
        let max_attempts = 1000;
        for _ in 0..max_attempts {
            if let Some(item) = self.queue.pop() {
                return match item {
                    Ok(batch) => Ok(Some(batch)),
                    Err(e) => {
                        self.exhausted = true;
                        Err(e)
                    }
                };
            }

            // Check if prefetch thread is still alive
            if let Some(ref thread) = self.prefetch_thread {
                if thread.is_finished() {
                    // Thread finished and queue is empty
                    self.exhausted = true;
                    return Ok(None);
                }
            }

            // Wait a bit before trying again
            thread::sleep(std::time::Duration::from_micros(100));
        }

        // Timeout waiting for prefetch
        Err(RuntimeError::dataset("prefetch", "timeout waiting for prefetched batch"))
    }

    /// Stops the prefetch thread and cleans up resources.
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);

        // Wait for the thread to finish
        if let Some(thread) = self.prefetch_thread.take() {
            let _ = thread.join();
        }
    }

    /// Returns true if prefetching is enabled.
    pub fn is_prefetching_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the number of items currently in the prefetch queue.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }
}

impl Iterator for PrefetchingIterator {
    type Item = Result<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

impl Drop for PrefetchingIterator {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Async prefetching iterator using tokio tasks.
///
/// This is an async version that integrates better with async runtimes.
pub struct AsyncPrefetchingIterator {
    /// Receiver for prefetched batches.
    receiver: tokio::sync::mpsc::Receiver<PrefetchItem>,
    /// Flag to signal stop (through dropping the sender).
    #[allow(dead_code)]
    stop_flag: Arc<AtomicBool>,
    /// Handle to the prefetch task.
    #[allow(dead_code)]
    task_handle: Option<tokio::task::JoinHandle<()>>,
    /// Whether we've seen the end.
    exhausted: bool,
}

impl AsyncPrefetchingIterator {
    /// Creates a new async prefetching iterator.
    ///
    /// Note: This must be called from within a tokio runtime.
    pub fn new(
        storage: Arc<dyn StorageBackend>,
        path: PathBuf,
        shard: ShardSpec,
        format: Arc<dyn RecordFormat>,
        iter_config: IteratorConfig,
        buffer_size: usize,
    ) -> Self {
        let (sender, receiver) = tokio::sync::mpsc::channel(buffer_size);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = stop_flag.clone();

        // Spawn prefetch task
        let task_handle = tokio::task::spawn_blocking(move || {
            let mut iter = ShardIterator::new(storage, path, shard, format, iter_config);

            loop {
                if stop_flag_clone.load(Ordering::Relaxed) {
                    break;
                }

                match iter.next_batch() {
                    Ok(Some(batch)) => {
                        // Use blocking send
                        if sender.blocking_send(Ok(batch)).is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let _ = sender.blocking_send(Err(e));
                        break;
                    }
                }
            }
        });

        Self {
            receiver,
            stop_flag,
            task_handle: Some(task_handle),
            exhausted: false,
        }
    }

    /// Gets the next batch asynchronously.
    pub async fn next_batch(&mut self) -> Result<Option<Batch>> {
        if self.exhausted {
            return Ok(None);
        }

        match self.receiver.recv().await {
            Some(Ok(batch)) => Ok(Some(batch)),
            Some(Err(e)) => {
                self.exhausted = true;
                Err(e)
            }
            None => {
                self.exhausted = true;
                Ok(None)
            }
        }
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

    /// Mock storage reader
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
    fn test_prefetching_iterator_basic() {
        let storage = Arc::new(MockStorage::new());
        let data = b"line1\nline2\nline3\nline4\nline5\n".to_vec();
        storage.add_file("test.txt", data.clone());

        let shard = ShardSpec {
            shard_id: 0,
            total_shards: 1,
            byte_start: 0,
            byte_end: data.len() as u64,
        };

        let format = Arc::new(NewlineDelimitedFormat::new());
        let iter_config = IteratorConfig {
            batch_size: 12,
            shard_id: 0,
        };
        let prefetch_config = PrefetchConfig {
            buffer_size: 2,
            enabled: true,
        };

        let mut iter = PrefetchingIterator::new(
            storage,
            PathBuf::from("test.txt"),
            shard,
            format,
            iter_config,
            prefetch_config,
        );

        let mut batches = vec![];
        while let Some(batch) = iter.next() {
            batches.push(batch.unwrap());
        }

        assert!(!batches.is_empty());
    }

    #[test]
    fn test_prefetching_disabled() {
        let storage = Arc::new(MockStorage::new());
        let data = b"line1\nline2\n".to_vec();
        storage.add_file("test.txt", data.clone());

        let shard = ShardSpec {
            shard_id: 0,
            total_shards: 1,
            byte_start: 0,
            byte_end: data.len() as u64,
        };

        let format = Arc::new(NewlineDelimitedFormat::new());
        let iter_config = IteratorConfig::default();
        let prefetch_config = PrefetchConfig {
            buffer_size: 2,
            enabled: false,
        };

        let iter = PrefetchingIterator::new(
            storage,
            PathBuf::from("test.txt"),
            shard,
            format,
            iter_config,
            prefetch_config,
        );

        assert!(!iter.is_prefetching_enabled());

        let batches: Vec<_> = iter.collect();
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_prefetch_config_default() {
        let config = PrefetchConfig::default();
        assert_eq!(config.buffer_size, 4);
        assert!(config.enabled);
    }
}
