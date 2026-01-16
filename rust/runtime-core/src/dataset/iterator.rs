// rust/runtime-core/src/dataset/iterator.rs

use std::path::PathBuf;
use std::sync::Arc;

use crate::error::Result;
use crate::storage::StorageBackend;

use super::traits::{Batch, RecordFormat, ShardSpec};

/// Configuration for the shard iterator
#[derive(Debug, Clone)]
pub struct IteratorConfig {
    /// Number of bytes to read per batch
    pub batch_size: usize,
    /// The shard ID this iterator is processing (for metadata)
    pub shard_id: u32,
}

impl Default for IteratorConfig {
    fn default() -> Self {
        Self {
            batch_size: 64 * 1024, // 64KB default
            shard_id: 0,
        }
    }
}

/// An iterator over batches within a shard.
///
/// The `ShardIterator` reads data from a shard in batches, ensuring that
/// each batch ends on a record boundary. This allows for efficient
/// streaming of data while maintaining record integrity.
pub struct ShardIterator {
    storage: Arc<dyn StorageBackend>,
    path: PathBuf,
    shard: ShardSpec,
    format: Arc<dyn RecordFormat>,
    batch_size: usize,
    current_offset: u64,
    batch_index: u64,
}

impl ShardIterator {
    /// Create a new shard iterator.
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend to read from
    /// * `path` - Path to the data file
    /// * `shard` - The shard specification defining the byte range
    /// * `format` - The record format for finding record boundaries
    /// * `config` - Iterator configuration
    pub fn new(
        storage: Arc<dyn StorageBackend>,
        path: PathBuf,
        shard: ShardSpec,
        format: Arc<dyn RecordFormat>,
        config: IteratorConfig,
    ) -> Self {
        Self {
            storage,
            path,
            current_offset: shard.byte_start,
            shard,
            format,
            batch_size: config.batch_size,
            batch_index: 0,
        }
    }

    /// Read the next batch of data from the shard.
    ///
    /// Returns `Ok(Some(Batch))` if data was read, `Ok(None)` if the shard
    /// is exhausted, or an error if reading fails.
    ///
    /// Each batch is guaranteed to end on a record boundary, except possibly
    /// the last batch if the shard doesn't end on a record boundary.
    pub fn next_batch(&mut self) -> Result<Option<Batch>> {
        // Check if we've exhausted the shard
        if self.current_offset >= self.shard.byte_end {
            return Ok(None);
        }

        // Calculate how much to read
        let remaining = self.shard.byte_end - self.current_offset;
        let mut read_size = (self.batch_size as u64).min(remaining) as usize;

        // Read the data
        let mut reader = self.storage.open_read(&self.path)?;
        let mut data = reader.read_range(self.current_offset, read_size)?;

        if data.is_empty() {
            return Ok(None);
        }

        // Find the last complete record boundary in the batch
        let mut batch_end = self.find_batch_end(&data);

        // If we couldn't find any record boundary and we're not at the end,
        // expand the read to find a boundary
        while batch_end == 0
            && (self.current_offset + data.len() as u64) < self.shard.byte_end
            && read_size < remaining as usize
        {
            // Double the read size, but don't exceed remaining bytes
            read_size = (read_size * 2).min(remaining as usize);
            data = reader.read_range(self.current_offset, read_size)?;
            batch_end = self.find_batch_end(&data);
        }

        // Determine the actual batch data
        let batch_data = if batch_end == 0 {
            // Return all remaining data - this is the last chunk or record spans entire shard
            data
        } else {
            data[..batch_end].to_vec()
        };

        let batch = Batch {
            offset: self.current_offset,
            shard_id: self.shard.shard_id,
            batch_index: self.batch_index,
            data: batch_data.clone(),
        };

        self.current_offset += batch_data.len() as u64;
        self.batch_index += 1;

        Ok(Some(batch))
    }

    /// Find the end position for a batch, ensuring it ends on a record boundary.
    ///
    /// Returns the byte offset within `data` where the batch should end.
    fn find_batch_end(&self, data: &[u8]) -> usize {
        let mut last_record_end = 0;
        let mut offset = 0;

        // Scan through the data finding record boundaries
        while offset < data.len() {
            if let Some(record_end) = self.format.find_record_end(data, offset) {
                last_record_end = record_end;
                offset = record_end;
            } else {
                break;
            }
        }

        last_record_end
    }

    /// Reset the iterator to the beginning of the shard.
    pub fn reset(&mut self) {
        self.current_offset = self.shard.byte_start;
        self.batch_index = 0;
    }

    /// Get the current progress through the shard as a fraction between 0.0 and 1.0.
    pub fn progress(&self) -> f64 {
        let total = self.shard.byte_end - self.shard.byte_start;
        if total == 0 {
            return 1.0;
        }

        let processed = self.current_offset - self.shard.byte_start;
        (processed as f64 / total as f64).min(1.0)
    }

    /// Get the current byte offset within the file.
    pub fn current_offset(&self) -> u64 {
        self.current_offset
    }

    /// Get the shard specification.
    pub fn shard(&self) -> &ShardSpec {
        &self.shard
    }
}

impl Iterator for ShardIterator {
    type Item = Result<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RuntimeError;
    use crate::storage::{ObjectMeta, StorageReader, StorageWriter};
    use std::collections::HashMap;
    use std::io::{Cursor, Read, Seek, SeekFrom, Write};
    use std::path::Path;
    use std::sync::Mutex;

    use super::super::traits::{FixedSizeFormat, NewlineDelimitedFormat};

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
            self.data
                .seek(SeekFrom::Start(start))
                .map_err(|e| RuntimeError::storage_with_source("mock", "seek failed", e))?;

            let mut buf = vec![0u8; length];
            let bytes_read = self
                .data
                .read(&mut buf)
                .map_err(|e| RuntimeError::storage_with_source("mock", "read failed", e))?;
            buf.truncate(bytes_read);
            Ok(buf)
        }
    }

    /// Mock storage writer (minimal implementation)
    struct MockWriter {
        data: Vec<u8>,
    }

    impl Write for MockWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.data.extend_from_slice(buf);
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

    /// Mock storage backend for testing
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
            let data = files
                .get(path)
                .ok_or_else(|| RuntimeError::storage(path, "not found"))?;

            Ok(ObjectMeta {
                size: data.len() as u64,
                modified: None,
                is_dir: false,
            })
        }

        fn open_read(&self, path: &Path) -> Result<Box<dyn StorageReader>> {
            let files = self.files.lock().unwrap();
            let data = files
                .get(path)
                .ok_or_else(|| RuntimeError::storage(path, "not found"))?
                .clone();

            let size = data.len() as u64;
            Ok(Box::new(MockReader {
                data: Cursor::new(data),
                size,
            }))
        }

        fn open_write(&self, _path: &Path) -> Result<Box<dyn StorageWriter>> {
            Ok(Box::new(MockWriter { data: vec![] }))
        }

        fn delete(&self, path: &Path) -> Result<()> {
            self.files.lock().unwrap().remove(path);
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
    fn test_iterator_produces_batches() {
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
        let config = IteratorConfig {
            batch_size: 12, // Small batch to force multiple batches
            shard_id: 0,
        };

        let mut iter =
            ShardIterator::new(storage, PathBuf::from("test.txt"), shard, format, config);

        let mut batches = vec![];
        while let Some(batch) = iter.next() {
            batches.push(batch.unwrap());
        }

        assert!(!batches.is_empty(), "Should produce at least one batch");

        // Verify batch indices are sequential
        for (i, batch) in batches.iter().enumerate() {
            assert_eq!(batch.batch_index, i as u64);
            assert_eq!(batch.shard_id, 0);
        }
    }

    #[test]
    fn test_iterator_exhausts() {
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
        let config = IteratorConfig {
            batch_size: 1024,
            shard_id: 0,
        };

        let mut iter =
            ShardIterator::new(storage, PathBuf::from("test.txt"), shard, format, config);

        // Consume all batches
        let batches: Vec<_> = iter.by_ref().collect();
        assert!(!batches.is_empty());

        // Next call should return None
        assert!(iter.next().is_none());
        assert!(iter.next().is_none()); // Multiple calls should still return None
    }

    #[test]
    fn test_iterator_coverage() {
        let storage = Arc::new(MockStorage::new());
        let data = vec![0u8; 100]; // 100 bytes of data
        storage.add_file("test.bin", data.clone());

        let shard = ShardSpec {
            shard_id: 0,
            total_shards: 1,
            byte_start: 0,
            byte_end: 100,
        };

        let format = Arc::new(FixedSizeFormat::new(10)); // 10-byte records
        let config = IteratorConfig {
            batch_size: 25, // Will get ~2 records per batch
            shard_id: 0,
        };

        let mut iter =
            ShardIterator::new(storage, PathBuf::from("test.bin"), shard, format, config);

        let mut total_bytes = 0;
        while let Some(batch) = iter.next() {
            let batch = batch.unwrap();
            total_bytes += batch.data.len();
        }

        assert_eq!(total_bytes, 100, "All bytes should be covered");
    }

    #[test]
    fn test_iterator_reset() {
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
        let config = IteratorConfig {
            batch_size: 1024,
            shard_id: 0,
        };

        let mut iter =
            ShardIterator::new(storage, PathBuf::from("test.txt"), shard, format, config);

        // Read all batches
        let first_pass: Vec<_> = iter.by_ref().map(|r| r.unwrap()).collect();

        // Reset and read again
        iter.reset();

        let second_pass: Vec<_> = iter.by_ref().map(|r| r.unwrap()).collect();

        assert_eq!(first_pass.len(), second_pass.len());
        for (a, b) in first_pass.iter().zip(second_pass.iter()) {
            assert_eq!(a.data, b.data);
            assert_eq!(a.offset, b.offset);
        }
    }

    #[test]
    fn test_iterator_progress() {
        let storage = Arc::new(MockStorage::new());
        let data = vec![0u8; 100];
        storage.add_file("test.bin", data);

        let shard = ShardSpec {
            shard_id: 0,
            total_shards: 1,
            byte_start: 0,
            byte_end: 100,
        };

        let format = Arc::new(FixedSizeFormat::new(10));
        let config = IteratorConfig {
            batch_size: 20, // 2 records per batch
            shard_id: 0,
        };

        let mut iter =
            ShardIterator::new(storage, PathBuf::from("test.bin"), shard, format, config);

        // Initial progress should be 0
        assert_eq!(iter.progress(), 0.0);

        let mut prev_progress = 0.0;
        while let Some(batch) = iter.next() {
            let _ = batch.unwrap();
            let progress = iter.progress();

            // Progress should monotonically increase
            assert!(
                progress >= prev_progress,
                "Progress decreased: {} -> {}",
                prev_progress,
                progress
            );
            prev_progress = progress;
        }

        // Final progress should be 1.0
        assert!((iter.progress() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_alignment() {
        let storage = Arc::new(MockStorage::new());
        // Create newline-delimited data
        let data = b"short\nlonger line here\nx\nanother\n".to_vec();
        storage.add_file("test.txt", data.clone());

        let shard = ShardSpec {
            shard_id: 0,
            total_shards: 1,
            byte_start: 0,
            byte_end: data.len() as u64,
        };

        let format = Arc::new(NewlineDelimitedFormat::new());
        let config = IteratorConfig {
            batch_size: 10, // Small batch to test alignment
            shard_id: 0,
        };

        let mut iter =
            ShardIterator::new(storage, PathBuf::from("test.txt"), shard, format, config);

        while let Some(batch) = iter.next() {
            let batch = batch.unwrap();
            // Each batch should end with a newline (record boundary)
            // except possibly the last one if data doesn't end with newline
            if !batch.data.is_empty() && iter.current_offset() < data.len() as u64 {
                assert_eq!(
                    batch.data.last(),
                    Some(&b'\n'),
                    "Batch should end on record boundary"
                );
            }
        }
    }

    #[test]
    fn test_iterator_partial_shard() {
        // Test iterator with a shard that's a subset of the file
        let storage = Arc::new(MockStorage::new());
        let data = b"line1\nline2\nline3\nline4\nline5\n".to_vec();
        storage.add_file("test.txt", data);

        // Shard covers only middle portion: "line2\nline3\n"
        let shard = ShardSpec {
            shard_id: 1,
            total_shards: 3,
            byte_start: 6, // After "line1\n"
            byte_end: 18,  // Up to and including "line3\n"
        };

        let format = Arc::new(NewlineDelimitedFormat::new());
        let config = IteratorConfig {
            batch_size: 1024,
            shard_id: 1,
        };

        let mut iter =
            ShardIterator::new(storage, PathBuf::from("test.txt"), shard, format, config);

        let mut all_data = vec![];
        while let Some(batch) = iter.next() {
            all_data.extend_from_slice(&batch.unwrap().data);
        }

        assert_eq!(all_data, b"line2\nline3\n");
    }

    #[test]
    fn test_empty_shard() {
        let storage = Arc::new(MockStorage::new());
        storage.add_file("test.txt", b"some data".to_vec());

        let shard = ShardSpec {
            shard_id: 0,
            total_shards: 1,
            byte_start: 5,
            byte_end: 5, // Empty range
        };

        let format = Arc::new(NewlineDelimitedFormat::new());
        let config = IteratorConfig::default();

        let mut iter =
            ShardIterator::new(storage, PathBuf::from("test.txt"), shard, format, config);

        // Should immediately return None
        assert!(iter.next().is_none());
        assert_eq!(iter.progress(), 1.0);
    }
}
