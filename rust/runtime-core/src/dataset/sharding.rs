// rust/runtime-core/src/dataset/sharding.rs

use crate::error::Result;
use crate::storage::StorageReader;

use super::traits::{RecordFormat, ShardSpec};

/// Buffer size for reading data when aligning to record boundaries
const ALIGNMENT_BUFFER_SIZE: usize = 64 * 1024; // 64KB

/// Calculate shard boundaries for a file.
///
/// This function divides a file into `total_shards` shards, ensuring that
/// each shard boundary aligns to a record boundary. This allows each shard
/// to be processed independently without splitting records across shards.
///
/// # Arguments
///
/// * `reader` - A storage reader for the file to shard
/// * `total_shards` - The number of shards to create
/// * `format` - The record format describing how to find record boundaries
///
/// # Returns
///
/// A vector of `ShardSpec` describing each shard's boundaries.
///
/// # Errors
///
/// Returns an error if reading from storage fails.
pub fn calculate_shards(
    reader: &mut dyn StorageReader,
    total_shards: u32,
    format: &dyn RecordFormat,
) -> Result<Vec<ShardSpec>> {
    let file_size = reader.size();

    // Handle edge cases
    if file_size == 0 || total_shards == 0 {
        return Ok(vec![]);
    }

    // If we have more shards than bytes, reduce to file_size shards
    let effective_shards = total_shards.min(file_size as u32);

    let mut shards = Vec::with_capacity(effective_shards as usize);
    let bytes_per_shard = file_size / effective_shards as u64;

    let mut current_start = 0u64;

    for shard_id in 0..effective_shards {
        let is_last_shard = shard_id == effective_shards - 1;

        let byte_end = if is_last_shard {
            // Last shard goes to the end of the file
            file_size
        } else {
            // Calculate the approximate end for this shard
            let approximate_end = current_start + bytes_per_shard;

            // Align to the next record boundary
            align_to_record_boundary(reader, approximate_end, format)?
        };

        // Only add non-empty shards (or the last shard which may be empty)
        if byte_end > current_start || is_last_shard {
            shards.push(ShardSpec {
                shard_id,
                total_shards: effective_shards,
                byte_start: current_start,
                byte_end,
            });
            current_start = byte_end;
        }

        // If we've reached the end of the file, stop creating shards
        if current_start >= file_size {
            break;
        }
    }

    // Update total_shards in all specs to reflect actual count
    let actual_count = shards.len() as u32;
    for (idx, shard) in shards.iter_mut().enumerate() {
        shard.shard_id = idx as u32;
        shard.total_shards = actual_count;
    }

    Ok(shards)
}

/// Align a byte offset to the next record boundary.
///
/// This function reads data starting at the given offset and finds the end
/// of the current record to ensure we don't split a record across shards.
///
/// # Arguments
///
/// * `reader` - A storage reader for the file
/// * `offset` - The byte offset to align
/// * `format` - The record format describing how to find record boundaries
///
/// # Returns
///
/// The aligned byte offset (pointing to the start of the next record).
fn align_to_record_boundary(
    reader: &mut dyn StorageReader,
    offset: u64,
    format: &dyn RecordFormat,
) -> Result<u64> {
    let file_size = reader.size();

    // If offset is at or beyond file end, return file_size
    if offset >= file_size {
        return Ok(file_size);
    }

    // Try to align without reading data (works for fixed-size records)
    if let Some(aligned) = format.try_align_offset(offset, file_size) {
        return Ok(aligned);
    }

    // Read a buffer starting at the offset to find the record boundary
    let read_size = ALIGNMENT_BUFFER_SIZE.min((file_size - offset) as usize);
    let data = reader.read_range(offset, read_size)?;

    // Find the end of the record that contains or starts at position 0 in our buffer
    if let Some(record_end) = format.find_record_end(&data, 0) {
        Ok(offset + record_end as u64)
    } else {
        // If no record boundary found in buffer, return file end
        // This handles cases where the last record is incomplete or
        // the buffer wasn't large enough
        Ok(file_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::StorageReader;
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use super::super::traits::{FixedSizeFormat, LengthPrefixedFormat, NewlineDelimitedFormat};

    /// Mock storage reader for testing
    struct MockReader {
        data: Cursor<Vec<u8>>,
        size: u64,
    }

    impl MockReader {
        fn new(data: Vec<u8>) -> Self {
            let size = data.len() as u64;
            Self {
                data: Cursor::new(data),
                size,
            }
        }
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
            self.data.seek(SeekFrom::Start(start)).map_err(|e| {
                crate::error::RuntimeError::storage_with_source("mock", "seek failed", e)
            })?;

            let mut buf = vec![0u8; length];
            let bytes_read = self.data.read(&mut buf).map_err(|e| {
                crate::error::RuntimeError::storage_with_source("mock", "read failed", e)
            })?;
            buf.truncate(bytes_read);
            Ok(buf)
        }
    }

    #[test]
    fn test_calculate_shards_count() {
        // 100 bytes of fixed-size records (10 bytes each = 10 records)
        let data = vec![0u8; 100];
        let mut reader = MockReader::new(data);
        let format = FixedSizeFormat::new(10);

        let shards = calculate_shards(&mut reader, 4, &format).unwrap();
        assert_eq!(shards.len(), 4);
    }

    #[test]
    fn test_calculate_shards_coverage() {
        // Ensure shards cover the entire file with no gaps
        let data = vec![0u8; 100];
        let mut reader = MockReader::new(data);
        let format = FixedSizeFormat::new(10);

        let shards = calculate_shards(&mut reader, 4, &format).unwrap();

        // First shard should start at 0
        assert_eq!(shards[0].byte_start, 0);

        // Last shard should end at file size
        assert_eq!(shards.last().unwrap().byte_end, 100);

        // Check continuity: each shard starts where the previous ended
        for i in 1..shards.len() {
            assert_eq!(
                shards[i].byte_start,
                shards[i - 1].byte_end,
                "Gap between shard {} and {}",
                i - 1,
                i
            );
        }
    }

    #[test]
    fn test_calculate_shards_no_overlap() {
        let data = vec![0u8; 100];
        let mut reader = MockReader::new(data);
        let format = FixedSizeFormat::new(10);

        let shards = calculate_shards(&mut reader, 4, &format).unwrap();

        for i in 0..shards.len() {
            // Each shard's start should be less than its end (non-empty)
            assert!(
                shards[i].byte_start <= shards[i].byte_end,
                "Shard {} has invalid range",
                i
            );

            // No overlap with next shard
            if i + 1 < shards.len() {
                assert!(
                    shards[i].byte_end <= shards[i + 1].byte_start,
                    "Shard {} overlaps with shard {}",
                    i,
                    i + 1
                );
            }
        }
    }

    #[test]
    fn test_calculate_shards_aligned() {
        // Test with newline-delimited format to ensure alignment
        let data = b"line1\nline2\nline3\nline4\nline5\nline6\n".to_vec();
        let mut reader = MockReader::new(data.clone());
        let format = NewlineDelimitedFormat::new();

        let shards = calculate_shards(&mut reader, 3, &format).unwrap();

        // Each shard boundary should be at a newline (record end)
        for shard in &shards {
            if shard.byte_end < data.len() as u64 {
                // The byte before shard end should be a newline
                assert_eq!(
                    data[shard.byte_end as usize - 1],
                    b'\n',
                    "Shard {} does not end at record boundary",
                    shard.shard_id
                );
            }
        }
    }

    #[test]
    fn test_calculate_shards_single() {
        let data = vec![0u8; 50];
        let mut reader = MockReader::new(data);
        let format = FixedSizeFormat::new(10);

        let shards = calculate_shards(&mut reader, 1, &format).unwrap();

        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].shard_id, 0);
        assert_eq!(shards[0].total_shards, 1);
        assert_eq!(shards[0].byte_start, 0);
        assert_eq!(shards[0].byte_end, 50);
    }

    #[test]
    fn test_calculate_shards_more_than_file() {
        // Request more shards than bytes in the file
        let data = vec![0u8; 10];
        let mut reader = MockReader::new(data);
        let format = FixedSizeFormat::new(1);

        let shards = calculate_shards(&mut reader, 100, &format).unwrap();

        // Should not have more shards than bytes
        assert!(shards.len() <= 10);

        // Should still cover the whole file
        assert_eq!(shards[0].byte_start, 0);
        assert_eq!(shards.last().unwrap().byte_end, 10);
    }

    #[test]
    fn test_calculate_shards_empty_file() {
        let data = vec![];
        let mut reader = MockReader::new(data);
        let format = FixedSizeFormat::new(10);

        let shards = calculate_shards(&mut reader, 4, &format).unwrap();
        assert!(shards.is_empty());
    }

    #[test]
    fn test_calculate_shards_zero_shards() {
        let data = vec![0u8; 100];
        let mut reader = MockReader::new(data);
        let format = FixedSizeFormat::new(10);

        let shards = calculate_shards(&mut reader, 0, &format).unwrap();
        assert!(shards.is_empty());
    }

    #[test]
    fn test_calculate_shards_with_length_prefixed() {
        // Create data with length-prefixed records
        let mut data = vec![];
        for i in 0..10 {
            let record = format!("record{}", i);
            data.extend_from_slice(&(record.len() as u32).to_be_bytes());
            data.extend_from_slice(record.as_bytes());
        }

        let mut reader = MockReader::new(data.clone());
        let format = LengthPrefixedFormat::new();

        let shards = calculate_shards(&mut reader, 3, &format).unwrap();

        // Verify coverage
        assert_eq!(shards[0].byte_start, 0);
        assert_eq!(shards.last().unwrap().byte_end, data.len() as u64);

        // Verify no gaps
        for i in 1..shards.len() {
            assert_eq!(shards[i].byte_start, shards[i - 1].byte_end);
        }
    }

    #[test]
    fn test_shard_ids_are_sequential() {
        let data = vec![0u8; 100];
        let mut reader = MockReader::new(data);
        let format = FixedSizeFormat::new(10);

        let shards = calculate_shards(&mut reader, 4, &format).unwrap();

        for (i, shard) in shards.iter().enumerate() {
            assert_eq!(shard.shard_id, i as u32);
            assert_eq!(shard.total_shards, shards.len() as u32);
        }
    }
}
