// rust/runtime-core/src/dataset/traits.rs

/// Specification for a single shard
#[derive(Debug, Clone)]
pub struct ShardSpec {
    pub shard_id: u32,
    pub total_shards: u32,
    pub byte_start: u64,
    pub byte_end: u64,
}

impl ShardSpec {
    /// Returns the size of this shard in bytes
    pub fn size(&self) -> u64 {
        self.byte_end.saturating_sub(self.byte_start)
    }
}

/// A batch of data from iteration
#[derive(Debug)]
pub struct Batch {
    pub data: Vec<u8>,
    pub offset: u64,
    pub shard_id: u32,
    pub batch_index: u64,
}

/// Describes how to find record boundaries
pub trait RecordFormat: Send + Sync {
    /// Minimum size of a record in bytes
    fn min_record_size(&self) -> usize;

    /// Find the end of the record that starts at or after `offset` in `data`.
    /// Returns the byte position after the record ends (exclusive end).
    /// Returns None if no complete record is found.
    fn find_record_end(&self, data: &[u8], offset: usize) -> Option<usize>;

    /// Name of this record format
    fn name(&self) -> &'static str;

    /// Try to align a file offset to the next record boundary without reading data.
    /// Returns Some(aligned_offset) if alignment can be calculated from the offset alone,
    /// or None if data must be read to find the boundary.
    ///
    /// This is primarily useful for fixed-size records where alignment can be
    /// calculated mathematically.
    fn try_align_offset(&self, file_offset: u64, file_size: u64) -> Option<u64> {
        let _ = (file_offset, file_size);
        None // Default: need to read data
    }
}

/// Fixed-size records
#[derive(Debug, Clone)]
pub struct FixedSizeFormat {
    pub record_size: usize,
}

impl FixedSizeFormat {
    pub fn new(record_size: usize) -> Self {
        Self { record_size }
    }
}

impl RecordFormat for FixedSizeFormat {
    fn min_record_size(&self) -> usize {
        self.record_size
    }

    fn find_record_end(&self, data: &[u8], offset: usize) -> Option<usize> {
        if offset >= data.len() {
            return None;
        }

        // Calculate how many bytes into the current record we are
        let bytes_into_record = offset % self.record_size;
        let bytes_to_end = if bytes_into_record == 0 {
            self.record_size
        } else {
            self.record_size - bytes_into_record
        };

        let end = offset + bytes_to_end;
        if end <= data.len() {
            Some(end)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "fixed-size"
    }

    fn try_align_offset(&self, file_offset: u64, file_size: u64) -> Option<u64> {
        if file_offset >= file_size {
            return Some(file_size);
        }

        let record_size = self.record_size as u64;
        let remainder = file_offset % record_size;
        let aligned = if remainder == 0 {
            // Already at a boundary, move to end of next record
            file_offset + record_size
        } else {
            // Move to end of current record
            file_offset + (record_size - remainder)
        };

        Some(aligned.min(file_size))
    }
}

/// Newline-delimited records (JSONL, CSV, etc.)
#[derive(Debug, Clone, Default)]
pub struct NewlineDelimitedFormat;

impl NewlineDelimitedFormat {
    pub fn new() -> Self {
        Self
    }
}

impl RecordFormat for NewlineDelimitedFormat {
    fn min_record_size(&self) -> usize {
        1 // At minimum, just a newline
    }

    fn find_record_end(&self, data: &[u8], offset: usize) -> Option<usize> {
        if offset >= data.len() {
            return None;
        }

        // Find the next newline starting from offset
        for (i, &byte) in data[offset..].iter().enumerate() {
            if byte == b'\n' {
                return Some(offset + i + 1); // Include the newline
            }
        }

        None
    }

    fn name(&self) -> &'static str {
        "newline-delimited"
    }
}

/// Length-prefixed records (4-byte big-endian length + data)
#[derive(Debug, Clone, Default)]
pub struct LengthPrefixedFormat;

impl LengthPrefixedFormat {
    pub fn new() -> Self {
        Self
    }
}

impl RecordFormat for LengthPrefixedFormat {
    fn min_record_size(&self) -> usize {
        4 // Minimum is just the length prefix with zero-length data
    }

    fn find_record_end(&self, data: &[u8], offset: usize) -> Option<usize> {
        if offset + 4 > data.len() {
            return None;
        }

        // Read 4-byte big-endian length
        let length_bytes: [u8; 4] = data[offset..offset + 4].try_into().ok()?;
        let length = u32::from_be_bytes(length_bytes) as usize;

        let end = offset + 4 + length;
        if end <= data.len() {
            Some(end)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "length-prefixed"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_size_format_boundary() {
        let format = FixedSizeFormat::new(10);

        // At record boundary (0), should find end at 10
        assert_eq!(format.find_record_end(&[0u8; 20], 0), Some(10));

        // At position 5 (middle of record), should find end at 10
        assert_eq!(format.find_record_end(&[0u8; 20], 5), Some(10));

        // At position 10 (second record start), should find end at 20
        assert_eq!(format.find_record_end(&[0u8; 20], 10), Some(20));

        // At position 15, should find end at 20
        assert_eq!(format.find_record_end(&[0u8; 20], 15), Some(20));

        // Not enough data for a complete record
        assert_eq!(format.find_record_end(&[0u8; 5], 0), None);

        // Offset beyond data
        assert_eq!(format.find_record_end(&[0u8; 10], 15), None);
    }

    #[test]
    fn test_newline_format_boundary() {
        let format = NewlineDelimitedFormat::new();

        // Simple case: find newline
        let data = b"hello\nworld\n";
        assert_eq!(format.find_record_end(data, 0), Some(6)); // "hello\n"
        assert_eq!(format.find_record_end(data, 6), Some(12)); // "world\n"

        // Starting in the middle of a line
        assert_eq!(format.find_record_end(data, 3), Some(6)); // "lo\n"

        // No newline found
        let data_no_newline = b"hello world";
        assert_eq!(format.find_record_end(data_no_newline, 0), None);

        // Empty after offset
        assert_eq!(format.find_record_end(data, 12), None);
    }

    #[test]
    fn test_length_prefixed_boundary() {
        let format = LengthPrefixedFormat::new();

        // Create a record with length 5 and 5 bytes of data
        let mut data = vec![];
        data.extend_from_slice(&5u32.to_be_bytes()); // Length = 5
        data.extend_from_slice(b"hello");
        data.extend_from_slice(&3u32.to_be_bytes()); // Length = 3
        data.extend_from_slice(b"bye");

        // First record: 4 bytes length + 5 bytes data = 9
        assert_eq!(format.find_record_end(&data, 0), Some(9));

        // Second record starts at 9: 4 bytes length + 3 bytes data = 7 more = 16
        assert_eq!(format.find_record_end(&data, 9), Some(16));

        // Not enough data for length prefix
        assert_eq!(format.find_record_end(&[0u8; 2], 0), None);

        // Length prefix says more data than available
        let mut incomplete = vec![];
        incomplete.extend_from_slice(&100u32.to_be_bytes());
        incomplete.extend_from_slice(b"short");
        assert_eq!(format.find_record_end(&incomplete, 0), None);

        // Zero-length record
        let mut zero_len = vec![];
        zero_len.extend_from_slice(&0u32.to_be_bytes());
        assert_eq!(format.find_record_end(&zero_len, 0), Some(4));
    }

    #[test]
    fn test_shard_spec_size() {
        let shard = ShardSpec {
            shard_id: 0,
            total_shards: 4,
            byte_start: 100,
            byte_end: 250,
        };
        assert_eq!(shard.size(), 150);

        // Edge case: empty shard
        let empty_shard = ShardSpec {
            shard_id: 0,
            total_shards: 1,
            byte_start: 100,
            byte_end: 100,
        };
        assert_eq!(empty_shard.size(), 0);
    }
}
