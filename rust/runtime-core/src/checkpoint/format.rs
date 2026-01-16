// rust/runtime-core/src/checkpoint/format.rs

//! Checkpoint file format specification.
//!
//! The checkpoint format is:
//! ```text
//! +-------------------+
//! | Header (bincode)  |  <- CheckpointHeader serialized with bincode
//! +-------------------+
//! | Compressed Data   |  <- Payload compressed according to header
//! +-------------------+
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Header for a checkpoint file.
///
/// The header contains metadata about the checkpoint including compression
/// settings and integrity verification data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointHeader {
    /// Magic bytes identifying this as a checkpoint file ("DTRC")
    pub magic: [u8; 4],
    /// Format version number
    pub version: u32,
    /// Compression algorithm used ("none", "lz4", or "zstd")
    pub compression: String,
    /// Size of the data before compression
    pub uncompressed_size: u64,
    /// XXHash64 checksum of the uncompressed data
    pub checksum: u64,
    /// User-defined metadata
    pub metadata: HashMap<String, String>,
}

impl CheckpointHeader {
    /// Magic bytes for checkpoint files
    pub const MAGIC: [u8; 4] = *b"DTRC";

    /// Current format version
    pub const VERSION: u32 = 1;

    /// Creates a new checkpoint header.
    ///
    /// # Arguments
    ///
    /// * `compression` - Compression algorithm name
    /// * `uncompressed_size` - Size of data before compression
    /// * `checksum` - XXHash64 checksum of uncompressed data
    pub fn new(compression: String, uncompressed_size: u64, checksum: u64) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            compression,
            uncompressed_size,
            checksum,
            metadata: HashMap::new(),
        }
    }

    /// Creates a new checkpoint header with metadata.
    pub fn with_metadata(
        compression: String,
        uncompressed_size: u64,
        checksum: u64,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            compression,
            uncompressed_size,
            checksum,
            metadata,
        }
    }

    /// Validates the header magic bytes.
    pub fn validate_magic(&self) -> bool {
        self.magic == Self::MAGIC
    }

    /// Validates the header version.
    pub fn validate_version(&self) -> bool {
        self.version == Self::VERSION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_new() {
        let header = CheckpointHeader::new("lz4".to_string(), 1000, 12345);

        assert_eq!(header.magic, CheckpointHeader::MAGIC);
        assert_eq!(header.version, CheckpointHeader::VERSION);
        assert_eq!(header.compression, "lz4");
        assert_eq!(header.uncompressed_size, 1000);
        assert_eq!(header.checksum, 12345);
        assert!(header.metadata.is_empty());
    }

    #[test]
    fn test_header_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("epoch".to_string(), "10".to_string());
        metadata.insert("step".to_string(), "5000".to_string());

        let header =
            CheckpointHeader::with_metadata("zstd".to_string(), 2000, 67890, metadata.clone());

        assert_eq!(header.metadata.get("epoch"), Some(&"10".to_string()));
        assert_eq!(header.metadata.get("step"), Some(&"5000".to_string()));
    }

    #[test]
    fn test_validate_magic() {
        let header = CheckpointHeader::new("none".to_string(), 100, 0);
        assert!(header.validate_magic());

        let mut invalid = header.clone();
        invalid.magic = *b"XXXX";
        assert!(!invalid.validate_magic());
    }

    #[test]
    fn test_validate_version() {
        let header = CheckpointHeader::new("none".to_string(), 100, 0);
        assert!(header.validate_version());

        let mut invalid = header.clone();
        invalid.version = 999;
        assert!(!invalid.validate_version());
    }

    #[test]
    fn test_header_serialization() {
        let header = CheckpointHeader::new("lz4".to_string(), 1000, 12345);

        // Serialize with bincode
        let encoded = bincode::serialize(&header).unwrap();

        // Deserialize
        let decoded: CheckpointHeader = bincode::deserialize(&encoded).unwrap();

        assert_eq!(decoded.magic, header.magic);
        assert_eq!(decoded.version, header.version);
        assert_eq!(decoded.compression, header.compression);
        assert_eq!(decoded.uncompressed_size, header.uncompressed_size);
        assert_eq!(decoded.checksum, header.checksum);
    }
}
