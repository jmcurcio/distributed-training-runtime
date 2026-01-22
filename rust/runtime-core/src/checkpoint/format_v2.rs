// rust/runtime-core/src/checkpoint/format_v2.rs

//! Version 2 checkpoint format for streaming writes.
//!
//! The V2 format places the checksum and metadata in a trailer at the end,
//! allowing streaming writes without knowing the final size upfront.
//!
//! ## Format Layout
//!
//! ```text
//! +----------------------------+
//! | Magic "DTR2" (4 bytes)     |
//! +----------------------------+
//! | Flags (4 bytes)            |
//! +----------------------------+
//! | Reserved (24 bytes)        |
//! +----------------------------+
//! | Compressed Data Chunks     |  <- Variable length
//! | ...                        |
//! +----------------------------+
//! | Trailer (bincode)          |  <- CheckpointTrailer
//! +----------------------------+
//! | Trailer Length (4 bytes)   |  <- u32 LE
//! +----------------------------+
//! ```
//!
//! ## Reading Strategy
//!
//! 1. Read last 4 bytes to get trailer length
//! 2. Read trailer (trailer_len bytes before the length field)
//! 3. Read header (first 32 bytes)
//! 4. Stream compressed data between header and trailer
//! 5. Verify checksum incrementally or at the end

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Magic bytes for V2 checkpoint files.
pub const MAGIC_V2: [u8; 4] = *b"DTR2";

/// Header size in bytes (magic + flags + reserved).
pub const HEADER_SIZE: usize = 32;

/// Size of the trailer length field at the end.
pub const TRAILER_LEN_SIZE: usize = 4;

/// Flag: Compression is LZ4.
pub const FLAG_COMPRESSION_LZ4: u32 = 0x0001;

/// Flag: Compression is ZSTD.
pub const FLAG_COMPRESSION_ZSTD: u32 = 0x0002;

/// Flag: No compression.
pub const FLAG_COMPRESSION_NONE: u32 = 0x0000;

/// Compression mask (lower 2 bits).
pub const FLAG_COMPRESSION_MASK: u32 = 0x0003;

/// Flag: Data is chunked (for incremental verification).
pub const FLAG_CHUNKED: u32 = 0x0010;

/// Header for V2 checkpoint files.
///
/// The header is a fixed-size structure at the beginning of the file.
#[derive(Debug, Clone, Copy)]
pub struct CheckpointHeaderV2 {
    /// Magic bytes identifying this as a V2 checkpoint file.
    pub magic: [u8; 4],
    /// Flags indicating compression and other options.
    pub flags: u32,
    /// Reserved bytes for future use.
    pub reserved: [u8; 24],
}

impl CheckpointHeaderV2 {
    /// Creates a new V2 header with the specified compression.
    pub fn new(compression: CompressionType) -> Self {
        Self {
            magic: MAGIC_V2,
            flags: compression.to_flag(),
            reserved: [0u8; 24],
        }
    }

    /// Creates a new V2 header with flags.
    pub fn with_flags(flags: u32) -> Self {
        Self {
            magic: MAGIC_V2,
            flags,
            reserved: [0u8; 24],
        }
    }

    /// Returns the compression type from flags.
    pub fn compression(&self) -> CompressionType {
        CompressionType::from_flag(self.flags & FLAG_COMPRESSION_MASK)
    }

    /// Returns true if the data is chunked.
    pub fn is_chunked(&self) -> bool {
        (self.flags & FLAG_CHUNKED) != 0
    }

    /// Validates the magic bytes.
    pub fn validate_magic(&self) -> bool {
        self.magic == MAGIC_V2
    }

    /// Serializes the header to bytes.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..8].copy_from_slice(&self.flags.to_le_bytes());
        bytes[8..32].copy_from_slice(&self.reserved);
        bytes
    }

    /// Deserializes a header from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < HEADER_SIZE {
            return None;
        }

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);

        let flags = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

        let mut reserved = [0u8; 24];
        reserved.copy_from_slice(&bytes[8..32]);

        Some(Self {
            magic,
            flags,
            reserved,
        })
    }
}

/// Compression type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    None,
    Lz4,
    Zstd,
}

impl CompressionType {
    /// Converts compression type to flag value.
    pub fn to_flag(self) -> u32 {
        match self {
            CompressionType::None => FLAG_COMPRESSION_NONE,
            CompressionType::Lz4 => FLAG_COMPRESSION_LZ4,
            CompressionType::Zstd => FLAG_COMPRESSION_ZSTD,
        }
    }

    /// Creates compression type from flag value.
    pub fn from_flag(flag: u32) -> Self {
        match flag & FLAG_COMPRESSION_MASK {
            FLAG_COMPRESSION_LZ4 => CompressionType::Lz4,
            FLAG_COMPRESSION_ZSTD => CompressionType::Zstd,
            _ => CompressionType::None,
        }
    }

    /// Creates compression type from string name.
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "lz4" => CompressionType::Lz4,
            "zstd" => CompressionType::Zstd,
            _ => CompressionType::None,
        }
    }

    /// Returns the string name of the compression type.
    pub fn as_str(&self) -> &'static str {
        match self {
            CompressionType::None => "none",
            CompressionType::Lz4 => "lz4",
            CompressionType::Zstd => "zstd",
        }
    }
}

/// Trailer for V2 checkpoint files.
///
/// The trailer is placed at the end of the file and contains
/// verification data and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointTrailer {
    /// XXHash64 checksum of the uncompressed data.
    pub checksum: u64,
    /// Size of the uncompressed data in bytes.
    pub uncompressed_size: u64,
    /// Size of the compressed data in bytes (excluding header and trailer).
    pub compressed_size: u64,
    /// Number of chunks (if chunked mode).
    pub chunk_count: u32,
    /// User-defined metadata.
    pub metadata: HashMap<String, String>,
}

impl CheckpointTrailer {
    /// Creates a new trailer.
    pub fn new(checksum: u64, uncompressed_size: u64, compressed_size: u64) -> Self {
        Self {
            checksum,
            uncompressed_size,
            compressed_size,
            chunk_count: 0,
            metadata: HashMap::new(),
        }
    }

    /// Creates a new trailer with metadata.
    pub fn with_metadata(
        checksum: u64,
        uncompressed_size: u64,
        compressed_size: u64,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            checksum,
            uncompressed_size,
            compressed_size,
            chunk_count: 0,
            metadata,
        }
    }

    /// Serializes the trailer to bytes using bincode.
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("trailer serialization should not fail")
    }

    /// Deserializes a trailer from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        bincode::deserialize(bytes).ok()
    }
}

/// Represents chunk information for incremental verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    /// Offset of the chunk in the compressed stream.
    pub offset: u64,
    /// Size of the compressed chunk.
    pub compressed_size: u32,
    /// Size of the uncompressed chunk.
    pub uncompressed_size: u32,
    /// Checksum of the uncompressed chunk.
    pub checksum: u64,
}

/// V2 format version indicator for compatibility checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointVersion {
    /// V1 format (header at start with size/checksum known upfront).
    V1,
    /// V2 format (trailer at end for streaming).
    V2,
}

impl CheckpointVersion {
    /// Detects the format version from the magic bytes.
    pub fn detect(magic: &[u8; 4]) -> Option<Self> {
        if magic == b"DTRC" {
            Some(CheckpointVersion::V1)
        } else if magic == &MAGIC_V2 {
            Some(CheckpointVersion::V2)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_v2_new() {
        let header = CheckpointHeaderV2::new(CompressionType::Lz4);
        assert_eq!(header.magic, MAGIC_V2);
        assert_eq!(header.compression(), CompressionType::Lz4);
        assert!(header.validate_magic());
    }

    #[test]
    fn test_header_v2_roundtrip() {
        let header = CheckpointHeaderV2::new(CompressionType::Zstd);
        let bytes = header.to_bytes();
        let restored = CheckpointHeaderV2::from_bytes(&bytes).unwrap();

        assert_eq!(restored.magic, header.magic);
        assert_eq!(restored.flags, header.flags);
        assert_eq!(restored.compression(), CompressionType::Zstd);
    }

    #[test]
    fn test_header_v2_chunked() {
        let header = CheckpointHeaderV2::with_flags(FLAG_COMPRESSION_LZ4 | FLAG_CHUNKED);
        assert_eq!(header.compression(), CompressionType::Lz4);
        assert!(header.is_chunked());
    }

    #[test]
    fn test_compression_type_conversions() {
        assert_eq!(CompressionType::parse("lz4"), CompressionType::Lz4);
        assert_eq!(CompressionType::parse("LZ4"), CompressionType::Lz4);
        assert_eq!(CompressionType::parse("zstd"), CompressionType::Zstd);
        assert_eq!(CompressionType::parse("ZSTD"), CompressionType::Zstd);
        assert_eq!(CompressionType::parse("none"), CompressionType::None);
        assert_eq!(CompressionType::parse("unknown"), CompressionType::None);

        assert_eq!(CompressionType::Lz4.as_str(), "lz4");
        assert_eq!(CompressionType::Zstd.as_str(), "zstd");
        assert_eq!(CompressionType::None.as_str(), "none");

        assert_eq!(CompressionType::from_flag(CompressionType::Lz4.to_flag()), CompressionType::Lz4);
        assert_eq!(CompressionType::from_flag(CompressionType::Zstd.to_flag()), CompressionType::Zstd);
        assert_eq!(CompressionType::from_flag(CompressionType::None.to_flag()), CompressionType::None);
    }

    #[test]
    fn test_trailer_new() {
        let trailer = CheckpointTrailer::new(12345, 1000, 500);
        assert_eq!(trailer.checksum, 12345);
        assert_eq!(trailer.uncompressed_size, 1000);
        assert_eq!(trailer.compressed_size, 500);
        assert_eq!(trailer.chunk_count, 0);
        assert!(trailer.metadata.is_empty());
    }

    #[test]
    fn test_trailer_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("epoch".to_string(), "10".to_string());

        let trailer = CheckpointTrailer::with_metadata(12345, 1000, 500, metadata);
        assert_eq!(trailer.metadata.get("epoch"), Some(&"10".to_string()));
    }

    #[test]
    fn test_trailer_roundtrip() {
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "value".to_string());

        let trailer = CheckpointTrailer::with_metadata(99999, 50000, 25000, metadata);
        let bytes = trailer.to_bytes();
        let restored = CheckpointTrailer::from_bytes(&bytes).unwrap();

        assert_eq!(restored.checksum, trailer.checksum);
        assert_eq!(restored.uncompressed_size, trailer.uncompressed_size);
        assert_eq!(restored.compressed_size, trailer.compressed_size);
        assert_eq!(restored.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_version_detection() {
        assert_eq!(CheckpointVersion::detect(b"DTRC"), Some(CheckpointVersion::V1));
        assert_eq!(CheckpointVersion::detect(b"DTR2"), Some(CheckpointVersion::V2));
        assert_eq!(CheckpointVersion::detect(b"XXXX"), None);
    }

    #[test]
    fn test_header_size_constant() {
        let header = CheckpointHeaderV2::new(CompressionType::None);
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);
    }
}
