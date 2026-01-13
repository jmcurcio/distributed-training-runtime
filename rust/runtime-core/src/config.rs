// rust/runtime-core/src/config.rs

//! Configuration management for the distributed training runtime.
//!
//! This module provides configuration parsing from TOML files, environment
//! variable overrides, and validation of configuration values.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::error::{Result, RuntimeError};

// Top-level runtime configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeConfig {
    pub storage: StorageConfig,
    pub dataset: DatasetConfig,
    pub checkpoint: CheckpointConfig,
    pub performance: PerformanceConfig,
}

// Storage configuration options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    // Base path for all storage operations.
    pub base_path: PathBuf,
    // Buffer size in bytes for I/O operations.
    pub buffer_size: usize,
    // Whether to use memory-mapped I/O.
    pub use_mmap: bool,
    // File size threshold (bytes) above which to use mmap.
    pub mmap_threshold: u64,
}

/// Dataset configuration options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DatasetConfig {
    // Default number of shards for dataset partitioning.
    pub default_shard_count: u32,
    // Number of batches to prefetch.
    pub prefetch_batches: usize,
    // Whether to shuffle the dataset.
    pub shuffle: bool,
    // Optional seed for reproducible shuffling.
    pub seed: Option<u64>,
}

// Checkpoint configuration options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CheckpointConfig {
    // Directory for storing checkpoints.
    pub checkpoint_dir: PathBuf,
    // Compression algorithm: "none", "lz4", or "zstd".
    pub compression: String,
    // Compression level (algorithm-specific).
    pub compression_level: i32,
    // Number of recent checkpoints to keep.
    pub keep_last_n: usize,
    // Whether to use atomic writes (write to temp then rename).
    pub atomic_writes: bool,
}

// Performance tuning options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PerformanceConfig {
    // Number of I/O threads.
    pub io_threads: usize,
    // Maximum memory (bytes) for buffering.
    pub max_buffer_memory: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("./data"),
            buffer_size: 64 * 1024, // 64 KB
            use_mmap: true,
            mmap_threshold: 1024 * 1024, // 1 MB
        }
    }
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            default_shard_count: 1,
            prefetch_batches: 2,
            shuffle: false,
            seed: None,
        }
    }
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("./checkpoints"),
            compression: "lz4".to_string(),
            compression_level: 1,
            keep_last_n: 3,
            atomic_writes: true,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            io_threads: 4,
            max_buffer_memory: 256 * 1024 * 1024, // 256 MB
        }
    }
}

impl FromStr for RuntimeConfig {
    type Err = RuntimeError;

    /// Parse configuration from a TOML string.
    fn from_str(s: &str) -> Result<Self> {
        toml::from_str(s)
            .map_err(|e| RuntimeError::config_with_source("failed to parse TOML config", e))
    }
}

impl RuntimeConfig {
    // Load configuration from a TOML file.
    //
    // # Errors
    //
    // Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| {
            RuntimeError::storage_with_source(path, "failed to read config file", e)
        })?;
        let config: Self = content.parse()?;
        config.validate()?;
        Ok(config)
    }

    // Apply environment variable overrides.
    //
    // Environment variables are prefixed with `DTR_` and use underscores
    // to separate nested fields. For example:
    // - `DTR_STORAGE_BASE_PATH` overrides `storage.base_path`
    // - `DTR_DATASET_SHUFFLE` overrides `dataset.shuffle`
    // - `DTR_CHECKPOINT_COMPRESSION` overrides `checkpoint.compression`
    #[must_use]
    pub fn with_env_overrides(mut self) -> Self {
        // Storage overrides
        if let Ok(val) = std::env::var("DTR_STORAGE_BASE_PATH") {
            self.storage.base_path = PathBuf::from(val);
        }
        if let Ok(val) = std::env::var("DTR_STORAGE_BUFFER_SIZE") {
            if let Ok(v) = val.parse() {
                self.storage.buffer_size = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_STORAGE_USE_MMAP") {
            if let Ok(v) = val.parse() {
                self.storage.use_mmap = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_STORAGE_MMAP_THRESHOLD") {
            if let Ok(v) = val.parse() {
                self.storage.mmap_threshold = v;
            }
        }

        // Dataset overrides
        if let Ok(val) = std::env::var("DTR_DATASET_DEFAULT_SHARD_COUNT") {
            if let Ok(v) = val.parse() {
                self.dataset.default_shard_count = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_DATASET_PREFETCH_BATCHES") {
            if let Ok(v) = val.parse() {
                self.dataset.prefetch_batches = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_DATASET_SHUFFLE") {
            if let Ok(v) = val.parse() {
                self.dataset.shuffle = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_DATASET_SEED") {
            if let Ok(v) = val.parse() {
                self.dataset.seed = Some(v);
            }
        }

        // Checkpoint overrides
        if let Ok(val) = std::env::var("DTR_CHECKPOINT_DIR") {
            self.checkpoint.checkpoint_dir = PathBuf::from(val);
        }
        if let Ok(val) = std::env::var("DTR_CHECKPOINT_COMPRESSION") {
            self.checkpoint.compression = val;
        }
        if let Ok(val) = std::env::var("DTR_CHECKPOINT_COMPRESSION_LEVEL") {
            if let Ok(v) = val.parse() {
                self.checkpoint.compression_level = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_CHECKPOINT_KEEP_LAST_N") {
            if let Ok(v) = val.parse() {
                self.checkpoint.keep_last_n = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_CHECKPOINT_ATOMIC_WRITES") {
            if let Ok(v) = val.parse() {
                self.checkpoint.atomic_writes = v;
            }
        }

        // Performance overrides
        if let Ok(val) = std::env::var("DTR_PERFORMANCE_IO_THREADS") {
            if let Ok(v) = val.parse() {
                self.performance.io_threads = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_PERFORMANCE_MAX_BUFFER_MEMORY") {
            if let Ok(v) = val.parse() {
                self.performance.max_buffer_memory = v;
            }
        }

        self
    }

    // Validate all configuration values.
    //
    // # Errors
    //
    // Returns an error if any configuration value is invalid.
    pub fn validate(&self) -> Result<()> {
        // Storage validation
        if self.storage.buffer_size == 0 {
            return Err(RuntimeError::config(
                "storage.buffer_size must be greater than 0",
            ));
        }

        // Dataset validation
        if self.dataset.default_shard_count == 0 {
            return Err(RuntimeError::config(
                "dataset.default_shard_count must be greater than 0",
            ));
        }

        // Checkpoint validation
        let valid_compression = ["none", "lz4", "zstd"];
        if !valid_compression.contains(&self.checkpoint.compression.as_str()) {
            return Err(RuntimeError::config(format!(
                "checkpoint.compression must be one of: {}",
                valid_compression.join(", ")
            )));
        }

        if self.checkpoint.keep_last_n == 0 {
            return Err(RuntimeError::config(
                "checkpoint.keep_last_n must be greater than 0",
            ));
        }

        // Performance validation
        if self.performance.io_threads == 0 {
            return Err(RuntimeError::config(
                "performance.io_threads must be greater than 0",
            ));
        }

        if self.performance.max_buffer_memory == 0 {
            return Err(RuntimeError::config(
                "performance.max_buffer_memory must be greater than 0",
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = RuntimeConfig::default();

        assert_eq!(config.storage.base_path, PathBuf::from("./data"));
        assert_eq!(config.storage.buffer_size, 64 * 1024);
        assert!(config.storage.use_mmap);
        assert_eq!(config.storage.mmap_threshold, 1024 * 1024);

        assert_eq!(config.dataset.default_shard_count, 1);
        assert_eq!(config.dataset.prefetch_batches, 2);
        assert!(!config.dataset.shuffle);
        assert!(config.dataset.seed.is_none());

        assert_eq!(
            config.checkpoint.checkpoint_dir,
            PathBuf::from("./checkpoints")
        );
        assert_eq!(config.checkpoint.compression, "lz4");
        assert_eq!(config.checkpoint.compression_level, 1);
        assert_eq!(config.checkpoint.keep_last_n, 3);
        assert!(config.checkpoint.atomic_writes);

        assert_eq!(config.performance.io_threads, 4);
        assert_eq!(config.performance.max_buffer_memory, 256 * 1024 * 1024);
    }

    #[test]
    fn test_default_validates() {
        let config = RuntimeConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_from_str_empty() {
        let config: RuntimeConfig = "".parse().unwrap();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_from_str_partial() {
        let toml = r#"
            [storage]
            base_path = "/custom/path"
            buffer_size = 128000
        "#;
        let config: RuntimeConfig = toml.parse().unwrap();

        assert_eq!(config.storage.base_path, PathBuf::from("/custom/path"));
        assert_eq!(config.storage.buffer_size, 128000);
        // Other storage fields should be defaults
        assert!(config.storage.use_mmap);
        // Other sections should be defaults
        assert_eq!(config.dataset.default_shard_count, 1);
    }

    #[test]
    fn test_from_str_full() {
        let toml = r#"
            [storage]
            base_path = "/data/training"
            buffer_size = 131072
            use_mmap = false
            mmap_threshold = 2097152

            [dataset]
            default_shard_count = 8
            prefetch_batches = 4
            shuffle = true
            seed = 42

            [checkpoint]
            checkpoint_dir = "/checkpoints"
            compression = "zstd"
            compression_level = 3
            keep_last_n = 5
            atomic_writes = false

            [performance]
            io_threads = 8
            max_buffer_memory = 536870912
        "#;

        let config: RuntimeConfig = toml.parse().unwrap();

        assert_eq!(config.storage.base_path, PathBuf::from("/data/training"));
        assert_eq!(config.storage.buffer_size, 131072);
        assert!(!config.storage.use_mmap);
        assert_eq!(config.storage.mmap_threshold, 2097152);

        assert_eq!(config.dataset.default_shard_count, 8);
        assert_eq!(config.dataset.prefetch_batches, 4);
        assert!(config.dataset.shuffle);
        assert_eq!(config.dataset.seed, Some(42));

        assert_eq!(
            config.checkpoint.checkpoint_dir,
            PathBuf::from("/checkpoints")
        );
        assert_eq!(config.checkpoint.compression, "zstd");
        assert_eq!(config.checkpoint.compression_level, 3);
        assert_eq!(config.checkpoint.keep_last_n, 5);
        assert!(!config.checkpoint.atomic_writes);

        assert_eq!(config.performance.io_threads, 8);
        assert_eq!(config.performance.max_buffer_memory, 536870912);
    }

    #[test]
    fn test_from_str_invalid_toml() {
        let result: std::result::Result<RuntimeConfig, _> = "invalid = [".parse();
        assert!(result.is_err());
    }

    #[test]
    fn test_from_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
            [storage]
            base_path = "/tmp/test"
            "#
        )
        .unwrap();

        let config = RuntimeConfig::from_file(file.path()).unwrap();
        assert_eq!(config.storage.base_path, PathBuf::from("/tmp/test"));
    }

    #[test]
    fn test_from_file_not_found() {
        let result = RuntimeConfig::from_file("/nonexistent/config.toml");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_invalid_buffer_size() {
        let mut config = RuntimeConfig::default();
        config.storage.buffer_size = 0;
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_invalid_shard_count() {
        let mut config = RuntimeConfig::default();
        config.dataset.default_shard_count = 0;
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_invalid_compression() {
        let mut config = RuntimeConfig::default();
        config.checkpoint.compression = "gzip".to_string();
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_valid_compression_none() {
        let mut config = RuntimeConfig::default();
        config.checkpoint.compression = "none".to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_valid_compression_zstd() {
        let mut config = RuntimeConfig::default();
        config.checkpoint.compression = "zstd".to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_keep_last_n() {
        let mut config = RuntimeConfig::default();
        config.checkpoint.keep_last_n = 0;
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_invalid_io_threads() {
        let mut config = RuntimeConfig::default();
        config.performance.io_threads = 0;
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_invalid_max_buffer_memory() {
        let mut config = RuntimeConfig::default();
        config.performance.max_buffer_memory = 0;
        let result = config.validate();
        assert!(result.is_err());
    }

    // Helper to clear all DTR_ environment variables for test isolation
    fn clear_dtr_env_vars() {
        for (key, _) in std::env::vars() {
            if key.starts_with("DTR_") {
                std::env::remove_var(&key);
            }
        }
    }

    // Environment variable tests are combined into a single test to avoid
    // race conditions when tests run in parallel, since env vars are global state.
    #[test]
    fn test_env_overrides() {
        // Ensure clean state
        clear_dtr_env_vars();

        // Test 1: Valid environment overrides
        std::env::set_var("DTR_STORAGE_BASE_PATH", "/env/path");
        std::env::set_var("DTR_STORAGE_BUFFER_SIZE", "32768");
        std::env::set_var("DTR_DATASET_SHUFFLE", "true");
        std::env::set_var("DTR_DATASET_SEED", "12345");
        std::env::set_var("DTR_CHECKPOINT_COMPRESSION", "zstd");
        std::env::set_var("DTR_PERFORMANCE_IO_THREADS", "16");

        let config = RuntimeConfig::default().with_env_overrides();

        assert_eq!(config.storage.base_path, PathBuf::from("/env/path"));
        assert_eq!(config.storage.buffer_size, 32768);
        assert!(config.dataset.shuffle);
        assert_eq!(config.dataset.seed, Some(12345));
        assert_eq!(config.checkpoint.compression, "zstd");
        assert_eq!(config.performance.io_threads, 16);

        // Clean up for next sub-test
        clear_dtr_env_vars();

        // Test 2: Invalid values should be ignored (keep defaults)
        std::env::set_var("DTR_STORAGE_BUFFER_SIZE", "not_a_number");

        let config = RuntimeConfig::default().with_env_overrides();

        // Should still have the default value since parsing failed
        assert_eq!(config.storage.buffer_size, 64 * 1024);

        // Final cleanup
        clear_dtr_env_vars();
    }

    #[test]
    fn test_serialize_roundtrip() {
        let original = RuntimeConfig::default();
        let toml_str = toml::to_string(&original).unwrap();
        let parsed: RuntimeConfig = toml_str.parse().unwrap();

        assert_eq!(original.storage.base_path, parsed.storage.base_path);
        assert_eq!(original.storage.buffer_size, parsed.storage.buffer_size);
        assert_eq!(original.dataset.shuffle, parsed.dataset.shuffle);
        assert_eq!(
            original.checkpoint.compression,
            parsed.checkpoint.compression
        );
    }
}
