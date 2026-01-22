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

/// Storage backend type.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageBackendType {
    /// Local filesystem storage.
    #[default]
    Local,
    /// S3-compatible object storage.
    S3,
}

// Storage configuration options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// Storage backend type: "local" or "s3".
    pub backend: StorageBackendType,
    // Base path for all storage operations (local path or S3 key prefix).
    pub base_path: PathBuf,
    // Buffer size in bytes for I/O operations.
    pub buffer_size: usize,
    // Whether to use memory-mapped I/O (local storage only).
    pub use_mmap: bool,
    // File size threshold (bytes) above which to use mmap (local storage only).
    pub mmap_threshold: u64,
    /// S3-specific configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub s3: Option<S3Config>,
}

/// S3-compatible storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct S3Config {
    /// S3 bucket name.
    pub bucket: String,
    /// AWS region (e.g., "us-east-1").
    pub region: String,
    /// Custom endpoint URL (for MinIO, LocalStack, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// AWS access key ID (if not using instance credentials).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub access_key_id: Option<String>,
    /// AWS secret access key (if not using instance credentials).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secret_access_key: Option<String>,
    /// AWS session token (for temporary credentials).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_token: Option<String>,
    /// Maximum number of concurrent connections.
    pub max_connections: usize,
    /// File size threshold (bytes) above which to use multipart upload.
    pub multipart_threshold: u64,
    /// Chunk size (bytes) for multipart upload parts.
    pub multipart_chunk_size: usize,
    /// Maximum number of retries for failed requests.
    pub max_retries: u32,
    /// Initial delay (milliseconds) between retries.
    pub retry_delay_ms: u64,
    /// Maximum delay (milliseconds) between retries.
    pub max_retry_delay_ms: u64,
    /// Connection timeout in milliseconds.
    pub connect_timeout_ms: u64,
    /// Request timeout in milliseconds.
    pub request_timeout_ms: u64,
    /// Whether to use path-style addressing (required for MinIO).
    pub force_path_style: bool,
    /// Whether to allow HTTP (non-TLS) connections.
    pub allow_http: bool,
}

impl Default for S3Config {
    fn default() -> Self {
        Self {
            bucket: String::new(),
            region: "us-east-1".to_string(),
            endpoint: None,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            max_connections: 64,
            multipart_threshold: 100 * 1024 * 1024, // 100 MB
            multipart_chunk_size: 32 * 1024 * 1024, // 32 MB
            max_retries: 5,
            retry_delay_ms: 100,
            max_retry_delay_ms: 30_000,
            connect_timeout_ms: 5_000,
            request_timeout_ms: 30_000,
            force_path_style: false,
            allow_http: false,
        }
    }
}

impl S3Config {
    /// Apply environment variable overrides to S3 configuration.
    #[must_use]
    pub fn with_env_overrides(mut self) -> Self {
        if let Ok(val) = std::env::var("DTR_S3_BUCKET") {
            self.bucket = val;
        }
        if let Ok(val) = std::env::var("DTR_S3_REGION") {
            self.region = val;
        }
        if let Ok(val) = std::env::var("DTR_S3_ENDPOINT") {
            self.endpoint = Some(val);
        }
        if let Ok(val) = std::env::var("DTR_S3_ACCESS_KEY_ID") {
            self.access_key_id = Some(val);
        }
        if let Ok(val) = std::env::var("DTR_S3_SECRET_ACCESS_KEY") {
            self.secret_access_key = Some(val);
        }
        if let Ok(val) = std::env::var("DTR_S3_SESSION_TOKEN") {
            self.session_token = Some(val);
        }
        if let Ok(val) = std::env::var("DTR_S3_MAX_CONNECTIONS") {
            if let Ok(v) = val.parse() {
                self.max_connections = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_S3_MULTIPART_THRESHOLD") {
            if let Ok(v) = val.parse() {
                self.multipart_threshold = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_S3_MULTIPART_CHUNK_SIZE") {
            if let Ok(v) = val.parse() {
                self.multipart_chunk_size = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_S3_MAX_RETRIES") {
            if let Ok(v) = val.parse() {
                self.max_retries = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_S3_RETRY_DELAY_MS") {
            if let Ok(v) = val.parse() {
                self.retry_delay_ms = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_S3_MAX_RETRY_DELAY_MS") {
            if let Ok(v) = val.parse() {
                self.max_retry_delay_ms = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_S3_CONNECT_TIMEOUT_MS") {
            if let Ok(v) = val.parse() {
                self.connect_timeout_ms = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_S3_REQUEST_TIMEOUT_MS") {
            if let Ok(v) = val.parse() {
                self.request_timeout_ms = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_S3_FORCE_PATH_STYLE") {
            if let Ok(v) = val.parse() {
                self.force_path_style = v;
            }
        }
        if let Ok(val) = std::env::var("DTR_S3_ALLOW_HTTP") {
            if let Ok(v) = val.parse() {
                self.allow_http = v;
            }
        }
        self
    }

    /// Validate S3 configuration.
    pub fn validate(&self) -> Result<()> {
        if self.bucket.is_empty() {
            return Err(RuntimeError::config("s3.bucket must not be empty"));
        }
        if self.region.is_empty() {
            return Err(RuntimeError::config("s3.region must not be empty"));
        }
        if self.max_connections == 0 {
            return Err(RuntimeError::config(
                "s3.max_connections must be greater than 0",
            ));
        }
        if self.multipart_chunk_size < 5 * 1024 * 1024 {
            return Err(RuntimeError::config(
                "s3.multipart_chunk_size must be at least 5 MB (S3 minimum)",
            ));
        }
        if self.multipart_chunk_size > 5 * 1024 * 1024 * 1024 {
            return Err(RuntimeError::config(
                "s3.multipart_chunk_size must be at most 5 GB (S3 maximum)",
            ));
        }
        if self.connect_timeout_ms == 0 {
            return Err(RuntimeError::config(
                "s3.connect_timeout_ms must be greater than 0",
            ));
        }
        if self.request_timeout_ms == 0 {
            return Err(RuntimeError::config(
                "s3.request_timeout_ms must be greater than 0",
            ));
        }
        Ok(())
    }
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
            backend: StorageBackendType::Local,
            base_path: PathBuf::from("./data"),
            buffer_size: 64 * 1024, // 64 KB
            use_mmap: true,
            mmap_threshold: 1024 * 1024, // 1 MB
            s3: None,
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
    // - `DTR_STORAGE_BACKEND` overrides `storage.backend` ("local" or "s3")
    // - `DTR_DATASET_SHUFFLE` overrides `dataset.shuffle`
    // - `DTR_CHECKPOINT_COMPRESSION` overrides `checkpoint.compression`
    // - `DTR_S3_BUCKET` overrides `storage.s3.bucket`
    #[must_use]
    pub fn with_env_overrides(mut self) -> Self {
        // Storage overrides
        if let Ok(val) = std::env::var("DTR_STORAGE_BACKEND") {
            match val.to_lowercase().as_str() {
                "local" => self.storage.backend = StorageBackendType::Local,
                "s3" => self.storage.backend = StorageBackendType::S3,
                _ => {} // ignore invalid values
            }
        }
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

        // S3 overrides - create S3Config if any S3 env vars are set
        if std::env::var("DTR_S3_BUCKET").is_ok() {
            let s3_config = self.storage.s3.take().unwrap_or_default().with_env_overrides();
            self.storage.s3 = Some(s3_config);
        } else if let Some(s3_config) = self.storage.s3.take() {
            self.storage.s3 = Some(s3_config.with_env_overrides());
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

        // S3 validation (when S3 backend is selected)
        if self.storage.backend == StorageBackendType::S3 {
            match &self.storage.s3 {
                Some(s3_config) => s3_config.validate()?,
                None => {
                    return Err(RuntimeError::config(
                        "storage.s3 configuration is required when backend is 's3'",
                    ));
                }
            }
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

        assert_eq!(config.storage.backend, StorageBackendType::Local);
        assert_eq!(config.storage.base_path, PathBuf::from("./data"));
        assert_eq!(config.storage.buffer_size, 64 * 1024);
        assert!(config.storage.use_mmap);
        assert_eq!(config.storage.mmap_threshold, 1024 * 1024);
        assert!(config.storage.s3.is_none());

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

    #[test]
    fn test_s3_config_default() {
        let config = S3Config::default();
        assert!(config.bucket.is_empty());
        assert_eq!(config.region, "us-east-1");
        assert!(config.endpoint.is_none());
        assert!(config.access_key_id.is_none());
        assert!(config.secret_access_key.is_none());
        assert_eq!(config.max_connections, 64);
        assert_eq!(config.multipart_threshold, 100 * 1024 * 1024);
        assert_eq!(config.multipart_chunk_size, 32 * 1024 * 1024);
        assert_eq!(config.max_retries, 5);
        assert!(!config.force_path_style);
        assert!(!config.allow_http);
    }

    #[test]
    fn test_s3_config_validate_empty_bucket() {
        let config = S3Config::default();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("bucket"));
    }

    #[test]
    fn test_s3_config_validate_empty_region() {
        let mut config = S3Config::default();
        config.bucket = "my-bucket".to_string();
        config.region = String::new();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("region"));
    }

    #[test]
    fn test_s3_config_validate_chunk_size_too_small() {
        let mut config = S3Config::default();
        config.bucket = "my-bucket".to_string();
        config.multipart_chunk_size = 1024; // Too small (< 5MB)
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("5 MB"));
    }

    #[test]
    fn test_s3_config_validate_success() {
        let mut config = S3Config::default();
        config.bucket = "my-bucket".to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_storage_backend_s3_requires_s3_config() {
        let mut config = RuntimeConfig::default();
        config.storage.backend = StorageBackendType::S3;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("s3 configuration is required"));
    }

    #[test]
    fn test_storage_backend_s3_with_config() {
        let mut config = RuntimeConfig::default();
        config.storage.backend = StorageBackendType::S3;
        config.storage.s3 = Some(S3Config {
            bucket: "my-bucket".to_string(),
            ..Default::default()
        });
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_parse_s3_config_from_toml() {
        let toml = r#"
            [storage]
            backend = "s3"
            base_path = "training-data/"

            [storage.s3]
            bucket = "my-training-bucket"
            region = "us-west-2"
            endpoint = "http://localhost:9000"
            max_connections = 32
            multipart_threshold = 52428800
            multipart_chunk_size = 16777216
            max_retries = 3
            force_path_style = true
            allow_http = true
        "#;

        let config: RuntimeConfig = toml.parse().unwrap();
        assert_eq!(config.storage.backend, StorageBackendType::S3);
        assert_eq!(config.storage.base_path, PathBuf::from("training-data/"));

        let s3 = config.storage.s3.unwrap();
        assert_eq!(s3.bucket, "my-training-bucket");
        assert_eq!(s3.region, "us-west-2");
        assert_eq!(s3.endpoint, Some("http://localhost:9000".to_string()));
        assert_eq!(s3.max_connections, 32);
        assert_eq!(s3.multipart_threshold, 52428800);
        assert_eq!(s3.multipart_chunk_size, 16777216);
        assert_eq!(s3.max_retries, 3);
        assert!(s3.force_path_style);
        assert!(s3.allow_http);
    }
}
