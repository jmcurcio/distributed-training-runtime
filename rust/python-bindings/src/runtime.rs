// rust/python-bindings/src/runtime.rs

//! Python bindings for the Runtime class.

// Allow useless_conversion - clippy has false positives with our error conversion pattern
#![allow(clippy::useless_conversion)]

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use runtime_core::config::{RuntimeConfig, StorageBackendType};
use runtime_core::Runtime;
use runtime_core::async_runtime::AsyncRuntime;
use tokio::runtime::Runtime as TokioRuntime;

#[cfg(feature = "s3")]
use runtime_core::config::S3Config;

use crate::dataset::PyDataset;
use crate::WrappedError;

/// Internal runtime type - either sync or async depending on backend.
enum RuntimeInner {
    /// Sync runtime for local storage
    Sync(Runtime),
    /// Async runtime for S3 storage
    Async(AsyncRuntime),
}

/// The main runtime for the distributed training system.
///
/// The Runtime manages storage, datasets, and checkpoints. Create one at the
/// start of your training script and use it throughout.
///
/// Examples
/// --------
/// Create with default configuration:
///
/// >>> runtime = Runtime()
///
/// Create with a configuration file:
///
/// >>> runtime = Runtime("config.toml")
///
/// Create with S3 storage (requires s3 feature):
///
/// >>> runtime = Runtime(
/// ...     backend="s3",
/// ...     s3_bucket="my-bucket",
/// ...     s3_region="us-east-1"
/// ... )
///
/// Register a dataset and iterate:
///
/// >>> dataset = runtime.register_dataset("data.jsonl", shards=4, format="newline")
/// >>> for batch in dataset.iter_shard(0):
/// ...     process(batch)
#[pyclass(name = "Runtime")]
pub struct PyRuntime {
    inner: RuntimeInner,
    config: RuntimeConfig,
    /// Tokio runtime for async operations
    tokio_runtime: Arc<TokioRuntime>,
}

#[pymethods]
impl PyRuntime {
    /// Create a new Runtime.
    ///
    /// Parameters
    /// ----------
    /// config_path : str, optional
    ///     Path to a TOML configuration file. If not provided, uses default
    ///     configuration.
    /// base_path : str, optional
    ///     Base path for storage operations. Overrides config file.
    /// backend : str, optional
    ///     Storage backend type: "local" or "s3". Defaults to "local".
    /// checkpoint_dir : str, optional
    ///     Directory for checkpoints. Relative to base_path.
    /// compression : str, optional
    ///     Compression algorithm: "none", "lz4", or "zstd". Defaults to "lz4".
    /// s3_bucket : str, optional
    ///     S3 bucket name (required if backend="s3").
    /// s3_region : str, optional
    ///     S3 region. Defaults to "us-east-1".
    /// s3_endpoint : str, optional
    ///     S3 endpoint URL (for MinIO or other S3-compatible storage).
    /// s3_access_key : str, optional
    ///     S3 access key ID. Falls back to environment variable.
    /// s3_secret_key : str, optional
    ///     S3 secret access key. Falls back to environment variable.
    ///
    /// Returns
    /// -------
    /// Runtime
    ///     A new runtime instance.
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If the configuration file cannot be read.
    /// ValueError
    ///     If the configuration is invalid.
    #[new]
    #[pyo3(signature = (
        config_path=None,
        base_path=None,
        backend=None,
        checkpoint_dir=None,
        compression=None,
        s3_bucket=None,
        s3_region=None,
        s3_endpoint=None,
        s3_access_key=None,
        s3_secret_key=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    #[allow(unused_variables)]  // S3 vars unused when s3 feature disabled
    fn new(
        config_path: Option<&str>,
        base_path: Option<&str>,
        backend: Option<&str>,
        checkpoint_dir: Option<&str>,
        compression: Option<&str>,
        s3_bucket: Option<&str>,
        s3_region: Option<&str>,
        s3_endpoint: Option<&str>,
        s3_access_key: Option<&str>,
        s3_secret_key: Option<&str>,
    ) -> PyResult<Self> {
        // Create tokio runtime for async operations
        let tokio_runtime = Arc::new(
            TokioRuntime::new()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );

        // Build configuration
        let mut config = if let Some(path) = config_path {
            RuntimeConfig::from_file(path)
                .map_err(WrappedError)?
                .with_env_overrides()
        } else {
            RuntimeConfig::default().with_env_overrides()
        };

        // Apply base_path override
        if let Some(path) = base_path {
            config.storage.base_path = PathBuf::from(path);
        }

        // Apply backend override
        if let Some(backend_str) = backend {
            config.storage.backend = match backend_str.to_lowercase().as_str() {
                "local" => StorageBackendType::Local,
                "s3" => StorageBackendType::S3,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown backend: '{}'. Expected 'local' or 's3'",
                        backend_str
                    )));
                }
            };
        }

        // Apply checkpoint config overrides
        if let Some(dir) = checkpoint_dir {
            config.checkpoint.checkpoint_dir = PathBuf::from(dir);
        }
        if let Some(comp) = compression {
            config.checkpoint.compression = comp.to_string();
        }

        // Apply S3 config if backend is S3
        #[cfg(feature = "s3")]
        if matches!(config.storage.backend, StorageBackendType::S3) {
            let bucket = s3_bucket.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "s3_bucket is required when backend='s3'",
                )
            })?;

            let mut s3_config = S3Config::default();
            s3_config.bucket = bucket.to_string();

            if let Some(region) = s3_region {
                s3_config.region = region.to_string();
            }
            if let Some(endpoint) = s3_endpoint {
                s3_config.endpoint = Some(endpoint.to_string());
                // Allow HTTP for custom endpoints (MinIO, LocalStack)
                s3_config.allow_http = endpoint.starts_with("http://");
                s3_config.force_path_style = true;
            }
            if let Some(key) = s3_access_key {
                s3_config.access_key_id = Some(key.to_string());
            }
            if let Some(secret) = s3_secret_key {
                s3_config.secret_access_key = Some(secret.to_string());
            }

            config.storage.s3 = Some(s3_config);
        }

        #[cfg(not(feature = "s3"))]
        if matches!(config.storage.backend, StorageBackendType::S3) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "S3 backend requested but 's3' feature is not enabled. \
                Reinstall with S3 support or use backend='local'.",
            ));
        }

        // Create the appropriate runtime based on backend
        let inner = match config.storage.backend {
            StorageBackendType::Local => {
                let runtime = Runtime::from_config(config.clone()).map_err(WrappedError)?;
                RuntimeInner::Sync(runtime)
            }
            StorageBackendType::S3 => {
                // Use async runtime for S3
                let async_runtime = tokio_runtime
                    .block_on(AsyncRuntime::from_config(config.clone()))
                    .map_err(WrappedError)?;
                RuntimeInner::Async(async_runtime)
            }
        };

        Ok(Self {
            inner,
            config,
            tokio_runtime,
        })
    }

    /// Register a dataset for sharded iteration.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to the dataset file (relative to storage base path).
    /// shards : int, optional
    ///     Number of shards to divide the dataset into. Defaults to 1.
    /// format : str, optional
    ///     Record format. One of:
    ///     - "newline": Newline-delimited records (JSONL, CSV, etc.)
    ///     - "fixed:N": Fixed-size records of N bytes
    ///     - "length-prefixed": 4-byte big-endian length prefix + data
    ///     Defaults to "newline".
    ///
    /// Returns
    /// -------
    /// Dataset
    ///     A dataset object that can be used to iterate over shards.
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If the file does not exist or cannot be read.
    /// ValueError
    ///     If the format string is invalid.
    #[pyo3(signature = (path, shards=None, format=None))]
    fn register_dataset(
        &self,
        path: &str,
        shards: Option<u32>,
        format: Option<&str>,
    ) -> PyResult<PyDataset> {
        let shard_count = shards.unwrap_or(1);
        let format_str = format.unwrap_or("newline");

        let dataset = match &self.inner {
            RuntimeInner::Sync(runtime) => {
                runtime
                    .register_dataset(path, shard_count, format_str)
                    .map_err(WrappedError)?
            }
            RuntimeInner::Async(runtime) => {
                // Block on async dataset registration
                self.tokio_runtime
                    .block_on(runtime.register_dataset(path, shard_count, format_str))
                    .map_err(WrappedError)?
                    .into()
            }
        };
        Ok(PyDataset::new(dataset))
    }

    /// Save a checkpoint.
    ///
    /// The checkpoint is compressed according to the runtime configuration
    /// and written atomically. When using S3 backend, checkpoints are saved
    /// directly to S3.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Base name for the checkpoint (e.g., "model", "optimizer").
    /// data : bytes
    ///     The data to checkpoint.
    ///
    /// Returns
    /// -------
    /// str
    ///     Path to the saved checkpoint file.
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If the checkpoint cannot be written.
    fn save_checkpoint(
        &self,
        py: Python<'_>,
        name: &str,
        data: &Bound<'_, PyBytes>,
    ) -> PyResult<String> {
        let bytes = data.as_bytes();

        let path = match &self.inner {
            RuntimeInner::Sync(runtime) => {
                // Release GIL during I/O operation
                py.allow_threads(|| runtime.save_checkpoint(name, bytes))
                    .map_err(WrappedError)?
            }
            RuntimeInner::Async(runtime) => {
                // Release GIL and run async operation
                py.allow_threads(|| {
                    self.tokio_runtime
                        .block_on(runtime.save_checkpoint(name, bytes))
                })
                .map_err(WrappedError)?
            }
        };

        Ok(path.to_string_lossy().into_owned())
    }

    /// Load a checkpoint.
    ///
    /// The checkpoint is decompressed and its integrity is verified.
    /// When using S3 backend, checkpoints are loaded directly from S3.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to the checkpoint file.
    ///
    /// Returns
    /// -------
    /// bytes
    ///     The decompressed checkpoint data.
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If the checkpoint cannot be read.
    /// RuntimeError
    ///     If the checkpoint is corrupted.
    fn load_checkpoint<'py>(&self, py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyBytes>> {
        let data = match &self.inner {
            RuntimeInner::Sync(runtime) => {
                // Release GIL during I/O operation
                py.allow_threads(|| runtime.load_checkpoint(path))
                    .map_err(WrappedError)?
            }
            RuntimeInner::Async(runtime) => {
                // Release GIL and run async operation
                py.allow_threads(|| {
                    self.tokio_runtime
                        .block_on(runtime.load_checkpoint(path))
                })
                .map_err(WrappedError)?
            }
        };

        Ok(PyBytes::new_bound(py, &data))
    }

    /// Download a file from S3 to local storage.
    ///
    /// This is useful for downloading dataset files from S3 before registering
    /// them with register_dataset(). Only available when using S3 backend.
    ///
    /// Parameters
    /// ----------
    /// s3_path : str
    ///     Path to the file in S3 (relative to base_path).
    /// local_path : str
    ///     Local path to save the file.
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If the file cannot be downloaded.
    /// ValueError
    ///     If not using S3 backend.
    #[pyo3(signature = (s3_path, local_path))]
    fn download_file(&self, py: Python<'_>, s3_path: &str, local_path: &str) -> PyResult<()> {
        match &self.inner {
            RuntimeInner::Sync(_) => {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "download_file() is only available when using S3 backend"
                ))
            }
            RuntimeInner::Async(runtime) => {
                let async_storage = runtime.async_storage();
                let s3_path_buf = std::path::Path::new(s3_path);
                let local_path_buf = PathBuf::from(local_path);

                py.allow_threads(|| {
                    self.tokio_runtime.block_on(async {
                        // Read from S3
                        let mut reader = async_storage.open_read(s3_path_buf).await
                            .map_err(WrappedError)?;
                        let data = reader.read_all().await.map_err(WrappedError)?;

                        // Write to local file
                        if let Some(parent) = local_path_buf.parent() {
                            std::fs::create_dir_all(parent)
                                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                        }
                        std::fs::write(&local_path_buf, &data)
                            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

                        Ok::<(), PyErr>(())
                    })
                })?;

                Ok(())
            }
        }
    }

    /// Upload a file from local storage to S3.
    ///
    /// This is useful for uploading dataset files to S3. Only available when
    /// using S3 backend.
    ///
    /// Parameters
    /// ----------
    /// local_path : str
    ///     Local path to the file.
    /// s3_path : str
    ///     Path to save in S3 (relative to base_path).
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If the file cannot be uploaded.
    /// ValueError
    ///     If not using S3 backend.
    #[pyo3(signature = (local_path, s3_path))]
    fn upload_file(&self, py: Python<'_>, local_path: &str, s3_path: &str) -> PyResult<()> {
        match &self.inner {
            RuntimeInner::Sync(_) => {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "upload_file() is only available when using S3 backend"
                ))
            }
            RuntimeInner::Async(runtime) => {
                let async_storage = runtime.async_storage();
                let s3_path_buf = std::path::Path::new(s3_path);
                let local_path_buf = PathBuf::from(local_path);

                py.allow_threads(|| {
                    self.tokio_runtime.block_on(async {
                        // Read local file
                        let data = std::fs::read(&local_path_buf)
                            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

                        // Write to S3
                        let mut writer = async_storage.open_write(s3_path_buf).await
                            .map_err(WrappedError)?;
                        writer.write_all_bytes(&data).await.map_err(WrappedError)?;
                        writer.finish().await.map_err(WrappedError)?;

                        Ok::<(), PyErr>(())
                    })
                })?;

                Ok(())
            }
        }
    }

    /// List files in an S3 path.
    ///
    /// Only available when using S3 backend.
    ///
    /// Parameters
    /// ----------
    /// prefix : str
    ///     S3 path prefix to list (relative to base_path).
    ///
    /// Returns
    /// -------
    /// list[str]
    ///     List of file paths.
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If the listing fails.
    /// ValueError
    ///     If not using S3 backend.
    #[pyo3(signature = (prefix=""))]
    fn list_files(&self, py: Python<'_>, prefix: &str) -> PyResult<Vec<String>> {
        match &self.inner {
            RuntimeInner::Sync(_) => {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "list_files() is only available when using S3 backend"
                ))
            }
            RuntimeInner::Async(runtime) => {
                let async_storage = runtime.async_storage();
                let prefix_path = std::path::Path::new(prefix);

                py.allow_threads(|| {
                    self.tokio_runtime.block_on(async {
                        async_storage.list(prefix_path).await.map_err(WrappedError)
                    })
                })
                .map_err(|e| e.into())
            }
        }
    }

    /// Check if a file exists in S3.
    ///
    /// Only available when using S3 backend.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     S3 path to check (relative to base_path).
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if the file exists.
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If the check fails.
    /// ValueError
    ///     If not using S3 backend.
    #[pyo3(signature = (path))]
    fn file_exists(&self, py: Python<'_>, path: &str) -> PyResult<bool> {
        match &self.inner {
            RuntimeInner::Sync(_) => {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "file_exists() is only available when using S3 backend"
                ))
            }
            RuntimeInner::Async(runtime) => {
                let async_storage = runtime.async_storage();
                let file_path = std::path::Path::new(path);

                py.allow_threads(|| {
                    self.tokio_runtime.block_on(async {
                        async_storage.exists(file_path).await.map_err(WrappedError)
                    })
                })
                .map_err(|e| e.into())
            }
        }
    }

    /// Get the storage base path.
    ///
    /// Returns
    /// -------
    /// str
    ///     The base path for all storage operations.
    #[getter]
    fn base_path(&self) -> String {
        self.config
            .storage
            .base_path
            .to_string_lossy()
            .into_owned()
    }

    /// Get the checkpoint directory.
    ///
    /// Returns
    /// -------
    /// str
    ///     The directory where checkpoints are stored.
    #[getter]
    fn checkpoint_dir(&self) -> String {
        self.config
            .checkpoint
            .checkpoint_dir
            .to_string_lossy()
            .into_owned()
    }

    /// Get the compression algorithm.
    ///
    /// Returns
    /// -------
    /// str
    ///     The compression algorithm used for checkpoints.
    #[getter]
    fn compression(&self) -> String {
        self.config.checkpoint.compression.clone()
    }

    /// Get the storage backend type.
    ///
    /// Returns
    /// -------
    /// str
    ///     The storage backend type: "local" or "s3".
    #[getter]
    fn backend(&self) -> String {
        match self.config.storage.backend {
            StorageBackendType::Local => "local".to_string(),
            StorageBackendType::S3 => "s3".to_string(),
        }
    }

    /// Check if S3 storage backend is enabled.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if using S3 backend.
    #[getter]
    fn is_s3(&self) -> bool {
        matches!(self.config.storage.backend, StorageBackendType::S3)
    }
}
