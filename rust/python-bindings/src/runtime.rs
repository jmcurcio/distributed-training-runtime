// rust/python-bindings/src/runtime.rs

//! Python bindings for the Runtime class.

// Allow useless_conversion - clippy has false positives with our error conversion pattern
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use runtime_core::Runtime;

use crate::dataset::PyDataset;
use crate::WrappedError;

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
/// Register a dataset and iterate:
///
/// >>> dataset = runtime.register_dataset("data.jsonl", shards=4, format="newline")
/// >>> for batch in dataset.iter_shard(0):
/// ...     process(batch)
#[pyclass(name = "Runtime")]
pub struct PyRuntime {
    inner: Runtime,
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
    #[pyo3(signature = (config_path=None))]
    fn new(config_path: Option<&str>) -> PyResult<Self> {
        let inner = match config_path {
            Some(path) => Runtime::from_config_file(path).map_err(WrappedError)?,
            None => Runtime::new().map_err(WrappedError)?,
        };
        Ok(Self { inner })
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

        let dataset = self
            .inner
            .register_dataset(path, shard_count, format_str)
            .map_err(WrappedError)?;
        Ok(PyDataset::new(dataset))
    }

    /// Save a checkpoint.
    ///
    /// The checkpoint is compressed according to the runtime configuration
    /// and written atomically.
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

        // Release GIL during I/O operation
        let path = py
            .allow_threads(|| self.inner.save_checkpoint(name, bytes))
            .map_err(WrappedError)?;

        Ok(path.to_string_lossy().into_owned())
    }

    /// Load a checkpoint.
    ///
    /// The checkpoint is decompressed and its integrity is verified.
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
        // Release GIL during I/O operation
        let data = py
            .allow_threads(|| self.inner.load_checkpoint(path))
            .map_err(WrappedError)?;

        Ok(PyBytes::new_bound(py, &data))
    }

    /// Get the storage base path.
    ///
    /// Returns
    /// -------
    /// str
    ///     The base path for all storage operations.
    #[getter]
    fn base_path(&self) -> String {
        self.inner
            .config()
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
        self.inner
            .config()
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
        self.inner.config().checkpoint.compression.clone()
    }
}
