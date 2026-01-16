// rust/python-bindings/src/dataset.rs

//! Python bindings for Dataset and ShardIterator.

// Allow useless_conversion - clippy has false positives with our error conversion pattern
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use runtime_core::{Dataset, ShardIterator};

use crate::WrappedError;

/// A registered dataset with pre-calculated shard boundaries.
///
/// Datasets are created through `Runtime.register_dataset()` and provide
/// methods for inspecting shard information and creating iterators.
///
/// Attributes
/// ----------
/// num_shards : int
///     The number of shards in this dataset.
/// total_bytes : int
///     The total size of the dataset in bytes.
/// path : str
///     The path to the dataset file.
///
/// Examples
/// --------
/// >>> dataset = runtime.register_dataset("data.jsonl", shards=4)
/// >>> print(f"Dataset has {dataset.num_shards} shards, {dataset.total_bytes} bytes")
/// >>> for shard_id in range(dataset.num_shards):
/// ...     for batch in dataset.iter_shard(shard_id):
/// ...         process(batch)
#[pyclass(name = "Dataset")]
pub struct PyDataset {
    inner: Dataset,
}

impl PyDataset {
    /// Create a new PyDataset wrapping a Rust Dataset.
    pub fn new(inner: Dataset) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDataset {
    /// The number of shards in this dataset.
    #[getter]
    fn num_shards(&self) -> u32 {
        self.inner.num_shards()
    }

    /// The total size of the dataset in bytes.
    #[getter]
    fn total_bytes(&self) -> u64 {
        self.inner.total_bytes()
    }

    /// The path to the dataset file.
    #[getter]
    fn path(&self) -> String {
        self.inner.path().to_string_lossy().into_owned()
    }

    /// Get information about a specific shard.
    ///
    /// Parameters
    /// ----------
    /// shard_id : int
    ///     The shard ID to get information for.
    ///
    /// Returns
    /// -------
    /// tuple[int, int]
    ///     A tuple of (byte_start, byte_end) for the shard.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the shard ID is out of range.
    fn shard_info(&self, shard_id: u32) -> PyResult<(u64, u64)> {
        let spec = self.inner.shard_info(shard_id).map_err(WrappedError)?;
        Ok((spec.byte_start, spec.byte_end))
    }

    /// Create an iterator for a specific shard.
    ///
    /// Parameters
    /// ----------
    /// shard_id : int
    ///     The shard ID to iterate over.
    /// batch_size : int, optional
    ///     Number of bytes per batch. Defaults to 1MB (1048576).
    ///
    /// Returns
    /// -------
    /// ShardIterator
    ///     An iterator that yields batches of bytes.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the shard ID is out of range.
    ///
    /// Examples
    /// --------
    /// >>> for batch in dataset.iter_shard(0, batch_size=64*1024):
    /// ...     records = batch.decode('utf-8').splitlines()
    /// ...     for record in records:
    /// ...         process(json.loads(record))
    #[pyo3(signature = (shard_id, batch_size=1048576))]
    fn iter_shard(&self, shard_id: u32, batch_size: usize) -> PyResult<PyShardIterator> {
        let iter = self
            .inner
            .iter_shard(shard_id, batch_size)
            .map_err(WrappedError)?;
        Ok(PyShardIterator::new(iter))
    }

    /// Get the byte range for all shards.
    ///
    /// Returns
    /// -------
    /// list[tuple[int, int]]
    ///     A list of (byte_start, byte_end) tuples for each shard.
    fn all_shard_info(&self) -> Vec<(u64, u64)> {
        self.inner
            .shards()
            .iter()
            .map(|s| (s.byte_start, s.byte_end))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Dataset(path='{}', num_shards={}, total_bytes={})",
            self.inner.path().display(),
            self.inner.num_shards(),
            self.inner.total_bytes()
        )
    }
}

/// An iterator over batches in a shard.
///
/// ShardIterators are created through `Dataset.iter_shard()` and yield
/// batches of bytes from the shard. Each batch ends on a record boundary.
///
/// Examples
/// --------
/// >>> iter = dataset.iter_shard(0)
/// >>> for batch in iter:
/// ...     print(f"Got {len(batch)} bytes")
/// >>> print(f"Progress: {iter.progress():.1%}")
#[pyclass(name = "ShardIterator")]
pub struct PyShardIterator {
    inner: ShardIterator,
}

impl PyShardIterator {
    /// Create a new PyShardIterator wrapping a Rust ShardIterator.
    pub fn new(inner: ShardIterator) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyShardIterator {
    /// Return self for iteration protocol.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Get the next batch from the iterator.
    ///
    /// Returns
    /// -------
    /// bytes or None
    ///     The next batch of data, or None if the shard is exhausted.
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If reading fails.
    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        // Release GIL during I/O operation
        let batch = py
            .allow_threads(|| self.inner.next_batch())
            .map_err(WrappedError)?;

        match batch {
            Some(b) => Ok(Some(PyBytes::new_bound(py, &b.data))),
            None => Ok(None),
        }
    }

    /// Get the current progress through the shard.
    ///
    /// Returns
    /// -------
    /// float
    ///     A value between 0.0 and 1.0 indicating progress.
    fn progress(&self) -> f64 {
        self.inner.progress()
    }

    /// Reset the iterator to the beginning of the shard.
    fn reset(&mut self) {
        self.inner.reset()
    }

    /// Get the current byte offset within the file.
    ///
    /// Returns
    /// -------
    /// int
    ///     The current byte position.
    #[getter]
    fn current_offset(&self) -> u64 {
        self.inner.current_offset()
    }

    /// Get the shard ID this iterator is processing.
    ///
    /// Returns
    /// -------
    /// int
    ///     The shard ID.
    #[getter]
    fn shard_id(&self) -> u32 {
        self.inner.shard().shard_id
    }

    fn __repr__(&self) -> String {
        format!(
            "ShardIterator(shard_id={}, progress={:.1}%)",
            self.inner.shard().shard_id,
            self.inner.progress() * 100.0
        )
    }
}
