// rust/python-bindings/src/dataset.rs

//! Python bindings for Dataset and ShardIterator.

// Allow useless_conversion - clippy has false positives with our error conversion pattern
#![allow(clippy::useless_conversion)]

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use runtime_core::dataset::{IteratorConfig, PrefetchConfig, PrefetchingIterator, RecordFormat};
use runtime_core::storage::StorageBackend;
use runtime_core::{Dataset, ShardIterator};

use crate::WrappedError;

/// Option type for prefetch parameter that can be bool or int.
#[derive(Debug, Clone)]
pub enum PyPrefetchOption {
    Bool(bool),
    Int(usize),
}

impl<'py> FromPyObject<'py> for PyPrefetchOption {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // Try to extract as bool first
        if let Ok(b) = ob.extract::<bool>() {
            return Ok(PyPrefetchOption::Bool(b));
        }
        // Then try as int
        if let Ok(i) = ob.extract::<usize>() {
            return Ok(PyPrefetchOption::Int(i));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "prefetch must be a bool or int",
        ))
    }
}

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

    /// Get the storage backend.
    fn storage(&self) -> Arc<dyn StorageBackend> {
        self.inner.storage()
    }

    /// Get the record format.
    fn format(&self) -> Arc<dyn RecordFormat> {
        self.inner.format()
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
    /// prefetch : bool or int, optional
    ///     Enable prefetching to reduce I/O stalls. If True, uses default
    ///     buffer size (4). If an integer, uses that as the buffer size.
    ///     If False or None, disables prefetching. Defaults to False.
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
    /// Basic iteration without prefetching:
    ///
    /// >>> for batch in dataset.iter_shard(0, batch_size=64*1024):
    /// ...     records = batch.decode('utf-8').splitlines()
    /// ...     for record in records:
    /// ...         process(json.loads(record))
    ///
    /// With prefetching for reduced I/O stalls:
    ///
    /// >>> for batch in dataset.iter_shard(0, prefetch=True):
    /// ...     process(batch)
    ///
    /// Custom prefetch buffer size:
    ///
    /// >>> for batch in dataset.iter_shard(0, prefetch=8):
    /// ...     process(batch)
    #[pyo3(signature = (shard_id, batch_size=1048576, prefetch=None))]
    fn iter_shard(
        &self,
        shard_id: u32,
        batch_size: usize,
        prefetch: Option<PyPrefetchOption>,
    ) -> PyResult<PyShardIterator> {
        let prefetch_config = match prefetch {
            Some(PyPrefetchOption::Bool(true)) => PrefetchConfig {
                buffer_size: 4,
                enabled: true,
            },
            Some(PyPrefetchOption::Bool(false)) | None => PrefetchConfig {
                buffer_size: 0,
                enabled: false,
            },
            Some(PyPrefetchOption::Int(size)) => PrefetchConfig {
                buffer_size: size,
                enabled: size > 0,
            },
        };

        // Get shard info
        let shard = self.inner.shard_info(shard_id).map_err(WrappedError)?.clone();

        if prefetch_config.enabled {
            let storage = self.storage();
            let format = self.format();
            let path = self.inner.path().to_path_buf();
            let iter_config = IteratorConfig {
                batch_size,
                shard_id,
            };

            let prefetch_iter = PrefetchingIterator::new(
                storage,
                path,
                shard,
                format,
                iter_config,
                prefetch_config,
            );

            Ok(PyShardIterator::new_prefetching_with_shard_id(prefetch_iter, shard_id))
        } else {
            let iter = self
                .inner
                .iter_shard(shard_id, batch_size)
                .map_err(WrappedError)?;
            Ok(PyShardIterator::new(iter))
        }
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

/// Internal enum for iterator type.
enum IteratorKind {
    Regular(ShardIterator),
    Prefetching(PrefetchingIterator),
}

/// An iterator over batches in a shard.
///
/// ShardIterators are created through `Dataset.iter_shard()` and yield
/// batches of bytes from the shard. Each batch ends on a record boundary.
///
/// Attributes
/// ----------
/// shard_id : int
///     The shard ID this iterator is processing.
/// current_offset : int
///     The current byte offset within the file.
/// prefetching : bool
///     Whether prefetching is enabled for this iterator.
///
/// Examples
/// --------
/// >>> iter = dataset.iter_shard(0)
/// >>> for batch in iter:
/// ...     print(f"Got {len(batch)} bytes")
/// >>> print(f"Progress: {iter.progress():.1%}")
#[pyclass(name = "ShardIterator")]
pub struct PyShardIterator {
    kind: IteratorKind,
    shard_id: u32,
}

impl PyShardIterator {
    /// Create a new PyShardIterator wrapping a regular ShardIterator.
    pub fn new(inner: ShardIterator) -> Self {
        let shard_id = inner.shard().shard_id;
        Self {
            kind: IteratorKind::Regular(inner),
            shard_id,
        }
    }

    /// Create a new PyShardIterator wrapping a PrefetchingIterator.
    pub fn new_prefetching(inner: PrefetchingIterator) -> Self {
        // Note: PrefetchingIterator doesn't have a shard() method, so we need
        // to track the shard_id separately. The caller should set this.
        // For now, we'll get it from the config.
        Self {
            kind: IteratorKind::Prefetching(inner),
            shard_id: 0, // Will be set properly by iter_shard
        }
    }

    /// Create a prefetching iterator with the specified shard_id.
    pub fn new_prefetching_with_shard_id(inner: PrefetchingIterator, shard_id: u32) -> Self {
        Self {
            kind: IteratorKind::Prefetching(inner),
            shard_id,
        }
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
        let batch = py.allow_threads(|| match &mut self.kind {
            IteratorKind::Regular(iter) => iter.next_batch(),
            IteratorKind::Prefetching(iter) => iter.next_batch(),
        });

        let batch = batch.map_err(WrappedError)?;

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
        match &self.kind {
            IteratorKind::Regular(iter) => iter.progress(),
            IteratorKind::Prefetching(_) => {
                // Prefetching iterator doesn't track progress directly
                // Return 0.0 as a fallback
                0.0
            }
        }
    }

    /// Reset the iterator to the beginning of the shard.
    ///
    /// Note: This is only supported for non-prefetching iterators.
    fn reset(&mut self) -> PyResult<()> {
        match &mut self.kind {
            IteratorKind::Regular(iter) => {
                iter.reset();
                Ok(())
            }
            IteratorKind::Prefetching(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "reset() is not supported for prefetching iterators",
            )),
        }
    }

    /// Get the current byte offset within the file.
    ///
    /// Returns
    /// -------
    /// int
    ///     The current byte position.
    #[getter]
    fn current_offset(&self) -> u64 {
        match &self.kind {
            IteratorKind::Regular(iter) => iter.current_offset(),
            IteratorKind::Prefetching(_) => 0, // Prefetching doesn't expose this
        }
    }

    /// Get the shard ID this iterator is processing.
    ///
    /// Returns
    /// -------
    /// int
    ///     The shard ID.
    #[getter]
    fn shard_id(&self) -> u32 {
        self.shard_id
    }

    /// Check if prefetching is enabled for this iterator.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if prefetching is enabled.
    #[getter]
    fn prefetching(&self) -> bool {
        match &self.kind {
            IteratorKind::Regular(_) => false,
            IteratorKind::Prefetching(iter) => iter.is_prefetching_enabled(),
        }
    }

    /// Get the number of batches currently in the prefetch queue.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of prefetched batches waiting. Returns 0 if prefetching
    ///     is not enabled.
    fn queue_len(&self) -> usize {
        match &self.kind {
            IteratorKind::Regular(_) => 0,
            IteratorKind::Prefetching(iter) => iter.queue_len(),
        }
    }

    fn __repr__(&self) -> String {
        match &self.kind {
            IteratorKind::Regular(iter) => {
                format!(
                    "ShardIterator(shard_id={}, progress={:.1}%)",
                    iter.shard().shard_id,
                    iter.progress() * 100.0
                )
            }
            IteratorKind::Prefetching(iter) => {
                format!(
                    "ShardIterator(shard_id={}, prefetching=True, queue_len={})",
                    self.shard_id,
                    iter.queue_len()
                )
            }
        }
    }
}
