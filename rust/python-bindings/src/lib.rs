// rust/python-bindings/src/lib.rs

//! Python bindings for the distributed training runtime.
//!
//! This module exposes the Rust runtime to Python using PyO3.

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use runtime_core::RuntimeError;

mod dataset;
mod runtime;

pub use dataset::{PyDataset, PyShardIterator};
pub use runtime::PyRuntime;

/// Wrapper for RuntimeError to enable conversion to PyErr.
///
/// This newtype pattern allows us to implement the conversion trait
/// since both RuntimeError and PyErr are foreign types.
pub struct WrappedError(pub RuntimeError);

impl From<RuntimeError> for WrappedError {
    fn from(err: RuntimeError) -> Self {
        WrappedError(err)
    }
}

impl From<WrappedError> for PyErr {
    fn from(err: WrappedError) -> PyErr {
        match &err.0 {
            RuntimeError::Storage { .. } => PyIOError::new_err(err.0.to_string()),
            RuntimeError::InvalidShard { .. } => PyValueError::new_err(err.0.to_string()),
            RuntimeError::Config { .. } => PyValueError::new_err(err.0.to_string()),
            RuntimeError::Dataset { .. } => PyValueError::new_err(err.0.to_string()),
            RuntimeError::Checkpoint { .. } => PyRuntimeError::new_err(err.0.to_string()),
            RuntimeError::Serialization { .. } => PyRuntimeError::new_err(err.0.to_string()),
            // Handle coordinator errors when runtime-core is compiled with coordinator feature
            // This can happen due to Cargo feature unification across the workspace
            #[allow(unreachable_patterns)]
            _ => PyRuntimeError::new_err(err.0.to_string()),
        }
    }
}

/// The main Python module for the distributed training runtime.
///
/// This module provides:
/// - `Runtime`: The main runtime class for managing datasets and checkpoints
/// - `Dataset`: A registered dataset with shard information
/// - `ShardIterator`: An iterator over batches in a shard
#[pymodule]
fn _dtr_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyDataset>()?;
    m.add_class::<PyShardIterator>()?;
    Ok(())
}
