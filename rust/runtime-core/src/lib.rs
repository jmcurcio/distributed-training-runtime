// rust/runtime-core/src/lib.rs

//! Distributed Training Runtime - Core Library
//!
//! This crate provides core functionality for distributed training workloads,
//! including error handling, storage abstractions, dataset management, and
//! checkpoint operations.

pub mod config;
pub mod error;

// Re-export commonly used types for convenience
pub use config::RuntimeConfig;
pub use error::{Result, RuntimeError};

// Placeholder modules for future implementation
pub mod checkpoint;
pub mod dataset;
pub mod runtime;
pub mod storage;