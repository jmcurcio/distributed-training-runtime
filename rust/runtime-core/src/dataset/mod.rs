// rust/runtime-core/src/dataset/mod.rs

//! Dataset management with sharding support.
//!
//! This module provides functionality for dividing datasets into shards
//! and iterating over batches within shards. It supports multiple record
//! formats including fixed-size records, newline-delimited records, and
//! length-prefixed records.
//!
//! # Features
//!
//! - **Sharding**: Divide large datasets into non-overlapping shards for parallel processing
//! - **Record Formats**: Support for fixed-size, newline-delimited, and length-prefixed records
//! - **Prefetching**: Background I/O to reduce stalls during training
//! - **Parallel Loading**: Concurrent loading of multiple shards
//!
//! # Example (Basic)
//!
//! ```ignore
//! use runtime_core::dataset::{calculate_shards, ShardIterator, IteratorConfig, NewlineDelimitedFormat};
//! use std::sync::Arc;
//!
//! // Calculate shards for a file
//! let mut reader = storage.open_read(Path::new("data.jsonl"))?;
//! let format = NewlineDelimitedFormat::new();
//! let shards = calculate_shards(&mut *reader, 4, &format)?;
//!
//! // Iterate over a shard
//! let config = IteratorConfig { batch_size: 64 * 1024, shard_id: 0 };
//! let iter = ShardIterator::new(
//!     storage.clone(),
//!     PathBuf::from("data.jsonl"),
//!     shards[0].clone(),
//!     Arc::new(format),
//!     config,
//! );
//!
//! for batch in iter {
//!     let batch = batch?;
//!     // Process batch.data
//! }
//! ```
//!
//! # Example (With Prefetching)
//!
//! ```ignore
//! use runtime_core::dataset::{PrefetchingIterator, PrefetchConfig};
//!
//! let prefetch_config = PrefetchConfig {
//!     buffer_size: 4,
//!     enabled: true,
//! };
//!
//! let iter = PrefetchingIterator::new(
//!     storage,
//!     path,
//!     shard,
//!     format,
//!     iter_config,
//!     prefetch_config,
//! );
//!
//! for batch in iter {
//!     // Batches are prefetched in background
//! }
//! ```

mod iterator;
mod sharding;
mod traits;

pub use iterator::{IteratorConfig, ShardIterator};
pub use sharding::calculate_shards;
pub use traits::{
    Batch, FixedSizeFormat, LengthPrefixedFormat, NewlineDelimitedFormat, RecordFormat, ShardSpec,
};

// Prefetching support
mod prefetch;
pub use prefetch::{AsyncPrefetchingIterator, PrefetchConfig, PrefetchingIterator};

// Parallel loading support
mod parallel;
pub use parallel::{
    BatchWithShard, ParallelLoadConfig, ParallelShardLoader, ParallelShardLoaderBuilder,
    RoundRobinShardIterator,
};
