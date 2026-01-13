// rust/runtime-core/src/error.rs

use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RuntimeError {

    #[error("Storage error at '{path}': {message}")]
    Storage {
        path: PathBuf,
        message: String,
        #[source]
        source: Option<std::io::Error>,
    },

    #[error("Dataset '{name}' error: {message}")]
    Dataset {
        name: String,
        message: String,
    },

    #[error("Configuration error: {message}")]
    Config {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Checkpoint error: {message}")]
    Checkpoint {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Shard {shard_id} out of range (total shards: {total_shards})")]
    InvalidShard {
        shard_id: u32,
        total_shards: u32,
    },

    #[error("Serialization error: {message}")]
    Serialization {
        message: String,
    },
}

pub type Result<T> = std::result::Result<T, RuntimeError>;

// Convenience constructors
impl RuntimeError {

    pub fn storage(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self::Storage {
            path: path.into(),
            message: message.into(),
            source: None,
        }
    }

    pub fn storage_with_source(
        path: impl Into<PathBuf>,
        message: impl Into<String>,
        source: std::io::Error,
    ) -> Self {
        Self::Storage {
            path: path.into(),
            message: message.into(),
            source: Some(source),
        }
    }

    pub fn dataset(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Dataset {
            name: name.into(),
            message: message.into(),
        }
    }

    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
            source: None,
        }
    }

    pub fn config_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Config {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub fn checkpoint(message: impl Into<String>) -> Self {
        Self::Checkpoint {
            message: message.into(),
            source: None,
        }
    }

    pub fn checkpoint_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Checkpoint {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub fn invalid_shard(shard_id: u32, total_shards: u32) -> Self {
        Self::InvalidShard { shard_id, total_shards }
    }

    pub fn serialization(message: impl Into<String>) -> Self {
        Self::Serialization {
            message: message.into(),
        }
    }
}