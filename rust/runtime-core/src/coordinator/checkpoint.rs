//! Distributed checkpoint protocol.
//!
//! This module implements the distributed checkpoint protocol for coordinating
//! checkpoint writes across multiple workers.
//!
//! ## Protocol Flow
//!
//! 1. Any worker (or external trigger) calls `InitiateCheckpoint`
//! 2. Coordinator creates `CheckpointPlan` with assignments for each worker
//! 3. Each worker writes its checkpoint shard to the assigned path
//! 4. Workers call `ConfirmCheckpointShard` with size and checksum
//! 5. Coordinator waits for all confirmations (with timeout)
//! 6. Coordinator writes `CheckpointManifest` listing all shards
//! 7. Atomic commit by renaming manifest to final path

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use twox_hash::XxHash64;

use super::protocol::{
    CheckpointManifest, CheckpointPlan, CheckpointShardConfirmation, CheckpointShardInfo,
};
use crate::checkpoint::{AsyncCheckpointReader, AsyncCheckpointWriter};
use crate::config::CheckpointConfig;
use crate::error::{Result, RuntimeError};
use crate::storage::AsyncStorageBackend;

/// Distributed checkpoint writer for workers.
pub struct DistributedCheckpointWriter {
    storage: std::sync::Arc<dyn AsyncStorageBackend>,
    compression: String,
}

impl DistributedCheckpointWriter {
    /// Create a new distributed checkpoint writer.
    pub fn new(storage: std::sync::Arc<dyn AsyncStorageBackend>, compression: String) -> Self {
        Self {
            storage,
            compression,
        }
    }

    /// Write a checkpoint shard according to the plan.
    ///
    /// # Arguments
    /// * `plan` - The checkpoint plan from the coordinator
    /// * `worker_id` - This worker's ID
    /// * `data` - The checkpoint data (key-value pairs)
    ///
    /// # Returns
    /// Confirmation details to send back to the coordinator.
    pub async fn write_shard(
        &self,
        plan: &CheckpointPlan,
        worker_id: &str,
        data: &HashMap<String, Vec<u8>>,
    ) -> Result<CheckpointShardConfirmation> {
        // Find our assignment
        let assignment = plan
            .assignments
            .iter()
            .find(|a| a.worker_id == worker_id)
            .ok_or_else(|| {
                RuntimeError::coordinator(format!(
                    "no checkpoint assignment for worker {}",
                    worker_id
                ))
            })?;

        // Filter data to only include assigned keys
        let shard_data: HashMap<String, Vec<u8>> = if assignment.assigned_keys.is_empty() {
            // If no specific keys assigned, write all data
            data.clone()
        } else {
            data.iter()
                .filter(|(k, _)| assignment.assigned_keys.contains(k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };

        // Serialize the HashMap to bytes
        let serialized_data = bincode::serialize(&shard_data).map_err(|e| {
            RuntimeError::checkpoint_with_source("failed to serialize shard data", e)
        })?;

        // Write checkpoint using the standard writer
        let shard_path = PathBuf::from(&assignment.shard_path);

        // Ensure parent directory exists
        if let Some(parent) = shard_path.parent() {
            self.storage.create_dir_all(parent).await?;
        }

        // Create checkpoint config for the writer
        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: shard_path
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf(),
            compression: self.compression.clone(),
            atomic_writes: true,
            keep_last_n: usize::MAX, // Don't clean up distributed checkpoint shards
            ..Default::default()
        };

        // Write checkpoint shard
        let writer = AsyncCheckpointWriter::new(self.storage.clone(), checkpoint_config);

        // Extract shard name from path
        let shard_name = shard_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("shard");

        writer.write(shard_name, &serialized_data).await?;

        // Calculate checksum and size
        let meta = self.storage.metadata(&shard_path).await?;
        let size_bytes = meta.size;

        // Read back and compute checksum
        let mut reader = self.storage.open_read(&shard_path).await?;
        let content = reader.read_all().await?;
        let checksum = compute_checksum(&content);

        Ok(CheckpointShardConfirmation {
            worker_id: worker_id.to_string(),
            session_token: String::new(), // Will be filled by client
            checkpoint_id: plan.checkpoint_id.clone(),
            shard_path: assignment.shard_path.clone(),
            shard_index: assignment.shard_index,
            size_bytes,
            checksum,
            success: true,
            error_message: String::new(),
        })
    }

    /// Create a failed confirmation for error reporting.
    pub fn create_failure_confirmation(
        plan: &CheckpointPlan,
        worker_id: &str,
        shard_index: u32,
        error: &str,
    ) -> CheckpointShardConfirmation {
        CheckpointShardConfirmation {
            worker_id: worker_id.to_string(),
            session_token: String::new(),
            checkpoint_id: plan.checkpoint_id.clone(),
            shard_path: String::new(),
            shard_index,
            size_bytes: 0,
            checksum: 0,
            success: false,
            error_message: error.to_string(),
        }
    }
}

/// Distributed checkpoint reader for loading sharded checkpoints.
pub struct DistributedCheckpointReader {
    storage: std::sync::Arc<dyn AsyncStorageBackend>,
}

impl DistributedCheckpointReader {
    /// Create a new distributed checkpoint reader.
    pub fn new(storage: std::sync::Arc<dyn AsyncStorageBackend>) -> Self {
        Self { storage }
    }

    /// Read the checkpoint manifest.
    pub async fn read_manifest(&self, manifest_path: &Path) -> Result<CheckpointManifest> {
        let mut reader = self.storage.open_read(manifest_path).await?;
        let content = reader.read_all().await?;

        let manifest: super::proto::CheckpointManifest =
            bincode::deserialize(&content).map_err(|e| {
                RuntimeError::checkpoint_with_source("failed to deserialize manifest", e)
            })?;

        Ok(CheckpointManifest::from(manifest))
    }

    /// Load a complete distributed checkpoint from a manifest.
    ///
    /// This reads all shards and combines them into a single data map.
    pub async fn load_checkpoint(
        &self,
        manifest_path: &Path,
    ) -> Result<(CheckpointManifest, HashMap<String, Vec<u8>>)> {
        let manifest = self.read_manifest(manifest_path).await?;

        // Verify manifest
        self.verify_manifest(&manifest).await?;

        // Load all shards
        let mut all_data = HashMap::new();

        for shard_info in &manifest.shards {
            let shard_path = PathBuf::from(&shard_info.path);
            let shard_data = self.load_shard(&shard_path, shard_info).await?;
            all_data.extend(shard_data);
        }

        Ok((manifest, all_data))
    }

    /// Load a single checkpoint shard.
    async fn load_shard(
        &self,
        path: &Path,
        info: &CheckpointShardInfo,
    ) -> Result<HashMap<String, Vec<u8>>> {
        // Read and verify checksum
        let mut reader = self.storage.open_read(path).await?;
        let content = reader.read_all().await?;

        let checksum = compute_checksum(&content);
        if checksum != info.checksum {
            return Err(RuntimeError::checkpoint(format!(
                "checksum mismatch for shard {}: expected {}, got {}",
                info.shard_index, info.checksum, checksum
            )));
        }

        // Use checkpoint reader to get decompressed data
        let checkpoint_reader = AsyncCheckpointReader::new(self.storage.clone());
        let decompressed_data = checkpoint_reader.read(path).await?;

        // Deserialize the HashMap from bytes
        let data: HashMap<String, Vec<u8>> =
            bincode::deserialize(&decompressed_data).map_err(|e| {
                RuntimeError::checkpoint_with_source("failed to deserialize shard data", e)
            })?;

        Ok(data)
    }

    /// Verify manifest integrity.
    async fn verify_manifest(&self, manifest: &CheckpointManifest) -> Result<()> {
        // Verify total checksum
        let combined_checksum = compute_combined_checksum(&manifest.shards);
        if combined_checksum != manifest.total_checksum {
            return Err(RuntimeError::checkpoint(format!(
                "manifest checksum mismatch: expected {}, got {}",
                manifest.total_checksum, combined_checksum
            )));
        }

        // Verify shard count
        if manifest.shards.len() != manifest.worker_count as usize {
            return Err(RuntimeError::checkpoint(format!(
                "shard count mismatch: expected {}, got {}",
                manifest.worker_count,
                manifest.shards.len()
            )));
        }

        // Verify all shards exist
        for shard in &manifest.shards {
            let path = PathBuf::from(&shard.path);
            if !self.storage.exists(&path).await? {
                return Err(RuntimeError::checkpoint(format!(
                    "missing checkpoint shard: {}",
                    shard.path
                )));
            }
        }

        Ok(())
    }

    /// List available distributed checkpoints in a directory.
    pub async fn list_checkpoints(&self, base_path: &Path) -> Result<Vec<CheckpointManifest>> {
        let entries = self.storage.list(base_path).await?;

        let mut manifests = Vec::new();
        for entry in entries {
            if entry.ends_with(".manifest") {
                let manifest_path = base_path.join(&entry);
                match self.read_manifest(&manifest_path).await {
                    Ok(manifest) => manifests.push(manifest),
                    Err(e) => {
                        // Log but continue
                        tracing::warn!("Failed to read manifest {}: {}", entry, e);
                    }
                }
            }
        }

        // Sort by step (newest first)
        manifests.sort_by(|a, b| b.step.cmp(&a.step));

        Ok(manifests)
    }
}

/// Compute xxHash64 checksum of data.
fn compute_checksum(data: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut hasher = XxHash64::default();
    hasher.write(data);
    hasher.finish()
}

/// Compute combined checksum of all shards.
fn compute_combined_checksum(shards: &[CheckpointShardInfo]) -> u64 {
    use std::hash::Hasher;
    let mut hasher = XxHash64::default();

    // Sort by shard index for deterministic order
    let mut sorted: Vec<_> = shards.iter().collect();
    sorted.sort_by_key(|s| s.shard_index);

    for shard in sorted {
        hasher.write_u64(shard.checksum);
    }

    hasher.finish()
}

/// Builder for creating checkpoint manifests (used by coordinator).
#[derive(Debug)]
pub struct ManifestBuilder {
    checkpoint_id: String,
    checkpoint_name: String,
    step: u64,
    metadata: HashMap<String, String>,
    shards: Vec<CheckpointShardInfo>,
}

impl ManifestBuilder {
    /// Create a new manifest builder.
    pub fn new(checkpoint_id: String, checkpoint_name: String, step: u64) -> Self {
        Self {
            checkpoint_id,
            checkpoint_name,
            step,
            metadata: HashMap::new(),
            shards: Vec::new(),
        }
    }

    /// Add metadata.
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Add a confirmed shard.
    pub fn add_shard(&mut self, confirmation: &CheckpointShardConfirmation, keys: Vec<String>) {
        self.shards.push(CheckpointShardInfo {
            shard_index: confirmation.shard_index,
            worker_id: confirmation.worker_id.clone(),
            path: confirmation.shard_path.clone(),
            size_bytes: confirmation.size_bytes,
            checksum: confirmation.checksum,
            keys,
        });
    }

    /// Build the manifest.
    pub fn build(self) -> CheckpointManifest {
        let total_size: u64 = self.shards.iter().map(|s| s.size_bytes).sum();
        let total_checksum = compute_combined_checksum(&self.shards);
        let worker_count = self.shards.len() as u32;

        CheckpointManifest {
            checkpoint_id: self.checkpoint_id,
            checkpoint_name: self.checkpoint_name,
            step: self.step,
            created_at_ms: chrono::Utc::now().timestamp_millis(),
            shards: self.shards,
            total_size_bytes: total_size,
            total_checksum,
            metadata: self.metadata,
            worker_count,
            format_version: 1,
        }
    }
}

/// Write a checkpoint manifest to storage.
pub async fn write_manifest(
    storage: &dyn AsyncStorageBackend,
    manifest: &CheckpointManifest,
    base_path: &Path,
) -> Result<PathBuf> {
    // Serialize manifest
    let proto_manifest: super::proto::CheckpointManifest = manifest.clone().into();
    let content = bincode::serialize(&proto_manifest)
        .map_err(|e| RuntimeError::checkpoint_with_source("failed to serialize manifest", e))?;

    // Write to temp path first
    let temp_path = base_path.join(format!("{}.manifest.tmp", manifest.checkpoint_id));
    let final_path = base_path.join(format!("{}.manifest", manifest.checkpoint_name));

    storage.create_dir_all(base_path).await?;

    let mut writer = storage.open_write(&temp_path).await?;
    writer.write_all_bytes(&content).await?;
    Box::new(writer).finish().await?;

    // Atomic rename
    storage.rename(&temp_path, &final_path).await?;

    Ok(final_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_checksum() {
        let data = b"hello world";
        let checksum1 = compute_checksum(data);
        let checksum2 = compute_checksum(data);
        assert_eq!(checksum1, checksum2);

        let different = b"hello World";
        let checksum3 = compute_checksum(different);
        assert_ne!(checksum1, checksum3);
    }

    #[test]
    fn test_combined_checksum_order_independent() {
        let shards1 = vec![
            CheckpointShardInfo {
                shard_index: 0,
                worker_id: "w0".to_string(),
                path: "/path/0".to_string(),
                size_bytes: 100,
                checksum: 12345,
                keys: vec![],
            },
            CheckpointShardInfo {
                shard_index: 1,
                worker_id: "w1".to_string(),
                path: "/path/1".to_string(),
                size_bytes: 200,
                checksum: 67890,
                keys: vec![],
            },
        ];

        let shards2 = vec![
            CheckpointShardInfo {
                shard_index: 1,
                worker_id: "w1".to_string(),
                path: "/path/1".to_string(),
                size_bytes: 200,
                checksum: 67890,
                keys: vec![],
            },
            CheckpointShardInfo {
                shard_index: 0,
                worker_id: "w0".to_string(),
                path: "/path/0".to_string(),
                size_bytes: 100,
                checksum: 12345,
                keys: vec![],
            },
        ];

        // Order shouldn't matter since we sort by shard_index
        assert_eq!(
            compute_combined_checksum(&shards1),
            compute_combined_checksum(&shards2)
        );
    }

    #[test]
    fn test_manifest_builder() {
        let mut builder = ManifestBuilder::new(
            "ckpt-123".to_string(),
            "epoch_10".to_string(),
            1000,
        );

        builder.add_shard(
            &CheckpointShardConfirmation {
                worker_id: "w0".to_string(),
                session_token: String::new(),
                checkpoint_id: "ckpt-123".to_string(),
                shard_path: "/ckpts/epoch_10/shard_0".to_string(),
                shard_index: 0,
                size_bytes: 1024,
                checksum: 12345,
                success: true,
                error_message: String::new(),
            },
            vec!["model.layer1".to_string()],
        );

        builder.add_shard(
            &CheckpointShardConfirmation {
                worker_id: "w1".to_string(),
                session_token: String::new(),
                checkpoint_id: "ckpt-123".to_string(),
                shard_path: "/ckpts/epoch_10/shard_1".to_string(),
                shard_index: 1,
                size_bytes: 2048,
                checksum: 67890,
                success: true,
                error_message: String::new(),
            },
            vec!["model.layer2".to_string()],
        );

        let manifest = builder.build();

        assert_eq!(manifest.checkpoint_id, "ckpt-123");
        assert_eq!(manifest.checkpoint_name, "epoch_10");
        assert_eq!(manifest.step, 1000);
        assert_eq!(manifest.shards.len(), 2);
        assert_eq!(manifest.worker_count, 2);
        assert_eq!(manifest.total_size_bytes, 3072);
    }
}
