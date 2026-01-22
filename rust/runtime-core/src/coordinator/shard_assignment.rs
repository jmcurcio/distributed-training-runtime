//! Shard assignment strategies for distributed training.
//!
//! This module provides different strategies for assigning data shards
//! to workers in a distributed training job.

use std::collections::HashMap;

use super::protocol::{ShardRange, WorkerCapabilities, WorkerInfo};
use crate::config::ShardStrategy;

/// Assignment of shards to a worker.
#[derive(Debug, Clone)]
pub struct WorkerAssignment {
    /// Worker ID.
    pub worker_id: String,
    /// Worker index (0-based).
    pub worker_index: u32,
    /// Assigned shard ranges.
    pub ranges: Vec<ShardRange>,
}

impl WorkerAssignment {
    /// Create a new worker assignment.
    pub fn new(worker_id: String, worker_index: u32) -> Self {
        Self {
            worker_id,
            worker_index,
            ranges: Vec::new(),
        }
    }

    /// Add a shard range to this assignment.
    pub fn add_range(&mut self, range: ShardRange) {
        self.ranges.push(range);
    }

    /// Get total number of shards assigned.
    pub fn total_shards(&self) -> u32 {
        self.ranges.iter().map(|r| r.count()).sum()
    }

    /// Get all shard IDs assigned to this worker.
    pub fn shard_ids(&self) -> Vec<u32> {
        self.ranges.iter().flat_map(|r| r.iter()).collect()
    }
}

/// Trait for shard assignment strategies.
pub trait ShardAssigner: Send + Sync {
    /// Assign shards to workers.
    ///
    /// # Arguments
    /// * `total_shards` - Total number of shards in the dataset
    /// * `workers` - List of workers to assign shards to
    ///
    /// # Returns
    /// A list of worker assignments, one per worker.
    fn assign_shards(&self, total_shards: u32, workers: &[WorkerInfo]) -> Vec<WorkerAssignment>;

    /// Rebalance shard assignments when workers change.
    ///
    /// # Arguments
    /// * `current` - Current assignments
    /// * `workers` - Updated list of workers
    /// * `total_shards` - Total number of shards
    ///
    /// # Returns
    /// New assignments, potentially redistributing shards.
    fn rebalance(
        &self,
        current: &[WorkerAssignment],
        workers: &[WorkerInfo],
        total_shards: u32,
    ) -> Vec<WorkerAssignment>;

    /// Get the strategy type.
    fn strategy(&self) -> ShardStrategy;
}

/// Static shard assignment: round-robin at job start.
///
/// Shards are divided evenly among workers based on their index.
/// Assignments don't change unless workers are added/removed.
#[derive(Debug, Default)]
pub struct StaticAssigner;

impl ShardAssigner for StaticAssigner {
    fn assign_shards(&self, total_shards: u32, workers: &[WorkerInfo]) -> Vec<WorkerAssignment> {
        if workers.is_empty() {
            return Vec::new();
        }

        let num_workers = workers.len() as u32;
        let shards_per_worker = total_shards / num_workers;
        let remainder = total_shards % num_workers;

        let mut assignments = Vec::with_capacity(workers.len());
        let mut start = 0u32;

        for (i, worker) in workers.iter().enumerate() {
            let idx = i as u32;
            // Workers with index < remainder get one extra shard
            let extra = if idx < remainder { 1 } else { 0 };
            let count = shards_per_worker + extra;

            let mut assignment = WorkerAssignment::new(worker.worker_id.clone(), worker.worker_index);

            if count > 0 {
                assignment.add_range(ShardRange::new(start, start + count));
            }

            assignments.push(assignment);
            start += count;
        }

        assignments
    }

    fn rebalance(
        &self,
        _current: &[WorkerAssignment],
        workers: &[WorkerInfo],
        total_shards: u32,
    ) -> Vec<WorkerAssignment> {
        // Static assigner just redistributes from scratch
        self.assign_shards(total_shards, workers)
    }

    fn strategy(&self) -> ShardStrategy {
        ShardStrategy::Static
    }
}

/// Dynamic shard assignment: rebalance as workers join/leave.
///
/// Attempts to minimize data movement when workers change by
/// only reassigning shards from departed workers.
#[derive(Debug, Default)]
pub struct DynamicAssigner;

impl ShardAssigner for DynamicAssigner {
    fn assign_shards(&self, total_shards: u32, workers: &[WorkerInfo]) -> Vec<WorkerAssignment> {
        // Initial assignment is the same as static
        StaticAssigner.assign_shards(total_shards, workers)
    }

    fn rebalance(
        &self,
        current: &[WorkerAssignment],
        workers: &[WorkerInfo],
        total_shards: u32,
    ) -> Vec<WorkerAssignment> {
        if workers.is_empty() {
            return Vec::new();
        }

        // Build map of current worker IDs
        let active_workers: HashMap<_, _> = workers
            .iter()
            .map(|w| (w.worker_id.clone(), w))
            .collect();

        // Find orphaned shards (from departed workers)
        let mut orphaned_shards: Vec<u32> = Vec::new();
        let mut surviving_assignments: Vec<WorkerAssignment> = Vec::new();

        for assignment in current {
            if active_workers.contains_key(&assignment.worker_id) {
                surviving_assignments.push(assignment.clone());
            } else {
                // Worker departed - collect their shards
                orphaned_shards.extend(assignment.shard_ids());
            }
        }

        // Find new workers without assignments
        let mut new_workers: Vec<&WorkerInfo> = workers
            .iter()
            .filter(|w| !current.iter().any(|a| a.worker_id == w.worker_id))
            .collect();

        // If no orphaned shards and no new workers, return current assignments
        if orphaned_shards.is_empty() && new_workers.is_empty() {
            return surviving_assignments;
        }

        // Distribute orphaned shards
        if !orphaned_shards.is_empty() {
            if surviving_assignments.is_empty() && !new_workers.is_empty() {
                // All workers departed, assign to new workers
                let mut assignments = Vec::new();
                for worker in new_workers.iter() {
                    assignments.push(WorkerAssignment::new(
                        worker.worker_id.clone(),
                        worker.worker_index,
                    ));
                }

                let shards_per_worker = orphaned_shards.len() / assignments.len();
                let remainder = orphaned_shards.len() % assignments.len();

                let mut shard_idx = 0;
                for (i, assignment) in assignments.iter_mut().enumerate() {
                    let extra = if i < remainder { 1 } else { 0 };
                    let count = shards_per_worker + extra;

                    for _ in 0..count {
                        if shard_idx < orphaned_shards.len() {
                            let shard = orphaned_shards[shard_idx];
                            assignment.add_range(ShardRange::new(shard, shard + 1));
                            shard_idx += 1;
                        }
                    }
                }

                return assignments;
            }

            // Distribute orphaned shards among surviving + new workers
            let all_workers: Vec<_> = surviving_assignments
                .iter()
                .map(|a| &a.worker_id)
                .chain(new_workers.iter().map(|w| &w.worker_id))
                .collect();

            // Add new workers to assignments
            for worker in &new_workers {
                surviving_assignments.push(WorkerAssignment::new(
                    worker.worker_id.clone(),
                    worker.worker_index,
                ));
            }

            // Round-robin distribute orphaned shards
            for (i, shard) in orphaned_shards.iter().enumerate() {
                let target_idx = i % surviving_assignments.len();
                surviving_assignments[target_idx].add_range(ShardRange::new(*shard, *shard + 1));
            }
        } else if !new_workers.is_empty() {
            // No orphaned shards but new workers - steal some shards
            let target_per_worker =
                total_shards as usize / (surviving_assignments.len() + new_workers.len());

            // Find workers with more than target
            let mut to_redistribute: Vec<u32> = Vec::new();

            for assignment in &mut surviving_assignments {
                while assignment.total_shards() as usize > target_per_worker {
                    if let Some(shard) = assignment.shard_ids().pop() {
                        to_redistribute.push(shard);
                        // Remove last range if it had only that shard
                        if let Some(last) = assignment.ranges.last() {
                            if last.count() == 1 {
                                assignment.ranges.pop();
                            } else {
                                // Shrink the range
                                if let Some(r) = assignment.ranges.last_mut() {
                                    *r = ShardRange::new(r.start_shard, r.end_shard - 1);
                                }
                            }
                        }
                    } else {
                        break;
                    }
                }
            }

            // Add new workers and give them shards
            for worker in new_workers {
                let mut assignment =
                    WorkerAssignment::new(worker.worker_id.clone(), worker.worker_index);

                while assignment.total_shards() < target_per_worker as u32 {
                    if let Some(shard) = to_redistribute.pop() {
                        assignment.add_range(ShardRange::new(shard, shard + 1));
                    } else {
                        break;
                    }
                }

                surviving_assignments.push(assignment);
            }
        }

        // Consolidate ranges
        for assignment in &mut surviving_assignments {
            assignment.ranges = consolidate_ranges(&assignment.ranges);
        }

        surviving_assignments
    }

    fn strategy(&self) -> ShardStrategy {
        ShardStrategy::Dynamic
    }
}

/// Locality-aware shard assignment.
///
/// Prefers assigning shards to workers that have local access to the data.
#[derive(Debug, Default)]
pub struct LocalityAwareAssigner {
    /// Mapping of shard ID to preferred paths.
    shard_locations: HashMap<u32, Vec<String>>,
}

impl LocalityAwareAssigner {
    /// Create a new locality-aware assigner.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the location preferences for shards.
    pub fn set_shard_locations(&mut self, locations: HashMap<u32, Vec<String>>) {
        self.shard_locations = locations;
    }

    /// Check if a worker can access a shard locally.
    fn can_access_locally(&self, worker: &WorkerInfo, shard: u32) -> bool {
        if let Some(locations) = self.shard_locations.get(&shard) {
            locations
                .iter()
                .any(|loc| worker.capabilities.accessible_paths.contains(loc))
        } else {
            false
        }
    }
}

impl ShardAssigner for LocalityAwareAssigner {
    fn assign_shards(&self, total_shards: u32, workers: &[WorkerInfo]) -> Vec<WorkerAssignment> {
        if workers.is_empty() {
            return Vec::new();
        }

        let mut assignments: Vec<_> = workers
            .iter()
            .map(|w| WorkerAssignment::new(w.worker_id.clone(), w.worker_index))
            .collect();

        let mut unassigned: Vec<u32> = (0..total_shards).collect();

        // First pass: assign shards to workers with local access
        let mut still_unassigned = Vec::new();
        for shard in unassigned {
            let mut assigned = false;

            // Find worker with local access and fewest shards
            let local_workers: Vec<_> = workers
                .iter()
                .enumerate()
                .filter(|(_, w)| self.can_access_locally(w, shard))
                .collect();

            if !local_workers.is_empty() {
                // Pick worker with fewest shards
                let min_idx = local_workers
                    .iter()
                    .min_by_key(|(i, _)| assignments[*i].total_shards())
                    .map(|(i, _)| *i)
                    .unwrap();

                assignments[min_idx].add_range(ShardRange::new(shard, shard + 1));
                assigned = true;
            }

            if !assigned {
                still_unassigned.push(shard);
            }
        }

        // Second pass: distribute remaining shards evenly
        for (i, shard) in still_unassigned.iter().enumerate() {
            let target = i % assignments.len();
            assignments[target].add_range(ShardRange::new(*shard, *shard + 1));
        }

        // Consolidate ranges
        for assignment in &mut assignments {
            assignment.ranges = consolidate_ranges(&assignment.ranges);
        }

        assignments
    }

    fn rebalance(
        &self,
        current: &[WorkerAssignment],
        workers: &[WorkerInfo],
        total_shards: u32,
    ) -> Vec<WorkerAssignment> {
        // For locality-aware, recompute from scratch to optimize locality
        // TODO: Implement incremental rebalancing that preserves locality
        self.assign_shards(total_shards, workers)
    }

    fn strategy(&self) -> ShardStrategy {
        ShardStrategy::LocalityAware
    }
}

/// Consolidate adjacent shard ranges.
fn consolidate_ranges(ranges: &[ShardRange]) -> Vec<ShardRange> {
    if ranges.is_empty() {
        return Vec::new();
    }

    let mut sorted: Vec<_> = ranges.to_vec();
    sorted.sort_by_key(|r| r.start_shard);

    let mut consolidated = Vec::new();
    let mut current = sorted[0];

    for range in sorted.into_iter().skip(1) {
        if range.start_shard <= current.end_shard {
            // Overlapping or adjacent
            current.end_shard = current.end_shard.max(range.end_shard);
        } else {
            consolidated.push(current);
            current = range;
        }
    }
    consolidated.push(current);

    consolidated
}

/// Create a shard assigner from a strategy.
pub fn create_assigner(strategy: ShardStrategy) -> Box<dyn ShardAssigner> {
    match strategy {
        ShardStrategy::Static => Box::new(StaticAssigner),
        ShardStrategy::Dynamic => Box::new(DynamicAssigner),
        ShardStrategy::LocalityAware => Box::new(LocalityAwareAssigner::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_worker(id: &str, index: u32) -> WorkerInfo {
        WorkerInfo {
            worker_id: id.to_string(),
            worker_index: index,
            status: super::super::protocol::WorkerStatus::Idle,
            last_heartbeat_ms: 0,
            step: 0,
            capabilities: WorkerCapabilities::default(),
            healthy: true,
            hostname: "localhost".to_string(),
        }
    }

    #[test]
    fn test_static_assignment_even() {
        let assigner = StaticAssigner;
        let workers = vec![
            make_worker("w0", 0),
            make_worker("w1", 1),
            make_worker("w2", 2),
            make_worker("w3", 3),
        ];

        let assignments = assigner.assign_shards(8, &workers);

        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0].total_shards(), 2);
        assert_eq!(assignments[1].total_shards(), 2);
        assert_eq!(assignments[2].total_shards(), 2);
        assert_eq!(assignments[3].total_shards(), 2);

        // Check no overlap
        let mut all_shards: Vec<_> = assignments.iter().flat_map(|a| a.shard_ids()).collect();
        all_shards.sort();
        assert_eq!(all_shards, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_static_assignment_uneven() {
        let assigner = StaticAssigner;
        let workers = vec![make_worker("w0", 0), make_worker("w1", 1), make_worker("w2", 2)];

        let assignments = assigner.assign_shards(10, &workers);

        assert_eq!(assignments.len(), 3);
        // 10 / 3 = 3 with remainder 1
        // First worker gets 4, others get 3
        assert_eq!(assignments[0].total_shards(), 4);
        assert_eq!(assignments[1].total_shards(), 3);
        assert_eq!(assignments[2].total_shards(), 3);
    }

    #[test]
    fn test_static_assignment_more_workers_than_shards() {
        let assigner = StaticAssigner;
        let workers = vec![
            make_worker("w0", 0),
            make_worker("w1", 1),
            make_worker("w2", 2),
            make_worker("w3", 3),
        ];

        let assignments = assigner.assign_shards(2, &workers);

        // Only first 2 workers get shards
        assert_eq!(assignments[0].total_shards(), 1);
        assert_eq!(assignments[1].total_shards(), 1);
        assert_eq!(assignments[2].total_shards(), 0);
        assert_eq!(assignments[3].total_shards(), 0);
    }

    #[test]
    fn test_dynamic_rebalance_worker_departure() {
        let assigner = DynamicAssigner;

        // Initial state: 2 workers with 4 shards each
        let initial = vec![
            {
                let mut a = WorkerAssignment::new("w0".to_string(), 0);
                a.add_range(ShardRange::new(0, 4));
                a
            },
            {
                let mut a = WorkerAssignment::new("w1".to_string(), 1);
                a.add_range(ShardRange::new(4, 8));
                a
            },
        ];

        // Worker w1 departed
        let remaining_workers = vec![make_worker("w0", 0)];

        let rebalanced = assigner.rebalance(&initial, &remaining_workers, 8);

        // w0 should have all 8 shards now
        assert_eq!(rebalanced.len(), 1);
        assert_eq!(rebalanced[0].total_shards(), 8);
    }

    #[test]
    fn test_consolidate_ranges() {
        let ranges = vec![
            ShardRange::new(0, 2),
            ShardRange::new(2, 4),
            ShardRange::new(6, 8),
        ];

        let consolidated = consolidate_ranges(&ranges);

        assert_eq!(consolidated.len(), 2);
        assert_eq!(consolidated[0].start_shard, 0);
        assert_eq!(consolidated[0].end_shard, 4);
        assert_eq!(consolidated[1].start_shard, 6);
        assert_eq!(consolidated[1].end_shard, 8);
    }

    #[test]
    fn test_worker_assignment_shard_ids() {
        let mut assignment = WorkerAssignment::new("w0".to_string(), 0);
        assignment.add_range(ShardRange::new(0, 3));
        assignment.add_range(ShardRange::new(5, 7));

        let ids = assignment.shard_ids();
        assert_eq!(ids, vec![0, 1, 2, 5, 6]);
    }

    #[test]
    fn test_eight_workers_unique_shards() {
        let assigner = StaticAssigner;
        let workers: Vec<_> = (0..8).map(|i| make_worker(&format!("w{}", i), i)).collect();

        let assignments = assigner.assign_shards(64, &workers);

        // Verify each worker gets 8 shards
        for (i, assignment) in assignments.iter().enumerate() {
            assert_eq!(
                assignment.total_shards(),
                8,
                "Worker {} should have 8 shards",
                i
            );
        }

        // Verify all shards are assigned exactly once
        let mut all_shards: Vec<_> = assignments.iter().flat_map(|a| a.shard_ids()).collect();
        all_shards.sort();
        let expected: Vec<_> = (0..64).collect();
        assert_eq!(all_shards, expected);

        // Verify no duplicates
        let unique: std::collections::HashSet<_> = all_shards.iter().collect();
        assert_eq!(unique.len(), 64);
    }
}
