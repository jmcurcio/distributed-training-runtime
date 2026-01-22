#!/usr/bin/env python3
"""
Multiprocess Data Loading - DTR Distributed Examples

Demonstrates:
- Loading data in parallel using Python multiprocessing
- Each process handles one shard independently
- Aggregating results from multiple workers

Prerequisites:
    cd rust/python-bindings && maturin develop
"""

import json
import tempfile
from multiprocessing import Process, Queue
from pathlib import Path

from dtr import Runtime


def create_test_dataset(data_dir: Path, num_records: int = 1000) -> Path:
    """Create a test dataset."""
    data_file = data_dir / "data.jsonl"
    with open(data_file, 'w') as f:
        for i in range(num_records):
            record = {"id": i, "value": i * 0.5}
            f.write(json.dumps(record) + '\n')
    return data_file


def worker_process(shard_id: int, config_path: str, data_path: str, result_queue: Queue):
    """
    Worker process that handles one shard.

    Each worker creates its own Runtime instance and processes
    its assigned shard independently.
    """
    # Each process creates its own runtime
    runtime = Runtime(config_path)
    dataset = runtime.register_dataset(data_path, shards=4, format="newline")

    record_count = 0
    byte_count = 0
    value_sum = 0.0

    for batch in dataset.iter_shard(shard_id, batch_size=4096):
        records = batch.decode('utf-8').splitlines()
        for record_str in records:
            if record_str.strip():
                record = json.loads(record_str)
                record_count += 1
                value_sum += record.get('value', 0)
        byte_count += len(batch)

    # Send results back to main process
    result_queue.put({
        'shard_id': shard_id,
        'records': record_count,
        'bytes': byte_count,
        'value_sum': value_sum,
    })


def example_multiprocess_basic():
    """
    Basic multiprocess data loading.

    Each worker process handles one shard of the dataset.
    Results are collected via a Queue.
    """
    print("Basic multiprocess loading:")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        create_test_dataset(data_dir, 1000)

        config_path = Path(tmpdir) / "config.toml"
        config_path.write_text(f'[storage]\nbase_path = "{data_dir}"\n')

        # Create dataset to get shard count
        runtime = Runtime(str(config_path))
        dataset = runtime.register_dataset("data.jsonl", shards=4)
        num_shards = dataset.num_shards
        print(f"  Dataset: {dataset.total_bytes:,} bytes, {num_shards} shards")

        # Launch worker processes
        result_queue = Queue()
        processes = []

        print(f"  Launching {num_shards} worker processes...")
        for shard_id in range(num_shards):
            p = Process(
                target=worker_process,
                args=(shard_id, str(config_path), "data.jsonl", result_queue)
            )
            p.start()
            processes.append(p)

        # Collect results
        results = []
        for _ in range(num_shards):
            results.append(result_queue.get())

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Aggregate results
        total_records = sum(r['records'] for r in results)
        total_bytes = sum(r['bytes'] for r in results)
        total_value = sum(r['value_sum'] for r in results)

        print(f"  Results:")
        for r in sorted(results, key=lambda x: x['shard_id']):
            print(f"    Shard {r['shard_id']}: {r['records']} records, {r['bytes']:,} bytes")

        print(f"  Total: {total_records} records, {total_bytes:,} bytes")
        print(f"  Sum of values: {total_value:.1f}")


def example_distributed_pattern():
    """
    Pattern for distributed training with DTR.
    """
    print("\nDistributed training pattern:")
    print('''
  # Pattern: One shard per worker

  def train_worker(rank: int, world_size: int, config_path: str):
      """Training worker for distributed setup."""

      # Each worker creates its own runtime
      runtime = Runtime(config_path)

      # Register dataset with world_size shards
      dataset = runtime.register_dataset(
          "training_data.jsonl",
          shards=world_size,  # One shard per worker
          format="newline"
      )

      # Each worker processes its own shard
      my_shard = rank

      for epoch in range(num_epochs):
          # Create fresh iterator for each epoch
          iterator = dataset.iter_shard(
              my_shard,
              batch_size=1024 * 1024,  # 1MB batches
              prefetch=4               # Enable prefetching
          )

          for batch in iterator:
              # Process batch
              records = parse_batch(batch)
              loss = train_step(model, records)

              # Synchronize gradients across workers
              all_reduce(model.gradients)

          # Checkpoint from rank 0 only
          if rank == 0:
              state = serialize(model.state_dict())
              runtime.save_checkpoint(f"epoch_{epoch}", state)

          # Barrier to ensure checkpoint is saved
          barrier()

  # Launch with your distributed framework:
  # - torchrun for PyTorch DDP
  # - mpirun for MPI
  # - Ray for Ray Train
''')


def example_sharding_strategies():
    """
    Different sharding strategies for distributed training.
    """
    print("\nSharding strategies:")
    print('''
  1. One shard per worker (simplest)
     - shards = world_size
     - Each worker gets equal portion
     - Best for homogeneous workers

  2. Multiple shards per worker
     - shards = world_size * N
     - Workers process N shards each
     - Better load balancing if data is uneven

  3. Dynamic sharding
     - Use a work queue
     - Workers pull shards as they finish
     - Best for heterogeneous workers

  Example - Multiple shards per worker:

  shards_per_worker = 4
  total_shards = world_size * shards_per_worker

  dataset = runtime.register_dataset("data.jsonl", shards=total_shards)

  # Each worker processes multiple shards
  my_shards = range(rank * shards_per_worker, (rank + 1) * shards_per_worker)

  for shard_id in my_shards:
      for batch in dataset.iter_shard(shard_id):
          process(batch)
''')


def main():
    """Run all multiprocess examples."""
    print("=" * 60)
    print("Multiprocess Data Loading - DTR Distributed Examples")
    print("=" * 60)

    print("\n--- Example 1: Basic Multiprocess Loading ---")
    example_multiprocess_basic()

    print("\n--- Example 2: Distributed Training Pattern ---")
    example_distributed_pattern()

    print("\n--- Example 3: Sharding Strategies ---")
    example_sharding_strategies()

    print("\n" + "=" * 60)
    print("All multiprocess examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
