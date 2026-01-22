#!/usr/bin/env python3
"""
Dataset and Iteration - DTR Phase 1 Examples

Demonstrates:
- Registering datasets with sharding
- Iterating over shards
- Batch processing
- Shard information and boundaries

Prerequisites:
    cd rust/python-bindings && maturin develop
"""

import json
import tempfile
from pathlib import Path

from dtr import Runtime


def create_test_dataset(data_dir: Path, num_records: int = 100) -> Path:
    """Create a test JSONL dataset."""
    data_file = data_dir / "test_data.jsonl"
    with open(data_file, 'w') as f:
        for i in range(num_records):
            record = {
                "id": i,
                "text": f"This is record number {i} with some sample text",
                "value": i * 1.5
            }
            f.write(json.dumps(record) + '\n')
    return data_file


def example_register_dataset(runtime: Runtime, data_path: str):
    """
    Register a dataset with sharding.

    Datasets are divided into shards for parallel processing.
    Each shard can be processed independently by different workers.
    """
    print(f"Registering dataset: {data_path}")

    # Register with 4 shards for parallel processing
    dataset = runtime.register_dataset(
        path=data_path,
        shards=4,           # Number of shards (typically = number of workers)
        format="newline"    # Records separated by newlines
    )

    print(f"  Path: {dataset.path}")
    print(f"  Total bytes: {dataset.total_bytes:,}")
    print(f"  Number of shards: {dataset.num_shards}")

    return dataset


def example_shard_info(dataset):
    """
    Get information about shard boundaries.

    Each shard has a start and end byte offset. Shards are designed
    to be roughly equal in size while respecting record boundaries.
    """
    print("Shard boundaries:")

    for shard_id in range(dataset.num_shards):
        start, end = dataset.shard_info(shard_id)
        size = end - start
        print(f"  Shard {shard_id}: bytes {start:,} - {end:,} ({size:,} bytes)")

    # Alternative: get all shard info at once
    print("\nUsing all_shard_info():")
    for shard_id, (start, end) in enumerate(dataset.all_shard_info()):
        print(f"  Shard {shard_id}: {start:,} - {end:,}")


def example_basic_iteration(dataset):
    """
    Iterate over a single shard.

    iter_shard() returns an iterator that yields batches of bytes.
    Each batch contains complete records (no partial records at boundaries).
    """
    print("Iterating over shard 0...")

    batch_size = 4096  # 4KB batches
    record_count = 0
    batch_count = 0

    for batch in dataset.iter_shard(shard_id=0, batch_size=batch_size):
        batch_count += 1
        # Decode batch and split into records
        text = batch.decode('utf-8')
        records = text.splitlines()

        for record in records:
            if record.strip():
                record_count += 1
                # Parse and process record
                data = json.loads(record)
                # In real code: process data...

    print(f"  Processed {batch_count} batches, {record_count} records")


def example_batch_sizes(dataset):
    """
    Compare different batch sizes.

    Batch size affects:
    - Memory usage (larger batches = more memory)
    - I/O efficiency (larger batches = fewer I/O calls)
    - Processing granularity (smaller batches = finer progress)
    """
    print("Comparing batch sizes:")

    for batch_size in [1024, 4096, 16384, 65536]:
        batch_count = 0
        for _batch in dataset.iter_shard(0, batch_size=batch_size):
            batch_count += 1

        print(f"  Batch size {batch_size:,}: {batch_count} batches")


def example_process_all_shards(dataset):
    """
    Process all shards sequentially.

    In distributed training, each worker would process one shard.
    Here we demonstrate sequential processing of all shards.
    """
    print("Processing all shards:")

    total_records = 0

    for shard_id in range(dataset.num_shards):
        shard_records = 0

        for batch in dataset.iter_shard(shard_id, batch_size=4096):
            records = batch.decode('utf-8').splitlines()
            shard_records += len([r for r in records if r.strip()])

        total_records += shard_records
        print(f"  Shard {shard_id}: {shard_records} records")

    print(f"  Total: {total_records} records across {dataset.num_shards} shards")


def main():
    """Run all dataset and iteration examples."""
    print("=" * 60)
    print("Dataset and Iteration - DTR Phase 1 Examples")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()

        # Create test data
        print("\nCreating test dataset...")
        data_file = create_test_dataset(data_dir, num_records=200)
        print(f"  Created: {data_file.name} ({data_file.stat().st_size:,} bytes)")

        # Create runtime with test directory
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(f'[storage]\nbase_path = "{data_dir}"\n')
        runtime = Runtime(str(config_file))

        print("\n--- Example 1: Register Dataset ---")
        dataset = example_register_dataset(runtime, "test_data.jsonl")

        print("\n--- Example 2: Shard Information ---")
        example_shard_info(dataset)

        print("\n--- Example 3: Basic Iteration ---")
        example_basic_iteration(dataset)

        print("\n--- Example 4: Batch Sizes ---")
        example_batch_sizes(dataset)

        print("\n--- Example 5: Process All Shards ---")
        example_process_all_shards(dataset)

    print("\n" + "=" * 60)
    print("All dataset and iteration examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
