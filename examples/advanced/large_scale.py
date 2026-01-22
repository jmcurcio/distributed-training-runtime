#!/usr/bin/env python3
"""
Large-Scale Data Processing - DTR Advanced Examples

Demonstrates:
- Memory-efficient streaming for TB-scale datasets
- Optimal batch sizing strategies
- Multi-pass processing patterns
- Performance optimization techniques

Prerequisites:
    cd rust/python-bindings && maturin develop
"""

import json
import tempfile
import time
from pathlib import Path

from dtr import Runtime


def create_large_dataset(data_dir: Path, num_records: int = 10000) -> Path:
    """Create a larger test dataset."""
    data_file = data_dir / "large_data.jsonl"
    with open(data_file, 'w') as f:
        for i in range(num_records):
            record = {
                "id": i,
                "features": [i * 0.001] * 100,  # 100 features
                "label": i % 10,
                "text": f"Record {i} " * 10,  # Some text content
            }
            f.write(json.dumps(record) + '\n')
    return data_file


def example_memory_efficient_streaming(dataset):
    """
    Process large datasets with constant memory usage.

    The streaming approach reads data in batches, processes each batch,
    and releases memory before the next batch. Memory usage stays constant
    regardless of total dataset size.
    """
    print("Memory-efficient streaming:")

    # Small batch size = low memory, large batch = better throughput
    batch_size = 4096  # 4KB batches

    peak_batch_bytes = 0
    total_bytes = 0
    batch_count = 0

    # Process all shards
    for shard_id in range(dataset.num_shards):
        for batch in dataset.iter_shard(shard_id, batch_size=batch_size):
            current_size = len(batch)
            peak_batch_bytes = max(peak_batch_bytes, current_size)
            total_bytes += current_size
            batch_count += 1

            # Process records (memory released after each iteration)
            records = batch.decode('utf-8').splitlines()
            for record_str in records:
                if record_str.strip():
                    _ = json.loads(record_str)

    print(f"  Total data: {total_bytes:,} bytes")
    print(f"  Batches: {batch_count}")
    print(f"  Peak batch size: {peak_batch_bytes:,} bytes")
    print(f"  Memory usage: O({batch_size}) regardless of dataset size")


def example_batch_size_tuning(dataset):
    """
    Find optimal batch size for your workload.

    Trade-offs:
    - Smaller batches: Lower memory, more I/O overhead
    - Larger batches: Higher memory, better throughput
    """
    print("\nBatch size comparison:")

    batch_sizes = [1024, 4096, 16384, 65536, 262144]

    for batch_size in batch_sizes:
        start = time.perf_counter()
        batch_count = 0
        total_bytes = 0

        for batch in dataset.iter_shard(0, batch_size=batch_size):
            batch_count += 1
            total_bytes += len(batch)

        elapsed = time.perf_counter() - start
        throughput = total_bytes / elapsed / 1024 / 1024  # MB/s

        print(f"  {batch_size:>7} bytes: {batch_count:>4} batches, "
              f"{elapsed:.3f}s, {throughput:.1f} MB/s")


def example_multi_pass_processing(dataset):
    """
    Multiple passes over data for statistics and normalization.

    Pattern for when you need to compute statistics before processing.
    """
    print("\nMulti-pass processing:")

    # Pass 1: Compute statistics
    print("  Pass 1: Computing statistics...")
    total_sum = 0.0
    count = 0

    iterator = dataset.iter_shard(0, batch_size=8192)
    for batch in iterator:
        records = batch.decode('utf-8').splitlines()
        for record_str in records:
            if record_str.strip():
                record = json.loads(record_str)
                total_sum += record.get('label', 0)
                count += 1

    mean = total_sum / count if count > 0 else 0
    print(f"    Computed mean: {mean:.4f} from {count} records")

    # Pass 2: Use statistics
    print("  Pass 2: Processing with statistics...")
    iterator.reset()

    normalized_count = 0
    for batch in iterator:
        records = batch.decode('utf-8').splitlines()
        for record_str in records:
            if record_str.strip():
                record = json.loads(record_str)
                # Normalize using computed mean
                normalized_value = record.get('label', 0) - mean
                normalized_count += 1

    print(f"    Processed {normalized_count} records with normalization")


def example_parallel_shard_processing(dataset):
    """
    Process shards in parallel for maximum throughput.

    In production, each shard would be processed by a different worker.
    """
    print("\nParallel shard processing (simulated):")

    shard_stats = []

    for shard_id in range(dataset.num_shards):
        start = time.perf_counter()
        record_count = 0
        byte_count = 0

        for batch in dataset.iter_shard(shard_id, batch_size=16384, prefetch=4):
            byte_count += len(batch)
            records = batch.decode('utf-8').splitlines()
            record_count += len([r for r in records if r.strip()])

        elapsed = time.perf_counter() - start
        shard_stats.append({
            'shard_id': shard_id,
            'records': record_count,
            'bytes': byte_count,
            'time': elapsed,
        })

    # Summary
    total_records = sum(s['records'] for s in shard_stats)
    total_bytes = sum(s['bytes'] for s in shard_stats)
    total_time = sum(s['time'] for s in shard_stats)
    parallel_time = max(s['time'] for s in shard_stats)

    for s in shard_stats:
        print(f"  Shard {s['shard_id']}: {s['records']:,} records, {s['time']:.3f}s")

    print(f"  Sequential time: {total_time:.3f}s")
    print(f"  Parallel time (ideal): {parallel_time:.3f}s")
    print(f"  Speedup potential: {total_time / parallel_time:.1f}x")


def main():
    """Run all large-scale processing examples."""
    print("=" * 60)
    print("Large-Scale Data Processing - DTR Advanced Examples")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()

        print("\nCreating test dataset...")
        create_large_dataset(data_dir, 5000)

        config = Path(tmpdir) / "config.toml"
        config.write_text(f'[storage]\nbase_path = "{data_dir}"\n')
        runtime = Runtime(str(config))
        dataset = runtime.register_dataset("large_data.jsonl", shards=4)
        print(f"  Dataset: {dataset.total_bytes:,} bytes, {dataset.num_shards} shards")

        print("\n--- Example 1: Memory-Efficient Streaming ---")
        example_memory_efficient_streaming(dataset)

        print("\n--- Example 2: Batch Size Tuning ---")
        example_batch_size_tuning(dataset)

        print("\n--- Example 3: Multi-Pass Processing ---")
        example_multi_pass_processing(dataset)

        print("\n--- Example 4: Parallel Shard Processing ---")
        example_parallel_shard_processing(dataset)

    print("\n" + "=" * 60)
    print("All large-scale processing examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
