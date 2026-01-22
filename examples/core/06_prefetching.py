#!/usr/bin/env python3
"""
Prefetching - DTR Core Examples

Demonstrates:
- Enabling prefetching for reduced I/O stalls
- Configuring prefetch buffer size
- Prefetching in training loops
- Performance comparison

Prerequisites:
    cd rust/python-bindings && maturin develop
"""

import json
import tempfile
import time
from pathlib import Path

from dtr import Runtime


def create_training_dataset(data_dir: Path, num_records: int = 500) -> Path:
    """Create a training dataset with features and labels."""
    data_file = data_dir / "train_data.jsonl"
    with open(data_file, 'w') as f:
        for i in range(num_records):
            record = {
                "id": i,
                "features": [(i % 10) * 0.1] * 10,
                "label": i % 3,
            }
            f.write(json.dumps(record) + '\n')
    return data_file


def example_prefetch_basic(dataset):
    """
    Basic prefetching usage.

    The prefetch parameter accepts:
    - True: Enable with default buffer size (4 batches)
    - False: Disable prefetching (default)
    - int: Enable with specified buffer size
    """
    print("Basic prefetching options:")

    # Without prefetching (default)
    print("  Without prefetching:")
    count = sum(1 for _ in dataset.iter_shard(0, batch_size=4096, prefetch=False))
    print(f"    Processed {count} batches")

    # With prefetching (default buffer)
    print("  With prefetching (default buffer=4):")
    count = sum(1 for _ in dataset.iter_shard(0, batch_size=4096, prefetch=True))
    print(f"    Processed {count} batches")

    # With custom buffer size
    print("  With prefetching (buffer=8):")
    count = sum(1 for _ in dataset.iter_shard(0, batch_size=4096, prefetch=8))
    print(f"    Processed {count} batches")


def example_prefetch_performance(dataset):
    """
    Compare performance with and without prefetching.

    Prefetching is most beneficial when:
    - I/O is slow relative to processing
    - Processing time is significant
    - Network storage (S3, NFS) is used
    """
    print("Performance comparison:")

    # Simulate some processing time
    def process_batch(batch):
        time.sleep(0.001)  # Simulate 1ms processing
        return len(batch)

    # Without prefetching
    start = time.perf_counter()
    total_bytes = 0
    for batch in dataset.iter_shard(0, batch_size=4096, prefetch=False):
        total_bytes += process_batch(batch)
    no_prefetch_time = time.perf_counter() - start
    print(f"  Without prefetch: {no_prefetch_time:.3f}s ({total_bytes:,} bytes)")

    # With prefetching
    start = time.perf_counter()
    total_bytes = 0
    for batch in dataset.iter_shard(0, batch_size=4096, prefetch=4):
        total_bytes += process_batch(batch)
    with_prefetch_time = time.perf_counter() - start
    print(f"  With prefetch:    {with_prefetch_time:.3f}s ({total_bytes:,} bytes)")

    if no_prefetch_time > with_prefetch_time:
        speedup = no_prefetch_time / with_prefetch_time
        print(f"  Speedup: {speedup:.2f}x")


def example_prefetch_training_loop(dataset):
    """
    Prefetching in a realistic training loop.

    This demonstrates how prefetching integrates with ML training,
    hiding I/O latency behind computation.
    """
    print("Training loop with prefetching:")

    # Simple linear model
    weights = [0.0] * 10
    bias = 0.0
    learning_rate = 0.01

    num_epochs = 2
    total_records = 0

    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_records = 0

        for shard_id in range(min(2, dataset.num_shards)):
            # Enable prefetching - I/O happens in background
            for batch in dataset.iter_shard(shard_id, batch_size=4096, prefetch=4):
                # While processing this batch, next batch is loading!
                records = batch.decode('utf-8').splitlines()

                for record_str in records:
                    if not record_str.strip():
                        continue
                    record = json.loads(record_str)

                    features = record.get('features', [0.0] * 10)
                    label = record.get('label', 0)

                    # Forward pass
                    prediction = sum(w * x for w, x in zip(weights, features)) + bias

                    # Loss
                    loss = (prediction - label) ** 2
                    epoch_loss += loss
                    epoch_records += 1

                    # Backward pass
                    error = prediction - label
                    for i in range(len(weights)):
                        if i < len(features):
                            weights[i] -= learning_rate * error * features[i]
                    bias -= learning_rate * error

        total_records += epoch_records
        avg_loss = epoch_loss / epoch_records if epoch_records > 0 else 0
        print(f"  Epoch {epoch}: {epoch_records} records, loss = {avg_loss:.4f}")

    elapsed = time.perf_counter() - start_time
    print(f"  Total: {total_records} records in {elapsed:.3f}s")
    print(f"  Throughput: {total_records / elapsed:.0f} records/sec")


def example_prefetch_buffer_sizing():
    """
    Guidelines for choosing prefetch buffer size.
    """
    print("Prefetch buffer sizing guidelines:")
    print('''
  Buffer Size | Use Case
  ------------|------------------
  2-4         | Local SSD, fast storage
  4-8         | Network storage, moderate latency
  8-16        | Cloud storage (S3), high latency
  16+         | Very high latency or slow processing

  Memory usage = buffer_size * batch_size
  Example: buffer=8, batch=1MB -> 8MB memory for prefetch

  Tips:
  - Start with buffer=4 and adjust based on profiling
  - Monitor queue length to detect I/O bottlenecks
  - Larger buffers help with bursty I/O patterns
''')


def main():
    """Run all prefetching examples."""
    print("=" * 60)
    print("Prefetching - DTR Core Examples")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()

        print("\nCreating training dataset...")
        create_training_dataset(data_dir, 500)

        config = Path(tmpdir) / "config.toml"
        config.write_text(f'[storage]\nbase_path = "{data_dir}"\n')
        runtime = Runtime(str(config))
        dataset = runtime.register_dataset("train_data.jsonl", shards=4)
        print(f"  Dataset: {dataset.total_bytes:,} bytes, {dataset.num_shards} shards")

        print("\n--- Example 1: Basic Prefetching ---")
        example_prefetch_basic(dataset)

        print("\n--- Example 2: Performance Comparison ---")
        example_prefetch_performance(dataset)

        print("\n--- Example 3: Training Loop ---")
        example_prefetch_training_loop(dataset)

        print("\n--- Example 4: Buffer Sizing Guidelines ---")
        example_prefetch_buffer_sizing()

    print("\n" + "=" * 60)
    print("All prefetching examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
