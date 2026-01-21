#!/usr/bin/env python3
"""
Comprehensive Python Examples for Distributed Training Runtime (Phase 1)

This module demonstrates all features of the DTR Python API including:
- Runtime initialization and configuration
- Dataset registration with different record formats
- Sharded iteration for parallel data loading
- Checkpoint save/load with compression
- Progress monitoring
- Distributed training patterns
- Error handling

Prerequisites:
    pip install dtr  # or: cd rust/python-bindings && maturin develop

Usage:
    python examples/python_examples.py
"""

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Iterator


# =============================================================================
# Example 1: Basic Runtime Usage
# =============================================================================

def example_basic_runtime():
    """
    Basic runtime initialization and configuration.

    The Runtime class is the main entry point for all DTR operations.
    It can be initialized with default settings or a custom config file.
    """
    from dtr import Runtime

    # Option 1: Default configuration
    # Uses ./data as base path, ./checkpoints for checkpoints, lz4 compression
    runtime = Runtime()

    print(f"Base path: {runtime.base_path}")
    print(f"Checkpoint dir: {runtime.checkpoint_dir}")
    print(f"Compression: {runtime.compression}")

    # Option 2: Custom configuration file
    # runtime = Runtime("config.toml")

    return runtime


# =============================================================================
# Example 2: Dataset Registration and Basic Iteration
# =============================================================================

def example_dataset_registration(runtime, data_path: str):
    """
    Register a dataset and iterate over it.

    Datasets are divided into shards for parallel processing.
    Each shard can be processed independently by different workers.
    """
    from dtr import Runtime

    # Register a newline-delimited dataset (JSONL, CSV, etc.)
    # Shards divide the dataset into equal byte ranges
    dataset = runtime.register_dataset(
        path=data_path,
        shards=4,           # Number of shards (typically = number of workers)
        format="newline"    # Record format: records separated by newlines
    )

    print(f"Dataset path: {dataset.path}")
    print(f"Total bytes: {dataset.total_bytes:,}")
    print(f"Number of shards: {dataset.num_shards}")

    # Get information about each shard
    for shard_id, (start, end) in enumerate(dataset.all_shard_info()):
        print(f"  Shard {shard_id}: bytes {start:,} - {end:,} ({end - start:,} bytes)")

    return dataset


def example_basic_iteration(dataset):
    """
    Iterate over a single shard.

    iter_shard() returns an iterator that yields batches of bytes.
    Each batch contains complete records (no partial records at boundaries).
    """
    # Iterate over shard 0 with 64KB batches
    batch_size = 64 * 1024  # 64 KB
    record_count = 0

    for batch in dataset.iter_shard(shard_id=0, batch_size=batch_size):
        # Decode batch and split into records
        text = batch.decode('utf-8')
        records = text.splitlines()
        record_count += len(records)

        # Process each record
        for record in records:
            if record.strip():  # Skip empty lines
                data = json.loads(record)
                # Process data...

    print(f"Processed {record_count} records from shard 0")


# =============================================================================
# Example 3: Different Record Formats
# =============================================================================

def example_newline_format(runtime, jsonl_path: str):
    """
    Newline-delimited format for JSONL, CSV, and text files.

    This is the default format. Records are separated by newline characters.
    Each batch ends at a complete newline boundary.
    """
    dataset = runtime.register_dataset(
        path=jsonl_path,
        shards=2,
        format="newline"  # Default, can be omitted
    )

    for batch in dataset.iter_shard(0):
        lines = batch.decode('utf-8').splitlines()
        for line in lines:
            record = json.loads(line)
            print(f"  ID: {record.get('id')}, text: {record.get('text', '')[:50]}...")
            break  # Just show first record
        break  # Just show first batch


def example_fixed_format(runtime, binary_path: str, record_size: int = 256):
    """
    Fixed-size format for binary data with uniform record sizes.

    Use for pre-processed embeddings, fixed-size images, or any data
    where all records have the same byte length.
    """
    dataset = runtime.register_dataset(
        path=binary_path,
        shards=4,
        format=f"fixed:{record_size}"  # Each record is exactly 256 bytes
    )

    # Shard boundaries are aligned to record size
    for shard_id, (start, end) in enumerate(dataset.all_shard_info()):
        assert start % record_size == 0, "Shards align to record boundaries"
        assert end % record_size == 0 or end == dataset.total_bytes

    for batch in dataset.iter_shard(0):
        # Process fixed-size records
        num_records = len(batch) // record_size
        for i in range(num_records):
            record = batch[i * record_size : (i + 1) * record_size]
            # Process record...
        break


def example_length_prefixed_format(runtime, data_path: str):
    """
    Length-prefixed format for variable-length binary records.

    Format: [4-byte big-endian length][data bytes]

    Useful for serialized protobufs, variable-length binary data, etc.
    """
    dataset = runtime.register_dataset(
        path=data_path,
        shards=2,
        format="length-prefixed"
    )

    for batch in dataset.iter_shard(0):
        # Parse length-prefixed records
        offset = 0
        while offset + 4 <= len(batch):
            # Read 4-byte big-endian length prefix
            length = int.from_bytes(batch[offset:offset+4], byteorder='big')
            offset += 4

            if offset + length > len(batch):
                break  # Incomplete record at end

            record_data = batch[offset:offset+length]
            offset += length
            # Process record_data...
        break


# =============================================================================
# Example 4: Progress Monitoring
# =============================================================================

def example_progress_monitoring(dataset):
    """
    Monitor iteration progress through a shard.

    Useful for progress bars and ETA calculations.
    """
    iterator = dataset.iter_shard(shard_id=0, batch_size=32 * 1024)

    batch_count = 0
    for batch in iterator:
        batch_count += 1
        progress = iterator.progress()  # Returns 0.0 to 1.0
        current_offset = iterator.current_offset

        # Update progress bar (using print for simplicity)
        if batch_count % 10 == 0:
            print(f"  Progress: {progress:.1%} | Offset: {current_offset:,}")

    print(f"  Final: 100% | Processed {batch_count} batches")


def example_with_tqdm(dataset):
    """
    Integration with tqdm progress bar.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        print("  tqdm not installed, skipping example")
        return

    start, end = dataset.shard_info(0)
    shard_bytes = end - start

    with tqdm(total=shard_bytes, unit='B', unit_scale=True) as pbar:
        iterator = dataset.iter_shard(0)
        last_offset = start

        for batch in iterator:
            current = iterator.current_offset
            pbar.update(current - last_offset)
            last_offset = current


# =============================================================================
# Example 5: Checkpointing
# =============================================================================

def example_save_checkpoint(runtime):
    """
    Save training state to a checkpoint.

    Checkpoints are:
    - Compressed (lz4 or zstd based on config)
    - Written atomically (safe against crashes)
    - Verified with XXHash64 checksums
    - Auto-cleaned (keeps last N based on config)
    """
    # Simulate training state
    training_state = {
        'epoch': 10,
        'global_step': 50000,
        'model_weights': [1.0, 2.0, 3.0, 4.0],  # Simplified
        'optimizer_state': {'lr': 0.001, 'beta1': 0.9},
        'loss_history': [0.5, 0.4, 0.3, 0.25, 0.2],
    }

    # Serialize state to bytes
    state_bytes = pickle.dumps(training_state)
    print(f"  State size: {len(state_bytes):,} bytes")

    # Save checkpoint - returns path to saved file
    checkpoint_path = runtime.save_checkpoint("training_state", state_bytes)
    print(f"  Saved to: {checkpoint_path}")

    return checkpoint_path


def example_load_checkpoint(runtime, checkpoint_path: str):
    """
    Load a checkpoint and restore training state.

    Automatically decompresses and verifies integrity.
    """
    # Load checkpoint bytes
    state_bytes = runtime.load_checkpoint(checkpoint_path)
    print(f"  Loaded {len(state_bytes):,} bytes")

    # Deserialize state
    training_state = pickle.loads(state_bytes)

    print(f"  Epoch: {training_state['epoch']}")
    print(f"  Global step: {training_state['global_step']}")
    print(f"  Learning rate: {training_state['optimizer_state']['lr']}")

    return training_state


def example_checkpoint_workflow():
    """
    Complete checkpoint save/load workflow.
    """
    from dtr import Runtime

    # Create runtime with checkpointing enabled (default)
    runtime = Runtime()

    # Training loop with periodic checkpointing
    def train_epoch(epoch: int, model_state: dict) -> dict:
        """Simulate one epoch of training."""
        model_state['epoch'] = epoch
        model_state['loss'] = 1.0 / (epoch + 1)  # Decreasing loss
        return model_state

    # Initialize or restore state
    model_state = {'weights': [0.0] * 100, 'epoch': 0, 'loss': 1.0}
    start_epoch = 0

    # Try to resume from checkpoint
    checkpoint_dir = Path(runtime.checkpoint_dir)
    existing_checkpoints = sorted(checkpoint_dir.glob("model_*.ckpt"))

    if existing_checkpoints:
        latest = str(existing_checkpoints[-1])
        print(f"  Resuming from: {latest}")
        state_bytes = runtime.load_checkpoint(latest)
        model_state = pickle.loads(state_bytes)
        start_epoch = model_state['epoch'] + 1

    # Continue training
    checkpoint_every = 5
    num_epochs = 20

    for epoch in range(start_epoch, num_epochs):
        model_state = train_epoch(epoch, model_state)
        print(f"  Epoch {epoch}: loss = {model_state['loss']:.4f}")

        # Save checkpoint periodically
        if (epoch + 1) % checkpoint_every == 0:
            state_bytes = pickle.dumps(model_state)
            path = runtime.save_checkpoint(f"model_epoch{epoch}", state_bytes)
            print(f"    Checkpoint saved: {path}")


# =============================================================================
# Example 6: Parallel Data Loading (Multi-Process)
# =============================================================================

def example_multiprocess_loading(runtime, data_path: str):
    """
    Load data in parallel using multiple processes.

    Each process handles one shard independently.
    This is the primary pattern for distributed training.
    """
    from multiprocessing import Process, Queue

    def worker(shard_id: int, data_path: str, result_queue: Queue):
        """Worker process that handles one shard."""
        from dtr import Runtime

        # Each process creates its own runtime
        runtime = Runtime()
        dataset = runtime.register_dataset(data_path, shards=4, format="newline")

        record_count = 0
        byte_count = 0

        for batch in dataset.iter_shard(shard_id):
            records = batch.decode('utf-8').splitlines()
            record_count += len(records)
            byte_count += len(batch)

        result_queue.put({
            'shard_id': shard_id,
            'records': record_count,
            'bytes': byte_count
        })

    # Create dataset to get shard count
    dataset = runtime.register_dataset(data_path, shards=4)

    # Launch workers
    result_queue = Queue()
    processes = []

    for shard_id in range(dataset.num_shards):
        p = Process(target=worker, args=(shard_id, data_path, result_queue))
        p.start()
        processes.append(p)

    # Collect results
    results = []
    for _ in range(dataset.num_shards):
        results.append(result_queue.get())

    # Wait for all processes
    for p in processes:
        p.join()

    # Summarize
    total_records = sum(r['records'] for r in results)
    total_bytes = sum(r['bytes'] for r in results)

    print(f"  Total records: {total_records:,}")
    print(f"  Total bytes: {total_bytes:,}")

    for r in sorted(results, key=lambda x: x['shard_id']):
        print(f"    Shard {r['shard_id']}: {r['records']:,} records, {r['bytes']:,} bytes")


# =============================================================================
# Example 7: Distributed Training with PyTorch
# =============================================================================

def example_distributed_pytorch():
    """
    Distributed training pattern with PyTorch DDP.

    Each rank processes a different shard of the data.

    Launch with: torchrun --nproc_per_node=4 script.py
    """
    # This is a template - actual execution requires PyTorch and torchrun
    code = '''
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dtr import Runtime

def main():
    # Initialize distributed environment
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Create model with DDP
    model = create_model().to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create DTR runtime and dataset
    runtime = Runtime("config.toml")
    dataset = runtime.register_dataset(
        "training_data.jsonl",
        shards=world_size,  # One shard per rank
        format="newline"
    )

    # Each rank processes its own shard
    my_shard = rank

    for epoch in range(num_epochs):
        # Reset iterator for new epoch
        iterator = dataset.iter_shard(my_shard, batch_size=1024 * 1024)

        for batch_data in iterator:
            # Parse records
            records = parse_records(batch_data)

            # Prepare batch tensors
            inputs, targets = prepare_batch(records)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = compute_loss(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Log progress
            if rank == 0:
                progress = iterator.progress()
                print(f"Epoch {epoch} | Progress: {progress:.1%} | Loss: {loss.item():.4f}")

        # Synchronize at epoch end
        dist.barrier()

        # Save checkpoint from rank 0
        if rank == 0:
            state = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            state_bytes = pickle.dumps(state)
            runtime.save_checkpoint(f"model_epoch_{epoch}", state_bytes)

        dist.barrier()

if __name__ == "__main__":
    main()
'''
    print("  PyTorch distributed training template:")
    print("  Launch with: torchrun --nproc_per_node=4 script.py")
    return code


# =============================================================================
# Example 8: Configuration
# =============================================================================

def example_configuration():
    """
    Configure the runtime via TOML file or environment variables.
    """
    # Example configuration file content
    config_toml = '''
# DTR Configuration File

[storage]
base_path = "./data"           # Base directory for datasets
buffer_size = 65536            # I/O buffer size (64KB default)
use_mmap = true                # Use memory mapping for large files
mmap_threshold = 1048576       # Min file size for mmap (1MB)

[dataset]
default_shard_count = 1        # Default shards if not specified
prefetch_batches = 2           # Batches to prefetch (future feature)
shuffle = false                # Shuffle records (future feature)

[checkpoint]
checkpoint_dir = "./checkpoints"  # Directory for checkpoint files
compression = "lz4"               # Compression: "none", "lz4", or "zstd"
compression_level = 1             # Compression level (1-22 for zstd)
keep_last_n = 3                   # Keep only N most recent checkpoints
atomic_writes = true              # Use atomic writes (rename pattern)

[performance]
io_threads = 4                    # I/O thread pool size
max_buffer_memory = 268435456     # Max memory for buffers (256MB)
'''

    print("  Example config.toml:")
    for line in config_toml.strip().split('\n')[:15]:
        print(f"    {line}")
    print("    ...")

    # Environment variable overrides
    print("\n  Environment variable overrides:")
    print("    DTR_STORAGE_BASE_PATH=/data/training")
    print("    DTR_CHECKPOINT_COMPRESSION=zstd")
    print("    DTR_CHECKPOINT_KEEP_LAST_N=5")


def example_environment_config():
    """
    Configure runtime using environment variables.

    Environment variables override config file settings.
    Prefix: DTR_, use underscores for nested keys.
    """
    from dtr import Runtime

    # Set environment variables before creating runtime
    os.environ["DTR_STORAGE_BASE_PATH"] = "/tmp/dtr_data"
    os.environ["DTR_CHECKPOINT_COMPRESSION"] = "zstd"
    os.environ["DTR_CHECKPOINT_KEEP_LAST_N"] = "5"

    # Create runtime - will use environment overrides
    runtime = Runtime()

    print(f"  Base path (from env): {runtime.base_path}")
    print(f"  Compression (from env): {runtime.compression}")

    # Clean up
    del os.environ["DTR_STORAGE_BASE_PATH"]
    del os.environ["DTR_CHECKPOINT_COMPRESSION"]
    del os.environ["DTR_CHECKPOINT_KEEP_LAST_N"]


# =============================================================================
# Example 9: Error Handling
# =============================================================================

def example_error_handling():
    """
    Handle common errors gracefully.
    """
    from dtr import Runtime

    runtime = Runtime()

    # 1. File not found
    print("  Testing file not found error...")
    try:
        dataset = runtime.register_dataset("nonexistent_file.jsonl")
    except IOError as e:
        print(f"    Caught IOError: {e}")

    # 2. Invalid shard ID
    print("  Testing invalid shard ID...")
    try:
        # Create a test file first
        test_dir = Path(runtime.base_path)
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / "test.jsonl"
        test_file.write_text('{"id": 1}\n{"id": 2}\n')

        dataset = runtime.register_dataset("test.jsonl", shards=2)
        iterator = dataset.iter_shard(99)  # Invalid shard ID
    except ValueError as e:
        print(f"    Caught ValueError: {e}")

    # 3. Invalid format string
    print("  Testing invalid format...")
    try:
        dataset = runtime.register_dataset("test.jsonl", format="invalid")
    except ValueError as e:
        print(f"    Caught ValueError: {e}")

    # 4. Checkpoint not found
    print("  Testing checkpoint not found...")
    try:
        runtime.load_checkpoint("nonexistent_checkpoint.ckpt")
    except IOError as e:
        print(f"    Caught IOError: {e}")


# =============================================================================
# Example 10: Complete Training Pipeline
# =============================================================================

def example_complete_pipeline():
    """
    A complete training pipeline demonstrating all features.
    """
    from dtr import Runtime

    print("=" * 60)
    print("Complete Training Pipeline Example")
    print("=" * 60)

    # Setup
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir()

        # Create sample training data
        train_file = data_dir / "train.jsonl"
        with open(train_file, 'w') as f:
            for i in range(1000):
                record = {"id": i, "features": [(i % 10) * 0.1] * 10, "label": i % 3}
                f.write(json.dumps(record) + '\n')

        print(f"\n1. Created training data: {train_file}")
        print(f"   File size: {train_file.stat().st_size:,} bytes")

        # Create configuration
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(f'''
[storage]
base_path = "{data_dir}"

[checkpoint]
checkpoint_dir = "{checkpoint_dir}"
compression = "lz4"
keep_last_n = 2
''')

        print(f"\n2. Created config: {config_file}")

        # Initialize runtime
        runtime = Runtime(str(config_file))
        print(f"\n3. Initialized runtime")
        print(f"   Base path: {runtime.base_path}")
        print(f"   Checkpoint dir: {runtime.checkpoint_dir}")
        print(f"   Compression: {runtime.compression}")

        # Register dataset
        dataset = runtime.register_dataset("train.jsonl", shards=4, format="newline")
        print(f"\n4. Registered dataset")
        print(f"   Total bytes: {dataset.total_bytes:,}")
        print(f"   Num shards: {dataset.num_shards}")

        # Simulate training
        print(f"\n5. Training simulation")

        model = {"weights": [0.0] * 10, "bias": 0.0}

        for epoch in range(3):
            total_records = 0
            total_loss = 0.0

            for shard_id in range(dataset.num_shards):
                iterator = dataset.iter_shard(shard_id, batch_size=4096)

                for batch in iterator:
                    records = batch.decode('utf-8').splitlines()

                    for record_str in records:
                        if not record_str.strip():
                            continue
                        record = json.loads(record_str)

                        # Simulate forward pass
                        features = record["features"]
                        prediction = sum(w * f for w, f in zip(model["weights"], features))
                        prediction += model["bias"]

                        # Simulate loss
                        loss = (prediction - record["label"]) ** 2
                        total_loss += loss
                        total_records += 1

                        # Simulate backward pass (gradient descent with clipping)
                        lr = 0.001
                        error = max(-10, min(10, prediction - record["label"]))  # Clip error
                        for i in range(len(model["weights"])):
                            model["weights"][i] -= lr * error * features[i]
                        model["bias"] -= lr * error

            avg_loss = total_loss / total_records if total_records > 0 else 0
            print(f"   Epoch {epoch}: {total_records} records, avg loss = {avg_loss:.6f}")

            # Save checkpoint
            state = {"epoch": epoch, "model": model, "loss": avg_loss}
            state_bytes = pickle.dumps(state)
            checkpoint_path = runtime.save_checkpoint(f"model_epoch_{epoch}", state_bytes)
            print(f"   Saved checkpoint: {Path(checkpoint_path).name}")

        # Final model evaluation
        print(f"\n6. Final model weights (first 3): {model['weights'][:3]}")
        print(f"   Final bias: {model['bias']:.4f}")

        # List checkpoints
        print(f"\n7. Checkpoints in {checkpoint_dir}:")
        for ckpt in sorted(checkpoint_dir.glob("*.ckpt")):
            print(f"   - {ckpt.name} ({ckpt.stat().st_size:,} bytes)")

        # Load latest checkpoint
        checkpoints = sorted(checkpoint_dir.glob("model_epoch_*.ckpt"))
        if checkpoints:
            latest = str(checkpoints[-1])
            print(f"\n8. Loading latest checkpoint: {Path(latest).name}")
            state_bytes = runtime.load_checkpoint(latest)
            state = pickle.loads(state_bytes)
            print(f"   Restored epoch: {state['epoch']}")
            print(f"   Restored loss: {state['loss']:.6f}")

        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)


# =============================================================================
# Example 11: Memory-Efficient Large Dataset Processing
# =============================================================================

def example_large_dataset_streaming():
    """
    Process large datasets without loading everything into memory.

    The streaming approach reads data in batches, processes each batch,
    and discards it before reading the next. This keeps memory usage constant
    regardless of dataset size.
    """
    from dtr import Runtime

    print("Memory-efficient streaming pattern:")
    print('''
# Process TB-scale datasets with constant memory usage

runtime = Runtime()
dataset = runtime.register_dataset("huge_dataset.jsonl", shards=1000)

# Each worker processes one shard
worker_id = get_worker_id()  # 0 to 999

# Small batch size = lower memory, larger = better throughput
batch_size = 512 * 1024  # 512KB batches

for batch in dataset.iter_shard(worker_id, batch_size=batch_size):
    # Parse records from this batch
    records = parse_batch(batch)

    # Process records (batch is ~512KB regardless of total dataset size)
    for record in records:
        process_record(record)

    # Memory for 'batch' is released before next iteration
    # Total memory usage stays constant

# Benefits:
# - Process petabyte datasets with megabytes of RAM
# - No OOM errors from loading full dataset
# - GIL released during I/O for better threading
''')


# =============================================================================
# Example 12: Iterator Reset and Multi-Pass Processing
# =============================================================================

def example_iterator_reset():
    """
    Reset iterators for multiple passes over data.

    Useful for multi-epoch training without re-creating datasets.
    """
    from dtr import Runtime

    print("Multi-pass iteration pattern:")
    print('''
runtime = Runtime()
dataset = runtime.register_dataset("data.jsonl", shards=4)

for epoch in range(num_epochs):
    for shard_id in range(dataset.num_shards):
        # Create iterator for this shard
        iterator = dataset.iter_shard(shard_id)

        for batch in iterator:
            process_batch(batch)

        # Or with explicit reset for multiple passes per epoch:
        iterator = dataset.iter_shard(shard_id)

        # First pass: compute statistics
        for batch in iterator:
            accumulate_stats(batch)

        # Reset for second pass
        iterator.reset()

        # Second pass: normalize using statistics
        for batch in iterator:
            normalize_and_process(batch)
''')


# =============================================================================
# Main: Run All Examples
# =============================================================================

def main():
    """Run all examples."""
    print("=" * 70)
    print("Distributed Training Runtime - Python Examples")
    print("=" * 70)

    # Create temporary directory for examples
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup test data
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir()

        # Create test JSONL file
        jsonl_file = data_dir / "test_data.jsonl"
        with open(jsonl_file, 'w') as f:
            for i in range(100):
                record = {
                    "id": i,
                    "text": f"This is record number {i} with some sample text",
                    "value": i * 1.5
                }
                f.write(json.dumps(record) + '\n')

        # Create config file for this test run
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(f'''
[storage]
base_path = "{data_dir}"

[checkpoint]
checkpoint_dir = "{checkpoint_dir}"
compression = "lz4"
''')

        from dtr import Runtime

        try:
            print("\n" + "-" * 70)
            print("Example 1: Basic Runtime")
            print("-" * 70)
            runtime = Runtime(str(config_file))
            print(f"Base path: {runtime.base_path}")
            print(f"Checkpoint dir: {runtime.checkpoint_dir}")
            print(f"Compression: {runtime.compression}")

            print("\n" + "-" * 70)
            print("Example 2: Dataset Registration")
            print("-" * 70)
            dataset = example_dataset_registration(runtime, "test_data.jsonl")

            print("\n" + "-" * 70)
            print("Example 3: Basic Iteration")
            print("-" * 70)
            example_basic_iteration(dataset)

            print("\n" + "-" * 70)
            print("Example 4: Newline Format")
            print("-" * 70)
            example_newline_format(runtime, "test_data.jsonl")

            print("\n" + "-" * 70)
            print("Example 5: Progress Monitoring")
            print("-" * 70)
            example_progress_monitoring(dataset)

            print("\n" + "-" * 70)
            print("Example 6: Save Checkpoint")
            print("-" * 70)
            checkpoint_path = example_save_checkpoint(runtime)

            print("\n" + "-" * 70)
            print("Example 7: Load Checkpoint")
            print("-" * 70)
            example_load_checkpoint(runtime, checkpoint_path)

            print("\n" + "-" * 70)
            print("Example 8: Configuration")
            print("-" * 70)
            example_configuration()

            print("\n" + "-" * 70)
            print("Example 9: Error Handling")
            print("-" * 70)
            example_error_handling()

            print("\n" + "-" * 70)
            print("Example 10: PyTorch Distributed Template")
            print("-" * 70)
            example_distributed_pytorch()

            print("\n" + "-" * 70)
            print("Example 11: Memory-Efficient Streaming")
            print("-" * 70)
            example_large_dataset_streaming()

            print("\n" + "-" * 70)
            print("Example 12: Iterator Reset Pattern")
            print("-" * 70)
            example_iterator_reset()

            print("\n" + "-" * 70)
            print("Example 13: Complete Training Pipeline")
            print("-" * 70)
            example_complete_pipeline()

        finally:
            pass  # Config file cleaned up with tmpdir

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
