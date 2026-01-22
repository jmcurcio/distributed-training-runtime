#!/usr/bin/env python3
"""
Progress Monitoring and Error Handling - DTR Phase 1 Examples

Demonstrates:
- Tracking iteration progress
- Progress bar integration (tqdm)
- Handling common errors gracefully
- Error types and recovery

Prerequisites:
    cd rust/python-bindings && maturin develop
    pip install tqdm  # Optional, for progress bar example
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
            record = {"id": i, "value": i * 1.5}
            f.write(json.dumps(record) + '\n')
    return data_file


def example_progress_monitoring(dataset):
    """
    Monitor iteration progress through a shard.

    The iterator provides:
    - progress(): Returns 0.0 to 1.0 completion ratio
    - current_offset: Current byte position in the shard
    """
    print("Monitoring progress:")

    iterator = dataset.iter_shard(shard_id=0, batch_size=1024)

    batch_count = 0
    for _batch in iterator:
        batch_count += 1
        progress = iterator.progress()
        offset = iterator.current_offset

        # Print progress every few batches
        if batch_count % 5 == 0 or progress >= 1.0:
            print(f"  Batch {batch_count}: {progress:.1%} complete (offset: {offset:,})")

    print(f"  Finished: {batch_count} batches processed")


def example_tqdm_integration(dataset):
    """
    Integration with tqdm progress bar.

    tqdm provides a nice visual progress bar in the terminal.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        print("  tqdm not installed, skipping example")
        print("  Install with: pip install tqdm")
        return

    print("Progress bar with tqdm:")

    start, end = dataset.shard_info(0)
    shard_bytes = end - start

    with tqdm(total=shard_bytes, unit='B', unit_scale=True, desc="Processing") as pbar:
        iterator = dataset.iter_shard(0, batch_size=1024)
        last_offset = start

        for _batch in iterator:
            current = iterator.current_offset
            pbar.update(current - last_offset)
            last_offset = current

    print("  Progress bar complete!")


def example_error_file_not_found(runtime: Runtime):
    """
    Handle file not found errors.

    Raised when the specified dataset file doesn't exist.
    """
    print("Testing file not found error...")

    try:
        dataset = runtime.register_dataset("nonexistent_file.jsonl")
        # This line won't be reached
        print(f"  Dataset: {dataset}")
    except IOError as e:
        print(f"  Caught IOError: {e}")
        print("  Recovery: Check file path, ensure file exists")


def example_error_invalid_shard(dataset):
    """
    Handle invalid shard ID errors.

    Raised when requesting a shard ID >= num_shards.
    """
    print("Testing invalid shard ID error...")

    try:
        # Request shard 99 when only 4 shards exist
        iterator = dataset.iter_shard(shard_id=99)
        for batch in iterator:
            print(batch)
    except ValueError as e:
        print(f"  Caught ValueError: {e}")
        print(f"  Recovery: Use shard_id < {dataset.num_shards}")


def example_error_invalid_format(runtime: Runtime):
    """
    Handle invalid format string errors.

    Raised when the format string is not recognized.
    """
    print("Testing invalid format error...")

    # First create a valid file
    data_dir = Path(runtime.base_path)
    test_file = data_dir / "test.jsonl"
    test_file.write_text('{"id": 1}\n')

    try:
        dataset = runtime.register_dataset("test.jsonl", format="invalid_format")
        print(f"  Dataset: {dataset}")
    except ValueError as e:
        print(f"  Caught ValueError: {e}")
        print("  Recovery: Use 'newline', 'fixed:N', or 'length-prefixed'")


def example_error_checkpoint_not_found(runtime: Runtime):
    """
    Handle checkpoint not found errors.

    Raised when trying to load a non-existent checkpoint.
    """
    print("Testing checkpoint not found error...")

    try:
        runtime.load_checkpoint("nonexistent_checkpoint.ckpt")
    except IOError as e:
        print(f"  Caught IOError: {e}")
        print("  Recovery: Check checkpoint path, list available checkpoints")


def example_error_recovery_pattern():
    """
    Demonstrate a robust error recovery pattern.
    """
    print("Error recovery pattern:")
    print('''
  # Robust training loop with error handling

  def train_with_recovery(runtime, dataset):
      checkpoint_dir = Path(runtime.checkpoint_dir)

      # Try to resume from checkpoint
      try:
          checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
          if checkpoints:
              state = load_checkpoint(runtime, checkpoints[-1])
              start_epoch = state['epoch'] + 1
          else:
              state = initialize_state()
              start_epoch = 0
      except IOError as e:
          print(f"Warning: Could not load checkpoint: {e}")
          state = initialize_state()
          start_epoch = 0

      # Training loop
      for epoch in range(start_epoch, num_epochs):
          for shard_id in range(dataset.num_shards):
              try:
                  for batch in dataset.iter_shard(shard_id):
                      process_batch(batch, state)
              except IOError as e:
                  print(f"I/O error on shard {shard_id}: {e}")
                  continue  # Skip to next shard

          # Save checkpoint after each epoch
          try:
              save_checkpoint(runtime, state)
          except IOError as e:
              print(f"Warning: Could not save checkpoint: {e}")
''')


def main():
    """Run all progress and error handling examples."""
    print("=" * 60)
    print("Progress Monitoring and Error Handling - DTR Phase 1 Examples")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        create_test_dataset(data_dir, 200)

        config = Path(tmpdir) / "config.toml"
        config.write_text(f'''
[storage]
base_path = "{data_dir}"

[checkpoint]
checkpoint_dir = "{tmpdir}/checkpoints"
''')
        runtime = Runtime(str(config))
        dataset = runtime.register_dataset("test_data.jsonl", shards=4)

        print("\n--- Example 1: Progress Monitoring ---")
        example_progress_monitoring(dataset)

        print("\n--- Example 2: tqdm Integration ---")
        example_tqdm_integration(dataset)

        print("\n--- Example 3: File Not Found Error ---")
        example_error_file_not_found(runtime)

        print("\n--- Example 4: Invalid Shard ID Error ---")
        example_error_invalid_shard(dataset)

        print("\n--- Example 5: Invalid Format Error ---")
        example_error_invalid_format(runtime)

        print("\n--- Example 6: Checkpoint Not Found Error ---")
        example_error_checkpoint_not_found(runtime)

        print("\n--- Example 7: Error Recovery Pattern ---")
        example_error_recovery_pattern()

    print("\n" + "=" * 60)
    print("All progress and error handling examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
