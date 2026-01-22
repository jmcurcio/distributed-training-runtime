#!/usr/bin/env python3
"""
Checkpointing - DTR Phase 1 Examples

Demonstrates:
- Saving checkpoints
- Loading checkpoints
- Compression options
- Checkpoint retention policies
- Checkpoint workflow for training

Prerequisites:
    cd rust/python-bindings && maturin develop
"""

import pickle
import tempfile
from pathlib import Path

from dtr import Runtime


def example_save_checkpoint(runtime: Runtime):
    """
    Save training state to a checkpoint.

    Checkpoints are:
    - Compressed (lz4 or zstd based on config)
    - Written atomically (safe against crashes)
    - Verified with XXHash64 checksums
    """
    print("Saving checkpoint...")

    # Simulate training state
    training_state = {
        'epoch': 10,
        'global_step': 50000,
        'model_weights': [1.0, 2.0, 3.0, 4.0],
        'optimizer_state': {'lr': 0.001, 'beta1': 0.9},
        'loss_history': [0.5, 0.4, 0.3, 0.25, 0.2],
    }

    # Serialize state to bytes
    state_bytes = pickle.dumps(training_state)
    print(f"  State size: {len(state_bytes):,} bytes")

    # Save checkpoint
    checkpoint_path = runtime.save_checkpoint("training_state", state_bytes)
    print(f"  Saved to: {checkpoint_path}")

    return checkpoint_path


def example_load_checkpoint(runtime: Runtime, checkpoint_path: str):
    """
    Load a checkpoint and restore training state.

    Loading automatically:
    - Decompresses the data
    - Verifies the checksum
    - Returns the original bytes
    """
    print(f"Loading checkpoint: {Path(checkpoint_path).name}")

    # Load checkpoint bytes
    state_bytes = runtime.load_checkpoint(checkpoint_path)
    print(f"  Loaded {len(state_bytes):,} bytes")

    # Deserialize state
    training_state = pickle.loads(state_bytes)

    print(f"  Epoch: {training_state['epoch']}")
    print(f"  Global step: {training_state['global_step']}")
    print(f"  Learning rate: {training_state['optimizer_state']['lr']}")

    return training_state


def example_compression_options():
    """
    Compare checkpoint compression options.

    Available options:
    - "none": No compression (fastest, largest files)
    - "lz4": Fast compression (default, good balance)
    - "zstd": High compression (slower, smallest files)
    """
    print("Compression options:")

    # Create test data
    test_data = pickle.dumps({'weights': [0.1] * 10000})
    print(f"  Original size: {len(test_data):,} bytes")

    with tempfile.TemporaryDirectory() as tmpdir:
        for compression in ["none", "lz4", "zstd"]:
            # Create runtime with specific compression
            config = Path(tmpdir) / f"config_{compression}.toml"
            config.write_text(f'''
[storage]
base_path = "{tmpdir}"

[checkpoint]
checkpoint_dir = "{tmpdir}/ckpt_{compression}"
compression = "{compression}"
''')
            runtime = Runtime(str(config))

            # Save checkpoint
            path = runtime.save_checkpoint("test", test_data)
            size = Path(path).stat().st_size

            print(f"  {compression:6}: {size:,} bytes ({size/len(test_data)*100:.1f}%)")


def example_checkpoint_retention(runtime: Runtime):
    """
    Demonstrate checkpoint retention policy.

    The keep_last_n setting automatically deletes old checkpoints,
    keeping only the N most recent ones.
    """
    print("Checkpoint retention (keep_last_n):")
    print("  Saving 5 checkpoints with keep_last_n=3...")

    checkpoint_dir = Path(runtime.checkpoint_dir)

    for i in range(5):
        state = pickle.dumps({'epoch': i})
        runtime.save_checkpoint(f"model_epoch_{i}", state)

        # Count remaining checkpoints
        checkpoints = list(checkpoint_dir.glob("model_epoch_*.ckpt"))
        print(f"    After epoch {i}: {len(checkpoints)} checkpoint(s)")

    # List final checkpoints
    checkpoints = sorted(checkpoint_dir.glob("model_epoch_*.ckpt"))
    print(f"  Final checkpoints: {[c.name for c in checkpoints]}")


def example_training_workflow():
    """
    Complete checkpoint workflow for training.

    Shows how to:
    - Resume from existing checkpoint
    - Save checkpoints periodically
    - Handle interruptions gracefully
    """
    print("Training workflow with checkpointing:")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Path(tmpdir) / "config.toml"
        config.write_text(f'''
[storage]
base_path = "{tmpdir}"

[checkpoint]
checkpoint_dir = "{tmpdir}/checkpoints"
compression = "lz4"
keep_last_n = 2
''')
        runtime = Runtime(str(config))
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        # Initialize or restore state
        model_state = {'weights': [0.0] * 10, 'epoch': 0, 'loss': 1.0}
        start_epoch = 0

        # Try to resume from checkpoint
        existing = sorted(checkpoint_dir.glob("model_*.ckpt")) if checkpoint_dir.exists() else []
        if existing:
            latest = str(existing[-1])
            print(f"  Resuming from: {Path(latest).name}")
            state_bytes = runtime.load_checkpoint(latest)
            model_state = pickle.loads(state_bytes)
            start_epoch = model_state['epoch'] + 1

        # Training loop
        num_epochs = 5
        checkpoint_every = 2

        for epoch in range(start_epoch, num_epochs):
            # Simulate training
            model_state['epoch'] = epoch
            model_state['loss'] = 1.0 / (epoch + 1)

            print(f"  Epoch {epoch}: loss = {model_state['loss']:.4f}")

            # Save checkpoint periodically
            if (epoch + 1) % checkpoint_every == 0:
                state_bytes = pickle.dumps(model_state)
                path = runtime.save_checkpoint(f"model_epoch_{epoch}", state_bytes)
                print(f"    Saved: {Path(path).name}")

        print("  Training complete!")


def main():
    """Run all checkpointing examples."""
    print("=" * 60)
    print("Checkpointing - DTR Phase 1 Examples")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create runtime
        config = Path(tmpdir) / "config.toml"
        config.write_text(f'''
[storage]
base_path = "{tmpdir}"

[checkpoint]
checkpoint_dir = "{tmpdir}/checkpoints"
compression = "lz4"
keep_last_n = 3
''')
        runtime = Runtime(str(config))

        print("\n--- Example 1: Save Checkpoint ---")
        checkpoint_path = example_save_checkpoint(runtime)

        print("\n--- Example 2: Load Checkpoint ---")
        example_load_checkpoint(runtime, checkpoint_path)

        print("\n--- Example 3: Compression Options ---")
        example_compression_options()

        print("\n--- Example 4: Checkpoint Retention ---")
        example_checkpoint_retention(runtime)

        print("\n--- Example 5: Training Workflow ---")
        example_training_workflow()

    print("\n" + "=" * 60)
    print("All checkpointing examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
