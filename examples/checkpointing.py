#!/usr/bin/env python3
"""
Comprehensive Checkpointing Examples with DTR

This module demonstrates checkpoint management including:
- Saving and loading checkpoints
- Compression options (none, lz4, zstd)
- Checkpoint retention policies
- PyTorch model checkpointing
- Recovery from interruptions

Prerequisites:
    cd rust/python-bindings && maturin develop
"""

import os
import pickle
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dtr import Runtime


# =============================================================================
# Basic Checkpointing
# =============================================================================

def example_basic_checkpoint():
    """
    Basic checkpoint save and load operations.

    Checkpoints are:
    - Compressed automatically (lz4 default)
    - Written atomically (crash-safe)
    - Verified with checksums on load
    """
    print("\n" + "=" * 60)
    print("Basic Checkpointing")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir()

        config = f"""
[storage]
base_path = "{data_dir}"

[checkpoint]
checkpoint_dir = "{checkpoint_dir}"
compression = "lz4"
keep_last_n = 5
"""
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(config)

        runtime = Runtime(str(config_file))

        # Save a simple checkpoint
        state = {
            "epoch": 5,
            "step": 10000,
            "learning_rate": 0.001,
            "model_weights": [1.0, 2.0, 3.0] * 1000,  # Simulated weights
        }

        state_bytes = pickle.dumps(state)
        original_size = len(state_bytes)
        print(f"Original state size: {original_size:,} bytes")

        # Save checkpoint
        start = time.time()
        checkpoint_path = runtime.save_checkpoint("model", state_bytes)
        save_time = time.time() - start

        compressed_size = Path(checkpoint_path).stat().st_size
        ratio = original_size / compressed_size

        print(f"Saved to: {checkpoint_path}")
        print(f"Compressed size: {compressed_size:,} bytes")
        print(f"Compression ratio: {ratio:.2f}x")
        print(f"Save time: {save_time*1000:.2f}ms")

        # Load checkpoint
        start = time.time()
        loaded_bytes = runtime.load_checkpoint(checkpoint_path)
        load_time = time.time() - start

        loaded_state = pickle.loads(loaded_bytes)

        print(f"Load time: {load_time*1000:.2f}ms")
        print(f"Loaded epoch: {loaded_state['epoch']}")
        print(f"Loaded step: {loaded_state['step']}")
        print(f"State matches: {state == loaded_state}")


# =============================================================================
# Compression Comparison
# =============================================================================

def example_compression_comparison():
    """
    Compare different compression algorithms.

    Available options:
    - "none": No compression (fastest save, largest files)
    - "lz4": Fast compression (good balance, default)
    - "zstd": Best compression (smaller files, slower)
    """
    print("\n" + "=" * 60)
    print("Compression Comparison")
    print("=" * 60)

    # Create test data (compressible)
    test_data = {
        "weights": [1.0] * 100000,  # Repetitive data compresses well
        "gradients": [0.01] * 100000,
        "metadata": {f"key_{i}": f"value_{i}" for i in range(100)},
    }
    data_bytes = pickle.dumps(test_data)
    original_size = len(data_bytes)

    print(f"Original size: {original_size:,} bytes\n")

    results = []

    for compression in ["none", "lz4", "zstd"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()

            config = f"""
[storage]
base_path = "{data_dir}"

[checkpoint]
checkpoint_dir = "{checkpoint_dir}"
compression = "{compression}"
"""
            config_file = Path(tmpdir) / "config.toml"
            config_file.write_text(config)

            runtime = Runtime(str(config_file))

            # Benchmark save
            start = time.time()
            for i in range(5):
                path = runtime.save_checkpoint(f"bench_{i}", data_bytes)
            save_time = (time.time() - start) / 5

            compressed_size = Path(path).stat().st_size

            # Benchmark load
            start = time.time()
            for i in range(5):
                loaded = runtime.load_checkpoint(path)
            load_time = (time.time() - start) / 5

            ratio = original_size / compressed_size
            results.append({
                "compression": compression,
                "size": compressed_size,
                "ratio": ratio,
                "save_ms": save_time * 1000,
                "load_ms": load_time * 1000,
            })

    # Print comparison table
    print(f"{'Algorithm':<12} {'Size':>12} {'Ratio':>8} {'Save':>10} {'Load':>10}")
    print("-" * 54)
    for r in results:
        print(f"{r['compression']:<12} {r['size']:>12,} {r['ratio']:>7.2f}x {r['save_ms']:>9.2f}ms {r['load_ms']:>9.2f}ms")


# =============================================================================
# Checkpoint Retention
# =============================================================================

def example_checkpoint_retention():
    """
    Automatic cleanup of old checkpoints.

    The keep_last_n config option limits how many checkpoints are retained.
    Older checkpoints are automatically deleted when new ones are saved.
    """
    print("\n" + "=" * 60)
    print("Checkpoint Retention (keep_last_n)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir()

        config = f"""
[storage]
base_path = "{data_dir}"

[checkpoint]
checkpoint_dir = "{checkpoint_dir}"
compression = "lz4"
keep_last_n = 3
"""
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(config)

        runtime = Runtime(str(config_file))

        # Save multiple checkpoints
        for epoch in range(10):
            state = {"epoch": epoch}
            state_bytes = pickle.dumps(state)
            path = runtime.save_checkpoint(f"model_epoch_{epoch}", state_bytes)

            # List current checkpoints
            checkpoints = sorted(checkpoint_dir.glob("model_epoch_*.ckpt"))
            ckpt_names = [c.name for c in checkpoints]

            print(f"After epoch {epoch}: {len(checkpoints)} checkpoints: {ckpt_names}")

        print("\nFinal checkpoints (should be last 3):")
        for ckpt in sorted(checkpoint_dir.glob("*.ckpt")):
            print(f"  {ckpt.name}")


# =============================================================================
# PyTorch Integration
# =============================================================================

def example_pytorch_checkpointing():
    """
    Checkpoint PyTorch models and optimizers.

    Shows the recommended pattern for saving all training state
    needed to resume training exactly.
    """
    print("\n" + "=" * 60)
    print("PyTorch Model Checkpointing")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed, showing template only")
        print('''
# PyTorch checkpointing pattern:

def save_training_state(runtime, model, optimizer, scheduler, epoch, metrics):
    """Save complete training state."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    state_bytes = pickle.dumps(state)
    return runtime.save_checkpoint(f"training_epoch_{epoch}", state_bytes)


def load_training_state(runtime, checkpoint_path, model, optimizer, scheduler):
    """Restore complete training state."""
    state_bytes = runtime.load_checkpoint(checkpoint_path)
    state = pickle.loads(state_bytes)

    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler and state['scheduler_state_dict']:
        scheduler.load_state_dict(state['scheduler_state_dict'])

    # Restore RNG state for exact reproducibility
    torch.set_rng_state(state['rng_state'])
    if state['cuda_rng_state'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng_state'])

    return state['epoch'], state['metrics']
''')
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()

        os.environ["DTR_STORAGE_BASE_PATH"] = str(data_dir)
        runtime = Runtime()

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # Simulate training
        print("Simulating training...")
        for epoch in range(3):
            # Fake training step
            x = torch.randn(32, 100)
            y = model(x).sum()
            y.backward()
            optimizer.step()
            scheduler.step()

        # Save checkpoint
        print("Saving checkpoint...")
        state = {
            'epoch': 3,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'rng_state': torch.get_rng_state(),
        }

        state_bytes = pickle.dumps(state)
        checkpoint_path = runtime.save_checkpoint("pytorch_model", state_bytes)
        print(f"Saved to: {checkpoint_path}")

        # Create new model and restore
        print("Restoring to new model...")
        new_model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=5, gamma=0.1)

        loaded_bytes = runtime.load_checkpoint(checkpoint_path)
        loaded_state = pickle.loads(loaded_bytes)

        new_model.load_state_dict(loaded_state['model_state_dict'])
        new_optimizer.load_state_dict(loaded_state['optimizer_state_dict'])
        new_scheduler.load_state_dict(loaded_state['scheduler_state_dict'])
        torch.set_rng_state(loaded_state['rng_state'])

        # Verify restoration
        print(f"Restored epoch: {loaded_state['epoch']}")
        print(f"Restored LR: {new_scheduler.get_last_lr()}")

        # Verify weights match
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2), "Weights should match exactly"

        print("Checkpoint restoration verified!")


# =============================================================================
# Crash Recovery Pattern
# =============================================================================

def example_crash_recovery():
    """
    Pattern for recovering from crashes during training.

    This shows how to:
    1. Check for existing checkpoints on startup
    2. Resume from the latest checkpoint
    3. Handle partial epochs
    """
    print("\n" + "=" * 60)
    print("Crash Recovery Pattern")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir()

        config = f"""
[storage]
base_path = "{data_dir}"

[checkpoint]
checkpoint_dir = "{checkpoint_dir}"
keep_last_n = 5
"""
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(config)

        def simulate_training(max_epochs: int = 10, crash_at_epoch: Optional[int] = None):
            """Simulate training with optional crash."""
            runtime = Runtime(str(config_file))

            # Try to resume from checkpoint
            start_epoch = 0
            model_state = {"weights": [0.0] * 100}

            existing = sorted(checkpoint_dir.glob("training_*.ckpt"))
            if existing:
                latest = str(existing[-1])
                print(f"  Found checkpoint: {Path(latest).name}")
                state_bytes = runtime.load_checkpoint(latest)
                state = pickle.loads(state_bytes)
                start_epoch = state["epoch"] + 1
                model_state = state["model_state"]
                print(f"  Resuming from epoch {start_epoch}")
            else:
                print("  No checkpoint found, starting fresh")

            # Training loop
            for epoch in range(start_epoch, max_epochs):
                # Simulate crash
                if crash_at_epoch and epoch == crash_at_epoch:
                    print(f"  CRASH at epoch {epoch}!")
                    return False

                # Simulate training
                model_state["weights"] = [w + 0.1 for w in model_state["weights"]]

                # Save checkpoint
                state = {"epoch": epoch, "model_state": model_state}
                state_bytes = pickle.dumps(state)
                runtime.save_checkpoint(f"training_epoch_{epoch}", state_bytes)
                print(f"  Completed epoch {epoch}")

            return True

        # First run: crash at epoch 3
        print("\nFirst run (will crash at epoch 3):")
        simulate_training(max_epochs=10, crash_at_epoch=3)

        # Second run: resume and complete
        print("\nSecond run (resume and complete):")
        simulate_training(max_epochs=10)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("DTR Checkpointing Examples")
    print("=" * 70)

    example_basic_checkpoint()
    example_compression_comparison()
    example_checkpoint_retention()
    example_pytorch_checkpointing()
    example_crash_recovery()

    print("\n" + "=" * 70)
    print("All examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
