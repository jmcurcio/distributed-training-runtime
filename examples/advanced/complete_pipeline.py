#!/usr/bin/env python3
"""
Complete Training Pipeline - DTR Advanced Examples

Demonstrates a full end-to-end training pipeline including:
- Dataset creation and registration
- Multi-epoch training with sharding
- Checkpoint save and resume
- Progress monitoring
- Error handling and recovery

Prerequisites:
    cd rust/python-bindings && maturin develop
"""

import json
import pickle
import tempfile
from pathlib import Path

from dtr import Runtime


def create_training_data(data_dir: Path, num_records: int = 1000) -> Path:
    """Create synthetic training data."""
    train_file = data_dir / "train.jsonl"
    with open(train_file, 'w') as f:
        for i in range(num_records):
            record = {
                "id": i,
                "features": [(i % 10) * 0.1] * 10,
                "label": i % 3
            }
            f.write(json.dumps(record) + '\n')
    return train_file


class SimpleModel:
    """A simple linear model for demonstration."""

    def __init__(self, input_dim: int = 10, output_dim: int = 3):
        self.weights = [[0.0] * output_dim for _ in range(input_dim)]
        self.bias = [0.0] * output_dim
        self.learning_rate = 0.01

    def forward(self, features: list) -> list:
        """Compute model output."""
        output = list(self.bias)
        for i, feat in enumerate(features):
            for j in range(len(output)):
                output[j] += feat * self.weights[i][j]
        return output

    def compute_loss(self, output: list, label: int) -> float:
        """Compute cross-entropy loss (simplified)."""
        # Simplified: squared error to correct class
        target = [0.0] * len(output)
        target[label] = 1.0
        return sum((o - t) ** 2 for o, t in zip(output, target))

    def backward(self, features: list, output: list, label: int):
        """Update weights with gradient descent."""
        target = [0.0] * len(output)
        target[label] = 1.0

        # Compute gradients and update
        for i, feat in enumerate(features):
            for j in range(len(output)):
                grad = 2 * (output[j] - target[j]) * feat
                self.weights[i][j] -= self.learning_rate * grad

        for j in range(len(output)):
            grad = 2 * (output[j] - target[j])
            self.bias[j] -= self.learning_rate * grad

    def state_dict(self) -> dict:
        """Get model state for checkpointing."""
        return {
            'weights': self.weights,
            'bias': self.bias,
        }

    def load_state_dict(self, state: dict):
        """Load model state from checkpoint."""
        self.weights = state['weights']
        self.bias = state['bias']


def train_epoch(model: SimpleModel, dataset, epoch: int) -> dict:
    """Train for one epoch, return metrics."""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for shard_id in range(dataset.num_shards):
        for batch in dataset.iter_shard(shard_id, batch_size=4096, prefetch=4):
            records = batch.decode('utf-8').splitlines()

            for record_str in records:
                if not record_str.strip():
                    continue

                record = json.loads(record_str)
                features = record['features']
                label = record['label']

                # Forward pass
                output = model.forward(features)
                loss = model.compute_loss(output, label)

                # Track metrics
                total_loss += loss
                predicted = output.index(max(output))
                if predicted == label:
                    total_correct += 1
                total_samples += 1

                # Backward pass
                model.backward(features, output, label)

    return {
        'epoch': epoch,
        'loss': total_loss / total_samples if total_samples > 0 else 0,
        'accuracy': total_correct / total_samples if total_samples > 0 else 0,
        'samples': total_samples,
    }


def save_checkpoint(runtime: Runtime, model: SimpleModel, metrics: dict, name: str) -> str:
    """Save model checkpoint with metrics."""
    state = {
        'model': model.state_dict(),
        'epoch': metrics['epoch'],
        'loss': metrics['loss'],
        'accuracy': metrics['accuracy'],
    }
    state_bytes = pickle.dumps(state)
    return runtime.save_checkpoint(name, state_bytes)


def load_checkpoint(runtime: Runtime, path: str) -> dict:
    """Load checkpoint and return state."""
    state_bytes = runtime.load_checkpoint(path)
    return pickle.loads(state_bytes)


def find_latest_checkpoint(checkpoint_dir: Path, prefix: str = "model_") -> Path | None:
    """Find the most recent checkpoint."""
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob(f"{prefix}*.ckpt"))
    return checkpoints[-1] if checkpoints else None


def main():
    """Run complete training pipeline."""
    print("=" * 60)
    print("Complete Training Pipeline - DTR Advanced Examples")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup directories
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir()

        # Create training data
        print("\n1. Creating training data...")
        train_file = create_training_data(data_dir, num_records=1000)
        print(f"   Created: {train_file.name} ({train_file.stat().st_size:,} bytes)")

        # Create configuration
        print("\n2. Creating configuration...")
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(f'''
[storage]
base_path = "{data_dir}"

[checkpoint]
checkpoint_dir = "{checkpoint_dir}"
compression = "lz4"
keep_last_n = 3
''')
        print(f"   Config: {config_file.name}")

        # Initialize runtime
        print("\n3. Initializing runtime...")
        runtime = Runtime(str(config_file))
        print(f"   Base path: {runtime.base_path}")
        print(f"   Checkpoint dir: {runtime.checkpoint_dir}")
        print(f"   Compression: {runtime.compression}")

        # Register dataset
        print("\n4. Registering dataset...")
        dataset = runtime.register_dataset("train.jsonl", shards=4, format="newline")
        print(f"   Total bytes: {dataset.total_bytes:,}")
        print(f"   Shards: {dataset.num_shards}")

        # Initialize model
        print("\n5. Initializing model...")
        model = SimpleModel(input_dim=10, output_dim=3)
        start_epoch = 0

        # Try to resume from checkpoint
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt:
            print(f"   Resuming from: {latest_ckpt.name}")
            state = load_checkpoint(runtime, str(latest_ckpt))
            model.load_state_dict(state['model'])
            start_epoch = state['epoch'] + 1
            print(f"   Restored epoch {state['epoch']}, loss={state['loss']:.4f}")
        else:
            print("   Starting fresh (no checkpoint found)")

        # Training loop
        print("\n6. Training...")
        num_epochs = 5
        checkpoint_every = 2

        for epoch in range(start_epoch, num_epochs):
            metrics = train_epoch(model, dataset, epoch)

            print(f"   Epoch {epoch}: loss={metrics['loss']:.4f}, "
                  f"accuracy={metrics['accuracy']:.2%}, "
                  f"samples={metrics['samples']}")

            # Save checkpoint periodically
            if (epoch + 1) % checkpoint_every == 0 or epoch == num_epochs - 1:
                path = save_checkpoint(runtime, model, metrics, f"model_epoch_{epoch}")
                print(f"      Saved: {Path(path).name}")

        # Final evaluation
        print("\n7. Final model state:")
        print(f"   Weights shape: {len(model.weights)}x{len(model.weights[0])}")
        print(f"   Bias: {[f'{b:.4f}' for b in model.bias]}")

        # List checkpoints
        print("\n8. Checkpoints saved:")
        for ckpt in sorted(checkpoint_dir.glob("*.ckpt")):
            print(f"   - {ckpt.name} ({ckpt.stat().st_size:,} bytes)")

        # Demonstrate checkpoint loading
        print("\n9. Loading final checkpoint...")
        final_ckpt = find_latest_checkpoint(checkpoint_dir)
        if final_ckpt:
            state = load_checkpoint(runtime, str(final_ckpt))
            print(f"   Loaded: {final_ckpt.name}")
            print(f"   Epoch: {state['epoch']}")
            print(f"   Loss: {state['loss']:.4f}")
            print(f"   Accuracy: {state['accuracy']:.2%}")

    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
