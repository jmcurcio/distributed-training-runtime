#!/usr/bin/env python3
"""
S3 Checkpoint Resume Example - Resume Training from S3 Checkpoint

This example demonstrates how to:
1. List available checkpoints in S3
2. Load the latest checkpoint
3. Resume training from saved state
4. Save new checkpoints to S3

Prerequisites:
    cd rust/python-bindings && maturin develop --features s3

Setup:
    1. First run s3_training_example.py to create initial checkpoints
    2. Then run this script to resume training

Environment Variables (set in .env):
    DTR_S3_BUCKET         - S3 bucket name (required)
    DTR_S3_REGION         - AWS region (default: us-east-1)
    DTR_S3_CHECKPOINT_DIR - S3 key prefix for checkpoints
    AWS_ACCESS_KEY_ID     - AWS access key
    AWS_SECRET_ACCESS_KEY - AWS secret key
"""

import json
import os
import pickle
import re
from pathlib import Path

from dtr import Runtime


# ============================================================================
# Configuration Loading
# ============================================================================

def load_dotenv(env_file: Path = None):
    """Load environment variables from .env file."""
    if env_file is None:
        env_file = Path(__file__).parent.parent / ".env"

    if not env_file.exists():
        return False

    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()
                if value and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = value
    return True


load_dotenv()

# Configuration from environment
S3_CONFIG = {
    'bucket': os.environ.get('DTR_S3_BUCKET', ''),
    'region': os.environ.get('DTR_S3_REGION', 'us-east-1'),
    'base_path': os.environ.get('DTR_S3_BASE_PATH', 'dtr-training/'),
    'checkpoint_dir': os.environ.get('DTR_S3_CHECKPOINT_DIR', 'dtr-training/checkpoints/'),
    'endpoint': os.environ.get('DTR_S3_ENDPOINT'),
}

S3_CREDENTIALS = {}
if os.environ.get('AWS_ACCESS_KEY_ID'):
    S3_CREDENTIALS['access_key'] = os.environ['AWS_ACCESS_KEY_ID']
if os.environ.get('AWS_SECRET_ACCESS_KEY'):
    S3_CREDENTIALS['secret_key'] = os.environ['AWS_SECRET_ACCESS_KEY']


# ============================================================================
# Model Definition (same as s3_training_example.py)
# ============================================================================

class SimpleModel:
    """Simple linear regression model for demonstration."""

    def __init__(self):
        self.weights = [0.0] * 10
        self.bias = 0.0
        self.learning_rate = 0.01

    def forward(self, features):
        return sum(w * f for w, f in zip(self.weights, features)) + self.bias

    def backward(self, features, output, label):
        error = output - label
        for i in range(len(self.weights)):
            if i < len(features):
                self.weights[i] -= self.learning_rate * error * features[i]
        self.bias -= self.learning_rate * error

    def state_dict(self):
        return {'weights': list(self.weights), 'bias': self.bias}

    def load_state_dict(self, state):
        self.weights = list(state['weights'])
        self.bias = state['bias']


# ============================================================================
# Runtime Creation
# ============================================================================

def create_s3_runtime():
    """Create runtime configured for S3 storage."""
    if not S3_CONFIG['bucket']:
        raise ValueError("DTR_S3_BUCKET not configured. Check your .env file.")

    kwargs = {
        'backend': 's3',
        'base_path': '.',
        's3_bucket': S3_CONFIG['bucket'],
        's3_region': S3_CONFIG['region'],
        'checkpoint_dir': S3_CONFIG['checkpoint_dir'],
        'compression': 'zstd',
    }

    if 'access_key' in S3_CREDENTIALS:
        kwargs['s3_access_key'] = S3_CREDENTIALS['access_key']
        kwargs['s3_secret_key'] = S3_CREDENTIALS['secret_key']

    if S3_CONFIG.get('endpoint'):
        kwargs['s3_endpoint'] = S3_CONFIG['endpoint']

    return Runtime(**kwargs)


# ============================================================================
# Checkpoint Management
# ============================================================================

def find_latest_checkpoint(runtime):
    """Find the latest checkpoint in S3 based on epoch number."""
    try:
        files = runtime.list_files(S3_CONFIG['checkpoint_dir'])
    except Exception as e:
        print(f"Error listing checkpoints: {e}")
        return None

    if not files:
        return None

    # Parse epoch numbers from checkpoint names
    checkpoints = []
    for f in files:
        # Match pattern like "model_epoch_5" or "model_epoch_5.dtr"
        match = re.search(r'model_epoch_(\d+)', f)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, f))

    if not checkpoints:
        return None

    # Return the checkpoint with highest epoch
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0]  # (epoch, filename)


def load_checkpoint(runtime, checkpoint_name):
    """Load checkpoint data from S3."""
    print(f"Loading checkpoint: {checkpoint_name}")
    data = runtime.load_checkpoint(checkpoint_name)
    return pickle.loads(data)


# ============================================================================
# Training
# ============================================================================

def train_model(runtime, dataset, model, start_epoch: int, num_epochs: int = 3):
    """Train model and save checkpoints."""
    print(f"\nTraining for {num_epochs} epochs starting from epoch {start_epoch}...")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0.0
        total_samples = 0

        for shard_id in range(dataset.num_shards):
            for batch in dataset.iter_shard(shard_id, batch_size=8192, prefetch=4):
                records = batch.decode('utf-8').splitlines()

                for record_str in records:
                    if not record_str.strip():
                        continue

                    record = json.loads(record_str)
                    features = record['features']
                    label = record['label']

                    output = model.forward(features)
                    loss = (output - label) ** 2
                    total_loss += loss
                    total_samples += 1

                    model.backward(features, output, label)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        print(f"   Epoch {epoch}: loss={avg_loss:.4f}, samples={total_samples}")

        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }
            state_bytes = pickle.dumps(state)
            path = runtime.save_checkpoint(f"model_epoch_{epoch}", state_bytes)
            print(f"      Checkpoint saved: {path}")

    return model


# ============================================================================
# Data Setup
# ============================================================================

def ensure_training_data(runtime):
    """Ensure training data exists locally (download from S3 if needed)."""
    s3_data_path = f"{S3_CONFIG['base_path']}train_data.jsonl"
    local_cache = Path("./.cache/train_data.jsonl")

    if local_cache.exists():
        print(f"Using cached data: {local_cache}")
        return local_cache

    print("Downloading training data from S3...")
    if not runtime.file_exists(s3_data_path):
        print(f"\nError: Training data not found in S3: {s3_data_path}")
        print("Run s3_training_example.py first to create training data.")
        return None

    local_cache.parent.mkdir(parents=True, exist_ok=True)
    runtime.download_file(s3_data_path, str(local_cache))
    print(f"   Downloaded to: {local_cache}")
    return local_cache


# ============================================================================
# Main
# ============================================================================

def main():
    """Resume training from S3 checkpoint."""
    print("=" * 60)
    print("S3 Checkpoint Resume Example")
    print("=" * 60)

    # Check configuration
    if not S3_CONFIG['bucket']:
        print("\nError: S3 not configured. Set DTR_S3_BUCKET in .env")
        return

    print(f"\nS3 Bucket: {S3_CONFIG['bucket']}")
    print(f"Checkpoint Dir: {S3_CONFIG['checkpoint_dir']}")

    # Create runtime
    print("\nConnecting to S3...")
    try:
        runtime = create_s3_runtime()
    except Exception as e:
        print(f"\nError: {e}")
        return

    # Find latest checkpoint
    print("\nLooking for checkpoints...")
    latest = find_latest_checkpoint(runtime)

    if latest is None:
        print("\nNo checkpoints found in S3.")
        print("Run s3_training_example.py first to create initial checkpoints.")
        return

    epoch, checkpoint_file = latest
    print(f"   Found: {checkpoint_file} (epoch {epoch})")

    # List all available checkpoints
    print("\nAll available checkpoints:")
    files = runtime.list_files(S3_CONFIG['checkpoint_dir'])
    for f in sorted(files):
        marker = " <-- resuming from here" if checkpoint_file in f else ""
        print(f"   {f}{marker}")

    # Load checkpoint
    print("\n" + "-" * 60)
    # Build full checkpoint path (checkpoint_dir + filename)
    checkpoint_path = f"{S3_CONFIG['checkpoint_dir']}{checkpoint_file}"

    state = load_checkpoint(runtime, checkpoint_path)
    print(f"   Loaded epoch: {state['epoch']}")
    print(f"   Previous loss: {state['loss']:.4f}")
    print(f"   Model weights: {state['model']['weights'][:3]}... (truncated)")

    # Create model and load state
    model = SimpleModel()
    model.load_state_dict(state['model'])

    # Ensure training data exists
    local_data = ensure_training_data(runtime)
    if local_data is None:
        return

    # Register dataset
    dataset = runtime.register_dataset(str(local_data), shards=4)
    print(f"\nDataset: {dataset.total_bytes:,} bytes, {dataset.num_shards} shards")

    # Continue training for 4 more epochs
    print("\n" + "-" * 60)
    start_epoch = state['epoch'] + 1
    train_model(runtime, dataset, model, start_epoch=start_epoch, num_epochs=4)

    # Show final results
    print("\n" + "=" * 60)
    print("Training resumed and completed!")
    print("=" * 60)

    print("\nFinal model state:")
    print(f"   Weights: {model.weights[:3]}... (truncated)")
    print(f"   Bias: {model.bias:.4f}")

    print("\nAll checkpoints now in S3:")
    files = runtime.list_files(S3_CONFIG['checkpoint_dir'])
    for f in sorted(files):
        print(f"   {f}")


if __name__ == "__main__":
    main()
