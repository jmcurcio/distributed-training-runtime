#!/usr/bin/env python3
"""
S3 Training Example - End-to-End Cloud Training Workflow

This example demonstrates using DTR with a real S3 bucket for a complete
cloud-native training workflow: data stored in S3, checkpoints saved to S3.

Prerequisites:
    cd rust/python-bindings && maturin develop --features s3

Setup:
    1. Copy examples/.env.example to examples/.env
    2. Edit examples/.env with your S3 bucket details and AWS credentials
    3. Run this script

Environment Variables (set in .env):
    DTR_S3_BUCKET         - S3 bucket name (required)
    DTR_S3_REGION         - AWS region (default: us-east-1)
    DTR_S3_BASE_PATH      - S3 key prefix for data (default: dtr-training/)
    DTR_S3_CHECKPOINT_DIR - S3 key prefix for checkpoints
    DTR_S3_ENDPOINT       - Custom endpoint for MinIO (optional)
    AWS_ACCESS_KEY_ID     - AWS access key (required unless using IAM role)
    AWS_SECRET_ACCESS_KEY - AWS secret key (required unless using IAM role)
"""

import json
import os
import pickle
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
_s3_bucket = os.environ.get('DTR_S3_BUCKET', '')
USE_S3 = bool(_s3_bucket and _s3_bucket != 'your-bucket-name')

S3_CONFIG = {
    'bucket': os.environ.get('DTR_S3_BUCKET', 'your-bucket-name'),
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
# Model Definition
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
# Data Generation
# ============================================================================

def create_sample_data(path: Path, num_records: int = 1000):
    """Create sample training data."""
    print(f"Creating sample training data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        for i in range(num_records):
            record = {
                "id": i,
                "features": [(i % 10) * 0.1] * 10,
                "label": i % 3,
            }
            f.write(json.dumps(record) + '\n')

    print(f"   Created {path} ({path.stat().st_size:,} bytes)")
    return path


# ============================================================================
# Runtime Creation
# ============================================================================

def create_s3_runtime():
    """Create runtime configured for S3 storage."""
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


def create_local_runtime():
    """Create runtime configured for local storage."""
    return Runtime()


# ============================================================================
# Training
# ============================================================================

def train_model(runtime, dataset, start_epoch: int = 0, num_epochs: int = 3):
    """Train model and save checkpoints to S3."""
    print("\nTraining model...")

    model = SimpleModel()

    # Resume from checkpoint if provided
    if start_epoch > 0:
        print(f"   (Resuming from epoch {start_epoch})")

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
# Main
# ============================================================================

def main():
    """Main training workflow."""
    print("=" * 60)
    print("S3 Training Example")
    print("=" * 60)

    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        print(f"\nConfiguration loaded from: {env_file}")
    else:
        print(f"\nNo .env file found at: {env_file}")

    # Local mode
    if not USE_S3:
        print("\nMode: Local storage (S3 not configured)")
        print("\nTo use S3, create examples/.env with your bucket details.")

        local_data = Path("./train_data.jsonl")
        if not local_data.exists():
            create_sample_data(local_data)

        runtime = create_local_runtime()
        dataset = runtime.register_dataset("train_data.jsonl", shards=4)
        print(f"\nDataset: {dataset.total_bytes:,} bytes, {dataset.num_shards} shards")

        train_model(runtime, dataset)
        print("\nTraining complete! (local mode)")
        return

    # S3 mode
    print("\nMode: S3 storage")
    print(f"Bucket: {S3_CONFIG['bucket']}")
    print(f"Region: {S3_CONFIG['region']}")
    if S3_CREDENTIALS:
        print("Credentials: Loaded from .env")
    else:
        print("Credentials: Using IAM role")

    # Connect to S3
    print("\nConnecting to S3...")
    try:
        runtime = create_s3_runtime()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nCheck your S3 configuration in .env")
        return

    # S3 data path
    s3_data_path = f"{S3_CONFIG['base_path']}train_data.jsonl"
    local_cache = Path("./.cache/train_data.jsonl")

    # Ensure data exists in S3
    print("\nChecking S3 for training data...")
    if not runtime.file_exists(s3_data_path):
        print(f"   Not found: {s3_data_path}")
        print("   Creating and uploading sample data...")
        create_sample_data(local_cache)
        runtime.upload_file(str(local_cache), s3_data_path)
        print(f"   Uploaded to: s3://{S3_CONFIG['bucket']}/{s3_data_path}")
    else:
        print(f"   Found: s3://{S3_CONFIG['bucket']}/{s3_data_path}")
        # Download to local cache
        print("\nDownloading to local cache...")
        local_cache.parent.mkdir(parents=True, exist_ok=True)
        runtime.download_file(s3_data_path, str(local_cache))
        print(f"   Cached: {local_cache}")

    # Register dataset
    print("\nRegistering dataset...")
    dataset = runtime.register_dataset(str(local_cache), shards=4)
    print(f"   {dataset.total_bytes:,} bytes, {dataset.num_shards} shards")

    # Train
    train_model(runtime, dataset)

    # Show results
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    print(f"\nCheckpoints in S3:")
    try:
        files = runtime.list_files(S3_CONFIG['checkpoint_dir'])
        for f in files[-5:]:  # Show last 5
            print(f"   {f}")
        if len(files) > 5:
            print(f"   ... and {len(files) - 5} more")
    except Exception as e:
        print(f"   Error listing: {e}")

    print(f"\nTo list all checkpoints:")
    print(f"   aws s3 ls s3://{S3_CONFIG['bucket']}/{S3_CONFIG['checkpoint_dir']}")


if __name__ == "__main__":
    main()
