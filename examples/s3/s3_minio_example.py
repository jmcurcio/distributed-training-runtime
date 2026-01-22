#!/usr/bin/env python3
"""
MinIO Example - Using DTR with S3-Compatible Local Storage

This example demonstrates using DTR with MinIO, a high-performance
S3-compatible object storage server. Useful for:
- Local development and testing
- On-premises deployments
- Air-gapped environments

Prerequisites:
    cd rust/python-bindings && maturin develop --features s3

MinIO Setup:
    # Start MinIO with Docker
    docker run -d --name minio \
        -p 9000:9000 \
        -p 9001:9001 \
        -e MINIO_ROOT_USER=minioadmin \
        -e MINIO_ROOT_PASSWORD=minioadmin \
        minio/minio server /data --console-address ":9001"

    # Access MinIO Console at http://localhost:9001
    # Create a bucket called "dtr-test" via the console or CLI

    # Or use mc (MinIO Client) to create bucket:
    mc alias set local http://localhost:9000 minioadmin minioadmin
    mc mb local/dtr-test

Environment Variables:
    Set these in .env or export them:

    DTR_S3_BUCKET=dtr-test
    DTR_S3_REGION=us-east-1
    DTR_S3_ENDPOINT=http://localhost:9000
    AWS_ACCESS_KEY_ID=minioadmin
    AWS_SECRET_ACCESS_KEY=minioadmin
"""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

from dtr import Runtime


# ============================================================================
# Configuration
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


# Default MinIO configuration (can be overridden by .env)
MINIO_DEFAULTS = {
    'DTR_S3_BUCKET': 'dtr-test',
    'DTR_S3_REGION': 'us-east-1',
    'DTR_S3_ENDPOINT': 'http://localhost:9000',
    'DTR_S3_BASE_PATH': 'training/',
    'DTR_S3_CHECKPOINT_DIR': 'training/checkpoints/',
    'AWS_ACCESS_KEY_ID': 'minioadmin',
    'AWS_SECRET_ACCESS_KEY': 'minioadmin',
}


def setup_minio_config():
    """Set up MinIO configuration with defaults."""
    # First try to load .env
    load_dotenv()

    # Apply MinIO defaults for any missing values
    for key, default in MINIO_DEFAULTS.items():
        if not os.environ.get(key):
            os.environ[key] = default


setup_minio_config()

# Configuration from environment
MINIO_CONFIG = {
    'bucket': os.environ.get('DTR_S3_BUCKET'),
    'region': os.environ.get('DTR_S3_REGION'),
    'endpoint': os.environ.get('DTR_S3_ENDPOINT'),
    'base_path': os.environ.get('DTR_S3_BASE_PATH'),
    'checkpoint_dir': os.environ.get('DTR_S3_CHECKPOINT_DIR'),
    'access_key': os.environ.get('AWS_ACCESS_KEY_ID'),
    'secret_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
}


# ============================================================================
# MinIO Status Check
# ============================================================================

def check_minio_available():
    """Check if MinIO is running and accessible."""
    import urllib.request
    import urllib.error

    endpoint = MINIO_CONFIG['endpoint']
    health_url = f"{endpoint}/minio/health/ready"

    try:
        req = urllib.request.Request(health_url, method='GET')
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


def print_minio_setup_instructions():
    """Print instructions for setting up MinIO."""
    print("""
MinIO Setup Instructions
========================

Option 1: Docker (Recommended)
------------------------------
docker run -d --name minio \\
    -p 9000:9000 \\
    -p 9001:9001 \\
    -e MINIO_ROOT_USER=minioadmin \\
    -e MINIO_ROOT_PASSWORD=minioadmin \\
    minio/minio server /data --console-address ":9001"

Then create a bucket:
    # Via MinIO Console: http://localhost:9001
    # Or via mc CLI:
    mc alias set local http://localhost:9000 minioadmin minioadmin
    mc mb local/dtr-test

Option 2: Direct Download
-------------------------
Download from: https://min.io/download
Run: minio server /data

Environment Variables
---------------------
Add to examples/.env:

    DTR_S3_BUCKET=dtr-test
    DTR_S3_ENDPOINT=http://localhost:9000
    AWS_ACCESS_KEY_ID=minioadmin
    AWS_SECRET_ACCESS_KEY=minioadmin
""")


# ============================================================================
# Runtime Creation
# ============================================================================

def create_minio_runtime():
    """Create runtime configured for MinIO."""
    return Runtime(
        backend='s3',
        base_path='.',
        s3_bucket=MINIO_CONFIG['bucket'],
        s3_region=MINIO_CONFIG['region'],
        s3_endpoint=MINIO_CONFIG['endpoint'],
        s3_access_key=MINIO_CONFIG['access_key'],
        s3_secret_key=MINIO_CONFIG['secret_key'],
        checkpoint_dir=MINIO_CONFIG['checkpoint_dir'],
        compression='zstd',
    )


# ============================================================================
# Model (simplified for demo)
# ============================================================================

class SimpleModel:
    """Simple model for demonstration."""

    def __init__(self):
        self.weights = [0.0] * 5
        self.bias = 0.0

    def forward(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def train_step(self, x, y, lr=0.01):
        pred = self.forward(x)
        error = pred - y
        for i in range(len(self.weights)):
            if i < len(x):
                self.weights[i] -= lr * error * x[i]
        self.bias -= lr * error
        return error ** 2

    def state_dict(self):
        return {'weights': self.weights.copy(), 'bias': self.bias}

    def load_state_dict(self, state):
        self.weights = state['weights'].copy()
        self.bias = state['bias']


# ============================================================================
# Demo Functions
# ============================================================================

def demo_basic_operations(runtime):
    """Demonstrate basic MinIO operations."""
    print("\n" + "-" * 50)
    print("Basic Operations")
    print("-" * 50)

    # Upload test file
    test_data = {"message": "Hello from DTR!", "backend": "MinIO"}
    test_path = f"{MINIO_CONFIG['base_path']}test.json"

    # Create temp file and upload
    temp_file = Path("/tmp/dtr_minio_test.json")
    with open(temp_file, 'w') as f:
        json.dump(test_data, f)

    print(f"\n1. Uploading test file to MinIO...")
    print(f"   Endpoint: {MINIO_CONFIG['endpoint']}")
    print(f"   Bucket:   {MINIO_CONFIG['bucket']}")
    print(f"   Key:      {test_path}")

    runtime.upload_file(str(temp_file), test_path)
    print("   Upload successful!")

    # Check exists
    print(f"\n2. Checking file exists...")
    exists = runtime.file_exists(test_path)
    print(f"   file_exists: {exists}")

    # Download and verify
    download_path = Path("/tmp/dtr_minio_test_download.json")
    print(f"\n3. Downloading file...")
    runtime.download_file(test_path, str(download_path))

    with open(download_path) as f:
        downloaded = json.load(f)

    if downloaded == test_data:
        print("   Download successful - contents verified!")
    else:
        print("   WARNING: Downloaded contents differ!")

    # Cleanup
    temp_file.unlink()
    download_path.unlink()


def demo_training_workflow(runtime):
    """Demonstrate training with checkpoints to MinIO."""
    print("\n" + "-" * 50)
    print("Training Workflow")
    print("-" * 50)

    # Create sample data
    print("\n1. Creating training data...")
    data_path = Path("/tmp/minio_train_data.jsonl")
    with open(data_path, 'w') as f:
        for i in range(100):
            record = {
                "features": [(i % 5) * 0.2] * 5,
                "label": (i % 5) * 0.5,
            }
            f.write(json.dumps(record) + '\n')
    print(f"   Created {data_path} with 100 records")

    # Upload to MinIO
    s3_data_path = f"{MINIO_CONFIG['base_path']}train_data.jsonl"
    print(f"\n2. Uploading data to MinIO...")
    runtime.upload_file(str(data_path), s3_data_path)
    print(f"   Uploaded to: {s3_data_path}")

    # Register dataset (from local file for iteration)
    print("\n3. Registering dataset...")
    dataset = runtime.register_dataset(str(data_path), shards=2)
    print(f"   Shards: {dataset.num_shards}")
    print(f"   Size: {dataset.total_bytes} bytes")

    # Train
    print("\n4. Training model...")
    model = SimpleModel()

    for epoch in range(3):
        total_loss = 0
        samples = 0

        for shard_id in range(dataset.num_shards):
            for batch in dataset.iter_shard(shard_id, batch_size=1024):
                for line in batch.decode().splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    loss = model.train_step(record['features'], record['label'])
                    total_loss += loss
                    samples += 1

        avg_loss = total_loss / samples if samples > 0 else 0
        print(f"   Epoch {epoch}: loss={avg_loss:.6f}")

        # Save checkpoint to MinIO
        state = {'model': model.state_dict(), 'epoch': epoch, 'loss': avg_loss}
        checkpoint_path = runtime.save_checkpoint(f"minio_model_epoch_{epoch}", pickle.dumps(state))
        print(f"      Checkpoint: {checkpoint_path}")

    # Cleanup local file
    data_path.unlink()

    # List checkpoints in MinIO
    print("\n5. Listing checkpoints in MinIO...")
    files = runtime.list_files(MINIO_CONFIG['checkpoint_dir'])
    for f in sorted(files):
        print(f"   {f}")


def demo_checkpoint_resume(runtime):
    """Demonstrate loading checkpoint from MinIO."""
    print("\n" + "-" * 50)
    print("Checkpoint Resume")
    print("-" * 50)

    # Find latest checkpoint
    files = runtime.list_files(MINIO_CONFIG['checkpoint_dir'])
    minio_checkpoints = [f for f in files if 'minio_model' in f]

    if not minio_checkpoints:
        print("\n   No MinIO checkpoints found. Run training workflow first.")
        return

    # Load latest
    latest = sorted(minio_checkpoints)[-1]
    checkpoint_name = latest.split('/')[-1].replace('.dtr', '')

    print(f"\n1. Loading checkpoint: {checkpoint_name}")
    data = runtime.load_checkpoint(checkpoint_name)
    state = pickle.loads(data)

    print(f"   Epoch: {state['epoch']}")
    print(f"   Loss: {state['loss']:.6f}")
    print(f"   Weights: {state['model']['weights']}")
    print(f"   Bias: {state['model']['bias']:.6f}")

    # Restore model
    print("\n2. Restoring model state...")
    model = SimpleModel()
    model.load_state_dict(state['model'])
    print("   Model restored successfully!")

    # Test inference
    print("\n3. Testing inference...")
    test_input = [0.2, 0.4, 0.6, 0.8, 1.0]
    prediction = model.forward(test_input)
    print(f"   Input: {test_input}")
    print(f"   Output: {prediction:.6f}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run MinIO demonstration."""
    print("=" * 60)
    print("MinIO Example - S3-Compatible Local Storage")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"   Endpoint:   {MINIO_CONFIG['endpoint']}")
    print(f"   Bucket:     {MINIO_CONFIG['bucket']}")
    print(f"   Base Path:  {MINIO_CONFIG['base_path']}")
    print(f"   Checkpoint: {MINIO_CONFIG['checkpoint_dir']}")

    # Check if MinIO is available
    print("\nChecking MinIO availability...")
    if not check_minio_available():
        print("   MinIO is not running or not accessible.")
        print_minio_setup_instructions()
        return

    print("   MinIO is available!")

    # Create runtime
    print("\nConnecting to MinIO...")
    try:
        runtime = create_minio_runtime()
    except Exception as e:
        print(f"\nError connecting to MinIO: {e}")
        print("\nMake sure:")
        print(f"  1. MinIO is running at {MINIO_CONFIG['endpoint']}")
        print(f"  2. Bucket '{MINIO_CONFIG['bucket']}' exists")
        print("  3. Credentials are correct")
        return

    print("   Connected!")

    # Run demos
    demo_basic_operations(runtime)
    demo_training_workflow(runtime)
    demo_checkpoint_resume(runtime)

    # Summary
    print("\n" + "=" * 60)
    print("MinIO demonstration complete!")
    print("=" * 60)

    print(f"\nMinIO Console: http://localhost:9001")
    print(f"Bucket: {MINIO_CONFIG['bucket']}")

    print("\nTo view files via MinIO Console or mc:")
    print(f"   mc ls local/{MINIO_CONFIG['bucket']}/{MINIO_CONFIG['base_path']}")

    print("\nTo clean up:")
    print(f"   mc rm --recursive local/{MINIO_CONFIG['bucket']}/{MINIO_CONFIG['base_path']}")


if __name__ == "__main__":
    main()
