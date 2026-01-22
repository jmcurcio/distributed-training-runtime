# DTR Examples

This directory contains examples for the Distributed Training Runtime (DTR).

## Prerequisites

```bash
# Build and install the Python bindings
cd rust/python-bindings
maturin develop

# For S3 support
maturin develop --features s3
```

## Quick Start

For a minimal example, run:

```bash
python quickstart.py
```

## Directory Structure

```
examples/
├── quickstart.py           # 5-minute getting started guide
├── .env.example            # Template for S3 configuration
│
├── core/                   # Core Features (single-node, local storage)
│   ├── 01_runtime_basics.py    # Runtime init, config, env vars
│   ├── 02_dataset_iteration.py # Datasets, sharding, iteration
│   ├── 03_record_formats.py    # Newline, fixed, length-prefixed
│   ├── 04_checkpointing.py     # Save/load, compression, retention
│   ├── 05_progress_errors.py   # Progress tracking, error handling
│   ├── 06_prefetching.py       # Background I/O prefetching
│   └── run_all.py              # Run all core examples
│
├── s3/                     # S3 Cloud Storage Examples
│   ├── s3_training_example.py  # End-to-end S3 training workflow
│   ├── s3_checkpoint_resume.py # Resume training from S3 checkpoint
│   ├── s3_file_operations.py   # Upload/download/list/exists ops
│   └── s3_minio_example.py     # Using MinIO (S3-compatible)
│
├── distributed/            # Distributed Training
│   ├── pytorch_ddp.py          # PyTorch DDP integration
│   └── multiprocess.py         # Python multiprocessing
│
└── advanced/               # Advanced Patterns
    ├── binary_formats.py       # Binary data deep dive
    ├── large_scale.py          # TB-scale processing
    └── complete_pipeline.py    # Full training pipeline
```

## Running Examples

### Individual Examples

```bash
# Run a specific example
python core/01_runtime_basics.py
python core/06_prefetching.py
python s3/s3_training_example.py
```

### All Core Examples

```bash
python core/run_all.py
```

### Distributed Examples

```bash
# PyTorch DDP (requires torch)
pip install torch
torchrun --nproc_per_node=4 distributed/pytorch_ddp.py

# Multiprocessing
python distributed/multiprocess.py
```

### S3 Examples

The `s3/` directory contains practical S3 examples:

```bash
# 1. Copy the environment template
cp .env.example .env

# 2. Edit .env with your S3 bucket details and AWS credentials
nano .env  # or your preferred editor

# 3. Run the examples
python s3/s3_training_example.py    # End-to-end training
python s3/s3_checkpoint_resume.py   # Resume from checkpoint
python s3/s3_file_operations.py     # File ops demo
python s3/s3_minio_example.py       # MinIO local testing
```

The `.env` file supports these variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DTR_S3_BUCKET` | S3 bucket name | (required) |
| `DTR_S3_REGION` | AWS region | `us-east-1` |
| `DTR_S3_BASE_PATH` | Key prefix for data | `dtr-training/` |
| `DTR_S3_CHECKPOINT_DIR` | Key prefix for checkpoints | `dtr-training/checkpoints/` |
| `DTR_S3_ENDPOINT` | Custom endpoint (MinIO) | (optional) |
| `AWS_ACCESS_KEY_ID` | AWS access key | (required unless using IAM) |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | (required unless using IAM) |

### MinIO (Local S3)

For local development without AWS:

```bash
# Start MinIO
docker run -d --name minio \
    -p 9000:9000 -p 9001:9001 \
    -e MINIO_ROOT_USER=minioadmin \
    -e MINIO_ROOT_PASSWORD=minioadmin \
    minio/minio server /data --console-address ":9001"

# Create bucket via console at http://localhost:9001
# Or via mc CLI: mc mb local/dtr-test

# Run MinIO example
python s3/s3_minio_example.py
```

## Example Categories

### Core (Single-Node)

| Example | Description |
|---------|-------------|
| 01_runtime_basics.py | Initialization, configuration, environment variables |
| 02_dataset_iteration.py | Registration, sharding, batch processing |
| 03_record_formats.py | Newline, fixed-size, length-prefixed formats |
| 04_checkpointing.py | Save/load, compression, retention policies |
| 05_progress_errors.py | Progress monitoring, error handling |
| 06_prefetching.py | Background I/O for reduced stalls |

### S3 (Cloud Storage)

| Example | Description |
|---------|-------------|
| s3_training_example.py | End-to-end cloud training workflow |
| s3_checkpoint_resume.py | Resume training from S3 checkpoint |
| s3_file_operations.py | Upload, download, list, exists operations |
| s3_minio_example.py | Local S3-compatible storage with MinIO |

### Distributed Training

| Example | Description |
|---------|-------------|
| pytorch_ddp.py | Integration with PyTorch DistributedDataParallel |
| multiprocess.py | Python multiprocessing patterns |

### Advanced Patterns

| Example | Description |
|---------|-------------|
| binary_formats.py | Deep dive into binary data handling |
| large_scale.py | TB-scale processing techniques |
| complete_pipeline.py | End-to-end training example |

## Common Patterns

### Basic Usage

```python
from dtr import Runtime

# Create runtime (uses defaults or config file)
runtime = Runtime()  # or Runtime("config.toml")

# Register dataset with sharding
dataset = runtime.register_dataset("data.jsonl", shards=4)

# Iterate over shards
for batch in dataset.iter_shard(0, batch_size=65536):
    records = batch.decode('utf-8').splitlines()
    for record in records:
        process(json.loads(record))
```

### With Prefetching

```python
# Enable prefetching for reduced I/O stalls
for batch in dataset.iter_shard(0, batch_size=65536, prefetch=4):
    process(batch)
```

### Checkpointing

```python
import pickle

# Save
state_bytes = pickle.dumps(model_state)
runtime.save_checkpoint("model_v1", state_bytes)

# Load
state_bytes = runtime.load_checkpoint("path/to/checkpoint.ckpt")
model_state = pickle.loads(state_bytes)
```

## Troubleshooting

### Import Error

```
ModuleNotFoundError: No module named 'dtr'
```

Solution: Build and install the Python bindings:
```bash
cd rust/python-bindings && maturin develop
```

### S3 Not Available

```
ValueError: S3 backend not available
```

Solution: Build with S3 feature:
```bash
cd rust/python-bindings && maturin develop --features s3
```

### Permission Denied

```
IOError: Storage error at '...': permission denied
```

Solution: Check file permissions and ensure the path is accessible.
