#!/usr/bin/env python3
"""
Distributed Training with PyTorch DDP and DTR

This example shows how to use DTR for data loading in a distributed
PyTorch training setup. Each rank processes a different shard of the data.

Usage:
    # Single GPU (or match your GPU count)
    torchrun --nproc_per_node=1 pytorch_ddp.py

    # Auto-detect GPU count
    torchrun --nproc_per_node=gpu pytorch_ddp.py

    # Multiple GPUs (adjust to your hardware)
    torchrun --nproc_per_node=4 pytorch_ddp.py

    # Multi-node (run on each node)
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
             --master_addr=<master_ip> --master_port=29500 \
             pytorch_ddp.py

Prerequisites:
    pip install torch
    cd rust/python-bindings && maturin develop
"""

import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from dtr import Runtime


# Simple model for demonstration
class SimpleModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_training_data(data_dir: Path, num_samples: int = 1000) -> Path:
    """Create synthetic training data."""
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / "training_data.jsonl"

    with open(data_file, 'w') as f:
        for i in range(num_samples):
            record = {
                'features': [float(i % 10) * 0.1] * 128,
                'label': i % 10,
            }
            f.write(json.dumps(record) + '\n')

    return data_file


def parse_batch(batch_bytes: bytes) -> tuple:
    """Parse a batch of JSONL records into tensors."""
    lines = batch_bytes.decode('utf-8').splitlines()

    features = []
    labels = []

    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        features.append(record['features'])
        labels.append(record['label'])

    if not features:
        return None, None

    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return features_tensor, labels_tensor


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset,
    shard_id: int,
    device: torch.device,
    epoch: int,
    rank: int
) -> float:
    """Train for one epoch on the given shard."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    iterator = dataset.iter_shard(shard_id, batch_size=64 * 1024)

    for batch_bytes in iterator:
        features, labels = parse_batch(batch_bytes)

        if features is None:
            continue

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Log progress periodically
        if num_batches % 100 == 0 and rank == 0:
            progress = iterator.progress()
            print(f"  Epoch {epoch} | Batch {num_batches} | Progress: {progress:.1%} | Loss: {loss.item():.4f}")

    return total_loss / max(num_batches, 1)


def save_checkpoint(runtime: Runtime, model: nn.Module, optimizer, epoch: int, loss: float):
    """Save training checkpoint."""
    # Get underlying model from DDP wrapper
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    state = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }

    state_bytes = pickle.dumps(state)
    path = runtime.save_checkpoint(f"model_epoch_{epoch}", state_bytes)
    return path


def load_checkpoint(runtime: Runtime, path: str, model: nn.Module, optimizer):
    """Load training checkpoint."""
    state_bytes = runtime.load_checkpoint(path)
    state = pickle.loads(state_bytes)

    # Load into underlying model (handle DDP wrapper)
    if hasattr(model, 'module'):
        model.module.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state['model_state_dict'])

    optimizer.load_state_dict(state['optimizer_state_dict'])

    return state['epoch'], state['loss']


def main():
    # Check CUDA availability before initializing distributed
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This example requires GPUs.")
        print("For CPU-only distributed training, modify the script to use 'gloo' backend.")
        return

    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank >= num_gpus:
        print(f"Error: Process local_rank={local_rank} but only {num_gpus} GPU(s) available.")
        print(f"Run with: torchrun --nproc_per_node={num_gpus} pytorch_ddp.py")
        print(f"Or use: torchrun --nproc_per_node=gpu pytorch_ddp.py (auto-detect)")
        return

    # Initialize distributed environment
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Starting distributed training with {world_size} workers")

    # Create temp directory for this run (rank 0 creates, broadcasts path)
    if rank == 0:
        tmpdir = tempfile.mkdtemp(prefix="dtr_ddp_")
        tmpdir_bytes = tmpdir.encode('utf-8')
        tmpdir_len = torch.tensor([len(tmpdir_bytes)], dtype=torch.int64, device=device)
    else:
        tmpdir_len = torch.tensor([0], dtype=torch.int64, device=device)

    # Broadcast temp directory path length
    dist.broadcast(tmpdir_len, src=0)

    # Broadcast temp directory path
    if rank == 0:
        tmpdir_tensor = torch.tensor(list(tmpdir_bytes), dtype=torch.uint8, device=device)
    else:
        tmpdir_tensor = torch.zeros(tmpdir_len.item(), dtype=torch.uint8, device=device)

    dist.broadcast(tmpdir_tensor, src=0)
    tmpdir = bytes(tmpdir_tensor.cpu().tolist()).decode('utf-8')

    data_dir = Path(tmpdir) / "data"
    checkpoint_dir = Path(tmpdir) / "checkpoints"

    # Rank 0 creates the training data
    if rank == 0:
        print(f"Creating training data in: {tmpdir}")
        data_file = create_training_data(data_dir, num_samples=1000)
        print(f"  Created: {data_file} ({data_file.stat().st_size:,} bytes)")

    # Wait for data to be created
    dist.barrier()

    # Create config file for runtime
    config_path = Path(tmpdir) / f"config_rank{rank}.toml"
    config_path.write_text(f'''
[storage]
base_path = "{data_dir}"

[checkpoint]
checkpoint_dir = "{checkpoint_dir}"
''')

    # Initialize DTR runtime
    runtime = Runtime(str(config_path))

    if rank == 0:
        print(f"DTR Runtime initialized:")
        print(f"  Base path: {runtime.base_path}")
        print(f"  Checkpoint dir: {runtime.checkpoint_dir}")

    # Register dataset with one shard per worker
    # All workers must register the same dataset configuration
    dataset = runtime.register_dataset(
        path="training_data.jsonl",
        shards=world_size,
        format="newline"
    )

    if rank == 0:
        print(f"Dataset registered:")
        print(f"  Total bytes: {dataset.total_bytes:,}")
        print(f"  Shards: {dataset.num_shards}")

    # Each rank processes its own shard
    my_shard = rank

    # Model configuration (must match your data)
    input_dim = 128
    hidden_dim = 256
    output_dim = 10

    # Create model with DDP
    model = SimpleModel(input_dim, hidden_dim, output_dim).to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Check for existing checkpoint to resume from
    start_epoch = 0

    if rank == 0 and checkpoint_dir.exists():
        existing = sorted(checkpoint_dir.glob("model_epoch_*.ckpt"))
        if existing:
            latest = str(existing[-1])
            print(f"Resuming from checkpoint: {latest}")
            start_epoch, _ = load_checkpoint(runtime, latest, model, optimizer)
            start_epoch += 1  # Start from next epoch

    # Broadcast start_epoch from rank 0 to all ranks
    start_tensor = torch.tensor([start_epoch], device=device)
    dist.broadcast(start_tensor, src=0)
    start_epoch = int(start_tensor.item())

    # Training loop
    num_epochs = 5
    checkpoint_every = 2

    for epoch in range(start_epoch, num_epochs):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs-1}")
            print(f"{'='*60}")

        # Train on this rank's shard
        avg_loss = train_epoch(model, optimizer, dataset, my_shard, device, epoch, rank)

        # Gather losses from all ranks
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_avg_loss = loss_tensor.item() / world_size

        if rank == 0:
            print(f"Epoch {epoch} complete | Global avg loss: {global_avg_loss:.4f}")

        # Synchronize before checkpoint
        dist.barrier()

        # Save checkpoint from rank 0 only
        if epoch % checkpoint_every == 0 and rank == 0:
            path = save_checkpoint(runtime, model, optimizer, epoch, global_avg_loss)
            print(f"Checkpoint saved: {path}")

        # Synchronize after checkpoint
        dist.barrier()

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

    # Rank 0 cleans up the temp directory
    if rank == 0:
        print(f"\nCleaning up temp directory: {tmpdir}")
        shutil.rmtree(tmpdir)
        print("Training complete!")


if __name__ == "__main__":
    main()
