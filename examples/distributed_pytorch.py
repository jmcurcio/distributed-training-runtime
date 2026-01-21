#!/usr/bin/env python3
"""
Distributed Training with PyTorch DDP and DTR

This example shows how to use DTR for data loading in a distributed
PyTorch training setup. Each rank processes a different shard of the data.

Usage:
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 distributed_pytorch.py

    # Multi-node (run on each node)
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
             --master_addr=<master_ip> --master_port=29500 \
             distributed_pytorch.py

Prerequisites:
    pip install torch
    cd rust/python-bindings && maturin develop
"""

import os
import pickle
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
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
    # Initialize distributed environment
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Starting distributed training with {world_size} workers")

    # Initialize DTR runtime
    runtime = Runtime()

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
    checkpoint_dir = Path(runtime.checkpoint_dir)

    if rank == 0:
        existing = sorted(checkpoint_dir.glob("model_epoch_*.ckpt"))
        if existing:
            latest = str(existing[-1])
            print(f"Resuming from checkpoint: {latest}")
            start_epoch, _ = load_checkpoint(runtime, latest, model, optimizer)
            start_epoch += 1  # Start from next epoch

    # Broadcast start_epoch from rank 0 to all ranks
    start_tensor = torch.tensor([start_epoch], device=device)
    dist.broadcast(start_tensor, src=0)
    start_epoch = start_tensor.item()

    # Training loop
    num_epochs = 10
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
    dist.destroy_process_group()

    if rank == 0:
        print("\nTraining complete!")


if __name__ == "__main__":
    main()
