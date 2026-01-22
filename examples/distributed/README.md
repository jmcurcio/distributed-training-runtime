# Distributed Training Examples

Examples for using DTR in distributed training setups.

## Files

| File | Description |
|------|-------------|
| `pytorch_ddp.py` | PyTorch Distributed Data Parallel with DTR |
| `multiprocess.py` | Python multiprocessing for parallel data loading |

## PyTorch DDP Setup

### Single Node, Multiple GPUs

```bash
# Install dependencies
pip install torch

# Run with 4 GPUs
torchrun --nproc_per_node=4 pytorch_ddp.py
```

### Multi-Node Training

```bash
# On node 0 (master)
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
         --master_addr=<master_ip> --master_port=29500 \
         pytorch_ddp.py

# On node 1
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
         --master_addr=<master_ip> --master_port=29500 \
         pytorch_ddp.py
```

## Key Concepts

### Sharding for Distributed Training

Each worker processes a different shard of the data:

```python
# world_size = total number of workers
dataset = runtime.register_dataset("data.jsonl", shards=world_size)

# Each worker processes its own shard
my_shard = rank  # rank = worker ID (0 to world_size-1)
for batch in dataset.iter_shard(my_shard):
    process(batch)
```

### Checkpointing in Distributed Training

Only rank 0 saves checkpoints to avoid conflicts:

```python
if rank == 0:
    state = serialize(model.state_dict())
    runtime.save_checkpoint("model", state)

# All workers wait for checkpoint to be saved
dist.barrier()
```

### Data Parallel vs Model Parallel

DTR is designed for **data parallelism**:
- Each worker gets a different subset of data
- All workers have the full model
- Gradients are synchronized across workers
 
For model parallelism (splitting the model across GPUs), you'll need additional frameworks like DeepSpeed or Megatron-LM.
