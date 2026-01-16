"""
Distributed Training Runtime

High-performance data loading and checkpointing for ML training.

Example
-------
>>> from dtr import Runtime
>>> runtime = Runtime()
>>> dataset = runtime.register_dataset("data.jsonl", shards=4, format="newline")
>>> for batch in dataset.iter_shard(0):
...     process(batch)
"""

from dtr._dtr_core import Dataset, Runtime, ShardIterator

__version__ = "0.1.0"
__all__ = ["Runtime", "Dataset", "ShardIterator"]
