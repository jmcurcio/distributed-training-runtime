#!/usr/bin/env python3
"""
Quick Start Example for Distributed Training Runtime

This minimal example shows the core workflow:
1. Create a runtime
2. Register a dataset with sharding
3. Iterate over shards
4. Save/load checkpoints

Prerequisites:
    cd rust/python-bindings && maturin develop
"""

import json
import os
import pickle
import tempfile
from pathlib import Path


def main():
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir()

        # Create sample JSONL dataset
        data_file = data_dir / "train.jsonl"
        with open(data_file, 'w') as f:
            for i in range(500):
                f.write(json.dumps({"id": i, "value": i * 0.1}) + '\n')

        print(f"Created dataset: {data_file} ({data_file.stat().st_size} bytes)")

        # --- Core DTR Usage ---

        # 1. Create runtime with config file
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(f'''
[storage]
base_path = "{data_dir}"

[checkpoint]
checkpoint_dir = "{checkpoint_dir}"
''')

        from dtr import Runtime
        runtime = Runtime(str(config_file))

        # 2. Register dataset with 4 shards
        dataset = runtime.register_dataset(
            path="train.jsonl",
            shards=4,
            format="newline"
        )
        print(f"Registered dataset: {dataset.num_shards} shards, {dataset.total_bytes} bytes")

        # 3. Iterate over each shard
        for shard_id in range(dataset.num_shards):
            record_count = 0

            for batch in dataset.iter_shard(shard_id, batch_size=4096):
                lines = batch.decode('utf-8').splitlines()
                for line in lines:
                    data = json.loads(line)
                    # Process record...
                    record_count += 1

            print(f"  Shard {shard_id}: {record_count} records")

        # 4. Save checkpoint
        model_state = {"weights": [1.0, 2.0, 3.0], "epoch": 1}
        state_bytes = pickle.dumps(model_state)
        checkpoint_path = runtime.save_checkpoint("model", state_bytes)
        print(f"Saved checkpoint: {checkpoint_path}")

        # 5. Load checkpoint
        loaded_bytes = runtime.load_checkpoint(checkpoint_path)
        loaded_state = pickle.loads(loaded_bytes)
        print(f"Loaded checkpoint: epoch={loaded_state['epoch']}")


if __name__ == "__main__":
    main()
