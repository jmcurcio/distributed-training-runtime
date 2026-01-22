#!/usr/bin/env python3
"""
Record Formats - DTR Phase 1 Examples

Demonstrates:
- Newline-delimited format (JSONL, CSV, text)
- Fixed-size format (embeddings, images)
- Length-prefixed format (protobuf, variable binary)

Prerequisites:
    cd rust/python-bindings && maturin develop
"""

import json
import struct
import tempfile
from pathlib import Path

from dtr import Runtime


def create_jsonl_file(path: Path, num_records: int = 50):
    """Create a JSONL (newline-delimited JSON) file."""
    with open(path, 'w') as f:
        for i in range(num_records):
            record = {"id": i, "text": f"Record {i}", "value": i * 0.5}
            f.write(json.dumps(record) + '\n')


def create_fixed_size_file(path: Path, num_records: int = 50, record_size: int = 64):
    """Create a file with fixed-size binary records."""
    with open(path, 'wb') as f:
        for i in range(num_records):
            # Create a fixed-size record: 4 bytes int + padding
            record = struct.pack(f'<I{record_size - 4}s', i, b'\x00' * (record_size - 4))
            f.write(record)


def create_length_prefixed_file(path: Path, num_records: int = 50):
    """Create a file with length-prefixed records."""
    with open(path, 'wb') as f:
        for i in range(num_records):
            # Variable-length data
            data = f"Record number {i} with variable length content".encode('utf-8')
            # Write 4-byte big-endian length prefix
            f.write(struct.pack('>I', len(data)))
            # Write data
            f.write(data)


def example_newline_format(runtime: Runtime, data_path: str):
    """
    Newline-delimited format for JSONL, CSV, and text files.

    This is the default format. Records are separated by newline characters.
    Each batch ends at a complete newline boundary.

    Use for:
    - JSONL files
    - CSV files
    - Plain text with one record per line
    """
    print(f"Reading newline-delimited file: {data_path}")

    dataset = runtime.register_dataset(
        path=data_path,
        shards=2,
        format="newline"  # Default, can be omitted
    )

    record_count = 0
    for batch in dataset.iter_shard(0, batch_size=1024):
        lines = batch.decode('utf-8').splitlines()
        for line in lines:
            if line.strip():
                record = json.loads(line)
                record_count += 1
                if record_count <= 3:  # Show first 3 records
                    print(f"  Record {record['id']}: {record['text']}")

    print(f"  Total records in shard 0: {record_count}")


def example_fixed_format(runtime: Runtime, data_path: str, record_size: int):
    """
    Fixed-size format for binary data with uniform record sizes.

    Use for:
    - Pre-computed embeddings (all same dimension)
    - Fixed-size image tensors
    - Tokenized sequences with padding
    """
    print(f"Reading fixed-size file: {data_path} (record_size={record_size})")

    dataset = runtime.register_dataset(
        path=data_path,
        shards=2,
        format=f"fixed:{record_size}"  # Each record is exactly N bytes
    )

    # Shard boundaries are aligned to record size
    for shard_id, (start, end) in enumerate(dataset.all_shard_info()):
        assert start % record_size == 0, "Shards align to record boundaries"
        print(f"  Shard {shard_id}: bytes {start} - {end}")

    # Read and parse records
    record_count = 0
    for batch in dataset.iter_shard(0, batch_size=record_size * 10):
        num_records = len(batch) // record_size
        for i in range(num_records):
            record_bytes = batch[i * record_size:(i + 1) * record_size]
            # Parse the fixed-size record
            record_id = struct.unpack('<I', record_bytes[:4])[0]
            record_count += 1
            if record_count <= 3:
                print(f"  Record ID: {record_id}")

    print(f"  Total records in shard 0: {record_count}")


def example_length_prefixed_format(runtime: Runtime, data_path: str):
    """
    Length-prefixed format for variable-length binary records.

    Format: [4-byte big-endian length][data bytes]

    Use for:
    - Serialized protocol buffers
    - Variable-length binary data
    - Mixed-size records
    """
    print(f"Reading length-prefixed file: {data_path}")

    dataset = runtime.register_dataset(
        path=data_path,
        shards=2,
        format="length-prefixed"
    )

    record_count = 0
    for batch in dataset.iter_shard(0, batch_size=4096):
        offset = 0
        while offset + 4 <= len(batch):
            # Read 4-byte big-endian length prefix
            length = struct.unpack('>I', batch[offset:offset + 4])[0]
            offset += 4

            if offset + length > len(batch):
                break  # Incomplete record at end of batch

            # Read record data
            record_data = batch[offset:offset + length]
            offset += length
            record_count += 1

            if record_count <= 3:
                print(f"  Record ({length} bytes): {record_data.decode('utf-8')[:50]}...")

    print(f"  Total records in shard 0: {record_count}")


def example_format_comparison():
    """
    Compare when to use each format.
    """
    print("Format Comparison:")
    print('''
  | Format          | Use Case                           | Pros                    |
  |-----------------|------------------------------------| ------------------------|
  | newline         | JSONL, CSV, text                   | Human-readable, simple  |
  | fixed:N         | Embeddings, fixed tensors          | Fast random access      |
  | length-prefixed | Protobuf, variable binary          | Efficient for binary    |

  Choosing a format:
  - Text data (JSON, CSV) -> "newline"
  - Fixed-size binary (embeddings) -> "fixed:N"
  - Variable binary (protobuf) -> "length-prefixed"
''')


def main():
    """Run all record format examples."""
    print("=" * 60)
    print("Record Formats - DTR Phase 1 Examples")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()

        # Create test files
        jsonl_file = data_dir / "data.jsonl"
        fixed_file = data_dir / "data.fixed"
        prefixed_file = data_dir / "data.prefixed"

        print("\nCreating test files...")
        create_jsonl_file(jsonl_file, 50)
        print(f"  Created: {jsonl_file.name}")

        record_size = 64
        create_fixed_size_file(fixed_file, 50, record_size)
        print(f"  Created: {fixed_file.name} (record_size={record_size})")

        create_length_prefixed_file(prefixed_file, 50)
        print(f"  Created: {prefixed_file.name}")

        # Create runtime
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(f'[storage]\nbase_path = "{data_dir}"\n')
        runtime = Runtime(str(config_file))

        print("\n--- Example 1: Newline Format (JSONL) ---")
        example_newline_format(runtime, "data.jsonl")

        print("\n--- Example 2: Fixed-Size Format ---")
        example_fixed_format(runtime, "data.fixed", record_size)

        print("\n--- Example 3: Length-Prefixed Format ---")
        example_length_prefixed_format(runtime, "data.prefixed")

        print("\n--- Example 4: Format Comparison ---")
        example_format_comparison()

    print("\n" + "=" * 60)
    print("All record format examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
