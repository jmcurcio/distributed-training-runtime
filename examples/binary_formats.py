#!/usr/bin/env python3
"""
Binary Data Formats with DTR

This example demonstrates working with binary data formats:
- Fixed-size records (embeddings, images, pre-processed tensors)
- Length-prefixed records (protobuf, variable-length data)

These formats are efficient for pre-processed ML data where records
don't need text parsing.

Prerequisites:
    cd rust/python-bindings && maturin develop
"""

import struct
import tempfile
from pathlib import Path
import numpy as np

from dtr import Runtime


def create_fixed_size_dataset(path: Path, num_records: int, record_size: int):
    """
    Create a dataset with fixed-size binary records.

    Each record is a numpy array serialized to bytes.
    This format is ideal for:
    - Pre-computed embeddings
    - Fixed-size image tensors
    - Tokenized sequences with padding
    """
    # Calculate embedding dimension from record size
    # Each float32 is 4 bytes
    embedding_dim = record_size // 4

    with open(path, 'wb') as f:
        for _ in range(num_records):
            # Create a fake embedding (in practice, these would be pre-computed)
            embedding = np.random.randn(embedding_dim).astype(np.float32)
            f.write(embedding.tobytes())

    print(f"Created fixed-size dataset: {path}")
    print(f"  Records: {num_records}")
    print(f"  Record size: {record_size} bytes ({embedding_dim} float32s)")
    print(f"  Total size: {path.stat().st_size:,} bytes")


def create_length_prefixed_dataset(path: Path, num_records: int):
    """
    Create a dataset with length-prefixed binary records.

    Format: [4-byte big-endian length][data bytes]

    This format is ideal for:
    - Variable-length sequences
    - Protocol buffers
    - Custom binary formats
    """
    with open(path, 'wb') as f:
        for i in range(num_records):
            # Variable-length data (simulating sequences of different lengths)
            seq_len = 50 + (i % 200)  # 50-249 floats
            data = np.random.randn(seq_len).astype(np.float32)
            data_bytes = data.tobytes()

            # Write 4-byte big-endian length prefix
            f.write(struct.pack('>I', len(data_bytes)))
            # Write data
            f.write(data_bytes)

    print(f"Created length-prefixed dataset: {path}")
    print(f"  Records: {num_records}")
    print(f"  Total size: {path.stat().st_size:,} bytes")


def example_fixed_size_records():
    """Iterate over fixed-size binary records."""
    print("\n" + "=" * 60)
    print("Fixed-Size Record Format")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()

        # Configuration
        record_size = 512  # 512 bytes = 128 float32s
        num_records = 10000
        embedding_dim = record_size // 4

        # Create dataset
        data_file = data_dir / "embeddings.bin"
        create_fixed_size_dataset(data_file, num_records, record_size)

        # Initialize runtime with config
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(f'[storage]\nbase_path = "{data_dir}"')
        runtime = Runtime(str(config_file))

        # Register with fixed format
        dataset = runtime.register_dataset(
            path="embeddings.bin",
            shards=4,
            format=f"fixed:{record_size}"
        )

        print(f"\nDataset info:")
        print(f"  Total bytes: {dataset.total_bytes:,}")
        print(f"  Num shards: {dataset.num_shards}")

        # Verify shard alignment
        print(f"\nShard boundaries (all aligned to {record_size}-byte records):")
        for shard_id, (start, end) in enumerate(dataset.all_shard_info()):
            records_in_shard = (end - start) // record_size
            assert start % record_size == 0, "Shards should be aligned"
            print(f"  Shard {shard_id}: bytes {start:>8} - {end:>8} ({records_in_shard} records)")

        # Iterate and process
        print(f"\nProcessing records:")
        total_records = 0
        total_sum = 0.0

        for shard_id in range(dataset.num_shards):
            shard_records = 0

            for batch in dataset.iter_shard(shard_id, batch_size=64 * 1024):
                # Each batch contains complete records only
                num_records_in_batch = len(batch) // record_size

                for i in range(num_records_in_batch):
                    # Extract one record
                    start = i * record_size
                    end = start + record_size
                    record_bytes = batch[start:end]

                    # Convert to numpy array
                    embedding = np.frombuffer(record_bytes, dtype=np.float32)
                    assert len(embedding) == embedding_dim

                    # Process (here we just compute sum)
                    total_sum += embedding.sum()
                    shard_records += 1

            total_records += shard_records
            print(f"  Shard {shard_id}: processed {shard_records} records")

        print(f"\nTotal records processed: {total_records}")
        print(f"Sum of all embeddings: {total_sum:.2f}")


def example_length_prefixed_records():
    """Iterate over length-prefixed binary records."""
    print("\n" + "=" * 60)
    print("Length-Prefixed Record Format")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()

        num_records = 5000

        # Create dataset
        data_file = data_dir / "sequences.bin"
        create_length_prefixed_dataset(data_file, num_records)

        # Initialize runtime with config
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(f'[storage]\nbase_path = "{data_dir}"')
        runtime = Runtime(str(config_file))

        # Register with length-prefixed format
        dataset = runtime.register_dataset(
            path="sequences.bin",
            shards=4,
            format="length-prefixed"
        )

        print(f"\nDataset info:")
        print(f"  Total bytes: {dataset.total_bytes:,}")
        print(f"  Num shards: {dataset.num_shards}")

        # Iterate and process
        print(f"\nProcessing variable-length records:")
        total_records = 0
        lengths = []

        for shard_id in range(dataset.num_shards):
            shard_records = 0

            for batch in dataset.iter_shard(shard_id, batch_size=64 * 1024):
                # Parse length-prefixed records from batch
                offset = 0

                while offset + 4 <= len(batch):
                    # Read 4-byte big-endian length
                    length = struct.unpack('>I', batch[offset:offset+4])[0]
                    offset += 4

                    if offset + length > len(batch):
                        # Incomplete record (shouldn't happen with proper batching)
                        break

                    # Extract record data
                    record_bytes = batch[offset:offset+length]
                    offset += length

                    # Convert to numpy array
                    sequence = np.frombuffer(record_bytes, dtype=np.float32)
                    lengths.append(len(sequence))

                    # Process sequence...
                    shard_records += 1

            total_records += shard_records
            print(f"  Shard {shard_id}: processed {shard_records} records")

        print(f"\nTotal records processed: {total_records}")
        print(f"Sequence length stats:")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")
        print(f"  Mean: {sum(lengths) / len(lengths):.1f}")


def example_image_dataset():
    """
    Simulated image dataset with fixed-size records.

    In practice, you would pre-process images to fixed-size tensors
    (e.g., 224x224x3 = 150528 bytes for uint8, or 602112 for float32).
    """
    print("\n" + "=" * 60)
    print("Fixed-Size Image Tensors")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()

        # Image configuration (small for demo)
        height, width, channels = 64, 64, 3
        record_size = height * width * channels  # uint8 pixels

        # Create dataset
        data_file = data_dir / "images.bin"
        num_images = 1000

        with open(data_file, 'wb') as f:
            for i in range(num_images):
                # Create random image tensor
                image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
                f.write(image.tobytes())

        print(f"Created image dataset:")
        print(f"  Image size: {width}x{height}x{channels}")
        print(f"  Record size: {record_size:,} bytes")
        print(f"  Num images: {num_images}")

        # Initialize runtime with config
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(f'[storage]\nbase_path = "{data_dir}"')
        runtime = Runtime(str(config_file))

        # Register dataset
        dataset = runtime.register_dataset(
            path="images.bin",
            shards=4,
            format=f"fixed:{record_size}"
        )

        # Process images
        print(f"\nProcessing images:")
        total_images = 0
        mean_pixel_values = []

        for shard_id in range(dataset.num_shards):
            for batch in dataset.iter_shard(shard_id, batch_size=record_size * 16):  # 16 images per batch
                num_images_in_batch = len(batch) // record_size

                for i in range(num_images_in_batch):
                    start = i * record_size
                    end = start + record_size

                    # Reconstruct image tensor
                    image = np.frombuffer(batch[start:end], dtype=np.uint8)
                    image = image.reshape(height, width, channels)

                    # Compute mean pixel value
                    mean_pixel_values.append(image.mean())
                    total_images += 1

        print(f"  Processed {total_images} images")
        print(f"  Mean pixel value across dataset: {sum(mean_pixel_values) / len(mean_pixel_values):.2f}")


def main():
    print("=" * 70)
    print("DTR Binary Format Examples")
    print("=" * 70)

    example_fixed_size_records()
    example_length_prefixed_records()
    example_image_dataset()

    print("\n" + "=" * 70)
    print("All examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
