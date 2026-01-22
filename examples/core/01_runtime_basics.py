#!/usr/bin/env python3
"""
Runtime Basics - DTR Phase 1 Examples

Demonstrates:
- Runtime initialization with defaults
- Configuration via TOML file
- Configuration via environment variables
- Accessing runtime properties

Prerequisites:
    cd rust/python-bindings && maturin develop
"""

import os
import tempfile
from pathlib import Path

from dtr import Runtime


def example_default_runtime():
    """
    Initialize runtime with default settings.

    Defaults:
    - base_path: ./data
    - checkpoint_dir: ./checkpoints
    - compression: lz4
    """
    print("Creating runtime with defaults...")

    # Use a temp directory to avoid leaving artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal config pointing to temp directory
        config_path = Path(tmpdir) / "config.toml"
        config_path.write_text(f'''
[storage]
base_path = "{tmpdir}/data"

[checkpoint]
checkpoint_dir = "{tmpdir}/checkpoints"
''')
        runtime = Runtime(str(config_path))

        print(f"  Base path: {runtime.base_path}")
        print(f"  Checkpoint dir: {runtime.checkpoint_dir}")
        print(f"  Compression: {runtime.compression}")


def example_config_file():
    """
    Initialize runtime from a TOML configuration file.

    Configuration files allow you to specify all runtime settings
    in a versioned, shareable format.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a config file with absolute paths to avoid artifacts
        config_path = Path(tmpdir) / "config.toml"
        config_path.write_text(f'''
[storage]
base_path = "{tmpdir}/data"
buffer_size = 131072  # 128KB

[checkpoint]
checkpoint_dir = "{tmpdir}/checkpoints"
compression = "zstd"
compression_level = 3
keep_last_n = 5
atomic_writes = true
''')

        print(f"Creating runtime from config file...")
        print(f"  Config path: {config_path}")

        runtime = Runtime(str(config_path))

        print(f"  Compression from config: {runtime.compression}")


def example_environment_variables():
    """
    Override configuration with environment variables.

    Environment variables take precedence over config file settings.
    Use prefix DTR_ with section_key format (e.g., DTR_STORAGE_BASE_PATH).
    """
    print("Setting environment variable overrides...")

    # Set environment variables
    os.environ["DTR_STORAGE_BASE_PATH"] = "/tmp/dtr_env_test"
    os.environ["DTR_CHECKPOINT_COMPRESSION"] = "zstd"
    os.environ["DTR_CHECKPOINT_KEEP_LAST_N"] = "10"

    try:
        runtime = Runtime()

        print(f"  Base path (from env): {runtime.base_path}")
        print(f"  Compression (from env): {runtime.compression}")

    finally:
        # Clean up environment
        del os.environ["DTR_STORAGE_BASE_PATH"]
        del os.environ["DTR_CHECKPOINT_COMPRESSION"]
        del os.environ["DTR_CHECKPOINT_KEEP_LAST_N"]


def example_configuration_reference():
    """
    Print a reference of all configuration options.
    """
    print("Configuration Reference:")
    print('''
  # Full TOML configuration example:

  [storage]
  base_path = "./data"           # Base directory for datasets
  buffer_size = 65536            # I/O buffer size (64KB default)
  use_mmap = true                # Use memory mapping for large files
  mmap_threshold = 1048576       # Min file size for mmap (1MB)

  [dataset]
  default_shard_count = 1        # Default shards if not specified

  [checkpoint]
  checkpoint_dir = "./checkpoints"  # Directory for checkpoint files
  compression = "lz4"               # Options: "none", "lz4", "zstd"
  compression_level = 1             # 1-22 for zstd, ignored for lz4
  keep_last_n = 3                   # Keep only N most recent checkpoints
  atomic_writes = true              # Use atomic writes (rename pattern)

  # Environment variable overrides:
  DTR_STORAGE_BASE_PATH=/data/training
  DTR_STORAGE_BUFFER_SIZE=131072
  DTR_CHECKPOINT_COMPRESSION=zstd
  DTR_CHECKPOINT_KEEP_LAST_N=5
''')


def main():
    """Run all runtime basics examples."""
    print("=" * 60)
    print("Runtime Basics - DTR Phase 1 Examples")
    print("=" * 60)

    print("\n--- Example 1: Default Runtime ---")
    example_default_runtime()

    print("\n--- Example 2: Config File ---")
    example_config_file()

    print("\n--- Example 3: Environment Variables ---")
    example_environment_variables()

    print("\n--- Example 4: Configuration Reference ---")
    example_configuration_reference()

    print("\n" + "=" * 60)
    print("All runtime basics examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
