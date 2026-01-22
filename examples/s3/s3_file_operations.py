#!/usr/bin/env python3
"""
S3 File Operations Example - Demonstrating DTR S3 File Management

This example demonstrates the file operations available with the S3 backend:
- upload_file: Upload local files to S3
- download_file: Download S3 objects to local files
- list_files: List objects in an S3 prefix
- file_exists: Check if an object exists in S3

Prerequisites:
    cd rust/python-bindings && maturin develop --features s3

Environment Variables (set in .env):
    DTR_S3_BUCKET         - S3 bucket name (required)
    DTR_S3_REGION         - AWS region (default: us-east-1)
    DTR_S3_BASE_PATH      - S3 key prefix for data
    AWS_ACCESS_KEY_ID     - AWS access key
    AWS_SECRET_ACCESS_KEY - AWS secret key
"""

import json
import os
import tempfile
from pathlib import Path

from dtr import Runtime


# ============================================================================
# Configuration Loading
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


load_dotenv()

# Configuration from environment
S3_CONFIG = {
    'bucket': os.environ.get('DTR_S3_BUCKET', ''),
    'region': os.environ.get('DTR_S3_REGION', 'us-east-1'),
    'base_path': os.environ.get('DTR_S3_BASE_PATH', 'dtr-training/'),
    'endpoint': os.environ.get('DTR_S3_ENDPOINT'),
}

S3_CREDENTIALS = {}
if os.environ.get('AWS_ACCESS_KEY_ID'):
    S3_CREDENTIALS['access_key'] = os.environ['AWS_ACCESS_KEY_ID']
if os.environ.get('AWS_SECRET_ACCESS_KEY'):
    S3_CREDENTIALS['secret_key'] = os.environ['AWS_SECRET_ACCESS_KEY']


# ============================================================================
# Runtime Creation
# ============================================================================

def create_s3_runtime():
    """Create runtime configured for S3 storage."""
    if not S3_CONFIG['bucket']:
        raise ValueError("DTR_S3_BUCKET not configured. Check your .env file.")

    kwargs = {
        'backend': 's3',
        'base_path': '.',
        's3_bucket': S3_CONFIG['bucket'],
        's3_region': S3_CONFIG['region'],
    }

    if 'access_key' in S3_CREDENTIALS:
        kwargs['s3_access_key'] = S3_CREDENTIALS['access_key']
        kwargs['s3_secret_key'] = S3_CREDENTIALS['secret_key']

    if S3_CONFIG.get('endpoint'):
        kwargs['s3_endpoint'] = S3_CONFIG['endpoint']

    return Runtime(**kwargs)


# ============================================================================
# Example Operations
# ============================================================================

def demo_upload_download(runtime):
    """Demonstrate file upload and download."""
    print("\n" + "=" * 60)
    print("Demo: Upload and Download")
    print("=" * 60)

    # Create a temporary file with sample data
    sample_data = {
        "name": "test_model",
        "version": "1.0.0",
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
        },
        "metrics": {
            "accuracy": 0.95,
            "loss": 0.05,
        }
    }

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_data, f, indent=2)
        local_path = f.name

    s3_path = f"{S3_CONFIG['base_path']}file_ops_demo/config.json"

    try:
        # Upload
        print(f"\n1. Uploading file to S3...")
        print(f"   Local: {local_path}")
        print(f"   S3:    s3://{S3_CONFIG['bucket']}/{s3_path}")
        runtime.upload_file(local_path, s3_path)
        print("   Upload complete!")

        # Verify exists
        print(f"\n2. Verifying file exists in S3...")
        exists = runtime.file_exists(s3_path)
        print(f"   file_exists('{s3_path}'): {exists}")

        # Download to different location
        download_path = local_path + ".downloaded"
        print(f"\n3. Downloading file from S3...")
        print(f"   S3:    s3://{S3_CONFIG['bucket']}/{s3_path}")
        print(f"   Local: {download_path}")
        runtime.download_file(s3_path, download_path)
        print("   Download complete!")

        # Verify contents match
        print(f"\n4. Verifying downloaded contents...")
        with open(download_path) as f:
            downloaded_data = json.load(f)

        if downloaded_data == sample_data:
            print("   Contents match original!")
        else:
            print("   WARNING: Contents do not match!")

        # Cleanup local files
        os.unlink(local_path)
        os.unlink(download_path)

    except Exception as e:
        print(f"\n   Error: {e}")
        # Cleanup on error
        if os.path.exists(local_path):
            os.unlink(local_path)


def demo_list_files(runtime):
    """Demonstrate listing files in S3."""
    print("\n" + "=" * 60)
    print("Demo: List Files")
    print("=" * 60)

    # Upload a few test files
    test_prefix = f"{S3_CONFIG['base_path']}file_ops_demo/list_test/"

    print(f"\n1. Creating test files in S3...")
    print(f"   Prefix: s3://{S3_CONFIG['bucket']}/{test_prefix}")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test content")
        temp_file = f.name

    test_files = ["file_a.txt", "file_b.txt", "file_c.txt", "subdir/nested.txt"]

    try:
        for filename in test_files:
            s3_path = test_prefix + filename
            runtime.upload_file(temp_file, s3_path)
            print(f"   Created: {filename}")

        # List files
        print(f"\n2. Listing files at prefix...")
        files = runtime.list_files(test_prefix)
        print(f"   Found {len(files)} files:")
        for f in sorted(files):
            print(f"      {f}")

        # List subdirectory
        print(f"\n3. Listing subdirectory...")
        subdir_files = runtime.list_files(test_prefix + "subdir/")
        print(f"   Found {len(subdir_files)} files in subdir:")
        for f in sorted(subdir_files):
            print(f"      {f}")

    except Exception as e:
        print(f"\n   Error: {e}")
    finally:
        os.unlink(temp_file)


def demo_file_exists(runtime):
    """Demonstrate checking file existence."""
    print("\n" + "=" * 60)
    print("Demo: File Exists")
    print("=" * 60)

    test_paths = [
        f"{S3_CONFIG['base_path']}file_ops_demo/config.json",
        f"{S3_CONFIG['base_path']}nonexistent/file.txt",
        f"{S3_CONFIG['base_path']}train_data.jsonl",
    ]

    print("\nChecking file existence:")
    for path in test_paths:
        try:
            exists = runtime.file_exists(path)
            status = "EXISTS" if exists else "NOT FOUND"
            print(f"   [{status:9}] {path}")
        except Exception as e:
            print(f"   [ERROR    ] {path}: {e}")


def demo_bulk_operations(runtime):
    """Demonstrate bulk upload/download patterns."""
    print("\n" + "=" * 60)
    print("Demo: Bulk Operations Pattern")
    print("=" * 60)

    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = Path(tmpdir)
        s3_prefix = f"{S3_CONFIG['base_path']}file_ops_demo/bulk/"

        # Create local files
        print("\n1. Creating local files for bulk upload...")
        files_to_upload = []
        for i in range(5):
            filepath = local_dir / f"batch_{i}.json"
            data = {"batch_id": i, "data": list(range(i * 10, (i + 1) * 10))}
            with open(filepath, 'w') as f:
                json.dump(data, f)
            files_to_upload.append(filepath)
            print(f"   Created: {filepath.name}")

        # Bulk upload
        print(f"\n2. Uploading {len(files_to_upload)} files to S3...")
        for filepath in files_to_upload:
            s3_path = s3_prefix + filepath.name
            runtime.upload_file(str(filepath), s3_path)
        print("   Bulk upload complete!")

        # List uploaded files
        print("\n3. Verifying uploaded files...")
        uploaded = runtime.list_files(s3_prefix)
        print(f"   Found {len(uploaded)} files in S3:")
        for f in sorted(uploaded):
            print(f"      {f}")

        # Bulk download to new directory
        download_dir = local_dir / "downloaded"
        download_dir.mkdir()

        print(f"\n4. Bulk downloading files...")
        for s3_file in uploaded:
            filename = s3_file.split('/')[-1]
            local_path = download_dir / filename
            # list_files returns just filenames, so prepend the prefix
            full_s3_path = s3_prefix + s3_file
            runtime.download_file(full_s3_path, str(local_path))
        print(f"   Downloaded {len(uploaded)} files to {download_dir}")

        # Verify downloads
        print("\n5. Downloaded files:")
        for filepath in sorted(download_dir.iterdir()):
            size = filepath.stat().st_size
            print(f"      {filepath.name} ({size} bytes)")


def show_summary(runtime):
    """Show summary of all files in the demo prefix."""
    print("\n" + "=" * 60)
    print("Summary: All Demo Files in S3")
    print("=" * 60)

    demo_prefix = f"{S3_CONFIG['base_path']}file_ops_demo/"

    try:
        all_files = runtime.list_files(demo_prefix)
        print(f"\nFiles at s3://{S3_CONFIG['bucket']}/{demo_prefix}")
        print(f"Total: {len(all_files)} files\n")

        for f in sorted(all_files):
            # Show relative path from demo prefix
            rel_path = f[len(demo_prefix):] if f.startswith(demo_prefix) else f
            print(f"   {rel_path}")

    except Exception as e:
        print(f"\nError listing files: {e}")

    print("\nTo clean up demo files:")
    print(f"   aws s3 rm --recursive s3://{S3_CONFIG['bucket']}/{demo_prefix}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run file operations demonstrations."""
    print("=" * 60)
    print("S3 File Operations Example")
    print("=" * 60)

    # Check configuration
    if not S3_CONFIG['bucket']:
        print("\nError: S3 not configured.")
        print("Set DTR_S3_BUCKET in examples/.env")
        return

    print(f"\nS3 Bucket: {S3_CONFIG['bucket']}")
    print(f"S3 Region: {S3_CONFIG['region']}")
    print(f"Base Path: {S3_CONFIG['base_path']}")

    # Create runtime
    print("\nConnecting to S3...")
    try:
        runtime = create_s3_runtime()
    except Exception as e:
        print(f"\nError: {e}")
        return

    print("   Connected!")

    # Run demonstrations
    demo_upload_download(runtime)
    demo_file_exists(runtime)
    demo_list_files(runtime)
    demo_bulk_operations(runtime)
    show_summary(runtime)

    print("\n" + "=" * 60)
    print("File operations demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
