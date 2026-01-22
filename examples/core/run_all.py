#!/usr/bin/env python3
"""
Run All Core Examples

This script runs all core (single-node) examples in sequence.

Prerequisites:
    cd rust/python-bindings && maturin develop
"""

import subprocess
import sys
from pathlib import Path


def run_example(script_path: Path) -> bool:
    """Run an example script and return True if successful."""
    print(f"\n{'=' * 70}")
    print(f"Running: {script_path.name}")
    print('=' * 70)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=script_path.parent
    )

    return result.returncode == 0


def main():
    """Run all core examples."""
    print("=" * 70)
    print("Core Examples - Running All")
    print("=" * 70)

    examples_dir = Path(__file__).parent
    examples = [
        "01_runtime_basics.py",
        "02_dataset_iteration.py",
        "03_record_formats.py",
        "04_checkpointing.py",
        "05_progress_errors.py",
        "06_prefetching.py",
    ]

    results = []
    for example in examples:
        script_path = examples_dir / example
        if script_path.exists():
            success = run_example(script_path)
            results.append((example, success))
        else:
            print(f"Warning: {example} not found")
            results.append((example, False))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for example, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {example}: {status}")

    print(f"\nTotal: {passed}/{total} examples passed")

    if passed == total:
        print("\nAll core examples completed successfully!")
        return 0
    else:
        print("\nSome examples failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
