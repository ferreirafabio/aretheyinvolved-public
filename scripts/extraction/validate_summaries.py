#!/usr/bin/env python3
"""Find and optionally delete corrupted summary JSON files.

Walks a summary directory, tries json.load() on each *_summary.json,
and reports files that fail to parse. With --delete, removes them so
they get regenerated on the next SLURM run (pipeline is idempotent).

Usage:
    # List corrupted files
    python scripts/extraction/validate_summaries.py data/processed/summaries/doj/

    # Delete corrupted files
    python scripts/extraction/validate_summaries.py data/processed/summaries/doj/ --delete
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Find/delete corrupted summary JSONs")
    parser.add_argument("directory", type=Path, help="Summary directory to scan")
    parser.add_argument("--delete", action="store_true", help="Delete corrupted files")
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    summary_files = sorted(args.directory.glob("*_summary.json"))
    corrupted = []
    empty = []

    for path in summary_files:
        try:
            with open(path) as f:
                content = f.read()
            if not content.strip():
                empty.append(path)
                continue
            json.loads(content)
        except json.JSONDecodeError:
            corrupted.append(path)

    total = len(summary_files)
    n_bad = len(corrupted) + len(empty)

    print(f"Scanned {total} summary files")
    print(f"  Valid:     {total - n_bad}")
    print(f"  Corrupted: {len(corrupted)}")
    print(f"  Empty:     {len(empty)}")

    if not n_bad:
        print("All files valid.")
        return

    all_bad = corrupted + empty
    for path in all_bad:
        print(path.name)

    if args.delete:
        for path in all_bad:
            path.unlink()
        print(f"\nDeleted {n_bad} files. Resubmit SLURM jobs to regenerate.")
    else:
        print(f"\nRun with --delete to remove {n_bad} corrupted files.")


if __name__ == "__main__":
    main()
