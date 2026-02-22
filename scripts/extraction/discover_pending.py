#!/usr/bin/env python3
"""Fast pending file discovery for SLURM NER jobs (Tier 1 and Tier 2).

Replaces the bash skip-gate loop that was taking 17+ hours on large datasets.
Uses os.listdir() (single syscall per directory) and set operations instead of
per-file stat checks.

Prints assigned file paths to stdout (one per line) for consumption by bash.
Prints diagnostics to stderr.

Modes:
    tier1 (default): Input = text .json files, Output = _ner.json files
    tier2:           Input = _ner.json files, Output = _names.json files

Usage:
    # Tier 1: discover text files needing NER
    python scripts/extraction/discover_pending.py \
        --input-dir data/processed/text/doj \
        --output-dir data/processed/names_v2/doj \
        --worker-id 0 --num-workers 32

    # Tier 2: discover _ner.json files needing LLM processing
    python scripts/extraction/discover_pending.py \
        --mode tier2 \
        --input-dir data/processed/names_v2/doj \
        --output-dir data/processed/names_v2/doj \
        --worker-id 0 --num-workers 12
"""

import argparse
import json
import os
import sys
import time


# Suffixes that mark output/intermediate files (not input text files)
OUTPUT_SUFFIXES = (
    "_names.json",
    "_names_llm.json",
    "_ner.json",
    "_clean.json",
    "_names_clean.json",
    "_summary.json",
)


def discover_pending(input_dir: str, output_dir: str,
                     worker_id: int, num_workers: int,
                     mode: str = "tier1") -> list[str]:
    """Discover pending files and return this worker's share.

    Args:
        input_dir: Directory containing input files.
        output_dir: Directory containing output files.
        worker_id: This worker's index (0-based).
        num_workers: Total number of workers.
        mode: "tier1" (text→ner) or "tier2" (ner→names).

    Returns:
        List of full file paths assigned to this worker.
    """
    # Collect input file stems (single listdir call)
    try:
        all_entries = os.listdir(input_dir)
    except FileNotFoundError:
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    input_stems = set()

    if mode == "tier2":
        # Tier 2: input = _ner.json files
        input_suffix = "_ner.json"
        input_suffix_len = len(input_suffix)
        output_suffix = "_names.json"
        output_suffix_len = len(output_suffix)
        file_ext = input_suffix  # for building output paths

        for entry in all_entries:
            if entry.endswith(input_suffix) and not entry.endswith(".tmp.json"):
                stem = entry[:-input_suffix_len]
                input_stems.add(stem)
    else:
        # Tier 1: input = text .json files (default)
        output_suffix = "_ner.json"
        output_suffix_len = len(output_suffix)
        file_ext = ".json"

        for entry in all_entries:
            if not entry.endswith(".json"):
                continue
            # Skip output/intermediate files that may be co-located
            if any(entry.endswith(suffix) for suffix in OUTPUT_SUFFIXES):
                continue
            stem = entry[:-5]  # strip .json
            input_stems.add(stem)

    if not input_stems:
        label = "_ner.json" if mode == "tier2" else ".json"
        print(f"ERROR: No input {label} files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect existing output stems (single listdir call)
    output_stems = set()
    if os.path.isdir(output_dir):
        try:
            for entry in os.listdir(output_dir):
                if entry.endswith(output_suffix) and not entry.endswith(".tmp.json"):
                    stem = entry[:-output_suffix_len]
                    output_stems.add(stem)
        except FileNotFoundError:
            pass  # Directory was removed between check and listdir

    # Set difference: pending files
    pending_stems = sorted(input_stems - output_stems)

    # Diagnostics to stderr
    print(f"Mode: {mode}", file=sys.stderr)
    print(f"Total input files: {len(input_stems)}", file=sys.stderr)
    print(f"Total existing outputs: {len(output_stems)}", file=sys.stderr)
    print(f"Pending: {len(pending_stems)}", file=sys.stderr)

    # Sanity check: if pending is 0 but output count is suspiciously low
    if not pending_stems:
        if len(output_stems) < len(input_stems) // 2:
            print(
                f"WARNING: 0 pending but only {len(output_stems)} outputs "
                f"for {len(input_stems)} inputs. Possible stem mismatch?",
                file=sys.stderr,
            )
        print(f"Assigned to worker {worker_id}: 0 files", file=sys.stderr)
        return []

    # Block-assign to this worker
    files_per_worker = (len(pending_stems) + num_workers - 1) // num_workers
    start_idx = worker_id * files_per_worker
    end_idx = min(start_idx + files_per_worker, len(pending_stems))

    if start_idx >= len(pending_stems):
        assigned = []
    else:
        assigned = pending_stems[start_idx:end_idx]

    print(f"Assigned to worker {worker_id}: {len(assigned)} files", file=sys.stderr)

    if assigned:
        preview = assigned[:3]
        print(f"First assigned: {', '.join(s + file_ext for s in preview)}", file=sys.stderr)

    if not assigned and pending_stems:
        print(
            f"WARNING: Worker {worker_id} got 0 files but {len(pending_stems)} "
            f"pending. Consider fewer workers.",
            file=sys.stderr,
        )

    # Return full paths
    return [os.path.join(input_dir, stem + file_ext) for stem in assigned]


def build_manifest(ner_dir: str) -> str:
    """Build _ner_manifest.jsonl from existing _ner.json files.

    Reads each _ner.json and extracts skip-gate fields into a compact
    JSONL manifest. Used to bootstrap manifests for directories that
    were created before manifest support was added.

    Returns:
        Path to the manifest file.
    """
    manifest_path = os.path.join(ner_dir, "_ner_manifest.jsonl")
    entries = os.listdir(ner_dir)
    ner_files = [e for e in entries if e.endswith("_ner.json") and not e.endswith(".tmp.json")]
    ner_files.sort()

    print(f"Building manifest from {len(ner_files)} _ner.json files...", file=sys.stderr)
    t0 = time.monotonic()
    count = 0
    errors = 0

    # Write to temp file then rename (atomic)
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as mf:
        for i, fname in enumerate(ner_files):
            if (i + 1) % 100000 == 0:
                print(f"  ...scanned {i + 1}/{len(ner_files)}", file=sys.stderr)
            stem = fname[:-len("_ner.json")]
            try:
                with open(os.path.join(ner_dir, fname), 'r') as f:
                    data = json.load(f)
                entry = json.dumps({
                    "stem": stem,
                    "person_spans": data.get("person_spans", 0),
                    "total_spans": data.get("total_spans", 0),
                    "clean_text_length": data.get("clean_text_length", 0),
                    "noise_score": round(data.get("noise_score", 0.0), 4),
                }, ensure_ascii=False)
                mf.write(entry + '\n')
                count += 1
            except (json.JSONDecodeError, OSError) as e:
                errors += 1
                if errors <= 5:
                    print(f"  WARN: {fname}: {e}", file=sys.stderr)

    os.rename(tmp_path, manifest_path)
    elapsed = time.monotonic() - t0
    print(
        f"Manifest built: {count} entries, {errors} errors, {elapsed:.1f}s → {manifest_path}",
        file=sys.stderr,
    )
    return manifest_path


def read_manifest(ner_dir: str) -> dict:
    """Read _ner_manifest.jsonl and return {stem: {person_spans, ...}}.

    Last-one-wins for duplicate stems (handles reruns).
    Returns empty dict if manifest doesn't exist.
    """
    manifest_path = os.path.join(ner_dir, "_ner_manifest.jsonl")
    if not os.path.isfile(manifest_path):
        return {}

    result = {}
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                result[entry["stem"]] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    return result


def query_manifest(ner_dir: str, min_spans: int = 0, max_spans: int = 999999) -> list:
    """Query manifest for files with person_spans in [min_spans, max_spans].

    Returns list of (stem, person_spans, clean_text_length, noise_score).
    Falls back to building manifest if it doesn't exist.
    """
    manifest = read_manifest(ner_dir)
    if not manifest:
        print("No manifest found, building...", file=sys.stderr)
        build_manifest(ner_dir)
        manifest = read_manifest(ner_dir)
        if not manifest:
            return []

    return [
        (e["stem"], e["person_spans"], e.get("clean_text_length", 0), e.get("noise_score", 0.0))
        for e in manifest.values()
        if min_spans <= e.get("person_spans", 0) <= max_spans
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Fast pending file discovery for SLURM NER jobs"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default: discover pending files (backwards compat via args)
    parser.add_argument("--input-dir",
                        help="Input directory (text JSONs for tier1, _ner.json for tier2)")
    parser.add_argument("--output-dir",
                        help="Output directory (_ner.json for tier1, _names.json for tier2)")
    parser.add_argument("--worker-id", type=int, help="Worker index (0-based)")
    parser.add_argument("--num-workers", type=int, help="Total workers")
    parser.add_argument("--mode", choices=["tier1", "tier2"], default="tier1",
                        help="Discovery mode: tier1 (text→ner) or tier2 (ner→names)")

    # build-manifest subcommand
    build_parser = subparsers.add_parser("build-manifest",
                                          help="Build _ner_manifest.jsonl from existing _ner.json files")
    build_parser.add_argument("ner_dir", help="Directory containing _ner.json files")

    # query subcommand
    query_parser = subparsers.add_parser("query",
                                          help="Query manifest for files matching span criteria")
    query_parser.add_argument("ner_dir", help="Directory containing _ner.json files")
    query_parser.add_argument("--min-spans", type=int, default=0, help="Minimum person_spans")
    query_parser.add_argument("--max-spans", type=int, default=999999, help="Maximum person_spans")
    query_parser.add_argument("--limit", type=int, default=0, help="Max results (0=all)")
    query_parser.add_argument("--sort", choices=["spans", "length"], default="spans",
                              help="Sort by person_spans or clean_text_length")

    args = parser.parse_args()

    if args.command == "build-manifest":
        build_manifest(args.ner_dir)
        return

    if args.command == "query":
        results = query_manifest(args.ner_dir, args.min_spans, args.max_spans)
        if args.sort == "spans":
            results.sort(key=lambda x: -x[1])
        else:
            results.sort(key=lambda x: -x[2])
        if args.limit > 0:
            results = results[:args.limit]
        print(f"Found {len(results)} files with person_spans in [{args.min_spans}, {args.max_spans}]",
              file=sys.stderr)
        for stem, spans, length, noise in results:
            print(f"{stem}\t{spans}\t{length}\t{noise:.4f}")
        return

    # Default: discover pending (original behavior)
    if not args.input_dir or not args.output_dir or args.worker_id is None or args.num_workers is None:
        parser.error("discover mode requires --input-dir, --output-dir, --worker-id, --num-workers")

    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        print(
            f"ERROR: worker-id {args.worker_id} out of range [0, {args.num_workers})",
            file=sys.stderr,
        )
        sys.exit(1)

    files = discover_pending(
        args.input_dir, args.output_dir,
        args.worker_id, args.num_workers,
        mode=args.mode,
    )

    # Print file paths to stdout (one per line)
    for f in files:
        print(f)


if __name__ == "__main__":
    main()
