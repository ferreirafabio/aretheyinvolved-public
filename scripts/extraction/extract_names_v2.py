#!/usr/bin/env python3
"""Extract names using NER + LLM pipeline v2.

This script processes extracted text files using the new 5-stage pipeline:
1. Deterministic cleaning (offset-preserving)
2. XLM-R NER (high recall)
3. LLM classification (filter/role assignment)
4. Hard validation (verify spans in source)
5. LLM repair (fix OCR errors)

Usage:
    # Process all files in directory
    python scripts/extract_names_v2.py --input-dir data/processed/text/doj/

    # Process specific files
    python scripts/extract_names_v2.py --files file1.json file2.json

    # Use mock extractors (for testing without GPU)
    python scripts/extract_names_v2.py --mock

    # Save intermediate stage outputs
    python scripts/extract_names_v2.py --save-intermediate
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ner.pipeline import (
    ExtractionPipeline,
    NerOnlyResult,
    PipelineConfig,
    PipelineResult,
    TIER2_SKIP_ESCAPE_HATCH_LENGTH,
    _get_pipeline_version,
    save_ner_only_result,
    save_pipeline_result,
)
from src.ner.llm_recovery import MIN_NOISE_SCORE


def validate_output_file(output_path: Path) -> bool:
    """Check if output file is valid and complete.

    Args:
        output_path: Path to the output JSON file.

    Returns:
        True if the file is valid and complete, False otherwise.
    """
    if not output_path.exists():
        return False

    # Ignore temp files
    if ".tmp." in output_path.name:
        return False

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check required fields
        required_fields = ["source_file", "names", "total_names"]
        if not all(field in data for field in required_fields):
            logger.warning(f"Output file missing required fields: {output_path}")
            return False

        # Validate count matches
        if data["total_names"] != len(data.get("names", [])):
            logger.warning(f"Output file count mismatch: {output_path}")
            return False

        return True

    except json.JSONDecodeError as e:
        logger.warning(f"Output file has invalid JSON: {output_path} ({e})")
        return False
    except Exception as e:
        logger.warning(f"Failed to validate output file: {output_path} ({e})")
        return False


def load_ocr_file(file_path: Path) -> dict[str, Any] | None:
    """Load extracted text JSON file.

    Args:
        file_path: Path to JSON file from extract_text.py.

    Returns:
        Parsed JSON data or None if failed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def process_file(
    pipeline: ExtractionPipeline,
    file_path: Path,
    output_dir: Path,
    force: bool = False,
    save_intermediate: bool = False
) -> PipelineResult | None:
    """Process a single text file through the pipeline.

    Args:
        pipeline: Extraction pipeline instance.
        file_path: Path to input JSON file.
        output_dir: Directory to save results.
        force: Overwrite existing output.
        save_intermediate: Save intermediate stage outputs.

    Returns:
        Pipeline result or None if skipped/failed.
    """
    # Check output path
    output_path = output_dir / f"{file_path.stem}_names.json"
    if not force and validate_output_file(output_path):
        logger.debug(f"Skipping (valid output exists): {file_path.name}")
        return None

    # Load OCR data
    ocr_data = load_ocr_file(file_path)
    if not ocr_data:
        return None

    source_file = Path(ocr_data.get("file_path", "")).name or file_path.stem

    logger.info(f"Processing: {file_path.name}")

    # Run pipeline
    result = pipeline.process_document(ocr_data, source_file=source_file)

    # Save results (atomic write handled inside save_pipeline_result)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        save_pipeline_result(
            result,
            output_dir,
            save_intermediate=save_intermediate
        )

        logger.info(
            f"Extracted {result.stats.final_names} names "
            f"(NER: {result.stats.stage1_ner_spans}, "
            f"validated: {result.stats.stage3_validated}, "
            f"repaired: {result.stats.stage4_repaired})"
        )

    except Exception as e:
        logger.error(f"Failed to save results for {file_path}: {e}")
        raise

    return result


def _prefetch_file(file_path: Path, output_dir: Path, force: bool) -> tuple[Path, dict[str, Any] | None, bool]:
    """Load file and check if output already exists.

    Args:
        file_path: Path to input file.
        output_dir: Output directory.
        force: Whether to force reprocessing.

    Returns:
        Tuple of (file_path, ocr_data, should_process).
    """
    # Check if we should skip this file
    output_path = output_dir / f"{file_path.stem}_names.json"
    if not force and validate_output_file(output_path):
        return (file_path, None, False)

    # Load the file
    ocr_data = load_ocr_file(file_path)
    return (file_path, ocr_data, True)


def validate_ner_output_file(output_path: Path) -> bool:
    """Check if a _ner.json output file is valid."""
    if not output_path.exists():
        return False
    # Skip temp files from interrupted atomic writes
    if ".tmp." in output_path.name:
        return False
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Accept both old format (with clean_text) and new format (with clean_text_length)
        has_required = "source_file" in data and "spans" in data
        has_text_info = "clean_text" in data or "clean_text_length" in data
        return has_required and has_text_info
    except (json.JSONDecodeError, Exception):
        return False


def process_directory_ner_only(
    pipeline: ExtractionPipeline,
    input_dir: Path,
    output_dir: Path,
    force: bool = False,
    file_list: list[Path] | None = None,
) -> dict[str, int]:
    """Process files through Stage 0 + Stage 1 only (Tier 1).

    Produces _ner.json files for later consumption by --from-ner (Tier 2).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_list:
        files = [Path(f) for f in file_list if Path(f).exists()]
    else:
        files = [
            f for f in input_dir.glob("**/*.json")
            if not f.name.endswith(("_names.json", "_names_llm.json",
                                     "_ner.json", "_clean.json",
                                     "_names_clean.json", "_summary.json"))
        ]

    files = sorted(files)
    logger.info(f"Found {len(files)} files for NER-only processing")

    stats = {"processed": 0, "skipped": 0, "failed": 0, "total_spans": 0}

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="io") as io_pool:
        prefetch: dict[Path, Future] = {}

        for i, file_path in enumerate(tqdm(files, desc="NER Tier 1 (Stage 0+1)")):
            for j in range(i + 1, min(i + 3, len(files))):
                if files[j] not in prefetch:
                    prefetch[files[j]] = io_pool.submit(
                        _prefetch_ner_only, files[j], output_dir, force
                    )

            try:
                if file_path in prefetch:
                    _, ocr_data, should_process = prefetch.pop(file_path).result()
                else:
                    _, ocr_data, should_process = _prefetch_ner_only(
                        file_path, output_dir, force
                    )

                if not should_process:
                    stats["skipped"] += 1
                    continue

                if ocr_data is None:
                    stats["failed"] += 1
                    continue

                source_file = Path(ocr_data.get("file_path", "")).name or file_path.stem

                result = pipeline.process_ner_only(ocr_data, source_file=source_file)
                save_ner_only_result(result, output_dir)

                stats["processed"] += 1
                stats["total_spans"] += result.total_spans

                # Log every 100th file (tqdm already shows progress)
                if stats["processed"] % 100 == 0:
                    logger.info(
                        f"Progress: {stats['processed']} processed, "
                        f"last: {file_path.name} "
                        f"({result.total_spans} spans, {result.person_spans} PER)"
                    )

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                stats["failed"] += 1

    logger.info(f"NER-only processing complete: {stats}")
    return stats


def _prefetch_ner_only(file_path: Path, output_dir: Path, force: bool) -> tuple:
    """Prefetch for NER-only mode: check if _ner.json exists."""
    output_path = output_dir / f"{file_path.stem}_ner.json"
    if not force and validate_ner_output_file(output_path):
        return (file_path, None, False)
    ocr_data = load_ocr_file(file_path)
    return (file_path, ocr_data, True)


def _early_skip_gate(ner_data: dict) -> bool:
    """Check if file can be skip-gated without loading text.json.

    Returns True if the file should be skipped (no LLM work needed).

    This gate is intentionally more aggressive than the pipeline gate in
    pipeline.py. It additionally skips files where recovery won't trigger
    (noise_score < MIN_NOISE_SCORE), since those files would go through
    the full pipeline only to produce zero names anyway.

    Skip conditions:
    1. person_spans == 0 AND clean_text_length < TIER2_SKIP_ESCAPE_HATCH_LENGTH:
       standard pipeline gate (always skip)
    2. person_spans == 0 AND clean_text_length >= TIER2_SKIP_ESCAPE_HATCH_LENGTH
       AND noise_score < MIN_NOISE_SCORE: recovery won't trigger, so skip
    """
    person_spans = ner_data.get("person_spans", 0)
    clean_text_length = ner_data.get("clean_text_length", 0)
    noise_score = ner_data.get("noise_score", 0.0)

    if person_spans > 0:
        return False

    # Standard skip gate: no spans AND short text
    if clean_text_length < TIER2_SKIP_ESCAPE_HATCH_LENGTH:
        return True

    # Enhanced skip: no spans, long text, but noise too low for recovery
    if noise_score < MIN_NOISE_SCORE:
        return True

    return False


def _save_minimal_names_json(stem: str, source_file: str, ner_data: dict,
                              output_dir: Path) -> None:
    """Write a minimal _names.json for skip-gated files (no LLM processing)."""
    names_path = output_dir / f"{stem}_names.json"
    names_data = {
        "source_file": source_file,
        "document_type": None,
        "pipeline_version": _get_pipeline_version(),
        "total_names": 0,
        "stats": {
            "stage0_clean_length": ner_data.get("clean_text_length", 0),
            "stage0_noise_score": ner_data.get("noise_score", 0.0),
            "stage1_ner_spans": ner_data.get("total_spans", 0),
            "stage1_person_spans": ner_data.get("person_spans", 0),
            "stage1_5_recovery_triggered": False,
            "stage1_5_recovered": 0,
            "stage2_classified": 0,
            "stage2_dropped": 0,
            "stage3_validated": 0,
            "stage3_failed": 0,
            "stage3_needs_repair": 0,
            "stage4_repaired": 0,
            "final_names": 0,
        },
        "names": [],
    }
    # Atomic write
    tmp_path = names_path.with_suffix(".tmp.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(names_data, f, ensure_ascii=False)
    os.rename(tmp_path, names_path)


def _prefetch_tier2(ner_file: Path, text_dir: Path, output_dir: Path,
                    force: bool) -> tuple[Path, dict | None, dict | None, bool, bool]:
    """Prefetch for Tier 2: load ner.json, optionally text.json.

    Returns:
        (ner_file, ner_data, ocr_data, should_process, early_skipped)
    """
    stem = ner_file.name.replace("_ner.json", "")
    output_path = output_dir / f"{stem}_names.json"

    # Quick existence check (discovery already filtered, but safety net)
    if not force and output_path.exists():
        return (ner_file, None, None, False, False)

    # Load NER data
    try:
        with open(ner_file, "r", encoding="utf-8") as f:
            ner_data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load NER file {ner_file}: {e}")
        return (ner_file, None, None, False, False)

    # Early skip gate: avoid loading text.json if no LLM work needed
    if not force and _early_skip_gate(ner_data):
        return (ner_file, ner_data, None, True, True)

    # Load text.json (needed for LLM stages)
    text_file = text_dir / f"{stem}.json"
    ocr_data = None
    if text_file.exists():
        try:
            with open(text_file, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)
        except Exception:
            pass

    return (ner_file, ner_data, ocr_data, True, False)


def _flush_batch(
    pipeline: ExtractionPipeline,
    batch_buffer: list[tuple[dict, dict, str]],
    output_dir: Path,
    inter_batch_size: int,
    stats: dict[str, int],
) -> None:
    """Process a buffer of documents through batched pipeline and save results."""
    if not batch_buffer:
        return

    try:
        results = pipeline.process_batch_from_ner(
            batch_buffer, inter_batch_size=inter_batch_size
        )

        for result in results:
            save_pipeline_result(result, output_dir)
            logger.info(
                f"  Final: {result.stats.final_names} names "
                f"(classified: {result.stats.stage2_classified}, "
                f"repaired: {result.stats.stage4_repaired}) "
                f"[{result.source_file}]"
            )
            stats["processed"] += 1
            stats["total_names"] += result.stats.final_names

    except Exception as e:
        logger.error(f"Batch processing failed ({len(batch_buffer)} docs): {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        # Fall back to sequential processing
        for ner_data, ocr_data, source_file in batch_buffer:
            try:
                result = pipeline.process_from_ner(
                    ner_data=ner_data, ocr_data=ocr_data,
                    source_file=source_file,
                )
                save_pipeline_result(result, output_dir)
                stats["processed"] += 1
                stats["total_names"] += result.stats.final_names
            except Exception as e2:
                logger.error(f"Sequential fallback failed for {source_file}: {e2}")
                stats["failed"] += 1


def process_directory_from_ner(
    pipeline: ExtractionPipeline,
    ner_dir: Path,
    text_dir: Path,
    output_dir: Path,
    force: bool = False,
    file_list: list[Path] | None = None,
    inter_batch_size: int = 1,
) -> dict[str, int]:
    """Process _ner.json files through Stages 1.5-4 (Tier 2).

    Reads _ner.json (from Tier 1) + original text JSON → produces _names.json.
    Features: early skip gate (avoids text.json I/O), I/O prefetching,
    inter-document batching (when inter_batch_size > 1).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wall_start = time.monotonic()

    if file_list:
        ner_files = [Path(f) for f in file_list if Path(f).exists()]
    else:
        ner_files = sorted(ner_dir.glob("**/*_ner.json"))

    logger.info(f"Found {len(ner_files)} _ner.json files for Tier 2 processing")
    if inter_batch_size > 1:
        logger.info(f"Inter-document batching enabled: batch_size={inter_batch_size}")

    stats = {
        "processed": 0, "skipped": 0, "skipped_gate": 0,
        "failed": 0, "total_names": 0
    }

    # Buffer for inter-document batching
    batch_buffer: list[tuple[dict, dict, str]] = []

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="io") as io_pool:
        prefetch: dict[Path, Future] = {}

        for i, ner_file in enumerate(tqdm(ner_files, desc="NER Tier 2 (Stages 1.5-4)")):
            # Prefetch next files while processing current
            lookahead = max(3, inter_batch_size + 1)
            for j in range(i + 1, min(i + lookahead, len(ner_files))):
                if ner_files[j] not in prefetch:
                    prefetch[ner_files[j]] = io_pool.submit(
                        _prefetch_tier2, ner_files[j], text_dir, output_dir, force
                    )

            try:
                # Get current file data (from prefetch if available, else load now)
                if ner_file in prefetch:
                    _, ner_data, ocr_data, should_process, early_skipped = prefetch.pop(ner_file).result()
                else:
                    _, ner_data, ocr_data, should_process, early_skipped = _prefetch_tier2(
                        ner_file, text_dir, output_dir, force
                    )

                if not should_process:
                    stats["skipped"] += 1
                    continue

                stem = ner_file.name.replace("_ner.json", "")
                source_file = ner_data.get("source_file", stem)

                # Early skip gate: write minimal output, no LLM work
                if early_skipped:
                    _save_minimal_names_json(stem, source_file, ner_data, output_dir)
                    stats["skipped_gate"] += 1
                    stats["processed"] += 1
                    continue

                # Need text.json for LLM stages
                if ocr_data is None:
                    logger.warning(f"Missing text file for {ner_file.name}, skipping")
                    stats["failed"] += 1
                    continue

                if inter_batch_size <= 1:
                    # Sequential mode (backward compatible)
                    logger.info(f"Processing (from-ner): {ner_file.name}")

                    result = pipeline.process_from_ner(
                        ner_data=ner_data,
                        ocr_data=ocr_data,
                        source_file=source_file,
                    )

                    save_pipeline_result(result, output_dir)

                    logger.info(
                        f"  Final: {result.stats.final_names} names "
                        f"(classified: {result.stats.stage2_classified}, "
                        f"repaired: {result.stats.stage4_repaired})"
                    )

                    stats["processed"] += 1
                    stats["total_names"] += result.stats.final_names
                else:
                    # Batched mode: accumulate into buffer
                    batch_buffer.append((ner_data, ocr_data, source_file))

                    if len(batch_buffer) >= inter_batch_size:
                        _flush_batch(
                            pipeline, batch_buffer, output_dir,
                            inter_batch_size, stats,
                        )
                        batch_buffer.clear()

            except Exception as e:
                logger.error(f"Failed to process {ner_file}: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                stats["failed"] += 1

        # Flush remaining buffer
        if batch_buffer:
            _flush_batch(
                pipeline, batch_buffer, output_dir,
                inter_batch_size, stats,
            )
            batch_buffer.clear()

    wall_elapsed = time.monotonic() - wall_start
    wall_min = wall_elapsed / 60

    # Throughput summary (stats["processed"] already includes skipped_gate)
    total_processed = stats["processed"]
    files_per_min = total_processed / wall_min if wall_min > 0 else 0

    logger.info(f"Tier 2 processing complete: {stats}")
    logger.info(
        f"=== Run Summary ==="
        f" | wall_time={wall_min:.1f}min"
        f" | files={total_processed}"
        f" | throughput={files_per_min:.1f} files/min"
        f" | skipped_existing={stats['skipped']}"
        f" | skipped_gate={stats['skipped_gate']}"
        f" | llm_processed={stats['processed'] - stats['skipped_gate']}"
        f" | failed={stats['failed']}"
        f" | total_names={stats['total_names']}"
    )

    # GPU batch stats if available
    if hasattr(pipeline, '_shared_llm_model') and pipeline._shared_llm_model is not None:
        bs = pipeline._shared_llm_model.batch_stats
        if bs.total_batches > 0:
            logger.info(
                f"=== GPU Batch Stats ==="
                f" | batches={bs.total_batches}"
                f" | prompts={bs.total_prompts}"
                f" | oom_splits={bs.oom_splits}"
                f" | padding_waste={bs.padding_waste_tokens} tokens"
            )

    # Classifier diagnostic counters (detect silent defaulting)
    if hasattr(pipeline, '_classifier') and pipeline._classifier is not None:
        clf = pipeline._classifier
        if hasattr(clf, 'parse_fail_count'):
            logger.info(
                f"=== Classifier Diagnostics ==="
                f" | parse_failures={clf.parse_fail_count}"
                f" | json_repairs_used={clf.json_repair_used_count}"
                f" | defaults_total={clf.defaults_count}"
            )

    return stats


def process_directory(
    pipeline: ExtractionPipeline,
    input_dir: Path,
    output_dir: Path,
    force: bool = False,
    file_list: list[Path] | None = None,
    save_intermediate: bool = False
) -> dict[str, int]:
    """Process all text files in a directory.

    Args:
        pipeline: Extraction pipeline instance.
        input_dir: Directory with extracted text JSON files.
        output_dir: Directory to save results.
        force: Overwrite existing outputs.
        file_list: Optional specific list of files to process.
        save_intermediate: Save intermediate stage outputs.

    Returns:
        Stats dict with counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get files to process
    if file_list:
        files = [Path(f) for f in file_list if Path(f).exists()]
    else:
        # Find all JSON files except output files
        files = [
            f for f in input_dir.glob("**/*.json")
            if not f.name.endswith("_names.json")
            and not f.name.endswith("_names_llm.json")
            and not f.name.endswith("_ner.json")
            and not f.name.endswith("_clean.json")
        ]

    files = sorted(files)
    logger.info(f"Found {len(files)} files to process")

    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "total_names": 0,
        "total_ner_spans": 0,
        "total_validated": 0,
        "total_repaired": 0,
    }

    # Use ThreadPoolExecutor for I/O prefetching
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="io") as io_pool:
        prefetch: dict[Path, Future] = {}

        for i, file_path in enumerate(tqdm(files, desc="Extracting names (v2)")):
            # Prefetch next 2 files while processing current
            for j in range(i + 1, min(i + 3, len(files))):
                if files[j] not in prefetch:
                    prefetch[files[j]] = io_pool.submit(_prefetch_file, files[j], output_dir, force)

            try:
                # Get current file data (from prefetch if available, else load now)
                if file_path in prefetch:
                    _, ocr_data, should_process = prefetch.pop(file_path).result()
                else:
                    _, ocr_data, should_process = _prefetch_file(file_path, output_dir, force)

                if not should_process:
                    logger.debug(f"Skipping (valid output exists): {file_path.name}")
                    stats["skipped"] += 1
                    continue

                if ocr_data is None:
                    stats["failed"] += 1
                    continue

                # Extract source file name
                source_file = Path(ocr_data.get("file_path", "")).name or file_path.stem

                logger.info(f"Processing: {file_path.name}")

                # Run pipeline
                result = pipeline.process_document(ocr_data, source_file=source_file)

                # Save results
                output_path = output_dir / f"{file_path.stem}_names.json"
                output_dir.mkdir(parents=True, exist_ok=True)

                saved = save_pipeline_result(
                    result,
                    output_dir,
                    save_intermediate=save_intermediate
                )

                logger.info(
                    f"Extracted {result.stats.final_names} names "
                    f"(NER: {result.stats.stage1_ner_spans}, "
                    f"validated: {result.stats.stage3_validated}, "
                    f"repaired: {result.stats.stage4_repaired})"
                )

                stats["processed"] += 1
                stats["total_names"] += result.stats.final_names
                stats["total_ner_spans"] += result.stats.stage1_ner_spans
                stats["total_validated"] += result.stats.stage3_validated
                stats["total_repaired"] += result.stats.stage4_repaired

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                stats["failed"] += 1

    logger.info(f"Processing complete: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract names using NER + LLM pipeline v2"
    )
    parser.add_argument(
        "--input-dir", "-i",
        default="data/processed/text",
        help="Input directory with extracted text JSON files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/processed/names_v2",
        help="Output directory for extracted names"
    )
    parser.add_argument(
        "--dataset",
        choices=["doj", "all"],
        default="all",
        help="Dataset to process"
    )
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        help="Specific files to process (optional)"
    )
    parser.add_argument(
        "--files-from",
        help="Read file paths from a text file (one per line). "
             "Use instead of --files to avoid argument list too long."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock extractors (for testing without GPU)"
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate stage outputs (_clean.json, _ner.json)"
    )
    parser.add_argument(
        "--ner-threshold",
        type=float,
        default=0.3,
        help="NER confidence threshold (lower = more recall)"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable 4-bit quantization (requires more VRAM)"
    )
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=50,
        help="Max spans per LLM classifier batch (default: 50)"
    )
    parser.add_argument(
        "--repair-batch-size",
        type=int,
        default=80,
        help="Max names per LLM repair batch (default: 80)"
    )
    parser.add_argument(
        "--llm-backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="LLM backend: 'transformers' (default, BNB 4-bit) or 'vllm' (AWQ, faster)"
    )
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="Disable Stage 1.5 LLM recovery (saves ~69s per corrupted doc)"
    )
    parser.add_argument(
        "--ner-batch-size",
        type=int,
        default=32,
        help="NER pipeline batch size (default: 32, reduce if OOM)"
    )

    parser.add_argument(
        "--inter-batch-size",
        type=int,
        default=1,
        help="Inter-document batch size for Tier 2 (default: 1 = sequential). "
             "Values >1 batch multiple documents into single GPU calls."
    )

    # Multi-tier split flags
    tier_group = parser.add_mutually_exclusive_group()
    tier_group.add_argument(
        "--ner-only",
        action="store_true",
        help="Tier 1: Run only Stage 0 + Stage 1 (XLM-R NER). "
             "Outputs _ner.json files. For Tier 1 GPU."
    )
    tier_group.add_argument(
        "--from-ner",
        action="store_true",
        help="Tier 2: Run Stages 1.5-4 from _ner.json files. "
             "Requires --ner-dir and --text-dir. For Tier 2 GPU."
    )
    parser.add_argument(
        "--ner-dir",
        help="Directory containing _ner.json files (for --from-ner)"
    )
    parser.add_argument(
        "--text-dir",
        help="Directory containing original text JSON files (for --from-ner)"
    )

    args = parser.parse_args()

    # Validate --from-ner args
    if args.from_ner and not args.text_dir:
        parser.error("--from-ner requires --text-dir")

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )

    # Build config
    config = PipelineConfig(
        use_gpu=not args.mock,
        mock=args.mock,
        quantize_4bit=not args.no_quantize,
        ner_confidence_threshold=args.ner_threshold,
        ner_batch_size=args.ner_batch_size,
        save_intermediate=args.save_intermediate,
        classifier_batch_size=args.classifier_batch_size,
        repair_batch_size=args.repair_batch_size,
        inter_batch_size=args.inter_batch_size,
        llm_backend=args.llm_backend,
        enable_recovery=not args.no_recovery,
    )

    # Create pipeline
    pipeline = ExtractionPipeline(config)

    try:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)

        # Adjust for dataset
        if args.dataset == "doj":
            input_dir = input_dir / "doj"
            output_dir = output_dir / "doj"

        file_list = None
        if args.files:
            file_list = [Path(f) for f in args.files]
        elif args.files_from:
            with open(args.files_from, "r") as fh:
                file_list = [Path(line.strip()) for line in fh if line.strip()]
            logger.info(f"Loaded {len(file_list)} files from {args.files_from}")

        if args.ner_only:
            # =============================================
            # Tier 1: NER-only (Stage 0 + Stage 1)
            # =============================================
            stats = process_directory_ner_only(
                pipeline,
                input_dir,
                output_dir,
                force=args.force,
                file_list=file_list,
            )

            print(f"\n{'=' * 50}")
            print(f"NER-Only (Tier 1) Complete")
            print(f"{'=' * 50}")
            print(f"Processed: {stats['processed']}")
            print(f"Skipped:   {stats['skipped']}")
            print(f"Failed:    {stats['failed']}")
            print(f"Total NER spans: {stats['total_spans']}")
            print(f"\nOutput directory: {output_dir}")

        elif args.from_ner:
            # =============================================
            # Tier 2: From NER (Stages 1.5-4)
            # =============================================
            ner_dir = Path(args.ner_dir) if args.ner_dir else output_dir
            text_dir = Path(args.text_dir)

            stats = process_directory_from_ner(
                pipeline,
                ner_dir,
                text_dir,
                output_dir,
                force=args.force,
                file_list=file_list,
                inter_batch_size=args.inter_batch_size,
            )

            print(f"\n{'=' * 50}")
            print(f"From-NER (Tier 2) Complete")
            print(f"{'=' * 50}")
            print(f"Processed: {stats['processed']}")
            print(f"Skipped:   {stats['skipped']}")
            print(f"Skipped (gate): {stats['skipped_gate']}")
            print(f"Failed:    {stats['failed']}")
            print(f"Total names: {stats['total_names']}")
            print(f"\nOutput directory: {output_dir}")

        else:
            # =============================================
            # Monolithic: Full pipeline (all stages)
            # =============================================
            stats = process_directory(
                pipeline,
                input_dir,
                output_dir,
                force=args.force,
                file_list=file_list,
                save_intermediate=args.save_intermediate
            )

            print(f"\n{'=' * 50}")
            print(f"Extraction Complete (v2 Pipeline)")
            print(f"{'=' * 50}")
            print(f"Processed: {stats['processed']}")
            print(f"Skipped:   {stats['skipped']}")
            print(f"Failed:    {stats['failed']}")
            print(f"\nTotals:")
            print(f"  NER spans:  {stats['total_ner_spans']}")
            print(f"  Validated:  {stats['total_validated']}")
            print(f"  Repaired:   {stats['total_repaired']}")
            print(f"  Final names: {stats['total_names']}")
            print(f"\nOutput directory: {output_dir}")

    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
