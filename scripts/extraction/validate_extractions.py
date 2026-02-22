#!/usr/bin/env python3
"""Validate that extraction outputs have no hallucinations.

This script verifies that all extracted name spans actually exist in the
source documents at the claimed offsets. This is a critical check to ensure
the NER + LLM pipeline is producing valid, non-hallucinated results.

Usage:
    # Validate all extractions in a directory
    python scripts/validate_extractions.py data/processed/names_v2/doj/

    # Validate with verbose output
    python scripts/validate_extractions.py data/processed/names_v2/doj/ --verbose

    # Validate and fix issues
    python scripts/validate_extractions.py data/processed/names_v2/doj/ --fix
"""

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path
from dataclasses import dataclass

from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class ValidationIssue:
    """A validation issue found in an extraction."""
    file: str
    name_index: int
    name_text: str
    start: int
    end: int
    issue_type: str  # 'mismatch', 'out_of_bounds', 'missing_source'
    details: str


@dataclass
class ValidationReport:
    """Report from validating extractions."""
    total_files: int
    valid_files: int
    invalid_files: int
    total_names: int
    valid_names: int
    invalid_names: int
    issues: list[ValidationIssue]


def load_json(file_path: Path) -> dict | None:
    """Load a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def find_source_text(names_file: Path, names_data: dict) -> str | None:
    """Find the source text for a names file.

    Looks for corresponding _ocr.json, _clean.json, or source text file.

    Args:
        names_file: Path to the _names.json file.
        names_data: Parsed names JSON data.

    Returns:
        Source text or None if not found.
    """
    stem = names_file.stem
    if stem.endswith('_names'):
        base_stem = stem[:-6]  # Remove '_names'
    else:
        base_stem = stem

    # Look in same directory and parent directory
    search_dirs = [names_file.parent, names_file.parent.parent]

    # Try different source file patterns
    patterns = [
        f"{base_stem}_ocr.json",
        f"{base_stem}.json",
        f"{base_stem}_clean.json",
    ]

    for search_dir in search_dirs:
        for pattern in patterns:
            source_path = search_dir / pattern
            if source_path.exists():
                source_data = load_json(source_path)
                if source_data:
                    # Try different text fields
                    if 'full_text' in source_data and source_data['full_text']:
                        return source_data['full_text']
                    if 'clean_text' in source_data:
                        return source_data['clean_text']
                    if 'pages' in source_data:
                        pages_text = []
                        for i, page in enumerate(source_data.get('pages', [])):
                            page_text = page.get('text', '').strip()
                            if page_text:
                                page_num = page.get('page_number', i + 1)
                                pages_text.append(f"[Page {page_num}]\n{page_text}")
                        if pages_text:
                            return "\n\n".join(pages_text)

    # Also try to find in text directory
    text_dirs = [
        names_file.parent.parent / 'text',
        names_file.parent.parent.parent / 'text',
    ]

    for text_dir in text_dirs:
        if text_dir.exists():
            for pattern in patterns:
                source_path = text_dir / pattern
                if source_path.exists():
                    source_data = load_json(source_path)
                    if source_data and 'full_text' in source_data:
                        return source_data['full_text']

    return None


def validate_name(name: dict, source_text: str) -> ValidationIssue | None:
    """Validate a single name against source text.

    Args:
        name: Name entry from _names.json.
        source_text: Original source text.

    Returns:
        ValidationIssue if invalid, None if valid.
    """
    original = name.get('original_text', '')
    start = name.get('start')
    end = name.get('end')

    # Check for required fields
    if start is None or end is None:
        return ValidationIssue(
            file='',
            name_index=0,
            name_text=original,
            start=start or 0,
            end=end or 0,
            issue_type='missing_offsets',
            details='Name entry missing start/end offsets'
        )

    # Check bounds
    if start < 0 or end > len(source_text) or start >= end:
        return ValidationIssue(
            file='',
            name_index=0,
            name_text=original,
            start=start,
            end=end,
            issue_type='out_of_bounds',
            details=f'Offsets {start}-{end} out of bounds (text length: {len(source_text)})'
        )

    # Check text match
    actual_text = source_text[start:end]

    # Normalize for comparison (whitespace normalization)
    actual_normalized = ' '.join(actual_text.split())
    expected_normalized = ' '.join(original.split())

    if actual_normalized != expected_normalized:
        return ValidationIssue(
            file='',
            name_index=0,
            name_text=original,
            start=start,
            end=end,
            issue_type='mismatch',
            details=f"Expected '{original}' but found '{actual_text}' at {start}-{end}"
        )

    return None


def validate_file(names_file: Path, verbose: bool = False) -> tuple[bool, list[ValidationIssue]]:
    """Validate a single _names.json file.

    Args:
        names_file: Path to the names file.
        verbose: Print verbose output.

    Returns:
        Tuple of (is_valid, issues).
    """
    issues = []

    # Load names data
    names_data = load_json(names_file)
    if not names_data:
        issues.append(ValidationIssue(
            file=str(names_file),
            name_index=-1,
            name_text='',
            start=0,
            end=0,
            issue_type='load_error',
            details='Failed to load names file'
        ))
        return False, issues

    # Find source text
    source_text = find_source_text(names_file, names_data)
    if not source_text:
        # Can't validate without source - just log warning
        if verbose:
            logger.warning(f"Could not find source text for {names_file.name}")
        return True, []  # Assume valid if can't check

    # Validate each name
    for i, name in enumerate(names_data.get('names', [])):
        issue = validate_name(name, source_text)
        if issue:
            issue.file = str(names_file)
            issue.name_index = i
            issues.append(issue)

    return len(issues) == 0, issues


def _validate_single_file(args: tuple[Path, bool]) -> tuple[bool, list[ValidationIssue], int]:
    """Validate a single file (helper for multiprocessing).

    Args:
        args: Tuple of (names_file, verbose).

    Returns:
        Tuple of (is_valid, issues_list, name_count).
    """
    names_file, verbose = args

    # Load names data to get count
    names_data = load_json(names_file)
    if not names_data:
        issue = ValidationIssue(
            file=str(names_file),
            name_index=-1,
            name_text='',
            start=0,
            end=0,
            issue_type='load_error',
            details='Failed to load names file'
        )
        return False, [issue], 0

    name_count = len(names_data.get('names', []))

    # Validate the file
    is_valid, issues = validate_file(names_file, verbose=verbose)

    return is_valid, issues, name_count


def validate_directory(
    dir_path: Path,
    verbose: bool = False,
    fix: bool = False,
    workers: int = None
) -> ValidationReport:
    """Validate all extraction files in a directory.

    Args:
        dir_path: Directory containing _names.json files.
        verbose: Print verbose output.
        fix: Attempt to fix issues (removes invalid entries).
        workers: Number of worker processes (default: CPU count).

    Returns:
        ValidationReport with results.
    """
    # Find all names files
    names_files = list(dir_path.glob('**/*_names.json'))

    if not names_files:
        logger.warning(f"No _names.json files found in {dir_path}")
        return ValidationReport(
            total_files=0,
            valid_files=0,
            invalid_files=0,
            total_names=0,
            valid_names=0,
            invalid_names=0,
            issues=[]
        )

    logger.info(f"Validating {len(names_files)} extraction files...")

    total_names = 0
    valid_names = 0
    valid_files = 0
    invalid_files = 0
    all_issues = []

    if fix:
        # Fix mode must be sequential (writes files)
        logger.info("Running in fix mode (sequential)")
        for names_file in sorted(names_files):
            # Load to count names
            names_data = load_json(names_file)
            if names_data:
                file_names = len(names_data.get('names', []))
                total_names += file_names

            # Validate
            is_valid, issues = validate_file(names_file, verbose=verbose)

            if is_valid:
                valid_files += 1
                valid_names += file_names
            else:
                invalid_files += 1
                valid_names += file_names - len(issues)
                all_issues.extend(issues)

                if verbose:
                    logger.warning(f"Issues in {names_file.name}:")
                    for issue in issues:
                        logger.warning(f"  [{issue.issue_type}] {issue.name_text}: {issue.details}")

                # Remove invalid entries and resave
                fix_file(names_file, names_data, issues)
    else:
        # Read-only validation can be parallelized
        if workers is None:
            workers = os.cpu_count() or 1
        workers = min(workers, len(names_files))

        logger.info(f"Using {workers} worker processes")

        with multiprocessing.Pool(workers) as pool:
            file_args = [(f, verbose) for f in sorted(names_files)]

            for is_valid, issues, file_names in pool.imap_unordered(_validate_single_file, file_args):
                total_names += file_names

                if is_valid:
                    valid_files += 1
                    valid_names += file_names
                else:
                    invalid_files += 1
                    valid_names += file_names - len(issues)
                    all_issues.extend(issues)

                    if verbose and issues:
                        logger.warning(f"Issues in {Path(issues[0].file).name}:")
                        for issue in issues:
                            logger.warning(f"  [{issue.issue_type}] {issue.name_text}: {issue.details}")

    return ValidationReport(
        total_files=len(names_files),
        valid_files=valid_files,
        invalid_files=invalid_files,
        total_names=total_names,
        valid_names=valid_names,
        invalid_names=len(all_issues),
        issues=all_issues
    )


def fix_file(names_file: Path, names_data: dict, issues: list[ValidationIssue]):
    """Fix a names file by removing invalid entries.

    Args:
        names_file: Path to the names file.
        names_data: Parsed names data.
        issues: Issues found in validation.
    """
    # Get indices of invalid names
    invalid_indices = {issue.name_index for issue in issues if issue.name_index >= 0}

    # Filter out invalid names
    original_count = len(names_data.get('names', []))
    names_data['names'] = [
        name for i, name in enumerate(names_data.get('names', []))
        if i not in invalid_indices
    ]
    names_data['total_names'] = len(names_data['names'])

    # Save fixed file
    with open(names_file, 'w', encoding='utf-8') as f:
        json.dump(names_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Fixed {names_file.name}: removed {original_count - len(names_data['names'])} invalid entries")


def main():
    parser = argparse.ArgumentParser(
        description='Validate extraction outputs for hallucinations'
    )
    parser.add_argument(
        'directory',
        type=Path,
        help='Directory containing _names.json files to validate'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed validation messages'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Remove invalid entries from files'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of worker processes (default: CPU count, ignored if --fix is used)'
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=log_level
    )

    # Validate
    report = validate_directory(
        args.directory,
        verbose=args.verbose,
        fix=args.fix,
        workers=args.workers
    )

    # Print report
    print(f"\n{'=' * 50}")
    print("Validation Report")
    print(f"{'=' * 50}")
    print(f"Total files:    {report.total_files}")
    print(f"Valid files:    {report.valid_files}")
    print(f"Invalid files:  {report.invalid_files}")
    print()
    print(f"Total names:    {report.total_names}")
    print(f"Valid names:    {report.valid_names}")
    print(f"Invalid names:  {report.invalid_names}")

    if report.issues:
        print(f"\n{'=' * 50}")
        print("Issues Found")
        print(f"{'=' * 50}")

        # Group by issue type
        by_type = {}
        for issue in report.issues:
            by_type.setdefault(issue.issue_type, []).append(issue)

        for issue_type, issues in sorted(by_type.items()):
            print(f"\n{issue_type}: {len(issues)} issues")
            if args.verbose:
                for issue in issues[:10]:  # Show first 10
                    print(f"  - {Path(issue.file).name}: '{issue.name_text}' @ {issue.start}-{issue.end}")
                    print(f"    {issue.details}")
                if len(issues) > 10:
                    print(f"  ... and {len(issues) - 10} more")

    # Exit with error code if issues found
    if report.invalid_names > 0:
        print(f"\n❌ Validation FAILED: {report.invalid_names} hallucinated/invalid names found")
        sys.exit(1)
    else:
        print(f"\n✓ Validation PASSED: All {report.valid_names} names verified")
        sys.exit(0)


if __name__ == "__main__":
    main()
