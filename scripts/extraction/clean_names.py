#!/usr/bin/env python3
"""JSON-level post-processing for extracted names.

Single-pass cleaning of _names.json files. Produces _names_clean.json files
that all downstream scripts (update_database, generate_summaries) should read from.

Cleaning steps:
1. Strip leading dots/spaces/hyphens from normalized_name
2. Filter garbage names (OCR artifacts, concatenated tokens, etc.)
3. Normalize non-standard roles to 'mentioned'
4. Deduplicate per document (keep highest-priority role per unique name)

Usage:
    # Dry run (preview changes)
    python scripts/extraction/clean_names.py --input data/processed/names_v2/doj

    # Apply (write _names_clean.json files)
    python scripts/extraction/clean_names.py --input data/processed/names_v2/doj --apply
"""

import argparse
import json
import multiprocessing
import os
import re
import sys
from pathlib import Path

from scripts.shared.constants import ROLE_PRIORITY


def clean_name(name: str) -> str:
    """Strip leading/trailing punctuation and collapse whitespace."""
    # Strip invisible Unicode characters (zero-width spaces, direction marks, BOM, etc.)
    name = re.sub(r'[\u200b-\u200f\u2028-\u202f\ufeff\u00ad]', '', name)
    # Replace underscores with spaces (OCR/extraction artifact)
    name = name.replace('_', ' ')
    # Collapse newlines and multi-spaces into single space
    name = re.sub(r'\s+', ' ', name)
    # Strip leading punctuation, OCR artifacts, and Unicode curly quotes
    name = name.lstrip("'. ,-;:!?°•\"_►▶◀▷◁▸◂▹◃·‣◦◉▣◆◇▰▱→⇒—–»›#…()[]{}=`~|/\\@^+<>\u2018\u2019\u201c\u201d\u201e\u2039")
    # Strip leading index numbers (page/line/list numbers, not years)
    # e.g., "2 A. Farmer" -> "A. Farmer", "14. Smith" -> "Smith", "2) Brune" -> "Brune"
    # Guard: only strip numbers <= 300 (avoids stripping years like "2005 Jeffrey Epstein")
    # or any number if followed by delimiter (., ), :, -)
    m = re.match(r'^(\d+)([.\s)\-:?\']+)\s*(.*)', name)
    if m:
        num, delim, rest = m.groups()
        if int(num) <= 300 or delim.strip() in ('.', ')', ':', '-'):
            if rest and len(rest) >= 3:
                name = rest
    # Strip fused leading digits (1-2 digits before letter or apostrophe, no delimiter)
    # e.g., "2Oc Saldana" -> "Oc Saldana", "34Enter Kiss" -> "Enter Kiss", "7'Erje" -> "'Erje" -> "Erje"
    # Only 1-2 digits to avoid stripping year prefixes like "2005Jeffrey"
    if not m:
        m2 = re.match(r'^(\d{1,2})[\']*([A-Za-z].*)', name)
        if m2:
            rest = m2.group(2)
            if len(rest) >= 3:
                name = rest
    # Strip trailing punctuation, OCR artifacts, and Unicode curly quotes
    name = re.sub(r'[,;:!?°•"\'►▶◀▷◁▸◂▹◃·‣◦◉▣◆◇▰▱→⇒—–»›‹#…()\[\]{}=`~|/\\@^+<>\u2018\u2019\u201c\u201d\u201e\u2039\u203a]+$', '', name)
    # Strip trailing ellipsis (2+ ASCII dots)
    name = re.sub(r'\.{2,}\s*$', '', name)
    # Strip trailing period NOT part of an initial (preceded by 2+ lowercase letters)
    # e.g., "bill gates." -> "bill gates", but "jeffrey e." stays
    name = re.sub(r'(?<=[a-z]{2})\.$', '', name)
    # Fix repeated name spans (NER/OCR artifacts)
    name = _fix_repeated_name(name)
    # Strip trailing email metadata tokens (NER over-extraction from email headers)
    # e.g., "Jeffrey Epstein Subject" → "Jeffrey Epstein", "Epstein Jeffrey Cc" → "Epstein Jeffrey"
    # Loop until stable: stripping guarded tokens may expose unguarded ones
    words_tmp = name.split()
    prev_len = -1
    while len(words_tmp) != prev_len:
        prev_len = len(words_tmp)
        while len(words_tmp) >= 2 and words_tmp[-1].lower() in _EMAIL_METADATA_TOKENS:
            words_tmp.pop()
        # Guarded tokens (to/from): only strip when 3+ tokens remain to preserve real surnames
        # e.g., "Jeffrey Epstein To" → "Jeffrey Epstein", but "Nina To" stays
        while len(words_tmp) >= 3 and words_tmp[-1].lower() in _EMAIL_METADATA_GUARDED:
            words_tmp.pop()
    name = ' '.join(words_tmp)
    # Strip truncated trailing token that is a prefix of a preceding token
    # (OCR truncation artifact: "epstein ep" → "epstein", "jeffrey epstein ep" → "jeffrey epstein")
    words_tmp = name.split()
    if len(words_tmp) >= 2:
        last_lower = words_tmp[-1].lower()
        if len(last_lower) >= 2:
            first_lower = words_tmp[0].lower()
            if len(first_lower) > len(last_lower) and first_lower.startswith(last_lower):
                name = ' '.join(words_tmp[:-1])
            elif len(words_tmp) >= 3:
                prev_lower = words_tmp[-2].lower()
                if len(prev_lower) > len(last_lower) and prev_lower.startswith(last_lower):
                    name = ' '.join(words_tmp[:-1])
    return name.strip()


_EMAIL_METADATA_TOKENS = frozenset({
    'subject', 'cc', 'bcc', 'unauthorized', 'attachments', 'fwd', 'fw',
    'sent', 'date', 'received', 'importance',
})

# Tokens that are only stripped when 3+ tokens remain (could be real surnames)
_EMAIL_METADATA_GUARDED = frozenset({'to', 'from'})

_SUFFIX_TOKENS = frozenset({'jr', 'sr', 'ii', 'iii', 'iv', 'v', 'esq', 'phd', 'md'})
_HONORIFICS = frozenset({
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'lady', 'rev', 'judge', 'sgt',
    'secretary', 'senator', 'governor', 'president', 'captain',
    'lieutenant', 'colonel', 'general', 'admiral', 'detective', 'officer',
    'agent', 'director', 'commissioner', 'attorney', 'counselor', 'justice',
})
_TRAILING_STOPWORDS = frozenset({
    # Articles
    'the',
    # Be verbs
    'is', 'was', 'are', 'were', 'be', 'been',
    # Have verbs
    'has', 'had', 'have',
    # Do verbs
    'do', 'did', 'does',
    # Subject pronouns
    'he', 'she', 'we', 'they', 'it',
    # Object pronouns
    'him', 'her', 'us', 'them',
    # Possessive
    'his', 'my', 'our', 'its', 'your', 'their',
    # Conjunctions
    'if', 'but', 'so', 'nor', 'and', 'or',
    # Prepositions (clearly not surnames)
    'of', 'for', 'with', 'from', 'into', 'as', 'at', 'by',
    # Informal
    'ok', 'im',
    # Adverbs
    'not', 'just', 'also', 'then', 'than',
    # Demonstratives / relatives
    'that', 'this', 'which', 'who', 'whom',
    # Modals
    'would', 'could', 'should', 'shall', 'might', 'can', 'will',
    # Other
    'behalf',
    # Email metadata (NER over-extraction from email headers)
    'subject', 'cc', 'bcc', 'unauthorized', 'attachments', 'fwd', 'fw',
    'sent', 'date', 'received', 'importance',
})


def _fix_repeated_name(name: str) -> str:
    """Fix repeated name patterns from NER/OCR span errors.

    Handles: exact duplication, tripled, trailing echo, leading echo,
    punctuation/dash/paren/slash/pipe separators between repetitions.
    Preserves suffixes (Jr., Sr., III, etc.) after block-repeat detection.
    """
    # Strip separator noise that splits repeated blocks
    # Commas, semicolons, em/en dashes (with or without spaces)
    # NOTE: Period NOT included — it's part of initials/suffixes (Jr., E.)
    cleaned = re.sub(r'\s*[,;—–()/|]\s*', ' ', name)
    # Hyphens only when surrounded by spaces (preserve intra-word like "Smith-Jones")
    cleaned = re.sub(r'\s+[-]\s+', ' ', cleaned)
    words = cleaned.split()
    if len(words) < 2:
        return name

    # 0. Period-separated same-token repeat: "epstein. epstein" → "epstein"
    #    Only fires when original name has a period acting as separator.
    #    Handles 2-word case that block-repeat (min block_size=2) can't.
    if '.' in name:
        stripped = [w.rstrip('.') for w in words]
        if len(set(s.lower() for s in stripped)) == 1:
            return stripped[0]

    # 1. Check for exact N-word block repeated 2+ times
    #    Try block sizes from len//2 down to 2
    for block_size in range(len(words) // 2, 1, -1):
        block = words[:block_size]
        # Check if the block repeats starting at each position
        pos = block_size
        repeats = 1
        while pos + block_size <= len(words) and words[pos:pos + block_size] == block:
            repeats += 1
            pos += block_size
        if repeats >= 2:
            # Block repeats. Check trailing tokens for suffixes to preserve.
            block_set = {w.lower() for w in block}
            suffix_tokens = []
            for token in words[pos:]:
                if token.lower().rstrip('.') in _SUFFIX_TOKENS:
                    suffix_tokens.append(token)
                elif token.lower() in block_set:
                    continue  # echo of block word, skip
                else:
                    break  # unknown trailing token, stop
            return ' '.join(block + suffix_tokens)

    # 2. Last-token echo: "David Oscar Markus Markus" -> "David Oscar Markus"
    if len(words) >= 3 and words[-1] == words[-2]:
        return ' '.join(words[:-1])

    # 3. First-token echo: "David David Oscar Markus" -> "David Oscar Markus"
    if len(words) >= 3 and words[0] == words[1]:
        return ' '.join(words[:1] + words[2:])

    # 4. Last hyphenated-token echo: "Anna-Marie Smith-Jones Smith-Jones"
    if len(words) >= 3 and '-' in words[-1] and words[-1] == words[-2]:
        return ' '.join(words[:-1])

    # 5. Wrap-around: first token matches last token ("Gates Bill Gates" -> "Bill Gates")
    if len(words) >= 3 and words[0].lower() == words[-1].lower():
        return ' '.join(words[1:])

    return name


def is_garbage(name: str) -> bool:
    """Check if a name is garbage that should be filtered out.

    Rules:
    - Too short (<=2 chars)
    - Contains 3+ consecutive digits (phone numbers, IDs)
    - No letters at all
    - Single token >12 chars with no separators (concatenated OCR artifact)
    - Mixed case within a single token (e.g., "EpsteinJeffrey")
    - Common pronouns / non-names
    - Contains newlines (multi-line NER artifact)
    - Looks like two names merged (e.g., "Epstein Joe", "Joe Joseph E")
    """
    # Too long (concatenated name lists, table headers, OCR-garbled rosters)
    if len(name) >= 60:
        return True
    # Too many tokens (same failure modes as above)
    if len(name.split()) >= 8:
        return True
    if len(name) <= 2:
        return True
    if re.search(r'\d{3,}', name):
        return True
    if not re.search(r'[a-zA-Z]', name):
        return True
    # Concatenated words: single token over 12 chars with no separators
    if len(name) > 12 and ' ' not in name and '.' not in name and '-' not in name:
        return True
    # Mixed case within a single token (e.g., "EpsteinJeffrey")
    if ' ' not in name and re.search(r'[a-z][A-Z]', name):
        return True
    # Common pronouns, honorifics, and non-names
    if name.lower().rstrip('.') in {
        'him', 'her', 'his', 'she', 'they', 'them', 'who', 'whom',
        'sir', 'mrs', 'mr', 'ms', 'dr', 'prof', 'esq',
        'inc', 'llc', 'etc', 'the', 'and',
        'none', 'unknown', 'redacted', 'undisclosed',
        'jr', 'sr', 'ii', 'iii', 'iv',
    }:
        return True
    # Contains newline — multi-line NER artifact
    if '\n' in name or '\r' in name:
        return True
    # All words are the same token: "Markus Markus Markus"
    words = name.split()
    if len(words) >= 2 and len(set(w.lower() for w in words)) == 1:
        return True
    # Contains redaction blocks or geometric shapes (█, ■, □, ▪, etc.)
    if re.search(r'[\u2580-\u259f\u25a0-\u25ff]', name):
        return True
    # Contains OCR artifacts (°, •, ►, ▶, etc.) in the middle of the name
    if re.search(r'[°•►▶◀▷◁▸◂▹◃·‣◦◉▣◆◇▰▱→⇒]', name):
        return True
    # Digits mixed with letters (names never contain digits)
    if re.search(r'\d', name) and re.search(r'[a-zA-Z]', name):
        return True
    # Structural punctuation in the name (parens, brackets, braces)
    if re.search(r'[)(}{[\]]', name):
        return True
    # Any individual token is a bare honorific (e.g., "Epstein Mr", "Dr Smith Jones")
    for token in words:
        if token.lower().rstrip('.') in _HONORIFICS:
            return True
    # Contains comma (comma-separated lists, malformed entries)
    if ',' in name:
        return True
    # Contains ampersand (two people merged: "Roy & Stephanie")
    if '&' in name:
        return True
    # Contains exclamation mark (OCR artifact: "de!son", "!gala")
    if '!' in name:
        return True
    # Contains colon (OCR artifact: "D: Edwards")
    if ':' in name:
        return True
    # Contains underscore (extraction artifact surviving clean_name)
    if '_' in name:
        return True
    # Trailing common stopword (NER over-extraction: "bill gates ok", "jeffrey epstein the")
    if len(words) >= 2 and words[-1].lower() in _TRAILING_STOPWORDS:
        return True
    # Leading common stopword (NER over-extraction: "is gates", "the jeffrey")
    if len(words) >= 2 and words[0].lower() in _TRAILING_STOPWORDS:
        return True
    # Bare lowercase single-letter token at FIRST or LAST position only
    # Middle single letters are likely initials (e.g., "alan m dershowitz" = Alan M. Dershowitz)
    # DB canonical_names are all lowercase, so we can't rely on case to distinguish initials
    # First position: "e epstein" (garbage), Last position: "andrew l" (truncated)
    if words:
        first = words[0]
        if len(first) == 1 and first.isalpha() and first.islower():
            return True
        if len(words) >= 2:
            last = words[-1]
            if len(last) == 1 and last.isalpha() and last.islower():
                return True
    # Last token is a prefix of first token (truncation: "epstein ep", "maxwell max")
    if len(words) >= 2:
        first_lower = words[0].lower()
        last_lower = words[-1].lower()
        if len(last_lower) >= 2 and len(first_lower) > len(last_lower) and first_lower.startswith(last_lower):
            return True
    return False


# Minimum confidence threshold — below this, likely not a real name
MIN_CONFIDENCE = 0.70


def normalize_role(role: str) -> str:
    """Normalize non-standard roles to 'mentioned'."""
    if role in ROLE_PRIORITY:
        return role
    return "mentioned"


def clean_names_file(data: dict) -> tuple[dict, dict]:
    """Clean names in a single extraction result.

    Args:
        data: Parsed JSON from a _names.json file.

    Returns:
        Tuple of (cleaned data dict, stats dict).
    """
    stats = {
        "original_count": 0,
        "cleaned_count": 0,
        "garbage_removed": 0,
        "roles_remapped": 0,
        "deduplicated": 0,
    }

    raw_names = data.get("names", [])
    stats["original_count"] = len(raw_names)

    # Phase 1: Clean and filter
    cleaned = []
    for entry in raw_names:
        name = entry.get("normalized_name") or entry.get("original_text", "")
        name = clean_name(name)
        if not name or is_garbage(name):
            stats["garbage_removed"] += 1
            continue

        # Filter low-confidence entries
        confidence = entry.get("confidence")
        if confidence is not None and confidence < MIN_CONFIDENCE:
            stats["garbage_removed"] += 1
            continue

        role = entry.get("role", "other")
        new_role = normalize_role(role)
        if new_role != role:
            stats["roles_remapped"] += 1

        cleaned_entry = dict(entry)
        if "normalized_name" in cleaned_entry:
            cleaned_entry["normalized_name"] = name
        if "original_text" in cleaned_entry:
            # Keep original_text as-is for provenance; update normalized_name
            pass
        cleaned_entry["role"] = new_role
        cleaned.append(cleaned_entry)

    # Phase 2: Deduplicate (keep highest-priority role per unique name)
    best = {}  # name_lower -> (priority, index)
    for i, entry in enumerate(cleaned):
        name = (entry.get("normalized_name") or entry.get("original_text", "")).lower()
        role = entry.get("role", "other")
        priority = ROLE_PRIORITY.get(role, 4)

        if name not in best or priority < best[name][0]:
            best[name] = (priority, i)

    before_dedup = len(cleaned)
    deduped = [cleaned[idx] for _, idx in sorted(best.values(), key=lambda x: x[1])]
    stats["deduplicated"] = before_dedup - len(deduped)
    stats["cleaned_count"] = len(deduped)

    # Build output (preserve all non-names metadata)
    output = {k: v for k, v in data.items() if k != "names"}
    output["names"] = deduped

    return output, stats


def _process_single_file(args: tuple[Path, bool]) -> tuple[dict, str]:
    """Process a single file (helper for multiprocessing).

    Args:
        args: Tuple of (file_path, apply).

    Returns:
        Tuple of (stats_dict, file_path_str).
    """
    file_path, apply = args

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {
            "error": str(e),
            "files_skipped": 1,
        }, str(file_path)

    if not isinstance(data, dict):
        return {
            "error": "not a dict",
            "files_skipped": 1,
        }, str(file_path)

    cleaned_data, stats = clean_names_file(data)

    # Build result stats
    result_stats = {
        "files_processed": 1,
        "files_skipped": 0,
        "total_original": stats["original_count"],
        "total_cleaned": stats["cleaned_count"],
        "total_garbage_removed": stats["garbage_removed"],
        "total_roles_remapped": stats["roles_remapped"],
        "total_deduplicated": stats["deduplicated"],
    }

    if apply:
        # Write _names_clean.json alongside the original
        output_path = file_path.parent / file_path.name.replace(
            "_names.json", "_names_clean.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    return result_stats, str(file_path)


def process_directory(input_dir: Path, apply: bool = False, workers: int = None) -> dict:
    """Process all _names.json files in a directory.

    Args:
        input_dir: Directory containing _names.json files.
        apply: If True, write _names_clean.json files.
        workers: Number of worker processes (default: CPU count).

    Returns:
        Aggregate statistics dict.
    """
    input_files = sorted(input_dir.glob("*_names.json"))
    # Exclude already-cleaned files
    input_files = [f for f in input_files if not f.name.endswith("_names_clean.json")]

    totals = {
        "files_processed": 0,
        "files_skipped": 0,
        "total_original": 0,
        "total_cleaned": 0,
        "total_garbage_removed": 0,
        "total_roles_remapped": 0,
        "total_deduplicated": 0,
    }

    if not input_files:
        print(f"No _names.json files found in {input_dir}")
        return totals

    print(f"Found {len(input_files)} _names.json files in {input_dir}")

    # Determine number of workers
    if workers is None:
        workers = os.cpu_count() or 1
    workers = min(workers, len(input_files))

    print(f"Using {workers} worker processes")

    # Process files in parallel
    with multiprocessing.Pool(workers) as pool:
        file_args = [(f, apply) for f in input_files]

        for result_stats, file_path in pool.imap_unordered(_process_single_file, file_args):
            # Aggregate stats
            if "error" in result_stats:
                print(f"  WARNING: Skipping {Path(file_path).name}: {result_stats['error']}")
                totals["files_skipped"] += result_stats.get("files_skipped", 0)
            else:
                totals["files_processed"] += result_stats["files_processed"]
                totals["total_original"] += result_stats["total_original"]
                totals["total_cleaned"] += result_stats["total_cleaned"]
                totals["total_garbage_removed"] += result_stats["total_garbage_removed"]
                totals["total_roles_remapped"] += result_stats["total_roles_remapped"]
                totals["total_deduplicated"] += result_stats["total_deduplicated"]

    return totals


def process_files(files: list[Path], apply: bool = False) -> dict:
    """Process specific _names.json files (for per-worker clean_names).

    Unlike process_directory() which scans a directory, this takes an explicit
    file list so each SLURM worker can clean only its chunk.

    Args:
        files: List of _names.json file paths to process.
        apply: If True, write _names_clean.json files.

    Returns:
        Aggregate statistics dict.
    """
    # Safety: filter out _names_clean.json files
    input_files = [f for f in files if not f.name.endswith("_names_clean.json")]

    totals = {
        "files_processed": 0,
        "files_skipped": 0,
        "total_original": 0,
        "total_cleaned": 0,
        "total_garbage_removed": 0,
        "total_roles_remapped": 0,
        "total_deduplicated": 0,
    }

    if not input_files:
        print(f"No _names.json files in provided list")
        return totals

    print(f"Processing {len(input_files)} _names.json files")

    workers = min(os.cpu_count() or 1, len(input_files))
    print(f"Using {workers} worker processes")

    with multiprocessing.Pool(workers) as pool:
        file_args = [(f, apply) for f in input_files]

        for result_stats, file_path in pool.imap_unordered(_process_single_file, file_args):
            if "error" in result_stats:
                print(f"  WARNING: Skipping {Path(file_path).name}: {result_stats['error']}")
                totals["files_skipped"] += result_stats.get("files_skipped", 0)
            else:
                totals["files_processed"] += result_stats["files_processed"]
                totals["total_original"] += result_stats["total_original"]
                totals["total_cleaned"] += result_stats["total_cleaned"]
                totals["total_garbage_removed"] += result_stats["total_garbage_removed"]
                totals["total_roles_remapped"] += result_stats["total_roles_remapped"]
                totals["total_deduplicated"] += result_stats["total_deduplicated"]

    return totals


def main():
    parser = argparse.ArgumentParser(
        description="Clean extracted names JSON files (post-processing)"
    )
    parser.add_argument(
        "--input", required=True, help="Directory with _names.json files"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Write _names_clean.json files (default: dry run)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of worker processes (default: CPU count)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"ERROR: Not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    totals = process_directory(input_dir, apply=args.apply, workers=args.workers)

    # Report
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"\n[{mode}] Post-processing summary:")
    print(f"  Files processed:   {totals['files_processed']}")
    print(f"  Files skipped:     {totals['files_skipped']}")
    print(f"  Names (original):  {totals['total_original']}")
    print(f"  Names (cleaned):   {totals['total_cleaned']}")
    print(f"  Garbage removed:   {totals['total_garbage_removed']}")
    print(f"  Roles remapped:    {totals['total_roles_remapped']}")
    print(f"  Deduplicated:      {totals['total_deduplicated']}")


if __name__ == "__main__":
    main()
