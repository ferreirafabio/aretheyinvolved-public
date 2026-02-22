#!/usr/bin/env python3
"""Generate short and long summaries for documents using page-aware map→reduce.

Architecture:
  1. MAP: Per-page summary (1-2 sentences each)
  2. REDUCE: Document-level short_summary + long_summary from page summaries

This handles multi-page documents without blowing context, keeps costs
predictable, and produces consistent summaries.

Usage:
    # Dry run (preview)
    python scripts/extraction/generate_summaries.py --dataset doj

    # Generate summaries
    python scripts/extraction/generate_summaries.py --dataset doj --apply

    # Resume from a specific file
    python scripts/extraction/generate_summaries.py --dataset doj --apply --resume-from EFTA00005000

    # Limit number of files
    python scripts/extraction/generate_summaries.py --dataset doj --apply --limit 100
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from loguru import logger

# Pattern to strip Qwen3 <think>...</think> blocks from output
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.DOTALL)


def _collapse_repetition(text: str, min_repeat_len: int = 50) -> str:
    """Detect and collapse repeated JSON blocks or text blocks."""
    if len(text) < min_repeat_len * 2:
        return text
    # Look for a repeated JSON block: {...} {...} {...}...
    first_close = text.find("}")
    if first_close > 0 and first_close < len(text) - 10:
        candidate = text[:first_close + 1].strip()
        if candidate.startswith("{"):
            rest = text[first_close + 1:].strip()
            if rest.startswith(candidate[:20]):
                return candidate  # Return just the first instance
    return text


def _strip_thinking(text: str) -> str:
    """Strip <think> blocks and collapse repetition loops."""
    text = _THINK_RE.sub("", text).strip()
    text = _collapse_repetition(text)
    return text


def _extract_short_summary(response: str) -> str:
    """Extract clean short summary from LLM response.

    SHORT_MULTI prompts ask for free text, but the model may return JSON
    or chain-of-thought reasoning. This extracts the actual summary.
    """
    text = str(response).strip() if response else ""
    if not text:
        return ""
    # Try JSON parse first (model may return JSON even for SHORT_MULTI)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "summary" in parsed:
            return str(parsed["summary"]).strip()
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end])
                if isinstance(parsed, dict) and "summary" in parsed:
                    return str(parsed["summary"]).strip()
            except json.JSONDecodeError:
                pass
    return text


# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.extraction.clean_names import process_files as clean_names_files
from scripts.shared.constants import DOCUMENT_TYPES, OCCUPATION_SYNONYMS, ROLE_PRIORITY

# Version tracking for cache invalidation
MODEL_VERSION = "qwen3-32b"
PROMPT_VERSION = "v6"

# Max prompts per batch to avoid OOM on large docs (e.g. 97-page police reports)
# Increased to 64 when vLLM is available (manages GPU memory via KV-cache paging)
BATCH_CHUNK_SIZE = 16
BATCH_CHUNK_SIZE_VLLM = 64

# Token budget per batch chunk — limits total input tokens to avoid KV cache pressure.
# Conservative for 80GB GPU with Qwen3-32B model weights + KV cache.
MAX_BATCH_TOKENS = 96_000

# Number of documents to batch together in cross-document mode
CROSS_DOC_BATCH_SIZE = 8

# Maximum page text to feed to the map step (per page)
MAX_PAGE_CHARS = 6000

# Minimum characters for a page to be worth summarizing
MIN_PAGE_CHARS = 10

# Max retries for individual failed prompts
MAX_PROMPT_RETRIES = 2

# Max model context length (must match vLLM's max_model_len).
# REDUCE prompts exceeding this are handled via hierarchical reduce.
MAX_MODEL_TOKENS = 32_768

# Safety margin for output tokens in REDUCE step
REDUCE_OUTPUT_MARGIN = 1_000

# Minimum characters for a page to be worth extracting occupations from
MIN_CHARS_OCCUPATION = 50

# Max pages per document — above this threshold, uses _deep_hierarchical_reduce
# instead of _hierarchical_reduce (single-level). Deep reduce processes one group
# prompt at a time to avoid OOM in vLLM KV cache.
MAX_DOCUMENT_PAGES = 5000

# File IDs to skip entirely (known to crash vLLM / cause OOM).
# EFTA01661868 (7778 pages) was here but is now handled by _deep_hierarchical_reduce.
SKIP_FILE_IDS: set[str] = set()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

MAP_PROMPT = """You are summarizing page {page_number} of a document ({total_pages} pages total).

Rules:
- Summarize ONLY what is stated on this page. Do not infer or speculate.
- Write 1-2 factual sentences for the summary.
- If the page is blank or a cover page, set summary to "Cover page" or "Blank page."
- If the text is garbled, set summary to "Text unclear, low OCR quality."
- Mention names only if they appear verbatim on this page.
- Extract professional occupations/titles only: when the text explicitly states someone's profession, title, or function (e.g. "Attorney", "Detective", "pilot", "housekeeper"), include it in occupation_mentions.
- Do NOT extract: honorifics (Mr., Mrs., Ms.), "Dr." unless clearly medical/academic, legal statuses (defendant, plaintiff, witness, victim), or relational terms (associate, girlfriend, friend).
- evidence_span MUST be copied exactly from the page text.
- person_name is the person WHO HOLDS the occupation (the professional), not the client, employer, or subject.
- CRITICAL: "Epstein's lawyer" → do NOT set person_name="Epstein". Set person_name to the lawyer's name if present, otherwise null.
- "Epstein's lawyer Alan Dershowitz" → person_name: "Alan Dershowitz"
- "X, Epstein's lawyer" → person_name: "X"
- "Detective Recarey investigated Epstein" → person_name: "Recarey" (NOT Epstein)
- Do NOT attribute occupation to the person who possesses/employs/is served by the professional.
- person_name should be the name as written, or null if no specific person.

PAGE {page_number} TEXT:
{page_text}

Respond with ONLY valid JSON:
{{"summary": "1-2 sentence summary", "occupation_mentions": [{{"occupation": "lawyer", "surface_form": "Attorney", "person_name": "Alan Dershowitz", "confidence": 0.95, "evidence_span": "Attorney Alan Dershowitz"}}]}}

If no occupations, use empty array. Output JSON now."""

SHORT_MULTI_PROMPT = """You are writing a short summary of a document based on page-by-page summaries.

Rules:
- Only mention people from the ALLOWED LIST below.
- If the document refers to others not on the list, say "an unidentified person."
- Write 1-2 sentences (max 60 words).
- State the document type, main subject, and key individuals.
- When relevant, mention people's occupations naturally (e.g. "attorney Alan Dershowitz").
- Summarize only what is stated. No allegations, no "implied" wrongdoing.

ALLOWED PEOPLE (name — role):
{names_list}

DOCUMENT INFO:
{doc_info}

PAGE SUMMARIES:
{page_summaries}

Write the short summary now (1-2 sentences, max 60 words)."""

LONG_MULTI_PROMPT = """You are writing a detailed summary of a document based on page-by-page summaries.

Rules:
- Only mention people from the ALLOWED LIST below.
- If the document refers to others not on the list, say "an unidentified person."
- Write a cohesive paragraph of 150-300 words for the summary. Do NOT use bullet points or lists.
- Cite page numbers where relevant, e.g. "(p. 12)".
- When relevant, mention people's occupations (e.g. "attorney Alan Dershowitz").
- No allegations, no "implied" wrongdoing. Summarize only stated facts.
- If any pages had garbled text, note this.
- For document_type, choose ONE from: {document_types}
- For date, use YYYY-MM-DD if exact, YYYY-MM or YYYY if approximate, or "undated" if unknown.

ALLOWED PEOPLE (name — role):
{names_list}

DOCUMENT INFO:
{doc_info}

PAGE SUMMARIES:
{page_summaries}

Respond with ONLY valid JSON:
{{"summary": "A cohesive paragraph summarizing the document content...", "document_type": "letter", "date": "2005-03-15"}}

Output JSON now."""

# Fallback for single-page documents (no map step needed)
SHORT_SINGLE_PROMPT = """You are summarizing one document.

Rules:
- Only mention people from the ALLOWED LIST. Others: "an unidentified person."
- Write 1-2 sentences (max 60 words). State document type, subject, key individuals.
- When relevant, mention occupations naturally.
- Summarize only what is stated. No allegations.
- Extract professional occupations/titles only (e.g. "lawyer", "pilot", "detective").
- Do NOT extract: honorifics (Mr., Mrs.), "Dr." unless medical/academic, legal statuses (defendant, witness), or relational terms (associate, girlfriend).
- evidence_span MUST be copied exactly from the document text.
- person_name is the person WHO HOLDS the occupation (the professional), not the client, employer, or subject.
- CRITICAL: "Epstein's lawyer" → do NOT set person_name="Epstein". Set person_name to the lawyer's name if present, otherwise null.
- "Epstein's lawyer Alan Dershowitz" → person_name: "Alan Dershowitz"
- "X, Epstein's lawyer" → person_name: "X"
- "Detective Recarey investigated Epstein" → person_name: "Recarey" (NOT Epstein)
- Do NOT attribute occupation to the person who possesses/employs/is served by the professional.

ALLOWED PEOPLE (name — role):
{names_list}

DOCUMENT TEXT:
{text}

Respond with ONLY valid JSON:
{{"summary": "1-2 sentence summary (max 60 words)", "occupation_mentions": [{{"occupation": "lawyer", "surface_form": "Attorney", "person_name": "Alan Dershowitz", "confidence": 0.95, "evidence_span": "Attorney Alan Dershowitz"}}]}}

If no occupations, use empty array. Output JSON now."""

LONG_SINGLE_PROMPT = """You are summarizing one document in detail.

Rules:
- Only mention people from the ALLOWED LIST. Others: "an unidentified person."
- Write a cohesive paragraph of 150-300 words. Do NOT use bullet points or lists.
- When relevant, mention occupations naturally.
- No allegations. Summarize only stated facts.
- If text is garbled, note this.
- Extract professional occupations/titles only (e.g. "lawyer", "pilot", "detective").
- Do NOT extract: honorifics (Mr., Mrs.), "Dr." unless medical/academic, legal statuses (defendant, witness), or relational terms (associate, girlfriend).
- evidence_span MUST be copied exactly from the document text.
- person_name is the person WHO HOLDS the occupation (the professional), not the client, employer, or subject.
- CRITICAL: "Epstein's lawyer" → do NOT set person_name="Epstein". Set person_name to the lawyer's name if present, otherwise null.
- "Epstein's lawyer Alan Dershowitz" → person_name: "Alan Dershowitz"
- "X, Epstein's lawyer" → person_name: "X"
- "Detective Recarey investigated Epstein" → person_name: "Recarey" (NOT Epstein)
- Do NOT attribute occupation to the person who possesses/employs/is served by the professional.
- For document_type, choose ONE from: {document_types}
- For date, use YYYY-MM-DD if exact, YYYY-MM or YYYY if approximate, or "undated" if unknown.

ALLOWED PEOPLE (name — role):
{names_list}

DOCUMENT TEXT:
{text}

Respond with ONLY valid JSON:
{{"summary": "A cohesive paragraph summarizing the document content...", "document_type": "letter", "date": "2005-03-15", "occupation_mentions": [{{"occupation": "lawyer", "surface_form": "Attorney", "person_name": "Alan Dershowitz", "confidence": 0.95, "evidence_span": "Attorney Alan Dershowitz"}}]}}

If no occupations, use empty array. Output JSON now."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_text_hash(text: str) -> str:
    """SHA-256 hash of document text for cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def check_names_cleaned(names_dir: Path) -> list[str]:
    """Check if names directory has uncleaned files (raw without clean).

    Returns list of file IDs that have _names.json but no _names_clean.json.
    Empty list means everything is clean (or no names files at all).
    """
    if not names_dir.exists():
        return []

    raw_files = set()
    clean_files = set()
    for f in names_dir.iterdir():
        if f.name.endswith("_names_clean.json"):
            clean_files.add(f.name.replace("_names_clean.json", ""))
        elif f.name.endswith("_names.json"):
            raw_files.add(f.name.replace("_names.json", ""))

    return sorted(raw_files - clean_files)


def load_names_for_document(names_dir: Path, file_id: str) -> tuple[list[dict], dict]:
    """Load extracted names from the NER pipeline for a document.

    Prefers _names_clean.json, falls back to _names.json.
    """
    names_file = names_dir / f"{file_id}_names_clean.json"
    if not names_file.exists():
        names_file = names_dir / f"{file_id}_names.json"
    if not names_file.exists():
        logger.warning(f"No names file for {file_id} — summary will have no grounded names")
        return [], {}

    try:
        with open(names_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Corrupt names file {names_file}: {e} — using empty names")
        return [], {}

    metadata = {}
    if isinstance(data, dict):
        raw_names = data.get("names", [])
        metadata = {k: v for k, v in data.items() if k != "names"}
    else:
        raw_names = data

    best = {}
    for entry in raw_names:
        name = entry.get("normalized_name") or entry.get("original_text", "")
        name = name.strip()
        if not name:
            continue
        role = entry.get("role", "other")
        if role not in ROLE_PRIORITY:
            role = "mentioned"
        priority = ROLE_PRIORITY[role]
        key = name.lower()
        if key not in best or priority < best[key]["priority"]:
            best[key] = {"name": name, "role": role, "priority": priority}

    return sorted(best.values(), key=lambda x: (x["priority"], x["name"])), metadata


def format_names_list(names: list[dict]) -> str:
    if not names:
        return "(none)"
    return "\n".join(f"- {n['name']} — {n['role']}" for n in names)


def load_document_data(file_path: Path) -> dict:
    """Load full document data from a text JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_pages(data: dict) -> list[dict]:
    """Extract page list from document data."""
    pages = data.get("pages", [])
    if not pages:
        # Fallback: treat full_text as a single page
        full_text = data.get("full_text", "")
        if full_text:
            return [{"page_number": 1, "text": full_text}]
    return pages


def get_full_text(data: dict) -> str:
    """Get concatenated text from document data."""
    if "full_text" in data and data["full_text"]:
        return data["full_text"]
    pages = data.get("pages", [])
    return "\n\n".join(p.get("text", "").strip() for p in pages if p.get("text", "").strip())


def build_doc_info(file_id: str, data: dict, names_metadata: dict) -> str:
    """Build document info header for reduce prompts."""
    parts = [f"File ID: {file_id}"]
    doc_type = names_metadata.get("document_type")
    if doc_type:
        parts.append(f"Source type: {doc_type}")
    file_type = data.get("file_type", "")
    if file_type:
        parts.append(f"File format: {file_type}")
    total_pages = data.get("total_pages", 0)
    if total_pages:
        parts.append(f"Pages: {total_pages}")
    method = data.get("extraction_method", "")
    if method:
        parts.append(f"Extraction: {method}")
    return "\n".join(parts)


def find_text_files(directory: Path) -> list[Path]:
    """Find all text JSON files (excluding _names.json, _summary.json)."""
    files = []
    for f in sorted(directory.glob("*.json")):
        if f.name.endswith("_names.json") or f.name.endswith("_names_clean.json"):
            continue
        if f.name.endswith("_summary.json"):
            continue
        files.append(f)
    return files


# ---------------------------------------------------------------------------
# Occupation extraction helpers
# ---------------------------------------------------------------------------

def normalize_occupation(raw: str) -> str:
    """Normalize an occupation string using OCCUPATION_SYNONYMS."""
    lower = raw.strip().lower()
    return OCCUPATION_SYNONYMS.get(lower, lower)


def parse_structured_response(
    response: str,
    page_text: str | None = None,
    allowed_names: set[str] | None = None,
) -> dict:
    """Parse MAP/DIRECT JSON output with occupation_mentions.

    Returns dict with 'summary' and 'occupation_mentions' keys.
    Falls back to plain text summary with empty mentions on parse failure.
    """
    text = str(response).strip() if response else ""
    result = {"summary": "", "occupation_mentions": []}

    if not text:
        return result

    # Try JSON parse
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from surrounding text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    if parsed is None or not isinstance(parsed, dict):
        # Fallback: treat entire response as summary text
        result["summary"] = text
        return result

    result["summary"] = str(parsed.get("summary", "")).strip()
    # Extract document_type and date if present (from LONG_SINGLE responses)
    if "document_type" in parsed:
        result["document_type"] = str(parsed["document_type"]).strip() or None
    if "date" in parsed:
        result["document_date"] = str(parsed["date"]).strip() or None
    raw_mentions = parsed.get("occupation_mentions", [])

    if not isinstance(raw_mentions, list):
        return result

    page_text_lower = page_text.lower() if page_text else None

    for mention in raw_mentions:
        if not isinstance(mention, dict):
            continue

        occupation = mention.get("occupation", "")
        surface_form = mention.get("surface_form", "")
        evidence_span = mention.get("evidence_span", "")

        if not occupation or not surface_form:
            continue

        # Validate evidence_span against page text
        if page_text and evidence_span:
            if evidence_span not in page_text:
                logger.debug(f"Dropping mention: evidence_span not in page text: {evidence_span!r}")
                continue

        # Normalize occupation
        canonical = normalize_occupation(occupation)

        # Possessive pattern detection: "Epstein's lawyer" with person_name="Epstein"
        # Only null when the possessor IS the attributed person (not a different person)
        person_name = mention.get("person_name")
        if person_name and evidence_span:
            pn_lower = person_name.lower()
            # Check if evidence_span has "{person_name}'s {occupation}" pattern
            # meaning the LLM incorrectly attributed the occupation to the possessor
            if re.search(rf"(?i)\b{re.escape(pn_lower)}'s\s+", evidence_span):
                person_name = None  # Possessive pattern — person_name is the client, not the professional

        # Validate person_name against allowed_names
        if person_name and allowed_names is not None:
            # Case-insensitive match
            matched = False
            for allowed in allowed_names:
                if person_name.lower() == allowed.lower():
                    person_name = allowed  # Use canonical casing
                    matched = True
                    break
            if not matched:
                person_name = None

        result["occupation_mentions"].append({
            "occupation": canonical,
            "surface_form": surface_form,
            "person_name": person_name,
            "confidence": float(mention.get("confidence", 0.0)),
            "evidence_span": evidence_span,
        })

    return result


def parse_long_multi_response(response: str) -> dict:
    """Parse LONG_MULTI JSON response: summary, document_type, date.

    Falls back to treating raw text as summary if JSON parse fails.
    Returns dict with 'summary', 'document_type', and 'document_date' keys.
    """
    text = str(response).strip() if response else ""
    result = {"summary": "", "document_type": None, "document_date": None}

    if not text:
        return result

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    if parsed is None or not isinstance(parsed, dict):
        result["summary"] = text
        return result

    result["summary"] = str(parsed.get("summary", "")).strip()
    if "document_type" in parsed:
        result["document_type"] = str(parsed["document_type"]).strip() or None
    if "date" in parsed:
        result["document_date"] = str(parsed["date"]).strip() or None

    return result


def aggregate_occupation_summary(
    page_summaries: list[dict],
    allowed_names: set[str] | None = None,
) -> dict:
    """Build doc-level occupation_summary from page-level occupation_mentions.

    Pure Python aggregation, no LLM call.
    """
    by_person: dict[str, set[str]] = {}
    by_occupation: dict[str, int] = {}
    unlinked: set[str] = set()

    for ps in page_summaries:
        for mention in ps.get("occupation_mentions", []):
            occ = mention.get("occupation", "")
            person = mention.get("person_name")
            if not occ:
                continue

            by_occupation[occ] = by_occupation.get(occ, 0) + 1

            if person:
                if person not in by_person:
                    by_person[person] = set()
                by_person[person].add(occ)
            else:
                unlinked.add(occ)

    return {
        "by_person": {k: sorted(v) for k, v in sorted(by_person.items())},
        "by_occupation": dict(sorted(by_occupation.items(), key=lambda x: -x[1])),
        "unlinked": sorted(unlinked),
    }


# ---------------------------------------------------------------------------
# Map step: per-page summaries
# ---------------------------------------------------------------------------

def _is_page_skippable(page: dict) -> bool:
    """Check if a page should be skipped (blank, photo-only, too short)."""
    text = page.get("text", "").strip()
    if not text or len(text) < MIN_PAGE_CHARS:
        return True
    # Skip photo-only pages (no extractable text)
    if page.get("text_source") == "none":
        return True
    return False


def build_page_prompt(page_num: int, page_text: str, total_pages: int) -> str | None:
    """Build a MAP prompt for one page, or None if blank."""
    text = page_text.strip()
    if not text or len(text) < MIN_PAGE_CHARS:
        return None

    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n[... truncated ...]"

    return MAP_PROMPT.format(
        page_number=page_num,
        total_pages=total_pages,
        page_text=text,
    )


def _estimate_prompt_tokens(prompt: str) -> int:
    """Conservative token estimate: ~3 chars per token.

    OCR text with special characters and punctuation tokenizes at ~2.5-3.5
    chars/token (not the typical ~4 for clean English).  Using 3 prevents
    prompt-too-long crashes on the remaining 3% of large documents.
    Chat template overhead (~200 tokens) is also absorbed by this margin.
    """
    return len(prompt) // 3


def _build_token_aware_chunks(
    prompts: list[str], max_count: int, max_tokens: int
) -> list[list[tuple[int, str]]]:
    """Split prompts into chunks bounded by both count and token budget.

    Returns list of chunks, where each chunk is [(original_index, prompt), ...].
    """
    chunks = []
    current = []
    current_tokens = 0

    for i, prompt in enumerate(prompts):
        est = _estimate_prompt_tokens(prompt)
        if current and (len(current) >= max_count or current_tokens + est > max_tokens):
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append((i, prompt))
        current_tokens += est

    if current:
        chunks.append(current)
    return chunks


def _chunked_batch(
    prompts: list[str], batch_llm_fn, chunk_size: int = 0, max_batch_tokens: int = 0,
) -> list[str]:
    """Run batch_llm_fn in chunks bounded by count AND token budget.

    Detects empty/failed responses and retries them individually (up to
    MAX_PROMPT_RETRIES times) at prompt granularity.
    """
    if not prompts:
        return []

    cs = chunk_size or BATCH_CHUNK_SIZE
    mbt = max_batch_tokens or MAX_BATCH_TOKENS
    chunks = _build_token_aware_chunks(prompts, max_count=cs, max_tokens=mbt)

    all_responses = [None] * len(prompts)

    for chunk in chunks:
        indices = [idx for idx, _ in chunk]
        chunk_prompts = [p for _, p in chunk]
        try:
            responses = batch_llm_fn(chunk_prompts)
        except (ValueError, RuntimeError) as e:
            err = str(e).lower()
            if "longer than" not in err and "too long" not in err and "max" not in err:
                raise
            # Prompt exceeds model context — truncate and retry each individually
            logger.warning(f"Chunk of {len(chunk_prompts)} prompts too long, truncating individually: {e}")
            responses = []
            max_chars = MAX_MODEL_TOKENS * 3  # conservative char limit matching // 3 estimate
            for cp in chunk_prompts:
                if len(cp) > max_chars:
                    cp = cp[:max_chars] + "\n[... truncated ...]"
                try:
                    r = batch_llm_fn([cp])
                    responses.append(r[0] if r else "")
                except (ValueError, RuntimeError):
                    logger.warning(f"Prompt still too long after truncation ({len(cp)} chars), skipping")
                    responses.append("")

        for idx, response in zip(indices, responses):
            all_responses[idx] = response

    # Per-item retry for empty/failed outputs
    failed = [
        i for i, r in enumerate(all_responses)
        if r is None or not str(r).strip()
    ]
    if failed:
        logger.warning(f"Retrying {len(failed)}/{len(prompts)} empty responses individually")
        for attempt in range(1, MAX_PROMPT_RETRIES + 1):
            still_failed = []
            for idx in failed:
                try:
                    retry_resp = batch_llm_fn([prompts[idx]])
                    if retry_resp and str(retry_resp[0]).strip():
                        all_responses[idx] = retry_resp[0]
                    else:
                        still_failed.append(idx)
                except Exception as e:
                    logger.warning(f"Retry {attempt} failed for prompt {idx}: {e}")
                    still_failed.append(idx)
            failed = still_failed
            if not failed:
                break

    # Replace any remaining None with empty string
    return [r if r is not None else "" for r in all_responses]


def summarize_pages_batched(
    pages: list[dict],
    total_pages: int,
    batch_llm_fn,
    allowed_names: set[str] | None = None,
) -> list[dict]:
    """Summarize all pages in chunked batches to avoid OOM on large docs."""
    # Build prompts, tracking which pages need LLM vs are blank/photo
    prompts = []
    prompt_indices = []  # maps prompt position → page index
    page_nums = []
    page_texts = {}  # index → original page text for evidence validation

    for i, page in enumerate(pages):
        page_num = page.get("page_number", page.get("page", i + 1))
        page_nums.append(page_num)

        if _is_page_skippable(page):
            continue

        page_text = page.get("text", "")
        prompt = build_page_prompt(page_num, page_text, total_pages)
        if prompt is not None:
            prompts.append(prompt)
            prompt_indices.append(i)
            page_texts[i] = page_text

    skipped = len(pages) - len(prompts)
    if skipped > 0:
        logger.debug(f"Skipped {skipped}/{len(pages)} blank/photo pages")

    # Chunked batch call
    if prompts:
        responses = _chunked_batch(prompts, batch_llm_fn)
    else:
        responses = []

    # Assemble results with per-item error handling
    results = [{"page": pn, "summary": "Blank page.", "occupation_mentions": []} for pn in page_nums]
    for idx, response in zip(prompt_indices, responses):
        try:
            page_text = page_texts.get(idx)
            parsed = parse_structured_response(response, page_text, allowed_names)
            results[idx]["summary"] = parsed["summary"] or "Blank page."
            results[idx]["occupation_mentions"] = parsed["occupation_mentions"]
        except Exception as e:
            logger.warning(f"Failed to parse response for page {page_nums[idx]}: {e}")
            results[idx]["summary"] = "Summary generation failed."

    return results


# ---------------------------------------------------------------------------
# Reduce step: document-level summaries
# ---------------------------------------------------------------------------

def format_page_summaries(page_summaries: list[dict]) -> str:
    """Format page summaries for the reduce prompt."""
    lines = []
    for ps in page_summaries:
        lines.append(f"Page {ps['page']}: {ps['summary']}")
    return "\n".join(lines)


def _estimate_reduce_tokens(names: list[dict], doc_info: str, page_summaries: list[dict]) -> int:
    """Estimate token count for the LONG reduce prompt (always the larger one)."""
    names_list = format_names_list(names)
    ps_text = format_page_summaries(page_summaries)
    prompt = LONG_MULTI_PROMPT.format(
        names_list=names_list, doc_info=doc_info, page_summaries=ps_text,
        document_types=", ".join(DOCUMENT_TYPES),
    )
    return _estimate_prompt_tokens(prompt)


def _hierarchical_reduce(
    page_summaries: list[dict],
    names: list[dict],
    doc_info: str,
    batch_llm_fn,
    max_tokens: int,
) -> list[dict]:
    """Reduce page summaries in groups until they fit the model context.

    Splits page summaries into groups, reduces each group to a single summary,
    then uses those group summaries as input to the final reduce.
    """
    # Determine group size: how many page summaries fit in the budget
    names_list = format_names_list(names)
    overhead = _estimate_prompt_tokens(
        LONG_MULTI_PROMPT.format(
            names_list=names_list, doc_info=doc_info, page_summaries="",
            document_types=", ".join(DOCUMENT_TYPES),
        )
    )
    available = max_tokens - overhead - REDUCE_OUTPUT_MARGIN
    if available <= 0:
        available = max_tokens // 2

    # Estimate tokens per page summary line
    total_ps_tokens = _estimate_prompt_tokens(format_page_summaries(page_summaries))
    tokens_per_page = max(total_ps_tokens // max(len(page_summaries), 1), 1)
    group_size = max(available // tokens_per_page, 2)

    # Cap group size so each MAP prompt stays within model context
    map_overhead = _estimate_prompt_tokens(
        MAP_PROMPT.format(page_number="1-999", total_pages=len(page_summaries), page_text="")
    )
    map_available = max_tokens - map_overhead - REDUCE_OUTPUT_MARGIN
    max_group_by_map = max(map_available // tokens_per_page, 2)
    group_size = min(group_size, max_group_by_map)

    logger.info(
        f"Hierarchical reduce: {len(page_summaries)} pages → groups of {group_size} "
        f"(~{total_ps_tokens} tokens total, budget {available})"
    )

    # Phase 1: Reduce each group into an intermediate summary
    group_prompts = []
    group_ranges = []
    for start in range(0, len(page_summaries), group_size):
        group = page_summaries[start : start + group_size]
        ps_text = format_page_summaries(group)
        first_page = group[0]["page"]
        last_page = group[-1]["page"]
        group_ranges.append(f"pp. {first_page}-{last_page}")
        group_prompts.append(
            MAP_PROMPT.format(
                page_number=f"{first_page}-{last_page}",
                total_pages=len(page_summaries),
                page_text=ps_text,
            )
        )

    group_responses = _chunked_batch(group_prompts, batch_llm_fn)

    # Build condensed page summaries from group responses
    condensed = []
    for i, (response, page_range) in enumerate(zip(group_responses, group_ranges)):
        text = str(response).strip() if response else "Group summary unavailable."
        condensed.append({"page": page_range, "summary": text})

    return condensed


def _deep_hierarchical_reduce(
    page_summaries: list[dict],
    names: list[dict],
    doc_info: str,
    batch_llm_fn,
    max_tokens: int,
) -> list[dict]:
    """Multi-level recursive reduce for extremely large documents (5000+ pages).

    Unlike _hierarchical_reduce (single-level, used by normal pipeline), this
    function recurses until the condensed output fits within max_tokens.
    Caps group sizes to stay within MAP prompt context limits too.

    Used only by process_oversized_document(), never by the main pipeline.
    """
    MAX_DEPTH = 5
    return _deep_reduce_recursive(
        page_summaries, names, doc_info, batch_llm_fn, max_tokens, depth=0, max_depth=MAX_DEPTH,
    )


def _deep_reduce_recursive(
    page_summaries: list[dict],
    names: list[dict],
    doc_info: str,
    batch_llm_fn,
    max_tokens: int,
    depth: int,
    max_depth: int,
) -> list[dict]:
    """Inner recursive implementation for _deep_hierarchical_reduce."""
    names_list = format_names_list(names)
    overhead = _estimate_prompt_tokens(
        LONG_MULTI_PROMPT.format(
            names_list=names_list, doc_info=doc_info, page_summaries="",
            document_types=", ".join(DOCUMENT_TYPES),
        )
    )
    available = max_tokens - overhead - REDUCE_OUTPUT_MARGIN
    if available <= 0:
        available = max_tokens // 2

    # Estimate tokens per page summary line
    total_ps_tokens = _estimate_prompt_tokens(format_page_summaries(page_summaries))
    tokens_per_page = max(total_ps_tokens // max(len(page_summaries), 1), 1)
    group_size = max(available // tokens_per_page, 2)

    # Cap group size so each MAP prompt stays within model context
    map_overhead = _estimate_prompt_tokens(
        MAP_PROMPT.format(page_number="1-999", total_pages=len(page_summaries), page_text="")
    )
    map_available = max_tokens - map_overhead - REDUCE_OUTPUT_MARGIN
    max_group_by_map = max(map_available // tokens_per_page, 2)
    group_size = min(group_size, max_group_by_map)

    n_groups = (len(page_summaries) + group_size - 1) // group_size

    logger.info(
        f"Deep hierarchical reduce (depth={depth}): {len(page_summaries)} summaries → "
        f"{n_groups} groups of {group_size} (~{total_ps_tokens} tokens, budget {available})"
    )

    # Reduce each group into an intermediate summary
    group_prompts = []
    group_ranges = []
    for start in range(0, len(page_summaries), group_size):
        group = page_summaries[start : start + group_size]
        ps_text = format_page_summaries(group)
        first_page = group[0]["page"]
        last_page = group[-1]["page"]
        group_ranges.append(f"pp. {first_page}-{last_page}")
        group_prompts.append(
            MAP_PROMPT.format(
                page_number=f"{first_page}-{last_page}",
                total_pages=len(page_summaries),
                page_text=ps_text,
            )
        )

    # Use conservative batching for deep reduce: each group prompt can be ~29K
    # tokens, so limit to 1 prompt at a time to avoid OOM in vLLM KV cache.
    group_responses = _chunked_batch(
        group_prompts, batch_llm_fn, chunk_size=1, max_batch_tokens=MAX_MODEL_TOKENS,
    )

    condensed = []
    for i, (response, page_range) in enumerate(zip(group_responses, group_ranges)):
        text = str(response).strip() if response else "Group summary unavailable."
        condensed.append({"page": page_range, "summary": text})

    # Check if condensed fits; if not, recurse
    condensed_tokens = _estimate_reduce_tokens(names, doc_info, condensed)
    if condensed_tokens > max_tokens and depth < max_depth and len(condensed) > 1:
        logger.info(
            f"Condensed still too large ({condensed_tokens} tokens > {max_tokens}), "
            f"recursing (depth {depth + 1})"
        )
        return _deep_reduce_recursive(
            condensed, names, doc_info, batch_llm_fn,
            max_tokens=max_tokens, depth=depth + 1, max_depth=max_depth,
        )

    return condensed


def reduce_summaries(
    page_summaries: list[dict],
    names: list[dict],
    doc_info: str,
    batch_llm_fn,
    *,
    document_types: str = "",
) -> dict:
    """Generate short + long summaries from page summaries (batched).

    If the reduce prompt would exceed the model context, automatically
    applies hierarchical reduce (groups → final) to stay within limits.

    Returns dict with keys: short, long, document_type, document_date.
    """
    doc_types = document_types or ", ".join(DOCUMENT_TYPES)

    # Check if reduce prompt fits in model context
    est_tokens = _estimate_reduce_tokens(names, doc_info, page_summaries)
    token_limit = MAX_MODEL_TOKENS - REDUCE_OUTPUT_MARGIN

    if est_tokens > token_limit:
        logger.warning(
            f"REDUCE prompt too large ({est_tokens} tokens > {token_limit} limit). "
            f"Using hierarchical reduce for {len(page_summaries)} pages."
        )
        page_summaries = _hierarchical_reduce(
            page_summaries, names, doc_info, batch_llm_fn,
            max_tokens=token_limit,
        )

    names_list = format_names_list(names)
    ps_text = format_page_summaries(page_summaries)

    prompts = [
        SHORT_MULTI_PROMPT.format(
            names_list=names_list,
            doc_info=doc_info,
            page_summaries=ps_text,
        ),
        LONG_MULTI_PROMPT.format(
            names_list=names_list,
            doc_info=doc_info,
            page_summaries=ps_text,
            document_types=doc_types,
        ),
    ]

    try:
        short_raw, long_raw = batch_llm_fn(prompts)
    except (ValueError, RuntimeError) as e:
        err = str(e).lower()
        if "too long" in err or "exceed" in err or "max" in err:
            logger.error(
                f"REDUCE prompt still too large after hierarchical reduce "
                f"({_estimate_prompt_tokens(prompts[1])} tokens). Truncating."
            )
            # Last resort: truncate page summaries text
            max_ps_chars = (token_limit - 500) * 4  # rough reverse estimate
            ps_text_trunc = ps_text[:max_ps_chars] + "\n[... truncated ...]"
            prompts = [
                SHORT_MULTI_PROMPT.format(
                    names_list=names_list, doc_info=doc_info, page_summaries=ps_text_trunc
                ),
                LONG_MULTI_PROMPT.format(
                    names_list=names_list, doc_info=doc_info, page_summaries=ps_text_trunc,
                    document_types=doc_types,
                ),
            ]
            short_raw, long_raw = batch_llm_fn(prompts)
        else:
            raise

    long_parsed = parse_long_multi_response(long_raw)

    return {
        "short": _extract_short_summary(short_raw),
        "long": long_parsed["summary"] or str(long_raw).strip(),
        "document_type": long_parsed["document_type"],
        "document_date": long_parsed["document_date"],
    }


def direct_summaries(
    text: str,
    names: list[dict],
    batch_llm_fn,
    allowed_names: set[str] | None = None,
    *,
    document_types: str = "",
) -> dict:
    """Generate summaries directly for short (1-page) documents (batched).

    Returns dict with keys: short, long, occupation_mentions, document_type, document_date.
    """
    doc_types = document_types or ", ".join(DOCUMENT_TYPES)
    names_list = format_names_list(names)

    original_text = text
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n[... truncated ...]"

    prompts = [
        SHORT_SINGLE_PROMPT.format(names_list=names_list, text=text),
        LONG_SINGLE_PROMPT.format(names_list=names_list, text=text, document_types=doc_types),
    ]
    short_raw, long_raw = batch_llm_fn(prompts)

    short_parsed = parse_structured_response(short_raw, original_text, allowed_names)
    long_parsed = parse_structured_response(long_raw, original_text, allowed_names)

    # Merge occupation mentions from both responses, dedup by (occupation, person_name, evidence_span)
    seen = set()
    all_mentions = []
    for mention in short_parsed["occupation_mentions"] + long_parsed["occupation_mentions"]:
        key = (mention["occupation"], mention.get("person_name"), mention.get("evidence_span"))
        if key not in seen:
            seen.add(key)
            all_mentions.append(mention)

    return {
        "short": short_parsed["summary"],
        "long": long_parsed["summary"],
        "occupation_mentions": all_mentions,
        "document_type": long_parsed.get("document_type"),
        "document_date": long_parsed.get("document_date"),
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def process_document(file_path: Path, names_dir: Path, batch_llm_fn) -> dict | None:
    """Generate summaries for a single document using map→reduce.

    For 1-page docs: direct summary (no map step).
    For multi-page: map per page (batched) → reduce to document-level (batched).
    """
    t0 = time.monotonic()
    file_id = file_path.stem
    data = load_document_data(file_path)
    full_text = get_full_text(data)

    if not full_text or len(full_text.strip()) < 20:
        logger.debug(f"Skipping {file_id}: text too short ({len(full_text)} chars)")
        return None

    # Load grounded names
    names, names_metadata = load_names_for_document(names_dir, file_id)
    doc_info = build_doc_info(file_id, data, names_metadata)
    text_hash = compute_text_hash(full_text)
    doc_types_str = ", ".join(DOCUMENT_TYPES)

    pages = get_pages(data)
    total_prompts = 0
    allowed_names_set = {n["name"] for n in names}
    doc_type = None
    doc_date = None

    if len(pages) <= 1:
        # Single page: direct summary, no map step
        page_text = pages[0].get("text", full_text) if pages else full_text
        result = direct_summaries(
            page_text, names, batch_llm_fn, allowed_names_set,
            document_types=doc_types_str,
        )
        short, long = result["short"], result["long"]
        direct_occ = result["occupation_mentions"]
        doc_type = result["document_type"]
        doc_date = result["document_date"]
        page_summaries = []
        # Build a synthetic page entry for aggregation
        occ_pages = [{"page": 1, "occupation_mentions": direct_occ}] if direct_occ else []
        total_prompts = 2
    else:
        # Multi-page: map→reduce
        total_pages = len(pages)
        non_blank = sum(1 for p in pages if not _is_page_skippable(p))

        # MAP: all pages in one batched call
        t_map = time.monotonic()
        page_summaries = summarize_pages_batched(
            pages, total_pages, batch_llm_fn, allowed_names_set
        )
        map_elapsed = time.monotonic() - t_map

        # REDUCE: short + long in one batched call
        t_reduce = time.monotonic()
        result = reduce_summaries(
            page_summaries, names, doc_info, batch_llm_fn,
            document_types=doc_types_str,
        )
        short, long = result["short"], result["long"]
        doc_type = result["document_type"]
        doc_date = result["document_date"]
        reduce_elapsed = time.monotonic() - t_reduce

        occ_pages = page_summaries
        total_prompts = non_blank + 2
        total_elapsed = time.monotonic() - t0
        est_tokens = _estimate_tokens(full_text)
        pages_per_sec = non_blank / map_elapsed if map_elapsed > 0 else 0

        logger.info(
            f"{file_id}: {total_pages}p ({non_blank} non-blank) | "
            f"batch={non_blank}+2 prompts | ~{est_tokens:,} input tok | "
            f"map={map_elapsed:.1f}s reduce={reduce_elapsed:.1f}s total={total_elapsed:.1f}s | "
            f"{pages_per_sec:.1f} pages/s"
        )

    return {
        "file_id": file_id,
        "model_version": MODEL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "text_hash": text_hash,
        "summary_short": short,
        "summary_long": long,
        "page_summaries": page_summaries,
        "allowed_names": [{"name": n["name"], "role": n["role"]} for n in names],
        "occupation_summary": aggregate_occupation_summary(occ_pages, allowed_names_set),
        "document_type": doc_type,
        "document_date": doc_date,
    }


def prepare_document(file_path: Path, names_dir: Path) -> dict | None:
    """Load and prepare a document for summary generation.

    Returns a context dict with all data needed for MAP and REDUCE steps,
    or None if the document should be skipped.
    """
    file_id = file_path.stem
    data = load_document_data(file_path)
    full_text = get_full_text(data)

    if not full_text or len(full_text.strip()) < 20:
        logger.debug(f"Skipping {file_id}: text too short ({len(full_text)} chars)")
        return None

    names, names_metadata = load_names_for_document(names_dir, file_id)
    doc_info = build_doc_info(file_id, data, names_metadata)
    text_hash = compute_text_hash(full_text)
    pages = get_pages(data)

    return {
        "file_id": file_id,
        "file_path": file_path,
        "data": data,
        "full_text": full_text,
        "names": names,
        "names_metadata": names_metadata,
        "doc_info": doc_info,
        "text_hash": text_hash,
        "pages": pages,
        "single_page": len(pages) <= 1,
        # Filled during processing
        "page_summaries": [],
        "summary_short": "",
        "summary_long": "",
    }


def process_documents_batched(
    files: list[Path],
    names_dir: Path,
    batch_llm_fn,
    output_dir: Path,
    chunk_size: int = 0,
) -> tuple[int, int]:
    """Process multiple documents with cross-document prompt batching.

    Instead of MAP→REDUCE per document, collects prompts across documents
    and sends them in larger batches for better GPU utilization with vLLM.

    Returns (processed_count, skipped_count).
    """
    processed = 0
    skipped = 0

    for batch_start in range(0, len(files), CROSS_DOC_BATCH_SIZE):
        batch_files = files[batch_start : batch_start + CROSS_DOC_BATCH_SIZE]
        t_batch = time.monotonic()

        # --- Phase 1: Prepare all documents ---
        doc_contexts = []
        for f in batch_files:
            ctx = prepare_document(f, names_dir)
            if ctx is not None:
                doc_contexts.append(ctx)
            else:
                skipped += 1

        if not doc_contexts:
            continue

        # --- Phase 2: Collect all MAP prompts across documents ---
        all_map_prompts = []
        prompt_routing = []  # (doc_idx, page_idx) for each prompt

        for doc_idx, ctx in enumerate(doc_contexts):
            if ctx["single_page"]:
                continue  # Single-page docs skip MAP, go straight to REDUCE
            total_pages = len(ctx["pages"])
            for page_idx, page in enumerate(ctx["pages"]):
                if _is_page_skippable(page):
                    continue
                page_num = page.get("page_number", page.get("page", page_idx + 1))
                prompt = build_page_prompt(page_num, page.get("text", ""), total_pages)
                if prompt is not None:
                    all_map_prompts.append(prompt)
                    prompt_routing.append((doc_idx, page_idx))

        # --- Phase 3: Batch MAP call ---
        t_map = time.monotonic()
        if all_map_prompts:
            map_responses = _chunked_batch(all_map_prompts, batch_llm_fn, chunk_size)
            assert len(map_responses) == len(all_map_prompts), (
                f"MAP response count mismatch: {len(map_responses)} != {len(all_map_prompts)}"
            )
        else:
            map_responses = []
        map_time = time.monotonic() - t_map

        # Route MAP responses back — build page_summaries per document
        # Build page_texts mapping for evidence validation
        page_texts_map = {}  # (doc_idx, page_idx) → page text
        for doc_idx, ctx in enumerate(doc_contexts):
            if ctx["single_page"]:
                continue
            page_nums = []
            for page_idx, page in enumerate(ctx["pages"]):
                pn = page.get("page_number", page.get("page", page_idx + 1))
                page_nums.append(pn)
                page_texts_map[(doc_idx, page_idx)] = page.get("text", "")
            ctx["page_summaries"] = [
                {"page": pn, "summary": "Blank page.", "occupation_mentions": []}
                for pn in page_nums
            ]

        for (doc_idx, page_idx), response in zip(prompt_routing, map_responses):
            ctx = doc_contexts[doc_idx]
            allowed_names_set = {n["name"] for n in ctx["names"]}
            page_text = page_texts_map.get((doc_idx, page_idx))
            parsed = parse_structured_response(response, page_text, allowed_names_set)
            ctx["page_summaries"][page_idx]["summary"] = parsed["summary"] or "Blank page."
            ctx["page_summaries"][page_idx]["occupation_mentions"] = parsed["occupation_mentions"]

        # --- Phase 4: Pre-reduce oversized docs, then collect REDUCE prompts ---
        token_limit = MAX_MODEL_TOKENS - REDUCE_OUTPUT_MARGIN
        for doc_idx, ctx in enumerate(doc_contexts):
            if ctx["single_page"] or not ctx["page_summaries"]:
                continue
            est = _estimate_reduce_tokens(ctx["names"], ctx["doc_info"], ctx["page_summaries"])
            if est > token_limit:
                n_pages = len(ctx["page_summaries"])
                if n_pages > MAX_DOCUMENT_PAGES:
                    logger.warning(
                        f"{ctx['file_id']}: REDUCE too large ({est} tokens > {token_limit}). "
                        f"Deep hierarchical reduce for {n_pages} pages (oversized doc)."
                    )
                    ctx["page_summaries"] = _deep_hierarchical_reduce(
                        ctx["page_summaries"], ctx["names"], ctx["doc_info"],
                        batch_llm_fn, max_tokens=token_limit,
                    )
                else:
                    logger.warning(
                        f"{ctx['file_id']}: REDUCE too large ({est} tokens > {token_limit}). "
                        f"Hierarchical reduce for {n_pages} pages."
                    )
                    ctx["page_summaries"] = _hierarchical_reduce(
                        ctx["page_summaries"], ctx["names"], ctx["doc_info"],
                        batch_llm_fn, max_tokens=token_limit,
                    )

        all_reduce_prompts = []
        reduce_routing = []  # (doc_idx, "short"|"long")
        doc_types_str = ", ".join(DOCUMENT_TYPES)

        for doc_idx, ctx in enumerate(doc_contexts):
            names_list = format_names_list(ctx["names"])

            if ctx["single_page"]:
                page_text = ctx["pages"][0].get("text", ctx["full_text"]) if ctx["pages"] else ctx["full_text"]
                if len(page_text) > MAX_PAGE_CHARS:
                    page_text = page_text[:MAX_PAGE_CHARS] + "\n[... truncated ...]"
                all_reduce_prompts.append(
                    SHORT_SINGLE_PROMPT.format(names_list=names_list, text=page_text)
                )
                all_reduce_prompts.append(
                    LONG_SINGLE_PROMPT.format(
                        names_list=names_list, text=page_text,
                        document_types=doc_types_str,
                    )
                )
            else:
                ps_text = format_page_summaries(ctx["page_summaries"])
                all_reduce_prompts.append(
                    SHORT_MULTI_PROMPT.format(
                        names_list=names_list,
                        doc_info=ctx["doc_info"],
                        page_summaries=ps_text,
                    )
                )
                all_reduce_prompts.append(
                    LONG_MULTI_PROMPT.format(
                        names_list=names_list,
                        doc_info=ctx["doc_info"],
                        page_summaries=ps_text,
                        document_types=doc_types_str,
                    )
                )
            reduce_routing.append((doc_idx, "short"))
            reduce_routing.append((doc_idx, "long"))

        # --- Phase 5: Batch REDUCE call ---
        t_reduce = time.monotonic()
        reduce_responses = _chunked_batch(all_reduce_prompts, batch_llm_fn, chunk_size)
        assert len(reduce_responses) == len(all_reduce_prompts), (
            f"REDUCE response count mismatch: {len(reduce_responses)} != {len(all_reduce_prompts)}"
        )
        reduce_time = time.monotonic() - t_reduce

        for (doc_idx, kind), response in zip(reduce_routing, reduce_responses):
            ctx = doc_contexts[doc_idx]
            if ctx["single_page"]:
                # SINGLE prompts return JSON with occupation_mentions + document_type/date
                allowed_names_set = {n["name"] for n in ctx["names"]}
                original_text = ctx["pages"][0].get("text", ctx["full_text"]) if ctx["pages"] else ctx["full_text"]
                parsed = parse_structured_response(response, original_text, allowed_names_set)
                ctx[f"summary_{kind}"] = parsed["summary"]
                # Accumulate occupation mentions from both short and long
                if "direct_occupation_mentions" not in ctx:
                    ctx["direct_occupation_mentions"] = []
                ctx["direct_occupation_mentions"].extend(parsed["occupation_mentions"])
                # Extract document_type/date from long response
                if kind == "long":
                    ctx["document_type"] = parsed.get("document_type")
                    ctx["document_date"] = parsed.get("document_date")
            else:
                if kind == "long":
                    # LONG_MULTI returns JSON with summary + document_type + date
                    long_parsed = parse_long_multi_response(response)
                    ctx["summary_long"] = long_parsed["summary"] or str(response).strip()
                    ctx["document_type"] = long_parsed["document_type"]
                    ctx["document_date"] = long_parsed["document_date"]
                else:
                    # SHORT_MULTI returns free text (extract summary from possible JSON/noise)
                    ctx["summary_short"] = _extract_short_summary(response)

        # --- Phase 6: Write outputs ---
        for ctx in doc_contexts:
            # Build occupation_summary
            if ctx["single_page"]:
                # Dedup direct mentions
                seen = set()
                deduped = []
                for m in ctx.get("direct_occupation_mentions", []):
                    key = (m["occupation"], m.get("person_name"), m.get("evidence_span"))
                    if key not in seen:
                        seen.add(key)
                        deduped.append(m)
                occ_pages = [{"page": 1, "occupation_mentions": deduped}] if deduped else []
            else:
                occ_pages = ctx["page_summaries"]
            allowed_names_set = {n["name"] for n in ctx["names"]}
            occ_summary = aggregate_occupation_summary(occ_pages, allowed_names_set)

            result = {
                "file_id": ctx["file_id"],
                "model_version": MODEL_VERSION,
                "prompt_version": PROMPT_VERSION,
                "text_hash": ctx["text_hash"],
                "summary_short": ctx["summary_short"],
                "summary_long": ctx["summary_long"],
                "page_summaries": ctx["page_summaries"],
                "allowed_names": [{"name": n["name"], "role": n["role"]} for n in ctx["names"]],
                "occupation_summary": occ_summary,
                "document_type": ctx.get("document_type"),
                "document_date": ctx.get("document_date"),
            }
            output_file = output_dir / f"{ctx['file_id']}_summary.json"
            tmp_file = output_file.with_suffix(".tmp.json")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            try:
                os.rename(tmp_file, output_file)
            except FileNotFoundError:
                # Another worker's cleanup may have deleted our tmp file — rewrite directly
                logger.warning(f"{ctx['file_id']}: tmp file vanished, writing directly")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            processed += 1

        total_time = time.monotonic() - t_batch
        throughput = len(doc_contexts) / total_time if total_time > 0 else 0
        logger.info(
            f"batch_summary | docs={len(doc_contexts)} "
            f"map_prompts={len(all_map_prompts)} "
            f"reduce_prompts={len(all_reduce_prompts)} "
            f"map_time={map_time:.1f}s reduce_time={reduce_time:.1f}s "
            f"throughput={throughput:.1f} docs/s"
        )

    return processed, skipped


def is_cache_valid(existing_path: Path, text_hash: str) -> bool:
    """Check if an existing summary is still valid (same text + model + prompt)."""
    try:
        with open(existing_path) as f:
            existing = json.load(f)
        return (
            existing.get("text_hash") == text_hash
            and existing.get("model_version") == MODEL_VERSION
            and existing.get("prompt_version") == PROMPT_VERSION
        )
    except (json.JSONDecodeError, KeyError):
        return False


def create_llm_fn(model_name: str = "Qwen/Qwen3-32B-AWQ") -> tuple:
    """Create a batched LLM function using vLLM or transformers.

    Returns (batch_generate_fn, is_vllm) tuple.
    vLLM processes the batch in parallel using continuous batching for high GPU utilization.
    """
    global BATCH_CHUNK_SIZE

    try:
        from vllm import LLM, SamplingParams

        tp_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1"))
        quantization = "awq_marlin" if "awq" in model_name.lower() else None
        llm = LLM(model=model_name, max_model_len=32768, tensor_parallel_size=tp_size,
                   quantization=quantization, trust_remote_code=True)
        params = SamplingParams(temperature=0.3, max_tokens=700, repetition_penalty=1.05)

        # Get tokenizer for chat template (suppresses Qwen3 thinking mode)
        tokenizer = llm.get_tokenizer()

        def _apply_template(prompt: str) -> str:
            """Wrap prompt in chat template with thinking disabled."""
            messages = [{"role": "user", "content": prompt}]
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                # Older models (Qwen2.5) don't support enable_thinking
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

        # vLLM handles memory management internally — use larger chunks
        BATCH_CHUNK_SIZE = BATCH_CHUNK_SIZE_VLLM
        logger.info(f"vLLM loaded, BATCH_CHUNK_SIZE={BATCH_CHUNK_SIZE}")

        def batch_generate(prompts: list[str]) -> list[str]:
            templated = [_apply_template(p) for p in prompts]
            outputs = llm.generate(templated, params)
            return [_strip_thinking(o.outputs[0].text) for o in outputs]

        return batch_generate, True

    except ImportError:
        logger.warning("vLLM not available, falling back to transformers (sequential)")
        from transformers import pipeline as hf_pipeline

        pipe = hf_pipeline("text-generation", model=model_name, max_new_tokens=700)

        def batch_generate(prompts: list[str]) -> list[str]:
            results = []
            for prompt in prompts:
                result = pipe(prompt, max_new_tokens=700, temperature=0.3)
                raw = result[0]["generated_text"][len(prompt):]
                results.append(_strip_thinking(raw))
            return results

        return batch_generate, False


def main():
    parser = argparse.ArgumentParser(description="Generate document summaries (page-aware map→reduce)")
    parser.add_argument("--dataset", required=True, help="Dataset name (doj, house-oversight)")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", help="Output directory (default: data/processed/summaries/{dataset})")
    parser.add_argument("--names-dir", help="Names directory (default: data/processed/names_v2/{dataset})")
    parser.add_argument("--model", default="Qwen/Qwen3-32B-AWQ", help="LLM model name")
    parser.add_argument("--apply", action="store_true", help="Actually generate summaries")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--resume-from", help="Resume from this file_id (skip earlier files)")
    parser.add_argument("--file-list", help="Path to file with one file_id per line (process only these)")
    parser.add_argument("--force", action="store_true", help="Regenerate even if cached summary exists")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM for testing")

    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )

    text_dir = Path(args.data_dir) / "processed" / "text" / args.dataset
    if not text_dir.exists():
        logger.error(f"Text directory not found: {text_dir}")
        sys.exit(1)

    names_dir = Path(args.names_dir) if args.names_dir else (
        Path(args.data_dir) / "processed" / "names_v2" / args.dataset
    )
    if names_dir.exists():
        logger.info(f"Names grounding from: {names_dir}")
    else:
        logger.warning(f"Names directory not found: {names_dir} — summaries will have no grounded names")

    text_files = find_text_files(text_dir)
    logger.info(f"Found {len(text_files)} text files in {text_dir}")

    # If --file-list provided, filter to only those IDs (avoids scanning all files)
    if args.file_list:
        with open(args.file_list) as fl:
            wanted_ids = {line.strip() for line in fl if line.strip()}
        text_files = [f for f in text_files if f.stem in wanted_ids]
        logger.info(f"Filtered to {len(text_files)} files from --file-list ({len(wanted_ids)} IDs)")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.data_dir) / "processed" / "summaries" / args.dataset
    )

    # Note: tmp.json cleanup removed — with multiple SLURM workers sharing the
    # same output dir, deleting all .tmp.json at startup causes a race condition
    # (worker A deletes worker B's in-progress temp file). Stale .tmp.json files
    # from truly interrupted runs are harmless and get overwritten on retry.

    # Filter files: skip cached, handle resume
    files_to_process = []
    cached = 0
    skip = bool(args.resume_from)

    skipped_ids = 0
    for f in text_files:
        file_id = f.stem
        if skip:
            if file_id == args.resume_from:
                skip = False
            else:
                continue

        if file_id in SKIP_FILE_IDS:
            skipped_ids += 1
            continue

        if not args.force:
            summary_path = output_dir / f"{file_id}_summary.json"
            if summary_path.exists():
                # Check cache validity
                doc_data = load_document_data(f)
                full_text = get_full_text(doc_data)
                text_hash = compute_text_hash(full_text)
                if is_cache_valid(summary_path, text_hash):
                    cached += 1
                    continue

        files_to_process.append(f)

    if skipped_ids:
        logger.warning(f"Skipped {skipped_ids} file(s) in SKIP_FILE_IDS")

    if args.limit:
        files_to_process = files_to_process[:args.limit]

    logger.info(f"Files to process: {len(files_to_process)} ({cached} cached, valid)")

    # Per-worker clean_names: clean ALL files in this worker's range (not just uncached)
    # This ensures clean files exist even for docs whose summaries are already cached.
    if names_dir.exists():
        # Use all files in the worker's range (before cache filtering), not just files_to_process
        all_worker_files = []
        skip_clean = bool(args.resume_from)
        for f in text_files:
            if skip_clean:
                if f.stem == args.resume_from:
                    skip_clean = False
                else:
                    continue
            all_worker_files.append(f)
            if args.limit and len(all_worker_files) >= args.limit:
                break
        worker_file_ids = [f.stem for f in all_worker_files]
        names_to_clean = []
        for fid in worker_file_ids:
            names_path = names_dir / f"{fid}_names.json"
            clean_path = names_dir / f"{fid}_names_clean.json"
            if names_path.exists() and not clean_path.exists():
                names_to_clean.append(names_path)

        if names_to_clean:
            logger.info(f"Cleaning {len(names_to_clean)}/{len(worker_file_ids)} uncleaned names files...")
            t_clean = time.monotonic()
            clean_stats = clean_names_files(names_to_clean, apply=True)
            logger.info(
                f"clean_names done in {time.monotonic() - t_clean:.1f}s: "
                f"{clean_stats['files_processed']} processed, "
                f"{clean_stats['total_original']} -> {clean_stats['total_cleaned']} "
                f"({clean_stats['total_garbage_removed']} garbage, {clean_stats['total_deduplicated']} deduped)"
            )

    if not args.apply:
        logger.info("[DRY RUN] Would process:")
        for f in files_to_process[:10]:
            file_id = f.stem
            names, _ = load_names_for_document(names_dir, file_id)
            doc_data = load_document_data(f)
            pages = get_pages(doc_data)
            logger.info(f"  {f.name} ({len(pages)} pages, {len(names)} names)")
        if len(files_to_process) > 10:
            logger.info(f"  ... and {len(files_to_process) - 10} more")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mock:
        batch_llm_fn = lambda prompts: ["Mock summary for testing."] * len(prompts)
        use_vllm = False
    else:
        logger.info(f"Loading model: {args.model}")
        batch_llm_fn, use_vllm = create_llm_fn(args.model)

    session_start = time.monotonic()

    if use_vllm and not args.mock:
        # Cross-document batching: collect prompts from multiple docs, send in bulk
        logger.info(
            f"Using cross-document batching (batch_size={CROSS_DOC_BATCH_SIZE}, "
            f"chunk_size={BATCH_CHUNK_SIZE})"
        )
        processed, skipped = process_documents_batched(
            files_to_process, names_dir, batch_llm_fn, output_dir,
            chunk_size=BATCH_CHUNK_SIZE,
        )
    else:
        # Sequential per-document processing (transformers fallback or mock)
        processed = 0
        skipped = 0
        for i, text_file in enumerate(files_to_process):
            result = process_document(text_file, names_dir, batch_llm_fn=batch_llm_fn)

            if result is None:
                skipped += 1
                continue

            output_file = output_dir / f"{result['file_id']}_summary.json"
            tmp_file = output_file.with_suffix(".tmp.json")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            os.rename(tmp_file, output_file)

            processed += 1

            if (i + 1) % 50 == 0:
                elapsed = time.monotonic() - session_start
                rate = processed / (elapsed / 60) if elapsed > 0 else 0
                logger.info(
                    f"Progress: {i + 1}/{len(files_to_process)} "
                    f"({processed} done, {skipped} skipped) | "
                    f"{elapsed:.0f}s elapsed, {rate:.1f} docs/min"
                )

    total_elapsed = time.monotonic() - session_start
    rate = processed / (total_elapsed / 60) if total_elapsed > 0 else 0
    logger.info(
        f"Done: {processed} summaries, {skipped} skipped | "
        f"{total_elapsed:.0f}s total, {rate:.1f} docs/min"
    )


if __name__ == "__main__":
    main()
