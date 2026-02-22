"""Text quality gate for PDF text layer validation.

Determines whether an embedded text layer is usable or if OCR is needed.
Uses multiple cheap signals (~0.2ms per file):

1. Minimum length — too little text → OCR
2. Dictionary word ratio — fraction of 3+ letter words in a dictionary
3. Alpha ratio — fraction of non-whitespace chars that are letters
4. Weird char ratio — replacement chars, control chars, box drawing
5. Token sanity — median token length, digit-mixed token ratio

Also provides page-level image analysis to distinguish scanned text pages
from photos/figures/blanks (~5ms per page, no ML):

    from src.text_quality import is_text_like_page

    if is_text_like_page(pil_image):
        # scanned text page → worth OCR'ing
    else:
        # photo/chart/blank → skip OCR

Usage:
    from src.text_quality import is_text_layer_good

    result = is_text_layer_good(text)
    if result.good:
        # use embedded text
    else:
        # fall back to OCR
        print(f"Bad text layer: {result.reason}")
"""

from __future__ import annotations

import os
import re
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

import numpy as np

# ---------------------------------------------------------------------------
# Thresholds (tuned against DS1 bad, DS6-8 good, DS12 good)
# ---------------------------------------------------------------------------
# Decision tree structure:
#   1. Too short → BAD
#   2. Weird chars → BAD (strong signal in any zone)
#   3. Token insanity → BAD (giant/tiny tokens, any zone)
#   4. word_ratio >= W_HIGH → GOOD (clearly real text, ignore alpha)
#   5. word_ratio <= W_LOW → BAD (clearly garbage)
#   6. Gray zone → use alpha_ratio to decide
#
MIN_CHARS = 200           # less → "too little text"
W_HIGH = 0.55             # clearly good: keep embedded text
W_LOW = 0.25              # clearly bad: force OCR
GRAY_ALPHA_MIN = 0.50     # gray zone: alpha must be at least this
MAX_WEIRD_RATIO = 0.05    # replacement / control / box-drawing chars
MIN_MEDIAN_TOKEN_LEN = 2  # median whitespace-split token length
MAX_MEDIAN_TOKEN_LEN = 20 # garbled = very long concatenated tokens
MAX_DIGIT_TOKEN_RATIO = 0.40  # tokens mixing letters + digits


# ---------------------------------------------------------------------------
# Dictionary word set (loaded once, cached)
# ---------------------------------------------------------------------------
_WORD_SET: set[str] | None = None


def _load_word_set() -> set[str]:
    """Load system dictionary, falling back to a curated set."""
    global _WORD_SET
    if _WORD_SET is not None:
        return _WORD_SET

    for dict_path in ["/usr/share/dict/words", "/usr/share/dict/american-english"]:
        if os.path.exists(dict_path):
            with open(dict_path) as f:
                _WORD_SET = {w.strip().lower() for w in f if len(w.strip()) >= 3}
            return _WORD_SET

    # Fallback: curated set (~500 common English + legal/DOJ terms)
    _WORD_SET = {
        "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
        "her", "was", "one", "our", "out", "day", "had", "has", "his", "how",
        "its", "may", "new", "now", "old", "see", "way", "who", "did", "get",
        "let", "say", "she", "too", "use", "man", "two", "been", "call", "come",
        "each", "find", "from", "give", "good", "have", "help", "here", "high",
        "just", "keep", "know", "last", "like", "line", "long", "look", "made",
        "make", "many", "more", "most", "much", "must", "name", "next", "only",
        "over", "part", "same", "some", "such", "take", "tell", "than", "that",
        "them", "then", "they", "this", "time", "very", "want", "well", "what",
        "when", "will", "with", "word", "work", "year", "your", "about", "after",
        "again", "being", "below", "between", "both", "could", "down", "first",
        "great", "house", "large", "later", "never", "number", "other", "people",
        "place", "point", "right", "small", "state", "still", "their", "there",
        "these", "thing", "think", "those", "three", "under", "water", "where",
        "which", "while", "world", "would", "write", "also", "back", "before",
        "city", "even", "hand", "home", "into", "life", "might", "need", "open",
        "own", "play", "said", "side", "turn", "went", "left", "best", "door",
        "done", "face", "fact", "head", "kind", "land", "live", "move", "real",
        "room", "show", "case", "close", "end", "eye", "form", "group", "hand",
        "john", "late", "mind", "miss", "office", "order", "person", "power",
        "read", "report", "school", "set", "since", "start", "stop", "sure",
        "through", "until", "upon", "week", "against", "air", "already", "always",
        "away", "because", "become", "big", "change", "children", "country",
        "dear", "different", "during", "early", "enough", "every", "example",
        "family", "far", "few", "follow", "general", "government", "half",
        "important", "information", "interest", "letter", "little", "local",
        "money", "morning", "mother", "night", "nothing", "often", "once",
        "possible", "president", "problem", "program", "public", "question",
        "quite", "rather", "second", "service", "several", "social", "south",
        "special", "subject", "today", "together", "top", "trying", "united",
        "war", "without", "yes", "young", "date", "dear", "court", "law",
        "file", "page", "document", "witness", "testimony", "deposition",
        "evidence", "attorney", "counsel", "plaintiff", "defendant", "judge",
        "trial", "motion", "hearing", "sworn", "affidavit", "exhibit",
        "statement", "interview", "record", "investigation", "federal",
        "department", "justice", "district", "southern", "florida",
        "island", "virgin", "york", "palm", "beach", "massage", "house",
        "travel", "flight", "telephone", "address", "email", "fax",
    }
    return _WORD_SET


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

# Regex: 3+ letter words
_WORD_RE = re.compile(r"[a-zA-Z]{3,}")

# Weird characters: replacement char, control chars (except \n\r\t), box drawing
_WEIRD_RE = re.compile(
    r"[\ufffd"           # replacement character
    r"\x00-\x08"         # control chars (before tab)
    r"\x0b\x0c"          # VT, FF
    r"\x0e-\x1f"         # control chars (after CR)
    r"\x7f"              # DEL
    r"\u2500-\u257f"     # box drawing
    r"\u2580-\u259f"     # block elements
    r"\ufff0-\ufffe"     # specials
    r"]"
)

# Token with mixed letters + digits (e.g. "h3llo", "t0p1c")
_MIXED_TOKEN_RE = re.compile(r"(?=[^ ]*[a-zA-Z])(?=[^ ]*\d)[^ ]+")


def compute_word_ratio(text: str) -> float:
    """Fraction of 3+ letter tokens that appear in a dictionary."""
    words = [m.group().lower() for m in _WORD_RE.finditer(text)]
    if len(words) < 5:
        return 0.0
    word_set = _load_word_set()
    return sum(1 for w in words if w in word_set) / len(words)


def compute_alpha_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are letters."""
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    return sum(1 for c in non_ws if c.isalpha()) / len(non_ws)


def compute_weird_ratio(text: str) -> float:
    """Fraction of all characters that are 'weird' (control, replacement, etc)."""
    if not text:
        return 0.0
    weird_count = len(_WEIRD_RE.findall(text))
    return weird_count / len(text)


def compute_token_sanity(text: str) -> tuple[float, float]:
    """Compute median token length and digit-mixed token ratio.

    Returns:
        (median_token_length, digit_mixed_ratio)
    """
    tokens = text.split()
    if not tokens:
        return (0.0, 0.0)

    lengths = [len(t) for t in tokens]
    median_len = statistics.median(lengths)

    # Tokens with both letters and digits mixed (e.g. "h3llo")
    mixed = sum(1 for t in tokens if _MIXED_TOKEN_RE.match(t))
    mixed_ratio = mixed / len(tokens)

    return (median_len, mixed_ratio)


# ---------------------------------------------------------------------------
# Quality gate result
# ---------------------------------------------------------------------------

@dataclass
class TextQualityResult:
    """Result of text quality assessment."""
    good: bool
    reason: str
    text_length: int = 0
    word_ratio: float = 0.0
    alpha_ratio: float = 0.0
    weird_ratio: float = 0.0
    median_token_len: float = 0.0
    digit_mixed_ratio: float = 0.0

    def __str__(self) -> str:
        status = "GOOD" if self.good else f"BAD ({self.reason})"
        return (
            f"{status} len={self.text_length} "
            f"word={self.word_ratio:.1%} alpha={self.alpha_ratio:.1%} "
            f"weird={self.weird_ratio:.1%} med_tok={self.median_token_len:.1f} "
            f"digit_mix={self.digit_mixed_ratio:.1%}"
        )


def is_text_layer_good(text: str) -> TextQualityResult:
    """Determine whether an embedded PDF text layer is usable.

    Uses a decision-tree approach (~0.2ms per file):
      1. Too short → BAD
      2. Weird chars / token insanity → BAD (strong signals, any zone)
      3. word_ratio >= W_HIGH → GOOD (clearly real text, skip alpha check)
      4. word_ratio <= W_LOW → BAD (clearly garbage)
      5. Gray zone → alpha_ratio decides

    This avoids false positives on number-heavy documents (calendars,
    financial tables) where alpha_ratio is low but word_ratio is high.

    Args:
        text: The extracted text to evaluate.

    Returns:
        TextQualityResult with .good bool and diagnostic fields.
    """
    stripped = text.strip() if text else ""
    length = len(stripped)

    # Step 1: minimum length
    if length < MIN_CHARS:
        return TextQualityResult(
            good=False,
            reason=f"too_short ({length} < {MIN_CHARS})",
            text_length=length,
        )

    # Compute all signals upfront (all are cheap)
    word_ratio = compute_word_ratio(stripped)
    alpha_ratio = compute_alpha_ratio(stripped)
    weird_ratio = compute_weird_ratio(stripped)
    median_tok, digit_mix = compute_token_sanity(stripped)

    base = dict(
        text_length=length,
        word_ratio=word_ratio,
        alpha_ratio=alpha_ratio,
        weird_ratio=weird_ratio,
        median_token_len=median_tok,
        digit_mixed_ratio=digit_mix,
    )

    # Step 2: strong BAD signals (override everything)
    if weird_ratio > MAX_WEIRD_RATIO:
        return TextQualityResult(
            good=False,
            reason=f"weird_chars ({weird_ratio:.1%} > {MAX_WEIRD_RATIO:.0%})",
            **base,
        )
    if median_tok < MIN_MEDIAN_TOKEN_LEN:
        return TextQualityResult(
            good=False,
            reason=f"tiny_tokens (median {median_tok:.1f} < {MIN_MEDIAN_TOKEN_LEN})",
            **base,
        )
    if median_tok > MAX_MEDIAN_TOKEN_LEN:
        return TextQualityResult(
            good=False,
            reason=f"giant_tokens (median {median_tok:.1f} > {MAX_MEDIAN_TOKEN_LEN})",
            **base,
        )
    if digit_mix > MAX_DIGIT_TOKEN_RATIO:
        return TextQualityResult(
            good=False,
            reason=f"digit_mixed ({digit_mix:.1%} > {MAX_DIGIT_TOKEN_RATIO:.0%})",
            **base,
        )

    # Step 3: clearly good — high word ratio means text is real
    if word_ratio >= W_HIGH:
        return TextQualityResult(good=True, reason="high_word_ratio", **base)

    # Step 4: clearly bad — very low word ratio
    if word_ratio <= W_LOW:
        return TextQualityResult(
            good=False,
            reason=f"low_word_ratio ({word_ratio:.1%} <= {W_LOW:.0%})",
            **base,
        )

    # Step 5: gray zone — alpha_ratio breaks the tie
    if alpha_ratio < GRAY_ALPHA_MIN:
        return TextQualityResult(
            good=False,
            reason=f"gray_low_alpha (word={word_ratio:.1%}, alpha={alpha_ratio:.1%})",
            **base,
        )

    return TextQualityResult(good=True, reason="gray_passed", **base)


# ---------------------------------------------------------------------------
# Page-level image analysis (scanned text vs photo/figure/blank)
# ---------------------------------------------------------------------------
# Two-signal approach: Gaussian-blurred peak counting (handles dense text)
# AND run-length band detection (handles sparse text). ~5ms per page, no ML.
#
# Key insight from DOJ data:
#   - Scanned text pages: dark_ratio 0.02-0.15, regular projection peaks
#   - Photos (headshots, objects): dark_ratio 0.20+, irregular texture
#   - dark_ratio alone separates text (0.03-0.07) from photos (0.26-0.40)
#     with a clean gap at 0.20
#
# Pipeline:
#   1. Gaussian blur → suppress photo texture (high-frequency noise)
#   2. Peak counting on blurred projection (works for dense text blocks)
#   3. Run-length band detection (works for sparse/well-separated text)
#   4. Spacing regularity (CV of inter-peak distances)
#   5. Decision: blank / text_like / photo_like / uncertain
#

# Analysis resolution (width x height) — enough for line detection, fast
_PAGE_ANALYSIS_W = 600
_PAGE_ANALYSIS_H = 800
# Gaussian blur sigma for texture suppression
_BLUR_SIGMA = 2.0
# Minimum distance between peaks (rows at 800px). 8px ≈ 1% of page.
_MIN_PEAK_DISTANCE = 8
# Minimum peaks/bands for "text_like" classification
_PAGE_MIN_LINES = 8
# Maximum coefficient of variation of line spacing for "regular" text
_MAX_SPACING_CV = 0.8
# dark_ratio bounds for text-like pages
# Scanned text: 0.02-0.15. Photos start at ~0.20+.
_DARK_RATIO_TEXT_MIN = 0.02
_DARK_RATIO_TEXT_MAX = 0.20
# dark_ratio below this → blank
_DARK_RATIO_BLANK = 0.005


@dataclass
class PageTextiness:
    """Result of page-level text-likeness analysis."""
    page_type: str          # "text_like" | "photo_like" | "blank" | "uncertain"
    dark_ratio: float       # fraction of dark pixels
    line_count: int         # detected text lines (best of peaks or bands)
    regularity: float       # spacing regularity (CV, lower = more regular)
    reason: str             # diagnostic string

    @property
    def is_text_like(self) -> bool:
        return self.page_type == "text_like"

    def __str__(self) -> str:
        return (
            f"{self.page_type} dark={self.dark_ratio:.3f} "
            f"lines={self.line_count} reg={self.regularity:.2f} ({self.reason})"
        )


def _gaussian_blur_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian blur via convolution (no scipy dependency)."""
    if sigma <= 0:
        return arr
    radius = int(3 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


def _count_peaks(
    projection: np.ndarray,
    min_distance: int = _MIN_PEAK_DISTANCE,
) -> tuple[int, list[int]]:
    """Count local maxima in a 1-D signal with minimum distance enforcement.

    Returns:
        (peak_count, peak_positions)
    """
    if len(projection) < 3:
        return 0, []

    threshold = projection.mean() + 0.5 * projection.std()
    if threshold <= 0:
        return 0, []

    peaks = []
    last_peak = -min_distance

    for i in range(1, len(projection) - 1):
        if (
            projection[i] > threshold
            and projection[i] >= projection[i - 1]
            and projection[i] >= projection[i + 1]
            and (i - last_peak) >= min_distance
        ):
            peaks.append(i)
            last_peak = i

    return len(peaks), peaks


def _count_text_bands(
    row_projection: np.ndarray,
    ink_threshold: float,
) -> tuple[int, list[int]]:
    """Count contiguous ink-row runs with plausible text-line thickness.

    Returns:
        (band_count, band_centers)
    """
    is_ink = row_projection > ink_threshold
    n = len(is_ink)

    centers = []
    i = 0
    while i < n:
        if is_ink[i]:
            start = i
            while i < n and is_ink[i]:
                i += 1
            thickness = i - start
            # Text lines at 800px: 2-15 px thick
            if 2 <= thickness <= 15:
                centers.append((start + i) // 2)
        else:
            i += 1

    return len(centers), centers


def _spacing_cv(positions: list[int]) -> float:
    """Coefficient of variation of distances between consecutive positions.

    Returns -1.0 if fewer than 2 positions.
    """
    if len(positions) < 2:
        return -1.0
    distances = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    mean_d = sum(distances) / len(distances)
    if mean_d <= 0:
        return 99.0
    std_d = (sum((d - mean_d) ** 2 for d in distances) / len(distances)) ** 0.5
    return std_d / mean_d


def analyze_page_textiness(image: "Image.Image") -> PageTextiness:
    """Analyze whether a rendered page image contains text lines.

    Uses two complementary signals on a Gaussian-blurred projection:
    - Peak counting: works for dense text (inter-line gaps too small for bands)
    - Run-length bands: works for sparse/well-separated text

    The best of the two is used, gated by dark_ratio (0.02-0.20 for text)
    and spacing regularity (CV < 0.8 for text).

    Args:
        image: PIL Image of the rendered page (any size/mode).

    Returns:
        PageTextiness with page_type, dark_ratio, line_count, regularity.
    """
    # Downscale to fixed size for consistent, fast analysis
    gray = np.array(
        image.convert("L").resize((_PAGE_ANALYSIS_W, _PAGE_ANALYSIS_H))
    )

    # Adaptive binarization: pixels darker than 70% of median are "dark".
    median_val = float(np.median(gray))
    binarize_thresh = median_val * 0.7
    dark = gray < binarize_thresh
    dark_ratio = float(dark.mean())

    # --- Tier 1: blank page ---
    if dark_ratio < _DARK_RATIO_BLANK:
        return PageTextiness(
            page_type="blank",
            dark_ratio=dark_ratio,
            line_count=0,
            regularity=-1.0,
            reason=f"dark_ratio {dark_ratio:.4f} < {_DARK_RATIO_BLANK}",
        )

    # --- Tier 2: photo (dark_ratio > 0.20) ---
    # Photos of people/objects have dark_ratio 0.20-0.50.
    # Scanned text rarely exceeds 0.15.
    if dark_ratio > _DARK_RATIO_TEXT_MAX:
        return PageTextiness(
            page_type="photo_like",
            dark_ratio=dark_ratio,
            line_count=0,
            regularity=-1.0,
            reason=f"dark={dark_ratio:.3f} > {_DARK_RATIO_TEXT_MAX}",
        )

    # Horizontal projection: fraction of dark pixels per row
    row_density = dark.sum(axis=1).astype(np.float64) / dark.shape[1]

    # Gaussian blur to suppress photo texture (hairlines, edges, noise)
    smoothed = _gaussian_blur_1d(row_density, _BLUR_SIGMA)

    # Signal 1: Peak counting (handles dense text blocks)
    peak_count, peak_positions = _count_peaks(smoothed, _MIN_PEAK_DISTANCE)
    peak_cv = _spacing_cv(peak_positions)

    # Signal 2: Run-length bands (handles sparse text)
    # Adaptive ink threshold: median of nonzero rows * 0.3
    nonzero = smoothed[smoothed > 0.001]
    ink_thresh = float(np.median(nonzero)) * 0.3 if len(nonzero) > 0 else 0.015
    band_count, band_centers = _count_text_bands(smoothed, ink_thresh)
    band_cv = _spacing_cv(band_centers)

    # Use the best signal (whichever found more lines)
    if band_count >= peak_count:
        line_count = band_count
        cv = band_cv
        method = "bands"
    else:
        line_count = peak_count
        cv = peak_cv
        method = "peaks"

    regular = cv >= 0 and cv < _MAX_SPACING_CV

    # --- Tier 3: text-like ---
    # Enough lines + dark ratio in text range + regular spacing
    if (
        line_count >= _PAGE_MIN_LINES
        and dark_ratio >= _DARK_RATIO_TEXT_MIN
        and regular
    ):
        return PageTextiness(
            page_type="text_like",
            dark_ratio=dark_ratio,
            line_count=line_count,
            regularity=cv,
            reason=f"{method}={line_count} dark={dark_ratio:.3f} cv={cv:.2f} text",
        )

    # --- Tier 4: low dark ratio with few lines → blank ---
    if dark_ratio < _DARK_RATIO_TEXT_MIN:
        return PageTextiness(
            page_type="blank",
            dark_ratio=dark_ratio,
            line_count=line_count,
            regularity=cv,
            reason=f"dark={dark_ratio:.3f} < {_DARK_RATIO_TEXT_MIN}",
        )

    # --- Tier 5: uncertain ---
    # In text dark range but not enough lines or irregular spacing
    return PageTextiness(
        page_type="uncertain",
        dark_ratio=dark_ratio,
        line_count=line_count,
        regularity=cv,
        reason=f"{method}={line_count} dark={dark_ratio:.3f} cv={cv:.2f} unclear",
    )


def is_text_like_page(image: "Image.Image") -> bool:
    """Convenience wrapper: True if page is text_like.

    For richer metadata, use analyze_page_textiness() directly.
    """
    return analyze_page_textiness(image).is_text_like
