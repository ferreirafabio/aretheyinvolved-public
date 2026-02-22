"""Stage 0: Deterministic text cleaning with offset preservation.

This module performs same-length cleaning operations that help NER models
see readable tokens without losing character offset alignment.

Key guarantee: len(output) == len(input) ALWAYS

Handles:
- Control characters (Cc, Cf categories)
- Zero-width spaces and joiners (\u200b, \u200c, \u200d, \ufeff)
- RTL/LTR direction marks (\u200e, \u200f, \u202a-\u202e)
- Various Unicode space types (\u00a0, \u2000-\u200a, \u3000)
- Combining marks (strikethrough, overline, etc.) - preserved but flagged
- OCR punctuation artifacts between letters
"""

import unicodedata


# Characters that should become spaces (invisible/formatting chars)
INVISIBLE_CHARS = {
    # Zero-width characters
    '\u200b',  # Zero-width space
    '\u200c',  # Zero-width non-joiner
    '\u200d',  # Zero-width joiner
    '\ufeff',  # Byte order mark / zero-width no-break space

    # Direction marks
    '\u200e',  # Left-to-right mark
    '\u200f',  # Right-to-left mark
    '\u202a',  # Left-to-right embedding
    '\u202b',  # Right-to-left embedding
    '\u202c',  # Pop directional formatting
    '\u202d',  # Left-to-right override
    '\u202e',  # Right-to-left override
    '\u2066',  # Left-to-right isolate
    '\u2067',  # Right-to-left isolate
    '\u2068',  # First strong isolate
    '\u2069',  # Pop directional isolate

    # Other invisible formatters
    '\u00ad',  # Soft hyphen (keep as hyphen? or space?)
    '\u034f',  # Combining grapheme joiner
    '\u061c',  # Arabic letter mark
    '\u115f',  # Hangul choseong filler
    '\u1160',  # Hangul jungseong filler
    '\u17b4',  # Khmer vowel inherent aq
    '\u17b5',  # Khmer vowel inherent aa
    '\u180e',  # Mongolian vowel separator
    '\u2060',  # Word joiner
    '\u2061',  # Function application
    '\u2062',  # Invisible times
    '\u2063',  # Invisible separator
    '\u2064',  # Invisible plus
    '\u206a',  # Inhibit symmetric swapping
    '\u206b',  # Activate symmetric swapping
    '\u206c',  # Inhibit arabic form shaping
    '\u206d',  # Activate arabic form shaping
    '\u206e',  # National digit shapes
    '\u206f',  # Nominal digit shapes
}

# Various Unicode space characters that should normalize to regular space
UNICODE_SPACES = {
    '\u00a0',  # Non-breaking space
    '\u1680',  # Ogham space mark
    '\u2000',  # En quad
    '\u2001',  # Em quad
    '\u2002',  # En space
    '\u2003',  # Em space
    '\u2004',  # Three-per-em space
    '\u2005',  # Four-per-em space
    '\u2006',  # Six-per-em space
    '\u2007',  # Figure space
    '\u2008',  # Punctuation space
    '\u2009',  # Thin space
    '\u200a',  # Hair space
    '\u202f',  # Narrow no-break space
    '\u205f',  # Medium mathematical space
    '\u3000',  # Ideographic space
}

# Punctuation that often appears as OCR errors between letters in names
OCR_PUNCT_IN_NAMES = set('/{}\u02dc|\\')

# Aggressive mode: Characters that are almost never part of names
# These get converted to spaces when in aggressive mode
GARBAGE_CHARS = {
    # Box drawing and block elements
    '█', '▓', '▒', '░', '▌', '▐', '▀', '▄',
    '│', '┤', '╡', '╢', '╖', '╕', '╣', '║',
    '╗', '╝', '╜', '╛', '┐', '└', '┴', '┬',
    '├', '─', '┼', '╞', '╟', '╚', '╔', '╩',
    '╦', '╠', '═', '╬', '╧', '╨', '╤', '╥',
    '╙', '╘', '╒', '╓', '╫', '╪', '┘', '┌',
    # Geometric shapes (often OCR artifacts)
    '■', '□', '▪', '▫', '●', '○', '◌', '◐',
    '◑', '◒', '◓', '◔', '◕', '◖', '◗', '◘',
    '◙', '◚', '◛', '◜', '◝', '◞', '◟', '◠',
    '◡', '◢', '◣', '◤', '◥', '◦', '◧', '◨',
    '◩', '◪', '◫', '◬', '◭', '◮', '◯', '◰',
    # Arrows (OCR often misreads these)
    '←', '→', '↑', '↓', '↔', '↕', '↖', '↗',
    '↘', '↙', '↚', '↛', '↜', '↝', '↞', '↟',
    '↠', '↡', '↢', '↣', '↤', '↥', '↦', '↧',
    '⇄', '⇅', '⇆', '⇇', '⇈', '⇉', '⇊', '⇋',
    '⇌', '⇍', '⇎', '⇏', '⇐', '⇑', '⇒', '⇓',
    '⇔', '⇕', '⇖', '⇗', '⇘', '⇙', '⇚', '⇛',
    '►', '▶', '◀', '▷', '◁', '▸', '◂', '▹', '◃',
    # Stars and decorative (often OCR garbage)
    '★', '☆', '✦', '✧', '✩', '✪', '✫', '✬',
    '✭', '✮', '✯', '✰', '※', '✱', '✲', '✳',
    '✴', '✵', '✶', '✷', '✸', '✹', '✺', '✻',
    # Mathematical operators rarely in names
    '∀', '∁', '∂', '∃', '∄', '∅', '∆', '∇',
    '∈', '∉', '∊', '∋', '∌', '∍', '∎', '∏',
    '∐', '∑', '−', '∓', '∔', '∕', '∖', '∗',
    '∘', '∙', '√', '∛', '∜', '∝', '∞', '∟',
    # Misc symbols
    '§', '¶', '†', '‡', '•', '‣', '⁃', '⁌',
    '⁍', '※', '‼', '⁇', '⁈', '⁉', '‽', '⸘',
}


def is_between_letters(text: str, index: int) -> bool:
    """Check if character at index is surrounded by letters."""
    if index <= 0 or index >= len(text) - 1:
        return False
    return text[index - 1].isalpha() and text[index + 1].isalpha()


def same_length_clean(raw_text: str, aggressive: bool = False) -> str:
    """Replace junk with spaces, normalize unicode. NO DELETION.

    This function performs deterministic cleaning that:
    1. Replaces invisible/zero-width characters with spaces
    2. Normalizes various Unicode space types to regular space
    3. Replaces control characters with spaces
    4. Normalizes unicode characters (NFKC, keeping same length)
    5. Replaces weird punctuation inside words with spaces
    6. (Aggressive mode) Replaces garbage symbols (boxes, arrows, stars)

    The key guarantee is that len(output) == len(input), so character
    offsets from NER models remain valid against the original text.

    Note: OCR digit errors (0→O, 1→I, 5→S) are NOT fixed here.
    That's handled in Stage 4 (LLM Repair) to preserve provenance.

    Args:
        raw_text: The raw OCR text to clean.
        aggressive: If True, also remove garbage chars (boxes, arrows, etc.)
                    that are almost never part of names. Use for heavily
                    corrupted documents.

    Returns:
        Cleaned text with exactly the same length as input.
    """
    if not raw_text:
        return raw_text

    clean = list(raw_text)

    for i, char in enumerate(raw_text):
        # Step 1: Replace invisible characters with space
        if char in INVISIBLE_CHARS:
            clean[i] = ' '
            continue

        # Step 2: Normalize Unicode spaces to regular space
        if char in UNICODE_SPACES:
            clean[i] = ' '
            continue

        # Step 3: (Aggressive) Replace garbage symbols with space
        if aggressive and char in GARBAGE_CHARS:
            clean[i] = ' '
            continue

        # Step 4: Unicode normalize - keep only first char if expansion happens
        # This handles decomposed accents (NFD → NFC) and compatibility chars
        try:
            normalized = unicodedata.normalize('NFKC', char)
            if normalized and len(normalized) >= 1:
                clean[i] = normalized[0]
            else:
                clean[i] = ' '
        except Exception:
            # If normalization fails, keep original
            pass

        # Step 5: Control chars → space (categories Cc=control, Cf=format)
        # Preserve \n, \r, \t — they are Cc but serve as NER boundary signals
        try:
            category = unicodedata.category(clean[i])
            if category in ('Cc', 'Cf') and clean[i] not in ('\n', '\r', '\t'):
                clean[i] = ' '
        except Exception:
            pass

        # Step 6: Weird punct inside words → space
        # These chars often appear as OCR errors splitting names
        if clean[i] in OCR_PUNCT_IN_NAMES and is_between_letters(raw_text, i):
            clean[i] = ' '

    result = ''.join(clean)
    assert len(result) == len(raw_text), f"Length mismatch: {len(result)} != {len(raw_text)}"
    return result


def detect_garbage_ratio(text: str) -> float:
    """Detect ratio of garbage characters in text.

    Useful for determining if a document needs aggressive cleaning.

    Args:
        text: Text to analyze.

    Returns:
        Ratio of garbage characters (0.0 to 1.0).
    """
    if not text:
        return 0.0

    garbage_count = sum(1 for c in text if c in GARBAGE_CHARS)
    return garbage_count / len(text)


def get_page_boundaries(ocr_data: dict) -> list[int]:
    """Extract page boundaries (character offsets) from OCR output.

    Args:
        ocr_data: OCR JSON data with 'pages' or 'full_text' field.

    Returns:
        List of character offsets where each page starts.
        First element is always 0.
    """
    boundaries = [0]

    if 'pages' in ocr_data and ocr_data['pages']:
        current_offset = 0
        for page in ocr_data['pages']:
            page_text = page.get('text', '')
            # Account for page markers and newlines we add
            page_marker = f"[Page {page.get('page_number', len(boundaries))}]\n"
            current_offset += len(page_marker) + len(page_text) + 2  # +2 for \n\n
            boundaries.append(current_offset)

    return boundaries


def get_page_number(char_offset: int, page_boundaries: list[int]) -> int:
    """Get 1-indexed page number for a character offset.

    Args:
        char_offset: Character offset in the full text.
        page_boundaries: List of offsets where each page starts.

    Returns:
        1-indexed page number (1, 2, 3, ...).
    """
    for i, boundary in enumerate(page_boundaries):
        if i + 1 < len(page_boundaries) and char_offset < page_boundaries[i + 1]:
            return i + 1
    return len(page_boundaries)


class CleaningResult:
    """Result of deterministic cleaning with metadata."""

    def __init__(self,
                 raw_text: str,
                 clean_text: str,
                 page_boundaries: list[int] | None = None):
        self.raw_text = raw_text
        self.clean_text = clean_text
        self.page_boundaries = page_boundaries or [0]

        # Verify length invariant
        assert len(self.clean_text) == len(self.raw_text)

    def get_page_number(self, char_offset: int) -> int:
        """Get page number for a character offset."""
        return get_page_number(char_offset, self.page_boundaries)

    def verify_span(self, start: int, end: int, expected_text: str) -> bool:
        """Verify a span exists in both raw and clean text."""
        if start < 0 or end > len(self.raw_text) or start >= end:
            return False
        return (self.raw_text[start:end] == expected_text or
                self.clean_text[start:end] == expected_text)


def clean_document(ocr_data: dict, aggressive: bool = False) -> CleaningResult:
    """Clean an OCR document while preserving offsets.

    Args:
        ocr_data: OCR JSON with 'full_text', 'pages', or 'paragraphs'.
        aggressive: If True, use aggressive cleaning mode for heavily
                    corrupted documents.

    Returns:
        CleaningResult with raw text, clean text, and page boundaries.
    """
    # Extract full text
    full_text = ocr_data.get('full_text', '')

    # Fallback: DOJ text files use 'text' key (not 'full_text')
    if not full_text:
        full_text = ocr_data.get('text', '')

    if not full_text and 'pages' in ocr_data:
        # Concatenate pages with markers
        page_texts = []
        for i, page in enumerate(ocr_data.get('pages', [])):
            page_text = page.get('text', '').strip()
            if page_text:
                page_num = page.get('page_number', i + 1)
                page_texts.append(f"[Page {page_num}]\n{page_text}")
        full_text = "\n\n".join(page_texts)

    if not full_text and 'paragraphs' in ocr_data:
        full_text = "\n\n".join(ocr_data['paragraphs'])

    # Get page boundaries before cleaning
    page_boundaries = get_page_boundaries(ocr_data)

    # Auto-detect if aggressive mode needed
    if not aggressive and detect_garbage_ratio(full_text) > 0.05:
        aggressive = True

    # Clean the text
    clean_text = same_length_clean(full_text, aggressive=aggressive)

    return CleaningResult(
        raw_text=full_text,
        clean_text=clean_text,
        page_boundaries=page_boundaries
    )


if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Basic control char replacement
        ("Hello\x00World", "length preserved"),
        # Zero-width space
        ("Ma\u200bria", "zero-width → space"),
        # Non-breaking space
        ("Anna\u00a0Müller", "nbsp → space"),
        # Various Unicode spaces
        ("A\u2003n\u2002n\u2004a", "em/en spaces → space"),
        # RTL marks
        ("Da\u200eniel\u200f", "direction marks → space"),
        # Weird punct in words
        ("John/Smith", "slash between letters"),
        ("Mary\\Brown", "backslash between letters"),
        # Keep OCR digit errors (for Stage 4)
        ("J0HN SM1TH", "OCR digits preserved"),
        # Soft hyphen
        ("Karl\u00adHeinz", "soft hyphen"),
    ]

    print("Testing same_length_clean:")
    for raw, description in test_cases:
        result = same_length_clean(raw)
        status = "PASS" if len(result) == len(raw) else "FAIL"
        print(f"  {status}: {description}")
        print(f"       {repr(raw)} -> {repr(result)}")
        print(f"       len: {len(raw)} == {len(result)}")

    # Test comprehensive Unicode garbage
    unicode_tests = [
        "★ Anna Müller ★",  # Emoji/symbols
        "F▶r◀o▶m: El◀i▶sa",  # Arrows
        "R\u0336o\u0336b\u0336e\u0336r\u0336t",  # Strikethrough
        "██▓▒░ ░▒▓██",  # Box drawing
        "张 Anna █ Müller □",  # CJK + box
    ]

    print("\nUnicode garbage tests:")
    for text in unicode_tests:
        result = same_length_clean(text)
        status = "PASS" if len(result) == len(text) else "FAIL"
        print(f"  {status}: len({len(text)}) == len({len(result)})")
        print(f"       {repr(text[:30])}")
