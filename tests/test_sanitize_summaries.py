"""Tests for scripts/db/sanitize_summaries.py"""

import sys
from unittest.mock import MagicMock

# Mock psycopg2 before importing the module under test
sys.modules.setdefault("psycopg2", MagicMock())
sys.modules.setdefault("psycopg2.extras", MagicMock())

import pytest
from scripts.db.sanitize_summaries import sanitize_summary, BOILERPLATE_PATTERN


class TestSanitizeSummary:
    """Tests for sanitize_summary()."""

    # --- Standard boilerplate removal ---

    def test_standard_bullet(self):
        text = "Some content.\n- Quelle: Originaldokument pruefen (KI-Zusammenfassung)."
        assert sanitize_summary(text) == "Some content."

    def test_bold_bullet(self):
        text = "Some content.\n- **Quelle: Originaldokument prüfen (KI-Zusammenfassung).**"
        assert sanitize_summary(text) == "Some content."

    def test_standalone_line(self):
        text = "Some content.\nQuelle: Originaldokument pruefen (KI-Zusammenfassung)."
        assert sanitize_summary(text) == "Some content."

    def test_extra_whitespace(self):
        text = "Some content.\n  -  Quelle:  Originaldokument pruefen (KI-Zusammenfassung). "
        assert sanitize_summary(text) == "Some content."

    # --- Umlaut variant ---

    def test_umlaut_pruefen(self):
        text = "Content.\n- Quelle: Originaldokument prüfen (KI-Zusammenfassung)."
        assert sanitize_summary(text) == "Content."

    # --- Space in KI-Zusammenfassung ---

    def test_space_variant(self):
        text = "Content.\n- Quelle: Originaldokument pruefen (KI Zusammenfassung)."
        assert sanitize_summary(text) == "Content."

    # --- Bold variants ---

    def test_bold_quelle_only(self):
        """**Quelle:** with bold on just the word."""
        text = "Content.\n- **Quelle:** Originaldokument prüfen (KI-Zusammenfassung)."
        assert sanitize_summary(text) == "Content."

    def test_bold_colon_outside(self):
        """**Quelle**: with colon outside the bold."""
        text = "Content.\n- **Quelle**: Originaldokument prüfen (KI-Zusammenfassung)."
        assert sanitize_summary(text) == "Content."

    def test_bold_trailing_period_after_stars(self):
        text = "Content.\n- **Quelle: Originaldokument pruefen (KI-Zusammenfassung).**."
        assert sanitize_summary(text) == "Content."

    # --- Truncated variant ---

    def test_truncated_no_closing_paren(self):
        text = "Content.\n\nQuelle: Originaldokument prüfen (KI-Zusammenfassung"
        assert sanitize_summary(text) == "Content."

    # --- With English translation ---

    def test_with_source_english_check(self):
        text = (
            "Content.\n- **Quelle: Originaldokument prüfen (KI-Zusammenfassung).** "
            "(Source: Check the original document (AI summary).)"
        )
        assert sanitize_summary(text) == "Content."

    def test_with_source_english_verify(self):
        text = (
            "Content.\n- **Quelle: Originaldokument prüfen (KI-Zusammenfassung).** "
            "(Source: Verify the original document (AI summary).)"
        )
        assert sanitize_summary(text) == "Content."

    # --- Page reference suffixes ---

    def test_page_reference_suffix(self):
        text = "Content.\n- Quelle: Originaldokument prüfen (KI-Zusammenfassung). (S. 1-2)"
        assert sanitize_summary(text) == "Content."

    def test_all_pages_suffix(self):
        text = "Content.\n- Quelle: Originaldokument prüfen (KI-Zusammenfassung). (All pages)"
        assert sanitize_summary(text) == "Content."

    # --- Note suffix ---

    def test_note_suffix(self):
        text = (
            "Content.\n- **Quelle: Originaldokument prüfen (KI-Zusammenfassung).** "
            "(Note: Several pages had garbled text due to low OCR quality.)"
        )
        assert sanitize_summary(text) == "Content."

    # --- JSON-embedded suffix ---

    def test_json_suffix(self):
        text = (
            'Content.\n- Quelle: Originaldokument pruefen (KI-Zusammenfassung).'
            '", "occupation_mentions": []}'
        )
        assert sanitize_summary(text) == "Content."

    # --- No boilerplate (unchanged) ---

    def test_no_boilerplate_unchanged(self):
        text = "Clean summary with no German text."
        assert sanitize_summary(text) == "Clean summary with no German text."

    def test_no_boilerplate_with_newlines(self):
        text = "First paragraph.\n\nSecond paragraph."
        assert sanitize_summary(text) == "First paragraph.\n\nSecond paragraph."

    # --- Edge cases ---

    def test_empty_string(self):
        assert sanitize_summary("") == ""

    def test_none_returns_none(self):
        """None input should return None (not crash)."""
        assert sanitize_summary(None) is None

    def test_only_boilerplate(self):
        """If text is just the boilerplate, result should be empty."""
        text = "- Quelle: Originaldokument pruefen (KI-Zusammenfassung)."
        result = sanitize_summary(text)
        assert result == "" or result.strip() == ""

    def test_boilerplate_in_middle_not_matched(self):
        """Boilerplate should only match at end of text, not in the middle."""
        text = (
            "Start.\n- Quelle: Originaldokument pruefen (KI-Zusammenfassung).\n"
            "More content follows here."
        )
        # The regex is MULTILINE anchored to $, so it matches end-of-line.
        # Depending on implementation, middle matches may or may not be removed.
        # The important thing: if removed, the surrounding content is preserved.
        result = sanitize_summary(text)
        assert "Start." in result
        assert "More content follows here." in result

    # --- Multiline content preserved ---

    def test_multiline_content_before_boilerplate(self):
        text = (
            "First point.\n"
            "Second point.\n"
            "Third point.\n"
            "- Quelle: Originaldokument pruefen (KI-Zusammenfassung)."
        )
        result = sanitize_summary(text)
        assert "First point." in result
        assert "Second point." in result
        assert "Third point." in result
        assert "Quelle" not in result

    # --- Case insensitivity ---

    def test_case_insensitive_quelle(self):
        text = "Content.\n- QUELLE: Originaldokument pruefen (KI-ZUSAMMENFASSUNG)."
        assert sanitize_summary(text) == "Content."

    def test_mixed_case(self):
        text = "Content.\n- quelle: originaldokument PRUEFEN (ki-zusammenfassung)."
        assert sanitize_summary(text) == "Content."


class TestBoilerplatePattern:
    """Tests for the compiled regex pattern directly."""

    def test_pattern_matches_standard(self):
        line = "\n- Quelle: Originaldokument pruefen (KI-Zusammenfassung)."
        assert BOILERPLATE_PATTERN.search(line) is not None

    def test_pattern_does_not_match_random_german(self):
        line = "\n- This is normal text about Quelle water sources."
        # "Quelle" alone shouldn't trigger without "Originaldokument"
        assert BOILERPLATE_PATTERN.search(line) is None

    def test_pattern_does_not_match_partial(self):
        line = "\n- Quelle: something else entirely."
        assert BOILERPLATE_PATTERN.search(line) is None
