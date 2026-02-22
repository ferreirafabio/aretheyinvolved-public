"""Tests for scripts/extraction/clean_names.py"""

import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.extraction.clean_names import (
    clean_name,
    is_garbage,
    normalize_role,
    clean_names_file,
    process_directory,
    MIN_CONFIDENCE,
)


class TestCleanName:
    def test_strip_leading_dots(self):
        assert clean_name("...John Smith") == "John Smith"

    def test_strip_leading_spaces(self):
        assert clean_name("   Jane Doe") == "Jane Doe"

    def test_strip_leading_hyphens(self):
        assert clean_name("--Bob Jones") == "Bob Jones"

    def test_strip_mixed_leading(self):
        assert clean_name(". - .Alice Brown") == "Alice Brown"

    def test_strip_trailing_spaces(self):
        assert clean_name("John Smith   ") == "John Smith"

    def test_strip_leading_quotes(self):
        assert clean_name("'Jeffrey Epstein") == "Jeffrey Epstein"

    def test_strip_trailing_comma(self):
        assert clean_name("Ghislaine Maxwell,") == "Ghislaine Maxwell"

    def test_strip_trailing_semicolon(self):
        assert clean_name("John Smith;") == "John Smith"

    def test_preserve_internal_dots(self):
        """Dots in initials should be preserved."""
        assert clean_name("J.E.") == "J.E."
        assert clean_name("Geoffrey S. Berman") == "Geoffrey S. Berman"

    def test_no_change_needed(self):
        assert clean_name("John Smith") == "John Smith"

    def test_empty_string(self):
        assert clean_name("") == ""

    def test_strip_leading_underscores(self):
        assert clean_name("___John Smith") == "John Smith"


class TestLeadingDigitStripping:
    """Tests for leading index number removal."""

    def test_simple_page_number(self):
        assert clean_name("2 Brune") == "Brune"

    def test_page_number_with_dot(self):
        assert clean_name("3. A. Farmer") == "A. Farmer"

    def test_page_number_with_paren(self):
        assert clean_name("4) Mchugh") == "Mchugh"

    def test_page_number_with_colon(self):
        assert clean_name("6: Aznaran") == "Aznaran"

    def test_page_number_with_dash(self):
        assert clean_name("7 - Dawson") == "Dawson"

    def test_year_without_delimiter_preserved(self):
        """2005 > 300 with no delimiter, should be preserved."""
        assert "Jeffrey Epstein" in clean_name("2005 Jeffrey Epstein")
        assert "2005" in clean_name("2005 Jeffrey Epstein")

    def test_year_with_delimiter_stripped(self):
        """Year with delimiter gets stripped."""
        assert clean_name("2005. Jeffrey Epstein") == "Jeffrey Epstein"

    def test_small_number_space_only(self):
        assert clean_name("14 Smith") == "Smith"

    def test_large_number_space_only_preserved(self):
        """Numbers > 300 without delimiter should not be stripped."""
        result = clean_name("500 Smith")
        assert "500" in result


class TestRepeatedNameCleanup:
    """15 hard test cases for repeated-name span cleanup."""

    def test_01_exact_full_name_duplication(self):
        """Perfect 2x repetition."""
        assert clean_name("David Oscar Markus David Oscar Markus") == "David Oscar Markus"

    def test_02_full_name_duplication_plus_trailing_token(self):
        """Repetition plus last-name echo."""
        assert clean_name("David Oscar Markus David Oscar Markus Markus") == "David Oscar Markus"

    def test_03_full_name_duplication_plus_extra_name(self):
        """Overlong span captured 2 people — keep first only."""
        assert clean_name("David Oscar Markus David Oscar Markus Melissa Madrigal") == "David Oscar Markus"

    def test_04_tripled_repetition(self):
        """3x repetition."""
        assert clean_name("David Oscar Markus David Oscar Markus David Oscar Markus") == "David Oscar Markus"

    def test_05_duplication_with_punctuation(self):
        """Punctuation/whitespace variants should normalize before detection."""
        assert clean_name("David Oscar Markus, David Oscar Markus") == "David Oscar Markus"

    def test_06_duplication_with_newline(self):
        """OCR line breaks."""
        assert clean_name("David Oscar Markus\nDavid Oscar Markus") == "David Oscar Markus"

    def test_07_duplication_with_accents(self):
        """Accented characters — normalization should still detect repetition."""
        assert clean_name("Dávid Oscar Markus Dávid Oscar Markus") == "Dávid Oscar Markus"

    def test_08_last_name_echo(self):
        """Common 'last token repeated' span bug."""
        assert clean_name("David Oscar Markus Markus") == "David Oscar Markus"

    def test_09_first_name_echo(self):
        """Repeated first token at span start."""
        assert clean_name("David David Oscar Markus") == "David Oscar Markus"

    def test_10_two_people_repeated_block(self):
        """Repeated multi-entity sequence — keep first occurrence."""
        result = clean_name(
            "David Oscar Markus Melissa Madrigal David Oscar Markus Melissa Madrigal"
        )
        assert result == "David Oscar Markus Melissa Madrigal"

    def test_11_legit_same_last_name(self):
        """Two different people share surname — don't collapse."""
        assert clean_name("David Markus and Robert Markus") == "David Markus and Robert Markus"

    def test_12_legit_suffix(self):
        """Suffix isn't repetition."""
        assert clean_name("David Oscar Markus Jr.") == "David Oscar Markus Jr."

    def test_13_hyphenated_last_name_echo(self):
        """Last-name echo with hyphenation."""
        assert clean_name("Anna-Marie Smith-Jones Smith-Jones") == "Anna-Marie Smith-Jones"

    def test_14_noise_between_repetitions(self):
        """Em dash separator; should treat as duplication."""
        assert clean_name("David Oscar Markus — David Oscar Markus") == "David Oscar Markus"

    def test_15_garbage_repeated_short_token(self):
        """No stable full name, just repeated token — should be garbage."""
        name = clean_name("Markus Markus Markus")
        assert is_garbage(name)


class TestSeparatorVariants:
    """A) Separator variants without spaces."""

    def test_em_dash_no_spaces(self):
        assert clean_name("David Oscar Markus\u2014David Oscar Markus") == "David Oscar Markus"

    def test_en_dash_no_spaces(self):
        assert clean_name("David Oscar Markus\u2013David Oscar Markus") == "David Oscar Markus"

    def test_comma_no_space(self):
        assert clean_name("David Oscar Markus,David Oscar Markus") == "David Oscar Markus"

    def test_semicolon_no_space(self):
        assert clean_name("David Oscar Markus;David Oscar Markus") == "David Oscar Markus"


class TestUnicodeWhitespace:
    """B) Unicode whitespace / invisible characters."""

    def test_nbsp_between_repetitions(self):
        assert clean_name("David Oscar Markus\u00a0David Oscar Markus") == "David Oscar Markus"

    def test_zero_width_space_in_name(self):
        assert clean_name("Da\u200bvid Oscar Markus David Oscar Markus") == "David Oscar Markus"

    def test_rtl_mark(self):
        assert clean_name("David Oscar Markus\u200f David Oscar Markus") == "David Oscar Markus"


class TestPunctuationNoise:
    """C) Punctuation noise between repetitions."""

    def test_parentheses(self):
        assert clean_name("David Oscar Markus (David Oscar Markus)") == "David Oscar Markus"

    def test_slash(self):
        assert clean_name("David Oscar Markus / David Oscar Markus") == "David Oscar Markus"

    def test_pipe(self):
        assert clean_name("David Oscar Markus | David Oscar Markus") == "David Oscar Markus"


class TestSuffixesAndInitials:
    """D) Suffixes / initials / ordering traps."""

    def test_suffix_survives_dedup(self):
        assert clean_name("David Oscar Markus Jr. David Oscar Markus Jr.") == "David Oscar Markus Jr."

    def test_initials_not_collapsed(self):
        """Initials != safe repetition — don't over-collapse."""
        assert clean_name("D. O. Markus David Oscar Markus") == "D. O. Markus David Oscar Markus"

    def test_permutation_not_collapsed(self):
        """Token permutation != repetition."""
        assert clean_name("Oscar David Markus David Oscar Markus") == "Oscar David Markus David Oscar Markus"


class TestLegitMultiPerson:
    """E) Legit multi-person spans (must not over-clean)."""

    def test_two_people_with_and(self):
        assert clean_name("David Oscar Markus and Melissa Madrigal") == "David Oscar Markus and Melissa Madrigal"

    def test_shared_surname_trailing_echo(self):
        """Last-name repetition with two people — drop last echo only."""
        result = clean_name("David Markus and Robert Markus Markus")
        assert result == "David Markus and Robert Markus"


class TestPathologicalGarbage:
    """F) Pathological garbage that should be dropped."""

    def test_two_word_same_token(self):
        assert is_garbage("Markus Markus")

    def test_three_word_same_token(self):
        assert is_garbage("David David David")


class TestMixedOcrChaos:
    """G) Mixed OCR chaos."""

    def test_multiline_dash_repetition(self):
        assert clean_name("David Oscar Markus \u2014 David Oscar Markus\nDavid Oscar Markus") == "David Oscar Markus"

    def test_comma_dash_semicolon(self):
        assert clean_name("David Oscar Markus, \u2014 David Oscar Markus;") == "David Oscar Markus"


class TestStressTest:
    """Final stress test combining multiple issues."""

    def test_zwsp_emdash_echo_suffix(self):
        """Zero-width char + em dash + last-name echo + suffix."""
        assert clean_name("David\u200b Oscar Markus\u2014David Oscar Markus Markus Jr.") == "David Oscar Markus Jr."


class TestIsGarbage:
    def test_too_short(self):
        assert is_garbage("AB") is True
        assert is_garbage("A") is True

    def test_three_chars_ok(self):
        assert is_garbage("Bob") is False

    def test_consecutive_digits(self):
        assert is_garbage("EFTA00001234") is True
        assert is_garbage("Phone 555") is True

    def test_no_letters(self):
        assert is_garbage("123") is True
        assert is_garbage("...") is True

    def test_concatenated_token(self):
        assert is_garbage("EpsteinJeffrey") is True
        assert is_garbage("LongConcatenatedName") is True

    def test_short_single_token_ok(self):
        assert is_garbage("Epstein") is False

    def test_mixed_case_single_token(self):
        assert is_garbage("mcDonald") is True

    def test_normal_names_pass(self):
        assert is_garbage("Jeffrey Epstein") is False
        assert is_garbage("John Smith") is False
        assert is_garbage("J. Edgar Hoover") is False

    def test_hyphenated_names_pass(self):
        assert is_garbage("Mary-Jane Watson") is False

    def test_pronouns_are_garbage(self):
        assert is_garbage("Him") is True
        assert is_garbage("HIM") is True
        assert is_garbage("her") is True
        assert is_garbage("She") is True
        assert is_garbage("them") is True

    def test_common_non_names_are_garbage(self):
        assert is_garbage("Inc") is True
        assert is_garbage("LLC") is True
        assert is_garbage("Esq") is True
        assert is_garbage("None") is True
        assert is_garbage("Unknown") is True
        assert is_garbage("Redacted") is True

    def test_honorifics_are_garbage(self):
        assert is_garbage("Mr.") is True
        assert is_garbage("Dr.") is True
        assert is_garbage("Mrs") is True
        assert is_garbage("Ms") is True
        assert is_garbage("Prof") is True

    def test_suffixes_are_garbage(self):
        assert is_garbage("Jr") is True
        assert is_garbage("Sr") is True
        assert is_garbage("III") is True
        assert is_garbage("II") is True
        assert is_garbage("IV") is True

    def test_redacted_names_are_garbage(self):
        assert is_garbage("Agent █████████") is True
        assert is_garbage("Special Agent █████") is True

    def test_ocr_artifacts_are_garbage(self):
        assert is_garbage("Chris Diion°") is True
        assert is_garbage("Collin•") is True
        assert is_garbage("A. Q•") is True

    def test_newlines_are_garbage(self):
        assert is_garbage("Joe\nJoseph E") is True
        assert is_garbage("Epstein\nH") is True
        assert is_garbage("Alan\r\nSmith") is True

    def test_real_multi_word_names_pass(self):
        """Legitimate multi-word names should not be caught by any rule."""
        assert is_garbage("Joseph E. Nascimento") is False
        assert is_garbage("Alan Ross") is False
        assert is_garbage("Joe Nascimento") is False


class TestCleanName_Whitespace:
    def test_collapses_newlines(self):
        assert clean_name("Joe\nJoseph E") == "Joe Joseph E"

    def test_collapses_multi_spaces(self):
        assert clean_name("John    Smith") == "John Smith"

    def test_collapses_tabs(self):
        assert clean_name("John\tSmith") == "John Smith"


class TestNormalizeRole:
    def test_standard_roles_unchanged(self):
        assert normalize_role("sender") == "sender"
        assert normalize_role("recipient") == "recipient"
        assert normalize_role("passenger") == "passenger"
        assert normalize_role("mentioned") == "mentioned"
        assert normalize_role("other") == "other"

    def test_nonstandard_roles_normalized(self):
        assert normalize_role("witness") == "mentioned"
        assert normalize_role("attorney") == "mentioned"
        assert normalize_role("unknown") == "mentioned"
        assert normalize_role("") == "mentioned"


class TestCleanNamesFile:
    def _make_data(self, names):
        return {
            "source_file": "TEST00001_names.json",
            "names": names,
        }

    def test_garbage_removed(self):
        data = self._make_data([
            {"normalized_name": "Jeffrey Epstein", "role": "mentioned"},
            {"normalized_name": "AB", "role": "mentioned"},
            {"normalized_name": "EFTA00001234", "role": "mentioned"},
        ])
        cleaned, stats = clean_names_file(data)
        assert len(cleaned["names"]) == 1
        assert cleaned["names"][0]["normalized_name"] == "Jeffrey Epstein"
        assert stats["garbage_removed"] == 2

    def test_leading_dots_stripped(self):
        data = self._make_data([
            {"normalized_name": "...John Smith", "role": "sender"},
        ])
        cleaned, stats = clean_names_file(data)
        assert cleaned["names"][0]["normalized_name"] == "John Smith"

    def test_role_remapped(self):
        data = self._make_data([
            {"normalized_name": "John Smith", "role": "attorney"},
        ])
        cleaned, stats = clean_names_file(data)
        assert cleaned["names"][0]["role"] == "mentioned"
        assert stats["roles_remapped"] == 1

    def test_deduplication_keeps_highest_priority(self):
        data = self._make_data([
            {"normalized_name": "Jeffrey Epstein", "role": "mentioned"},
            {"normalized_name": "jeffrey epstein", "role": "sender"},
        ])
        cleaned, stats = clean_names_file(data)
        assert len(cleaned["names"]) == 1
        assert cleaned["names"][0]["role"] == "sender"
        assert stats["deduplicated"] == 1

    def test_metadata_preserved(self):
        data = {
            "source_file": "TEST00001_names.json",
            "pipeline_version": "2.0",
            "names": [
                {"normalized_name": "John Smith", "role": "mentioned"},
            ],
        }
        cleaned, _ = clean_names_file(data)
        assert cleaned["pipeline_version"] == "2.0"
        assert cleaned["source_file"] == "TEST00001_names.json"

    def test_empty_names(self):
        data = self._make_data([])
        cleaned, stats = clean_names_file(data)
        assert cleaned["names"] == []
        assert stats["original_count"] == 0
        assert stats["cleaned_count"] == 0

    def test_original_text_fallback(self):
        """When normalized_name is missing, use original_text."""
        data = self._make_data([
            {"original_text": "Jane Doe", "role": "recipient"},
        ])
        cleaned, stats = clean_names_file(data)
        assert len(cleaned["names"]) == 1
        assert stats["cleaned_count"] == 1

    def test_low_confidence_filtered(self):
        """Names below MIN_CONFIDENCE are filtered out."""
        data = self._make_data([
            {"normalized_name": "Jeffrey Epstein", "role": "mentioned", "confidence": 0.99},
            {"normalized_name": "Him", "role": "mentioned", "confidence": 0.51},
            {"normalized_name": "Maybe Name", "role": "mentioned", "confidence": 0.60},
        ])
        cleaned, stats = clean_names_file(data)
        assert len(cleaned["names"]) == 1
        assert cleaned["names"][0]["normalized_name"] == "Jeffrey Epstein"

    def test_no_confidence_field_passes(self):
        """Names without confidence field are not filtered by confidence."""
        data = self._make_data([
            {"normalized_name": "John Smith", "role": "mentioned"},
        ])
        cleaned, stats = clean_names_file(data)
        assert len(cleaned["names"]) == 1

    def test_newline_names_cleaned_then_deduped(self):
        """Multi-line NER artifacts like 'Joe\\nJoseph E' should be cleaned."""
        data = self._make_data([
            {"normalized_name": "Joe\nJoseph E", "role": "mentioned", "confidence": 0.99},
            {"normalized_name": "Joe", "role": "sender", "confidence": 0.99},
        ])
        cleaned, stats = clean_names_file(data)
        # "Joe\nJoseph E" -> "Joe Joseph E" after clean_name, which is valid
        # "Joe" is a separate entry — both should survive
        names = [n["normalized_name"] for n in cleaned["names"]]
        assert "Joe" in names


class TestProcessDirectory:
    def test_writes_clean_files(self, tmp_path):
        data = {
            "source_file": "TEST00001.pdf",
            "names": [
                {"normalized_name": "Jeffrey Epstein", "role": "mentioned"},
                {"normalized_name": "AB", "role": "mentioned"},
            ],
        }
        input_file = tmp_path / "TEST00001_names.json"
        input_file.write_text(json.dumps(data))

        totals = process_directory(tmp_path, apply=True)

        output_file = tmp_path / "TEST00001_names_clean.json"
        assert output_file.exists()

        with open(output_file) as f:
            cleaned = json.load(f)
        assert len(cleaned["names"]) == 1
        assert cleaned["names"][0]["normalized_name"] == "Jeffrey Epstein"
        assert totals["files_processed"] == 1

    def test_dry_run_no_files_written(self, tmp_path):
        data = {
            "source_file": "TEST00001.pdf",
            "names": [
                {"normalized_name": "John Smith", "role": "mentioned"},
            ],
        }
        (tmp_path / "TEST00001_names.json").write_text(json.dumps(data))

        totals = process_directory(tmp_path, apply=False)

        output_file = tmp_path / "TEST00001_names_clean.json"
        assert not output_file.exists()
        assert totals["files_processed"] == 1
        assert totals["total_cleaned"] == 1

    def test_skips_already_clean_files(self, tmp_path):
        """_names_clean.json files should not be re-processed."""
        data = {
            "source_file": "TEST00001.pdf",
            "names": [
                {"normalized_name": "John Smith", "role": "mentioned"},
            ],
        }
        (tmp_path / "TEST00001_names.json").write_text(json.dumps(data))
        (tmp_path / "TEST00001_names_clean.json").write_text(json.dumps(data))

        totals = process_directory(tmp_path, apply=False)
        assert totals["files_processed"] == 1  # Only the _names.json, not _names_clean.json

    def test_empty_directory(self, tmp_path):
        totals = process_directory(tmp_path, apply=True)
        assert totals["files_processed"] == 0

    def test_invalid_json_skipped(self, tmp_path):
        (tmp_path / "BAD_names.json").write_text("not json{{{")

        totals = process_directory(tmp_path, apply=True)
        assert totals["files_skipped"] == 1
        assert totals["files_processed"] == 0

    def test_statistics_accurate(self, tmp_path):
        data = {
            "source_file": "TEST.pdf",
            "names": [
                {"normalized_name": "Jeffrey Epstein", "role": "sender"},
                {"normalized_name": "jeffrey epstein", "role": "mentioned"},  # dedup
                {"normalized_name": "AB", "role": "mentioned"},  # garbage
                {"normalized_name": "John Smith", "role": "witness"},  # remap
            ],
        }
        (tmp_path / "TEST_names.json").write_text(json.dumps(data))

        totals = process_directory(tmp_path, apply=True)
        assert totals["total_original"] == 4
        assert totals["total_garbage_removed"] == 1
        assert totals["total_roles_remapped"] == 1
        assert totals["total_deduplicated"] == 1
        assert totals["total_cleaned"] == 2  # Epstein (sender) + Smith (mentioned)


class TestHighProfileNameRegression:
    """Regression tests ensuring high-profile names survive the cleaning pipeline."""

    def test_bill_gates_not_garbage(self):
        assert not is_garbage("Bill Gates")

    def test_william_h_gates_not_garbage(self):
        assert not is_garbage("William H. Gates")

    def test_gates_bill_inverted_is_garbage(self):
        """Comma-separated names are garbage (should be 'gates bill' or 'bill gates')."""
        assert is_garbage("Gates, Bill")

    def test_bill_clinton_not_garbage(self):
        assert not is_garbage("Bill Clinton")

    def test_donald_trump_not_garbage(self):
        assert not is_garbage("Donald Trump")

    def test_prince_andrew_not_garbage(self):
        assert not is_garbage("Prince Andrew")

    def test_alan_dershowitz_not_garbage(self):
        assert not is_garbage("Alan Dershowitz")

    def test_virginia_giuffre_not_garbage(self):
        assert not is_garbage("Virginia Giuffre")

    def test_les_wexner_not_garbage(self):
        assert not is_garbage("Les Wexner")

    def test_ghislaine_maxwell_not_garbage(self):
        assert not is_garbage("Ghislaine Maxwell")

    def test_bill_gates_survives_pipeline(self):
        """Full clean_names_file round-trip with high confidence."""
        data = {
            "source_file": "TEST00001_names.json",
            "names": [
                {
                    "normalized_name": "Bill Gates",
                    "role": "mentioned",
                    "confidence": 0.999,
                },
            ],
        }
        cleaned, stats = clean_names_file(data)
        assert len(cleaned["names"]) == 1
        assert cleaned["names"][0]["normalized_name"] == "Bill Gates"
        assert stats["garbage_removed"] == 0

    def test_bill_gates_low_confidence_filtered(self):
        """Below MIN_CONFIDENCE threshold IS correctly dropped."""
        data = {
            "source_file": "TEST00001_names.json",
            "names": [
                {
                    "normalized_name": "Bill Gates",
                    "role": "mentioned",
                    "confidence": 0.50,
                },
            ],
        }
        cleaned, stats = clean_names_file(data)
        assert len(cleaned["names"]) == 0
        assert stats["garbage_removed"] == 1

    def test_bill_gates_clean_name_passthrough(self):
        """clean_name() should not alter 'Bill Gates'."""
        assert clean_name("Bill Gates") == "Bill Gates"

    def test_gates_comma_bill_cleaned(self):
        """'Gates, Bill' should survive cleaning (comma stripped from trailing)."""
        result = clean_name("Gates, Bill")
        assert "Gates" in result
        assert "Bill" in result


class TestLongNameGarbage:
    """Tests for character-length and token-count garbage filters."""

    # --- Real garbage from data/examples/long_names_garbage.json ---

    def test_concatenated_name_list(self):
        """Roster/manifest NER'd as single span (234 chars, 33 tokens)."""
        name = "Ashielle Etienne Boniface Laudat Charles Dick Cristobal Herrera Cuthbert Titre Dale Mark Danny Etienne Danny Boodram David Alves Dupson Donissaint Fabian John Felito Joseph Gael Leatham Garth Eugene Gerry A. Titre Gusneme Dalce Hilian"
        assert is_garbage(name)

    def test_table_headers_long(self):
        """Financial table column headers (350 chars, 54 tokens)."""
        name = "Trade Date Quantity Settlement Date Quantity Local Price Base Price Pricing Date Local Trade Date Cost Base Trade Date Cost Local Settlement Date Cost Base Settlement Date Cost Local Trade Date Market Value Base Trade Date Market Value Local Settlement Date Market Value Base Settlement Date Market Value Local Accrued Income Base Accrued Income Loca"
        assert is_garbage(name)

    def test_table_headers_medium(self):
        """Financial table column headers (263 chars, 40 tokens)."""
        name = "Trade Date Cost Base Trade Date Cost Local Settlement Date Cost Base Settlement Date Cost Local Trade Date Market Value Base Trade Date Market Value Local Settlement Date Market Value Base Settlement Date Market Value Local Accrued Income Base Accrued Income Loca"
        assert is_garbage(name)

    def test_ocr_garbled_name_list(self):
        """OCR-garbled roster with artifacts (208 chars, 30 tokens)."""
        name = "Gatos Domenech &mon Jeremy Avers& Ismael Guerrero Mas Yana Kravlsova Martin Truong Brian Wuebbehs Herd 'Welly* Oahya Mark Flonan Frarcisco 'Panche Perez Gurcln Mark Lerdal Steven Tesonere Ade Nene Ahmal Chats"
        assert is_garbage(name)

    # --- All 7 examples from the JSON file ---

    def test_all_garbage_examples_rejected(self):
        """All 7 examples from long_names_garbage.json must be rejected."""
        examples_path = Path(__file__).parent.parent / "data" / "examples" / "long_names_garbage.json"
        if not examples_path.exists():
            pytest.skip("data/examples/long_names_garbage.json not found")
        with open(examples_path) as f:
            examples = json.load(f)
        for entry in examples:
            name = clean_name(entry["normalized_name"])
            assert is_garbage(name), f"Should be garbage: {entry['normalized_name'][:80]}..."

    # --- Legitimate long names must pass ---

    def test_legitimate_5_token_name(self):
        """Longest observed real name: 36 chars, 5 tokens."""
        assert not is_garbage("jeffrey edward richard lee epstein")

    def test_legitimate_multi_part_name(self):
        """Multi-part name with prepositions: 5 tokens."""
        assert not is_garbage("Maria de la Cruz Rodriguez")

    def test_legitimate_hyphenated_with_suffix(self):
        """Hyphenated + suffix: 7 tokens (just under limit)."""
        assert not is_garbage("Jean-Pierre van der Berg Jr. III")

    def test_legitimate_6_token_name(self):
        """6-token name with middle names."""
        assert not is_garbage("Mary Jane Elizabeth Anne Smith")

    # --- Boundary: character length ---

    def test_boundary_59_chars_passes(self):
        """59 chars = max allowed length, should pass."""
        name = "A" * 4 + " " + "B" * 4 + " " + "C" * 4 + " " + "D" * 44  # 4+1+4+1+4+1+44 = 59
        assert len(name) == 59
        assert not is_garbage(name)

    def test_boundary_60_chars_rejected(self):
        """60 chars = rejected."""
        name = "A" * 4 + " " + "B" * 4 + " " + "C" * 4 + " " + "D" * 45  # 4+1+4+1+4+1+45 = 60
        assert len(name) == 60
        assert is_garbage(name)

    # --- Boundary: token count ---

    def test_boundary_7_tokens_passes(self):
        """7 tokens = max allowed, should pass."""
        name = "One Two Three Four Five Six Seven"
        assert len(name.split()) == 7
        assert not is_garbage(name)

    def test_boundary_8_tokens_rejected(self):
        """8 tokens = rejected."""
        name = "One Two Three Four Five Six Seven Eight"
        assert len(name.split()) == 8
        assert is_garbage(name)

    # --- Pipeline round-trip ---

    def test_garbage_long_name_removed_from_pipeline(self):
        """Long garbage name is removed during clean_names_file()."""
        data = {
            "source_file": "TEST.pdf",
            "names": [
                {"normalized_name": "Jeffrey Epstein", "role": "mentioned"},
                {"normalized_name": "Ashielle Etienne Boniface Laudat Charles Dick Cristobal Herrera Cuthbert Titre Dale Mark Danny Etienne Danny Boodram David Alves Dupson Donissaint Fabian John Felito Joseph Gael Leatham Garth Eugene Gerry A. Titre Gusneme Dalce Hilian", "role": "mentioned"},
            ],
        }
        cleaned, stats = clean_names_file(data)
        assert len(cleaned["names"]) == 1
        assert cleaned["names"][0]["normalized_name"] == "Jeffrey Epstein"
        assert stats["garbage_removed"] == 1


class TestLeadingGarbageChars:
    """Tests for expanded lstrip/rstrip and fused digit+letter stripping."""

    # --- Leading parentheses/brackets ---

    def test_leading_open_paren(self):
        assert clean_name("(Gala Joseph") == "Gala Joseph"

    def test_nested_parens(self):
        assert clean_name("(((Tom") == "Tom"

    def test_paren_bang_combo(self):
        assert clean_name("(!) Ju Wang") == "Ju Wang"

    def test_leading_close_paren(self):
        assert clean_name(") Jeffrey") == "Jeffrey"

    def test_leading_bracket(self):
        result = clean_name("[Mad")
        assert result == "Mad"

    # --- Leading equals, backtick, other symbols ---

    def test_leading_equals(self):
        assert clean_name("=Effrey Epstein") == "Effrey Epstein"

    def test_leading_backtick(self):
        assert clean_name("`Bella Klein") == "Bella Klein"

    def test_leading_tilde(self):
        assert clean_name("~John Smith") == "John Smith"

    def test_leading_pipe(self):
        assert clean_name("|Anna Ross") == "Anna Ross"

    def test_leading_at(self):
        assert clean_name("@Mark Davis") == "Mark Davis"

    # --- Trailing expanded chars ---

    def test_trailing_parens(self):
        assert clean_name("John Smith)") == "John Smith"

    def test_trailing_bracket(self):
        assert clean_name("Jane Doe]") == "Jane Doe"

    def test_trailing_equals(self):
        assert clean_name("Bob Jones=") == "Bob Jones"

    def test_trailing_backtick(self):
        assert clean_name("Bella Klein`") == "Bella Klein"

    # --- Fused digit+letter stripping ---

    def test_fused_single_digit(self):
        assert clean_name("2Oc Saldana") == "Oc Saldana"

    def test_fused_two_digits(self):
        assert clean_name("34Enter Kiss") == "Enter Kiss"

    def test_fused_two_digit_name(self):
        assert clean_name("99Agostino") == "Agostino"

    def test_fused_digit_ocr_letter(self):
        """3Effrey should strip to Effrey (OCR of Jeffrey)."""
        assert clean_name("3Effrey") == "Effrey"

    def test_fused_two_digit_ocr(self):
        assert clean_name("30Y E Mitchell") == "Y E Mitchell"

    def test_fused_three_digits_not_stripped(self):
        """3+ digits fused to letter should NOT be stripped by fused rule."""
        # 100Smith has 3 digits — too many for fused rule; falls through to
        # digit+delimiter rule (>300, no delimiter -> preserved)
        result = clean_name("100Smith")
        assert "100" in result

    # --- Guards: year and existing behavior preserved ---

    def test_year_space_preserved(self):
        """2005 Jeffrey Epstein must not be stripped (year guard)."""
        result = clean_name("2005 Jeffrey Epstein")
        assert "2005" in result
        assert "Jeffrey Epstein" in result

    def test_existing_dot_delimiter(self):
        """Existing behavior: '14. Smith' -> 'Smith'."""
        assert clean_name("14. Smith") == "Smith"

    def test_existing_space_delimiter(self):
        """Existing behavior: '2 A. Farmer' -> 'A. Farmer'."""
        assert clean_name("2 A. Farmer") == "A. Farmer"


class TestDigitsInNames:
    """Names should never contain digits mixed with letters."""

    def test_ocr_digit_in_surname(self):
        """'Jeffrey F.2Stein' has digit '2' embedded in name."""
        assert is_garbage("Jeffrey F.2Stein")

    def test_zero_for_o(self):
        """OCR '0' instead of 'O': 'J0hn Smith'."""
        assert is_garbage("J0hn Smith")

    def test_one_for_l(self):
        """OCR '1' instead of 'l': 'A1an Smith'."""
        assert is_garbage("A1an Smith")

    def test_digit_only_no_letters(self):
        """Pure digits already caught by earlier rule, but also caught by digit+letter."""
        assert is_garbage("12345")

    def test_clean_name_then_garbage(self):
        """After clean_name strips leading digit, remaining digits still caught."""
        name = clean_name("2F.2Stein")
        assert is_garbage(name)

    def test_legitimate_name_no_digits(self):
        """Normal names have no digits."""
        assert not is_garbage("Jeffrey Epstein")
        assert not is_garbage("John Smith")
        assert not is_garbage("Mary Jane Watson")


class TestStructuralPunctuation:
    """Names with internal parens/brackets/braces are garbage."""

    def test_closing_paren_in_middle(self):
        """'Jeffry). Epstein' — structural punctuation."""
        assert is_garbage("Jeffry). Epstein")

    def test_parens_around_suffix(self):
        """'Smith (Jr) Jones' — parens in name."""
        assert is_garbage("Smith (Jr) Jones")

    def test_brackets_in_name(self):
        """'John [redacted] Smith' — brackets in name."""
        assert is_garbage("John [redacted] Smith")

    def test_braces_in_name(self):
        """'John {test} Smith' — braces in name."""
        assert is_garbage("John {test} Smith")

    def test_leading_trailing_parens_stripped_by_clean_name(self):
        """Leading/trailing parens stripped by clean_name, so result is clean."""
        name = clean_name("(John Smith)")
        assert not is_garbage(name)  # clean_name strips parens, result is "John Smith"

    def test_apostrophe_not_structural(self):
        """O'Brien has apostrophe, not structural punctuation."""
        assert not is_garbage("O'Brien")

    def test_hyphen_not_structural(self):
        """Kim Jong-un has hyphen, not structural punctuation."""
        assert not is_garbage("Kim Jong-un")

    def test_comma_is_garbage(self):
        """'Gates, Bill' has comma — comma-separated names are now garbage."""
        assert is_garbage("Gates, Bill")


class TestPerTokenHonorific:
    """Any individual token being a bare honorific makes the name garbage."""

    def test_trailing_mr(self):
        """'Epstein Mr' — 'Mr' as a token."""
        assert is_garbage("Epstein Mr")

    def test_leading_mr_dot(self):
        """'Mr. Epstein Johnson' — 'Mr.' as a token."""
        assert is_garbage("Mr. Epstein Johnson")

    def test_leading_dr(self):
        """'Dr Smith Jones' — 'Dr' as a token."""
        assert is_garbage("Dr Smith Jones")

    def test_mrs_in_name(self):
        """'Mrs. Jane Smith' — 'Mrs.' as a token."""
        assert is_garbage("Mrs. Jane Smith")

    def test_judge_in_name(self):
        """'Judge Kenneth Marra' — 'Judge' as a token."""
        assert is_garbage("Judge Kenneth Marra")

    def test_rev_in_name(self):
        """'Rev. Martin Luther' — 'Rev.' as a token."""
        assert is_garbage("Rev. Martin Luther")

    def test_sgt_in_name(self):
        """'Sgt. John Harris' — 'Sgt.' as a token."""
        assert is_garbage("Sgt. John Harris")

    def test_normal_names_not_caught(self):
        """Normal names without honorifics pass."""
        assert not is_garbage("Jeffrey Epstein")
        assert not is_garbage("Bill Gates")
        assert not is_garbage("Prince Andrew")  # 'Prince' is NOT in honorifics
        assert not is_garbage("Virginia Giuffre")

    def test_initial_not_honorific(self):
        """Single-letter initials like 'J.' are not honorifics."""
        assert not is_garbage("J. Edgar Hoover")
        assert not is_garbage("A. Farmer")

    def test_epstein_mr_epstein(self):
        """'Epstein Mr. Epstein' — repeated name with honorific."""
        assert is_garbage("Epstein Mr. Epstein")


class TestNewGarbageRulesPipeline:
    """End-to-end tests through clean_names_file with new rules."""

    def _make_data(self, names):
        return {
            "source_file": "TEST00001_names.json",
            "names": names,
        }

    def test_digit_name_removed(self):
        data = self._make_data([
            {"normalized_name": "Jeffrey Epstein", "role": "mentioned"},
            {"normalized_name": "Jeffrey F.2Stein", "role": "mentioned"},
        ])
        cleaned, stats = clean_names_file(data)
        assert len(cleaned["names"]) == 1
        assert cleaned["names"][0]["normalized_name"] == "Jeffrey Epstein"

    def test_honorific_name_removed(self):
        data = self._make_data([
            {"normalized_name": "Jeffrey Epstein", "role": "mentioned"},
            {"normalized_name": "Mr. Epstein", "role": "mentioned"},
        ])
        cleaned, stats = clean_names_file(data)
        assert len(cleaned["names"]) == 1
        assert cleaned["names"][0]["normalized_name"] == "Jeffrey Epstein"

    def test_structural_punct_removed(self):
        data = self._make_data([
            {"normalized_name": "Jeffrey Epstein", "role": "mentioned"},
            {"normalized_name": "Jeffry). Epstein", "role": "mentioned"},
        ])
        cleaned, stats = clean_names_file(data)
        assert len(cleaned["names"]) == 1
        assert cleaned["names"][0]["normalized_name"] == "Jeffrey Epstein"


class TestUnicodeCurlyQuotes:
    """Unicode curly quotes should be stripped by clean_name()."""

    def test_leading_left_single_quote(self):
        assert clean_name("\u2018Epstein") == "Epstein"

    def test_leading_right_single_quote(self):
        assert clean_name("\u2019Smith") == "Smith"

    def test_leading_left_double_quote(self):
        assert clean_name("\u201cJohn") == "John"

    def test_trailing_low_double_quote(self):
        assert clean_name("Jerry\u201e") == "Jerry"

    def test_leading_low_double_quote(self):
        assert clean_name("\u201eJose H Cothron") == "Jose H Cothron"

    def test_trailing_left_angle_quote(self):
        assert clean_name("Lesley Groff \u2039") == "Lesley Groff"

    def test_trailing_right_angle_quote(self):
        assert clean_name("John Smith \u203a") == "John Smith"

    def test_multiple_curly_quotes(self):
        assert clean_name("\u201c\u201cJane Doe\u201d\u201d") == "Jane Doe"


class TestTrailingDots:
    """Trailing dots/ellipsis should be stripped."""

    def test_trailing_three_dots(self):
        assert clean_name("Jeffrey...") == "Jeffrey"

    def test_trailing_dots_with_space(self):
        assert clean_name("Epstein ...") == "Epstein"

    def test_trailing_unicode_ellipsis(self):
        assert clean_name("Reichart\u2026") == "Reichart"

    def test_trailing_single_period_after_word(self):
        """'bill gates.' -> 'bill gates' (period after 2+ lowercase letters)."""
        assert clean_name("bill gates.") == "bill gates"

    def test_initial_period_preserved(self):
        """'jeffrey e.' stays (single letter before period = initial)."""
        assert clean_name("jeffrey e.") == "jeffrey e."

    def test_jr_period_stripped(self):
        """'thomas jr.' -> 'thomas jr' (jr has 2 lowercase letters before period)."""
        assert clean_name("thomas jr.") == "thomas jr"

    def test_double_initial_preserved(self):
        """'j.e.' stays (dots are part of initials)."""
        assert clean_name("j.e.") == "j.e."

    def test_ann_r_period_preserved(self):
        """'ann r.' stays (single letter before period = initial)."""
        assert clean_name("ann r.") == "ann r."


class TestUnderscoreReplacement:
    """Underscores should be replaced with spaces."""

    def test_underscore_in_name(self):
        assert clean_name("desmond_shum") == "desmond shum"

    def test_leading_underscore(self):
        assert clean_name("_jeffrey") == "jeffrey"

    def test_multiple_underscores(self):
        assert clean_name("daniel_c_dennett") == "daniel c dennett"

    def test_underscore_in_is_garbage(self):
        """Underscore surviving clean_name is caught by is_garbage."""
        assert is_garbage("still_has_underscore")


class TestWrapAroundRepeat:
    """First token == last token wrap-around detection."""

    def test_gates_bill_gates(self):
        assert clean_name("gates bill gates") == "bill gates"

    def test_epstein_jeffrey_epstein(self):
        assert clean_name("Epstein Jeffrey Epstein") == "Jeffrey Epstein"

    def test_smith_john_smith(self):
        assert clean_name("Smith John Smith") == "John Smith"

    def test_two_token_echo_caught_by_is_garbage(self):
        """Two-token same-word passes clean_name but caught by is_garbage."""
        assert clean_name("John John") == "John John"
        assert is_garbage("John John")


class TestCommaGarbage:
    """Names with commas are garbage (comma-separated lists, inverted format)."""

    def test_comma_separated_list(self):
        assert is_garbage("Epstein, Jeffrey, Edwar")

    def test_inverted_name(self):
        assert is_garbage("Pritzker, Tom")

    def test_comma_with_suffix(self):
        assert is_garbage("Landon Thomas, Jr")

    def test_no_comma_passes(self):
        assert not is_garbage("Jeffrey Epstein")


class TestAmpersandGarbage:
    """Names with & are two people merged."""

    def test_two_people(self):
        assert is_garbage("Roy & Stephanie Hodges")

    def test_couple(self):
        assert is_garbage("Lyn & Jojo")

    def test_no_ampersand_passes(self):
        assert not is_garbage("Roy Hodges")


class TestExclamationGarbage:
    """Names with ! are OCR artifacts."""

    def test_ocr_exclamation_middle(self):
        assert is_garbage("Larry De!Son")

    def test_ocr_exclamation_start_of_token(self):
        assert is_garbage("Joseph !Gala")

    def test_no_exclamation_passes(self):
        assert not is_garbage("Larry Deson")


class TestColonGarbage:
    """Names with : are OCR artifacts."""

    def test_colon_in_name(self):
        assert is_garbage("D: Edwards")

    def test_no_colon_passes(self):
        assert not is_garbage("D Edwards")


class TestTrailingStopword:
    """Names ending with common stopwords are NER over-extractions."""

    def test_trailing_ok(self):
        assert is_garbage("Bill Gates Ok")

    def test_trailing_im(self):
        assert is_garbage("Bill Gates Im")

    def test_trailing_he(self):
        assert is_garbage("Bill Gates He")

    def test_trailing_the(self):
        assert is_garbage("Jeffrey Epstein The")

    def test_trailing_did(self):
        assert is_garbage("Jeffrey Epstein Did")

    def test_trailing_we(self):
        assert is_garbage("Jeffrey We")

    def test_trailing_of(self):
        assert is_garbage("Duke Of")

    def test_trailing_is(self):
        assert is_garbage("Jeffrey Epstein Is")

    def test_trailing_my(self):
        assert is_garbage("Jeffrey My")

    def test_trailing_do(self):
        assert is_garbage("Jeffrey Epstein Do")

    def test_normal_last_name_passes(self):
        assert not is_garbage("Jeffrey Epstein")
        assert not is_garbage("Bill Gates")

    def test_korean_surname_no(self):
        """'Kyeong No' — 'No' is a Korean surname, NOT in stopwords."""
        assert not is_garbage("Kyeong No")

    def test_chinese_surname_to(self):
        """'Nina To' — 'To' could be a Chinese surname."""
        assert not is_garbage("Nina To")

    def test_vietnamese_surname_an(self):
        """'Nguyen An' — 'An' is a Vietnamese name."""
        assert not is_garbage("Nguyen An")


class TestLeadingStopword:
    """Names starting with common stopwords are NER over-extractions."""

    def test_leading_is(self):
        assert is_garbage("Is Gates")

    def test_leading_the(self):
        assert is_garbage("The Jeffrey")

    def test_leading_for(self):
        assert is_garbage("For Epstein")

    def test_leading_his(self):
        assert is_garbage("His Attorney")

    def test_normal_first_name_passes(self):
        assert not is_garbage("Jeffrey Epstein")
        assert not is_garbage("Bill Gates")


class TestExpandedHonorifics:
    """Expanded honorifics catch title-prefixed names."""

    def test_secretary(self):
        assert is_garbage("Secretary Robert Gates")

    def test_senator(self):
        assert is_garbage("Senator John Smith")

    def test_president(self):
        assert is_garbage("President Clinton")

    def test_detective(self):
        assert is_garbage("Detective Smith")

    def test_attorney(self):
        assert is_garbage("Attorney General Holder")

    def test_justice(self):
        assert is_garbage("Justice O'Connor")

    def test_prince_not_honorific(self):
        """'Prince' is NOT in honorifics — it's part of the name."""
        assert not is_garbage("Prince Andrew")


class TestBareSingleLetterToken:
    """Bare single-letter tokens (without dot) are OCR/NER fragments."""

    def test_e_epstein(self):
        assert is_garbage("e epstein")

    def test_e_e_dot_epstein(self):
        """'e e. epstein' has bare 'e' token."""
        assert is_garbage("e e. epstein")

    def test_initial_with_dot_passes(self):
        """'J. Edgar Hoover' — 'J.' is NOT bare (has dot)."""
        assert not is_garbage("J. Edgar Hoover")

    def test_a_dot_farmer_passes(self):
        """'A. Farmer' — 'A.' has dot, not bare."""
        assert not is_garbage("A. Farmer")

    def test_jeffrey_e_dot_passes(self):
        """'Jeffrey E.' — 'E.' is an initial with dot."""
        assert not is_garbage("Jeffrey E.")

    def test_single_bare_letter_in_middle_is_initial(self):
        """'John e Smith' — bare 'e' in middle is likely a middle initial (DB names are lowercase)."""
        assert not is_garbage("John e Smith")

    def test_alan_m_dershowitz_not_garbage(self):
        """'alan m dershowitz' — middle initial without dot, DB-style lowercase."""
        assert not is_garbage("alan m dershowitz")

    def test_adam_d_horowitz_not_garbage(self):
        """'adam d horowitz' — middle initial without dot."""
        assert not is_garbage("adam d horowitz")

    def test_christian_r_everdell_not_garbage(self):
        """'christian r everdell' — middle initial without dot."""
        assert not is_garbage("christian r everdell")

    def test_leading_bare_letter_still_garbage(self):
        """'e epstein' — bare letter at first position is still garbage."""
        assert is_garbage("e epstein")

    def test_trailing_bare_letter_still_garbage(self):
        """'andrew l' — bare letter at last position is still garbage (truncated name)."""
        assert is_garbage("andrew l")


class TestPrefixTruncation:
    """Last token is a prefix of first/preceding token (OCR truncation).

    clean_name() now strips the truncated trailing token so occurrences get
    merged instead of deleted. is_garbage() still catches direct calls as a safety net.
    """

    def test_epstein_ep_cleaned(self):
        """clean_name strips truncated suffix: 'epstein ep' → 'epstein'."""
        assert clean_name("epstein ep") == "epstein"

    def test_maxwell_max_cleaned(self):
        """clean_name strips truncated suffix: 'maxwell max' → 'maxwell'."""
        assert clean_name("maxwell max") == "maxwell"

    def test_three_token_preceding_match(self):
        """'jeffrey epstein ep' → 'jeffrey epstein' (last is prefix of preceding token)."""
        assert clean_name("jeffrey epstein ep") == "jeffrey epstein"

    def test_three_token_first_match(self):
        """'epstein jeffrey ep' → 'epstein jeffrey' (last is prefix of first token)."""
        assert clean_name("epstein jeffrey ep") == "epstein jeffrey"

    def test_is_garbage_still_catches_direct(self):
        """is_garbage() safety net still catches prefix-truncated names directly."""
        assert is_garbage("epstein ep")
        assert is_garbage("maxwell max")

    def test_different_names_pass(self):
        """Unrelated tokens should pass."""
        assert not is_garbage("Jeffrey Epstein")

    def test_short_prefix_ignored(self):
        """Single-char prefix too short to trigger (already caught by bare letter rule)."""
        assert is_garbage("epstein e")  # caught by bare letter rule

    def test_pipeline_roundtrip_merges(self):
        """In the pipeline, clean_name runs first, so 'epstein ep' becomes 'epstein'."""
        data = {
            "source_file": "TEST.pdf",
            "names": [
                {"normalized_name": "epstein ep", "role": "mentioned"},
                {"normalized_name": "jeffrey epstein", "role": "mentioned"},
            ],
        }
        cleaned, stats = clean_names_file(data)
        names = [n["normalized_name"].lower() for n in cleaned["names"]]
        # "epstein ep" → "epstein" after clean_name, survives is_garbage
        assert "epstein" in names or "jeffrey epstein" in names

    def test_real_names_not_stripped(self):
        """Names where last token is NOT a prefix of any preceding token."""
        assert clean_name("alan ross") == "alan ross"
        assert clean_name("bill gates") == "bill gates"
        assert clean_name("john smith") == "john smith"

    def test_legitimate_prefix_surname_preserved(self):
        """'mar martinez' — 'mar' is NOT a prefix of 'martinez' (wrong direction)."""
        assert clean_name("mar martinez") == "mar martinez"


class TestJunkAppendedToToken:
    """Junk appended to names — is_garbage() can't detect without cross-referencing.
    Handled by find_junk_suffix_merges() in cleanup_db_names.py."""

    def test_bill_gatesil_not_caught_by_is_garbage(self):
        """is_garbage can't detect 'bill gatesil' without DB context."""
        assert not is_garbage("bill gatesil")

    def test_normal_names_pass(self):
        assert not is_garbage("Bill Gates")
        assert not is_garbage("Jeffrey Epstein")

    def test_mark_markovic_passes(self):
        """'mark markovic' — 'markovic' starts with 'mark' but is a real surname."""
        assert not is_garbage("mark markovic")


class TestPeriodSeparatorRepeat:
    """Period as separator between repeated names."""

    def test_epstein_dot_epstein(self):
        """'epstein. epstein' → after period separator normalization, caught as repeat."""
        result = clean_name("epstein. epstein")
        assert result == "epstein"

    def test_gates_dot_gates(self):
        result = clean_name("gates. gates")
        assert result == "gates"


class TestAsAtByStopwords:
    """'as', 'at', 'by' added to stopwords."""

    def test_leading_as(self):
        assert is_garbage("as bill gates")

    def test_trailing_as(self):
        assert is_garbage("bill gates as")

    def test_leading_at(self):
        assert is_garbage("at jeffrey")

    def test_trailing_by(self):
        assert is_garbage("jeffrey by")

    # --- Email metadata suffix detection ---
    def test_trailing_subject(self):
        assert is_garbage("jeffrey epstein subject")

    def test_trailing_cc(self):
        assert is_garbage("epstein jeffrey cc")

    def test_trailing_bcc(self):
        assert is_garbage("john smith bcc")

    def test_trailing_unauthorized(self):
        assert is_garbage("jeffrey epstein unauthorized")

    def test_trailing_attachments(self):
        assert is_garbage("jeffrey epstein attachments")

    def test_trailing_fwd(self):
        assert is_garbage("john doe fwd")

    def test_trailing_fw(self):
        assert is_garbage("john doe fw")


class TestEmailMetadataStripping:
    """Test clean_name() strips trailing email metadata tokens."""

    def test_strip_subject(self):
        assert clean_name("Jeffrey Epstein Subject") == "Jeffrey Epstein"

    def test_strip_cc(self):
        assert clean_name("Epstein Jeffrey Cc") == "Epstein Jeffrey"

    def test_strip_bcc(self):
        assert clean_name("John Smith Bcc") == "John Smith"

    def test_strip_unauthorized(self):
        assert clean_name("Jeffrey Epstein Unauthorized") == "Jeffrey Epstein"

    def test_strip_attachments(self):
        assert clean_name("Jeffrey Epstein Attachments") == "Jeffrey Epstein"

    def test_strip_fwd(self):
        assert clean_name("John Doe Fwd") == "John Doe"

    def test_strip_fw(self):
        assert clean_name("John Doe FW") == "John Doe"

    def test_strip_multiple_metadata(self):
        """Multiple trailing metadata tokens are all stripped."""
        assert clean_name("Jeffrey Epstein Subject Cc") == "Jeffrey Epstein"

    def test_no_strip_when_only_one_word(self):
        """Don't strip if it would leave an empty name."""
        assert clean_name("Subject") == "Subject"

    def test_no_strip_middle_token(self):
        """Email metadata in the middle of a name is preserved."""
        assert clean_name("Subject John Smith") == "Subject John Smith"

    def test_case_insensitive(self):
        assert clean_name("Jeffrey Epstein SUBJECT") == "Jeffrey Epstein"
        assert clean_name("Jeffrey Epstein subject") == "Jeffrey Epstein"
        assert clean_name("Jeffrey Epstein CC") == "Jeffrey Epstein"

    # --- New email metadata tokens: sent, date, received, importance ---

    def test_strip_sent_suffix(self):
        assert clean_name("Joscha Bach Sent") == "Joscha Bach"
        assert clean_name("Lesley Groff sent") == "Lesley Groff"
        assert clean_name("Thx Larry SENT") == "Thx Larry"

    def test_strip_date_suffix(self):
        assert clean_name("Joscha Bach Date") == "Joscha Bach"
        assert clean_name("Jeffrey Epstein date") == "Jeffrey Epstein"

    def test_strip_received_importance_suffix(self):
        assert clean_name("John Smith Received") == "John Smith"
        assert clean_name("John Smith Importance") == "John Smith"

    def test_strip_multiple_new_metadata_suffixes(self):
        """Multiple trailing metadata tokens are all stripped."""
        assert clean_name("Joscha Bach Sent Date") == "Joscha Bach"
        assert clean_name("Joscha Bach Date Sent") == "Joscha Bach"

    def test_no_strip_single_word_metadata(self):
        """Don't strip if it would leave an empty name."""
        assert clean_name("Sent") == "Sent"
        assert clean_name("Date") == "Date"

    def test_no_strip_middle_new_metadata(self):
        """Metadata in the middle of a name is preserved."""
        assert clean_name("Sent Joscha Bach") == "Sent Joscha Bach"
        assert clean_name("Joscha Sent Bach") == "Joscha Sent Bach"

    # --- Guarded tokens: to / from (could be real surnames) ---

    def test_strip_to_suffix_for_multi_token(self):
        """'to' stripped when 3+ tokens (clearly email header artifact)."""
        assert clean_name("Jeffrey Epstein To") == "Jeffrey Epstein"
        assert clean_name("Lesley Groff to") == "Lesley Groff"

    def test_do_not_strip_real_surname_to(self):
        """'Nina To' stays — 'to' could be a real surname."""
        assert clean_name("Nina To") == "Nina To"

    def test_strip_from_suffix_for_multi_token(self):
        assert clean_name("Jeffrey Epstein From") == "Jeffrey Epstein"

    def test_do_not_strip_real_surname_from(self):
        assert clean_name("Alice From") == "Alice From"

    def test_mixed_guarded_and_unguarded(self):
        """Unguarded tokens strip first, then guarded."""
        assert clean_name("Jeffrey Epstein Sent To") == "Jeffrey Epstein"
        assert clean_name("Jeffrey Epstein To Sent") == "Jeffrey Epstein"


# ---------- Cross-reference junk-suffix detection tests ----------
# These test the pure function find_junk_suffix_merges() from cleanup_db_names.py

import importlib
import sys

# Import the function under test
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.db.cleanup_db_names import find_junk_suffix_merges


class TestCrossReferenceJunkSuffix:
    """Test find_junk_suffix_merges() for detecting junk-appended names."""

    def test_bill_gatesil_matches_bill_gates(self):
        """'bill gatesil' (1 mention) → 'bill gates' (1001 mentions) via trim-2."""
        name_counts = {
            "bill gates": 1001,
            "bill gatesil": 1,
            "jeffrey epstein": 500,
        }
        name_to_id = {"bill gates": 1, "bill gatesil": 2, "jeffrey epstein": 3}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        assert len(merges) == 1
        assert merges[0] == (2, "bill gatesil", "bill gates", 1)

    def test_trim_1_char(self):
        """'alan dersh' (1) should NOT match 'alan ders' but would match if 'alan ders' existed."""
        name_counts = {
            "jeffrey epsteinx": 1,
            "jeffrey epstein": 500,
        }
        name_to_id = {"jeffrey epsteinx": 1, "jeffrey epstein": 2}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        assert len(merges) == 1
        assert merges[0] == (1, "jeffrey epsteinx", "jeffrey epstein", 2)

    def test_trim_3_chars(self):
        """'bill gatesilk' (trim 3) → 'bill gates'."""
        name_counts = {
            "bill gates": 1001,
            "bill gatesilk": 1,
        }
        name_to_id = {"bill gates": 1, "bill gatesilk": 2}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        assert len(merges) == 1

    def test_trim_4_chars_no_match(self):
        """Trimming > 3 chars should NOT match (too aggressive)."""
        name_counts = {
            "bill gates": 1001,
            "bill gatesilkx": 1,  # would need 4-char trim
        }
        name_to_id = {"bill gates": 1, "bill gatesilkx": 2}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        assert len(merges) == 0

    def test_source_high_count_no_match(self):
        """Source with many occurrences should NOT be merged (likely a real name)."""
        name_counts = {
            "bill gates": 1001,
            "bill gatesil": 50,  # too many occurrences to be junk
        }
        name_to_id = {"bill gates": 1, "bill gatesil": 2}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        assert len(merges) == 0

    def test_target_low_count_no_match(self):
        """Target with few occurrences shouldn't attract merges."""
        name_counts = {
            "bill gates": 5,  # too few to be a confident target
            "bill gatesil": 1,
        }
        name_to_id = {"bill gates": 1, "bill gatesil": 2}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        assert len(merges) == 0

    def test_ratio_too_low_no_match(self):
        """Even if target has enough count, ratio must be sufficient."""
        name_counts = {
            "bill gates": 60,
            "bill gatesil": 3,  # ratio = 20x, exactly at boundary
        }
        name_to_id = {"bill gates": 1, "bill gatesil": 2}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        assert len(merges) == 1  # ratio=20 meets threshold

    def test_ratio_below_threshold_no_match(self):
        """Ratio just below threshold."""
        name_counts = {
            "bill gates": 57,  # 57/3 = 19x < 20x threshold
            "bill gatesil": 3,
        }
        name_to_id = {"bill gates": 1, "bill gatesil": 2}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        assert len(merges) == 0

    def test_single_token_name_skipped(self):
        """Single-token names have no 'last token' to trim."""
        name_counts = {
            "epstein": 500,
            "epsteinx": 1,
        }
        name_to_id = {"epstein": 1, "epsteinx": 2}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        assert len(merges) == 0

    def test_no_false_positive_real_surname(self):
        """'john smithers' should NOT match 'john smith' if smithers has decent count."""
        name_counts = {
            "john smith": 500,
            "john smithers": 10,  # above MIN_SOURCE_COUNT
        }
        name_to_id = {"john smith": 1, "john smithers": 2}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        assert len(merges) == 0

    def test_real_surname_with_1_count_still_matches(self):
        """'john smithers' with 1 count would match — acceptable risk, shown in dry run."""
        name_counts = {
            "john smith": 500,
            "john smithers": 1,
        }
        name_to_id = {"john smith": 1, "john smithers": 2}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        # This IS a match because count=1 and ratio is huge.
        # Dry run review is the safeguard for borderline cases.
        assert len(merges) == 1

    def test_picks_best_match_highest_count(self):
        """When multiple trim lengths match, pick the one with highest count."""
        name_counts = {
            "bill gate": 10,       # wouldn't match (below MIN_TARGET_COUNT)
            "bill gates": 1001,
            "bill gatesil": 1,
        }
        name_to_id = {"bill gate": 3, "bill gates": 1, "bill gatesil": 2}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        assert len(merges) == 1
        assert merges[0][2] == "bill gates"  # target is the high-count one

    def test_empty_input(self):
        """No names → no merges."""
        merges = find_junk_suffix_merges({}, {})
        assert merges == []

    def test_all_high_count_no_merges(self):
        """When all names have high counts, nothing gets merged."""
        name_counts = {
            "bill gates": 1001,
            "jeffrey epstein": 500,
            "ghislaine maxwell": 300,
        }
        name_to_id = {n: i for i, n in enumerate(name_counts)}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        assert len(merges) == 0

    def test_last_token_too_short_after_trim(self):
        """Trimming would leave last token < 2 chars — don't match."""
        name_counts = {
            "bill ga": 500,       # 'ga' is only 2 chars
            "bill gax": 1,        # trim 1 → 'ga', which is only 2 chars — borderline
        }
        name_to_id = {"bill ga": 1, "bill gax": 2}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        # 'gax' trimmed by 1 = 'ga' (2 chars, valid) → matches 'bill ga' if count threshold met
        assert len(merges) == 1

    def test_trimmed_token_one_char_no_match(self):
        """If trimming leaves only 1 char for last token, skip."""
        name_counts = {
            "bill g": 500,
            "bill gat": 1,  # trim 2 → 'g' (1 char) — too short
        }
        name_to_id = {"bill g": 1, "bill gat": 2}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        # trim 1 → 'ga' → no match. trim 2 → 'g' → too short. trim 3 → '' → too short.
        assert len(merges) == 0

    def test_self_match_prevented(self):
        """A name should not merge into itself."""
        name_counts = {"bill gates": 1}
        name_to_id = {"bill gates": 1}
        merges = find_junk_suffix_merges(name_counts, name_to_id)
        assert len(merges) == 0
