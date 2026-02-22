"""Tests for scripts/extraction/validate_extractions.py"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from scripts.extraction.validate_extractions import (
    ValidationIssue,
    ValidationReport,
    find_source_text,
    fix_file,
    load_json,
    validate_directory,
    validate_file,
    validate_name,
)


class TestValidateName:
    """Tests for validate_name()."""

    def test_valid_names_pass(self):
        """Correct offsets produce no issue."""
        source = "Hello Jeffrey Epstein how are you"
        name = {"original_text": "Jeffrey Epstein", "start": 6, "end": 21}
        assert validate_name(name, source) is None

    def test_mismatch_detected(self):
        """Wrong offset text is caught as mismatch."""
        source = "Hello Jeffrey Epstein how are you"
        name = {"original_text": "Ghislaine Maxwell", "start": 6, "end": 21}
        issue = validate_name(name, source)
        assert issue is not None
        assert issue.issue_type == "mismatch"

    def test_out_of_bounds_detected(self):
        """Offset beyond text length is caught."""
        source = "Short text"
        name = {"original_text": "Name", "start": 100, "end": 104}
        issue = validate_name(name, source)
        assert issue is not None
        assert issue.issue_type == "out_of_bounds"

    def test_negative_start_detected(self):
        """Negative start offset is caught."""
        source = "Some text here"
        name = {"original_text": "Some", "start": -1, "end": 4}
        issue = validate_name(name, source)
        assert issue is not None
        assert issue.issue_type == "out_of_bounds"

    def test_start_equals_end_detected(self):
        """start == end (zero-length span) is caught."""
        source = "Some text here"
        name = {"original_text": "X", "start": 5, "end": 5}
        issue = validate_name(name, source)
        assert issue is not None
        assert issue.issue_type == "out_of_bounds"

    def test_missing_offsets_detected(self):
        """Missing start/end fields are caught."""
        source = "Some text"
        name = {"original_text": "Some"}
        issue = validate_name(name, source)
        assert issue is not None
        assert issue.issue_type == "missing_offsets"

    def test_whitespace_normalization(self):
        """Trailing/extra whitespace still matches via normalization."""
        source = "Hello Jeffrey  Epstein here"
        name = {"original_text": "Jeffrey Epstein", "start": 6, "end": 22}
        # source[6:22] = "Jeffrey  Epstein" (double space)
        # normalized: "Jeffrey Epstein" == "Jeffrey Epstein"
        issue = validate_name(name, source)
        assert issue is None

    def test_boundary_start_zero(self):
        """Name at the very start of text (offset 0)."""
        source = "Jeffrey Epstein is mentioned"
        name = {"original_text": "Jeffrey Epstein", "start": 0, "end": 15}
        assert validate_name(name, source) is None

    def test_boundary_end_at_length(self):
        """Name at the very end of text."""
        source = "Mentioned by Jeffrey Epstein"
        name = {"original_text": "Jeffrey Epstein", "start": 13, "end": 28}
        assert len(source) == 28
        assert validate_name(name, source) is None


class TestValidateFile:
    """Tests for validate_file()."""

    def test_valid_file(self, tmp_path):
        """File with correct offsets passes validation."""
        source_text = "Hello Jeffrey Epstein how are you"
        names_data = {
            "source_file": "TEST001.pdf",
            "names": [
                {"original_text": "Jeffrey Epstein", "start": 6, "end": 21}
            ]
        }
        names_file = tmp_path / "TEST001_names.json"
        source_file = tmp_path / "TEST001_ocr.json"
        with open(names_file, "w") as f:
            json.dump(names_data, f)
        with open(source_file, "w") as f:
            json.dump({"full_text": source_text}, f)

        is_valid, issues = validate_file(names_file)
        assert is_valid is True
        assert len(issues) == 0

    def test_empty_names_list(self, tmp_path):
        """File with no names passes gracefully."""
        names_data = {"source_file": "TEST002.pdf", "names": []}
        names_file = tmp_path / "TEST002_names.json"
        source_file = tmp_path / "TEST002_ocr.json"
        with open(names_file, "w") as f:
            json.dump(names_data, f)
        with open(source_file, "w") as f:
            json.dump({"full_text": "Some text"}, f)

        is_valid, issues = validate_file(names_file)
        assert is_valid is True
        assert len(issues) == 0

    def test_missing_source_text(self, tmp_path):
        """Missing source text assumes valid (can't check)."""
        names_data = {
            "source_file": "NOTEXT.pdf",
            "names": [
                {"original_text": "Some Name", "start": 0, "end": 9}
            ]
        }
        names_file = tmp_path / "NOTEXT_names.json"
        with open(names_file, "w") as f:
            json.dump(names_data, f)

        is_valid, issues = validate_file(names_file)
        assert is_valid is True
        assert len(issues) == 0

    def test_invalid_json_file(self, tmp_path):
        """Malformed JSON is caught as load_error."""
        names_file = tmp_path / "BAD_names.json"
        names_file.write_text("not valid json {{{")

        is_valid, issues = validate_file(names_file)
        assert is_valid is False
        assert len(issues) == 1
        assert issues[0].issue_type == "load_error"

    def test_mismatch_in_file(self, tmp_path):
        """File with offset mismatch is caught."""
        source_text = "Hello World"
        names_data = {
            "source_file": "TEST003.pdf",
            "names": [
                {"original_text": "Wrong Text", "start": 0, "end": 5}
            ]
        }
        names_file = tmp_path / "TEST003_names.json"
        source_file = tmp_path / "TEST003_ocr.json"
        with open(names_file, "w") as f:
            json.dump(names_data, f)
        with open(source_file, "w") as f:
            json.dump({"full_text": source_text}, f)

        is_valid, issues = validate_file(names_file)
        assert is_valid is False
        assert len(issues) == 1
        assert issues[0].issue_type == "mismatch"


class TestValidateDirectory:
    """Tests for validate_directory()."""

    def test_empty_directory(self, tmp_path):
        """Empty directory returns zero-count report."""
        report = validate_directory(tmp_path)
        assert report.total_files == 0
        assert report.valid_files == 0
        assert report.invalid_files == 0

    def test_report_counts_correct(self, tmp_path):
        """Report accurately counts valid/invalid files and names."""
        source_text = "Hello Jeffrey Epstein and Ghislaine Maxwell"

        # Valid file
        valid_data = {
            "source_file": "VALID.pdf",
            "names": [
                {"original_text": "Jeffrey Epstein", "start": 6, "end": 21}
            ]
        }
        valid_file = tmp_path / "VALID_names.json"
        valid_source = tmp_path / "VALID_ocr.json"
        with open(valid_file, "w") as f:
            json.dump(valid_data, f)
        with open(valid_source, "w") as f:
            json.dump({"full_text": source_text}, f)

        # Invalid file
        invalid_data = {
            "source_file": "INVALID.pdf",
            "names": [
                {"original_text": "Wrong Name", "start": 6, "end": 21},
                {"original_text": "Ghislaine Maxwell", "start": 26, "end": 43}
            ]
        }
        invalid_file = tmp_path / "INVALID_names.json"
        invalid_source = tmp_path / "INVALID_ocr.json"
        with open(invalid_file, "w") as f:
            json.dump(invalid_data, f)
        with open(invalid_source, "w") as f:
            json.dump({"full_text": source_text}, f)

        report = validate_directory(tmp_path)
        assert report.total_files == 2
        assert report.valid_files == 1
        assert report.invalid_files == 1
        assert report.total_names == 3
        assert report.invalid_names == 1


class TestFixFile:
    """Tests for fix_file()."""

    def test_fix_removes_invalid_entries(self, tmp_path):
        """--fix removes entries with bad offsets."""
        names_data = {
            "source_file": "FIX.pdf",
            "names": [
                {"original_text": "Good Name", "start": 0, "end": 9},
                {"original_text": "Bad Name", "start": 999, "end": 1007},
            ],
            "total_names": 2
        }
        names_file = tmp_path / "FIX_names.json"
        with open(names_file, "w") as f:
            json.dump(names_data, f)

        issues = [
            ValidationIssue(
                file=str(names_file),
                name_index=1,
                name_text="Bad Name",
                start=999,
                end=1007,
                issue_type="out_of_bounds",
                details="offset out of bounds"
            )
        ]

        fix_file(names_file, names_data, issues)

        with open(names_file) as f:
            fixed = json.load(f)

        assert len(fixed["names"]) == 1
        assert fixed["names"][0]["original_text"] == "Good Name"
        assert fixed["total_names"] == 1


class TestFindSourceText:
    """Tests for find_source_text()."""

    def test_finds_ocr_json(self, tmp_path):
        """Finds source text from _ocr.json file."""
        source_file = tmp_path / "TEST_ocr.json"
        with open(source_file, "w") as f:
            json.dump({"full_text": "The source text"}, f)

        names_file = tmp_path / "TEST_names.json"
        result = find_source_text(names_file, {})
        assert result == "The source text"

    def test_finds_clean_json(self, tmp_path):
        """Finds source text from _clean.json file."""
        source_file = tmp_path / "TEST_clean.json"
        with open(source_file, "w") as f:
            json.dump({"clean_text": "Cleaned text"}, f)

        names_file = tmp_path / "TEST_names.json"
        result = find_source_text(names_file, {})
        assert result == "Cleaned text"

    def test_finds_pages_format(self, tmp_path):
        """Finds source text from pages-format JSON."""
        source_file = tmp_path / "TEST_ocr.json"
        with open(source_file, "w") as f:
            json.dump({
                "pages": [
                    {"page_number": 1, "text": "Page one text"},
                    {"page_number": 2, "text": "Page two text"}
                ]
            }, f)

        names_file = tmp_path / "TEST_names.json"
        result = find_source_text(names_file, {})
        assert "[Page 1]" in result
        assert "Page one text" in result
        assert "Page two text" in result

    def test_returns_none_when_no_source(self, tmp_path):
        """Returns None when no source file exists."""
        names_file = tmp_path / "MISSING_names.json"
        result = find_source_text(names_file, {})
        assert result is None


class TestLoadJson:
    """Tests for load_json()."""

    def test_loads_valid_json(self, tmp_path):
        """Valid JSON file loads correctly."""
        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}')
        result = load_json(f)
        assert result == {"key": "value"}

    def test_returns_none_for_invalid(self, tmp_path):
        """Invalid JSON returns None."""
        f = tmp_path / "bad.json"
        f.write_text("not json")
        result = load_json(f)
        assert result is None

    def test_returns_none_for_missing(self, tmp_path):
        """Missing file returns None."""
        result = load_json(tmp_path / "nonexistent.json")
        assert result is None
