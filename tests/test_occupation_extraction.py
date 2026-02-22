"""Tests for occupation extraction in generate_summaries.py."""

import json
from pathlib import Path

import pytest

from scripts.extraction.generate_summaries import (
    parse_structured_response,
    normalize_occupation,
    aggregate_occupation_summary,
    summarize_pages_batched,
    direct_summaries,
    process_document,
    process_documents_batched,
    is_cache_valid,
    PROMPT_VERSION,
    MODEL_VERSION,
)
from scripts.shared.constants import OCCUPATION_SYNONYMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tagged_batch_llm(prompts: list[str]) -> list[str]:
    """Return tagged responses."""
    return [f"[response-{i}]" for i in range(len(prompts))]


def tracking_batch_llm_factory():
    calls = []

    def batch_llm(prompts: list[str]) -> list[str]:
        calls.append(prompts)
        return [f"[resp-{len(calls)-1}-{i}]" for i in range(len(prompts))]

    return batch_llm, calls


def json_batch_llm(prompts: list[str]) -> list[str]:
    """Return valid JSON responses with occupation_mentions."""
    results = []
    for p in prompts:
        results.append(json.dumps({
            "summary": "Mock summary for testing.",
            "occupation_mentions": [],
        }))
    return results


def json_with_mentions_llm(prompts: list[str]) -> list[str]:
    """Return JSON responses that include occupation mentions."""
    results = []
    for p in prompts:
        if "Detective Recarey" in p:
            results.append(json.dumps({
                "summary": "Detective Recarey interviewed the victim.",
                "occupation_mentions": [{
                    "occupation": "detective",
                    "surface_form": "Detective",
                    "person_name": "Recarey",
                    "confidence": 0.92,
                    "evidence_span": "Detective Recarey",
                }],
            }))
        elif "Attorney" in p:
            results.append(json.dumps({
                "summary": "Attorney Dershowitz filed a motion.",
                "occupation_mentions": [{
                    "occupation": "lawyer",
                    "surface_form": "Attorney",
                    "person_name": "Alan Dershowitz",
                    "confidence": 0.95,
                    "evidence_span": "Attorney Dershowitz",
                }],
            }))
        else:
            results.append(json.dumps({
                "summary": "Mock summary.",
                "occupation_mentions": [],
            }))
    return results


def _make_doc(tmp_path, file_id, pages=None, text=None, names=None):
    """Helper to create test document files."""
    if pages:
        data = {"pages": pages, "file_type": "pdf", "total_pages": len(pages)}
    else:
        data = {"full_text": text or "Default text.", "file_type": "pdf", "total_pages": 1}
    (tmp_path / f"{file_id}.json").write_text(json.dumps(data))
    if names:
        (tmp_path / f"{file_id}_names.json").write_text(json.dumps({"names": names}))


# ===========================================================================
# Unit tests: normalize_occupation
# ===========================================================================

class TestNormalizeOccupation:
    def test_synonym_mapping(self):
        assert normalize_occupation("attorney") == "lawyer"
        assert normalize_occupation("counsel") == "lawyer"
        assert normalize_occupation("detective") == "detective"

    def test_unknown_passthrough(self):
        assert normalize_occupation("firefighter") == "firefighter"

    def test_case_insensitive(self):
        assert normalize_occupation("Attorney") == "lawyer"
        assert normalize_occupation("DETECTIVE") == "detective"

    def test_whitespace_stripped(self):
        assert normalize_occupation("  attorney  ") == "lawyer"


# ===========================================================================
# Unit tests: parse_structured_response
# ===========================================================================

class TestParseStructuredResponse:
    def test_valid_json_with_mentions(self):
        response = json.dumps({
            "summary": "Test summary.",
            "occupation_mentions": [{
                "occupation": "detective",
                "surface_form": "Detective",
                "person_name": "Recarey",
                "confidence": 0.92,
                "evidence_span": "Detective Recarey",
            }],
        })
        result = parse_structured_response(
            response, "Detective Recarey interviewed the victim."
        )
        assert result["summary"] == "Test summary."
        assert len(result["occupation_mentions"]) == 1
        assert result["occupation_mentions"][0]["occupation"] == "detective"

    def test_title_prefix_extraction(self):
        response = json.dumps({
            "summary": "Attorney Dershowitz filed a motion.",
            "occupation_mentions": [{
                "occupation": "attorney",
                "surface_form": "Attorney",
                "person_name": "Alan Dershowitz",
                "confidence": 0.95,
                "evidence_span": "Attorney Dershowitz",
            }],
        })
        result = parse_structured_response(
            response, "Attorney Dershowitz filed a motion."
        )
        # attorney -> lawyer via normalization
        assert result["occupation_mentions"][0]["occupation"] == "lawyer"

    def test_unnamed_occupation(self):
        response = json.dumps({
            "summary": "The pilot flew the plane.",
            "occupation_mentions": [{
                "occupation": "pilot",
                "surface_form": "pilot",
                "person_name": None,
                "confidence": 0.8,
                "evidence_span": "The pilot flew",
            }],
        })
        result = parse_structured_response(
            response, "The pilot flew the plane."
        )
        assert result["occupation_mentions"][0]["person_name"] is None

    def test_no_occupations(self):
        response = json.dumps({
            "summary": "A brief summary.",
            "occupation_mentions": [],
        })
        result = parse_structured_response(response)
        assert result["occupation_mentions"] == []

    def test_malformed_json_fallback(self):
        result = parse_structured_response("This is not JSON at all.")
        assert result["summary"] == "This is not JSON at all."
        assert result["occupation_mentions"] == []

    def test_json_embedded_in_text(self):
        response = 'Here is the result: {"summary": "Test.", "occupation_mentions": []}'
        result = parse_structured_response(response)
        assert result["summary"] == "Test."

    def test_multiple_mentions(self):
        response = json.dumps({
            "summary": "Summary.",
            "occupation_mentions": [
                {"occupation": "detective", "surface_form": "Det.", "person_name": "Smith",
                 "confidence": 0.9, "evidence_span": "Det. Smith"},
                {"occupation": "nurse", "surface_form": "nurse", "person_name": None,
                 "confidence": 0.7, "evidence_span": "a nurse"},
            ],
        })
        result = parse_structured_response(
            response, "Det. Smith spoke with a nurse at the hospital."
        )
        assert len(result["occupation_mentions"]) == 2

    def test_normalization_applied(self):
        response = json.dumps({
            "summary": "Summary.",
            "occupation_mentions": [{
                "occupation": "solicitor",
                "surface_form": "solicitor",
                "person_name": None,
                "confidence": 0.8,
                "evidence_span": "solicitor",
            }],
        })
        result = parse_structured_response(response, "A solicitor was present.")
        assert result["occupation_mentions"][0]["occupation"] == "lawyer"


# ===========================================================================
# Unit tests: evidence validation
# ===========================================================================

class TestEvidenceValidation:
    def test_exact_substring_match(self):
        response = json.dumps({
            "summary": "S.",
            "occupation_mentions": [{
                "occupation": "pilot",
                "surface_form": "pilot",
                "person_name": None,
                "confidence": 0.9,
                "evidence_span": "the pilot",
            }],
        })
        result = parse_structured_response(response, "He was the pilot of the aircraft.")
        assert len(result["occupation_mentions"]) == 1

    def test_hallucinated_span_dropped(self):
        response = json.dumps({
            "summary": "S.",
            "occupation_mentions": [{
                "occupation": "pilot",
                "surface_form": "pilot",
                "person_name": None,
                "confidence": 0.9,
                "evidence_span": "Captain Rogers flew the plane",
            }],
        })
        result = parse_structured_response(
            response, "He was the pilot of the aircraft."
        )
        assert len(result["occupation_mentions"]) == 0

    def test_no_page_text_skips_validation(self):
        response = json.dumps({
            "summary": "S.",
            "occupation_mentions": [{
                "occupation": "pilot",
                "surface_form": "pilot",
                "person_name": None,
                "confidence": 0.9,
                "evidence_span": "anything",
            }],
        })
        result = parse_structured_response(response, page_text=None)
        assert len(result["occupation_mentions"]) == 1


# ===========================================================================
# Unit tests: name linking
# ===========================================================================

class TestNameLinking:
    def test_person_not_in_allowed_list_set_to_null(self):
        response = json.dumps({
            "summary": "S.",
            "occupation_mentions": [{
                "occupation": "lawyer",
                "surface_form": "Attorney",
                "person_name": "Unknown Person",
                "confidence": 0.9,
                "evidence_span": "Attorney Unknown",
            }],
        })
        result = parse_structured_response(
            response, "Attorney Unknown Person filed the motion.",
            allowed_names={"Alan Dershowitz", "Jeffrey Epstein"},
        )
        assert result["occupation_mentions"][0]["person_name"] is None

    def test_case_insensitive_match(self):
        response = json.dumps({
            "summary": "S.",
            "occupation_mentions": [{
                "occupation": "lawyer",
                "surface_form": "Attorney",
                "person_name": "alan dershowitz",
                "confidence": 0.9,
                "evidence_span": "Attorney alan dershowitz",
            }],
        })
        result = parse_structured_response(
            response, "Attorney alan dershowitz was present.",
            allowed_names={"Alan Dershowitz"},
        )
        assert result["occupation_mentions"][0]["person_name"] == "Alan Dershowitz"

    def test_null_allowed_names_keeps_person(self):
        response = json.dumps({
            "summary": "S.",
            "occupation_mentions": [{
                "occupation": "pilot",
                "surface_form": "pilot",
                "person_name": "John Smith",
                "confidence": 0.9,
                "evidence_span": "pilot John Smith",
            }],
        })
        result = parse_structured_response(
            response, "pilot John Smith flew the aircraft.",
            allowed_names=None,
        )
        assert result["occupation_mentions"][0]["person_name"] == "John Smith"


# ===========================================================================
# Unit tests: edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_response(self):
        result = parse_structured_response("")
        assert result["summary"] == ""
        assert result["occupation_mentions"] == []

    def test_none_response(self):
        result = parse_structured_response(None)
        assert result["summary"] == ""

    def test_occupation_mentions_not_list(self):
        response = json.dumps({
            "summary": "S.",
            "occupation_mentions": "not a list",
        })
        result = parse_structured_response(response)
        assert result["occupation_mentions"] == []

    def test_mention_missing_required_fields(self):
        response = json.dumps({
            "summary": "S.",
            "occupation_mentions": [
                {"occupation": "lawyer"},  # missing surface_form
                {"surface_form": "Dr."},  # missing occupation
                {"occupation": "pilot", "surface_form": "pilot", "confidence": 0.9,
                 "evidence_span": "pilot"},  # valid, no person_name
            ],
        })
        result = parse_structured_response(response, "The pilot landed safely.")
        assert len(result["occupation_mentions"]) == 1
        assert result["occupation_mentions"][0]["occupation"] == "pilot"


# ===========================================================================
# Unit tests: aggregate_occupation_summary
# ===========================================================================

class TestAggregateOccupationSummary:
    def test_aggregation_by_person(self):
        pages = [
            {"page": 1, "occupation_mentions": [
                {"occupation": "lawyer", "person_name": "Dershowitz"},
                {"occupation": "professor", "person_name": "Dershowitz"},
            ]},
            {"page": 2, "occupation_mentions": [
                {"occupation": "detective", "person_name": "Recarey"},
            ]},
        ]
        result = aggregate_occupation_summary(pages)
        assert result["by_person"]["Dershowitz"] == ["lawyer", "professor"]
        assert result["by_person"]["Recarey"] == ["detective"]

    def test_aggregation_by_occupation(self):
        pages = [
            {"page": 1, "occupation_mentions": [
                {"occupation": "lawyer", "person_name": "A"},
                {"occupation": "lawyer", "person_name": "B"},
            ]},
        ]
        result = aggregate_occupation_summary(pages)
        assert result["by_occupation"]["lawyer"] == 2

    def test_unlinked_occupations(self):
        pages = [
            {"page": 1, "occupation_mentions": [
                {"occupation": "nurse", "person_name": None},
            ]},
        ]
        result = aggregate_occupation_summary(pages)
        assert "nurse" in result["unlinked"]

    def test_empty_pages(self):
        result = aggregate_occupation_summary([])
        assert result["by_person"] == {}
        assert result["by_occupation"] == {}
        assert result["unlinked"] == []

    def test_no_occupation_mentions_key(self):
        pages = [{"page": 1, "summary": "Test."}]
        result = aggregate_occupation_summary(pages)
        assert result["by_person"] == {}


# ===========================================================================
# Integration tests
# ===========================================================================

class TestProcessDocumentOccupation:
    def test_single_page_with_occupations(self, tmp_path):
        """Single-page doc using direct path includes occupation_summary."""
        text_file = tmp_path / "DOC001.json"
        text_file.write_text(json.dumps({
            "full_text": "Attorney Dershowitz filed a motion in court.",
            "file_type": "pdf",
            "total_pages": 1,
        }))
        (tmp_path / "DOC001_names_clean.json").write_text(json.dumps({
            "names": [{"normalized_name": "Alan Dershowitz", "role": "mentioned"}]
        }))

        result = process_document(text_file, tmp_path, json_with_mentions_llm)
        assert result is not None
        assert "occupation_summary" in result
        assert isinstance(result["occupation_summary"], dict)

    def test_multi_page_with_occupations(self, tmp_path):
        """Multi-page doc using MAP path includes occupation_summary."""
        text_file = tmp_path / "DOC002.json"
        text_file.write_text(json.dumps({
            "pages": [
                {"page_number": 1, "text": "Detective Recarey interviewed the victim at the station."},
                {"page_number": 2, "text": "Attorney Dershowitz filed a motion in court today."},
            ],
            "file_type": "pdf",
            "total_pages": 2,
        }))
        (tmp_path / "DOC002_names_clean.json").write_text(json.dumps({
            "names": [
                {"normalized_name": "Recarey", "role": "mentioned"},
                {"normalized_name": "Alan Dershowitz", "role": "mentioned"},
            ]
        }))

        result = process_document(text_file, tmp_path, json_with_mentions_llm)
        assert result is not None
        assert "occupation_summary" in result
        # Page summaries should have occupation_mentions
        for ps in result["page_summaries"]:
            assert "occupation_mentions" in ps

    def test_output_has_all_required_fields(self, tmp_path):
        """Output contains occupation_summary field."""
        text_file = tmp_path / "FIELDS.json"
        text_file.write_text(json.dumps({
            "full_text": "Document text for testing output field completeness.",
            "file_type": "pdf",
            "total_pages": 1,
        }))

        result = process_document(text_file, tmp_path, json_batch_llm)
        for field in ["file_id", "model_version", "prompt_version", "text_hash",
                       "summary_short", "summary_long", "page_summaries",
                       "allowed_names", "occupation_summary"]:
            assert field in result, f"Missing field: {field}"


class TestProcessDocumentsBatchedOccupation:
    def test_cross_doc_batched_includes_occupations(self, tmp_path):
        """process_documents_batched includes occupation_summary in output."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _make_doc(tmp_path, "DOCA", text="Attorney Dershowitz filed a motion in the court.")
        _make_doc(tmp_path, "DOCB", pages=[
            {"page_number": 1, "text": "Detective Recarey interviewed the victim at the station."},
            {"page_number": 2, "text": "The nurse administered treatment to the patient carefully."},
        ])

        files = [tmp_path / f for f in ["DOCA.json", "DOCB.json"]]
        processed, _ = process_documents_batched(
            files, tmp_path, json_batch_llm, output_dir
        )
        assert processed == 2

        for doc_id in ["DOCA", "DOCB"]:
            out_file = output_dir / f"{doc_id}_summary.json"
            assert out_file.exists()
            result = json.loads(out_file.read_text())
            assert "occupation_summary" in result

    def test_backward_compat_no_mentions(self, tmp_path):
        """Documents with no occupation mentions still work."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _make_doc(tmp_path, "PLAIN", text="A plain document without any occupations mentioned.")

        files = [tmp_path / "PLAIN.json"]
        processed, _ = process_documents_batched(
            files, tmp_path, json_batch_llm, output_dir
        )
        assert processed == 1

        result = json.loads((output_dir / "PLAIN_summary.json").read_text())
        assert result["occupation_summary"]["by_person"] == {}
        assert result["occupation_summary"]["by_occupation"] == {}
        assert result["occupation_summary"]["unlinked"] == []


# ===========================================================================
# Regression tests
# ===========================================================================

class TestRegressionOccupation:
    def test_prompt_version_is_v6(self):
        """PROMPT_VERSION should be v6 (v5→v6: possessive disambiguation)."""
        assert PROMPT_VERSION == "v6"

    def test_cache_invalidation_on_v3(self, tmp_path):
        """Old v2 summaries are invalidated by v3 prompt version."""
        summary = {
            "text_hash": "abc123",
            "model_version": MODEL_VERSION,
            "prompt_version": "v2",
        }
        path = tmp_path / "DOC_summary.json"
        path.write_text(json.dumps(summary))
        assert is_cache_valid(path, "abc123") is False

    def test_v6_cache_valid(self, tmp_path):
        """v6 summaries with matching hash are valid."""
        summary = {
            "text_hash": "abc123",
            "model_version": MODEL_VERSION,
            "prompt_version": "v6",
        }
        path = tmp_path / "DOC_summary.json"
        path.write_text(json.dumps(summary))
        assert is_cache_valid(path, "abc123") is True

    def test_direct_summaries_returns_dict(self):
        """direct_summaries returns dict with short, long, occupation_mentions."""
        result = direct_summaries(
            "Some document text here.", [], json_batch_llm
        )
        assert isinstance(result, dict)
        assert isinstance(result["short"], str)
        assert isinstance(result["long"], str)
        assert isinstance(result["occupation_mentions"], list)

    def test_summarize_pages_batched_has_occupation_mentions(self):
        """Page summaries include occupation_mentions key."""
        pages = [
            {"page_number": 1, "text": "Page one has enough text for a summary."},
        ]
        results = summarize_pages_batched(pages, 1, json_batch_llm)
        assert "occupation_mentions" in results[0]

    def test_occupation_synonyms_table_exists(self):
        """OCCUPATION_SYNONYMS is imported and has expected entries."""
        assert "attorney" in OCCUPATION_SYNONYMS
        assert OCCUPATION_SYNONYMS["attorney"] == "lawyer"
        assert "detective" in OCCUPATION_SYNONYMS
        assert OCCUPATION_SYNONYMS["detective"] == "detective"
        assert "masseuse" in OCCUPATION_SYNONYMS

    def test_legal_status_excluded(self):
        """Legal statuses should not appear in OCCUPATION_SYNONYMS."""
        for excluded in ["defendant", "plaintiff", "witness", "victim"]:
            assert excluded not in OCCUPATION_SYNONYMS

    def test_relational_terms_excluded(self):
        """Relational terms should not appear in OCCUPATION_SYNONYMS."""
        for excluded in ["associate", "girlfriend", "friend", "co-conspirator"]:
            assert excluded not in OCCUPATION_SYNONYMS
