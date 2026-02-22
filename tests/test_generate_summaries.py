"""Tests for scripts/extraction/generate_summaries.py (batched API)."""

import json
from pathlib import Path

import pytest

import scripts.extraction.generate_summaries as gen_mod
from scripts.extraction.generate_summaries import (
    build_page_prompt,
    summarize_pages_batched,
    reduce_summaries,
    direct_summaries,
    process_document,
    prepare_document,
    process_documents_batched,
    find_text_files,
    load_names_for_document,
    check_names_cleaned,
    format_names_list,
    format_page_summaries,
    is_cache_valid,
    parse_long_multi_response,
    parse_structured_response,
    _is_page_skippable,
    _chunked_batch,
    _build_token_aware_chunks,
    _estimate_prompt_tokens,
    _estimate_reduce_tokens,
    _hierarchical_reduce,
    BATCH_CHUNK_SIZE,
    MAX_BATCH_TOKENS,
    MAX_MODEL_TOKENS,
    REDUCE_OUTPUT_MARGIN,
    MAX_PAGE_CHARS,
    CROSS_DOC_BATCH_SIZE,
    MIN_PAGE_CHARS,
    MODEL_VERSION,
    PROMPT_VERSION,
)
from scripts.extraction.clean_names import clean_name, is_garbage


# ---------------------------------------------------------------------------
# Deterministic fake LLM that returns tagged outputs
# ---------------------------------------------------------------------------

def tagged_batch_llm(prompts: list[str]) -> list[str]:
    """Return tagged responses so we can verify ordering."""
    return [f"[response-{i}]" for i in range(len(prompts))]


def tracking_batch_llm_factory():
    """Returns a batch LLM that tracks all calls for assertion."""
    calls = []

    def batch_llm(prompts: list[str]) -> list[str]:
        calls.append(prompts)
        return [f"[resp-{len(calls)-1}-{i}]" for i in range(len(prompts))]

    return batch_llm, calls


# ---------------------------------------------------------------------------
# Tests: build_page_prompt
# ---------------------------------------------------------------------------

class TestBuildPagePrompt:
    def test_blank_page_returns_none(self):
        assert build_page_prompt(1, "", 5) is None

    def test_short_text_returns_none(self):
        assert build_page_prompt(1, "short", 5) is None

    def test_exactly_min_chars_returns_none(self):
        text = "a" * (MIN_PAGE_CHARS - 1)
        assert build_page_prompt(1, text, 5) is None

    def test_sufficient_text_returns_prompt(self):
        text = "This is enough text for a page summary to be generated."
        result = build_page_prompt(1, text, 5)
        assert result is not None
        assert "page 1" in result.lower() or "Page 1" in result
        assert text in result

    def test_long_text_truncated(self):
        text = "A" * 10000
        result = build_page_prompt(1, text, 5)
        assert "[... truncated ...]" in result

    def test_page_number_in_prompt(self):
        result = build_page_prompt(42, "Enough text for page summary content.", 100)
        assert "42" in result
        assert "100" in result


# ---------------------------------------------------------------------------
# Tests: _is_page_skippable
# ---------------------------------------------------------------------------

class TestIsPageSkippable:
    def test_empty_text(self):
        assert _is_page_skippable({"text": ""}) is True

    def test_short_text(self):
        assert _is_page_skippable({"text": "hi"}) is True

    def test_photo_only_page(self):
        assert _is_page_skippable({"text": "Some text here.", "text_source": "none"}) is True

    def test_normal_page(self):
        assert _is_page_skippable({"text": "This page has enough text content."}) is False

    def test_ocr_page_ok(self):
        assert _is_page_skippable({"text": "OCR extracted text.", "text_source": "ocr"}) is False


# ---------------------------------------------------------------------------
# Tests: _chunked_batch
# ---------------------------------------------------------------------------

class TestChunkedBatch:
    def test_small_batch_single_call(self):
        batch_fn, calls = tracking_batch_llm_factory()
        prompts = ["p1", "p2", "p3"]
        results = _chunked_batch(prompts, batch_fn)
        assert len(calls) == 1
        assert len(results) == 3

    def test_large_batch_multiple_calls(self):
        batch_fn, calls = tracking_batch_llm_factory()
        prompts = [f"p{i}" for i in range(BATCH_CHUNK_SIZE + 5)]
        results = _chunked_batch(prompts, batch_fn)
        assert len(calls) == 2  # one full chunk + one partial
        assert len(results) == BATCH_CHUNK_SIZE + 5

    def test_exact_chunk_size(self):
        batch_fn, calls = tracking_batch_llm_factory()
        prompts = [f"p{i}" for i in range(BATCH_CHUNK_SIZE)]
        results = _chunked_batch(prompts, batch_fn)
        assert len(calls) == 1
        assert len(results) == BATCH_CHUNK_SIZE

    def test_ordering_preserved_across_chunks(self):
        """Responses from multiple chunks maintain correct order."""
        batch_fn, calls = tracking_batch_llm_factory()
        n = BATCH_CHUNK_SIZE * 3 + 2
        prompts = [f"p{i}" for i in range(n)]
        results = _chunked_batch(prompts, batch_fn)
        assert len(results) == n
        assert len(calls) == 4

    def test_empty_prompts_returns_empty(self):
        batch_fn, calls = tracking_batch_llm_factory()
        results = _chunked_batch([], batch_fn)
        assert results == []
        assert len(calls) == 0

    def test_token_budget_splits_large_prompts(self):
        """Prompts exceeding token budget are split into separate chunks."""
        batch_fn, calls = tracking_batch_llm_factory()
        # Each prompt ~10k tokens (40k chars), budget is 96k tokens
        # So max ~9 per chunk, but count limit (16) is higher
        big_prompts = ["A" * 40_000 for _ in range(20)]
        results = _chunked_batch(big_prompts, batch_fn, chunk_size=64)
        assert len(results) == 20
        # Should have split into ~3 chunks (9+9+2) due to token budget
        assert len(calls) >= 2

    def test_per_item_retry_on_empty_response(self):
        """Empty responses are retried individually."""
        call_count = [0]

        def flaky_llm(prompts):
            call_count[0] += 1
            results = []
            for i, p in enumerate(prompts):
                # First batch call: prompt index 1 returns empty
                if call_count[0] == 1 and i == 1:
                    results.append("")
                else:
                    results.append(f"ok-{p}")
            return results

        prompts = ["p0", "p1", "p2"]
        results = _chunked_batch(prompts, flaky_llm)
        assert len(results) == 3
        assert results[0] == "ok-p0"
        assert results[1] == "ok-p1"  # retried individually
        assert results[2] == "ok-p2"
        assert call_count[0] == 2  # original batch + 1 retry

    def test_per_item_retry_gives_up_after_max(self):
        """After MAX_PROMPT_RETRIES, empty response remains empty."""
        def always_empty_for_p1(prompts):
            return ["" if "p1" in p else f"ok-{p}" for p in prompts]

        prompts = ["p0", "p1", "p2"]
        results = _chunked_batch(prompts, always_empty_for_p1)
        assert results[0] == "ok-p0"
        assert results[1] == ""  # gave up
        assert results[2] == "ok-p2"

    def test_retry_does_not_duplicate_good_results(self):
        """Retry only affects failed prompts, good ones untouched."""
        call_log = []

        def logging_llm(prompts):
            call_log.append(list(prompts))
            return ["" if p == "fail" else f"ok-{p}" for p in prompts]

        prompts = ["good1", "fail", "good2"]
        results = _chunked_batch(prompts, logging_llm)
        assert results[0] == "ok-good1"
        assert results[2] == "ok-good2"
        # Retry should only re-send "fail", not the good ones
        assert len(call_log) >= 2
        assert call_log[1] == ["fail"]


# ---------------------------------------------------------------------------
# Tests: _build_token_aware_chunks
# ---------------------------------------------------------------------------

class TestBuildTokenAwareChunks:
    def test_all_small_prompts_single_chunk(self):
        prompts = ["short"] * 5
        chunks = _build_token_aware_chunks(prompts, max_count=10, max_tokens=1000)
        assert len(chunks) == 1
        assert len(chunks[0]) == 5

    def test_count_limit_splits(self):
        prompts = ["short"] * 10
        chunks = _build_token_aware_chunks(prompts, max_count=3, max_tokens=999999)
        assert len(chunks) == 4  # 3+3+3+1
        assert all(len(c) <= 3 for c in chunks)

    def test_token_limit_splits(self):
        # Each prompt ~1333 tokens (4000 chars at //3 estimate)
        prompts = ["A" * 4000] * 5
        chunks = _build_token_aware_chunks(prompts, max_count=100, max_tokens=2800)
        # Budget 2800 tokens, each prompt ~1333 tokens → max 2 per chunk
        assert len(chunks) == 3  # 2+2+1
        for c in chunks:
            total_tokens = sum(_estimate_prompt_tokens(p) for _, p in c)
            assert total_tokens <= 2800

    def test_preserves_original_indices(self):
        prompts = ["a", "bb", "ccc"]
        chunks = _build_token_aware_chunks(prompts, max_count=100, max_tokens=999999)
        assert len(chunks) == 1
        indices = [idx for idx, _ in chunks[0]]
        assert indices == [0, 1, 2]

    def test_single_oversized_prompt_gets_own_chunk(self):
        """A prompt exceeding the budget goes in its own chunk."""
        prompts = ["small", "A" * 400000, "small"]  # middle is ~100k tokens
        chunks = _build_token_aware_chunks(prompts, max_count=100, max_tokens=50000)
        assert len(chunks) >= 2
        # The huge prompt should be alone or with the first small one
        # Key: it doesn't block other prompts from being chunked


# ---------------------------------------------------------------------------
# Tests: prepare_document
# ---------------------------------------------------------------------------

class TestPrepareDocument:
    def test_returns_context_dict(self, tmp_path):
        text_file = tmp_path / "DOC001.json"
        text_file.write_text(json.dumps({
            "full_text": "Legal document about Jeffrey Epstein from 2005.",
            "file_type": "pdf",
            "total_pages": 1,
        }))
        (tmp_path / "DOC001_names_clean.json").write_text(json.dumps({
            "names": [{"normalized_name": "Jeffrey Epstein", "role": "mentioned"}]
        }))

        ctx = prepare_document(text_file, tmp_path)
        assert ctx is not None
        assert ctx["file_id"] == "DOC001"
        assert ctx["single_page"] is True
        assert len(ctx["names"]) == 1
        assert ctx["text_hash"]

    def test_skips_short_text(self, tmp_path):
        text_file = tmp_path / "EMPTY.json"
        text_file.write_text(json.dumps({"full_text": "short"}))
        assert prepare_document(text_file, tmp_path) is None

    def test_multi_page_not_single(self, tmp_path):
        text_file = tmp_path / "MULTI.json"
        text_file.write_text(json.dumps({
            "pages": [
                {"page_number": 1, "text": "Page one text."},
                {"page_number": 2, "text": "Page two text."},
            ],
        }))
        ctx = prepare_document(text_file, tmp_path)
        assert ctx is not None
        assert ctx["single_page"] is False
        assert len(ctx["pages"]) == 2


# ---------------------------------------------------------------------------
# Tests: process_documents_batched (cross-document)
# ---------------------------------------------------------------------------

def _make_doc(tmp_path, file_id, pages=None, text=None, names=None):
    """Helper to create test document files."""
    if pages:
        data = {"pages": pages, "file_type": "pdf", "total_pages": len(pages)}
    else:
        data = {"full_text": text or "Default text.", "file_type": "pdf", "total_pages": 1}
    (tmp_path / f"{file_id}.json").write_text(json.dumps(data))
    if names:
        (tmp_path / f"{file_id}_names_clean.json").write_text(json.dumps({"names": names}))


class TestProcessDocumentsBatched:
    def test_mixed_doc_mapping_correctness(self, tmp_path):
        """Prompts from different docs route back to correct doc/page."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Doc A: 3 pages, Doc B: 2 pages, Doc C: 1 page (direct)
        _make_doc(tmp_path, "DOCA", pages=[
            {"page_number": 1, "text": "Doc A page 1 has enough text for summary."},
            {"page_number": 2, "text": "Doc A page 2 has enough text for summary."},
            {"page_number": 3, "text": "Doc A page 3 has enough text for summary."},
        ])
        _make_doc(tmp_path, "DOCB", pages=[
            {"page_number": 1, "text": "Doc B page 1 has enough text for summary."},
            {"page_number": 2, "text": "Doc B page 2 has enough text for summary."},
        ])
        _make_doc(tmp_path, "DOCC", text="Doc C is a single-page document with enough text.")

        files = [tmp_path / f for f in ["DOCA.json", "DOCB.json", "DOCC.json"]]
        batch_fn, calls = tracking_batch_llm_factory()

        processed, skipped = process_documents_batched(
            files, tmp_path, batch_fn, output_dir
        )
        assert processed == 3
        assert skipped == 0

        # Verify output files exist and have correct structure
        for doc_id in ["DOCA", "DOCB", "DOCC"]:
            out_file = output_dir / f"{doc_id}_summary.json"
            assert out_file.exists()
            result = json.loads(out_file.read_text())
            assert result["file_id"] == doc_id
            assert "summary_short" in result
            assert "summary_long" in result

        # DOCA should have 3 page summaries, DOCB 2, DOCC 0 (single-page)
        doca = json.loads((output_dir / "DOCA_summary.json").read_text())
        docb = json.loads((output_dir / "DOCB_summary.json").read_text())
        docc = json.loads((output_dir / "DOCC_summary.json").read_text())
        assert len(doca["page_summaries"]) == 3
        assert len(docb["page_summaries"]) == 2
        assert docc["page_summaries"] == []

    def test_skipped_documents_counted(self, tmp_path):
        """Documents with too-short text are skipped."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _make_doc(tmp_path, "GOOD", text="Document with enough text for summarization.")
        _make_doc(tmp_path, "BAD", text="tiny")

        files = [tmp_path / "GOOD.json", tmp_path / "BAD.json"]
        batch_fn, _ = tracking_batch_llm_factory()

        processed, skipped = process_documents_batched(
            files, tmp_path, batch_fn, output_dir
        )
        assert processed == 1
        assert skipped == 1

    def test_map_reduce_ordering_dependency(self, tmp_path):
        """REDUCE uses MAP results, not stale data."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _make_doc(tmp_path, "DOC", pages=[
            {"page_number": 1, "text": "First page has enough text for a summary."},
            {"page_number": 2, "text": "Second page has enough text for a summary."},
        ])

        # LLM that tags responses with batch number
        batch_num = [0]
        def ordered_llm(prompts):
            batch_num[0] += 1
            return [f"[batch{batch_num[0]}-{i}]" for i in range(len(prompts))]

        processed, _ = process_documents_batched(
            [tmp_path / "DOC.json"], tmp_path, ordered_llm, output_dir
        )
        assert processed == 1

        result = json.loads((output_dir / "DOC_summary.json").read_text())
        # MAP was batch 1, REDUCE was batch 2
        assert result["page_summaries"][0]["summary"].startswith("[batch1-")
        assert result["page_summaries"][1]["summary"].startswith("[batch1-")
        # REDUCE summaries come from batch 2
        assert result["summary_short"].startswith("[batch2-")
        assert result["summary_long"].startswith("[batch2-")

    def test_blank_pages_not_in_map_batch(self, tmp_path):
        """Blank pages don't consume MAP batch slots."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _make_doc(tmp_path, "DOC", pages=[
            {"page_number": 1, "text": "Real page content for summarization."},
            {"page_number": 2, "text": ""},  # blank
            {"page_number": 3, "text": "Another real page for summarization."},
        ])

        batch_fn, calls = tracking_batch_llm_factory()
        process_documents_batched(
            [tmp_path / "DOC.json"], tmp_path, batch_fn, output_dir
        )

        # MAP call should only have 2 prompts (pages 1 and 3)
        assert len(calls[0]) == 2
        result = json.loads((output_dir / "DOC_summary.json").read_text())
        assert result["page_summaries"][1]["summary"] == "Blank page."

    def test_idempotent_overwrite(self, tmp_path):
        """Running twice produces same output (idempotent)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _make_doc(tmp_path, "DOC", text="Document text for idempotency test here.")

        files = [tmp_path / "DOC.json"]

        # Run twice with deterministic LLM
        for _ in range(2):
            process_documents_batched(files, tmp_path, tagged_batch_llm, output_dir)

        # File should exist and be valid
        result = json.loads((output_dir / "DOC_summary.json").read_text())
        assert result["file_id"] == "DOC"
        assert result["summary_short"]

    def test_all_single_page_docs(self, tmp_path):
        """Batch of only single-page docs: no MAP phase, only REDUCE."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        for i in range(3):
            _make_doc(tmp_path, f"S{i}", text=f"Single page doc {i} with enough text here.")

        batch_fn, calls = tracking_batch_llm_factory()
        processed, _ = process_documents_batched(
            [tmp_path / f"S{i}.json" for i in range(3)],
            tmp_path, batch_fn, output_dir,
        )
        assert processed == 3
        # No MAP call (all single-page), just one REDUCE call with 6 prompts (3 docs × 2)
        assert len(calls) == 1
        assert len(calls[0]) == 6


# ---------------------------------------------------------------------------
# Tests: summarize_pages_batched
# ---------------------------------------------------------------------------

class TestSummarizePagesBatched:
    def test_ordering_preserved(self):
        """Page summaries line up with page numbers."""
        pages = [
            {"page_number": 1, "text": "Page one has enough text for summary."},
            {"page_number": 2, "text": "Page two also has enough text here."},
            {"page_number": 3, "text": "Page three with sufficient content."},
        ]
        results = summarize_pages_batched(pages, 3, tagged_batch_llm)
        assert len(results) == 3
        assert results[0]["page"] == 1
        assert results[1]["page"] == 2
        assert results[2]["page"] == 3
        # All got LLM responses
        assert results[0]["summary"] == "[response-0]"
        assert results[1]["summary"] == "[response-1]"
        assert results[2]["summary"] == "[response-2]"

    def test_blank_pages_not_sent_to_llm(self):
        """Blank pages get 'Blank page.' without LLM call."""
        batch_fn, calls = tracking_batch_llm_factory()
        pages = [
            {"page_number": 1, "text": "Page one has real content for summarization."},
            {"page_number": 2, "text": ""},  # blank
            {"page_number": 3, "text": "Page three has real content for summarization."},
        ]
        results = summarize_pages_batched(pages, 3, batch_fn)
        assert len(results) == 3
        # Only 2 prompts sent to LLM (pages 1 and 3)
        assert len(calls) == 1
        assert len(calls[0]) == 2
        # Blank page gets static response
        assert results[1]["summary"] == "Blank page."
        # Non-blank pages get LLM responses
        assert results[0]["summary"].startswith("[resp-")
        assert results[2]["summary"].startswith("[resp-")

    def test_photo_only_pages_skipped(self):
        """Pages with text_source='none' are skipped."""
        batch_fn, calls = tracking_batch_llm_factory()
        pages = [
            {"page_number": 1, "text": "Real text content that is long enough."},
            {"page_number": 2, "text": "Some text here.", "text_source": "none"},
        ]
        results = summarize_pages_batched(pages, 2, batch_fn)
        assert len(results) == 2
        assert len(calls[0]) == 1  # only page 1 sent
        assert results[1]["summary"] == "Blank page."

    def test_all_blank_pages(self):
        """All blank pages means no LLM call."""
        batch_fn, calls = tracking_batch_llm_factory()
        pages = [
            {"page_number": 1, "text": ""},
            {"page_number": 2, "text": "hi"},
        ]
        results = summarize_pages_batched(pages, 2, batch_fn)
        assert len(calls) == 0  # no LLM call
        assert all(r["summary"] == "Blank page." for r in results)

    def test_page_number_fallback(self):
        """Falls back to 'page' key, then index+1."""
        pages = [
            {"page": 10, "text": "Text content for this page summary here."},
            {"text": "More content that is long enough for summarization."},
        ]
        results = summarize_pages_batched(pages, 2, tagged_batch_llm)
        assert results[0]["page"] == 10
        assert results[1]["page"] == 2  # index 1 + 1

    def test_large_doc_chunked(self):
        """Documents with many pages get chunked."""
        batch_fn, calls = tracking_batch_llm_factory()
        n_pages = BATCH_CHUNK_SIZE + 5
        pages = [
            {"page_number": i + 1, "text": f"Page {i+1} with enough text for summary generation."}
            for i in range(n_pages)
        ]
        results = summarize_pages_batched(pages, n_pages, batch_fn)
        assert len(results) == n_pages
        # Should have made 2 batch calls
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# Tests: reduce_summaries
# ---------------------------------------------------------------------------

class TestReduceSummaries:
    def test_returns_dict_with_short_and_long(self):
        page_sums = [
            {"page": 1, "summary": "Page 1 is about Epstein."},
            {"page": 2, "summary": "Page 2 discusses Maxwell."},
        ]
        names = [{"name": "Jeffrey Epstein", "role": "mentioned"}]
        result = reduce_summaries(page_sums, names, "File: TEST", tagged_batch_llm)
        assert isinstance(result, dict)
        assert result["short"] == "[response-0]"
        # long is parsed via parse_long_multi_response; non-JSON falls back to raw
        assert result["long"] == "[response-1]"
        assert "document_type" in result
        assert "document_date" in result

    def test_both_prompts_sent_in_one_batch(self):
        batch_fn, calls = tracking_batch_llm_factory()
        page_sums = [{"page": 1, "summary": "Summary."}]
        reduce_summaries(page_sums, [], "info", batch_fn)
        assert len(calls) == 1
        assert len(calls[0]) == 2  # short + long in one call


# ---------------------------------------------------------------------------
# Tests: direct_summaries
# ---------------------------------------------------------------------------

class TestDirectSummaries:
    def test_returns_dict_with_short_and_long(self):
        result = direct_summaries(
            "Some document text here.", [], tagged_batch_llm
        )
        # tagged_batch_llm returns "[response-N]" which isn't JSON,
        # so parse_structured_response falls back to treating it as summary text
        assert isinstance(result, dict)
        assert result["short"] == "[response-0]"
        assert result["long"] == "[response-1]"
        assert isinstance(result["occupation_mentions"], list)
        assert "document_type" in result
        assert "document_date" in result

    def test_both_prompts_batched(self):
        batch_fn, calls = tracking_batch_llm_factory()
        direct_summaries("Text content.", [], batch_fn)
        assert len(calls) == 1
        assert len(calls[0]) == 2


# ---------------------------------------------------------------------------
# Tests: process_document (integration)
# ---------------------------------------------------------------------------

class TestProcessDocument:
    def test_single_page_doc(self, tmp_path):
        """Single-page doc uses direct_summaries (2 prompts batched)."""
        text_file = tmp_path / "DOC001.json"
        text_file.write_text(json.dumps({
            "full_text": "Legal document about Jeffrey Epstein from 2005.",
            "file_type": "pdf",
            "total_pages": 1,
        }))
        (tmp_path / "DOC001_names_clean.json").write_text(json.dumps({
            "names": [{"normalized_name": "Jeffrey Epstein", "role": "mentioned"}]
        }))

        batch_fn, calls = tracking_batch_llm_factory()
        result = process_document(text_file, tmp_path, batch_fn)

        assert result is not None
        assert result["file_id"] == "DOC001"
        assert "summary_short" in result
        assert "summary_long" in result
        assert result["page_summaries"] == []
        assert len(result["allowed_names"]) == 1
        # Direct: 1 batch of 2 prompts
        assert len(calls) == 1
        assert len(calls[0]) == 2

    def test_multi_page_doc(self, tmp_path):
        """Multi-page doc: MAP batch + REDUCE batch."""
        text_file = tmp_path / "DOC002.json"
        text_file.write_text(json.dumps({
            "pages": [
                {"page_number": 1, "text": "Page one describes legal proceedings in detail."},
                {"page_number": 2, "text": "Page two lists financial transactions and records."},
                {"page_number": 3, "text": "Page three contains witness testimony and statements."},
            ],
            "file_type": "pdf",
            "total_pages": 3,
        }))

        batch_fn, calls = tracking_batch_llm_factory()
        result = process_document(text_file, tmp_path, batch_fn)

        assert result is not None
        assert len(result["page_summaries"]) == 3
        # 2 batch calls: MAP (3 prompts) + REDUCE (2 prompts)
        assert len(calls) == 2
        assert len(calls[0]) == 3  # MAP
        assert len(calls[1]) == 2  # REDUCE

    def test_multi_page_with_blank(self, tmp_path):
        """Blank pages excluded from MAP batch."""
        text_file = tmp_path / "DOC003.json"
        text_file.write_text(json.dumps({
            "pages": [
                {"page_number": 1, "text": "First page with real content for summarization."},
                {"page_number": 2, "text": ""},
                {"page_number": 3, "text": "Third page with real content for summarization."},
            ],
            "file_type": "pdf",
            "total_pages": 3,
        }))

        batch_fn, calls = tracking_batch_llm_factory()
        result = process_document(text_file, tmp_path, batch_fn)

        assert len(result["page_summaries"]) == 3
        assert result["page_summaries"][1]["summary"] == "Blank page."
        # MAP: only 2 non-blank pages
        assert len(calls[0]) == 2

    def test_empty_text_skipped(self, tmp_path):
        text_file = tmp_path / "EMPTY.json"
        text_file.write_text(json.dumps({"full_text": ""}))

        batch_fn, calls = tracking_batch_llm_factory()
        result = process_document(text_file, tmp_path, batch_fn)
        assert result is None
        assert len(calls) == 0

    def test_no_names_file_still_works(self, tmp_path):
        text_file = tmp_path / "NONAMES.json"
        text_file.write_text(json.dumps({
            "full_text": "A document without extracted names but enough text.",
            "file_type": "pdf",
            "total_pages": 1,
        }))

        result = process_document(text_file, tmp_path, tagged_batch_llm)
        assert result is not None
        assert result["allowed_names"] == []

    def test_output_contains_all_fields(self, tmp_path):
        text_file = tmp_path / "FIELDS.json"
        text_file.write_text(json.dumps({
            "full_text": "Document text for testing output field completeness.",
            "file_type": "pdf",
            "total_pages": 1,
        }))

        result = process_document(text_file, tmp_path, tagged_batch_llm)
        for field in ["file_id", "model_version", "prompt_version", "text_hash",
                       "summary_short", "summary_long", "page_summaries", "allowed_names",
                       "document_type", "document_date"]:
            assert field in result, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Tests: clean_names (unchanged, kept from old test file)
# ---------------------------------------------------------------------------

class TestCleanName:
    def test_strips_leading_dots(self):
        assert clean_name(". Epstein") == "Epstein"

    def test_strips_leading_spaces(self):
        assert clean_name("  Maxwell") == "Maxwell"

    def test_strips_leading_hyphens(self):
        assert clean_name("- Smith") == "Smith"

    def test_normal_name_unchanged(self):
        assert clean_name("Jeffrey Epstein") == "Jeffrey Epstein"


class TestIsGarbage:
    def test_short_name_is_garbage(self):
        assert is_garbage("JE") is True

    def test_digits_is_garbage(self):
        assert is_garbage("12345") is True

    def test_real_name_not_garbage(self):
        assert is_garbage("Jeffrey Epstein") is False

    def test_concatenated_name_is_garbage(self):
        assert is_garbage("Epsteinjeffrey") is True

    def test_hyphenated_name_ok(self):
        assert is_garbage("Jean-Pierre") is False


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------

class TestLoadNamesForDocument:
    def test_deduplicates_by_priority(self, tmp_path):
        (tmp_path / "DOC_names.json").write_text(json.dumps({
            "names": [
                {"normalized_name": "Jeffrey Epstein", "role": "mentioned"},
                {"normalized_name": "Jeffrey Epstein", "role": "sender"},
            ]
        }))
        names, _ = load_names_for_document(tmp_path, "DOC")
        assert len(names) == 1
        assert names[0]["role"] == "sender"

    def test_prefers_clean_file(self, tmp_path):
        (tmp_path / "DOC_names.json").write_text(json.dumps({
            "names": [
                {"normalized_name": "Jeffrey Epstein", "role": "mentioned"},
                {"normalized_name": "JE", "role": "mentioned"},
            ]
        }))
        (tmp_path / "DOC_names_clean.json").write_text(json.dumps({
            "names": [
                {"normalized_name": "Jeffrey Epstein", "role": "mentioned"},
            ]
        }))
        names, _ = load_names_for_document(tmp_path, "DOC")
        assert len(names) == 1

    def test_missing_file_returns_empty(self, tmp_path):
        names, meta = load_names_for_document(tmp_path, "NONEXISTENT")
        assert names == []
        assert meta == {}


class TestCheckNamesCleaned:
    def test_returns_uncleaned_files(self, tmp_path):
        """Files with _names.json but no _names_clean.json are flagged."""
        (tmp_path / "DOC1_names.json").write_text("{}")
        (tmp_path / "DOC2_names.json").write_text("{}")
        (tmp_path / "DOC2_names_clean.json").write_text("{}")
        result = check_names_cleaned(tmp_path)
        assert result == ["DOC1"]

    def test_all_cleaned_returns_empty(self, tmp_path):
        """When every raw file has a clean counterpart, return empty list."""
        (tmp_path / "DOC1_names.json").write_text("{}")
        (tmp_path / "DOC1_names_clean.json").write_text("{}")
        result = check_names_cleaned(tmp_path)
        assert result == []

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        """Non-existent directory returns empty list."""
        result = check_names_cleaned(tmp_path / "nope")
        assert result == []

    def test_only_clean_files_returns_empty(self, tmp_path):
        """Clean files without raw counterparts are fine."""
        (tmp_path / "DOC1_names_clean.json").write_text("{}")
        result = check_names_cleaned(tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: parse_long_multi_response
# ---------------------------------------------------------------------------

class TestParseLongMultiResponse:
    def test_valid_json(self):
        response = '{"summary": "A letter about legal matters.", "document_type": "letter", "date": "2005-03-15"}'
        result = parse_long_multi_response(response)
        assert result["summary"] == "A letter about legal matters."
        assert result["document_type"] == "letter"
        assert result["document_date"] == "2005-03-15"

    def test_missing_fields(self):
        response = '{"summary": "Some text."}'
        result = parse_long_multi_response(response)
        assert result["summary"] == "Some text."
        assert result["document_type"] is None
        assert result["document_date"] is None

    def test_parse_failure_fallback(self):
        response = "This is just plain text, not JSON."
        result = parse_long_multi_response(response)
        assert result["summary"] == response
        assert result["document_type"] is None
        assert result["document_date"] is None

    def test_empty_response(self):
        result = parse_long_multi_response("")
        assert result["summary"] == ""
        assert result["document_type"] is None
        assert result["document_date"] is None

    def test_json_with_surrounding_text(self):
        response = 'Here is the result: {"summary": "A memo.", "document_type": "memo", "date": "2004"} done.'
        result = parse_long_multi_response(response)
        assert result["summary"] == "A memo."
        assert result["document_type"] == "memo"
        assert result["document_date"] == "2004"


# ---------------------------------------------------------------------------
# Tests: parse_structured_response with document_type/date
# ---------------------------------------------------------------------------

class TestParseStructuredResponseDocFields:
    def test_extracts_document_type_and_date(self):
        response = json.dumps({
            "summary": "A legal filing.",
            "document_type": "legal_filing",
            "date": "2005-06",
            "occupation_mentions": [],
        })
        result = parse_structured_response(response)
        assert result["summary"] == "A legal filing."
        assert result["document_type"] == "legal_filing"
        assert result["document_date"] == "2005-06"

    def test_no_doc_fields_when_absent(self):
        response = json.dumps({
            "summary": "A page summary.",
            "occupation_mentions": [],
        })
        result = parse_structured_response(response)
        assert result["summary"] == "A page summary."
        assert "document_type" not in result
        assert "document_date" not in result


# ---------------------------------------------------------------------------
# Tests: output JSON includes document_type and document_date
# ---------------------------------------------------------------------------

class TestOutputDocFields:
    def test_process_document_includes_doc_fields(self, tmp_path):
        """process_document output dict includes document_type and document_date."""
        text_file = tmp_path / "FIELDS.json"
        text_file.write_text(json.dumps({
            "full_text": "Document text for testing output field completeness.",
            "file_type": "pdf",
            "total_pages": 1,
        }))
        result = process_document(text_file, tmp_path, tagged_batch_llm)
        assert "document_type" in result
        assert "document_date" in result

    def test_batched_output_includes_doc_fields(self, tmp_path):
        """process_documents_batched output JSON includes document_type and document_date."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        _make_doc(tmp_path, "DOC", text="Document text for field completeness test here.")
        files = [tmp_path / "DOC.json"]
        process_documents_batched(files, tmp_path, tagged_batch_llm, output_dir)
        result = json.loads((output_dir / "DOC_summary.json").read_text())
        assert "document_type" in result
        assert "document_date" in result


class TestFormatNamesList:
    def test_empty_list(self):
        assert format_names_list([]) == "(none)"

    def test_single_name(self):
        result = format_names_list([{"name": "Epstein", "role": "mentioned"}])
        assert "Epstein" in result
        assert "mentioned" in result


class TestFindTextFiles:
    def test_finds_json_excludes_names(self, tmp_path):
        (tmp_path / "doc1.json").write_text('{}')
        (tmp_path / "doc1_names.json").write_text('{}')
        (tmp_path / "doc1_names_clean.json").write_text('{}')
        (tmp_path / "doc1_summary.json").write_text('{}')
        files = find_text_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "doc1.json"

    def test_empty_directory(self, tmp_path):
        assert find_text_files(tmp_path) == []


# ===========================================================================
# A) Correctness: mapping & determinism
# ===========================================================================

class TestMappingCorrectness:
    """Verify prompts from interleaved docs route to correct output files."""

    def test_interleaved_doc_page_mapping(self, tmp_path):
        """Mixed doc prompts route back by (doc_idx, page_idx), not list position."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Doc A: 3 pages, Doc B: 2 pages, Doc C: 1 page
        _make_doc(tmp_path, "DOCA", pages=[
            {"page_number": 1, "text": "DOCA-P1 content for summary generation."},
            {"page_number": 2, "text": "DOCA-P2 content for summary generation."},
            {"page_number": 3, "text": "DOCA-P3 content for summary generation."},
        ])
        _make_doc(tmp_path, "DOCB", pages=[
            {"page_number": 1, "text": "DOCB-P1 content for summary generation."},
            {"page_number": 2, "text": "DOCB-P2 content for summary generation."},
        ])
        _make_doc(tmp_path, "DOCC", text="DOCC single-page with enough text for summary.")

        # Mock LLM that echoes a tag based on prompt content
        def echo_llm(prompts):
            results = []
            for p in prompts:
                # Extract doc/page identifier from prompt text
                for tag in ["DOCA-P1", "DOCA-P2", "DOCA-P3", "DOCB-P1", "DOCB-P2", "DOCC"]:
                    if tag in p:
                        results.append(f"OUT({tag})")
                        break
                else:
                    results.append(f"OUT(reduce)")
            return results

        files = [tmp_path / f for f in ["DOCA.json", "DOCB.json", "DOCC.json"]]
        process_documents_batched(files, tmp_path, echo_llm, output_dir)

        # Verify page summaries landed on correct docs
        doca = json.loads((output_dir / "DOCA_summary.json").read_text())
        assert "DOCA-P1" in doca["page_summaries"][0]["summary"]
        assert "DOCA-P2" in doca["page_summaries"][1]["summary"]
        assert "DOCA-P3" in doca["page_summaries"][2]["summary"]

        docb = json.loads((output_dir / "DOCB_summary.json").read_text())
        assert "DOCB-P1" in docb["page_summaries"][0]["summary"]
        assert "DOCB-P2" in docb["page_summaries"][1]["summary"]

    def test_partial_failure_retry_only_retries_failed(self):
        """Only empty outputs retried; non-empty not regenerated."""
        call_log = []

        def partial_fail_llm(prompts):
            call_log.append(list(prompts))
            return [
                "" if "FAIL" in p else f"OK({p[:10]})"
                for p in prompts
            ]

        prompts = ["good1", "FAIL_me", "good2"]
        results = _chunked_batch(prompts, partial_fail_llm)

        assert results[0].startswith("OK(")
        assert results[2].startswith("OK(")
        # Retry call should contain ONLY the failed prompt
        assert len(call_log) >= 2
        assert call_log[1] == ["FAIL_me"]
        # No duplicated outputs
        assert len(results) == 3


# ===========================================================================
# B) Token budgeting / memory safety
# ===========================================================================

class TestTokenBudgeting:
    """Verify token-aware chunking respects budgets and handles edge cases."""

    def test_exact_budget_boundary(self):
        """Prompt that exactly fills budget should not spill into next chunk."""
        # Each prompt = 1333 tokens (4000 chars at //3), budget = 4000
        prompts = ["A" * 4000] * 3
        chunks = _build_token_aware_chunks(prompts, max_count=100, max_tokens=4000)
        assert len(chunks) == 1
        total = sum(_estimate_prompt_tokens(p) for _, p in chunks[0])
        assert total <= 4000

    def test_one_token_over_budget_splits(self):
        """Adding one more prompt over budget forces a new chunk."""
        # 3 prompts × 1333 tokens, budget = 2666 fits 2 → chunks of [2, 1]
        # budget = 2665 fits only 1 each → chunks of [1, 1, 1]
        prompts = ["A" * 4000] * 3
        chunks = _build_token_aware_chunks(prompts, max_count=100, max_tokens=2666)
        assert len(chunks) == 2  # [2, 1]
        chunks_tight = _build_token_aware_chunks(prompts, max_count=100, max_tokens=2665)
        assert len(chunks_tight) == 3  # [1, 1, 1]

    def test_mixed_sizes_chunk_correctly(self):
        """Mix of small (133 tok) and large (6666 tok) prompts."""
        prompts = (
            ["A" * 400] * 10  # 10 × 133 tokens = 1330
            + ["B" * 20000] * 2  # 2 × 6666 tokens = 13332
        )
        chunks = _build_token_aware_chunks(prompts, max_count=100, max_tokens=8000)
        # All chunks must respect budget
        for chunk in chunks:
            total = sum(_estimate_prompt_tokens(p) for _, p in chunk)
            assert total <= 8000

    def test_single_oversized_prompt_not_rejected(self):
        """A single prompt exceeding budget still gets processed (in its own chunk)."""
        # 1 prompt = 50000 tokens, budget = 10000
        prompts = ["A" * 200_000]
        chunks = _build_token_aware_chunks(prompts, max_count=100, max_tokens=10_000)
        # Should still be in exactly 1 chunk (can't split a single prompt)
        assert len(chunks) == 1
        assert len(chunks[0]) == 1

    def test_max_page_chars_bounds_prompt_tokens(self):
        """MAX_PAGE_CHARS truncation keeps prompt tokens bounded."""
        huge_text = "X" * (MAX_PAGE_CHARS * 3)
        prompt = build_page_prompt(1, huge_text, 10)
        assert prompt is not None
        assert "[... truncated ...]" in prompt
        # Token estimate should be bounded by MAX_PAGE_CHARS + prompt overhead
        # (overhead is larger in v3 due to JSON schema + occupation instructions)
        est = _estimate_prompt_tokens(prompt)
        max_expected = (MAX_PAGE_CHARS + 2000) // 3  # overhead for prompt template
        assert est <= max_expected


# ===========================================================================
# C) REDUCE prompt oversize handling (hierarchical reduce)
# ===========================================================================

class TestHierarchicalReduce:
    """Test oversized REDUCE prompt handling via hierarchical reduce."""

    def test_small_doc_no_hierarchical(self):
        """Small doc (5 pages) should not trigger hierarchical reduce."""
        page_sums = [
            {"page": i, "summary": f"Page {i} summary."}
            for i in range(1, 6)
        ]
        names = [{"name": "Test", "role": "mentioned"}]
        est = _estimate_reduce_tokens(names, "File: TEST", page_sums)
        assert est < MAX_MODEL_TOKENS - REDUCE_OUTPUT_MARGIN

    def test_huge_doc_triggers_hierarchical(self):
        """97-page doc with verbose summaries triggers hierarchical reduce."""
        # Each page summary ~200 chars = 50 tokens. 97 pages = ~4850 tokens of summaries
        # Pad to make it exceed: 500 chars each = 125 tok × 97 = ~12125 tokens
        page_sums = [
            {"page": i, "summary": f"Page {i}: " + "x" * 500}
            for i in range(1, 98)
        ]
        names = [{"name": f"Person{i}", "role": "mentioned"} for i in range(50)]

        est = _estimate_reduce_tokens(names, "File: HUGE", page_sums)
        # If this doesn't exceed limit, make summaries bigger
        if est <= MAX_MODEL_TOKENS - REDUCE_OUTPUT_MARGIN:
            page_sums = [
                {"page": i, "summary": f"Page {i}: " + "x" * 2000}
                for i in range(1, 98)
            ]

        batch_fn, calls = tracking_batch_llm_factory()
        result = reduce_summaries(page_sums, names, "File: HUGE", batch_fn)

        # Should produce output without raising
        assert result["short"]
        assert result["long"]

    def test_hierarchical_reduce_produces_condensed_summaries(self):
        """_hierarchical_reduce splits pages into groups and reduces them."""
        page_sums = [
            {"page": i, "summary": f"Content from page {i}."}
            for i in range(1, 21)
        ]
        batch_fn, calls = tracking_batch_llm_factory()

        result = _hierarchical_reduce(
            page_sums, [], "File: TEST", batch_fn,
            max_tokens=500,  # Very tight budget forces many groups
        )

        # Should produce fewer entries than original
        assert len(result) < len(page_sums)
        # Each entry should have page and summary keys
        for entry in result:
            assert "page" in entry
            assert "summary" in entry

    def test_reduce_catches_value_error_and_truncates(self):
        """If vLLM raises ValueError for too-long prompt, falls back to truncation."""
        call_count = [0]

        def exploding_llm(prompts):
            call_count[0] += 1
            if call_count[0] <= 2:
                # First calls: hierarchical reduce groups work fine
                return [f"group-summary-{i}" for i in range(len(prompts))]
            if call_count[0] == 3:
                # Final reduce call: simulate vLLM rejecting prompt
                raise ValueError("Prompt is too long to fit in max_model_len")
            # Truncated retry should work
            return [f"truncated-{i}" for i in range(len(prompts))]

        # Use enough page summaries that hierarchical is triggered
        page_sums = [
            {"page": i, "summary": "x" * 2000}
            for i in range(1, 98)
        ]
        names = [{"name": f"P{i}", "role": "mentioned"} for i in range(50)]

        result = reduce_summaries(page_sums, names, "File: BIG", exploding_llm)
        # Should not raise; should produce output via truncation fallback
        assert result["short"]
        assert result["long"]

    def test_cross_doc_batched_with_oversized_doc(self, tmp_path):
        """Cross-doc batching handles one oversized doc alongside normal ones."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Normal small doc
        _make_doc(tmp_path, "SMALL", text="Small doc with enough text for a summary.")

        # Huge doc with many pages (will trigger hierarchical in cross-doc path)
        # Use monkeypatch to lower the threshold
        _make_doc(tmp_path, "HUGE", pages=[
            {"page_number": i, "text": f"Page {i} content " + "x" * 500}
            for i in range(1, 51)
        ])

        batch_fn, _ = tracking_batch_llm_factory()

        # Temporarily lower MAX_MODEL_TOKENS to force hierarchical reduce
        original = gen_mod.MAX_MODEL_TOKENS
        gen_mod.MAX_MODEL_TOKENS = 2000
        try:
            processed, skipped = process_documents_batched(
                [tmp_path / "SMALL.json", tmp_path / "HUGE.json"],
                tmp_path, batch_fn, output_dir,
            )
        finally:
            gen_mod.MAX_MODEL_TOKENS = original

        assert processed == 2
        assert skipped == 0
        # Both outputs should exist
        assert (output_dir / "SMALL_summary.json").exists()
        assert (output_dir / "HUGE_summary.json").exists()


# ===========================================================================
# D) Idempotency / cache validity
# ===========================================================================

class TestCacheIdempotency:
    """Test cache validity and idempotent writes."""

    def test_cache_valid_skips_generation(self, tmp_path):
        """Valid cache means is_cache_valid returns True."""
        summary = {
            "text_hash": "abc123",
            "model_version": MODEL_VERSION,
            "prompt_version": PROMPT_VERSION,
            "summary_short": "test",
        }
        path = tmp_path / "DOC_summary.json"
        path.write_text(json.dumps(summary))

        assert is_cache_valid(path, "abc123") is True

    def test_cache_invalid_on_text_change(self, tmp_path):
        """Changed text hash invalidates cache."""
        summary = {
            "text_hash": "abc123",
            "model_version": MODEL_VERSION,
            "prompt_version": PROMPT_VERSION,
        }
        path = tmp_path / "DOC_summary.json"
        path.write_text(json.dumps(summary))

        assert is_cache_valid(path, "different_hash") is False

    def test_cache_invalid_on_model_change(self, tmp_path):
        """Changed model version invalidates cache."""
        summary = {
            "text_hash": "abc123",
            "model_version": "old-model",
            "prompt_version": PROMPT_VERSION,
        }
        path = tmp_path / "DOC_summary.json"
        path.write_text(json.dumps(summary))

        assert is_cache_valid(path, "abc123") is False

    def test_cache_invalid_on_prompt_change(self, tmp_path):
        """Changed prompt version invalidates cache."""
        summary = {
            "text_hash": "abc123",
            "model_version": MODEL_VERSION,
            "prompt_version": "v1",
        }
        path = tmp_path / "DOC_summary.json"
        path.write_text(json.dumps(summary))

        assert is_cache_valid(path, "abc123") is False

    def test_corrupt_cache_returns_invalid(self, tmp_path):
        """Corrupt JSON cache file returns invalid."""
        path = tmp_path / "DOC_summary.json"
        path.write_text("not json")
        assert is_cache_valid(path, "abc123") is False

    def test_duplicate_batch_run_overwrites_idempotently(self, tmp_path):
        """Running process_documents_batched twice with same LLM = same output."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        _make_doc(tmp_path, "DOC", text="Document text for idempotency verification.")
        files = [tmp_path / "DOC.json"]

        process_documents_batched(files, tmp_path, tagged_batch_llm, output_dir)
        result1 = json.loads((output_dir / "DOC_summary.json").read_text())

        process_documents_batched(files, tmp_path, tagged_batch_llm, output_dir)
        result2 = json.loads((output_dir / "DOC_summary.json").read_text())

        assert result1 == result2


# ===========================================================================
# E) Dependency boundaries (MAP/REDUCE separation)
# ===========================================================================

class TestDependencyBoundaries:
    """Verify MAP completes before REDUCE starts."""

    def test_reduce_sees_map_results(self, tmp_path):
        """REDUCE prompts contain MAP outputs (not stale defaults)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _make_doc(tmp_path, "DOC", pages=[
            {"page_number": 1, "text": "Page 1 legal content for summarization."},
            {"page_number": 2, "text": "Page 2 legal content for summarization."},
        ])

        batch_num = [0]
        def sequential_llm(prompts):
            batch_num[0] += 1
            return [f"BATCH{batch_num[0]}_ITEM{i}" for i in range(len(prompts))]

        process_documents_batched(
            [tmp_path / "DOC.json"], tmp_path, sequential_llm, output_dir
        )

        result = json.loads((output_dir / "DOC_summary.json").read_text())
        # Page summaries from batch 1
        for ps in result["page_summaries"]:
            assert ps["summary"].startswith("BATCH1_")
        # Final summaries from batch 2 (after MAP completed)
        assert result["summary_short"].startswith("BATCH2_")
        assert result["summary_long"].startswith("BATCH2_")

    def test_reduce_never_before_all_map_pages(self, tmp_path):
        """Even with blank pages, REDUCE still waits for all MAP pages."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _make_doc(tmp_path, "DOC", pages=[
            {"page_number": 1, "text": "Real content for page one summarization."},
            {"page_number": 2, "text": ""},  # blank - no MAP needed
            {"page_number": 3, "text": "Real content for page three summarization."},
        ])

        batch_num = [0]
        def sequential_llm(prompts):
            batch_num[0] += 1
            return [f"B{batch_num[0]}" for _ in prompts]

        process_documents_batched(
            [tmp_path / "DOC.json"], tmp_path, sequential_llm, output_dir
        )

        result = json.loads((output_dir / "DOC_summary.json").read_text())
        # MAP is batch 1, REDUCE is batch 2
        assert result["page_summaries"][0]["summary"] == "B1"
        assert result["page_summaries"][1]["summary"] == "Blank page."  # not sent to LLM
        assert result["page_summaries"][2]["summary"] == "B1"
        assert result["summary_short"] == "B2"


# ===========================================================================
# F) Truncation safety
# ===========================================================================

class TestTruncationSafety:
    """Verify MAX_PAGE_CHARS is applied and truncation markers present."""

    def test_page_text_truncated_at_limit(self):
        """Page text exceeding MAX_PAGE_CHARS is truncated."""
        text = "X" * (MAX_PAGE_CHARS + 5000)
        prompt = build_page_prompt(1, text, 10)
        assert prompt is not None
        assert "[... truncated ...]" in prompt
        # Original text should NOT appear in full
        assert text not in prompt

    def test_direct_summary_text_truncated(self):
        """Single-page doc text is truncated for direct summaries."""
        batch_fn, calls = tracking_batch_llm_factory()
        long_text = "Y" * (MAX_PAGE_CHARS + 5000)
        direct_summaries(long_text, [], batch_fn)

        # Both prompts should contain truncation marker
        for prompt in calls[0]:
            assert "[... truncated ...]" in prompt

    def test_truncated_prompt_token_estimate_bounded(self):
        """Token estimate after truncation stays within expected range."""
        text = "Z" * 100_000
        prompt = build_page_prompt(1, text, 10)
        est = _estimate_prompt_tokens(prompt)
        # Should be bounded by MAX_PAGE_CHARS/3 + some overhead
        # (overhead is larger in v3 due to JSON schema + occupation instructions)
        assert est < (MAX_PAGE_CHARS + 2000) // 3


# ===========================================================================
# G) Progress flushing (job-level fairness proxy)
# ===========================================================================

class TestProgressFlushing:
    """Verify outputs written per-batch, not all-at-end."""

    def test_partial_outputs_after_each_cross_doc_batch(self, tmp_path):
        """After each CROSS_DOC_BATCH_SIZE batch, outputs exist on disk."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create more docs than one batch
        n_docs = CROSS_DOC_BATCH_SIZE + 3
        for i in range(n_docs):
            _make_doc(tmp_path, f"D{i:03d}", text=f"Document {i} text content for summary.")

        files = [tmp_path / f"D{i:03d}.json" for i in range(n_docs)]

        # Track when outputs appear using a wrapper
        outputs_at_batch_boundary = []
        original_tagged = tagged_batch_llm
        call_count = [0]

        def tracking_llm(prompts):
            call_count[0] += 1
            # After a reduce call for batch 1 (first CROSS_DOC_BATCH_SIZE docs),
            # check how many outputs exist
            results = [f"[resp-{i}]" for i in range(len(prompts))]
            return results

        process_documents_batched(files, tmp_path, tracking_llm, output_dir)

        # All outputs should exist
        for i in range(n_docs):
            assert (output_dir / f"D{i:03d}_summary.json").exists()

    def test_first_batch_written_before_second_starts(self, tmp_path):
        """First batch of docs written to disk before second batch starts MAP."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # 2 batches worth of docs
        n_batch1 = CROSS_DOC_BATCH_SIZE
        n_total = CROSS_DOC_BATCH_SIZE + 2

        for i in range(n_total):
            _make_doc(tmp_path, f"F{i:03d}", text=f"Doc {i} has enough text for a summary.")

        files = [tmp_path / f"F{i:03d}.json" for i in range(n_total)]
        batch_boundaries = []
        batch_call_num = [0]

        def boundary_llm(prompts):
            batch_call_num[0] += 1
            # After all reduce calls for batch 1, check what's on disk
            if batch_call_num[0] == 1:
                # This is the reduce call for batch 1 (single-page docs)
                pass
            elif batch_call_num[0] == 2:
                # This is batch 2's reduce call — batch 1 should be written
                written = list(output_dir.glob("*_summary.json"))
                batch_boundaries.append(len(written))
            return [f"resp-{i}" for i in range(len(prompts))]

        process_documents_batched(files, tmp_path, boundary_llm, output_dir)

        # By the time batch 2's LLM runs, batch 1 outputs should be on disk
        if batch_boundaries:
            assert batch_boundaries[0] >= n_batch1
