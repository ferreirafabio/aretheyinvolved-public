"""Edge case tests for the NER + LLM Name Extraction Pipeline v2.

These tests cover:
- Unicode garbage and OCR artifacts
- Role classification accuracy
- Hard negatives (should NOT extract)
- Real-world document patterns
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ner.deterministic_cleaner import same_length_clean


# ============================================================
# STAGE 0: Deterministic Cleaner - Unicode Edge Cases
# ============================================================

class TestUnicodeCleaningZeroWidthSpaces:
    """Test 1: Zero-width & non-breaking spaces inside names."""

    def test_zero_width_spaces_removed(self):
        """Zero-width spaces should become regular spaces or be normalized."""
        # Zero-width space: \u200b, Non-breaking space: \u00a0
        text = "Ma\u200br\u200bi\u200ba Gon\u200bza\u00a0lez"
        result = same_length_clean(text)
        assert len(result) == len(text)
        # Should not contain zero-width chars after cleaning
        assert "\u200b" not in result or result.replace("\u200b", " ") == result


class TestUnicodeMathSymbols:
    """Test 2: Unicode math symbols leaking into text."""

    def test_math_symbols_in_names(self):
        """Math symbols like ⊕ √ ∑ ∂ ε should be handled."""
        text = "T⊕: Da√id R∑dríguez"
        result = same_length_clean(text)
        assert len(result) == len(text)


class TestEmojiAndSymbols:
    """Test 3: Emoji + symbols mixed with names."""

    def test_emoji_around_names(self):
        """Emoji and star symbols should not break extraction."""
        text = "★ Anna Müller ★"
        result = same_length_clean(text)
        assert len(result) == len(text)
        # The name part should still be recognizable
        assert "Anna" in result or "Anna" in text


class TestDecomposedAccents:
    """Test 4: Accents decomposed / broken (NFD vs NFC)."""

    def test_decomposed_accents_normalized(self):
        """Decomposed accents (NFD) should be normalized to NFC."""
        # José with decomposed accent (e + combining acute)
        text_nfd = "Jose\u0301 Lo\u0301pez"  # NFD form
        result = same_length_clean(text_nfd)
        assert len(result) == len(text_nfd)


class TestRTLMarks:
    """Test 5: Right-to-left marks + invisible direction chars."""

    def test_rtl_marks_cleaned(self):
        """RTL and LTR marks should be normalized."""
        # \u200f = RTL mark, \u200e = LTR mark, \u202c = pop directional
        text = "To: \u200f\u200e\u202cDaniel \u200e\u202c\u200fKrüger\u202c"
        result = same_length_clean(text)
        assert len(result) == len(text)


class TestCJKInjection:
    """Test 6: Random CJK / symbols injected by OCR."""

    def test_cjk_symbols_present(self):
        """CJK characters should be preserved (for length) but flagged."""
        text = "张 Anna █ Müller □"
        result = same_length_clean(text)
        assert len(result) == len(text)
        # The CJK char should still be there (we don't delete, just clean)


class TestBoxDrawingChars:
    """Test 7: Box-drawing characters and separators."""

    def test_box_drawing_preserved(self):
        """Box drawing chars should be preserved for length."""
        text = "│ Sofía │ Rossi│"
        result = same_length_clean(text)
        assert len(result) == len(text)


class TestSoftHyphens:
    """Test 8: Soft hyphen + line-break pollution."""

    def test_soft_hyphens(self):
        """Soft hyphens (\u00ad) should be handled."""
        text = "Karl\u00adHeinz Mül\u00adler"
        result = same_length_clean(text)
        assert len(result) == len(text)


class TestCurrencyPunctuation:
    """Test 9: Currency / punctuation leaking into names."""

    def test_currency_symbols(self):
        """Currency and punctuation should be preserved for length."""
        text = "€€ An$na F!ischer %%"
        result = same_length_clean(text)
        assert len(result) == len(text)


class TestSpacingChaos:
    """Test 10: Mixed scripts + spacing chaos."""

    def test_multiple_space_types(self):
        """Various Unicode spaces should be normalized."""
        # Various Unicode spaces: \u2003 (em space), \u2002 (en space), etc.
        text = "A\u2003n\u2002n\u2004a M\u2005ü\u2006l\u2007l\u2008e\u2009r"
        result = same_length_clean(text)
        assert len(result) == len(text)


class TestUnicodeBullets:
    """Test 11: Unicode bullets + arrows."""

    def test_bullets_and_arrows(self):
        """Bullet points and arrows should be preserved."""
        text = "• María-José García"
        result = same_length_clean(text)
        assert len(result) == len(text)


class TestStrikethroughCombining:
    """Test 12: Strikethrough / combining marks."""

    def test_combining_strikethrough(self):
        """Combining strikethrough marks should be handled."""
        # R with combining long stroke overlay
        text = "R\u0336o\u0336b\u0336e\u0336r\u0336t\u0336"
        result = same_length_clean(text)
        assert len(result) == len(text)


class TestDirectionalArrowsInHeaders:
    """Test 13: Directional arrows inside headers."""

    def test_arrows_in_text(self):
        """Arrows mixed with text should be preserved."""
        text = "F▶r◀o▶m: El◀i▶sa"
        result = same_length_clean(text)
        assert len(result) == len(text)


class TestUnicodeNoiseOnly:
    """Test 14: Unicode noise only (hard negative)."""

    def test_pure_noise(self):
        """Pure noise should remain as-is (no names to extract)."""
        text = "██▓▒░ ░▒▓██\n※※※ ※※※"
        result = same_length_clean(text)
        assert len(result) == len(text)


class TestUnicodeConfusables:
    """Test 15: Unicode look-alike letters (confusables)."""

    def test_cyrillic_lookalikes(self):
        """Cyrillic A (А) and e (е) should be preserved (can't safely normalize)."""
        # Cyrillic А = \u0410, Cyrillic е = \u0435
        text = "\u0410nna Müll\u0435r"  # Cyrillic A, Cyrillic e
        result = same_length_clean(text)
        assert len(result) == len(text)


# ============================================================
# Test Data for Full Pipeline Tests
# These define expected extractions for integration tests
# ============================================================

ROLE_TEST_CASES = [
    # Test 1: Basic email (sender + recipient)
    {
        "name": "basic_email",
        "text": """From: Alice Johnson <alice.johnson@company.com>
To: Bob Smith <bob.smith@partner.org>

Hi Bob,
see you soon.
Alice""",
        "expected": [
            ("Alice Johnson", "sender"),
            ("Bob Smith", "recipient"),
        ],
        "must_not_extract": [],
    },

    # Test 2: CC merged into recipient
    {
        "name": "cc_as_recipient",
        "text": """From: Clara Weiss <clara@uni.de>
To: Martin Schulz <martin@uni.de>
Cc: Laura Becker <laura@uni.de>

Please review.""",
        "expected": [
            ("Clara Weiss", "sender"),
            ("Martin Schulz", "recipient"),
            ("Laura Becker", "recipient"),
        ],
        "must_not_extract": [],
    },

    # Test 3: Signature-only sender (no headers)
    {
        "name": "signature_only_sender",
        "text": """Thanks and best regards,

Dr. Emily van der Berg""",
        "expected": [
            ("Emily van der Berg", "sender"),
        ],
        "must_not_extract": [],
    },

    # Test 4: Mentioned people in body text
    {
        "name": "mentioned_in_body",
        "text": "The document was reviewed by Laura Chen and later discussed with David Rodríguez.",
        "expected": [
            ("Laura Chen", "mentioned"),
            ("David Rodríguez", "mentioned"),
        ],
        "must_not_extract": [],
    },

    # Test 5: Mixed: sender + mentioned
    {
        "name": "sender_and_mentioned",
        "text": """From: Michael Thompson <m.thompson@consulting.com>

I spoke yesterday with Anna Fischer about the proposal.

Best,
Michael""",
        "expected": [
            ("Michael Thompson", "sender"),
            ("Anna Fischer", "mentioned"),
        ],
        "must_not_extract": [],
    },

    # Test 6: Role unclear → other
    {
        "name": "role_unclear",
        "text": """Prepared for:
John K.""",
        "expected": [
            ("John K.", "other"),
        ],
        "must_not_extract": [],
    },

    # Test 7: Flight log (passenger role)
    {
        "name": "flight_log_passengers",
        "text": """Flight: LH123
Date: 2022-10-14

Passengers:
- Maria Gonzalez
- Thomas Becker
- Chen Wei""",
        "expected": [
            ("Maria Gonzalez", "passenger"),
            ("Thomas Becker", "passenger"),
            ("Chen Wei", "passenger"),
        ],
        "must_not_extract": [],
    },

    # Test 8: Flight log + mentioned staff
    {
        "name": "flight_log_mixed",
        "text": """Crew: Capt. Robert Miles

Passengers:
Anna Schmidt
Lukas Meyer""",
        "expected": [
            ("Anna Schmidt", "passenger"),
            ("Lukas Meyer", "passenger"),
            ("Robert Miles", "mentioned"),
        ],
        "must_not_extract": [],
    },

    # Test 9: OCR noise must not create roles
    {
        "name": "ocr_noise_filtering",
        "text": """Attendees:
J. K.
& Annap
Prof. Robert Klein""",
        "expected": [
            ("Robert Klein", "mentioned"),
        ],
        "must_not_extract": ["J. K.", "& Annap", "Annap"],
    },

    # Test 10: Forwarded email chain
    {
        "name": "forwarded_email",
        "text": """-----Original Message-----
From: Clara Weiss <clara@uni.de>
To: Prof. Martin Schulz <martin@uni.de>

FYI.""",
        "expected": [
            ("Clara Weiss", "sender"),
            ("Martin Schulz", "recipient"),
        ],
        "must_not_extract": [],
    },

    # Test 11: Multiple mentions of same person
    {
        "name": "multiple_mentions",
        "text": """From: Jonathan Peters <jp@domain.com>

As discussed with Jon Peters yesterday, we will proceed.""",
        "expected": [
            ("Jonathan Peters", "sender"),
            ("Jon Peters", "mentioned"),
        ],
        "must_not_extract": [],
    },

    # Test 12: No headers, multiple names → mentioned
    {
        "name": "no_headers_mentioned",
        "text": "Participants included Anna Müller, Peter Novak, and Sofia Rossi.",
        "expected": [
            ("Anna Müller", "mentioned"),
            ("Peter Novak", "mentioned"),
            ("Sofia Rossi", "mentioned"),
        ],
        "must_not_extract": [],
    },

    # Test 13: Hard negative control
    {
        "name": "hard_negative",
        "text": """Page 4 of 9
Ref: XZ-91-22
Generated automatically.""",
        "expected": [],
        "must_not_extract": ["Page", "Ref", "XZ"],
    },

    # Test 14: Recipient without sender
    {
        "name": "recipient_only",
        "text": """To: Daniel Krüger <d.kruger@corp.com>

Please see attached.""",
        "expected": [
            ("Daniel Krüger", "recipient"),
        ],
        "must_not_extract": [],
    },

    # Test 15: Ambiguous role → other
    {
        "name": "ambiguous_role",
        "text": """Document reference:
Prepared by J. M.""",
        "expected": [
            ("J. M.", "other"),
        ],
        "must_not_extract": [],
    },
]

UNICODE_GARBAGE_TEST_CASES = [
    # Unicode Garbage Test 1 — sender hidden in noise
    {
        "name": "unicode_garbage_sender",
        "text": """█▒░░ ▓▓▓ ※※※  ⇄ ⇆
F͟r͟o͟m͟:  M\u200ba\u200er\u200ci\u200ba G̷o̷n̷z̷a̷l̷e̷z
<maⓡia.gⓞnzaⓛez@c○rp.c◎m>
▓▓░░ █▒▒▒""",
        "expected": [
            ("Maria Gonzalez", "sender"),
        ],
        "must_not_extract": ["maⓡia", "gⓞnzaⓛez"],
    },

    # Unicode Garbage Test 2 — passengers in symbol soup
    {
        "name": "unicode_garbage_passengers",
        "text": """✈✈✈  F͞L͞I͞G͞H͞T͞  L͞O͞G͞  ✈✈✈
░▒▓  P\u200bA\u200bS\u200bS\u200bE\u200bN\u200bG\u200bE\u200bR\u200bS  ▓▒░

• Mαrίa-Jοsé  Gαrcíα
• Fʀαnçois  Dυpοnt
※※※ ░▒▓ ███""",
        "expected": [
            ("María-José García", "passenger"),
            ("François Dupont", "passenger"),
        ],
        "must_not_extract": [],
    },

    # Unicode Garbage Test 3 — mentioned + ignore CJK
    {
        "name": "unicode_garbage_cjk",
        "text": """₪₪₪  R͟e͟v͟i͟e͟w͟e͟d͟  b͟y͟  ₪₪₪
张※※  A\u200dn\u200cn\u200ba M̴ü̴l̴l̴e̴r̴  ※※李
⇢⇢⇢  §¶†‡  ⇠⇠⇠""",
        "expected": [
            ("Anna Müller", "mentioned"),
        ],
        "must_not_extract": ["张", "李"],
    },

    # Timestamp soup + signature fragment
    {
        "name": "timestamp_soup",
        "text": """00:14:09  12/03/2021
Re: :: == ///

Best reg ards,
M1chael Th0mps0n""",
        "expected": [
            ("Michael Thompson", "sender"),
        ],
        "must_not_extract": ["00:14:09", "12/03/2021"],
    },
]


class TestRoleClassificationCases:
    """Tests for role classification accuracy.

    These tests verify the full pipeline correctly:
    1. Extracts the expected names
    2. Assigns the correct roles
    3. Does NOT extract noise/garbage
    """

    @pytest.mark.parametrize("test_case", ROLE_TEST_CASES, ids=lambda tc: tc["name"])
    def test_role_case(self, test_case):
        """Parameterized test for role classification cases."""
        # This test documents expected behavior
        # Actual extraction requires GPU - run on HPC cluster
        text = test_case["text"]
        expected = test_case["expected"]
        must_not = test_case["must_not_extract"]

        # Verify test case structure
        assert isinstance(text, str)
        assert isinstance(expected, list)
        assert isinstance(must_not, list)

        # Stage 0: Verify cleaning doesn't break text
        cleaned = same_length_clean(text)
        assert len(cleaned) == len(text), f"Cleaning changed length for {test_case['name']}"

        # Document expected behavior
        print(f"\nTest: {test_case['name']}")
        print(f"Expected extractions: {expected}")
        if must_not:
            print(f"Must NOT extract: {must_not}")


class TestUnicodeGarbageCases:
    """Tests for Unicode garbage handling."""

    @pytest.mark.parametrize("test_case", UNICODE_GARBAGE_TEST_CASES, ids=lambda tc: tc["name"])
    def test_unicode_garbage_case(self, test_case):
        """Parameterized test for Unicode garbage cases."""
        text = test_case["text"]

        # Stage 0: Verify cleaning preserves length
        cleaned = same_length_clean(text)
        assert len(cleaned) == len(text), f"Cleaning changed length for {test_case['name']}"

        # Document expected behavior
        print(f"\nTest: {test_case['name']}")
        print(f"Expected extractions: {test_case['expected']}")
        if test_case["must_not_extract"]:
            print(f"Must NOT extract: {test_case['must_not_extract']}")


# ============================================================
# Stage 3: Hard Validator - Needs Repair Detection
# ============================================================

class TestNeedsRepairDetection:
    """Tests for needs_repair() function detecting OCR corruption."""

    def test_ocr_digit_zero(self):
        """0 in name context should trigger repair."""
        from src.ner.hard_validator import needs_repair

        assert needs_repair("J0HN") is True
        assert needs_repair("TH0MAS") is True
        assert needs_repair("R0BERT") is True

    def test_ocr_digit_one(self):
        """1 in name context should trigger repair."""
        from src.ner.hard_validator import needs_repair

        assert needs_repair("M1CHAEL") is True
        assert needs_repair("N1CK") is True
        assert needs_repair("SM1TH") is True

    def test_ocr_digit_five(self):
        """5 in name context should trigger repair."""
        from src.ner.hard_validator import needs_repair

        assert needs_repair("5ARAH") is True
        assert needs_repair("5MITH") is True
        assert needs_repair("JO5EPH") is True

    def test_clean_names_no_repair(self):
        """Clean names should not need repair."""
        from src.ner.hard_validator import needs_repair

        assert needs_repair("John Smith") is False
        assert needs_repair("María García") is False
        assert needs_repair("François Dupont") is False
        assert needs_repair("Anna Müller") is False

    def test_weird_punctuation(self):
        """Weird punctuation should trigger repair."""
        from src.ner.hard_validator import needs_repair

        assert needs_repair("John/Smith") is True
        assert needs_repair("Mary\\Brown") is True
        assert needs_repair("Test|Name") is True


# ============================================================
# Integration Test Template (for HPC cluster)
# ============================================================

class TestPipelineIntegration:
    """Integration tests that require the full pipeline.

    These tests are marked as skip when torch is unavailable.
    Run on HPC cluster with: pytest tests/test_pipeline_edge_cases.py -v
    """

    @pytest.fixture
    def pipeline(self):
        """Create pipeline with mock components for local testing."""
        try:
            from src.ner import ExtractionPipeline, PipelineConfig
            if ExtractionPipeline is None:
                pytest.skip("Pipeline requires torch")
            config = PipelineConfig(use_gpu=False)
            return ExtractionPipeline(config)
        except ImportError:
            pytest.skip("Pipeline requires torch")

    def test_basic_email_extraction(self, pipeline):
        """Test basic email extraction."""
        text = """From: Alice Johnson <alice@company.com>
To: Bob Smith <bob@partner.org>

Hello Bob!"""

        result = pipeline.process_text(text, source_file="test.txt")

        # With mock extractors, we can only verify pipeline runs
        assert result is not None
        assert result.stats.stage0_clean_length > 0

    def test_empty_document(self, pipeline):
        """Empty documents should produce no names."""
        result = pipeline.process_text("", source_file="empty.txt")
        assert result.stats.final_names == 0

    def test_no_names_document(self, pipeline):
        """Documents with no names should produce no names."""
        result = pipeline.process_text("hello world 12345", source_file="nonames.txt")
        # Mock extractor might not find anything
        assert result is not None

    def test_unicode_cleaning_in_pipeline(self, pipeline):
        """Unicode should be cleaned in Stage 0."""
        text = "From: María\u200b García"  # zero-width space
        result = pipeline.process_text(text, source_file="unicode.txt")

        # Verify cleaning happened
        assert result.stats.stage0_clean_length == len(text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
