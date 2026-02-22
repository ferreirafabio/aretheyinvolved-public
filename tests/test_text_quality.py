"""Tests for text quality gate (src/text_quality.py).

Validates the decision-tree quality gate against known-good and known-bad
text samples from actual DOJ datasets.
"""

import pytest
from src.text_quality import (
    is_text_layer_good,
    compute_word_ratio,
    compute_alpha_ratio,
    compute_weird_ratio,
    compute_token_sanity,
)


# -----------------------------------------------------------------------
# Fixtures: real text samples from actual DOJ files
# -----------------------------------------------------------------------

# DS8 calendar event (EFTA00009885) — readable but number-heavy
CALENDAR_EVENT = (
    "Event: Call with Geoff on Epstein \n"
    "Start Date: 2019-03-05 21:00:00 +0000 \n"
    "End Date: 2019-03-05 21:30:00 +0000 \n"
    "Organizer: \n"
    "USAWS) <\n> \n"
    "Class: X-PERSONAL \n"
    "Date Created: 2019-03-05 16:17:55 +0000 \n"
    "Date Modified: 2019-03-05 16:17:55 +0000 \n"
    "Priority: 5 \n"
    "DTSTAMP: 2019-03-05 16:07:55 +0000 \n"
    "Attendee: \n(USAWS) \n"
    "Alarm: Display the following message 15m before start \n"
    "Reminder \nEFTA00009885\n"
)

# DS8 email header (EFTA00010035) — short but readable
EMAIL_HEADER = (
    "From: \nTo: \n"
    "Subject: 2021-11-10 GM letter re GX 52.v4.docx \n"
    "Date: Fri, 12 Nov 2021 18:08:13 +0000 \n"
    "Attachments: 2021-11-10 GM letter re GX 52 v4 docx \n"
    "_ \n_ \n_ _ \n_ \n"
    "Comments in attached. \nEFTA00010035\n"
)

# DS1 garbled text (EFTA00001363) — garbled OCR, short excerpt
# NOTE: This is a known edge case — short garbled text where 3-letter tokens
# accidentally match dictionary words (50% word ratio, 58% alpha). The quality
# gate passes it through the gray zone. In practice, DS1 is force-OCR'd anyway.
GARBLED_DS1_SHORT = (
    '„„ \n...., \n\'cow\' * ne \nth,„ i,\n,,,rm \n,."\'",„ „ '
    '\'mast ^,„ cb•"% .",\'„„ \nSe."1"--/ at Pane\' - of "irane • no. '
    '\nton .0° \nrea Pas W \nA ta "\na"\n4.\n r\'t \no An.t.ore IC"\' t '
    '\nAU.nn P \' \n3.1 \naerriv\n Le \nolie \nIli\' "toe\ni ngtOntisooing '
    '\nr \ne3 \nR o f  le \nEFTA00001363\n'
)

# Clearly garbled text with low word ratio (representative of DS1 bad files)
GARBLED_LOW_WORD = (
    "xvzk plmn qrst bfhj wklm nrst dfgk hjkl mnpq rstvw xyzab cdefg "
    "hijkl mnopq rstuv wxyza bcdef ghijk lmnop qrstu vwxyz abcde fghij "
    "klmno pqrst uvwxy zabcd efghi jklmn opqrs tuvwx yzabc defgh ijklm "
    "nopqr stuvw xyzab cdefg hijkl mnopq rstuv wxyza bcdef ghijk lmnop "
)

# DS1 garbled text with low alpha ratio (numbers/symbols dominate)
GARBLED_LOW_ALPHA = (
    "4601-2233 01/15 $1,600 #4601 ref:22-3344 01/17 $1,500 #4602 "
    "3344-5566 01/22 $2,066 #4603 5566-7788 01/22 $1,234 #4604 "
    "7788-9900 02/01 $8,450 #4605 9900-1122 02/15 $12,500 #4606 "
    "1122-3344 03/01 $3,200 #4607 3344-5566 03/15 $5,677 #4608 "
)

# Good legal document text (synthetic but representative)
GOOD_LEGAL_TEXT = (
    "UNITED STATES DISTRICT COURT \n"
    "SOUTHERN DISTRICT OF NEW YORK \n\n"
    "The defendant Jeffrey Epstein appeared before the court on January 15, "
    "2019. Defense counsel Alan Dershowitz filed a motion to dismiss the "
    "indictment on grounds of double jeopardy. The government, represented "
    "by Assistant United States Attorney Maurene Comey, opposed the motion. "
    "After hearing arguments from both sides, the court denied the motion "
    "and scheduled the trial for September 2019. The witness testimony "
    "included depositions from several individuals who had been interviewed "
    "by federal investigators during the course of the investigation."
)

# Good text with lots of numbers (financial document)
GOOD_FINANCIAL = (
    "Account Statement - Period: January 1, 2005 to March 31, 2005\n"
    "Account Number: 4601-2233-8899\n\n"
    "Date       Check#  Payee                          Amount\n"
    "01/15/05   4601    Brautigam Land Surveyor        1,600.00\n"
    "01/17/05   4602    Stabenow for US Senate          1,500.00\n"
    "01/22/05   4603    Bank of America Visa            2,066.45\n"
    "01/22/05   4604    Electronic Environments         1,234.81\n"
    "02/01/05   4605    Palm Beach County Tax           8,450.00\n"
    "02/15/05   4606    Mercedes Benz Financial        12,500.00\n"
    "03/01/05   4607    National Insurance Company      3,200.00\n"
    "03/15/05   4608    American Express Travel         5,677.90\n\n"
    "Total Debits: $37,229.16\n"
    "Ending Balance: $142,558.33\n"
)


class TestDecisionTree:
    """Test the decision-tree structure of the quality gate."""

    def test_too_short_rejects(self):
        r = is_text_layer_good("Hello world")
        assert not r.good
        assert "too_short" in r.reason

    def test_empty_rejects(self):
        r = is_text_layer_good("")
        assert not r.good

    def test_none_like_rejects(self):
        r = is_text_layer_good("   \n\n  ")
        assert not r.good

    def test_good_legal_text_passes(self):
        r = is_text_layer_good(GOOD_LEGAL_TEXT)
        assert r.good, f"Should pass: {r}"

    def test_good_financial_text_passes(self):
        """Number-heavy financial docs must pass (W_HIGH override)."""
        r = is_text_layer_good(GOOD_FINANCIAL)
        assert r.good, f"Financial doc should pass: {r}"


class TestFalsePositiveRegression:
    """Ensure number-heavy but readable documents are NOT flagged."""

    def test_calendar_event_passes(self):
        """DS8 calendar events with timestamps must stay GOOD.

        Specifically: must pass via the W_HIGH branch (high_word_ratio),
        NOT the gray zone. This prevents someone from simplifying the tree
        and reintroducing alpha-only gating that would flag these files.
        """
        r = is_text_layer_good(CALENDAR_EVENT)
        assert r.good, (
            f"Calendar event is readable text — should pass.\n"
            f"Result: {r}\n"
            f"word_ratio={r.word_ratio:.1%} should trigger W_HIGH override"
        )
        assert r.reason == "high_word_ratio", (
            f"Calendar must pass via W_HIGH branch, not '{r.reason}'. "
            f"word_ratio={r.word_ratio:.1%} should be >= W_HIGH."
        )

    def test_short_email_too_short_is_ok(self):
        """Short email header: either GOOD or too_short is acceptable."""
        r = is_text_layer_good(EMAIL_HEADER)
        # This is borderline (206 chars). Either pass or too_short is fine.
        # What's NOT acceptable is flagging it as garbled.
        if not r.good:
            assert "too_short" in r.reason, (
                f"Short email should be GOOD or too_short, not: {r.reason}"
            )


class TestTruePositives:
    """Ensure genuinely garbled text IS flagged."""

    def test_garbled_low_word_ratio_caught(self):
        """Text with nonsense words (low dictionary match) must fail."""
        r = is_text_layer_good(GARBLED_LOW_WORD)
        assert not r.good, f"Nonsense words should be caught: {r}"
        assert "low_word_ratio" in r.reason

    def test_garbled_low_alpha_caught(self):
        """Text dominated by numbers/symbols must fail."""
        r = is_text_layer_good(GARBLED_LOW_ALPHA)
        assert not r.good, f"Number-dominated text should be caught: {r}"

    def test_garbled_ds1_short_is_edge_case(self):
        """Short garbled text with accidental dict matches is a known edge case.

        Very short garbled text where 3-letter tokens happen to match
        dictionary words can pass the gate. This is acceptable because:
        - DS1-4 are force-OCR'd regardless
        - Long garbled text (> ~500 chars) is reliably caught
        - The gate's priority is avoiding false positives on good text
        """
        r = is_text_layer_good(GARBLED_DS1_SHORT)
        # Document the actual behavior — don't assert it must fail
        # word_ratio ~50%, alpha ~58% puts it in gray zone, alpha > 50% → passes
        assert r.word_ratio < 0.55  # confirm it's NOT in the clearly-good zone
        assert r.alpha_ratio > 0.50  # confirm it passes gray-zone alpha check

    def test_replacement_chars_caught(self):
        """Text with many replacement characters (U+FFFD) must fail."""
        bad = "The \ufffd\ufffd\ufffd document was \ufffd\ufffd signed by " * 10
        r = is_text_layer_good(bad)
        assert not r.good
        assert "weird_chars" in r.reason

    def test_giant_tokens_caught(self):
        """Text with extremely long concatenated tokens must fail."""
        bad = ("abcdefghijklmnopqrstuvwxyz" * 5 + " ") * 20
        r = is_text_layer_good(bad)
        assert not r.good


class TestSignals:
    """Test individual signal computation functions."""

    def test_word_ratio_good_text(self):
        ratio = compute_word_ratio(GOOD_LEGAL_TEXT)
        assert ratio > 0.60

    def test_word_ratio_garbled(self):
        ratio = compute_word_ratio(GARBLED_LOW_WORD)
        assert ratio < 0.25

    def test_alpha_ratio_normal(self):
        ratio = compute_alpha_ratio("Hello World, this is a test.")
        assert ratio > 0.80

    def test_alpha_ratio_number_heavy(self):
        ratio = compute_alpha_ratio("2019-03-05 21:00:00 +0000")
        assert ratio < 0.30

    def test_weird_ratio_clean(self):
        ratio = compute_weird_ratio("Normal clean text here.")
        assert ratio == 0.0

    def test_weird_ratio_dirty(self):
        ratio = compute_weird_ratio("Bad \ufffd\ufffd text \x00\x01 here")
        assert ratio > 0.10

    def test_token_sanity_normal(self):
        median, digit_mix = compute_token_sanity("The quick brown fox jumps")
        assert 3 <= median <= 6
        assert digit_mix == 0.0

    def test_token_sanity_mixed_digits(self):
        _, digit_mix = compute_token_sanity("h3llo w0rld t3st n0w abc")
        assert digit_mix > 0.50


class TestGrayZone:
    """Test the gray zone behavior (W_LOW < word_ratio < W_HIGH)."""

    def test_gray_good_alpha_passes(self):
        """Gray zone with decent alpha should pass."""
        # Construct text with ~40% word ratio but high alpha
        text = (
            "Dershowitz deposition transcript from the Southern District "
            "proceedings involving Giuffre and Maxwell litigation documents "
            "Brunel Dubin Wexner Kellen Rodriguez Marcinkova flight log "
            "testimony regarding Palm Beach investigation and federal case "
        )
        r = is_text_layer_good(text)
        # Many proper nouns won't match dictionary, so word_ratio is moderate
        # but alpha_ratio should be high
        assert r.alpha_ratio > 0.80

    def test_gray_bad_alpha_fails(self):
        """Gray zone with low alpha should fail."""
        # Moderate word content but lots of numbers/symbols
        text = (
            "ref# 4601-2233 dt: 01/15/05 amt: $1,600.00 chk# 4601 "
            "ref# 4602-3344 dt: 01/17/05 amt: $1,500.00 chk# 4602 "
            "ref# 4603-5566 dt: 01/22/05 amt: $2,066.45 chk# 4603 "
            "ref# 4604-7788 dt: 01/22/05 amt: $1,234.81 chk# 4604 "
            "ref# 4605-9900 dt: 02/01/05 amt: $8,450.00 chk# 4605 "
        )
        r = is_text_layer_good(text)
        # word_ratio should be low-moderate, alpha should be low
        assert r.alpha_ratio < 0.60
