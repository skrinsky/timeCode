"""
Unit tests for SMPTE timecode parsing and sample offset conversion.

Critical correctness tests — especially drop-frame edge cases.
One-frame error at 29.97 = 33.4ms, just above the A/V sync threshold.
"""

import pytest
from timecode_audio.core.timecode import (
    parse_smpte,
    smpte_to_sample_offset,
    sample_offset_to_smpte,
)

SAMPLE_RATE = 48000


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParseSMPTE:
    def test_basic(self):
        assert parse_smpte("01:02:03:04", "30") == (1, 2, 3, 4)

    def test_zero(self):
        assert parse_smpte("00:00:00:00", "30") == (0, 0, 0, 0)

    def test_semicolon_separator(self):
        # ';' is conventional for drop-frame timecodes
        assert parse_smpte("01:00:00;00", "29.97df") == (1, 0, 0, 0)

    def test_mixed_separators(self):
        assert parse_smpte("00:01:00;02", "29.97df") == (0, 1, 0, 2)

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid SMPTE"):
            parse_smpte("01:02:03", "30")

    def test_invalid_hours(self):
        with pytest.raises(ValueError, match="Hours out of range"):
            parse_smpte("25:00:00:00", "30")

    def test_invalid_minutes(self):
        with pytest.raises(ValueError, match="Minutes out of range"):
            parse_smpte("00:60:00:00", "30")

    def test_invalid_frames(self):
        with pytest.raises(ValueError, match="Frames out of range"):
            parse_smpte("00:00:00:30", "30")  # frame 30 invalid at 30fps

    def test_non_integer(self):
        with pytest.raises(ValueError, match="Non-integer"):
            parse_smpte("00:00:00:ab", "30")


# ---------------------------------------------------------------------------
# Non-drop-frame sample offsets
# ---------------------------------------------------------------------------

class TestNonDropFrame:
    def test_zero_offset(self):
        assert smpte_to_sample_offset("00:00:00:00", "30", SAMPLE_RATE) == 0

    def test_one_frame_30fps(self):
        # 1 frame at 30fps / 48000 Hz = 1600 samples
        assert smpte_to_sample_offset("00:00:00:01", "30", SAMPLE_RATE) == 1600

    def test_one_second_30fps(self):
        assert smpte_to_sample_offset("00:00:01:00", "30", SAMPLE_RATE) == 48000

    def test_one_minute_30fps(self):
        assert smpte_to_sample_offset("00:01:00:00", "30", SAMPLE_RATE) == 48000 * 60

    def test_one_hour_30fps(self):
        assert smpte_to_sample_offset("01:00:00:00", "30", SAMPLE_RATE) == 48000 * 3600

    def test_24fps(self):
        # 1 frame at 24fps / 48000 Hz = 2000 samples
        assert smpte_to_sample_offset("00:00:00:01", "24", SAMPLE_RATE) == 2000

    def test_25fps(self):
        # 1 frame at 25fps / 48000 Hz = 1920 samples
        assert smpte_to_sample_offset("00:00:00:01", "25", SAMPLE_RATE) == 1920

    def test_23976_rational_arithmetic(self):
        # 23.976 = 24000/1001; 1 frame ≈ 2002.08 samples
        # Must use rational arithmetic — float would accumulate error
        offset = smpte_to_sample_offset("00:00:00:01", "23.976", SAMPLE_RATE)
        # At 24 nominal fps, 1 frame = 48000 * 1001/24000 = 48000/23.976... ≈ 2002
        assert offset == 2002  # floor of 48000 * 1001/24000

    def test_23976_one_hour_no_drift(self):
        # Verify rational arithmetic doesn't drift over 1 hour
        offset = smpte_to_sample_offset("01:00:00:00", "23.976", SAMPLE_RATE)
        expected = int(3600 * 24 * SAMPLE_RATE * 1001 / 24000)
        assert offset == expected

    def test_2997nd_one_second(self):
        # 29.97 ND uses nominal 30fps labeling: 1 second = 30 timecode frames
        # actual sample offset = 30 * 48000 * 1001/30000 = 48048
        offset = smpte_to_sample_offset("00:00:01:00", "29.97", SAMPLE_RATE)
        assert offset == 48048  # NOT 46446 (which would result from fps=29)

    def test_2997nd_round_trip(self):
        # smpte → samples → smpte must round-trip cleanly for 29.97 ND
        tc = "01:23:45:15"
        offset = smpte_to_sample_offset(tc, "29.97", SAMPLE_RATE)
        recovered = sample_offset_to_smpte(offset, "29.97", SAMPLE_RATE)
        assert recovered == tc


# ---------------------------------------------------------------------------
# Drop-frame edge cases (29.97 DF) — the critical correctness tests
# ---------------------------------------------------------------------------

class TestDropFrame:
    """
    29.97 DF drops frame numbers 0 and 1 at the start of every minute,
    except every 10th minute (00, 10, 20, 30, 40, 50).

    These edge cases are where off-by-one errors occur. Each test verifies
    that the sample offset jumps correctly across the drop boundary.
    """

    def test_zero(self):
        assert smpte_to_sample_offset("00:00:00:00", "29.97df", SAMPLE_RATE) == 0

    def test_before_first_drop(self):
        # Drop-frame skips label numbers :00 and :01 at non-tenth minutes.
        # 00:00:59:29 and 00:01:00:02 are consecutive actual frames (label jump of 3,
        # but only 1 real frame elapses). The sample offset delta should be ~1 frame.
        # One frame at 29.97 = 48000 * 1001/30000 ≈ 1601.6 samples.
        offset_before = smpte_to_sample_offset("00:00:59:29", "29.97df", SAMPLE_RATE)
        offset_at = smpte_to_sample_offset("00:01:00:02", "29.97df", SAMPLE_RATE)
        from fractions import Fraction
        one_frame = int(Fraction(SAMPLE_RATE * 1001, 30000))  # 1601
        assert abs((offset_at - offset_before) - one_frame) <= 2

    def test_at_10_minute_no_drop(self):
        # At every 10th minute there is NO drop — frames :00 and :01 exist.
        # Consecutive frames should be ~1 frame apart in samples.
        offset_before = smpte_to_sample_offset("00:09:59:29", "29.97df", SAMPLE_RATE)
        offset_at_00 = smpte_to_sample_offset("00:10:00:00", "29.97df", SAMPLE_RATE)
        offset_at_01 = smpte_to_sample_offset("00:10:00:01", "29.97df", SAMPLE_RATE)
        from fractions import Fraction
        one_frame = int(Fraction(SAMPLE_RATE * 1001, 30000))  # 1601
        assert abs((offset_at_00 - offset_before) - one_frame) <= 2
        assert abs((offset_at_01 - offset_at_00) - one_frame) <= 2

    def test_at_one_hour_known_frame_count(self):
        # 1 hour at 29.97 DF = exactly 107892 frames (by SMPTE spec)
        offset = smpte_to_sample_offset("01:00:00:00", "29.97df", SAMPLE_RATE)
        expected_frames = 107892
        # sample_offset = frames / (30000/1001) * 48000
        from fractions import Fraction
        expected = int(Fraction(expected_frames * SAMPLE_RATE) / Fraction(30000, 1001))
        assert offset == expected

    def test_drop_is_two_frames_wide(self):
        # The gap between 00:00:59:29 and 00:01:00:02 should be exactly 2 frames
        # (frames 00:01:00:00 and 00:01:00:01 are dropped).
        # This is verified indirectly: the sample offset jump from :29 to :02
        # should equal 3 frame-periods (29→30≡02 with 2 dropped = 3 nominal frames).
        offset_a = smpte_to_sample_offset("00:00:59:28", "29.97df", SAMPLE_RATE)
        offset_b = smpte_to_sample_offset("00:00:59:29", "29.97df", SAMPLE_RATE)
        offset_c = smpte_to_sample_offset("00:01:00:02", "29.97df", SAMPLE_RATE)
        # Each nominal frame ≈ 1601.6 samples at 48kHz/29.97; the gap a→b and b→c
        # should each be one frame apart.
        one_frame = round(SAMPLE_RATE * 1001 / 30000)
        assert abs((offset_b - offset_a) - one_frame) <= 1   # one frame step
        assert abs((offset_c - offset_b) - one_frame) <= 1   # one frame step (drop absorbed)


# ---------------------------------------------------------------------------
# Round-trip: smpte → samples → smpte
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize("tc,fps", [
        ("00:00:00:00", "30"),
        ("01:23:45:15", "30"),
        ("00:00:01:00", "25"),
        ("02:00:00:00", "24"),
    ])
    def test_round_trip(self, tc, fps):
        offset = smpte_to_sample_offset(tc, fps, SAMPLE_RATE)
        recovered = sample_offset_to_smpte(offset, fps, SAMPLE_RATE)
        assert recovered == tc


# ---------------------------------------------------------------------------
# Unsupported frame rate
# ---------------------------------------------------------------------------

def test_unsupported_frame_rate():
    with pytest.raises(ValueError, match="Unsupported frame rate"):
        smpte_to_sample_offset("00:00:01:00", "60", SAMPLE_RATE)
