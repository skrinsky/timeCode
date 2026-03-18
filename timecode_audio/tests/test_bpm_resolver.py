"""
Unit tests for BPM mode resolver.
"""

import pytest
from timecode_audio.core.bpm_resolver import BPMSession, position_to_ms, duration_to_ms


# ---------------------------------------------------------------------------
# BPMSession construction
# ---------------------------------------------------------------------------

class TestBPMSession:
    def test_basic(self):
        s = BPMSession(bpm=120.0)
        assert s.beats_per_bar == 4
        assert s.beat_unit == 4
        assert s.beat_duration_ms == pytest.approx(500.0)
        assert s.ticks_per_bar == 1920

    def test_three_four(self):
        s = BPMSession(bpm=120.0, time_signature="3/4")
        assert s.beats_per_bar == 3
        assert s.ticks_per_bar == 1440

    def test_six_eight(self):
        s = BPMSession(bpm=120.0, time_signature="6/8")
        assert s.beats_per_bar == 6

    def test_invalid_bpm(self):
        with pytest.raises(ValueError, match="BPM must be positive"):
            BPMSession(bpm=0.0)

    def test_invalid_time_sig(self):
        with pytest.raises(ValueError):
            BPMSession(bpm=120.0, time_signature="4")


# ---------------------------------------------------------------------------
# position_to_ms
# ---------------------------------------------------------------------------

class TestPositionToMs:
    def setup_method(self):
        self.s = BPMSession(bpm=120.0)  # beat = 500ms, bar = 2000ms

    def test_downbeat(self):
        assert position_to_ms("1.1.0", self.s) == pytest.approx(0.0)

    def test_bar_two(self):
        assert position_to_ms("2.1.0", self.s) == pytest.approx(2000.0)

    def test_beat_two(self):
        assert position_to_ms("1.2.0", self.s) == pytest.approx(500.0)

    def test_beat_three(self):
        assert position_to_ms("1.3.0", self.s) == pytest.approx(1000.0)

    def test_half_beat_swing(self):
        # Tick 240 = half of 480 ticks/beat = half a beat = 250ms
        assert position_to_ms("1.1.240", self.s) == pytest.approx(250.0)

    def test_session_start_offset(self):
        s = BPMSession(bpm=120.0, session_start_ms=1000.0)
        assert position_to_ms("1.1.0", s) == pytest.approx(1000.0)
        assert position_to_ms("1.2.0", s) == pytest.approx(1500.0)

    def test_bar_out_of_range_beat(self):
        with pytest.raises(ValueError, match="Beat 5 out of range"):
            position_to_ms("1.5.0", self.s)

    def test_negative_tick(self):
        with pytest.raises(ValueError, match="Tick"):
            position_to_ms("1.1.-1", self.s)

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid position"):
            position_to_ms("1.1", self.s)

    def test_bar_zero_invalid(self):
        with pytest.raises(ValueError, match="Bar must be >= 1"):
            position_to_ms("0.1.0", self.s)

    def test_100bpm(self):
        s = BPMSession(bpm=100.0)  # beat = 600ms
        assert position_to_ms("1.2.0", s) == pytest.approx(600.0)
        assert position_to_ms("2.1.0", s) == pytest.approx(2400.0)


# ---------------------------------------------------------------------------
# duration_to_ms
# ---------------------------------------------------------------------------

class TestDurationToMs:
    def setup_method(self):
        self.s = BPMSession(bpm=120.0)
        # At 120 BPM: beat = 500ms, whole note = 2000ms

    def test_quarter_note(self):
        assert duration_to_ms("quarter note", self.s) == pytest.approx(500.0)

    def test_half_note(self):
        assert duration_to_ms("half note", self.s) == pytest.approx(1000.0)

    def test_whole_note(self):
        assert duration_to_ms("whole note", self.s) == pytest.approx(2000.0)

    def test_eighth_note(self):
        assert duration_to_ms("eighth note", self.s) == pytest.approx(250.0)

    def test_8th_note_alias(self):
        assert duration_to_ms("8th note", self.s) == pytest.approx(250.0)

    def test_sixteenth_note(self):
        assert duration_to_ms("sixteenth note", self.s) == pytest.approx(125.0)

    def test_16th_alias(self):
        assert duration_to_ms("16th note", self.s) == pytest.approx(125.0)

    def test_dotted_quarter(self):
        assert duration_to_ms("dotted quarter note", self.s) == pytest.approx(750.0)

    def test_dotted_eighth(self):
        assert duration_to_ms("dotted eighth note", self.s) == pytest.approx(375.0)

    def test_dotted_half(self):
        assert duration_to_ms("dotted half note", self.s) == pytest.approx(1500.0)

    def test_1_bar(self):
        assert duration_to_ms("1 bar", self.s) == pytest.approx(2000.0)

    def test_2_bars(self):
        assert duration_to_ms("2 bars", self.s) == pytest.approx(4000.0)

    def test_4_bars(self):
        assert duration_to_ms("4 bars", self.s) == pytest.approx(8000.0)

    def test_case_insensitive(self):
        assert duration_to_ms("Quarter Note", self.s) == pytest.approx(500.0)
        assert duration_to_ms("DOTTED HALF NOTE", self.s) == pytest.approx(1500.0)

    def test_without_note_word(self):
        # "quarter" alone (no "note")
        assert duration_to_ms("quarter", self.s) == pytest.approx(500.0)

    def test_unrecognized(self):
        with pytest.raises(ValueError, match="Unrecognized duration"):
            duration_to_ms("triplet quarter", self.s)

    def test_three_four_bar(self):
        s = BPMSession(bpm=120.0, time_signature="3/4")
        # 3/4 bar = 3 beats = 1500ms at 120 BPM
        assert duration_to_ms("1 bar", s) == pytest.approx(1500.0)

    def test_tempo_scaling(self):
        s = BPMSession(bpm=60.0)  # beat = 1000ms
        assert duration_to_ms("quarter note", s) == pytest.approx(1000.0)
        assert duration_to_ms("2 bars", s) == pytest.approx(8000.0)
