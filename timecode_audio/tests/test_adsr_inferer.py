"""
Unit tests for ADSR inference and resolution hierarchy.
"""

import pytest
from timecode_audio.core.cue_event import CueEvent
from timecode_audio.core.adsr_inferer import infer_adsr, infer_cue_list
from timecode_audio.core.adsr_defaults import GLOBAL_DEFAULT, INSTRUMENT_DEFAULTS


# ---------------------------------------------------------------------------
# Resolution hierarchy
# ---------------------------------------------------------------------------

class TestResolutionHierarchy:
    def test_explicit_wins(self):
        cue = CueEvent(sound="staccato piano C4", timecode="00:00:01:00",
                       A=999.0, R=888.0)
        resolved = infer_adsr(cue)
        assert resolved.A == 999.0
        assert resolved.R == 888.0

    def test_explicit_partial(self):
        # Only A explicit; D/S/R should be inferred
        cue = CueEvent(sound="piano C4", timecode="00:00:01:00", A=5.0)
        resolved = infer_adsr(cue)
        assert resolved.A == 5.0
        # D, S, R come from piano instrument class
        assert resolved.D == INSTRUMENT_DEFAULTS["piano"]["D"]
        assert resolved.S == INSTRUMENT_DEFAULTS["piano"]["S"]
        assert resolved.R == INSTRUMENT_DEFAULTS["piano"]["R"]

    def test_articulation_overrides_instrument(self):
        # staccato → R: 30ms, overrides piano default R: 500ms
        cue = CueEvent(sound="staccato piano C4", timecode="00:00:01:00")
        resolved = infer_adsr(cue)
        assert resolved.R == 30.0  # from staccato articulation

    def test_instrument_overrides_global(self):
        # snare: A=2, not global default A=10
        cue = CueEvent(sound="snare drum hit", timecode="00:00:01:00")
        resolved = infer_adsr(cue)
        assert resolved.A == INSTRUMENT_DEFAULTS["snare"]["A"]
        assert resolved.D == INSTRUMENT_DEFAULTS["snare"]["D"]

    def test_global_default_fallback(self):
        # Unknown instrument → global defaults
        cue = CueEvent(sound="mysterious sound", timecode="00:00:01:00")
        resolved = infer_adsr(cue)
        assert resolved.A == GLOBAL_DEFAULT["A"]
        assert resolved.D == GLOBAL_DEFAULT["D"]
        assert resolved.S == GLOBAL_DEFAULT["S"]
        assert resolved.R == GLOBAL_DEFAULT["R"]


# ---------------------------------------------------------------------------
# Articulation parsing
# ---------------------------------------------------------------------------

class TestArticulationParsing:
    def test_pizzicato(self):
        cue = CueEvent(sound="pizzicato strings", timecode="00:00:01:00")
        r = infer_adsr(cue)
        assert r.A == 5.0
        assert r.D == 60.0
        assert r.S == 0.0
        assert r.R == 150.0

    def test_sforzando(self):
        cue = CueEvent(sound="sforzando brass", timecode="00:00:01:00")
        r = infer_adsr(cue)
        assert r.A == 2.0
        assert r.velocity == 1.0

    def test_staccatissimo(self):
        cue = CueEvent(sound="staccatissimo violin", timecode="00:00:01:00")
        r = infer_adsr(cue)
        assert r.R == 10.0

    def test_con_sordino_no_adsr_effect(self):
        # con sordino is timbral only — ADSR should come from violin class
        cue = CueEvent(sound="violin con sordino", timecode="00:00:01:00")
        r = infer_adsr(cue)
        assert r.A == INSTRUMENT_DEFAULTS["violin"]["A"]

    def test_conflict_staccato_explicit_R(self):
        # User specified R=2000 explicitly even though sound is staccato
        cue = CueEvent(sound="staccato piano C4", timecode="00:00:01:00", R=2000.0)
        r = infer_adsr(cue)
        # Explicit wins over articulation
        assert r.R == 2000.0


# ---------------------------------------------------------------------------
# Inferred fields log
# ---------------------------------------------------------------------------

class TestInferredFieldsLog:
    def test_all_explicit_no_inferred(self):
        cue = CueEvent(sound="piano C4", timecode="00:00:01:00",
                       A=5.0, D=200.0, S=0.4, R=500.0)
        r = infer_adsr(cue)
        # velocity and duration_ms were not specified — those should be inferred
        assert "A" not in r.inferred_fields
        assert "D" not in r.inferred_fields
        assert "velocity" in r.inferred_fields

    def test_all_null_all_inferred(self):
        cue = CueEvent(sound="kick drum", timecode="00:00:01:00")
        r = infer_adsr(cue)
        assert "A" in r.inferred_fields
        assert "D" in r.inferred_fields
        assert "S" in r.inferred_fields
        assert "R" in r.inferred_fields


# ---------------------------------------------------------------------------
# Duration inference
# ---------------------------------------------------------------------------

class TestDurationInference:
    def test_explicit_duration(self):
        cue = CueEvent(sound="piano C4", timecode="00:00:01:00", duration_ms=1234.0)
        r = infer_adsr(cue)
        assert r.duration_ms == 1234.0

    def test_next_cue_gap(self):
        cue = CueEvent(sound="piano C4", timecode="00:00:01:00")
        # next cue is 600ms later, R will be inferred from piano (500ms)
        # gap = 600 - R = 600 - 500 = 100ms
        r = infer_adsr(cue, next_cue_gap_ms=600.0)
        assert r.duration_ms == pytest.approx(100.0, abs=1.0)

    def test_last_cue_uses_instrument_default(self):
        # No next cue → falls through to instrument default
        cue = CueEvent(sound="piano C4", timecode="00:00:01:00")
        r = infer_adsr(cue, next_cue_gap_ms=None)
        assert r.duration_ms == INSTRUMENT_DEFAULTS["piano"]["duration_ms"]

    def test_text_long(self):
        cue = CueEvent(sound="long pad chord", timecode="00:00:01:00")
        r = infer_adsr(cue, next_cue_gap_ms=None)
        assert r.duration_ms == 2000.0

    def test_text_short(self):
        cue = CueEvent(sound="short click", timecode="00:00:01:00")
        r = infer_adsr(cue, next_cue_gap_ms=None)
        assert r.duration_ms == 200.0


# ---------------------------------------------------------------------------
# infer_cue_list
# ---------------------------------------------------------------------------

class TestInferCueList:
    def test_list_resolution(self):
        cues = [
            CueEvent(sound="kick drum", timecode="00:00:00:00"),
            CueEvent(sound="snare drum", timecode="00:00:00:15"),
        ]
        onset_ms = [0.0, 500.0]
        resolved = infer_cue_list(cues, onset_ms)
        assert len(resolved) == 2
        assert resolved[0].A == INSTRUMENT_DEFAULTS["kick"]["A"]
        assert resolved[1].A == INSTRUMENT_DEFAULTS["snare"]["A"]

    def test_length_mismatch(self):
        cues = [CueEvent(sound="kick", timecode="00:00:00:00")]
        with pytest.raises(ValueError, match="same length"):
            infer_cue_list(cues, [0.0, 500.0])

    def test_last_cue_no_next(self):
        cues = [CueEvent(sound="piano note", timecode="00:00:01:00")]
        resolved = infer_cue_list(cues, [1000.0])
        # Last cue falls to instrument default for duration
        assert resolved[0].duration_ms == INSTRUMENT_DEFAULTS["piano"]["duration_ms"]
