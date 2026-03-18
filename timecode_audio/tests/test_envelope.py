"""
Unit tests for the analytic ADSR gate and stage_position functions.
"""

import numpy as np
import pytest
from timecode_audio.core.envelope import adsr_gate, stage_position, validate_adsr_constraint


def t(start, end, n=1000):
    return np.linspace(start, end, n, dtype=np.float32)


# ---------------------------------------------------------------------------
# adsr_gate correctness
# ---------------------------------------------------------------------------

class TestADSRGate:
    def test_basic_shape(self):
        # A=10, D=20, S=0.5, R=30, dur=50
        times = np.array([0, 5, 10, 20, 30, 50, 65, 80], dtype=np.float32)
        g = adsr_gate(times, A=10, D=20, S=0.5, R=30, note_duration_ms=50)
        # t=0: start of attack, g=0
        assert g[0] == pytest.approx(0.0, abs=1e-4)
        # t=5: halfway through attack, g=0.5
        assert g[1] == pytest.approx(0.5, abs=1e-4)
        # t=10: peak, g=1.0
        assert g[2] == pytest.approx(1.0, abs=1e-4)
        # t=20: halfway through decay (D=20 → at A+D/2=20), g = 1 - 0.5*(20-10)/20 = 0.75
        assert g[3] == pytest.approx(0.75, abs=1e-4)
        # t=30: sustain, g=0.5
        assert g[4] == pytest.approx(0.5, abs=1e-4)
        # t=50: note_off, release starts, g=0.5
        assert g[5] == pytest.approx(0.5, abs=1e-4)
        # t=65: halfway through release (R=30), g = 0.5 * (1 - 15/30) = 0.25
        assert g[6] == pytest.approx(0.25, abs=1e-4)
        # t=80: after release, g=0
        assert g[7] == pytest.approx(0.0, abs=1e-4)

    def test_zero_attack(self):
        # A=0: instant peak at t=0
        times = np.array([0, 5, 10, 50], dtype=np.float32)
        g = adsr_gate(times, A=0, D=10, S=0.5, R=20, note_duration_ms=50)
        # t=0: should be at peak (1.0) going into decay
        # D=10: at t=5 (halfway), g = 1 - 0.5*(5/10) = 0.75
        assert g[1] == pytest.approx(0.75, abs=1e-4)
        # sustain
        assert g[2] == pytest.approx(0.5, abs=1e-4)

    def test_zero_decay(self):
        # D=0: instant transition from peak to sustain
        times = np.array([5, 10, 15], dtype=np.float32)
        g = adsr_gate(times, A=10, D=0, S=0.7, R=20, note_duration_ms=30)
        assert g[1] == pytest.approx(0.7, abs=1e-4)  # sustain immediately after A

    def test_zero_release(self):
        # R=0: instant silence at note_off
        times = np.array([49.9, 50.0, 50.1], dtype=np.float32)
        g = adsr_gate(times, A=5, D=10, S=0.5, R=0, note_duration_ms=50)
        assert g[0] == pytest.approx(0.5, abs=1e-4)  # sustain
        assert g[2] == pytest.approx(0.0, abs=1e-4)  # silence immediately after note_off

    def test_zero_sustain(self):
        # S=0: no sustain (percussive)
        times = np.array([30, 40, 50], dtype=np.float32)
        g = adsr_gate(times, A=5, D=20, S=0.0, R=50, note_duration_ms=100)
        assert g[0] == pytest.approx(0.0, abs=1e-4)  # at sustain, g=S=0

    def test_all_zero(self):
        # A=D=R=0, S=anything: g is S during note, 0 at/after note_off
        times = np.array([0, 10, 49.9, 50.0, 50.1], dtype=np.float32)
        g = adsr_gate(times, A=0, D=0, S=0.6, R=0, note_duration_ms=50)
        assert g[0] == pytest.approx(0.6, abs=1e-4)   # t=0: in note
        assert g[1] == pytest.approx(0.6, abs=1e-4)   # t=10: in note
        assert g[2] == pytest.approx(0.6, abs=1e-4)   # t=49.9: still in note
        assert g[3] == pytest.approx(0.0, abs=1e-4)   # t=50.0: at note_off, R=0 → instant silence
        assert g[4] == pytest.approx(0.0, abs=1e-4)   # t=50.1: after note_off

    def test_interrupted_during_attack(self):
        # note_off during attack (at t=5, A=20)
        times = np.array([3, 5, 6, 20], dtype=np.float32)
        g = adsr_gate(times, A=20, D=10, S=0.5, R=15, note_duration_ms=5)
        # t=3: still in attack, g=3/20=0.15
        assert g[0] == pytest.approx(0.15, abs=1e-4)
        # t=5: note_off — g(note_off) = 5/20 = 0.25, release starts
        # t=6: 1ms into release (R=15), g = 0.25 * (1 - 1/15)
        assert g[2] == pytest.approx(0.25 * (1 - 1/15), abs=1e-4)

    def test_output_bounded(self):
        # g should always be in [0, 1]
        times = t(-10, 200, 2000)
        g = adsr_gate(times, A=10, D=30, S=0.6, R=40, note_duration_ms=100)
        assert np.all(g >= 0.0)
        assert np.all(g <= 1.0)

    def test_non_physical_slow_attack_piano(self):
        # Non-physical combination should just work
        times = t(0, 3000, 1000)
        g = adsr_gate(times, A=2000, D=100, S=0.9, R=500, note_duration_ms=2500)
        assert np.all(g >= 0.0)
        assert np.all(g <= 1.0)
        # Peak at A=2000ms
        idx_peak = np.argmin(np.abs(times - 2000))
        assert g[idx_peak] == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# stage_position
# ---------------------------------------------------------------------------

class TestStagePosition:
    def test_basic_shape(self):
        # A=10, D=20, note_duration=100
        times = np.array([0, 5, 10, 20, 30, 50, 110], dtype=np.float32)
        alpha = stage_position(times, A=10, D=20, note_duration_ms=100)
        # t=0: start of attack, alpha=0
        assert alpha[0] == pytest.approx(0.0, abs=1e-4)
        # t=5: halfway through attack, alpha=0.5
        assert alpha[1] == pytest.approx(0.5, abs=1e-4)
        # t=10: peak, alpha=1.0
        assert alpha[2] == pytest.approx(1.0, abs=1e-4)
        # t=20: halfway through decay, alpha=0.5
        assert alpha[3] == pytest.approx(0.5, abs=1e-4)
        # t=30: sustain, alpha=0
        assert alpha[4] == pytest.approx(0.0, abs=1e-4)
        # t=50: sustain, alpha=0
        assert alpha[5] == pytest.approx(0.0, abs=1e-4)
        # t=110: release, alpha stays 0 (no revert to attack spectrum)
        assert alpha[6] == pytest.approx(0.0, abs=1e-4)

    def test_release_stays_at_sustain_spectrum(self):
        # Key invariant: alpha never rises during release
        times = t(0, 200, 500)
        alpha = stage_position(times, A=10, D=20, note_duration_ms=80)
        release_mask = times >= 80
        assert np.all(alpha[release_mask] == pytest.approx(0.0, abs=1e-4))

    def test_zero_attack(self):
        # A=0: no attack ramp, alpha goes straight to decay
        times = np.array([0, 5, 10, 20], dtype=np.float32)
        alpha = stage_position(times, A=0, D=10, note_duration_ms=50)
        # With A=0, the "peak" is instantaneous — decay starts at t=0
        # t=5: halfway through D=10, alpha=0.5
        assert alpha[1] == pytest.approx(0.5, abs=1e-4)

    def test_bounded(self):
        times = t(0, 200, 1000)
        alpha = stage_position(times, A=20, D=30, note_duration_ms=100)
        assert np.all(alpha >= 0.0)
        assert np.all(alpha <= 1.0)


# ---------------------------------------------------------------------------
# validate_adsr_constraint
# ---------------------------------------------------------------------------

class TestValidateADSRConstraint:
    def test_valid(self):
        assert validate_adsr_constraint(10, 20, 100) is True

    def test_boundary_fail(self):
        with pytest.raises(ValueError, match="sustain phase would never be reached"):
            validate_adsr_constraint(50, 60, 100, strict=True)

    def test_equal_fails(self):
        # A+D == note_duration: sustain never reached
        with pytest.raises(ValueError):
            validate_adsr_constraint(50, 50, 100, strict=True)

    def test_non_strict_returns_false(self):
        result = validate_adsr_constraint(50, 60, 100, strict=False)
        assert result is False

    def test_zero_adsr_valid(self):
        assert validate_adsr_constraint(0, 0, 100) is True
