"""
Analytic piecewise-linear ADSR gate g(t) and stage_position(t).

No ML — these are pure math functions computed from the four ADSR scalars.
Both functions are called per synthesis frame (Option A) and serve as the
ground-truth envelope shape for evaluation.

Design choices (deliberate v1 simplifications):
    - Piecewise-linear ramps (not exponential). Swap to exponential in this
      file only if linear sounds mechanical during evaluation.
    - A=0, D=0, R=0 are valid — handled without division by zero.
    - Interrupted envelope (note_off before A+D completes) is supported.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def adsr_gate(
    t_ms: NDArray[np.float32],
    A: float,
    D: float,
    S: float,
    R: float,
    note_duration_ms: float,
) -> NDArray[np.float32]:
    """
    Compute the analytic piecewise-linear ADSR amplitude gate g(t).

    Parameters
    ----------
    t_ms : array of time values in milliseconds (relative to note-on = 0)
    A    : Attack time in ms  (>= 0)
    D    : Decay time in ms   (>= 0)
    S    : Sustain level      (0.0–1.0)
    R    : Release time in ms (>= 0)
    note_duration_ms : time from note-on to note-off in ms

    Returns
    -------
    g    : gate values in [0.0, 1.0], same shape as t_ms

    Stage boundaries:
        note-on    = 0
        peak       = A
        sustain    = A + D
        note-off   = note_duration_ms
        silence    = note_duration_ms + R

    Interrupted envelope: if note_off occurs during Attack or Decay,
    Release begins from whatever g(note_off) was at that point.

    Edge cases (zero-duration stages):
        A=0  → g(0) = 1.0  (instant peak, no ramp)
        D=0  → no decay phase; sustain level equals 1.0 implicitly at peak
               (S is still honored — if D=0 and S<1, gain jumps to S at note-on+0)
        R=0  → g drops to 0 immediately at note-off
    """
    t = np.asarray(t_ms, dtype=np.float32)
    g = np.zeros_like(t)

    note_off = float(note_duration_ms)

    # --- Compute g(note_off) for interrupted-envelope release ---
    g_at_note_off = _g_before_release(note_off, A, D, S)

    # --- Attack ---
    if A > 0:
        mask = (t >= 0) & (t < A) & (t < note_off)
        g[mask] = t[mask] / A
    else:
        # Instant attack: peak at t=0
        pass  # handled by decay/sustain sections below

    # --- Decay ---
    if D > 0:
        mask = (t >= A) & (t < A + D) & (t < note_off)
        g[mask] = 1.0 - (1.0 - S) * (t[mask] - A) / D
    # If D=0, no decay phase — fall through to sustain at 1.0 (A=0) or wherever ramp lands

    # --- Sustain (or instant-peak when A=0 and D=0) ---
    mask = (t >= max(A, 0) + max(D, 0)) & (t < note_off)
    if A == 0 and D == 0:
        # Instant peak, instant settle to S
        mask_full = (t >= 0) & (t < note_off)
        g[mask_full] = S
    else:
        g[mask] = S

    # Correction: at t < A with A=0, gate should be at peak (1.0)
    # This is handled by the decay block covering t>=0 when A==0
    if A == 0:
        if D > 0:
            mask = (t >= 0) & (t < D) & (t < note_off)
            g[mask] = 1.0 - (1.0 - S) * t[mask] / D
        # sustain covered above

    # --- Release ---
    if R > 0:
        mask = (t >= note_off) & (t < note_off + R)
        g[mask] = g_at_note_off * (1.0 - (t[mask] - note_off) / R)
    # R=0: g remains 0 after note_off (already initialized to 0)

    return np.clip(g, 0.0, 1.0)


def _g_before_release(t: float, A: float, D: float, S: float) -> float:
    """Compute g(t) for t <= note_off, used to find the release start level."""
    if A == 0 and D == 0:
        return S
    if A == 0:
        if t < D:
            return 1.0 - (1.0 - S) * t / D
        return S
    if t < A:
        return t / A
    if D > 0 and t < A + D:
        return 1.0 - (1.0 - S) * (t - A) / D
    return S


def stage_position(
    t_ms: NDArray[np.float32],
    A: float,
    D: float,
    note_duration_ms: float,
) -> NDArray[np.float32]:
    """
    Compute the spectral interpolation schedule α(t) ∈ [0.0, 1.0].

    α(t) = 1.0 at attack peak (t = A)
    α(t) = 0.0 during sustain and release

    During Attack:  linearly rises 0 → 1
    During Decay:   linearly falls 1 → 0
    During Sustain: holds at 0.0
    During Release: holds at 0.0  (real instruments don't revert to attack
                    spectrum on release — piano release is darker than attack)

    This drives frame_harmonics(t) = α(t)*params_peak + (1-α(t))*params_sustain
    in the DDSP synthesizer.
    """
    t = np.asarray(t_ms, dtype=np.float32)
    alpha = np.zeros_like(t)

    # Attack ramp: 0 → 1
    if A > 0:
        mask = (t >= 0) & (t < A)
        alpha[mask] = t[mask] / A
    else:
        # Instant attack: alpha spikes to 1 at t=0, then immediately decays
        # Represented as a single-frame peak — handled by decay block below
        pass

    # Decay ramp: 1 → 0
    if D > 0:
        mask = (t >= A) & (t < A + D)
        alpha[mask] = 1.0 - (t[mask] - A) / D
    elif A == 0:
        # A=0, D=0: no spectral evolution — stays at sustain spectrum
        pass

    # Sustain + Release: alpha remains 0 (already initialized)

    return np.clip(alpha, 0.0, 1.0)


def envelope_duration_ms(A: float, D: float, R: float, note_duration_ms: float) -> float:
    """Total duration of the sound in ms (note_duration + release tail)."""
    return note_duration_ms + R


def validate_adsr_constraint(
    A: float, D: float, note_duration_ms: float, strict: bool = True
) -> bool:
    """
    Check the A + D < note_duration constraint.

    When strict=True (training data generation), raise if violated.
    When strict=False (inference), return False so caller can log a warning.
    """
    if A + D >= note_duration_ms:
        if strict:
            raise ValueError(
                f"A ({A}ms) + D ({D}ms) = {A+D}ms >= note_duration ({note_duration_ms}ms). "
                f"The sustain phase would never be reached. "
                f"Reduce A or D, or increase note_duration_ms."
            )
        return False
    return True
