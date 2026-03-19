"""
ADSR reconstruction accuracy metrics for Option A evaluation.

Metrics (per Section 7a of implementation plan):
    ATE  — Attack Time Error   |fitted_A - target_A| in ms   target: median < 15ms
    DTE  — Decay Time Error    |fitted_D - target_D| in ms   target: median < 30ms
    SLE  — Sustain Level Error |fitted_S - target_S|         target: median < 0.05
    RTE  — Release Time Error  |fitted_R - target_R| in ms   target: median < 50ms

Usage
-----
    from timecode_audio.eval.adsr_metrics import compute_batch_adsr_errors, print_adsr_report

    errors = compute_batch_adsr_errors(
        audio_batch,      # [N, n_samples] generated audio
        A_targets,        # [N] ground-truth attack ms
        D_targets, S_targets, R_targets,
        sample_rate=48000,
    )
    print_adsr_report(errors)
"""

from __future__ import annotations
import math
import numpy as np
import torch
from typing import NamedTuple


class ADSRErrors(NamedTuple):
    ATE: np.ndarray   # [N] ms
    DTE: np.ndarray   # [N] ms
    SLE: np.ndarray   # [N] dimensionless
    RTE: np.ndarray   # [N] ms


# ---------------------------------------------------------------------------
# Envelope extraction
# ---------------------------------------------------------------------------

def extract_amplitude_envelope(
    audio: np.ndarray,
    sample_rate: int,
    window_ms: float = 5.0,
) -> np.ndarray:
    """
    Compute RMS amplitude envelope with a short sliding window.

    Returns array of same length as audio (in samples).
    """
    hop = max(1, int(window_ms / 1000.0 * sample_rate))
    n = len(audio)
    envelope = np.zeros(n, dtype=np.float32)
    for i in range(0, n, hop):
        segment = audio[i : i + hop]
        rms = np.sqrt(np.mean(segment ** 2) + 1e-10)
        envelope[i : i + hop] = rms
    return envelope


# ---------------------------------------------------------------------------
# ADSR fitting
# ---------------------------------------------------------------------------

def fit_adsr_to_envelope(
    envelope: np.ndarray,
    sample_rate: int,
    note_duration_ms: float | None = None,
) -> dict[str, float] | None:
    """
    Fit a piecewise-linear ADSR model to an amplitude envelope.

    Returns dict with keys A_ms, D_ms, S_level, R_ms,
    or None if fitting fails (e.g., envelope is silent).

    Strategy:
        - Peak sample → Attack time
        - Time from peak to plateau → Decay time
        - Plateau level → Sustain level
        - Time from note-off to silence → Release time
    """
    if envelope.max() < 1e-5:
        return None   # silent clip

    env_norm = envelope / (envelope.max() + 1e-10)

    # Attack: time to first peak
    peak_idx = int(np.argmax(env_norm))
    A_ms = peak_idx / sample_rate * 1000.0

    # Sustain level: median of the middle third of the envelope
    # (avoids attack transient and release tail)
    n = len(env_norm)
    mid = env_norm[n // 3 : 2 * n // 3]
    S_level = float(np.median(mid)) if len(mid) > 0 else 0.0

    # Decay: time from peak to sustain level crossing
    decay_target = S_level + (1.0 - S_level) * 0.1   # 90% of the way to sustain
    D_ms = 0.0
    for i in range(peak_idx, min(peak_idx + int(sample_rate), n)):
        if env_norm[i] <= decay_target:
            D_ms = (i - peak_idx) / sample_rate * 1000.0
            break

    # Release: time from sustain end to silence
    # Find where envelope first drops below 5% of peak (after middle of clip)
    release_start = 2 * n // 3
    R_ms = 0.0
    for i in range(release_start, n):
        if env_norm[i] < 0.05:
            R_ms = (n - i) / sample_rate * 1000.0
            break

    return {"A_ms": A_ms, "D_ms": D_ms, "S_level": S_level, "R_ms": R_ms}


# ---------------------------------------------------------------------------
# Per-clip and batch metrics
# ---------------------------------------------------------------------------

def compute_adsr_error(
    audio: np.ndarray,
    target_A: float,
    target_D: float,
    target_S: float,
    target_R: float,
    sample_rate: int,
) -> dict[str, float] | None:
    """
    Compute ATE/DTE/SLE/RTE for a single generated clip.
    Returns None if envelope fitting fails.
    """
    envelope = extract_amplitude_envelope(audio, sample_rate)
    fitted = fit_adsr_to_envelope(envelope, sample_rate)
    if fitted is None:
        return None
    return {
        "ATE": abs(fitted["A_ms"]    - target_A),
        "DTE": abs(fitted["D_ms"]    - target_D),
        "SLE": abs(fitted["S_level"] - target_S),
        "RTE": abs(fitted["R_ms"]    - target_R),
    }


def compute_batch_adsr_errors(
    audio_batch: torch.Tensor,    # [N, n_samples]
    A_targets:   torch.Tensor,    # [N] ms
    D_targets:   torch.Tensor,    # [N] ms
    S_targets:   torch.Tensor,    # [N]
    R_targets:   torch.Tensor,    # [N] ms
    sample_rate: int = 48000,
) -> ADSRErrors:
    """
    Compute ADSR reconstruction errors for a batch of generated clips.

    Clips where fitting fails are excluded from the returned arrays
    (they should be counted separately as "fitting failed").
    """
    ATE_list, DTE_list, SLE_list, RTE_list = [], [], [], []

    audio_np = audio_batch.detach().cpu().numpy()
    A_np = A_targets.detach().cpu().numpy()
    D_np = D_targets.detach().cpu().numpy()
    S_np = S_targets.detach().cpu().numpy()
    R_np = R_targets.detach().cpu().numpy()

    for i in range(len(audio_np)):
        err = compute_adsr_error(
            audio_np[i], float(A_np[i]), float(D_np[i]),
            float(S_np[i]), float(R_np[i]), sample_rate
        )
        if err is None:
            continue
        ATE_list.append(err["ATE"])
        DTE_list.append(err["DTE"])
        SLE_list.append(err["SLE"])
        RTE_list.append(err["RTE"])

    return ADSRErrors(
        ATE=np.array(ATE_list, dtype=np.float32),
        DTE=np.array(DTE_list, dtype=np.float32),
        SLE=np.array(SLE_list, dtype=np.float32),
        RTE=np.array(RTE_list, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

TARGETS = {"ATE": 15.0, "DTE": 30.0, "SLE": 0.05, "RTE": 50.0}


def print_adsr_report(errors: ADSRErrors, label: str = "") -> None:
    """Print median errors and pass/fail against Option A targets."""
    header = f"ADSR Metrics{' — ' + label if label else ''}"
    print(f"\n{'='*50}")
    print(header)
    print(f"{'='*50}")
    print(f"  N clips evaluated: {len(errors.ATE)}")
    print(f"  {'Metric':<8} {'Median':>10}  {'Target':>10}  {'Pass?':>6}")
    print(f"  {'-'*40}")

    fields = [("ATE", errors.ATE, "ms"), ("DTE", errors.DTE, "ms"),
              ("SLE", errors.SLE, ""),   ("RTE", errors.RTE, "ms")]
    for name, vals, unit in fields:
        if len(vals) == 0:
            print(f"  {name:<8} {'N/A':>10}")
            continue
        median = float(np.median(vals))
        target = TARGETS[name]
        passed = "✓" if median < target else "✗"
        print(f"  {name:<8} {median:>8.2f}{unit:>3}  {target:>8.2f}{unit:>3}  {passed:>6}")
    print()
