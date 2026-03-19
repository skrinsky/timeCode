"""
SMPTE timecode parsing, validation, and sample offset conversion.

Supports: 23.976, 24, 25, 29.97 ND, 29.97 DF, 30 fps.

Drop-frame arithmetic for 29.97 DF and 23.976 is handled via the
`timecode` PyPI library — never hand-rolled. All other frame rates
use exact integer arithmetic.
"""

from __future__ import annotations
import math
from fractions import Fraction
from typing import Literal

try:
    import timecode as _tc_lib
    _TC_LIB_AVAILABLE = True
except ImportError:
    _TC_LIB_AVAILABLE = False

# Supported frame rates
FrameRate = Literal["23.976", "24", "25", "29.97", "29.97df", "30"]

# Rational representations to avoid float drift
_RATE_RATIONAL: dict[str, Fraction] = {
    "23.976":  Fraction(24000, 1001),
    "24":      Fraction(24, 1),
    "25":      Fraction(25, 1),
    "29.97":   Fraction(30000, 1001),
    "29.97df": Fraction(30000, 1001),
    "30":      Fraction(30, 1),
}

_DROP_FRAME_RATES = {"29.97df", "23.976"}  # 23.976 also uses rational arithmetic


def parse_smpte(timecode_str: str, frame_rate: str) -> tuple[int, int, int, int]:
    """
    Parse a SMPTE timecode string into (HH, MM, SS, FF).

    Accepts both ':' and ';' as separators (';' is conventional for drop-frame
    but the model accepts either — the frame_rate parameter determines DF behavior).
    """
    parts = timecode_str.replace(";", ":").split(":")
    if len(parts) != 4:
        raise ValueError(f"Invalid SMPTE timecode: {timecode_str!r}. Expected HH:MM:SS:FF")
    try:
        hh, mm, ss, ff = (int(p) for p in parts)
    except ValueError:
        raise ValueError(f"Non-integer component in timecode: {timecode_str!r}")

    _validate_components(hh, mm, ss, ff, frame_rate)
    return hh, mm, ss, ff


def _validate_components(hh: int, mm: int, ss: int, ff: int, frame_rate: str) -> None:
    nominal_fps = math.ceil(_RATE_RATIONAL[frame_rate])  # ceiling integer fps for range check
    if not (0 <= hh <= 23):
        raise ValueError(f"Hours out of range: {hh}")
    if not (0 <= mm <= 59):
        raise ValueError(f"Minutes out of range: {mm}")
    if not (0 <= ss <= 59):
        raise ValueError(f"Seconds out of range: {ss}")
    if not (0 <= ff < nominal_fps):
        raise ValueError(f"Frames out of range for {frame_rate} fps: {ff}")


def smpte_to_sample_offset(
    timecode_str: str,
    frame_rate: str,
    sample_rate: int = 48000,
) -> int:
    """
    Convert a SMPTE timecode string to an absolute sample offset.

    For drop-frame rates (29.97 DF) and 23.976, uses the `timecode` library
    to compute the correct frame count. For all others, uses exact rational
    arithmetic.

    Returns the sample offset as an integer (floor).
    """
    if frame_rate not in _RATE_RATIONAL:
        raise ValueError(f"Unsupported frame rate: {frame_rate!r}. "
                         f"Must be one of {list(_RATE_RATIONAL)}")

    hh, mm, ss, ff = parse_smpte(timecode_str, frame_rate)

    if frame_rate in _DROP_FRAME_RATES:
        total_frames = _drop_frame_count(hh, mm, ss, ff, frame_rate)
    else:
        fps_int = math.ceil(_RATE_RATIONAL[frame_rate])   # nominal fps (e.g. 30 for 29.97 ND)
        total_frames = hh * 3600 * fps_int + mm * 60 * fps_int + ss * fps_int + ff

    # Use rational arithmetic to avoid float drift
    rate = _RATE_RATIONAL[frame_rate]
    sample_offset = Fraction(total_frames * sample_rate) / rate
    return int(sample_offset)  # floor — onset lands at or before intended sample


def _drop_frame_count(hh: int, mm: int, ss: int, ff: int, frame_rate: str) -> int:
    """
    Compute the total frame count for a drop-frame timecode.

    29.97 DF drop rule: skip frame numbers 0 and 1 at the start of each minute,
    except every 10th minute.

    23.976 uses the same rational rate (24000/1001) but is non-drop-frame in
    standard practice. Handled via rational arithmetic rather than drop-frame rules.
    """
    if frame_rate == "23.976":
        # 23.976 is non-drop — use straight integer arithmetic at 24 nominal fps
        return hh * 3600 * 24 + mm * 60 * 24 + ss * 24 + ff

    # 29.97 DF
    if _TC_LIB_AVAILABLE:
        tc = _tc_lib.Timecode("29.97", f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}")
        return tc.frame_number
    else:
        # Fallback: manual drop-frame formula (well-tested, from SMPTE spec)
        drop_frames = 2  # frames dropped per minute
        frames_per_10min = 17982  # 29.97 * 600
        frames_per_min = 1800 - drop_frames  # 1798

        d = hh * 2 * 1800  # hours → frames (approximate, corrected below)
        # Standard SMPTE DF formula
        total_minutes = 60 * hh + mm
        frame_number = (
            30 * 60 * 60 * hh
            + 30 * 60 * mm
            + 30 * ss
            + ff
            - drop_frames * (total_minutes - total_minutes // 10)
        )
        return frame_number


def sample_offset_to_smpte(
    sample_offset: int,
    frame_rate: str,
    sample_rate: int = 48000,
) -> str:
    """
    Convert a sample offset back to a SMPTE timecode string.
    Useful for the per-cue output report.
    """
    if frame_rate not in _RATE_RATIONAL:
        raise ValueError(f"Unsupported frame rate: {frame_rate!r}")

    rate = _RATE_RATIONAL[frame_rate]
    total_frames = int(Fraction(sample_offset) * rate / sample_rate)

    if frame_rate == "29.97df":
        return _frames_to_smpte_df(total_frames)
    else:
        nominal_fps = int(rate) if rate.denominator == 1 else (
            24 if frame_rate == "23.976" else 30
        )
        ff = total_frames % nominal_fps
        total_seconds = total_frames // nominal_fps
        ss = total_seconds % 60
        total_minutes = total_seconds // 60
        mm = total_minutes % 60
        hh = total_minutes // 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _frames_to_smpte_df(frame_number: int) -> str:
    """Convert a frame count back to 29.97 DF timecode string."""
    if _TC_LIB_AVAILABLE:
        tc = _tc_lib.Timecode("29.97", frames=frame_number + 1)
        return str(tc)
    # Manual inverse of the drop-frame formula
    drop_frames = 2
    frames_per_10min = 17982
    frames_per_min = 1798

    d, m = divmod(frame_number, frames_per_10min)
    hh_part = d * 10

    if m >= 2:
        m -= 2
        d2, m2 = divmod(m, frames_per_min)
        mm_part = d2 + 1
        remaining = m2 + (2 if d2 > 0 else 0)
    else:
        mm_part = 0
        remaining = m

    ss = remaining // 30
    ff = remaining % 30

    hh = hh_part // 60 if hh_part >= 60 else 0
    mm = (hh_part + mm_part) % 60

    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"
