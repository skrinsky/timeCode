"""
BPM mode resolver: converts BAR.BEAT.TICK positions and musical duration
strings to absolute milliseconds, then to SMPTE timecode.

Constant BPM only (v1). Variable tempo maps are a v2 extension.
"""

from __future__ import annotations
import re
from fractions import Fraction
from dataclasses import dataclass, field
from typing import Optional

from .timecode import sample_offset_to_smpte


@dataclass
class BPMSession:
    """Session-level tempo settings for BPM mode."""
    bpm: float
    time_signature: str = "4/4"      # e.g. "4/4", "3/4", "6/8"
    ticks_per_beat: int = 480         # standard MIDI resolution
    session_start_ms: float = 0.0     # where bar 1 beat 1 lands (ms from file start)
    frame_rate: str = "30"            # used when converting back to SMPTE

    def __post_init__(self) -> None:
        if self.bpm <= 0:
            raise ValueError(f"BPM must be positive, got {self.bpm}")
        self._beats_per_bar, self._beat_unit = _parse_time_signature(self.time_signature)

    @property
    def beats_per_bar(self) -> int:
        return self._beats_per_bar

    @property
    def beat_unit(self) -> int:
        """Denominator of time signature (4 = quarter note, 8 = eighth note)."""
        return self._beat_unit

    @property
    def beat_duration_ms(self) -> float:
        """Duration of one beat (quarter note by default) in milliseconds."""
        return 60000.0 / self.bpm

    @property
    def ticks_per_bar(self) -> int:
        return self.ticks_per_beat * self.beats_per_bar

    @property
    def tick_duration_ms(self) -> float:
        return self.beat_duration_ms / self.ticks_per_beat


def _parse_time_signature(sig: str) -> tuple[int, int]:
    parts = sig.strip().split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid time signature: {sig!r}. Expected format '4/4'")
    return int(parts[0]), int(parts[1])


def position_to_ms(position: str, session: BPMSession) -> float:
    """
    Convert a BAR.BEAT.TICK position string to absolute milliseconds.

    Position format: "BAR.BEAT.TICK" where BAR and BEAT are 1-indexed.
    Examples:
        "1.1.0"    → bar 1, beat 1, tick 0 (downbeat)
        "2.3.240"  → bar 2, beat 3, tick 240 (half-beat into beat 3 at 480 tpb)
        "4.1.0"    → bar 4, beat 1 (start of bar 4)
    """
    parts = position.strip().split(".")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid position: {position!r}. Expected BAR.BEAT.TICK (e.g. '2.3.0')"
        )
    try:
        bar, beat, tick = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        raise ValueError(f"Non-integer component in position: {position!r}")

    if bar < 1:
        raise ValueError(f"Bar must be >= 1, got {bar}")
    if beat < 1 or beat > session.beats_per_bar:
        raise ValueError(
            f"Beat {beat} out of range for {session.time_signature} "
            f"(1–{session.beats_per_bar})"
        )
    if tick < 0 or tick >= session.ticks_per_beat:
        raise ValueError(
            f"Tick {tick} out of range (0–{session.ticks_per_beat - 1})"
        )

    total_ticks = (
        (bar - 1) * session.ticks_per_bar
        + (beat - 1) * session.ticks_per_beat
        + tick
    )
    absolute_ms = session.session_start_ms + total_ticks * session.tick_duration_ms
    return absolute_ms


def position_to_smpte(position: str, session: BPMSession, sample_rate: int = 48000) -> str:
    """Convert a BAR.BEAT.TICK position to a SMPTE timecode string."""
    absolute_ms = position_to_ms(position, session)
    sample_offset = int(absolute_ms * sample_rate / 1000.0)
    return sample_offset_to_smpte(sample_offset, session.frame_rate, sample_rate)


# ---------------------------------------------------------------------------
# Musical duration string → milliseconds
# ---------------------------------------------------------------------------

# Maps duration name to fraction of a whole note
_DURATION_NAMES: dict[str, Fraction] = {
    "whole":          Fraction(1, 1),
    "half":           Fraction(1, 2),
    "quarter":        Fraction(1, 4),
    "eighth":         Fraction(1, 8),
    "8th":            Fraction(1, 8),
    "sixteenth":      Fraction(1, 16),
    "16th":           Fraction(1, 16),
    "thirty-second":  Fraction(1, 32),
    "32nd":           Fraction(1, 32),
}

_DOTTED_RE = re.compile(
    r"^(dotted\s+)?(\d+\s+)?"          # optional "dotted" prefix and numeric bars
    r"(whole|half|quarter|eighth|8th|sixteenth|16th|thirty-second|32nd)"
    r"(\s+note)?$",
    re.IGNORECASE,
)

_BARS_RE = re.compile(r"^(\d+)\s+bar[s]?$", re.IGNORECASE)


def duration_to_ms(duration: str, session: BPMSession) -> float:
    """
    Convert a musical duration string to milliseconds at the current BPM.

    Supported formats:
        "quarter note"        → one quarter note
        "dotted quarter note" → 1.5× quarter note
        "8th note"            → one eighth note
        "2 bars"              → two full bars
        "1 bar"               → one bar
        "half"                → one half note

    The beat unit in the time signature is always a quarter note for BPM
    purposes (standard interpretation). A "quarter note" = one beat.
    """
    duration = duration.strip()

    # "N bar(s)"
    m = _BARS_RE.match(duration)
    if m:
        n_bars = int(m.group(1))
        bar_ms = session.beats_per_bar * session.beat_duration_ms
        return n_bars * bar_ms

    # Named note values (with optional "dotted" prefix)
    m = _DOTTED_RE.match(duration)
    if m:
        is_dotted = m.group(1) is not None
        note_name = m.group(3).lower()
        fraction_of_whole = _DURATION_NAMES[note_name]
        # A whole note = 4 quarter notes = 4 beats
        whole_note_ms = 4.0 * session.beat_duration_ms
        note_ms = float(fraction_of_whole) * whole_note_ms
        if is_dotted:
            note_ms *= 1.5
        return note_ms

    raise ValueError(
        f"Unrecognized duration: {duration!r}. "
        f"Examples: '2 bars', 'quarter note', 'dotted eighth note', '16th note'"
    )
