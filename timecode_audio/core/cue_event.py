"""
CueEvent schema: the atomic unit of a cue list.

All ADSR fields are optional (null = infer from text via adsr_inferer).
Accepts both SMPTE mode (timecode field) and BPM mode (position field).
"""

from __future__ import annotations
from typing import Optional, Union
from dataclasses import dataclass, field


@dataclass
class CueEvent:
    """
    One sound event on the timeline.

    Either `timecode` (SMPTE mode) or `position` (BPM mode) must be provided,
    but not both. The pipeline resolves whichever is present to a sample offset
    before generation.

    ADSR fields:
        A  — Attack time in milliseconds  (null = infer)
        D  — Decay time in milliseconds   (null = infer)
        S  — Sustain level 0.0–1.0        (null = infer)
        R  — Release time in milliseconds (null = infer)

    All timing values (A, D, R, duration_ms) are in milliseconds.
    S is dimensionless (gain fraction).
    """

    sound: str                          # free-text description, e.g. "staccato piano C4"
    timecode: Optional[str] = None      # SMPTE mode: "HH:MM:SS:FF"
    position: Optional[str] = None      # BPM mode:   "BAR.BEAT.TICK"

    A: Optional[float] = None           # Attack ms
    D: Optional[float] = None           # Decay ms
    S: Optional[float] = None           # Sustain level
    R: Optional[float] = None           # Release ms

    velocity: Optional[float] = None    # 0.0–1.0; null → 1.0
    duration_ms: Optional[float] = None # null → infer from next-cue gap or instrument default
    duration: Optional[str] = None      # BPM mode only: e.g. "2 bars", "quarter note"
    seed: Optional[int] = None          # reproducibility; null → derived deterministically

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        # Exactly one of timecode / position required
        if self.timecode is None and self.position is None:
            raise ValueError(
                f"CueEvent for {self.sound!r}: must provide either 'timecode' or 'position'"
            )
        if self.timecode is not None and self.position is not None:
            raise ValueError(
                f"CueEvent for {self.sound!r}: provide 'timecode' or 'position', not both"
            )

        # ADSR range checks (only when explicitly provided)
        if self.A is not None and self.A < 0:
            raise ValueError(f"A must be >= 0, got {self.A}")
        if self.D is not None and self.D < 0:
            raise ValueError(f"D must be >= 0, got {self.D}")
        if self.S is not None and not (0.0 <= self.S <= 1.0):
            raise ValueError(f"S must be in [0.0, 1.0], got {self.S}")
        if self.R is not None and self.R < 0:
            raise ValueError(f"R must be >= 0, got {self.R}")
        if self.velocity is not None and not (0.0 <= self.velocity <= 1.0):
            raise ValueError(f"velocity must be in [0.0, 1.0], got {self.velocity}")
        if self.duration_ms is not None and self.duration_ms <= 0:
            raise ValueError(f"duration_ms must be > 0, got {self.duration_ms}")

        if not self.sound.strip():
            raise ValueError("sound description must not be empty")

    @property
    def is_bpm_mode(self) -> bool:
        return self.position is not None

    @property
    def adsr_is_fully_specified(self) -> bool:
        return all(x is not None for x in (self.A, self.D, self.S, self.R))

    def adsr_null_fields(self) -> list[str]:
        """Return list of ADSR field names that are null and need inference."""
        return [name for name, val in [("A", self.A), ("D", self.D),
                                        ("S", self.S), ("R", self.R)]
                if val is None]

    def with_resolved_adsr(
        self,
        A: float,
        D: float,
        S: float,
        R: float,
        velocity: float,
        duration_ms: float,
        inferred_fields: Optional[list] = None,
    ) -> "ResolvedCueEvent":
        """Return a fully resolved version of this cue (no null fields)."""
        return ResolvedCueEvent(
            sound=self.sound,
            timecode=self.timecode,
            position=self.position,
            A=A, D=D, S=S, R=R,
            velocity=velocity,
            duration_ms=duration_ms,
            seed=self.seed,
            inferred_fields=inferred_fields or [],
        )


@dataclass
class ResolvedCueEvent:
    """
    A CueEvent with all fields populated (no nulls).
    Produced by the ADSR inferer after Stage 2.5 of the inference pipeline.
    """
    sound: str
    A: float
    D: float
    S: float
    R: float
    velocity: float
    duration_ms: float

    timecode: Optional[str] = None
    position: Optional[str] = None
    seed: Optional[int] = None

    # Set by the pipeline after timecode/position resolution
    sample_offset: Optional[int] = None

    # Set by the pipeline: which fields were inferred vs. explicit
    inferred_fields: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.A < 0 or self.D < 0 or self.R < 0:
            raise ValueError("A, D, R must be >= 0")
        if not (0.0 <= self.S <= 1.0):
            raise ValueError(f"S must be in [0.0, 1.0], got {self.S}")
        if not (0.0 <= self.velocity <= 1.0):
            raise ValueError(f"velocity must be in [0.0, 1.0], got {self.velocity}")
        if self.duration_ms <= 0:
            raise ValueError(f"duration_ms must be > 0, got {self.duration_ms}")
