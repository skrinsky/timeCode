"""
ADSR inferer: resolves null ADSR fields in a CueEvent using the lookup tables
in adsr_defaults.py.

Resolution hierarchy (per field):
    1. Explicit value in CueEvent      — always wins
    2. Articulation term override      — parsed from sound description
    3. Instrument-class default        — parsed from sound description
    4. Global default                  — last resort

Produces a ResolvedCueEvent with no null fields. Logs which fields were
inferred vs. explicit for the per-cue output report.
"""

from __future__ import annotations
from typing import Optional
import logging

from .cue_event import CueEvent, ResolvedCueEvent
from .adsr_defaults import (
    get_articulation_override,
    get_instrument_defaults,
    GLOBAL_DEFAULT,
)

logger = logging.getLogger(__name__)


def infer_adsr(
    cue: CueEvent,
    next_cue_gap_ms: Optional[float] = None,
) -> ResolvedCueEvent:
    """
    Resolve all null ADSR fields and return a ResolvedCueEvent.

    Parameters
    ----------
    cue             : CueEvent with possibly null ADSR fields
    next_cue_gap_ms : time gap (ms) from this cue's onset to the next cue's
                      onset (i.e. next_onset_ms - this_onset_ms). Used for
                      duration inference. None if this is the last cue.

    Returns
    -------
    ResolvedCueEvent with all fields populated and inferred_fields log.
    """
    text = cue.sound
    inferred: list[str] = []

    # Step 1: get lookup layers
    articulation = get_articulation_override(text)
    instrument   = get_instrument_defaults(text)

    def resolve(
        field: str,
        explicit_value: Optional[float],
    ) -> float:
        if explicit_value is not None:
            return explicit_value

        inferred.append(field)

        # Articulation override (may be None for fields not specified by articulation)
        art_val = articulation.get(field)
        if art_val is not None:
            logger.debug("  %s: inferred from articulation → %s", field, art_val)
            return art_val

        # Instrument-class default
        inst_val = instrument.get(field)
        if inst_val is not None:
            logger.debug("  %s: inferred from instrument class → %s", field, inst_val)
            return inst_val

        # Global default
        global_val = GLOBAL_DEFAULT[field]
        logger.debug("  %s: inferred from global default → %s", field, global_val)
        return global_val

    A        = resolve("A",        cue.A)
    D        = resolve("D",        cue.D)
    S        = resolve("S",        cue.S)
    R        = resolve("R",        cue.R)
    velocity = resolve("velocity", cue.velocity)

    # Duration resolution
    duration_ms = _resolve_duration(cue, A, D, R, next_cue_gap_ms, instrument, inferred)

    if inferred:
        logger.debug("Cue %r — inferred fields: %s", text, inferred)

    return cue.with_resolved_adsr(
        A=A, D=D, S=S, R=R,
        velocity=velocity,
        duration_ms=duration_ms,
        inferred_fields=inferred,
    )


def _resolve_duration(
    cue: CueEvent,
    A: float,
    D: float,
    R: float,
    next_cue_gap_ms: Optional[float],
    instrument: dict[str, float],
    inferred: list[str],
) -> float:
    """
    Resolve note_duration_ms using the four-level hierarchy:
        1. Explicit duration_ms in CueEvent
        2. Next-cue gap heuristic (last cue skips this — no next cue)
        3. Text-inferred (short/long/sustained keywords)
        4. Instrument-class default
    """
    # Explicit
    if cue.duration_ms is not None:
        return cue.duration_ms

    inferred.append("duration_ms")

    # Duration = time-to-next-cue minus release so the tail ends just before
    # the next cue's onset (next_cue_gap_ms is relative: next_onset - this_onset).
    if next_cue_gap_ms is not None:
        gap = next_cue_gap_ms - R
        if gap > 0:
            return gap

    # Text-inferred keywords
    text_lower = cue.sound.lower()
    if any(w in text_lower for w in ("long", "sustained", "sustain", "hold")):
        return 2000.0
    if any(w in text_lower for w in ("short", "brief", "quick")):
        return 200.0
    if "staccato" in text_lower or "staccatissimo" in text_lower:
        return max(A + D + 10.0, 100.0)

    # Instrument-class default
    return instrument.get("duration_ms", GLOBAL_DEFAULT["duration_ms"])


def infer_cue_list(
    cues: list[CueEvent],
    onset_ms_list: list[float],
) -> list[ResolvedCueEvent]:
    """
    Resolve all cues in a list, passing next-cue onset for duration inference.

    Parameters
    ----------
    cues          : list of CueEvents (all modes accepted)
    onset_ms_list : absolute onset in ms for each cue (same length as cues)

    Returns
    -------
    List of ResolvedCueEvents in the same order.
    """
    if len(cues) != len(onset_ms_list):
        raise ValueError("cues and onset_ms_list must have the same length")

    resolved = []
    for i, (cue, onset_ms) in enumerate(zip(cues, onset_ms_list)):
        next_onset = onset_ms_list[i + 1] if i + 1 < len(onset_ms_list) else None
        next_onset_relative = (next_onset - onset_ms) if next_onset is not None else None
        resolved_cue = infer_adsr(cue, next_cue_gap_ms=next_onset_relative)
        resolved_cue.sample_offset = None  # set by pipeline after this stage
        resolved.append(resolved_cue)

    return resolved
