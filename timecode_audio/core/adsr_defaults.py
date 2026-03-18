"""
ADSR default lookup tables.

Resolution hierarchy (per field, highest priority first):
    1. Explicit value in CueEvent
    2. Articulation term parsed from text prompt (this file)
    3. Instrument-class default parsed from text prompt (this file)
    4. Global default (this file)

All timing values in milliseconds. S is dimensionless (0.0–1.0).
Stored as plain dicts — easy to audit and extend.
"""

from __future__ import annotations
from typing import Optional

# ---------------------------------------------------------------------------
# Articulation term overrides
# Only fields listed are overridden — others fall through to instrument class.
# ---------------------------------------------------------------------------

ARTICULATION_DEFAULTS: dict[str, dict[str, Optional[float]]] = {
    "staccato": {
        "R": 30.0,         # very short release; A/D/S inherit from instrument
    },
    "staccatissimo": {
        "R": 10.0,
    },
    "legato": {
        # legato implies long duration — no ADSR override, handled in duration inference
    },
    "marcato": {
        "A": None,         # fast attack, inherits instrument class
        "velocity": 0.9,
    },
    "tenuto": {
        "S": 0.9,          # held at high sustain
    },
    "pizzicato": {
        "A": 5.0,
        "D": 60.0,
        "S": 0.0,
        "R": 150.0,
    },
    "sforzando": {
        "A": 2.0,
        "velocity": 1.0,
    },
    "sfz": {               # alias
        "A": 2.0,
        "velocity": 1.0,
    },
    "sffz": {
        "A": 1.0,
        "velocity": 1.0,
    },
    "tremolo": {
        # duration_ms = one repetition note; flagged separately
        # No direct ADSR override — handled in adsr_inferer
    },
    "con sordino": {
        # Timbral only — passes to text encoder, no ADSR implication
    },
    "sordino": {
        # Same
    },
    "arco": {
        # Bowing direction — timbral only
    },
    "sul ponticello": {
        # Timbral only
    },
    "col legno": {
        # Timbral only
    },
}

# ---------------------------------------------------------------------------
# Instrument-class defaults
# ---------------------------------------------------------------------------

INSTRUMENT_DEFAULTS: dict[str, dict[str, float]] = {
    "piano": {
        "A": 5.0,
        "D": 200.0,
        "S": 0.4,
        "R": 500.0,
        "velocity": 0.8,
        "duration_ms": 500.0,
    },
    "strings": {
        "A": 80.0,
        "D": 100.0,
        "S": 0.8,
        "R": 400.0,
        "velocity": 0.7,
        "duration_ms": 1000.0,
    },
    "violin": {
        "A": 60.0,
        "D": 80.0,
        "S": 0.8,
        "R": 300.0,
        "velocity": 0.7,
        "duration_ms": 800.0,
    },
    "cello": {
        "A": 80.0,
        "D": 100.0,
        "S": 0.85,
        "R": 500.0,
        "velocity": 0.7,
        "duration_ms": 1000.0,
    },
    "brass": {
        "A": 30.0,
        "D": 80.0,
        "S": 0.9,
        "R": 200.0,
        "velocity": 0.8,
        "duration_ms": 800.0,
    },
    "trumpet": {
        "A": 20.0,
        "D": 60.0,
        "S": 0.9,
        "R": 150.0,
        "velocity": 0.8,
        "duration_ms": 600.0,
    },
    "horn": {
        "A": 40.0,
        "D": 80.0,
        "S": 0.9,
        "R": 250.0,
        "velocity": 0.75,
        "duration_ms": 800.0,
    },
    "woodwind": {
        "A": 20.0,
        "D": 60.0,
        "S": 0.85,
        "R": 150.0,
        "velocity": 0.7,
        "duration_ms": 600.0,
    },
    "flute": {
        "A": 15.0,
        "D": 50.0,
        "S": 0.85,
        "R": 120.0,
        "velocity": 0.65,
        "duration_ms": 500.0,
    },
    "plucked": {
        "A": 5.0,
        "D": 100.0,
        "S": 0.1,
        "R": 200.0,
        "velocity": 0.75,
        "duration_ms": 400.0,
    },
    "guitar": {
        "A": 5.0,
        "D": 120.0,
        "S": 0.15,
        "R": 300.0,
        "velocity": 0.75,
        "duration_ms": 500.0,
    },
    "harp": {
        "A": 5.0,
        "D": 150.0,
        "S": 0.05,
        "R": 400.0,
        "velocity": 0.7,
        "duration_ms": 600.0,
    },
    "pad": {
        "A": 400.0,
        "D": 200.0,
        "S": 0.9,
        "R": 800.0,
        "velocity": 0.6,
        "duration_ms": 2000.0,
    },
    "synth": {
        "A": 10.0,
        "D": 100.0,
        "S": 0.7,
        "R": 200.0,
        "velocity": 0.8,
        "duration_ms": 500.0,
    },
    "organ": {
        "A": 5.0,
        "D": 0.0,
        "S": 1.0,
        "R": 50.0,
        "velocity": 0.8,
        "duration_ms": 800.0,
    },
    "snare": {
        "A": 2.0,
        "D": 80.0,
        "S": 0.0,
        "R": 50.0,
        "velocity": 0.9,
        "duration_ms": 200.0,
    },
    "kick": {
        "A": 1.0,
        "D": 120.0,
        "S": 0.0,
        "R": 80.0,
        "velocity": 1.0,
        "duration_ms": 250.0,
    },
    "hihat": {
        "A": 1.0,
        "D": 40.0,
        "S": 0.0,
        "R": 20.0,
        "velocity": 0.7,
        "duration_ms": 80.0,
    },
    "cymbal": {
        "A": 5.0,
        "D": 500.0,
        "S": 0.1,
        "R": 1000.0,
        "velocity": 0.8,
        "duration_ms": 1500.0,
    },
    "tom": {
        "A": 2.0,
        "D": 100.0,
        "S": 0.0,
        "R": 60.0,
        "velocity": 0.85,
        "duration_ms": 200.0,
    },
    "percussion": {
        "A": 2.0,
        "D": 100.0,
        "S": 0.0,
        "R": 80.0,
        "velocity": 0.8,
        "duration_ms": 200.0,
    },
    "foley": {
        "A": 5.0,
        "D": 150.0,
        "S": 0.2,
        "R": 200.0,
        "velocity": 0.7,
        "duration_ms": 400.0,
    },
    "sfx": {
        "A": 5.0,
        "D": 150.0,
        "S": 0.2,
        "R": 200.0,
        "velocity": 0.7,
        "duration_ms": 400.0,
    },
    "voice": {
        "A": 50.0,
        "D": 80.0,
        "S": 0.85,
        "R": 200.0,
        "velocity": 0.7,
        "duration_ms": 600.0,
    },
    "choir": {
        "A": 80.0,
        "D": 100.0,
        "S": 0.9,
        "R": 400.0,
        "velocity": 0.7,
        "duration_ms": 1000.0,
    },
}

# Aliases: map surface forms to canonical instrument-class keys
INSTRUMENT_ALIASES: dict[str, str] = {
    "string":     "strings",
    "pizz":       "plucked",    # "pizz" is shorthand for pizzicato strings
    "bass guitar": "guitar",
    "electric guitar": "guitar",
    "acoustic guitar": "guitar",
    "hi-hat":     "hihat",
    "hi hat":     "hihat",
    "bass drum":  "kick",
    "kick drum":  "kick",
    "snare drum": "snare",
    "sound effect": "sfx",
    "sound effects": "sfx",
    "fx":         "sfx",
    "synth pad":  "pad",
    "synthesizer": "synth",
}

# ---------------------------------------------------------------------------
# Global default (last resort)
# ---------------------------------------------------------------------------

GLOBAL_DEFAULT: dict[str, float] = {
    "A": 10.0,
    "D": 100.0,
    "S": 0.5,
    "R": 300.0,
    "velocity": 1.0,
    "duration_ms": 500.0,
}


def get_articulation_override(text: str) -> dict[str, Optional[float]]:
    """
    Scan text for known articulation terms and return their ADSR overrides.
    If multiple articulation terms are found, later ones in ARTICULATION_DEFAULTS
    take precedence (order of definition).
    """
    text_lower = text.lower()
    result: dict[str, Optional[float]] = {}
    for term, overrides in ARTICULATION_DEFAULTS.items():
        if term in text_lower:
            result.update(overrides)
    return result


def get_instrument_defaults(text: str) -> dict[str, float]:
    """
    Scan text for known instrument-class keywords and return defaults.
    Returns the most specific match found (checked in order of INSTRUMENT_DEFAULTS).
    Falls back to GLOBAL_DEFAULT if nothing matches.
    """
    text_lower = text.lower()

    # Check aliases first
    for alias, canonical in INSTRUMENT_ALIASES.items():
        if alias in text_lower and canonical in INSTRUMENT_DEFAULTS:
            return INSTRUMENT_DEFAULTS[canonical]

    # Check canonical names
    for name, defaults in INSTRUMENT_DEFAULTS.items():
        if name in text_lower:
            return defaults

    return GLOBAL_DEFAULT
