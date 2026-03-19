"""
Synthetic training data generator.

Generates audio clips with exactly-known ADSR parameters using programmatic
synthesis. These are the ground-truth training pairs for Phase 1.

Synthesis uses simple waveforms (sine, sawtooth, square, FM) whose
parameters are fully controlled — no ambiguity about ADSR ground truth.

Constraint: A + D < note_duration_ms is enforced for every clip.
Interrupted-envelope cases (note_off during A or D) are included at ~5%.

Output: directory of .wav files + a metadata JSON file.
"""

from __future__ import annotations
import json
import math
import random
import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from timecode_audio.core.envelope import adsr_gate, validate_adsr_constraint

SAMPLE_RATE = 48000


@dataclass
class SyntheticClip:
    """Metadata for one generated clip."""
    audio_path: str
    text_prompt: str
    A_ms: float
    D_ms: float
    S_level: float
    R_ms: float
    pitch_midi: int
    instrument: str
    source: str = "synthetic"
    adsr_confidence: float = 1.0
    adsr_inferred: list = None
    note_duration: float = 0.0    # ms
    sample_rate: int = SAMPLE_RATE
    velocity: float = 1.0

    def __post_init__(self):
        if self.adsr_inferred is None:
            self.adsr_inferred = []


# ---------------------------------------------------------------------------
# ADSR coverage grid
# ---------------------------------------------------------------------------

A_VALUES_MS  = [0, 1, 5, 10, 20, 50, 100, 200, 500]
D_VALUES_MS  = [0, 5, 20, 50, 100, 200, 500]
S_VALUES     = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
R_VALUES_MS  = [0, 10, 50, 100, 200, 500, 1000, 2000]
DURATIONS_MS = [500, 1000, 2000, 4000]
VELOCITIES   = [0.25, 0.5, 0.75, 1.0]
MIDI_RANGE   = list(range(21, 109))   # A0 to C8

INSTRUMENTS = {
    "sine":      {"waveform": "sine",      "fm_ratio": None,  "fm_index": None},
    "sawtooth":  {"waveform": "sawtooth",  "fm_ratio": None,  "fm_index": None},
    "square":    {"waveform": "square",    "fm_ratio": None,  "fm_index": None},
    "fm_2op":    {"waveform": "fm",        "fm_ratio": 2.0,   "fm_index": 3.0},
    "fm_4op":    {"waveform": "fm",        "fm_ratio": 3.0,   "fm_index": 5.0},
}


def midi_to_hz(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12.0)


# ---------------------------------------------------------------------------
# Waveform generators (no envelope — envelope applied analytically)
# ---------------------------------------------------------------------------

def generate_sine(f0: float, n_samples: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(n_samples) / sr
    return np.sin(2 * math.pi * f0 * t).astype(np.float32)


def generate_sawtooth(f0: float, n_samples: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Bandlimited sawtooth via Fourier series (avoids aliasing)."""
    n_harmonics = min(int(sr / 2 / f0), 64)
    t = np.arange(n_samples) / sr
    signal = np.zeros(n_samples, dtype=np.float32)
    for k in range(1, n_harmonics + 1):
        signal += ((-1) ** (k + 1)) / k * np.sin(2 * math.pi * k * f0 * t)
    return (2 / math.pi * signal).astype(np.float32)


def generate_square(f0: float, n_samples: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Bandlimited square via Fourier series (odd harmonics only)."""
    n_harmonics = min(int(sr / 2 / f0), 64)
    t = np.arange(n_samples) / sr
    signal = np.zeros(n_samples, dtype=np.float32)
    for k in range(1, n_harmonics + 1, 2):  # odd only
        signal += (1 / k) * np.sin(2 * math.pi * k * f0 * t)
    return (4 / math.pi * signal).astype(np.float32)


def generate_fm(
    f0: float,
    n_samples: int,
    fm_ratio: float = 2.0,
    fm_index: float = 3.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """2-operator FM synthesis: carrier at f0, modulator at f0 * fm_ratio."""
    t = np.arange(n_samples) / sr
    modulator = fm_index * np.sin(2 * math.pi * f0 * fm_ratio * t)
    carrier   = np.sin(2 * math.pi * f0 * t + modulator)
    return carrier.astype(np.float32)


def generate_waveform(
    instrument: str,
    f0: float,
    n_samples: int,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    cfg = INSTRUMENTS[instrument]
    wf = cfg["waveform"]
    if wf == "sine":
        return generate_sine(f0, n_samples, sr)
    elif wf == "sawtooth":
        return generate_sawtooth(f0, n_samples, sr)
    elif wf == "square":
        return generate_square(f0, n_samples, sr)
    elif wf == "fm":
        return generate_fm(f0, n_samples, cfg["fm_ratio"], cfg["fm_index"], sr)
    raise ValueError(f"Unknown waveform: {wf}")


# ---------------------------------------------------------------------------
# Clip generation
# ---------------------------------------------------------------------------

def generate_clip(
    instrument: str,
    midi: int,
    A: float,
    D: float,
    S: float,
    R: float,
    note_duration_ms: float,
    velocity: float = 1.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate one audio clip: waveform × analytic ADSR gate × velocity.

    Total clip length = note_duration_ms + R (includes release tail).
    """
    total_ms = note_duration_ms + R
    n_samples = int(total_ms / 1000.0 * sr) + sr // 100  # +10ms pad

    f0 = midi_to_hz(midi)
    raw = generate_waveform(instrument, f0, n_samples, sr)

    t_ms = np.arange(n_samples, dtype=np.float32) / sr * 1000.0
    gate = adsr_gate(t_ms, A=A, D=D, S=S, R=R, note_duration_ms=note_duration_ms)

    clip = (raw * gate * velocity).astype(np.float32)
    return clip


def sample_adsr_valid(
    rng: random.Random,
    allow_interrupted: bool = False,
) -> tuple[float, float, float, float, float]:
    """
    Sample (A, D, S, R, note_duration) satisfying A + D < note_duration.

    Returns (A, D, S, R, note_duration) all in ms (S is dimensionless).
    """
    for _ in range(100):  # retry until constraint satisfied
        A   = rng.choice(A_VALUES_MS)
        D   = rng.choice(D_VALUES_MS)
        S   = rng.choice(S_VALUES)
        R   = rng.choice(R_VALUES_MS)
        dur = rng.choice(DURATIONS_MS)

        if allow_interrupted and rng.random() < 0.05:
            # Deliberately interrupted: A+D > dur (~5% of data)
            return float(A), float(D), float(S), float(R), float(dur)

        if validate_adsr_constraint(A, D, dur, strict=False):
            return float(A), float(D), float(S), float(R), float(dur)

    # Fallback: use safe values
    return 10.0, 50.0, 0.5, 200.0, 500.0


def make_text_prompt(instrument: str, midi: int) -> str:
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = midi // 12 - 1
    note  = note_names[midi % 12]
    return f"synthesizer {instrument} {note}{octave}"


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    output_dir: str,
    n_clips: int = 500_000,
    stage: str = "stage1",          # "stage1" = sine+sawtooth only
    seed: int = 42,
    sr: int = SAMPLE_RATE,
) -> None:
    """
    Generate n_clips synthetic training clips and save to output_dir.

    output_dir/
        audio/
            000000.wav
            ...
        metadata.jsonl   ← one JSON object per line
    """
    out = Path(output_dir)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    if stage == "stage1":
        instruments = ["sine", "sawtooth"]
    else:
        instruments = list(INSTRUMENTS.keys())

    metadata_path = out / "metadata.jsonl"
    generated = 0

    with open(metadata_path, "w") as f_meta:
        while generated < n_clips:
            instrument = rng.choice(instruments)
            midi       = rng.choice(MIDI_RANGE)
            velocity   = rng.choice(VELOCITIES)
            A, D, S, R, dur = sample_adsr_valid(rng, allow_interrupted=True)

            try:
                clip = generate_clip(instrument, midi, A, D, S, R, dur, velocity, sr)
            except Exception:
                continue

            # Normalize to prevent clipping
            peak = np.abs(clip).max()
            if peak > 0:
                clip = clip / peak * 0.9 * velocity

            filename = f"{generated:06d}.wav"
            audio_path = audio_dir / filename

            sf.write(str(audio_path), clip, sr, subtype="FLOAT")

            meta = SyntheticClip(
                audio_path=str(audio_path.relative_to(out)),
                text_prompt=make_text_prompt(instrument, midi),
                A_ms=A,
                D_ms=D,
                S_level=S,
                R_ms=R,
                pitch_midi=midi,
                instrument=instrument,
                note_duration=dur,
                velocity=velocity,
            )
            f_meta.write(json.dumps(asdict(meta)) + "\n")
            generated += 1

            if generated % 10_000 == 0:
                print(f"Generated {generated}/{n_clips} clips")

    print(f"Done. {generated} clips saved to {output_dir}")


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SyntheticDataset(torch.utils.data.Dataset):
    """
    Dataset that loads from a metadata.jsonl file generated by generate_dataset().
    """

    def __init__(
        self,
        data_dir: str,
        max_duration_ms: float = 6000.0,  # clip at 6s
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_ms / 1000.0 * sample_rate)

        meta_path = self.data_dir / "metadata.jsonl"
        self.records: list[dict] = []
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        audio_path = self.data_dir / rec["audio_path"]

        waveform, sr = torchaudio.load(str(audio_path))
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        waveform = waveform.squeeze(0)  # [n_samples]

        # Pad or trim
        if waveform.shape[0] < self.max_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.max_samples - waveform.shape[0])
            )
        else:
            waveform = waveform[:self.max_samples]

        return {
            "audio":          waveform,                        # [n_samples]
            "f0_hz":          torch.tensor(midi_to_hz(rec["pitch_midi"]), dtype=torch.float32),
            "A":              torch.tensor(rec["A_ms"],    dtype=torch.float32),
            "D":              torch.tensor(rec["D_ms"],    dtype=torch.float32),
            "S":              torch.tensor(rec["S_level"], dtype=torch.float32),
            "R":              torch.tensor(rec["R_ms"],    dtype=torch.float32),
            "velocity":       torch.tensor(rec["velocity"],   dtype=torch.float32),
            "note_duration":  torch.tensor(rec["note_duration"], dtype=torch.float32),
            "text_prompt":    rec["text_prompt"],
        }
