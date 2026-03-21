"""
Dataset for adapter training.

Each record in metadata.jsonl:
    {
        "audio_path":   "/abs/path/to/clip.wav",
        "text_prompt":  "piano note C4",
        "seconds_total": 4.0
    }

The envelope is computed on-the-fly from the audio at VAE latent frame rate
(one RMS value per 2048 samples = ~21.53 Hz). No pre-extracted envelopes saved.

Returns:
    audio         : [2, n_samples] stereo float32 at 44100 Hz
    envelope      : [T_latent] float32 in [0, 1]
    text_prompt   : str
    seconds_total : float
"""

from __future__ import annotations
import json
import random
import soundfile as sf
import numpy as np
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset

from timecode_audio.model.stable_audio_adapter import (
    audio_to_envelope,
    apply_median_filter,
    VAE_DOWNSAMPLE,
    SAMPLE_RATE,
)

MAX_DURATION_S  = 47.0      # Stable Audio Open max clip length
MAX_SAMPLES     = int(MAX_DURATION_S * SAMPLE_RATE)


class AdapterDataset(Dataset):
    """
    Loads audio clips and computes envelopes on-the-fly for adapter training.

    augment=True:  applies random median filter (Sketch2Sound augmentation).
                   Use during training.
    augment=False: no augmentation. Use for validation/eval.
    """

    def __init__(
        self,
        data_dir: str,
        augment: bool = True,
        max_duration_s: float = MAX_DURATION_S,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.augment  = augment
        self.max_samples = int(max_duration_s * SAMPLE_RATE)

        meta_path = self.data_dir / "metadata.jsonl"
        self.records: list[dict] = []
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        print(f"AdapterDataset: {len(self.records)} clips from {data_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        audio_path   = rec["audio_path"]
        text_prompt  = rec["text_prompt"]
        seconds_total = float(rec["seconds_total"])

        # --- Load audio ---
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)   # [channels, n_samples]

        # Ensure stereo
        if waveform.shape[0] == 1:
            waveform = waveform.expand(2, -1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        # Resample if needed
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        # Pad or trim to max_samples
        n = waveform.shape[1]
        if n < self.max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - n))
        else:
            waveform = waveform[:, :self.max_samples]

        # --- Compute envelope from mono mix at latent frame rate ---
        mono = waveform.mean(0)   # [n_samples]
        envelope = audio_to_envelope(mono, frame_size=VAE_DOWNSAMPLE)

        # --- Augment: random median filter (training only) ---
        if self.augment:
            envelope = apply_median_filter(envelope, window_size=None, max_window=25)

        return {
            "audio":        waveform,           # [2, max_samples]
            "envelope":     envelope,           # [T_latent]
            "text_prompt":  text_prompt,
            "seconds_total": seconds_total,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad envelopes to the same length within a batch."""
    max_env_len = max(b["envelope"].shape[0] for b in batch)

    audios, envelopes, texts, durations = [], [], [], []
    for b in batch:
        audios.append(b["audio"])
        env = b["envelope"]
        # Pad envelope to max length
        pad = max_env_len - env.shape[0]
        if pad > 0:
            env = torch.nn.functional.pad(env, (0, pad))
        envelopes.append(env)
        texts.append(b["text_prompt"])
        durations.append(b["seconds_total"])

    return {
        "audio":        torch.stack(audios),        # [B, 2, n_samples]
        "envelope":     torch.stack(envelopes),     # [B, T_latent]
        "text_prompt":  texts,
        "seconds_total": torch.tensor(durations, dtype=torch.float32),
    }
