"""
Envelope adapter for Stable Audio Open.

Implements the Sketch2Sound pattern (ICASSP 2025):
  ADSR params → adsr_gate_samples() → envelope curve [T_samples]
              → frame RMS at VAE latent rate (2048x = ~21.5 Hz)
              → Linear(1, 64) → add to noisy latents [B, 64, T_latent]

The adapter is the only trainable component. Stable Audio Open is frozen.
Zero-initialized so training starts from the pretrained model's behavior.

VAE latent shape: (B, 64, T_latent)  where T_latent = audio_samples / 2048
Latent frame rate: 44100 / 2048 ≈ 21.53 Hz
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

VAE_DOWNSAMPLE = 2048          # Stable Audio Open VAE temporal downsampling factor
LATENT_CHANNELS = 64           # VAE latent channel count
SAMPLE_RATE = 44100            # Stable Audio Open native sample rate
LATENT_FRAME_RATE = SAMPLE_RATE / VAE_DOWNSAMPLE  # ≈ 21.53 Hz


# ---------------------------------------------------------------------------
# Envelope extraction
# ---------------------------------------------------------------------------

def audio_to_envelope(
    audio: torch.Tensor,   # [n_samples] or [2, n_samples] stereo
    frame_size: int = VAE_DOWNSAMPLE,
) -> torch.Tensor:
    """
    Compute RMS envelope at VAE latent frame rate.

    Frames align exactly with the VAE's downsampling (frame_size = 2048),
    so the returned envelope has exactly T_latent frames — no resampling needed.

    Returns: [T_latent] float tensor, values in [0, 1]
    """
    if audio.dim() == 2:
        audio = audio.mean(0)   # stereo → mono for envelope

    audio_np = audio.float().cpu().numpy()
    n = len(audio_np)
    n_frames = max(1, n // frame_size)
    envelope = np.empty(n_frames, dtype=np.float32)

    for i in range(n_frames):
        frame = audio_np[i * frame_size : (i + 1) * frame_size]
        envelope[i] = math.sqrt(float(np.mean(frame ** 2)) + 1e-10)

    # Normalize to [0, 1]
    peak = float(envelope.max())
    if peak > 1e-8:
        envelope /= peak

    return torch.from_numpy(envelope)


def apply_median_filter(
    envelope: torch.Tensor,   # [T_latent]
    window_size: Optional[int] = None,
    max_window: int = 25,
) -> torch.Tensor:
    """
    Apply median filter with random window size (Sketch2Sound augmentation).

    Teaches the model to follow imprecise/sketched curves, preventing
    overfitting to exact envelope shapes during training.

    window_size: fixed size (for inference); None = random 1–max_window (training)
    """
    if window_size is None:
        window_size = int(torch.randint(1, max_window + 1, (1,)).item())

    if window_size <= 1:
        return envelope

    # Use avg_pool1d as a fast approximation (true median is slow for long sequences)
    # For training augmentation, average pooling is sufficient
    x = envelope.unsqueeze(0).unsqueeze(0)   # [1, 1, T]
    pad = window_size // 2
    x = F.avg_pool1d(x, kernel_size=window_size, stride=1, padding=pad)
    return x.squeeze(0).squeeze(0)[:len(envelope)]


# ---------------------------------------------------------------------------
# Adapter module
# ---------------------------------------------------------------------------

class EnvelopeAdapter(nn.Module):
    """
    Single-layer envelope adapter for Stable Audio Open DiT.

    Maps a 1D amplitude envelope (one value per VAE latent frame) to a
    [B, 64, T_latent] tensor that is added element-wise to the noisy latents
    before the DiT forward pass.

    Zero-initialized: at the start of training the adapter adds nothing,
    so the model inherits the pretrained backbone's behavior exactly.
    """

    def __init__(self, latent_channels: int = LATENT_CHANNELS) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        # Single linear layer: projects scalar envelope value → latent channel dim
        self.proj = nn.Linear(1, latent_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        envelope: torch.Tensor,   # [B, T_latent] values in [0, 1]
    ) -> torch.Tensor:
        """
        Returns: [B, latent_channels, T_latent] — add to noisy latents before DiT.
        """
        x = envelope.unsqueeze(-1)          # [B, T_latent, 1]
        x = self.proj(x)                    # [B, T_latent, latent_channels]
        return x.transpose(1, 2)            # [B, latent_channels, T_latent]

    def forward_with_cfg(
        self,
        envelope: torch.Tensor,   # [B, T_latent]
        guidance_scale: float = 3.0,
    ) -> torch.Tensor:
        """
        Compute adapter embedding for CFG at inference.

        Returns the guided embedding:
            embed_null + guidance_scale * (embed_full - embed_null)

        embed_null = adapter(zeros), embed_full = adapter(envelope)
        """
        embed_full = self.forward(envelope)
        embed_null = self.forward(torch.zeros_like(envelope))
        return embed_null + guidance_scale * (embed_full - embed_null)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "EnvelopeAdapter":
        adapter = cls()
        adapter.load_state_dict(torch.load(path, map_location=device))
        return adapter
