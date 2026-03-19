"""
ADSR Estimator: predicts A, D, S, R from an audio clip.

Architecture:
    audio → log-mel spectrogram → 4-block CNN → global avg pool → 4 heads

Trained on synthetic clips where ADSR ground truth is known exactly.
Applied to real audio (NSynth, etc.) to generate pseudo-labels for Stage 3.

Output convention:
    A, D, R : predicted in log(x+1) space internally; returned in ms
    S       : predicted via sigmoid; returned in [0, 1]
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

SAMPLE_RATE = 48000
N_MELS      = 128


class _ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU, stride 2 in both dimensions."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ADSREstimator(nn.Module):
    """
    Predicts (A, D, S, R) from a raw audio waveform.

    Internal representation:
        A, D, R → log(ms + 1)   (unbounded regression)
        S       → sigmoid        (bounded [0, 1])

    Use `predict()` to get back physical units.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        # Log-mel frontend (pure PyTorch, no audio backend)
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels,
            f_min=20.0,
            f_max=float(sample_rate // 2),
        )

        # CNN encoder: 4 blocks, each halving spatial dims
        # Input: [B, 1, n_mels, T]  (T ≈ 563 for 6s clips)
        # After 4 blocks: [B, hidden_dim, n_mels/16, T/16]
        self.encoder = nn.Sequential(
            _ConvBlock(1,           hidden_dim // 8),   # → 32 ch
            _ConvBlock(hidden_dim // 8, hidden_dim // 4),   # → 64 ch
            _ConvBlock(hidden_dim // 4, hidden_dim // 2),   # → 128 ch
            _ConvBlock(hidden_dim // 2, hidden_dim),         # → 256 ch
        )

        # Global average pool → [B, hidden_dim]
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Output heads
        self.head_A = nn.Linear(hidden_dim, 1)   # log(A_ms + 1)
        self.head_D = nn.Linear(hidden_dim, 1)   # log(D_ms + 1)
        self.head_S = nn.Linear(hidden_dim, 1)   # sigmoid → [0, 1]
        self.head_R = nn.Linear(hidden_dim, 1)   # log(R_ms + 1)

    def forward(
        self,
        audio: torch.Tensor,   # [batch, n_samples]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (log_A, log_D, S, log_R), each [batch].
        log_A = log(A_ms + 1), etc.
        S is in [0, 1] via sigmoid.
        """
        # Mel spectrogram
        mel = self.mel(audio)                          # [B, n_mels, T]
        mel = torch.log(mel + 1e-7)                    # log-mel
        mel = mel.unsqueeze(1)                          # [B, 1, n_mels, T]

        # Encode
        h = self.encoder(mel)                           # [B, hidden_dim, H, W]
        h = self.pool(h).squeeze(-1).squeeze(-1)        # [B, hidden_dim]

        log_A = self.head_A(h).squeeze(-1)              # [B]
        log_D = self.head_D(h).squeeze(-1)              # [B]
        S     = torch.sigmoid(self.head_S(h)).squeeze(-1)  # [B]
        log_R = self.head_R(h).squeeze(-1)              # [B]

        return log_A, log_D, S, log_R

    @torch.no_grad()
    def predict(
        self,
        audio: torch.Tensor,   # [batch, n_samples] or [n_samples]
    ) -> dict[str, torch.Tensor]:
        """
        Returns predicted ADSR in physical units:
            A, D, R in ms  (float tensors [batch])
            S in [0, 1]    (float tensor  [batch])
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        log_A, log_D, S, log_R = self(audio)
        return {
            "A": torch.exp(log_A) - 1.0,
            "D": torch.exp(log_D) - 1.0,
            "S": S,
            "R": torch.exp(log_R) - 1.0,
        }


def adsr_estimator_loss(
    log_A_pred: torch.Tensor,
    log_D_pred: torch.Tensor,
    S_pred:     torch.Tensor,
    log_R_pred: torch.Tensor,
    A_ms:       torch.Tensor,   # ground-truth ms
    D_ms:       torch.Tensor,
    S_level:    torch.Tensor,   # ground-truth [0, 1]
    R_ms:       torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss in log space for A/D/R (wide dynamic range),
    MSE in linear space for S (already bounded).

    Returns scalar loss.
    """
    log_A_true = torch.log(A_ms   + 1.0)
    log_D_true = torch.log(D_ms   + 1.0)
    log_R_true = torch.log(R_ms   + 1.0)

    loss = (
        F.mse_loss(log_A_pred, log_A_true)
        + F.mse_loss(log_D_pred, log_D_true)
        + F.mse_loss(S_pred,     S_level)
        + F.mse_loss(log_R_pred, log_R_true)
    )
    return loss
