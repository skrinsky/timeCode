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
    """Conv2d → BatchNorm → ReLU → Dropout2d, stride 2 in both dimensions."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
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

    @torch.no_grad()
    def predict_mc(
        self,
        audio: torch.Tensor,   # [batch, n_samples] or [n_samples]
        n_passes: int = 20,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Monte Carlo dropout inference: run n_passes forward passes with dropout
        active to estimate prediction uncertainty.

        Returns
        -------
        means     : dict {A, D, S, R} — mean predictions in physical units
        variances : dict {A, D, S, R} — variance across passes (uncertainty proxy)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        was_training = self.training
        self.train()   # keep Dropout2d active

        results_A, results_D, results_S, results_R = [], [], [], []
        for _ in range(n_passes):
            log_A, log_D, S, log_R = self(audio)
            results_A.append(torch.exp(log_A) - 1.0)
            results_D.append(torch.exp(log_D) - 1.0)
            results_S.append(S)
            results_R.append(torch.exp(log_R) - 1.0)

        if not was_training:
            self.eval()

        stack = lambda lst: torch.stack(lst, dim=0)   # [n_passes, batch]
        means = {
            "A": stack(results_A).mean(0),
            "D": stack(results_D).mean(0),
            "S": stack(results_S).mean(0),
            "R": stack(results_R).mean(0),
        }
        variances = {
            "A": stack(results_A).var(0),
            "D": stack(results_D).var(0),
            "S": stack(results_S).var(0),
            "R": stack(results_R).var(0),
        }
        return means, variances

    def predict_with_confidence(
        self,
        audio: torch.Tensor,
        platt_scaler: "PlattScaler | None" = None,
        n_passes: int = 20,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns (predictions_dict, confidence_score [batch]).
        confidence_score is calibrated if platt_scaler is provided,
        otherwise raw (1 / (1 + mean_variance)).
        """
        means, variances = self.predict_mc(audio, n_passes=n_passes)
        mean_var = sum(variances.values()) / 4.0   # scalar per batch item
        if platt_scaler is not None:
            confidence = platt_scaler.transform(mean_var)
        else:
            confidence = 1.0 / (1.0 + mean_var)   # simple uncalibrated score
        return means, confidence


class PlattScaler:
    """
    Calibrates raw MC-dropout variance to a confidence score in [0, 1].

    Fits logistic regression: P(correct) = sigmoid(a * log(var) + b)
    on a held-out synthetic set where ground-truth ADSR is known.

    Usage
    -----
    # Fit (once, on held-out synthetic validation set):
    scaler = PlattScaler()
    scaler.fit(variances, is_correct)

    # Apply at pseudo-label time:
    confidence = scaler.transform(variances)
    """

    def __init__(self) -> None:
        self.a: float = -1.0   # default: higher variance → lower confidence
        self.b: float = 0.0
        self._fitted: bool = False

    def fit(
        self,
        variances: torch.Tensor,   # [N] — mean variance per clip
        is_correct: torch.Tensor,  # [N] bool — True if prediction within threshold
    ) -> None:
        """Fit a · log(var) + b via gradient descent (no sklearn required)."""
        import numpy as np
        x = torch.log(variances.float() + 1e-8)   # [N]
        y = is_correct.float()                     # [N]

        # Gradient descent on binary cross-entropy
        a = torch.tensor(-1.0, requires_grad=True)
        b = torch.tensor(0.0,  requires_grad=True)
        opt = torch.optim.LBFGS([a, b], max_iter=200, tolerance_grad=1e-7)

        def closure():
            opt.zero_grad()
            logits = a * x + b
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            return loss

        opt.step(closure)
        self.a = float(a.item())
        self.b = float(b.item())
        self._fitted = True

    def transform(self, variances: torch.Tensor) -> torch.Tensor:
        """Returns calibrated confidence scores in [0, 1]."""
        log_var = torch.log(variances.float() + 1e-8)
        return torch.sigmoid(self.a * log_var + self.b)

    def save(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump({"a": self.a, "b": self.b}, f)

    @classmethod
    def load(cls, path: str) -> "PlattScaler":
        import json
        scaler = cls()
        with open(path) as f:
            d = json.load(f)
        scaler.a = d["a"]
        scaler.b = d["b"]
        scaler._fitted = True
        return scaler


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
