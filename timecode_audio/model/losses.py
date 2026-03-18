"""
Training losses.

Option A: Multi-scale spectral loss (MSS)
    L1 on STFT magnitudes at multiple FFT sizes.
    Captures timbral quality across frequency scales — low FFT sizes
    catch fast transients, high FFT sizes catch spectral resolution.

Option B: OT-CFM + confidence-weighted loss (Phase 6).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torchaudio.transforms as T


class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-scale spectral loss (MSS).

    For each FFT size n:
        loss_n = mean(|mag_pred - mag_target|)   (L1 on magnitude)

    Total loss = mean over all scales.

    FFT sizes chosen to span the range from transient resolution (64)
    to spectral resolution (2048) at 48kHz.
    """

    FFT_SIZES = [64, 128, 256, 512, 1024, 2048]

    def __init__(
        self,
        fft_sizes: list[int] = None,
        sample_rate: int = 48000,
        log_magnitude: bool = True,
    ) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes or self.FFT_SIZES
        self.sample_rate = sample_rate
        self.log_magnitude = log_magnitude

        # Build STFT transforms (one per scale)
        self.stfts = nn.ModuleList([
            _STFTMagnitude(n_fft=n, hop_length=n // 4)
            for n in self.fft_sizes
        ])

    def forward(
        self,
        pred: torch.Tensor,    # [batch, n_samples]
        target: torch.Tensor,  # [batch, n_samples]
    ) -> torch.Tensor:
        """Returns scalar loss."""
        # Pad or trim to same length
        min_len = min(pred.shape[-1], target.shape[-1])
        pred   = pred[..., :min_len]
        target = target[..., :min_len]

        total = torch.tensor(0.0, device=pred.device)
        for stft in self.stfts:
            mag_pred   = stft(pred)    # [batch, freq, time]
            mag_target = stft(target)

            if self.log_magnitude:
                mag_pred   = torch.log(mag_pred   + 1e-7)
                mag_target = torch.log(mag_target + 1e-7)

            total = total + torch.mean(torch.abs(mag_pred - mag_target))

        return total / len(self.stfts)


class _STFTMagnitude(nn.Module):
    """Wrapper: waveform → STFT magnitude spectrum."""

    def __init__(self, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_samples]
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )
        return stft.abs()  # [batch, freq, time]
