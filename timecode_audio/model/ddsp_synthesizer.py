"""
ADSR-extended harmonic-plus-noise synthesizer at 48kHz.

Architecture:
    text_prompt + pitch_hz  →  PitchEncoder + (optional) CLAP text encoder
                            →  shared MLP trunk
                            →  params_peak    [N_harmonics + N_noise_bands]
                               params_sustain [N_harmonics + N_noise_bands]

Per-frame synthesis:
    α(t)              = stage_position(t)               (from ADSREncoder)
    harmonic_amps(t)  = α * params_peak[:H] + (1-α) * params_sustain[:H]
    noise_mags(t)     = α * params_peak[H:] + (1-α) * params_sustain[H:]
    audio(t)          = HarmonicSynth(harmonic_amps, f0) + FilteredNoise(noise_mags)
    output(t)         = audio(t) * g(t) * velocity      (g from adsr_encoder)

All synthesis runs at 48kHz. No 16kHz internal representation.

Deliberate choices:
    - Piecewise-linear ADSR (swap to exponential in envelope.py if needed)
    - Frequency-adaptive Nyquist masking: harmonics above 24kHz zeroed per note
    - Phase accumulation across frames for click-free synthesis
    - Sample-accurate ADSR gate (applied per-sample, not per-frame)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

SAMPLE_RATE = 48000
FRAME_SIZE = 256           # samples per synthesis frame (~5.33ms)
N_HARMONICS = 128          # max harmonics; Nyquist-masked per note
N_NOISE_BANDS = 65         # half-spectrum of FFT size 128
NYQUIST = SAMPLE_RATE // 2  # 24000 Hz


# ---------------------------------------------------------------------------
# Pitch encoder
# ---------------------------------------------------------------------------

class PitchEncoder(nn.Module):
    """
    Encodes fundamental frequency (Hz) as a sinusoidal embedding.
    Works in log-frequency space so the network sees octave distances linearly.

    Output dim: 2 * n_freqs (sin + cos pairs)
    """

    def __init__(self, n_freqs: int = 32) -> None:
        super().__init__()
        self.n_freqs = n_freqs
        # Fixed frequency bands spanning piano range (A0=27.5Hz to C8=4186Hz)
        # in log space
        log_freqs = torch.linspace(
            math.log(20.0), math.log(20000.0), n_freqs
        )
        self.register_buffer("log_freqs", log_freqs)

    def forward(self, f0_hz: torch.Tensor) -> torch.Tensor:
        """
        f0_hz : [batch] or [batch, 1]
        Returns : [batch, 2 * n_freqs]
        """
        f0 = f0_hz.view(-1, 1)  # [batch, 1]
        log_f0 = torch.log(f0.clamp(min=1.0))  # [batch, 1]
        phases = log_f0 * self.log_freqs.unsqueeze(0)  # [batch, n_freqs]
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)  # [batch, 2*n_freqs]


# ---------------------------------------------------------------------------
# Spectral prediction network
# ---------------------------------------------------------------------------

class SpectralPredictor(nn.Module):
    """
    Maps (pitch_embedding [+ text_embedding]) → (params_peak, params_sustain).

    Outputs are log-scale amplitudes (before softplus), shape [batch, N_harmonics + N_noise_bands].
    The network outputs two states: peak (at attack) and sustain (during sustain/release).
    """

    def __init__(
        self,
        pitch_dim: int = 64,       # 2 * PitchEncoder.n_freqs
        text_dim: int = 0,         # 0 = no text conditioning (Stage 1)
        hidden_dim: int = 256,
        n_harmonics: int = N_HARMONICS,
        n_noise_bands: int = N_NOISE_BANDS,
    ) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        output_dim = n_harmonics + n_noise_bands

        in_dim = pitch_dim + text_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Two heads: peak state and sustain state
        self.head_peak    = nn.Linear(hidden_dim, output_dim)
        self.head_sustain = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        pitch_emb: torch.Tensor,            # [batch, pitch_dim]
        text_emb: Optional[torch.Tensor] = None,  # [batch, text_dim] or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        params_peak    : [batch, n_harmonics + n_noise_bands]  log-scale amps
        params_sustain : [batch, n_harmonics + n_noise_bands]  log-scale amps
        """
        if text_emb is not None:
            x = torch.cat([pitch_emb, text_emb], dim=-1)
        else:
            x = pitch_emb

        h = self.trunk(x)
        params_peak    = self.head_peak(h)     # log-scale, unbounded
        params_sustain = self.head_sustain(h)

        return params_peak, params_sustain


# ---------------------------------------------------------------------------
# Harmonic synthesizer (additive)
# ---------------------------------------------------------------------------

def harmonic_synth(
    harmonic_amps: torch.Tensor,  # [batch, n_frames, n_harmonics]  (positive, linear scale)
    f0_hz: torch.Tensor,          # [batch] or [batch, 1]
    n_samples: int,
    sample_rate: int = SAMPLE_RATE,
    frame_size: int = FRAME_SIZE,
) -> torch.Tensor:
    """
    Phase-accumulation additive synthesis.

    Each harmonic k is a sinusoid at k * f0_hz. Amplitudes are interpolated
    linearly between frames to avoid magnitude discontinuities.

    Anti-aliasing: harmonics whose frequency exceeds Nyquist are zeroed.

    Returns : [batch, n_samples]
    """
    batch_size, n_frames, n_harmonics = harmonic_amps.shape
    device = harmonic_amps.device
    f0 = f0_hz.view(batch_size, 1).float()  # [batch, 1]

    # Build harmonic number tensor [1, 1, n_harmonics]
    k = torch.arange(1, n_harmonics + 1, device=device).float().view(1, 1, -1)

    # Nyquist mask: zero out harmonics above 24kHz — [batch, 1, n_harmonics]
    nyquist = sample_rate / 2.0
    harmonic_freqs = f0.unsqueeze(-1) * k  # [batch, 1, n_harmonics]
    nyquist_mask = (harmonic_freqs < nyquist).float()  # [batch, 1, n_harmonics]

    # Upsample frame-level amplitudes to sample level via linear interpolation
    # harmonic_amps: [batch, n_frames, n_harmonics] → [batch, n_harmonics, n_frames]
    amps_t = harmonic_amps.permute(0, 2, 1)  # [batch, n_harmonics, n_frames]
    # Interpolate to n_samples
    amps_samples = F.interpolate(
        amps_t, size=n_samples, mode="linear", align_corners=True
    )  # [batch, n_harmonics, n_samples]
    amps_samples = amps_samples.permute(0, 2, 1)  # [batch, n_samples, n_harmonics]

    # nyquist_mask is [batch, 1, n_harmonics]; amps_samples is [batch, n_samples, n_harmonics]
    # The middle dimension 1 broadcasts to n_samples automatically — no reshape needed
    amps_samples = amps_samples * nyquist_mask  # [batch, n_samples, n_harmonics]

    # Phase accumulation
    # Instantaneous angular frequency per sample per harmonic: ω_k = 2π * k * f0 / sr
    omega = 2 * math.pi * f0.unsqueeze(-1) * k / sample_rate  # [batch, 1, n_harmonics]
    # Phase at each sample: cumulative sum of ω
    t = torch.arange(n_samples, device=device).float().view(1, -1, 1)  # [1, n_samples, 1]
    phase = omega * t  # [batch, n_samples, n_harmonics]

    # Synthesize
    signal = (amps_samples * torch.sin(phase)).sum(dim=-1)  # [batch, n_samples]

    return signal


# ---------------------------------------------------------------------------
# Filtered noise synthesizer
# ---------------------------------------------------------------------------

def filtered_noise(
    noise_mags: torch.Tensor,   # [batch, n_frames, n_noise_bands]  (positive, linear)
    n_samples: int,
    sample_rate: int = SAMPLE_RATE,
    frame_size: int = FRAME_SIZE,
    fft_size: int = 128,
) -> torch.Tensor:
    """
    Frame-wise filtered noise via overlap-add (fully vectorized).

    All frames are processed in a single batched FFT/IFFT, then overlap-added
    using F.fold. No Python loop over frames.

    Returns : [batch, n_samples]
    """
    batch_size, n_frames, n_bands = noise_mags.shape
    device = noise_mags.device
    hop = frame_size
    window = torch.hann_window(fft_size, device=device)  # [fft_size]

    # All noise frames at once: [B, n_frames, fft_size]
    noise = torch.randn(batch_size, n_frames, fft_size, device=device)
    noise = noise * window  # [fft_size] broadcasts over [B, n_frames, fft_size]

    # Batch FFT: [B, n_frames, fft_size//2 + 1]
    noise_fft = torch.fft.rfft(noise, n=fft_size)

    # Apply spectral envelope (magnitude only, preserve random phase)
    noise_fft_filtered = noise_fft * noise_mags  # [B, n_frames, n_bands]

    # Batch IFFT: [B, n_frames, fft_size]
    noise_frames = torch.fft.irfft(noise_fft_filtered, n=fft_size)
    noise_frames = noise_frames * window  # synthesis window

    # Overlap-add via F.fold
    # fold expects [B, fft_size, n_frames]; output [B, 1, 1, fold_length]
    fold_length = (n_frames - 1) * hop + fft_size
    output = F.fold(
        noise_frames.permute(0, 2, 1),        # [B, fft_size, n_frames]
        output_size=(1, fold_length),
        kernel_size=(1, fft_size),
        stride=(1, hop),
    ).squeeze(1).squeeze(1)                    # [B, fold_length]

    # Pad with zeros if fold_length < n_samples (last partial frame gap)
    if fold_length < n_samples:
        output = torch.cat(
            [output, torch.zeros(batch_size, n_samples - fold_length, device=device)],
            dim=1,
        )

    return output[:, :n_samples]


# ---------------------------------------------------------------------------
# Full DDSP synthesizer model
# ---------------------------------------------------------------------------

class DDSPSynthesizer(nn.Module):
    """
    ADSR-extended harmonic-plus-noise synthesizer at 48kHz.

    Forward pass:
        1. Encode pitch (and optionally text) → params_peak, params_sustain
        2. Interpolate per frame using stage_position(t)
        3. Synthesize harmonic signal + filtered noise
        4. Apply sample-accurate ADSR gate × velocity

    Returns : [batch, n_samples]
    """

    def __init__(
        self,
        n_harmonics: int = N_HARMONICS,
        n_noise_bands: int = N_NOISE_BANDS,
        hidden_dim: int = 256,
        text_dim: int = 0,           # 0 = Stage 1 (pitch only)
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
    ) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.text_dim = text_dim

        pitch_encoder_freqs = 32
        pitch_dim = 2 * pitch_encoder_freqs

        self.pitch_encoder = PitchEncoder(n_freqs=pitch_encoder_freqs)
        self.spectral_predictor = SpectralPredictor(
            pitch_dim=pitch_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            n_harmonics=n_harmonics,
            n_noise_bands=n_noise_bands,
        )

    def forward(
        self,
        f0_hz: torch.Tensor,            # [batch]
        stage_pos: torch.Tensor,        # [batch, n_frames]  from ADSREncoder
        gate: torch.Tensor,             # [batch, n_samples] analytic ADSR gate
        velocity: torch.Tensor,         # [batch]
        text_emb: Optional[torch.Tensor] = None,  # [batch, text_dim] or None
    ) -> torch.Tensor:
        """
        Returns audio : [batch, n_samples]
        """
        batch_size, n_frames = stage_pos.shape
        n_samples = gate.shape[1]

        # 1. Predict spectral states from pitch (and text)
        pitch_emb = self.pitch_encoder(f0_hz)  # [batch, pitch_dim]
        params_peak, params_sustain = self.spectral_predictor(pitch_emb, text_emb)

        # 2. Split into harmonic amps and noise mags; apply softplus for positivity
        H = self.n_harmonics
        peak_h    = F.softplus(params_peak[:, :H])      # [batch, H]
        peak_n    = F.softplus(params_peak[:, H:])      # [batch, N_noise_bands]
        sust_h    = F.softplus(params_sustain[:, :H])
        sust_n    = F.softplus(params_sustain[:, H:])

        # 3. Per-frame spectral interpolation: α * peak + (1-α) * sustain
        # stage_pos: [batch, n_frames], expand to [batch, n_frames, 1]
        alpha = stage_pos.unsqueeze(-1)  # [batch, n_frames, 1]

        # Expand static params to [batch, 1, dim] for broadcast
        harm_amps = alpha * peak_h.unsqueeze(1) + (1 - alpha) * sust_h.unsqueeze(1)
        # [batch, n_frames, n_harmonics]

        noise_mags = alpha * peak_n.unsqueeze(1) + (1 - alpha) * sust_n.unsqueeze(1)
        # [batch, n_frames, n_noise_bands]

        # 4. Synthesize
        harmonic_signal = harmonic_synth(harm_amps, f0_hz, n_samples,
                                         self.sample_rate, self.frame_size)
        noise_signal = filtered_noise(noise_mags, n_samples,
                                      self.sample_rate, self.frame_size)

        audio = harmonic_signal + noise_signal  # [batch, n_samples]

        # 5. Apply ADSR gate × velocity (sample-accurate)
        # velocity: [batch] → [batch, 1]
        audio = audio * gate * velocity.unsqueeze(-1)

        return audio


# ---------------------------------------------------------------------------
# Convenience: compute n_frames for a given duration
# ---------------------------------------------------------------------------

def duration_to_frames(duration_ms: float, frame_size: int = FRAME_SIZE,
                        sample_rate: int = SAMPLE_RATE) -> int:
    """Number of synthesis frames needed to cover duration_ms + some release tail."""
    n_samples = int(duration_ms / 1000.0 * sample_rate) + frame_size
    return math.ceil(n_samples / frame_size)


def duration_to_samples(duration_ms: float, sample_rate: int = SAMPLE_RATE) -> int:
    return int(duration_ms / 1000.0 * sample_rate)
