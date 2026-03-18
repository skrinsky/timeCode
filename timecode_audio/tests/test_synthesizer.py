"""
Smoke tests for the DDSP synthesizer and related model components.
Verifies shapes, forward pass, and key audio properties — not perceptual quality.
"""

import math
import pytest
import torch
import numpy as np

from timecode_audio.model.ddsp_synthesizer import (
    DDSPSynthesizer,
    PitchEncoder,
    SpectralPredictor,
    harmonic_synth,
    filtered_noise,
    duration_to_samples,
    SAMPLE_RATE,
    FRAME_SIZE,
    N_HARMONICS,
    N_NOISE_BANDS,
)
from timecode_audio.model.adsr_encoder import ADSREncoder, adsr_gate_samples
from timecode_audio.model.losses import MultiScaleSpectralLoss


BATCH = 2
F0 = 440.0     # A4
DUR = 500.0    # ms


def make_adsr_batch(batch=BATCH, device="cpu"):
    return {
        "A": torch.tensor([10.0] * batch, device=device),
        "D": torch.tensor([50.0] * batch, device=device),
        "S": torch.tensor([0.6]  * batch, device=device),
        "R": torch.tensor([100.0] * batch, device=device),
        "note_duration": torch.tensor([DUR] * batch, device=device),
        "f0_hz": torch.tensor([F0] * batch, device=device),
        "velocity": torch.tensor([0.8] * batch, device=device),
    }


class TestPitchEncoder:
    def test_output_shape(self):
        enc = PitchEncoder(n_freqs=32)
        f0 = torch.tensor([440.0, 220.0])
        out = enc(f0)
        assert out.shape == (2, 64)

    def test_different_pitches_differ(self):
        enc = PitchEncoder(n_freqs=32)
        e1 = enc(torch.tensor([440.0]))
        e2 = enc(torch.tensor([880.0]))
        assert not torch.allclose(e1, e2)


class TestSpectralPredictor:
    def test_output_shapes(self):
        pred = SpectralPredictor(pitch_dim=64, text_dim=0)
        pitch = torch.randn(BATCH, 64)
        pk, su = pred(pitch)
        assert pk.shape == (BATCH, N_HARMONICS + N_NOISE_BANDS)
        assert su.shape == pk.shape

    def test_peak_sustain_differ(self):
        pred = SpectralPredictor(pitch_dim=64)
        pitch = torch.randn(BATCH, 64)
        pk, su = pred(pitch)
        # Network should (usually) produce different states for peak vs sustain
        assert not torch.allclose(pk, su)


class TestADSREncoder:
    def test_stage_pos_shape(self):
        enc = ADSREncoder(adsr_dim=128)
        b = make_adsr_batch()
        n_frames = 50
        stage_pos, adsr_vec = enc(
            b["A"], b["D"], b["S"], b["R"], b["note_duration"],
            n_frames=n_frames,
        )
        assert stage_pos.shape == (BATCH, n_frames)
        assert adsr_vec.shape == (BATCH, 128)

    def test_stage_pos_bounded(self):
        enc = ADSREncoder()
        b = make_adsr_batch()
        stage_pos, _ = enc(b["A"], b["D"], b["S"], b["R"], b["note_duration"], n_frames=100)
        assert stage_pos.min() >= 0.0
        assert stage_pos.max() <= 1.0

    def test_stage_pos_peak_during_attack(self):
        # A=100ms, D=200ms, stage_pos should peak at ~frame corresponding to 100ms
        enc = ADSREncoder()
        A = torch.tensor([100.0])
        D = torch.tensor([200.0])
        S = torch.tensor([0.5])
        R = torch.tensor([100.0])
        dur = torch.tensor([500.0])
        n_frames = 100  # 100 frames × 5.33ms ≈ 533ms
        stage_pos, _ = enc(A, D, S, R, dur, n_frames=n_frames)
        sp = stage_pos[0]
        # Frame at A=100ms: frame index ≈ 100ms / 5.33ms ≈ 18-19
        peak_frame = sp.argmax().item()
        assert 15 <= peak_frame <= 25

    def test_adsr_gate_samples_shape(self):
        g = adsr_gate_samples(10, 50, 0.6, 100, DUR, n_samples=48000)
        assert g.shape == (48000,)

    def test_adsr_gate_samples_bounded(self):
        g = adsr_gate_samples(10, 50, 0.6, 100, DUR, n_samples=24000)
        assert g.min() >= 0.0
        assert g.max() <= 1.0


class TestHarmonicSynth:
    def test_output_shape(self):
        n_samples = duration_to_samples(DUR)
        n_frames = math.ceil(n_samples / FRAME_SIZE)
        amps = torch.ones(BATCH, n_frames, N_HARMONICS) * 0.01
        f0 = torch.tensor([F0] * BATCH)
        out = harmonic_synth(amps, f0, n_samples)
        assert out.shape == (BATCH, n_samples)

    def test_nyquist_masking(self):
        # High f0: many harmonics should be zeroed by Nyquist mask
        n_samples = duration_to_samples(200)
        n_frames = math.ceil(n_samples / FRAME_SIZE)
        # f0=5000Hz: harmonics 5 and above (25kHz+) should be zeroed
        amps = torch.ones(1, n_frames, N_HARMONICS) * 0.1
        f0 = torch.tensor([5000.0])
        out = harmonic_synth(amps, f0, n_samples)
        # Should not be all zero (first 4 harmonics pass)
        assert out.abs().max() > 0

    def test_silence_for_zero_amps(self):
        n_samples = duration_to_samples(200)
        n_frames = math.ceil(n_samples / FRAME_SIZE)
        amps = torch.zeros(1, n_frames, N_HARMONICS)
        f0 = torch.tensor([440.0])
        out = harmonic_synth(amps, f0, n_samples)
        assert out.abs().max() < 1e-6

    def test_frequency_content(self):
        # Sine at 440Hz: fundamental should dominate the spectrum
        n_samples = SAMPLE_RATE  # 1 second
        n_frames = math.ceil(n_samples / FRAME_SIZE)
        amps = torch.zeros(1, n_frames, N_HARMONICS)
        amps[:, :, 0] = 1.0   # only fundamental (k=1)
        f0 = torch.tensor([440.0])
        out = harmonic_synth(amps, f0, n_samples)[0]  # [n_samples]

        fft = torch.fft.rfft(out)
        freqs = torch.fft.rfftfreq(n_samples, d=1/SAMPLE_RATE)
        peak_freq = freqs[fft.abs().argmax()].item()
        # Peak should be close to 440Hz
        assert abs(peak_freq - 440.0) < 5.0  # within 5Hz


class TestFilteredNoise:
    def test_output_shape(self):
        n_samples = duration_to_samples(DUR)
        n_frames = math.ceil(n_samples / FRAME_SIZE)
        mags = torch.ones(BATCH, n_frames, N_NOISE_BANDS) * 0.01
        out = filtered_noise(mags, n_samples)
        assert out.shape == (BATCH, n_samples)

    def test_non_silent(self):
        n_samples = duration_to_samples(200)
        n_frames = math.ceil(n_samples / FRAME_SIZE)
        mags = torch.ones(1, n_frames, N_NOISE_BANDS) * 0.1
        out = filtered_noise(mags, n_samples)
        assert out.abs().max() > 0


class TestDDSPSynthesizer:
    def test_forward_shape(self):
        model = DDSPSynthesizer()
        b = make_adsr_batch()
        n_samples = duration_to_samples(DUR + 100)  # note + release
        n_frames = math.ceil(n_samples / FRAME_SIZE)

        enc = ADSREncoder()
        stage_pos, _ = enc(
            b["A"], b["D"], b["S"], b["R"], b["note_duration"],
            n_frames=n_frames,
        )
        gate = torch.stack([
            adsr_gate_samples(10, 50, 0.6, 100, DUR, n_samples)
            for _ in range(BATCH)
        ])

        out = model(
            f0_hz=b["f0_hz"],
            stage_pos=stage_pos,
            gate=gate,
            velocity=b["velocity"],
        )
        assert out.shape == (BATCH, n_samples)

    def test_velocity_scales_output(self):
        model = DDSPSynthesizer()
        n_samples = duration_to_samples(300)
        n_frames = math.ceil(n_samples / FRAME_SIZE)
        enc = ADSREncoder()

        A = torch.tensor([10.0, 10.0])
        D = torch.tensor([50.0, 50.0])
        S = torch.tensor([0.6,  0.6])
        R = torch.tensor([50.0, 50.0])
        dur = torch.tensor([200.0, 200.0])
        f0 = torch.tensor([440.0, 440.0])

        stage_pos, _ = enc(A, D, S, R, dur, n_frames=n_frames)
        gate = torch.stack([adsr_gate_samples(10, 50, 0.6, 50, 200, n_samples)] * 2)

        # velocity 0.5 vs 1.0
        vel_lo = torch.tensor([0.5, 1.0])
        out = model(f0_hz=f0, stage_pos=stage_pos, gate=gate, velocity=vel_lo)

        # output[0] amplitude should be roughly half of output[1]
        rms_lo = out[0].pow(2).mean().sqrt().item()
        rms_hi = out[1].pow(2).mean().sqrt().item()
        assert rms_hi > rms_lo * 1.5   # hi should be notably louder

    def test_zero_velocity_silence(self):
        model = DDSPSynthesizer()
        n_samples = duration_to_samples(300)
        n_frames = math.ceil(n_samples / FRAME_SIZE)
        enc = ADSREncoder()

        A = torch.tensor([10.0])
        D = torch.tensor([50.0])
        S = torch.tensor([0.6])
        R = torch.tensor([50.0])
        dur = torch.tensor([200.0])
        f0 = torch.tensor([440.0])
        vel = torch.tensor([0.0])

        stage_pos, _ = enc(A, D, S, R, dur, n_frames=n_frames)
        gate = adsr_gate_samples(10, 50, 0.6, 50, 200, n_samples).unsqueeze(0)

        out = model(f0_hz=f0, stage_pos=stage_pos, gate=gate, velocity=vel)
        assert out.abs().max() < 1e-6


class TestMSSLoss:
    def test_zero_loss_identical(self):
        loss_fn = MultiScaleSpectralLoss()
        x = torch.randn(2, 48000)
        loss = loss_fn(x, x)
        assert loss.item() < 1e-4

    def test_nonzero_loss_different(self):
        loss_fn = MultiScaleSpectralLoss()
        x = torch.randn(2, 48000)
        y = torch.randn(2, 48000)
        assert loss_fn(x, y).item() > 0.01

    def test_differentiable(self):
        loss_fn = MultiScaleSpectralLoss()
        x = torch.randn(2, 8192, requires_grad=True)
        y = torch.randn(2, 8192)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
