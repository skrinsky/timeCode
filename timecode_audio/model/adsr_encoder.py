"""
ADSREncoder: converts ADSR scalars into the two signals the synthesizer needs.

    stage_position(t)  — per-frame spectral interpolation schedule [0,1]
                         (wraps the analytic function from core/envelope.py)
    adsr_vector        — learned dense projection of ADSR scalars used as
                         conditioning in Option B's cross-attention pathway

For Option A, only stage_position is used in synthesis.
For Option B, adsr_vector is fed to the ControlNet adapter.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray

from timecode_audio.core.envelope import stage_position as analytic_stage_position


class ADSREncoder(nn.Module):
    """
    Encodes (A, D, S, R, note_duration) into:
        - stage_position_frames : Tensor [batch, n_frames]  — analytic, no grad
        - adsr_vector           : Tensor [batch, adsr_dim]  — learned projection
    """

    def __init__(self, adsr_dim: int = 128) -> None:
        super().__init__()
        # Log-scale input projection (A, D, R are log-scale; S is linear)
        # Input: [log(A+1), log(D+1), S, log(R+1), log(dur+1)] → 5 dims
        self.projection = nn.Sequential(
            nn.Linear(5, 64),
            nn.SiLU(),
            nn.Linear(64, adsr_dim),
            nn.SiLU(),
            nn.Linear(adsr_dim, adsr_dim),
        )

    def forward(
        self,
        A: torch.Tensor,       # [batch] attack ms
        D: torch.Tensor,       # [batch] decay ms
        S: torch.Tensor,       # [batch] sustain level 0-1
        R: torch.Tensor,       # [batch] release ms
        note_duration: torch.Tensor,  # [batch] ms
        n_frames: int,
        frame_size: int = 256,
        sample_rate: int = 48000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        stage_pos : [batch, n_frames]  in [0, 1]
        adsr_vec  : [batch, adsr_dim]
        """
        # --- Learned projection (Option B conditioning) ---
        x = torch.stack([
            torch.log(A + 1),
            torch.log(D + 1),
            S,
            torch.log(R + 1),
            torch.log(note_duration + 1),
        ], dim=-1)  # [batch, 5]
        adsr_vec = self.projection(x)  # [batch, adsr_dim]

        # --- Analytic stage_position (Option A synthesis) ---
        # Compute frame timestamps in ms
        frame_duration_ms = frame_size / sample_rate * 1000.0
        t_ms = np.arange(n_frames, dtype=np.float32) * frame_duration_ms

        batch_size = A.shape[0]
        stage_pos_list = []
        for i in range(batch_size):
            sp = analytic_stage_position(
                t_ms,
                A=float(A[i].item()),
                D=float(D[i].item()),
                note_duration_ms=float(note_duration[i].item()),
            )
            stage_pos_list.append(sp)

        stage_pos = torch.tensor(
            np.stack(stage_pos_list, axis=0),
            dtype=torch.float32,
            device=A.device,
        )  # [batch, n_frames]

        return stage_pos, adsr_vec


def adsr_gate_frames(
    A: float,
    D: float,
    S: float,
    R: float,
    note_duration_ms: float,
    n_frames: int,
    frame_size: int = 256,
    sample_rate: int = 48000,
) -> NDArray:
    """
    Compute analytic ADSR gate at frame timestamps (numpy, no grad).
    Returns array of shape [n_frames].
    """
    from timecode_audio.core.envelope import adsr_gate
    frame_duration_ms = frame_size / sample_rate * 1000.0
    t_ms = np.arange(n_frames, dtype=np.float32) * frame_duration_ms
    return adsr_gate(t_ms, A=A, D=D, S=S, R=R, note_duration_ms=note_duration_ms)


def adsr_gate_samples(
    A: float,
    D: float,
    S: float,
    R: float,
    note_duration_ms: float,
    n_samples: int,
    sample_rate: int = 48000,
) -> torch.Tensor:
    """
    Compute analytic ADSR gate at every sample (more accurate than frame-level).
    Used for the final amplitude multiplication in the synthesizer.
    Returns Tensor of shape [n_samples].
    """
    from timecode_audio.core.envelope import adsr_gate
    t_ms = np.arange(n_samples, dtype=np.float32) / sample_rate * 1000.0
    g = adsr_gate(t_ms, A=A, D=D, S=S, R=R, note_duration_ms=note_duration_ms)
    return torch.from_numpy(g)
