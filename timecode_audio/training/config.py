"""
Training hyperparameters and curriculum configuration.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    # --- Audio ---
    sample_rate: int = 48000
    frame_size: int = 256          # samples per synthesis frame
    n_harmonics: int = 128
    n_noise_bands: int = 65

    # --- Model ---
    hidden_dim: int = 256
    adsr_dim: int = 128
    text_dim: int = 0              # 0 = Stage 1 (no text); 512 = CLAP

    # --- Training ---
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 16
    total_steps: int = 500_000
    warmup_steps: int = 2_000
    grad_clip: float = 1.0
    adsr_lr_scale:     float = 1.0            # multiplier for ADSR encoder LR (set to 0.3 for Stage 3)
    log_every: int = 100
    checkpoint_every: int = 5_000
    eval_every: int = 5_000

    # --- Data ---
    train_data_dir: str = "data/synthetic"
    val_split: float = 0.02        # fraction of data held out for validation

    # --- Curriculum stages ---
    # Each entry: (start_step, end_step, description)
    curriculum: list[tuple[int, int, str]] = field(default_factory=lambda: [
        (0,       50_000,  "stage1_sine_sawtooth"),
        (50_000,  200_000, "stage2_all_synthetic"),
        (200_000, 400_000, "stage3_synthetic_nsynth"),
        (400_000, 500_000, "stage4_augmentation"),
    ])

    # --- CFG (text dropout for Stage 2+) ---
    text_dropout_prob: float = 0.30   # 30% null text during training

    # --- Output ---
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    def curriculum_stage(self, step: int) -> str:
        for start, end, name in self.curriculum:
            if start <= step < end:
                return name
        return self.curriculum[-1][2]

    def use_text(self, step: int) -> bool:
        """Text conditioning active from Stage 2 onwards."""
        return step >= self.curriculum[1][0]
