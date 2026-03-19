"""
Stage 3 training: NSynth fine-tuning with CLAP text conditioning.

Run after:
    1. Stage 2 training complete (checkpoints/best.pt exists)
    2. ADSR estimator trained (checkpoints_estimator/best.pt)
    3. run_pseudo_label.py completed (data/nsynth_estimated/metadata.jsonl exists)

Key differences from Stage 2:
    - text_dim=512: CLAP text conditioning active
    - adsr_lr_scale=0.3: ADSR encoder trains at 30% of synthesizer LR
    - MixedDataset: 70% synthetic + 30% NSynth
    - Partial weight load: text_dim 0 → 512 expansion
    - total_steps=400_000 (runs 200K new steps from Stage 2 step)
"""

from timecode_audio.data.synthetic_gen import SyntheticDataset
from timecode_audio.data.nsynth_loader import NSynthDataset, MixedDataset
from timecode_audio.training.config import TrainingConfig
from timecode_audio.training.trainer import Trainer

config = TrainingConfig(
    text_dim       = 512,
    adsr_lr_scale  = 0.3,
    total_steps    = 400_000,
    batch_size     = 16,
    warmup_steps   = 1_000,
)

synthetic = SyntheticDataset('data/synthetic')
nsynth    = NSynthDataset('data/nsynth_estimated', min_confidence=0.5)
dataset   = MixedDataset(synthetic, nsynth, synthetic_ratio=0.7)

trainer = Trainer(config, dataset)
trainer.load_checkpoint_for_stage3('checkpoints/best.pt')
trainer.train()
