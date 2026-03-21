"""
Launch adapter training.

Prerequisites:
    1. generate_training_data.py completed (data/adapter_training/metadata.jsonl exists)
    2. HF_TOKEN set and Stability AI license accepted
    3. stable-audio-tools installed: pip install stable-audio-tools

Usage:
    python run_adapter_train.py
    python run_adapter_train.py --data_dir data/adapter_training --steps 40000
"""

import argparse
from timecode_audio.training.adapter_trainer import AdapterTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",     default="data/adapter_training")
parser.add_argument("--checkpoint_dir", default="checkpoints_adapter")
parser.add_argument("--steps",        type=int,   default=40_000)
parser.add_argument("--batch_size",   type=int,   default=4)
parser.add_argument("--lr",           type=float, default=1e-4)
args = parser.parse_args()

trainer = AdapterTrainer(
    data_dir        = args.data_dir,
    checkpoint_dir  = args.checkpoint_dir,
    learning_rate   = args.lr,
    batch_size      = args.batch_size,
    total_steps     = args.steps,
)
trainer.train()
