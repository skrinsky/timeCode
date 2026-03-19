from timecode_audio.data.synthetic_gen import SyntheticDataset
from timecode_audio.training.config import TrainingConfig
from timecode_audio.training.trainer import Trainer

config = TrainingConfig(total_steps=50_000)
dataset = SyntheticDataset('data/synthetic')
Trainer(config, dataset).train()
