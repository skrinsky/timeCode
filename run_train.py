from timecode_audio.data.synthetic_gen import SyntheticDataset
from timecode_audio.training.config import TrainingConfig
from timecode_audio.training.trainer import Trainer

config = TrainingConfig(total_steps=200_000)
dataset = SyntheticDataset('data/synthetic')
trainer = Trainer(config, dataset)
trainer.load_checkpoint('checkpoints/best.pt')
trainer.train()
