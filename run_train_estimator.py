from timecode_audio.data.synthetic_gen import SyntheticDataset
from timecode_audio.training.estimator_trainer import EstimatorConfig, EstimatorTrainer

config  = EstimatorConfig()
dataset = SyntheticDataset('data/synthetic')
EstimatorTrainer(config, dataset).train()
