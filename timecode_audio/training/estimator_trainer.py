"""
Training loop for the ADSR estimator.

Input : synthetic audio clips (known ground-truth ADSR)
Output: trained ADSREstimator checkpoint

Usage:
    dataset = SyntheticDataset('data/synthetic')
    trainer = EstimatorTrainer(EstimatorConfig(), dataset)
    trainer.train()
"""

from __future__ import annotations
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from timecode_audio.model.adsr_estimator import ADSREstimator, adsr_estimator_loss


class EstimatorConfig:
    sample_rate:       int   = 48000
    n_mels:            int   = 128
    hidden_dim:        int   = 256
    learning_rate:     float = 1e-3
    weight_decay:      float = 1e-5
    batch_size:        int   = 64
    total_steps:       int   = 50_000
    warmup_steps:      int   = 1_000
    grad_clip:         float = 1.0
    val_split:         float = 0.02
    log_every:         int   = 100
    eval_every:        int   = 2_000
    checkpoint_every:  int   = 5_000
    train_data_dir:    str   = 'data/synthetic'
    checkpoint_dir:    str   = 'checkpoints_estimator'


class EstimatorTrainer:

    def __init__(
        self,
        config: EstimatorConfig,
        dataset: torch.utils.data.Dataset,
        device: str = "auto",
    ) -> None:
        self.config = config
        self.device = self._resolve_device(device)

        self.model = ADSREstimator(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.total_steps - config.warmup_steps),
            eta_min=config.learning_rate * 0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.warmup_steps],
        )

        val_size   = max(1, int(len(dataset) * config.val_split))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.best_val_loss = float("inf")

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def train(self) -> None:
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Training ADSR estimator on {self.device} | {n_params:,} params")

        self.model.train()
        data_iter = iter(self.train_loader)

        while self.step < self.config.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            loss = self._train_step(batch)

            if self.step % self.config.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"step {self.step:6d} | loss {loss:.4f} | lr {lr:.2e}")

            if self.step % self.config.eval_every == 0 and self.step > 0:
                val_loss = self._evaluate()
                print(f"  val_loss {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best.pt")

            if self.step % self.config.checkpoint_every == 0 and self.step > 0:
                self._save_checkpoint(f"step_{self.step:06d}.pt")

            self.step += 1

        self._save_checkpoint("final.pt")
        print("Training complete.")

    def _train_step(self, batch: dict) -> float:
        self.optimizer.zero_grad()

        audio    = batch["audio"].to(self.device)       # [B, n_samples]
        A_ms     = batch["A"].to(self.device)
        D_ms     = batch["D"].to(self.device)
        S_level  = batch["S"].to(self.device)
        R_ms     = batch["R"].to(self.device)

        log_A, log_D, S_pred, log_R = self.model(audio)
        loss = adsr_estimator_loss(log_A, log_D, S_pred, log_R, A_ms, D_ms, S_level, R_ms)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.val_loader:
            audio   = batch["audio"].to(self.device)
            A_ms    = batch["A"].to(self.device)
            D_ms    = batch["D"].to(self.device)
            S_level = batch["S"].to(self.device)
            R_ms    = batch["R"].to(self.device)

            log_A, log_D, S_pred, log_R = self.model(audio)
            loss = adsr_estimator_loss(log_A, log_D, S_pred, log_R, A_ms, D_ms, S_level, R_ms)
            total_loss += loss.item()
            n_batches  += 1

        self.model.train()
        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save({
            "step":           self.step,
            "model":          self.model.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "scheduler":      self.scheduler.state_dict(),
            "best_val_loss":  self.best_val_loss,
        }, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step           = ckpt["step"]
        self.best_val_loss  = ckpt.get("best_val_loss", float("inf"))
        print(f"Loaded estimator checkpoint from step {self.step}")
