"""
Training loop for Option A (ADSR-extended DDSP).

Curriculum:
    Stage 1 (0–50K):    sine + sawtooth, pitch-only (text_dim=0)
    Stage 2 (50K–200K): all synths, pitch-only (text_dim=0, preserves Stage 1 checkpoint shape)
    Stage 3 (200K–400K): synthetic + NSynth, CLAP text conditioning + 30% null dropout (text_dim=512)
    Stage 4 (400K+):    + augmentation, upsample edge cases

Loss: Multi-Scale Spectral (MSS) only.
      No envelope reconstruction loss — gate is analytic (guaranteed).
"""

from __future__ import annotations
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from timecode_audio.model.ddsp_synthesizer import (
    DDSPSynthesizer, duration_to_frames, duration_to_samples
)
from timecode_audio.model.adsr_encoder import ADSREncoder, adsr_gate_samples
from timecode_audio.model.losses import MultiScaleSpectralLoss
from timecode_audio.training.config import TrainingConfig


class Trainer:
    """
    Trains the DDSPSynthesizer model end-to-end.

    Usage:
        config = TrainingConfig()
        dataset = SyntheticDataset(config.train_data_dir)
        trainer = Trainer(config, dataset)
        trainer.train()
    """

    def __init__(
        self,
        config: TrainingConfig,
        dataset: torch.utils.data.Dataset,
        device: str = "auto",
    ) -> None:
        self.config = config
        self.device = self._resolve_device(device)

        # --- Models ---
        self.synthesizer = DDSPSynthesizer(
            n_harmonics=config.n_harmonics,
            n_noise_bands=config.n_noise_bands,
            hidden_dim=config.hidden_dim,
            text_dim=config.text_dim,
            sample_rate=config.sample_rate,
            frame_size=config.frame_size,
        ).to(self.device)

        self.adsr_encoder = ADSREncoder(adsr_dim=config.adsr_dim).to(self.device)

        # --- Loss ---
        self.loss_fn = MultiScaleSpectralLoss(sample_rate=config.sample_rate).to(self.device)

        # --- Text encoder (None for Stages 1-2; CLAP for Stage 3) ---
        if config.text_dim > 0:
            from timecode_audio.model.text_encoder import CLAPTextEncoder
            self.text_encoder = CLAPTextEncoder(device=str(self.device))
            self.text_encoder._model = self.text_encoder._model.to(self.device)
        else:
            self.text_encoder = None

        # --- Optimizer (split param groups for differential LR at Stage 3) ---
        self.optimizer = torch.optim.AdamW(
            [
                {"params": list(self.synthesizer.parameters()), "lr": config.learning_rate},
                {"params": list(self.adsr_encoder.parameters()), "lr": config.learning_rate * config.adsr_lr_scale},
            ],
            weight_decay=config.weight_decay,
        )

        # LR schedule: linear warmup then cosine decay
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.total_steps - config.start_step - config.warmup_steps),
            eta_min=config.learning_rate * 0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.warmup_steps],
        )

        # --- Data ---
        val_size = max(1, int(len(dataset) * config.val_split))
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

        # --- Checkpointing ---
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
        print(f"Training on {self.device} | "
              f"{sum(p.numel() for p in self.synthesizer.parameters()):,} params")

        self.synthesizer.train()
        self.adsr_encoder.train()

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
                stage = self.config.curriculum_stage(self.step)
                print(f"step {self.step:6d} | loss {loss:.4f} | lr {lr:.2e} | {stage}")

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

        audio_target = batch["audio"].to(self.device)
        f0_hz        = batch["f0_hz"].to(self.device)
        A            = batch["A"].to(self.device)
        D            = batch["D"].to(self.device)
        S            = batch["S"].to(self.device)
        R            = batch["R"].to(self.device)
        velocity     = batch["velocity"].to(self.device)
        note_dur     = batch["note_duration"].to(self.device)

        n_samples = audio_target.shape[1]
        n_frames  = math.ceil(n_samples / self.config.frame_size)

        stage_pos, _ = self.adsr_encoder(
            A, D, S, R, note_dur,
            n_frames=n_frames,
            frame_size=self.config.frame_size,
            sample_rate=self.config.sample_rate,
        )
        gate = self._compute_batch_gate(A, D, S, R, note_dur, n_samples)

        # Text embedding with 30% null dropout (CFG training)
        text_emb = None
        if self.text_encoder is not None:
            texts = batch["text_prompt"]
            if torch.rand(1).item() >= self.config.text_dropout_prob:
                text_emb = self.text_encoder(texts)   # [B, 512]

        audio_pred = self.synthesizer(
            f0_hz=f0_hz,
            stage_pos=stage_pos,
            gate=gate,
            velocity=velocity,
            text_emb=text_emb,
        )

        # Confidence-weighted loss (Stage 3: down-weight uncertain pseudo-labels)
        if "adsr_confidence" in batch:
            confidence = batch["adsr_confidence"].to(self.device)  # [B]
            per_sample = self.loss_fn(audio_pred, audio_target, reduction="none")  # [B]
            loss = (per_sample * confidence).mean()
        else:
            loss = self.loss_fn(audio_pred, audio_target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.synthesizer.parameters()) + list(self.adsr_encoder.parameters()),
            self.config.grad_clip,
        )
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.synthesizer.eval()
        self.adsr_encoder.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            audio_target = batch["audio"].to(self.device)
            f0_hz        = batch["f0_hz"].to(self.device)
            A  = batch["A"].to(self.device)
            D  = batch["D"].to(self.device)
            S  = batch["S"].to(self.device)
            R  = batch["R"].to(self.device)
            velocity  = batch["velocity"].to(self.device)
            note_dur  = batch["note_duration"].to(self.device)

            n_samples = audio_target.shape[1]
            n_frames  = math.ceil(n_samples / self.config.frame_size)

            stage_pos, _ = self.adsr_encoder(
                A, D, S, R, note_dur,
                n_frames=n_frames,
                frame_size=self.config.frame_size,
                sample_rate=self.config.sample_rate,
            )
            gate = self._compute_batch_gate(A, D, S, R, note_dur, n_samples)

            # Always use text at eval (no dropout) when text encoder is active
            text_emb = None
            if self.text_encoder is not None:
                texts = batch["text_prompt"]
                text_emb = self.text_encoder(texts)

            audio_pred = self.synthesizer(
                f0_hz=f0_hz,
                stage_pos=stage_pos,
                gate=gate,
                velocity=velocity,
                text_emb=text_emb,
            )
            loss = self.loss_fn(audio_pred, audio_target)
            total_loss += loss.item()
            n_batches += 1

        self.synthesizer.train()
        self.adsr_encoder.train()
        return total_loss / max(n_batches, 1)

    def _compute_batch_gate(
        self,
        A: torch.Tensor,
        D: torch.Tensor,
        S: torch.Tensor,
        R: torch.Tensor,
        note_dur: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Compute per-sample ADSR gate for each clip in the batch."""
        gates = []
        for i in range(A.shape[0]):
            g = adsr_gate_samples(
                A=float(A[i].item()),
                D=float(D[i].item()),
                S=float(S[i].item()),
                R=float(R[i].item()),
                note_duration_ms=float(note_dur[i].item()),
                n_samples=n_samples,
                sample_rate=self.config.sample_rate,
            )
            gates.append(g)
        return torch.stack(gates, dim=0).to(A.device)   # [B, n_samples]

    def _save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save({
            "step":         self.step,
            "synthesizer":  self.synthesizer.state_dict(),
            "adsr_encoder": self.adsr_encoder.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "scheduler":    self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.synthesizer.load_state_dict(ckpt["synthesizer"])
        self.adsr_encoder.load_state_dict(ckpt["adsr_encoder"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step = ckpt["step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Loaded checkpoint from step {self.step}")

    def load_checkpoint_for_stage3(self, path: str) -> None:
        """
        Load a Stage 2 checkpoint (text_dim=0) into a Stage 3 model (text_dim=512).

        The SpectralPredictor's first Linear layer changes shape:
            Stage 2: Linear(pitch_dim=64, hidden_dim)
            Stage 3: Linear(pitch_dim + text_dim = 576, hidden_dim)

        Copies all weight-shape-matching parameters. For the one mismatched layer
        (spectral_predictor.trunk.0.weight), copies the pitch columns and leaves
        the text columns at zero (they will be trained on NSynth).
        """
        ckpt = torch.load(path, map_location=self.device)

        # --- Synthesizer: partial load for text_dim expansion ---
        new_state = self.synthesizer.state_dict()
        old_synth  = ckpt["synthesizer"]
        for k, v in old_synth.items():
            if k not in new_state:
                continue
            if new_state[k].shape == v.shape:
                new_state[k] = v
            elif k == "spectral_predictor.trunk.0.weight":
                # [hidden_dim, old_in_dim] → [hidden_dim, new_in_dim]
                # Copy pitch columns; zero the text columns so Stage 3 starts at
                # Stage 2 behavior for null-text passes (zeros × weight = 0 regardless
                # of init; but for text-conditioned passes random init can cause large
                # initial perturbations, so zeros are more stable).
                new_state[k].zero_()
                new_state[k][:, : v.shape[1]] = v
        self.synthesizer.load_state_dict(new_state)

        # --- ADSR encoder: direct load (architecture unchanged) ---
        self.adsr_encoder.load_state_dict(ckpt["adsr_encoder"])

        # Do NOT load optimizer/scheduler state — we're starting a new LR schedule
        self.step = ckpt["step"]
        self.best_val_loss = float("inf")   # reset for Stage 3
        print(
            f"Loaded Stage 2 checkpoint (step {self.step}) into Stage 3 model "
            f"(text_dim expanded 0 → {self.config.text_dim}). "
            f"Optimizer and scheduler reset."
        )
