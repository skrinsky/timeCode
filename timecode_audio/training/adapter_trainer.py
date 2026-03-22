"""
Adapter training loop for Stable Audio Open envelope conditioning.

Uses stable-audio-tools to load the pretrained model.
Only the EnvelopeAdapter (single Linear layer) trains — everything else frozen.

Training objective: v-prediction MSE on the DiT output.
Noise schedule: cosine (t ∈ [0,1]), matching stable-audio-tools convention.

Conditioning dropout: 20% probability of zeroing the envelope embedding per clip,
enabling classifier-free guidance over the envelope at inference.
"""

from __future__ import annotations
import math
import json
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# k_diffusion (dep of stable-audio-tools) imports openai-clip which uses
# pkg_resources — removed in Python 3.12+. Shim it before the import chain fires.
try:
    import pkg_resources  # noqa: F401
except ImportError:
    import packaging as _packaging
    _mod = types.ModuleType("pkg_resources")
    _mod.packaging = _packaging
    sys.modules["pkg_resources"] = _mod
from torch.utils.data import DataLoader, random_split

from timecode_audio.model.stable_audio_adapter import EnvelopeAdapter
from timecode_audio.data.adapter_dataset import AdapterDataset, collate_fn


LATENT_CHANNELS = 64


# ---------------------------------------------------------------------------
# Noise schedule helpers
# ---------------------------------------------------------------------------

def get_alphas_sigmas(t: torch.Tensor):
    """
    Cosine noise schedule: alpha = cos(t*pi/2), sigma = sin(t*pi/2).
    t ∈ [0, 1]. Returns (alphas, sigmas) each shaped [B, 1, 1] for broadcasting.
    """
    alphas = torch.cos(t * math.pi / 2).view(-1, 1, 1)
    sigmas = torch.sin(t * math.pi / 2).view(-1, 1, 1)
    return alphas, sigmas


def v_prediction_target(
    latents: torch.Tensor,
    noise: torch.Tensor,
    alphas: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """v-prediction target: v = noise * alpha - latent * sigma."""
    return noise * alphas - latents * sigmas


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AdapterTrainer:
    """
    Trains the EnvelopeAdapter on top of frozen Stable Audio Open.

    Loading the model requires stable-audio-tools:
        pip install stable-audio-tools

    The model weights are downloaded from HuggingFace on first run
    (requires HF_TOKEN and accepting the Stability AI community license).
    """

    def __init__(
        self,
        data_dir: str,
        checkpoint_dir: str = "checkpoints_adapter",
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        total_steps: int = 40_000,
        warmup_steps: int = 500,
        grad_clip: float = 1.0,
        envelope_dropout: float = 0.20,
        log_every: int = 50,
        save_every: int = 2_000,
        val_split: float = 0.02,
        device: str = "auto",
    ) -> None:
        self.lr               = learning_rate
        self.batch_size       = batch_size
        self.total_steps      = total_steps
        self.warmup_steps     = warmup_steps
        self.grad_clip        = grad_clip
        self.env_dropout      = envelope_dropout
        self.log_every        = log_every
        self.save_every       = save_every
        self.checkpoint_dir   = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = self._resolve_device(device)
        print(f"AdapterTrainer: device={self.device}")

        # --- Load Stable Audio Open (frozen) ---
        print("Loading Stable Audio Open from HuggingFace...")
        try:
            from stable_audio_tools import get_pretrained_model
        except ImportError:
            raise ImportError(
                "stable-audio-tools is required. "
                "Install with: pip install stable-audio-tools"
            )
        self.sa_model, self.model_config = get_pretrained_model(
            "stabilityai/stable-audio-open-1.0"
        )
        self.sa_model = self.sa_model.to(self.device)
        self.sa_model.eval()
        for p in self.sa_model.parameters():
            p.requires_grad_(False)

        self.sample_rate = self.model_config["sample_rate"]   # 44100
        print(f"Stable Audio Open loaded. Sample rate: {self.sample_rate}")

        # --- Adapter (only trainable component) ---
        self.adapter = EnvelopeAdapter(latent_channels=LATENT_CHANNELS).to(self.device)

        # --- Optimizer and schedule ---
        self.optimizer = torch.optim.AdamW(
            self.adapter.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=learning_rate * 0.01
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        # --- Data ---
        full_dataset = AdapterDataset(data_dir, augment=True)
        val_size   = max(1, int(len(full_dataset) * val_split))
        train_size = len(full_dataset) - val_size
        train_set, val_set = random_split(full_dataset, [train_size, val_size])

        # Val set: no augmentation — use a fresh dataset instance
        val_dataset = AdapterDataset(data_dir, augment=False)
        val_indices = val_set.indices
        val_subset  = torch.utils.data.Subset(val_dataset, val_indices)

        self.train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True, collate_fn=collate_fn,
        )

        self.step = 0
        self.best_val_loss = float("inf")

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode stereo audio to VAE latents.
        audio: [B, 2, n_samples] at 44100 Hz
        Returns: [B, 64, T_latent]
        """
        with torch.no_grad():
            latents = self.sa_model.pretransform.encode(audio)
        return latents

    def _get_conditioning(self, texts: list[str], seconds_total: torch.Tensor) -> object:
        """Build conditioning dict for stable-audio-tools conditioner."""
        metadata = [
            {
                "prompt":        texts[i],
                "seconds_start": 0,
                "seconds_total": float(seconds_total[i].item()),
            }
            for i in range(len(texts))
        ]
        with torch.no_grad():
            cond = self.sa_model.conditioner(metadata, self.device)
        return cond

    def _train_step(self, batch: dict) -> float:
        audio        = batch["audio"].to(self.device)         # [B, 2, n_samples]
        envelope     = batch["envelope"].to(self.device)      # [B, T_latent]
        texts        = batch["text_prompt"]
        seconds_total = batch["seconds_total"].to(self.device)

        B = audio.shape[0]

        # 1. Encode audio → latents
        latents = self._encode_audio(audio)                   # [B, 64, T_latent]
        T_latent = latents.shape[2]

        # 2. Trim / pad envelope to match latent length
        if envelope.shape[1] != T_latent:
            envelope = F.interpolate(
                envelope.unsqueeze(1), size=T_latent, mode="linear", align_corners=False
            ).squeeze(1)

        # 3. Sample timestep and compute noisy latents
        t = torch.rand(B, device=self.device)
        alphas, sigmas = get_alphas_sigmas(t)
        noise = torch.randn_like(latents)
        noisy_latents = latents * alphas + noise * sigmas     # [B, 64, T_latent]

        # 4. Adapter embedding with envelope dropout (CFG training)
        adapter_embed = self.adapter(envelope)                # [B, 64, T_latent]
        # Zero out adapter for dropout_prob fraction of the batch
        dropout_mask = (
            torch.rand(B, device=self.device) > self.env_dropout
        ).float().view(B, 1, 1)
        adapter_embed = adapter_embed * dropout_mask

        noisy_latents_adapted = noisy_latents + adapter_embed

        # 5. Get text+timing conditioning
        cond = self._get_conditioning(texts, seconds_total)

        # 6. Forward through frozen DiT (outside no_grad so grads flow to adapter_embed)
        # DiT params have requires_grad=False so only adapter gets updated.
        model_output = self.sa_model.model(noisy_latents_adapted, t, cond)

        # 7. V-prediction loss
        target = v_prediction_target(latents.detach(), noise, alphas, sigmas)
        loss = F.mse_loss(model_output, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.adapter.eval()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.val_loader:
            audio        = batch["audio"].to(self.device)
            envelope     = batch["envelope"].to(self.device)
            texts        = batch["text_prompt"]
            seconds_total = batch["seconds_total"].to(self.device)
            B = audio.shape[0]

            latents   = self._encode_audio(audio)
            T_latent  = latents.shape[2]

            if envelope.shape[1] != T_latent:
                envelope = F.interpolate(
                    envelope.unsqueeze(1), size=T_latent, mode="linear", align_corners=False
                ).squeeze(1)

            t = torch.rand(B, device=self.device)
            alphas, sigmas = get_alphas_sigmas(t)
            noise = torch.randn_like(latents)
            noisy_latents = latents * alphas + noise * sigmas

            adapter_embed = self.adapter(envelope)
            noisy_latents_adapted = noisy_latents + adapter_embed

            cond = self._get_conditioning(texts, seconds_total)
            model_output = self.sa_model.model(noisy_latents_adapted, t, cond)

            target = v_prediction_target(latents, noise, alphas, sigmas)
            total_loss += F.mse_loss(model_output, target).item()
            n_batches  += 1

        self.adapter.train()
        return total_loss / max(n_batches, 1)

    def train(self) -> None:
        print(f"Training adapter | {sum(p.numel() for p in self.adapter.parameters())} params")
        self.adapter.train()
        data_iter = iter(self.train_loader)

        while self.step < self.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            loss = self._train_step(batch)

            if self.step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"step {self.step:6d} | loss {loss:.4f} | lr {lr:.2e}")

            if self.step % self.save_every == 0 and self.step > 0:
                val_loss = self._evaluate()
                print(f"  val_loss {val_loss:.4f}")
                self.adapter.save(str(self.checkpoint_dir / f"adapter_step_{self.step:06d}.pt"))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.adapter.save(str(self.checkpoint_dir / "adapter_best.pt"))
                    print(f"  Saved best adapter (val_loss {val_loss:.4f})")

            self.step += 1

        self.adapter.save(str(self.checkpoint_dir / "adapter_final.pt"))
        print("Adapter training complete.")
