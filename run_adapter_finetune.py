"""
Stage 2 adapter fine-tuning: unfreeze last 4 DiT layers.

Run after the ADSR sensitivity test fails (adapter-only training insufficient).
Loads adapter_best.pt, unfreezes the last 4 transformer blocks, and trains
20K more steps with discriminative learning rates + L2-to-init regularization.

Usage:
    uv run python run_adapter_finetune.py
    uv run python run_adapter_finetune.py --adapter checkpoints_adapter/adapter_best.pt
    uv run python run_adapter_finetune.py --total_steps 20000 --dit_layers 4

Outputs checkpoints to checkpoints_adapter_ft/
"""

from __future__ import annotations
import argparse
import copy
import math
import json
import sys
import types
import torch
import torch.nn.functional as F
from pathlib import Path

# pkg_resources shim (needed by k_diffusion on Python 3.12+)
try:
    import pkg_resources  # noqa: F401
except ImportError:
    import packaging as _packaging
    _mod = types.ModuleType("pkg_resources")
    _mod.packaging = _packaging
    sys.modules["pkg_resources"] = _mod

from torch.utils.data import DataLoader, random_split

from timecode_audio.model.stable_audio_adapter import EnvelopeAdapter, LATENT_CHANNELS
from timecode_audio.data.adapter_dataset import AdapterDataset, collate_fn


VAE_DOWNSAMPLE = 2048


# ---------------------------------------------------------------------------
# Noise schedule helpers (copied from adapter_trainer.py)
# ---------------------------------------------------------------------------

def get_alphas_sigmas(t: torch.Tensor):
    alphas = torch.cos(t * math.pi / 2).view(-1, 1, 1)
    sigmas = torch.sin(t * math.pi / 2).view(-1, 1, 1)
    return alphas, sigmas


def v_prediction_target(latents, noise, alphas, sigmas):
    return noise * alphas - latents * sigmas


# ---------------------------------------------------------------------------
# Fine-tune trainer
# ---------------------------------------------------------------------------

class AdapterFinetuner:
    """
    Fine-tunes adapter + last N DiT transformer layers.

    Key differences from AdapterTrainer:
    - Loads pre-trained adapter weights
    - Unfreezes last `dit_layers` transformer blocks of sa_model
    - Two optimizer param groups: adapter @ lr_adapter, DiT @ lr_dit
    - L2-to-init regularization on unfrozen DiT layers (lambda_l2)
    """

    def __init__(
        self,
        data_dir: str,
        adapter_ckpt: str,
        checkpoint_dir: str = "checkpoints_adapter_ft",
        lr_adapter: float = 1e-4,
        lr_dit: float = 1e-5,
        lambda_l2: float = 1e-3,
        dit_layers: int = 4,
        batch_size: int = 4,
        total_steps: int = 20_000,
        warmup_steps: int = 200,
        grad_clip: float = 1.0,
        envelope_dropout: float = 0.20,
        log_every: int = 50,
        save_every: int = 2_000,
        val_split: float = 0.02,
        device: str = "auto",
    ) -> None:
        self.lr_adapter    = lr_adapter
        self.lr_dit        = lr_dit
        self.lambda_l2     = lambda_l2
        self.dit_layers    = dit_layers
        self.batch_size    = batch_size
        self.total_steps   = total_steps
        self.warmup_steps  = warmup_steps
        self.grad_clip     = grad_clip
        self.env_dropout   = envelope_dropout
        self.log_every     = log_every
        self.save_every    = save_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else torch.device(device)
        print(f"AdapterFinetuner: device={self.device}")

        # --- Load Stable Audio Open ---
        print("Loading Stable Audio Open...")
        try:
            from stable_audio_tools import get_pretrained_model
        except ImportError:
            raise ImportError("Install stable-audio-tools first.")
        self.sa_model, self.model_config = get_pretrained_model(
            "stabilityai/stable-audio-open-1.0"
        )
        self.sa_model = self.sa_model.to(self.device).eval()

        # Freeze everything first
        for p in self.sa_model.parameters():
            p.requires_grad_(False)

        # Unfreeze last `dit_layers` transformer blocks
        # Stable Audio Open DiT: sa_model.model.transformer.layers (24 blocks)
        # Try known path; if it fails, print the model structure to help debug.
        try:
            dit_blocks = self.sa_model.model.transformer.layers
        except AttributeError:
            print("\n[ERROR] Could not find sa_model.model.transformer.layers")
            print("Top-level attributes of sa_model.model:")
            for name, _ in self.sa_model.model.named_children():
                print(f"  {name}")
            raise
        n_total = len(dit_blocks)
        print(f"DiT has {n_total} transformer layers. Unfreezing last {dit_layers}.")
        self.unfrozen_blocks = dit_blocks[-dit_layers:]
        for block in self.unfrozen_blocks:
            for p in block.parameters():
                p.requires_grad_(True)

        # Store initial weights for L2-to-init regularization
        self.dit_init = {
            name: param.data.clone()
            for name, param in self.sa_model.named_parameters()
            if param.requires_grad
        }

        n_dit_trainable = sum(
            p.numel() for p in self.sa_model.parameters() if p.requires_grad
        )
        print(f"Unfrozen DiT params: {n_dit_trainable:,}")

        self.sample_rate = self.model_config["sample_rate"]

        # --- Load adapter ---
        print(f"Loading adapter from {adapter_ckpt}...")
        self.adapter = EnvelopeAdapter.load(adapter_ckpt, device=str(self.device))
        self.adapter.train()

        n_adapter = sum(p.numel() for p in self.adapter.parameters())
        print(f"Adapter params: {n_adapter:,}")

        # --- Optimizer: two param groups ---
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.adapter.parameters(), "lr": lr_adapter},
                {"params": [p for p in self.sa_model.parameters() if p.requires_grad],
                 "lr": lr_dit},
            ],
            weight_decay=1e-5,
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_steps - warmup_steps),
            eta_min=lr_adapter * 0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        # --- Data ---
        full_dataset = AdapterDataset(data_dir, augment=True)
        val_size     = max(1, int(len(full_dataset) * val_split))
        train_size   = len(full_dataset) - val_size
        train_set, val_set = random_split(full_dataset, [train_size, val_size])

        val_dataset = AdapterDataset(data_dir, augment=False)
        val_subset  = torch.utils.data.Subset(val_dataset, val_set.indices)

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

    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.sa_model.pretransform.encode(audio)

    def _get_conditioning(self, texts, seconds_total):
        metadata = [
            {"prompt": texts[i], "seconds_start": 0,
             "seconds_total": float(seconds_total[i].item())}
            for i in range(len(texts))
        ]
        with torch.no_grad():
            return self.sa_model.conditioner(metadata, self.device)

    def _l2_to_init_loss(self) -> torch.Tensor:
        """L2 regularization pulling unfrozen DiT weights back to their init values."""
        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.sa_model.named_parameters():
            if param.requires_grad and name in self.dit_init:
                loss = loss + ((param - self.dit_init[name]) ** 2).sum()
        return self.lambda_l2 * loss

    def _train_step(self, batch: dict) -> tuple[float, float]:
        audio         = batch["audio"].to(self.device)
        envelope      = batch["envelope"].to(self.device)
        texts         = batch["text_prompt"]
        seconds_total = batch["seconds_total"].to(self.device)

        B = audio.shape[0]

        with torch.no_grad():
            latents = self.sa_model.pretransform.encode(audio)
        T_latent = latents.shape[2]

        if envelope.shape[1] != T_latent:
            envelope = F.interpolate(
                envelope.unsqueeze(1), size=T_latent, mode="linear", align_corners=False
            ).squeeze(1)

        t = torch.rand(B, device=self.device)
        alphas, sigmas = get_alphas_sigmas(t)
        noise = torch.randn_like(latents)
        noisy_latents = latents * alphas + noise * sigmas

        adapter_embed = self.adapter(envelope)
        dropout_mask = (
            torch.rand(B, device=self.device) > self.env_dropout
        ).float().view(B, 1, 1)
        adapter_embed = adapter_embed * dropout_mask

        noisy_latents_adapted = noisy_latents + adapter_embed

        cond = self._get_conditioning(texts, seconds_total)
        model_output = self.sa_model(noisy_latents_adapted, t, cond)

        target = v_prediction_target(latents.detach(), noise, alphas, sigmas)
        diffusion_loss = F.mse_loss(model_output, target)
        reg_loss = self._l2_to_init_loss()
        loss = diffusion_loss + reg_loss

        self.optimizer.zero_grad()
        loss.backward()

        # Clip adapter and DiT grads separately
        torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.sa_model.parameters() if p.requires_grad],
            self.grad_clip,
        )

        self.optimizer.step()
        self.scheduler.step()

        return diffusion_loss.item(), reg_loss.item()

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.adapter.eval()
        self.sa_model.eval()
        total_loss, n_batches = 0.0, 0

        for batch in self.val_loader:
            audio         = batch["audio"].to(self.device)
            envelope      = batch["envelope"].to(self.device)
            texts         = batch["text_prompt"]
            seconds_total = batch["seconds_total"].to(self.device)
            B = audio.shape[0]

            latents  = self.sa_model.pretransform.encode(audio)
            T_latent = latents.shape[2]

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
            model_output = self.sa_model(noisy_latents_adapted, t, cond)

            target = v_prediction_target(latents, noise, alphas, sigmas)
            total_loss += F.mse_loss(model_output, target).item()
            n_batches  += 1

        self.adapter.train()
        self.sa_model.train()
        # Keep frozen params in eval mode
        self.sa_model.pretransform.eval()
        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, tag: str) -> None:
        """Save adapter + unfrozen DiT layers."""
        self.adapter.save(str(self.checkpoint_dir / f"adapter_{tag}.pt"))
        # Save unfrozen DiT state (just the last N layers)
        dit_state = {
            name: param.data
            for name, param in self.sa_model.named_parameters()
            if param.requires_grad
        }
        torch.save(dit_state, str(self.checkpoint_dir / f"dit_layers_{tag}.pt"))

    def train(self) -> None:
        n_adapter = sum(p.numel() for p in self.adapter.parameters())
        n_dit = sum(p.numel() for p in self.sa_model.parameters() if p.requires_grad)
        print(f"Fine-tuning: {n_adapter} adapter params + {n_dit} DiT params")
        print(f"LRs: adapter={self.lr_adapter:.0e}, DiT={self.lr_dit:.0e}, "
              f"λ_l2={self.lambda_l2:.0e}")

        self.adapter.train()
        self.sa_model.train()
        self.sa_model.pretransform.eval()   # keep VAE frozen/eval

        data_iter = iter(self.train_loader)

        while self.step < self.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            diff_loss, reg_loss = self._train_step(batch)

            if self.step % self.log_every == 0:
                lr_a = self.optimizer.param_groups[0]["lr"]
                lr_d = self.optimizer.param_groups[1]["lr"]
                print(f"step {self.step:6d} | loss {diff_loss:.4f} "
                      f"| reg {reg_loss:.4f} | lr_a {lr_a:.1e} | lr_d {lr_d:.1e}")

            if self.step % self.save_every == 0 and self.step > 0:
                val_loss = self._evaluate()
                print(f"  val_loss {val_loss:.4f}")
                self._save_checkpoint(f"step_{self.step:06d}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best")
                    print(f"  Saved best (val_loss {val_loss:.4f})")

            self.step += 1

        self._save_checkpoint("final")
        print("Fine-tuning complete.")
        print(f"Adapter: {self.checkpoint_dir}/adapter_final.pt")
        print(f"DiT layers: {self.checkpoint_dir}/dit_layers_final.pt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     default="data/adapter_training")
    parser.add_argument("--adapter",      default="checkpoints_adapter/adapter_best.pt")
    parser.add_argument("--out_dir",      default="checkpoints_adapter_ft")
    parser.add_argument("--dit_layers",   type=int,   default=4)
    parser.add_argument("--total_steps",  type=int,   default=20_000)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lr_adapter",   type=float, default=1e-4)
    parser.add_argument("--lr_dit",       type=float, default=1e-5)
    parser.add_argument("--lambda_l2",    type=float, default=1e-3)
    args = parser.parse_args()

    finetuner = AdapterFinetuner(
        data_dir=args.data_dir,
        adapter_ckpt=args.adapter,
        checkpoint_dir=args.out_dir,
        lr_adapter=args.lr_adapter,
        lr_dit=args.lr_dit,
        lambda_l2=args.lambda_l2,
        dit_layers=args.dit_layers,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
    )
    finetuner.train()


if __name__ == "__main__":
    main()
