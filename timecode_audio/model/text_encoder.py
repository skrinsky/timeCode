"""
Text encoder wrapper.

Stage 1–2: pitch-only conditioning — text encoder returns None (text_dim=0).
Stage 3+: CLAP text encoder (frozen weights, text_dim=512).

CLAP note: CLAP was trained at 48kHz audio but its text encoder is a
standard transformer (BERT-style). Text input does not need resampling.
Plug in the real CLAP encoder here when moving to Stage 3.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


class TextEncoderStub(nn.Module):
    """
    Stage 1–2 stub: returns None, signalling pitch-only conditioning.
    Replace with CLAPTextEncoder for Stage 3.
    """

    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 0

    def forward(self, texts: list[str]) -> Optional[torch.Tensor]:
        return None


class CLAPTextEncoder(nn.Module):
    """
    Frozen CLAP text encoder (LAION-CLAP or Microsoft CLAP).

    Weights are frozen — we use CLAP purely as a feature extractor.
    get_text_features() runs both the text model and the projection head,
    returning the projected 512-dim embedding (not raw 768-dim pooler output).

    output_dim: 512 (LAION-CLAP projected)
    """

    def __init__(
        self,
        model_name: str = "laion/larger_clap_music_and_speech",
        output_dim: int = 512,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self._device = device

        try:
            from transformers import ClapModel, ClapProcessor
            self._model = ClapModel.from_pretrained(model_name)
            self._processor = ClapProcessor.from_pretrained(model_name)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
        except ImportError:
            raise ImportError(
                "transformers is required for CLAPTextEncoder. "
                "Install with: uv pip install transformers"
            )

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        texts : list of strings, length = batch_size
        Returns : [batch, output_dim]  — projected CLAP text embeddings
        """
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        # get_text_features() applies both the text model and projection head → 512-dim
        emb = self._model.get_text_features(**inputs)
        # Some transformers versions return a ModelOutput object instead of a tensor
        if not isinstance(emb, torch.Tensor):
            emb = emb.text_embeds if hasattr(emb, "text_embeds") else emb.pooler_output
        return emb
