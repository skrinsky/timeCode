"""
Text encoder wrapper.

Stage 1 (current): pitch-only conditioning — text encoder returns None.
Stage 2+: CLAP text encoder (frozen weights).

CLAP note: CLAP was trained at 48kHz audio but its text encoder is a
standard transformer (BERT-style). Text input does not need resampling.
Plug in the real CLAP encoder here when moving to Stage 2.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


class TextEncoderStub(nn.Module):
    """
    Stage 1 stub: returns None, signalling pitch-only conditioning.
    Replace with CLAPTextEncoder for Stage 2.
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
    Only the text branch is loaded to keep memory footprint small.

    output_dim: 512 (LAION-CLAP) or 1024 (CLAP-2023) depending on checkpoint.
    """

    def __init__(
        self,
        model_name: str = "laion/larger_clap_music_and_speech",
        output_dim: int = 512,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        try:
            from transformers import ClapModel, ClapProcessor
            self._model = ClapModel.from_pretrained(model_name).text_model
            self._processor = ClapProcessor.from_pretrained(model_name)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
            self._available = True
        except ImportError:
            raise ImportError(
                "transformers is required for CLAPTextEncoder. "
                "Install with: uv pip install transformers"
            )

        self._device = device

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        texts : list of strings, length = batch_size
        Returns : [batch, output_dim]
        """
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        # Use pooled output (CLS token or mean pool depending on CLAP version)
        emb = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0]
        return emb
