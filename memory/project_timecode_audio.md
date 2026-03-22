---
name: timecode_audio_project
description: Full project state — architecture, completed work, current path, key technical decisions, and next steps for ADSR-conditioned timecode-aligned generative audio
type: project
---

## What this project is

Generative audio system that maps explicit ADSR envelope parameters (Attack, Decay, Sustain, Release) to SMPTE timecode cue positions to synthesize audio. No existing system does this — closest prior art is Sketch2Sound (Adobe, ICASSP 2025) which does amplitude curves but not discrete ADSR parameters or timecodes.

**Two input modes:**
- Easy: `{"timecode": "00:01:23:14", "sound": "staccato piano C4"}` — ADSR inferred from text
- Precise: add explicit A/D/S/R fields — override any or all inferred defaults
- BPM mode: `{"position": "1.1.0", "sound": "kick drum"}` with session-level bpm/time_signature

**Key creative feature:** Text prompt controls timbre, ADSR controls envelope — independently. Physically impossible combinations are valid ("sustained snare drum", "slow-attack piano").

---

## Current architecture: Option B — Stable Audio Open + envelope adapter

**Why we moved away from DDSP (Option A):**
- DDSP has a hard timbral quality ceiling (harmonic-plus-noise, can't do foley/SFX/drums)
- NSynth is 16kHz — upsampling adds no information above 8kHz
- SLE and RTE evaluation metrics failed due to fundamental architectural issue: DDSP synthesizer's spectral energy varies across ADSR stages independently of the gate, so S in the gate doesn't map to S in the output amplitude. Not fixable without losing natural amplitude evolution.
- Option A ATE=5ms ✓, DTE=10ms ✓, SLE=0.36 ✗ (target 0.05), RTE=82.5ms ✗ (target 50ms)

**Why ETTA was dropped as the backbone:**
- ETTA (NVIDIA, ICML 2025) was original planned backbone. Weights announced October 2024, still "coming soon" as of March 2026. GitHub has training code but no checkpoint.

**Current backbone: Stable Audio Open (Stability AI)**
- Available on HuggingFace: `stabilityai/stable-audio-open-1.0` (gated — requires license acceptance)
- 44.1kHz stereo, up to 47 seconds
- VAE: AutoencoderOobleck, 64 latent channels, 2048x temporal downsampling → ~21.53 Hz latent frame rate
- DiT: 24 layers, inner_dim=1536, 24 attention heads (GQA 12 KV heads), cross_attn_dim=768
- Text encoder: T5-base (768-dim), frozen
- Timing conditioning: seconds_start + seconds_total via projection model
- Diffusion objective: v-prediction with cosine schedule (t ∈ [0,1])
- Trained on Freesound CC0 — covers instruments, SFX, foley, ambiences, everything

**Adapter architecture (Sketch2Sound pattern, ICASSP 2025):**
- Single trainable `Linear(1, 64)` — zero initialized
- ADSR params → `adsr_gate_samples()` → 1D piecewise-linear envelope curve [T_samples]
- Curve aligns exactly with VAE frame rate (frame_size=2048 samples = one latent frame)
- Linear projects [B, T_latent, 1] → [B, T_latent, 64] → transpose → [B, 64, T_latent]
- Added element-wise to noisy latents before DiT forward pass
- 20% envelope dropout per clip during training → enables CFG over envelope at inference
- Random median filter augmentation (window 1–25 frames) during training

**Output sample rate:** 44.1kHz from Stable Audio → upsample to 48kHz with `sinc_interp_kaiser` at pipeline output (post-production standard)

---

## Completed work

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Infrastructure | ✅ Complete | SMPTE parser, BPM resolver, CueEvent schema, ADSR gate/stage_position, adsr_inferer, adsr_defaults — 134 tests passing |
| Option A Stage 1 (DDSP) | ✅ Complete | 50K steps, sine+sawtooth |
| Option A Stage 2 (DDSP) | ✅ Complete | 200K steps, all 5 synth types |
| Option A Stage 3 (DDSP) | ✅ Complete | 400K steps, NSynth fine-tuning with CLAP |
| ADSR Estimator | ✅ Complete | CNN on log-mel spec, MC dropout confidence (fixed: log-space variance), Platt scaler not yet fitted |
| NSynth pseudo-labeling | ✅ Complete | 143,861 clips above confidence 0.5 at data/nsynth_estimated/metadata.jsonl |
| Option B adapter code | ✅ Complete | stable_audio_adapter.py, adapter_dataset.py, adapter_trainer.py |
| Option B data generation | ✅ Code written | generate_training_data.py — 80+ prompts, needs to run on cluster |

---

## Next steps (in order)

1. **On cluster:** `pip install stable-audio-tools einops`
2. **Accept HF license** at huggingface.co/stabilityai/stable-audio-open-1.0, set HF_TOKEN
3. **Generate training data:** `python generate_training_data.py --n_clips 20000` (runs overnight)
4. **Train adapter:** `python run_adapter_train.py`
5. **ADSR sensitivity test:** generate same prompt with A=10ms vs A=500ms — if ATE > 100ms, unfreeze last 4 DiT layers
6. **Inference pipeline rewrite** for Stable Audio Open backbone (pipeline.py, mixer.py)
7. **Freesound dataset** (Phase 5) for additional diversity if adapter precision insufficient

---

## Key files

```
timecode_audio/
  core/
    timecode.py, bpm_resolver.py, cue_event.py, envelope.py
    adsr_defaults.py, adsr_inferer.py
  model/
    stable_audio_adapter.py   ← NEW: EnvelopeAdapter, audio_to_envelope()
    adsr_estimator.py         ← ADSR estimator (pseudo-labeling)
    ddsp_synthesizer.py       ← Option A reference only
  data/
    adapter_dataset.py        ← NEW: training data loader for adapter
    nsynth_loader.py          ← NSynth with pseudo-labels
    synthetic_gen.py
  training/
    adapter_trainer.py        ← NEW: Option B training loop
    trainer.py                ← Option A trainer
    config.py
  eval/
    adsr_metrics.py           ← ATE/DTE/SLE/RTE (uses note_duration for fitting)
generate_training_data.py     ← NEW: generate clips from Stable Audio Open
run_adapter_train.py          ← NEW: launch adapter training
run_train_stage3.py           ← Option A Stage 3 (complete)
run_eval.py                   ← Option A evaluation
research/implementation_plan.md  ← Full plan, up to date
```

---

## Dev environment

**Local machine:**
- Package manager: uv — `uv run python`, `uv run pytest`, `uv pip install`
- Python 3.11.14 in .venv managed by uv
- Bare `python` not on PATH — always use `uv run python`

**Cluster (training):**
- Server: cmn17, /scratch/summerk/timeCode, tmux "train2"
- GPU: 4090 (CUDA)
- Standard python/pip work fine on cluster

---

## Important technical decisions

- **ADSR as envelope curve, not 4 scalars:** Adapter receives the output of `adsr_gate_samples()` — a 1D time series — not discrete A/D/S/R values. This is how Sketch2Sound works and what diffusion models can follow.
- **Zero initialization:** Adapter starts by adding nothing — model inherits pretrained behavior exactly at step 0.
- **Training data source:** Generate from Stable Audio Open itself (self-supervised, in-distribution). Do NOT use synthetic DDSP audio — completely out of Stable Audio's distribution.
- **Platt scaler not fitted:** ADSR estimator confidence uses uncalibrated `1/(1+mean_var)`. Acceptable for current pseudo-labels since threshold (0.5) is already filtering well empirically. Fit Platt scaler before Phase 5 (Freesound pseudo-labeling at scale).
- **v-prediction + cosine schedule:** stable-audio-tools uses t ∈ [0,1], alphas=cos(t*π/2), sigmas=sin(t*π/2). This is what adapter_trainer.py implements.
