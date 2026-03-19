---
name: timecode_audio_project
description: Overview of the ADSR-conditioned timecode-aligned generative audio project — what it is, the research gap, architecture, where files live, and current build status
type: project
---

This project is building a novel generative audio system that maps ADSR envelope parameters (Attack, Decay, Sustain, Release) to SMPTE timecode cue positions to synthesize audio. No existing system does this.

**Why:** Generative audio models can't align to timecode. Existing approaches control when sounds occur (PicoAudio, FreeAudio) or draw amplitude curves (Music ControlNet, T-Foley) but none accept discrete ADSR parameters per timecode cue and use them to shape both amplitude envelope and spectral evolution.

**The gap (verified genuine):** No system maps explicit A/D/S/R as four named values to timecode positions to drive generative synthesis. Closest prior art: MPEG-4 SASL (1999, deterministic only), JASCO (chords/rhythm not ADSR), SynthCloner (ADSR transfer but no timeline), DDSP (implicit envelope, not user-specified). Sketch2Sound (Adobe, ICASSP 2025) validates the adapter+CFG approach but is amplitude-only, no ADSR parameterization, no timecode.

**Key creative feature:** Timbre and envelope are independent pathways — users can combine any sound description with any ADSR shape, including physically impossible combinations (sustained snare, slow-attack piano, instant strings).

**Two-mode input:**
- Easy: `{"timecode": "00:01:23:14", "sound": "staccato piano C4"}` — ADSR inferred from text
- Precise: add explicit A/D/S/R fields — override any or all inferred defaults
- BPM mode: `{"position": "1.1.0", "sound": "kick drum"}` with session-level bpm/time_signature

**Architecture:**
- Option A (built): ADSR-extended DDSP at 48kHz. Analytic ADSR gate controls amplitude; stage_position(t) drives spectral interpolation between two learned states (params_peak, params_sustain). Fast, ADSR-guaranteed, limited to harmonic sounds.
- Option B (Phase 6): ETTA-DiT backbone + ADSR cross-attention ControlNet adapter. Text via AdaLayerNorm (frozen), ADSR via cross-attention (trained). Handles drums, foley, complex sounds.

**Data strategy:**
- Option A: 500K synthetic clips (known ADSR) + NSynth estimated (200K) + augmentation
- Option B: AF3 off-the-shelf for sound type labels; ADSR estimator (synthetic pre-trained) for ADSR pseudo-labels; ~100-200 clip human evaluation gate before scaling

**File locations:**
- Repo: /Users/summerkrinsky/Documents/GitHub/timeCode/
- Research + implementation plan: research/implementation_plan.md (full, detailed, ~830 lines)
- Prior art survey: research/timecode_aligned_audio.md
- Code: timecode_audio/

**Implementation phases and status:**
- ✅ Phase 0: Infrastructure complete (SMPTE parser, BPM resolver, CueEvent schema, ADSR gate/stage_position, ADSR inferer, defaults lookup) — 123 tests passing
- ✅ Phase 1: DDSP baseline built and all bugs fixed (filtered_noise vectorized, batch_size 64→16, LR warmup, STFT window buffer, 29.97ND timecode fix) — 134 tests passing
- ✅ Stage 1 training: COMPLETE on 4090. 50K steps, final loss 0.93, checkpoint saved to checkpoints_stage1_backup/
- ⏳ Stage 2 training: IN PROGRESS on 4090. 200K total steps, resuming from step 45K, all 5 instrument types (sine/sawtooth/square/FM2op/FM4op), pitch-only conditioning. ~8-9 hrs.
- ✅ Phase 4 (partial): ADSR estimator built (adsr_estimator.py, estimator_trainer.py) — CNN on log-mel spectrograms, 4 regression heads, 11 tests passing. Ready to train after Stage 2.
- 🔲 NEXT: Wire CLAP into trainer before Stage 3 (text_emb=None is hardcoded — must fix before NSynth fine-tuning)
- 🔲 Phase 2: Inference pipeline (CueList → WAV)
- 🔲 Phase 3: Real instrument fine-tuning (NSynth — pitched only, no drums)
- 🔲 Phase 5: Pseudo-label + dataset validation
- 🔲 Phase 6: Diffusion upgrade (Option B, ETTA backbone) — needed for drums/foley/percussion

**Training setup:**
- Server: cmn17, /scratch/summerk/timeCode, tmux session "train2"
- Train on 4090 (CUDA), NOT M1 Metal — MPS has incomplete op support for torch.stft and torch.fft.rfft
- CUDA PyTorch: `uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`
- torchaudio 2.5+ breaks save/load — use soundfile directly (already fixed in synthetic_gen.py)
- Stage 2 data: 500K clips, all 5 instrument types, in data/synthetic/
- After Stage 2: run run_train_estimator.py, then wire CLAP, then Stage 3 on NSynth
- NSynth: pitched instruments only (piano, strings, brass, guitar, etc.) — no drums/percussion

**CLAP integration (TODO before Stage 3):**
- CLAPTextEncoder already written in text_encoder.py
- Trainer hardcodes text_emb=None — needs to be wired up
- Need: text dropout (30%), CLAP embeddings passed through SpectralPredictor, text_dim=512 in config

**Why:** Motivated by the problem that generative audio can't align to post-production timecode. Intended for film/TV/game audio use cases. BPM mode also supports music production workflows.
