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
- ✅ Phase 0: Infrastructure complete (SMPTE parser, BPM resolver, CueEvent schema, ADSR gate/stage_position, ADSR inferer, defaults lookup) — 100 tests passing
- ✅ Phase 1: DDSP baseline built (PitchEncoder, SpectralPredictor, harmonic synth at 48kHz, filtered noise, ADSREncoder, MSS loss, synthetic data generator, training loop) — 121 tests passing
- ⏳ Phase 1 PENDING: vectorize filtered_noise loop before training (see project_pending_tasks.md)
- ⏳ Phase 1 PENDING: push to git ✅ done — pull on 4090 server and run training
- 🔲 Phase 2: Inference pipeline (CueList → WAV)
- 🔲 Phase 3: Real instrument fine-tuning
- 🔲 Phase 4: ADSR estimator + AF3 labeling pipeline
- 🔲 Phase 5: Pseudo-label + dataset validation
- 🔲 Phase 6: Diffusion upgrade (Option B, ETTA backbone)

**Training setup:**
- Train on 4090 (CUDA), NOT M1 Metal — MPS has incomplete op support for torch.stft and torch.fft.rfft
- CUDA PyTorch: `uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`
- Stage 1 curriculum: sine + sawtooth only, pitch-only conditioning (no text), 0–50K steps
- After training: Stage 2 adds all synths + CLAP text encoder

**Why:** Motivated by the problem that generative audio can't align to post-production timecode. Intended for film/TV/game audio use cases. BPM mode also supports music production workflows.
