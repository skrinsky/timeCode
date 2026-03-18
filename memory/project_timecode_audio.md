---
name: timecode_audio_project
description: Overview of the ADSR-conditioned timecode-aligned generative audio project — what it is, the research gap, architecture, and where files live
type: project
---

This project is building a novel generative audio system that maps ADSR envelope parameters (Attack, Decay, Sustain, Release) to SMPTE timecode cue positions to synthesize audio. No existing system does this.

**Why:** Generative audio models can't align to timecode. Existing approaches control when sounds occur (PicoAudio, FreeAudio) or draw amplitude curves (Music ControlNet, T-Foley) but none accept discrete ADSR parameters per timecode cue and use them to shape both amplitude envelope and spectral evolution.

**The gap (verified genuine):** No system maps explicit A/D/S/R as four named values to timecode positions to drive generative synthesis. Closest prior art: MPEG-4 SASL (1999, deterministic only), JASCO (chords/rhythm not ADSR), SynthCloner (ADSR transfer but no timeline), DDSP (implicit envelope, not user-specified).

**Key creative feature:** Timbre and envelope are independent pathways — users can combine any sound description with any ADSR shape, including physically impossible combinations (sustained snare, slow-attack piano, instant strings).

**Two-mode input:**
- Easy: `{"timecode": "00:01:23:14", "sound": "staccato piano C4"}` — ADSR inferred from text
- Precise: add explicit A/D/S/R fields — override any or all inferred defaults

**Architecture:**
- Option A (build first): ADSR-extended DDSP. Analytic ADSR gate controls amplitude; stage_position(t) drives spectral interpolation between two learned states (params_peak, params_sustain). Fast, ADSR-guaranteed, limited to harmonic sounds.
- Option B (upgrade): ETTA-DiT backbone + ADSR cross-attention ControlNet adapter. Text via AdaLayerNorm (frozen), ADSR via cross-attention (trained). Handles drums, foley, complex sounds. Uses AF3 + synthetic-trained ADSR estimator for pseudo-labeling at scale.

**Data strategy:**
- Option A: 500K synthetic clips (known ADSR) + NSynth estimated (200K) + augmentation
- Option B: AF3 off-the-shelf for sound type labels; ADSR estimator (synthetic pre-trained, no human labels needed) for ADSR pseudo-labels; ~100-200 clip human evaluation gate before scaling

**File locations:**
- Research dir: /Users/summerkrinsky/Documents/GitHub/timeCode/research/
- Prior art: research/timecode_aligned_audio.md
- Implementation plan: research/timecode_aligned_audio.md (full, detailed)
- Planned code: timecode_audio/ (not yet built)

**Implementation phases:**
0. Infrastructure (SMPTE parser, CueEvent schema, ADSR inferer, defaults)
1. DDSP baseline (Option A)
2. Inference pipeline (CueList → WAV)
3. Real instrument fine-tuning
4. ADSR estimator + AF3 labeling pipeline
5. Pseudo-label + dataset validation
6. Diffusion upgrade (Option B, ETTA backbone)

**Why:** Motivated by the problem that generative audio can't align to post-production timecode. Intended for film/TV/game audio use cases.
