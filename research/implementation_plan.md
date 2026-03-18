# ADSR-Conditioned Timecode-Aligned Generative Audio: Implementation Plan

---

## 1. Precise Problem Formulation

### What "ADSR-conditioned generation" means architecturally

The problem is distinct from all prior work in three simultaneous dimensions:

1. **Explicit four-parameter ADSR conditioning.** A, D, S, R are not inferred, not approximated by a loudness curve, and not globally applied. They are discrete named scalars attached to each cue event. The model must understand that Attack=10ms means a sharp onset, Decay=80ms shapes the peak-to-sustain slope, Sustain=0.6 is a gain coefficient held until note-off, and Release=300ms governs the tail. These encode *shape intent* at a specific level of abstraction — they are not the same as "loudness over time."

2. **Per-cue timecode anchoring.** Each event is keyed to a SMPTE-style timecode position (HH:MM:SS:FF) independent of what surrounds it. The timecode is an absolute reference, not a relative offset from clip start. This is a different conditioning modality than PicoAudio's float-second windows.

3. **Sound identity conditioning.** The text description (e.g., "piano note C4") controls the timbral and spectral character of the sound while ADSR controls its envelope. These two pathways must be architecturally separable so the model can generate the same C4 piano note with A=10ms or A=500ms and produce correctly differentiated results.

### Two usage modes

**Easy mode:** Describe the sound in plain language including articulation. The system infers ADSR defaults from the text.
```json
{"timecode": "00:01:23:14", "sound": "staccato piano C4"}
```

**Precise mode:** Specify any or all ADSR fields explicitly. Explicit values always override inferred ones.
```json
{"timecode": "00:01:23:14", "sound": "piano C4", "A": 10, "R": 2000}
```

These mix freely per-parameter. `"staccato piano C4"` with only `R=2000` specified uses inferred A/D/S from "staccato piano" and the explicit `R=2000ms` — useful for "staccato character but with an unusually long tail."

### How text and ADSR interact

Articulation terms (staccato, legato, pizzicato, etc.) sit at the boundary of both pathways:

- **Envelope pathway (ADSR):** Articulation implies certain envelope shapes. These become inferred defaults for any unspecified ADSR fields. Explicit values always win.
- **Timbral pathway (text):** Articulation also affects timbre — a staccato piano has a harder hammer strike and different harmonic character than a legato piano. This information flows through the text encoder regardless of whether ADSR is explicit or inferred.

**Resolution hierarchy (per field):**
```
explicit ADSR value > text-inferred default > instrument-class default > global default
```

A conflict (e.g., `"staccato piano C4"` + `R=2000ms`) is valid — the system generates staccato-style timbre with a long release tail. The text informs *what the sound is*; the explicit ADSR parameters govern *how its envelope behaves*. Making them consistent is the user's responsibility; the system will honor both.

### Arbitrary and non-physical envelopes are a feature

Because timbre and envelope are independent pathways, users are not constrained to envelope shapes that exist in the physical world for a given instrument. This is a first-class creative capability, not a side effect:

```
"piano C4"      + A=2000ms, S=0.9  → slow-attack, fully sustained piano (no real piano does this)
"snare drum"    + S=0.8,   R=2000ms → sustained snare with long tail (physically impossible)
"string section"+ A=0ms,   R=0ms   → instant on/off strings (no bow physics)
"kick drum"     + A=400ms, R=1500ms → pad-like kick envelope
"choir"         + A=5ms,   D=10ms, S=0.0 → percussive choir hit
```

A sample library cannot produce any of these. The system's value is precisely that it decouples *what a sound is* from *how its envelope behaves*. Users can design sounds that don't exist in the real world by specifying any combination of text description and ADSR parameters.

The system places no restrictions on ADSR values relative to instrument class. Unusual combinations are valid inputs. The output may sound strange — that is the user's creative choice.

### What the system must NOT do

This is not a sample player applying a post-hoc envelope to pre-recorded audio. The generative model must produce audio whose waveform naturally reflects the ADSR shape — because post-processing envelope multiplication cannot reproduce how real instruments respond to different attack profiles (a fast-attacked piano differs from a slow-attacked one in harmonic content, not just amplitude).

### Formal input/output specification

```
Input:  List[CueEvent]
        CueEvent = {
          timecode:       SMPTE string "HH:MM:SS:FF",
          sound:          free-text description (e.g., "staccato piano C4"),
          A:              float | null  (null = infer from text),
          D:              float | null  (null = infer from text),
          S:              float | null  (null = infer from text),
          R:              float | null  (null = infer from text),
          velocity:       float | null  (null = infer from text or default 1.0),
          duration_ms:    float | null  (null = infer from next cue or text)
        }

        # Two usage modes — same schema:
        # Easy:    {"timecode": "00:01:23:14", "sound": "staccato piano C4"}
        #          → all ADSR fields inferred from articulation + instrument class
        # Precise: {"timecode": "00:01:23:14", "sound": "piano C4", "A": 10, "R": 2000}
        #          → explicit fields used as-is; null fields inferred
        # Mixed:   articulation terms still inform timbre even when ADSR is explicit
        frame_rate:       SMPTE frame rate (23.976, 24, 25, 29.97, 30)
        output_duration:  SMPTE end timecode or total seconds

        # BPM mode (alternative to per-cue SMPTE timecodes):
        # Provide session-level tempo instead of per-cue SMPTE strings.
        # CueEvent.timecode is replaced by CueEvent.position ("BAR.BEAT.TICK").
        # The BPM resolver converts position → absolute ms → SMPTE → sample offset
        # before the rest of the pipeline. Everything downstream is identical.
        #
        # BPM session header (replaces frame_rate when in BPM mode):
        bpm:              float (e.g., 120.0)
        time_signature:   string (e.g., "4/4")
        ticks_per_beat:   int (default 480, standard MIDI resolution)
        session_start:    SMPTE string — where bar 1 beat 1 lands on the timeline (default "00:00:00:00")
        #
        # BPM mode CueEvent:
        # {"position": "1.1.0",   "sound": "kick drum"}
        # {"position": "1.3.0",   "sound": "snare"}
        # {"position": "2.1.240", "sound": "open hi-hat, R:400"}   ← swung 16th
        #
        # duration_ms can also be expressed as a musical value in BPM mode:
        # {"position": "3.1.0", "sound": "pad", "duration": "2 bars"}
        # → resolver converts "2 bars" → ms at current BPM before generation
        #
        # V1: constant BPM only. V2: tempo map (variable BPM) via list of
        # (position, bpm) breakpoints — resolver integrates the tempo curve.

Output: PCM audio file (WAV/FLAC), 48000 Hz stereo (post-production standard; 44100 Hz supported for non-broadcast use)
        Each CueEvent's sound starts at its timecode offset,
        with its amplitude envelope matching the specified A/D/S/R profile.
```

---

## 2. ADSR vs. RMS Curves: Why the Distinction Matters

This is the architectural fulcrum of the entire design.

### What an RMS curve is

A time-series of amplitude values sampled every 10–100ms. Music ControlNet, T-Foley, and Stable-V2A all use this. It can approximate ADSR shapes but has no semantic structure — it cannot tell you whether a soft start is a slow attack or a faded-in sustain.

### What ADSR parameters are

A parameterized envelope model with exactly 4 scalars (plus implicit note-duration). Each has semantic meaning:

- **Attack** — time from note-on to peak amplitude (governs onset sharpness)
- **Decay** — time from peak to sustain level (governs how the initial transient settles)
- **Sustain** — steady-state amplitude fraction held until note-off
- **Release** — tail duration after note-off

This allows interpolation with physical meaning, generalization to instrument-appropriate shapes, and direct human control with predictable outcomes.

| Dimension | RMS curve | ADSR params |
|-----------|-----------|-------------|
| Dimensionality | T/frame_rate floats | 4 floats |
| Semantic content | None | Each param has physical meaning |
| Distinguishes slow attack from fade-in sustain | No | Yes |
| Compatible with note-off concept | Only by convention | Yes (R is defined relative to note-off) |
| Model must learn to decode | Yes | No — envelope applied analytically; spectral states learned but ADSR drives interpolation schedule, not the network |
| Matches music production vocabulary | Weakly | Exactly |

### Key architectural consequence

In the recommended architecture, the ADSR gate `g(t)` and the spectral interpolation schedule `α(t)` are both computed analytically from the four ADSR scalars — neither is predicted by the neural network. The network learns two spectral states (peak and sustain) and ADSR positions each frame within them. The temporal envelope shape is guaranteed by construction.

---

## 3. Data Strategy

### The core data problem

No dataset exists pairing (sound description, A, D, S, R) with audio where those parameters are ground-truth known. The data strategy must be **constructive** — you build the dataset rather than curate it.

### 3a. Synthetic synthesizer layer (highest-quality ground truth)

Use programmatic synthesis where ADSR parameters are exactly known.

**Tools:** FluidSynth, SuperCollider, Csound, or Python-native (pyaudio + scipy, or music21 + fluidsynth). For each (instrument, note, A, D, S, R):
1. Generate raw oscillator/wavetable signal without envelope
2. Apply ADSR as a piecewise-linear gain multiplier
3. Render to 48000 Hz mono WAV

**Coverage grid:**
- Instruments: sine, sawtooth, square, FM (2-op, 4-op), physical model piano, strings, brass, plucked string
- Pitches: MIDI 21–108 (full piano range)
- A: {0, 1, 5, 10, 20, 50, 100, 200, 500} ms
- D: {0, 5, 20, 50, 100, 200, 500} ms
- S: {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
- R: {0, 10, 50, 100, 200, 500, 1000, 2000} ms
- Note duration: {0.5, 1.0, 2.0, 4.0} seconds
- Velocity: {0.25, 0.5, 0.75, 1.0}

**Constraint:** Enforce A + D < note_duration for every generated clip. If a sampled combination violates this, resample D (or A) until the constraint holds. This ensures the sustain phase always exists and the envelope is fully defined. Interrupted-envelope cases (note_off during attack or decay) are handled by the gate definition above — include a small fraction (~5%) of such cases deliberately for robustness.

Full grid = ~57M combinations (before velocity). Sample ~500K clips via stratified random sampling.

**Text descriptions:** Programmatically generated — "synthesizer sawtooth wave note G#3", "FM synthesis bass note D2", "simple sine tone E5". Cheap, consistent, sufficient for v1.

### 3b. Real instrument estimation layer (NSynth + inverse synthesis)

Use inverse-synthesis tools (InverSynth, DiffMoog) to estimate ADSR from real recordings:
- **NSynth Dataset** (Google Magenta) — 305,979 annotated notes across 1006 instruments. Run inverse synthesis to estimate A, D, S, R per clip. ~20% noise on Attack estimation, ~35% on Decay. Use only for fine-tuning, not as the foundation.
- **URMP Dataset** — clean monophonic instrument stems, good timbral diversity.

### 3c. AF3-based labeling pipeline for complex sounds (drums, foley) — Option B data strategy

Inverse synthesis breaks down for non-pitched sounds (drums, foley, SFX) where the ADSR model doesn't map cleanly to the waveform. The solution is a two-model labeling pipeline inspired by how NVIDIA uses Audio Flamingo 3 to generate synthetic captions for training ETTA.

**The pipeline:**
```
unlabeled audio (foley/drums/SFX libraries)
  → AF3 (off-the-shelf)   → sound_type label ("tight snare, dry room")
  → ADSR estimator        → A, D, S, R values
  → CLAP similarity check → filter low-confidence pairs (threshold ≥ 0.45)
  → labeled training pair
```

**Component 1 — Sound type labels via AF3:**
AF3 (Audio Flamingo 3, NVIDIA ADLR, NeurIPS 2025) is a 7B audio language model that generates rich text descriptions of arbitrary audio. Used off-the-shelf with prompt: *"Briefly describe what you hear in this audio clip."* Generate 5 captions per clip, keep the one with the highest CLAP similarity to the audio. No fine-tuning required for sound type labeling — AF3 already handles this.

**Component 2 — ADSR labels via a learned estimator:**
The ADSR estimator is trained on synthetic data where parameters are exactly known, then applied directly to unlabeled real audio. No human annotation is required for the labeling pipeline.

*Stage A — Synthetic pre-training:*
Train a waveform-to-ADSR regressor on the 500K synthetic clips from Section 3a. Architecture: 1D CNN or lightweight transformer over mel-spectrogram → 4 scalar outputs (A, D, S, R).

**Loss function:** Log-scale MSE for timing parameters (A, D, R), standard MSE for Sustain:
```
loss = MSE(log(A_pred+1), log(A_true+1))
     + MSE(log(D_pred+1), log(D_true+1))
     + MSE(S_pred, S_true)
     + MSE(log(R_pred+1), log(R_true+1))
```
Log-scale is critical for timing params: a 10ms error on a 5ms attack is catastrophic; a 10ms error on a 500ms release is negligible. Linear MSE treats both equally.

**Confidence score via MC dropout:** Add dropout layers (p=0.1) to the estimator, kept active at inference. Run 20 forward passes per clip, use variance across passes as the confidence signal. Requires calibration before use as a threshold — fit a Platt scaler mapping variance → reliability on a held-out synthetic set. Do not use raw variance as a threshold without calibration.

*Why no human seed fine-tune is needed:*
The domain gap between synthetic and real audio is primarily timbral, not envelope-geometric. A piano's fast attack and slow release look structurally similar to a synthetic oscillator's fast attack and slow release — the estimator should generalize. Estimation error will be noisier on real audio than on synthetic, but Option B has two safeguards that tolerate this: confidence filtering and confidence-weighted loss. Systematic bias (not noise) would be the failure mode — mitigated by the evaluation gate in Stage B below.

*Stage B — Small human evaluation set (~100–200 clips, not for training):*
Before running the estimator over 500K–1M clips, validate that it generalizes to real audio. Collect ~100–200 real foley/drum clips and have 3 listeners per clip answer: "Does this ADSR estimate roughly match the envelope you hear?" (yes / partially / no). This is a binary perceptual check, not precise annotation — much lower burden than producing exact values.

If pass rate > 80%: proceed to Stage C.
If pass rate 60–80%: inspect failures by sound class. If failures cluster on specific classes (e.g., cymbals, sustained textures), exclude those classes from pseudo-labeling. Proceed.
If pass rate < 60%: the estimator is not generalizing. Investigate whether systematic bias exists before scaling.

*Stage C — Pseudo-label the large unlabeled dataset:*
Run the estimator over unlabeled audio libraries (FreeSound CC0, BBC SFX where licensed, film foley archives). Keep only predictions above the calibrated confidence threshold. These become pseudo-labeled training pairs.

**Key property:** For sounds where ADSR is semantically ambiguous (crowd noise, sustained texture), the estimator will produce high-variance, low-confidence outputs — these are automatically filtered out. The training data self-selects to sounds where ADSR is a meaningful descriptor.

### 3d. Timbre-transfer augmentation

Take ADSR envelope from one instrument and timbre from another (SynthCloner-style). Multiply dataset diversity without increasing labeling cost. E.g., violin timbre + piano ADSR profile. Used only as augmentation, not primary training data.

**ADSR perturbation augmentation (from Sketch2Sound):** During Option B training, randomly apply small perturbations to conditioning ADSR values (±10–20% of each parameter). This prevents the model from overfitting to exact conditioning values and makes it accept slightly imprecise inputs gracefully at inference — equivalent to the median filter trick Sketch2Sound applies to time-varying control curves.

### 3e. Dataset schema

```json
{
  "audio_path":      "path/to/clip.wav",
  "text_prompt":     "tight snare drum, dry room",
  "A_ms":            5.0,
  "D_ms":            40.0,
  "S_level":         0.1,
  "R_ms":            80.0,
  "pitch_midi":      null,
  "instrument":      "snare",
  "source":          "af3_pseudo_labeled",
  "adsr_confidence": 0.87,
  "adsr_inferred":   ["A", "D", "S"],
  "note_duration":   0.3,
  "sample_rate":     48000
}
```

The `adsr_confidence` field gates whether a pseudo-labeled example is included in training (threshold set per curriculum stage).

### 3f. Dataset sizing targets

| Phase | Source | Clips | Quality |
|-------|--------|-------|---------|
| Option A Phase 1 | Synthetic synths | 500K | Ground truth |
| Option A Phase 2 | NSynth estimated | 200K | Noisy labels |
| Option A Phase 3 | URMP + augmentation | 100K | Mixed |
| Option B eval | Human perceptual check (evaluation only, not training) | 100–200 | Pass/fail per clip |
| Option B large | AF3 + estimator pseudo-labeled | 500K–1M | Filtered pseudo-labels |
| **Option A total** | | **~800K** | |
| **Option B total** | | **~1.5M** | |

### 3g. ADSR inference defaults

For any null ADSR field, the system resolves a default value in order:

**1. Articulation term lookup (parsed from text prompt):**
```
staccato    → duration short, R: 10–50ms
legato      → duration long,  A: (inherit instrument class)
marcato     → velocity: 0.9,  A: fast (inherit instrument class)
tenuto      → duration full,  S: high (inherit instrument class)
pizzicato   → A: 5ms, D: 60ms,  S: 0.0, R: 150ms
sforzando   → velocity: 1.0,  A: 2ms
tremolo     → (flag: duration_ms = one repetition, not note duration)
con sordino → (timbral only — no ADSR implication, passes to text encoder)
```

**2. Instrument-class defaults (from text prompt, if no articulation match):**
```
piano       → A: 5ms,   D: 200ms, S: 0.4,  R: 500ms
strings     → A: 80ms,  D: 100ms, S: 0.8,  R: 400ms
brass       → A: 30ms,  D: 80ms,  S: 0.9,  R: 200ms
plucked     → A: 5ms,   D: 100ms, S: 0.1,  R: 200ms
pad         → A: 400ms, D: 200ms, S: 0.9,  R: 800ms
snare       → A: 2ms,   D: 80ms,  S: 0.0,  R: 50ms
kick        → A: 1ms,   D: 120ms, S: 0.0,  R: 80ms
hihat       → A: 1ms,   D: 40ms,  S: 0.0,  R: 20ms
foley/SFX   → A: 5ms,   D: 150ms, S: 0.2,  R: 200ms
```

**3. Global default (last resort):**
```
A: 10ms, D: 100ms, S: 0.5, R: 300ms, velocity: 1.0
```

These defaults are stored in `core/adsr_defaults.py` as a plain dictionary — easy to audit, extend, and override. The lookup is rule-based for v1; a future version could replace or augment it with an LLM query to AF3.

The per-cue output report (Stage 6 of inference pipeline) logs which fields were inferred vs. explicit, so users can inspect and refine.

---

## 4. Model Architecture

### 4a. Three options and the recommended path

**Option A: ADSR-Extended DDSP** ← Build this first
Extend Google's DDSP with an explicit ADSR conditioning pathway. The envelope gate is analytic (not learned). Fast inference, small model, ADSR precision guaranteed. Limited to pitched/harmonic sounds. Good for v1.

**Option B: Diffusion ControlNet Adapter** ← Upgrade path
Add an ADSR ControlNet adapter to a frozen pretrained T2A backbone (ETTA). Higher perceptual quality, general sound design. ADSR conditioning is learned, not analytic. ~30–50M additional parameters. The lightweight adapter + CFG dropout approach is validated by Sketch2Sound (Adobe Research, ICASSP 2025) and Audio Palette (2024) for analogous time-varying conditioning on DiT backbones — this is an extension of a known-working pattern, not an untested architectural bet.

**Option C: DDSP + Diffusion Polish (hybrid)** ← Do not build first
DDSP for guaranteed envelope, diffusion for timbral richness. Most complex; risk of Stage 2 altering the envelope shape.

**Recommendation:** Option A as foundation. Option B as a separate parallel research track after Option A is validated.

### 4b. Option A architecture in detail

ADSR controls two distinct things that must be handled separately:
1. **Overall amplitude envelope** — handled analytically via the ADSR gate `g(t)`
2. **Timbral/spectral evolution** — handled by the neural network, conditioned on ADSR stage position

Real sounds change timbre across ADSR stages, not just amplitude. A piano's attack has strong high harmonics from the hammer strike; they fall off faster than the fundamental during decay — the sound gets darker as it sustains. Brass has a "brassy" high-harmonic transient at attack that settles during sustain. FM synthesis classically applies ADSR to the modulation index, changing which harmonics exist. Applying a single scalar gate uniformly across all harmonics ignores this and produces unnatural results.

The architecture captures both effects:

```
Inputs:
  text_prompt + pitch_hz → CLAP encoder (frozen) + sinusoidal pitch embed
                         → two spectral state predictions (via shared MLP trunk):
    params_peak    [N_harmonics + noise]  # harmonic profile at attack peak
    params_sustain [N_harmonics + noise]  # harmonic profile during sustain

  (A, D, S, R, dur) → ADSREncoder → stage_position(t) per frame
    # stage_position(t): continuous value in [0, 1] indicating where in the
    # ADSR lifecycle each synthesis frame falls.
    # 1.0 = at attack peak, 0.0 = full sustain, interpolated through D.
    # Holds at 0.0 through release — timbral character stays at sustain during the tail.
    # (Real instruments don't revert to attack spectrum during release — piano release
    # is darker than sustain, not brighter. sustain spectrum held through R is correct.)

Per-frame synthesis (DDSP frame rate, ~every 64 samples):
  # 1. Spectral interpolation — ADSR shapes harmonic content each frame
  α(t) = stage_position(t)
  frame_harmonics(t) = α(t) * params_peak + (1 - α(t)) * params_sustain
  frame_noise(t)     = α(t) * params_peak_noise + (1 - α(t)) * params_sustain_noise

  # 2. Amplitude gate — analytic, ADSR shapes overall loudness
  ADSR gate g(t):
    # Edge cases: A=0, D=0, R=0 are valid (e.g., percussive instant attack)
    # When a stage duration is 0, skip that stage — do not divide by zero
    # Interrupted envelope: if note_off occurs before A+D completes,
    # begin release from whatever g(note_off) was at.
    if A_ms == 0: peak at t=0 (g(0) = 1)
    else:         t ∈ [0, A_ms):        g(t) = t / A_ms
    if D_ms == 0: g = 1 during sustain
    else:         t ∈ [A_ms, A+D):      g(t) = 1 - (1-S)*(t-A)/D
                  t ∈ [A+D, note_off):  g(t) = S
    if R_ms == 0: g = 0 immediately at note_off
    else:         t ∈ [note_off, note_off+R): g(t) = g(note_off) * (1 - (t-note_off)/R)
    t ≥ note_off + R: g(t) = 0

  # 3. Final output per frame (velocity applied once here — not again in inference pipeline)
  output(t) = DDSP_synth(frame_harmonics(t), f0) * g(t) * velocity
            + FilteredNoise(frame_noise(t)) * g(t) * velocity

  # Output is mono. For stereo output: duplicate to both channels (Option A).
  # True stereo generation arrives with Option B (ETTA outputs stereo natively).
```

**What the neural network learns:** How the harmonic profile of a sound described by `text_prompt` at pitch `pitch_hz` differs between its attack peak and its sustained body. This is a genuine timbral fact about each instrument class.

**What is analytic (not learned):** The amplitude envelope shape `g(t)` and the interpolation schedule `α(t)` — both computed directly from the four ADSR scalars.

**The ADSREncoder's precise role:** It does not feed into a static conditioning blob. Its output is `stage_position(t)` — a per-frame scalar that tells the synthesis network where in the ADSR lifecycle each frame sits. The network uses this to interpolate between the two learned spectral states. This is the principled version of ADSR influencing harmonic content.

### 4c. Option B architecture in detail

Option B extends the system to complex sounds (drums, foley, SFX) where DDSP's harmonic-plus-noise model is insufficient. The backbone is a pretrained T2A diffusion model (ETTA is the natural candidate given its design space analysis, but AudioLDM2 or Stable Audio also work). The ADSR conditioning is learned, not analytic — which requires both a stronger data strategy (Section 3c) and explicit architectural enforcement.

```
Inputs:
  text_prompt       → CLAP or T5 text encoder (frozen)   → timbre_embedding [512–1024]
  # Note: AF3 is an audio-in model (processes audio, not text) — not used here.
  # ETTA uses T5 for text conditioning; CLAP is an alternative if music-text alignment matters.
  (A, D, S, R, dur) → ADSREncoder (2-layer MLP)          → adsr_vector [128]

Backbone:
  Frozen pretrained T2A DiT (e.g., ETTA-DiT, 1.29B params)
  — weights frozen, only the ControlNet adapter trains

ADSR ControlNet adapter (~40M params, trains from scratch):
  adsr_vector → project to DiT hidden dim → adsr_kv (key/value for cross-attention)

  In each DiT transformer block:
    x = AdaLayerNorm(x, text_embedding)        # existing text pathway — weights frozen, unchanged
    x = x + CrossAttention(x, adsr_kv)         # separate ADSR pathway — additive, trains from scratch

  Why separate pathways:
  - AdaLayerNorm for both signals would fuse them before the transformer sees them,
    letting the model learn to weight text heavily and ADSR lightly (path of least resistance
    given the pretrained backbone's text prior). Additive cross-attention cannot be
    "overridden" the same way — it contributes independently at every block.
  - The pretrained backbone's text conditioning is completely undisturbed (frozen AdaLayerNorm).
  - ADSR guidance scale at inference has a cleaner effect when the pathway is separate.
  - Gradients for ADSR and text flow through distinct paths during training.

Training losses (Option B):
  - Standard diffusion loss (flow matching objective, matching ETTA's OT-CFM)
  - Confidence-weighted loss:
      Scale each training example's loss by adsr_confidence score from the pseudo-label pipeline.
      Low-confidence pseudo-labels contribute less to gradient updates.

  Note: No ADSR inverse-prediction auxiliary loss. Sketch2Sound and Audio Palette both show that
  a lightweight adapter + CFG dropout is sufficient to prevent the text prior from overriding
  explicit conditioning — no auxiliary loss through the denoising loop is needed (and it would
  be non-differentiable through the full OT-CFM trajectory anyway). The ADSR-guided CFG at
  inference is the primary mechanism for enforcing conditioning strength.

ADSR-guided classifier-free guidance at inference (novel):
  Applied inside the denoising loop at each diffusion step, not post-generation.
  At each step t, predict the score/flow field twice:
    v_null = model(x_t, t, text=prompt, adsr=null_adsr)
    v_full = model(x_t, t, text=prompt, adsr=input_adsr)
    v_guided = v_null + adsr_guidance_scale × (v_full - v_null)
  Use v_guided to take the denoising step.
  This is analogous to text CFG but applied to the ADSR conditioning signal,
  and must operate on the flow field at each step — not on the final decoded audio.
  adsr_guidance_scale is a separate hyperparameter from text guidance scale (default: 3.0).
  Cost: doubles inference compute (two forward passes per step). Can be mitigated by
  applying ADSR guidance only on the first half of denoising steps where structure forms.
```

**Why ETTA as the backbone:**
ETTA's design space analysis (the "elucidated" paper) means its architecture choices are individually validated — you know what each component does and why. The RoPE positional embeddings and OT-CFM training objective are compatible with the per-step AdaLayerNorm injection used by the ControlNet adapter. ETTA also outputs 44.1 kHz stereo via its VAE, which matches the post-production target format.

**Frozen vs. partially unfrozen backbone — decision tree:**

Start fully frozen. Only unfreeze if the ADSR sensitivity test fails. The goal is to do the minimum unfreezing necessary — the backbone's pretrained weights are expensive to recover if degraded.

```
Step 1: Train fully frozen for 50K steps.

Step 2: ADSR sensitivity test.
  Generate the same prompt with A=10ms vs A=500ms.
  → ATE < 60ms:   adapter is working. Stay frozen. Continue training.
  → ATE > 100ms:  adapter alone is insufficient. Proceed to Step 3.

Step 3: Unfreeze last 4 layers (layers 21–24 of 24).
  Use discriminative learning rates — smaller for backbone layers
  to limit how much they drift from their pretrained values:
    Adapter:               lr = 1e-4  (unchanged)
    Unfrozen layers 21–24: lr = 1e-5  (10× lower)
  Add L2-to-init regularization on the unfrozen layers:
    loss += λ * ||current_weights - pretrained_weights||²
    (λ = 1e-3 as starting point)
  This keeps unfrozen layers close to their pretrained position
  while still allowing small ADSR-relevant adjustments.

Step 4: Train for another 50K steps.

Step 5: Two-check gate.
  a. ADSR sensitivity test:   is ATE now < 100ms?
  b. Generation quality check: has FAD on held-out general audio degraded > 20%?

  → ADSR passing + FAD stable:   done.
  → ADSR still failing:           unfreeze layers 17–24 (last 8). Repeat from Step 4.
  → FAD degrading:                reduce unfrozen lr to 5e-6, increase λ to 3e-3. Repeat.

Hard ceiling: never unfreeze more than 12 of 24 layers.
Early layers encode low-level acoustic features that the entire backbone depends on.
Unfreezing them risks degrading general generation quality irreversibly.
```

**Why this works:** Transformer layers do different jobs at different depths. Early layers extract low-level acoustic structure (frequencies, temporal patterns). Later layers handle high-level semantics ("what is this sound's character"). ADSR is a high-level semantic concept, so the last few layers are where unfreezing has the most effect with the least risk. The L2-to-init term acts as a leash — the unfrozen layers can adjust, but only within a bounded radius of their pretrained values.

---

## 5. Timecode Handling

### 5a. SMPTE to sample offset conversion

```python
# For non-drop-frame:
total_frames = HH*3600*fps + MM*60*fps + SS*fps + FF
sample_offset = total_frames * (sample_rate / fps)

# For 29.97 drop-frame: use a dedicated SMPTE library
# (never implement drop-frame arithmetic from scratch)
```

At 30fps / 48000 Hz: one frame = 1600 samples = 33.3ms.

**Supported frame rates:** 23.976, 24, 25, 29.97 ND, 29.97 DF, 30.

**Drop-frame risk:** 29.97 DF is the dominant broadcast standard. Its frame-dropping rule (skip frames 0 and 1 of every minute except every 10th minute) is the single highest-risk correctness-critical step. One-frame error = 33.4ms — just above the human A/V sync threshold. Use a battle-tested library (`timecode`, `smpte` on PyPI). Write unit tests for 00:01:00:00 DF, 00:10:00:00 DF, 01:00:00:00 DF specifically.

### 5b. BPM mode resolver

When the session header contains `bpm` instead of `frame_rate`, cue positions are expressed as `"BAR.BEAT.TICK"` strings. The resolver converts these to absolute milliseconds before the SMPTE conversion step:

```python
beat_duration_ms = 60000.0 / bpm
ticks_per_bar    = ticks_per_beat * beats_per_bar   # e.g., 480 * 4 = 1920
total_ticks      = (bar - 1) * ticks_per_bar + (beat - 1) * ticks_per_beat + tick
absolute_ms      = session_start_ms + total_ticks * (beat_duration_ms / ticks_per_beat)
```

The resulting `absolute_ms` is then converted to a SMPTE timecode using the session `frame_rate` (default 30fps if not specified in BPM mode). Everything downstream — the SMPTE→sample conversion, generation, timeline assembly — is identical to SMPTE mode. The model never sees BPM.

ADSR parameters remain in milliseconds regardless of BPM. A snare's 2ms attack is 2ms at any tempo.

Musical duration values in BPM mode (`"2 bars"`, `"quarter note"`, `"8th note"`) are converted to milliseconds by the same resolver before generation.

### 5c. Timecode is a scheduling parameter, not a model input

The generative model does not need to know the timecode position. It generates a waveform of the right shape and duration. The timecode only determines *where in the output buffer* to write that waveform. The separation is clean.

### 5d. Note duration inference

Three strategies (explicit takes precedence):
1. **Explicit:** `duration_ms` field in CueEvent
2. **Next-cue heuristic:** gap to next cue onset minus R_ms
3. **Text-inferred:** parse "short"/"long"/"sustained" from text prompt → map to fixed durations

### 5e. Overlap handling

Multiple cues may overlap in time. Generate each independently, sum (mix) into the output buffer. Apply headroom normalization if the buffer clips. Standard audio mixing — not a generation problem.

---

## 6. Training Procedure

### 6a. Loss functions (Option A)

- **Multi-scale spectral loss (MSS):** L1 on STFT magnitudes at FFT sizes {64, 128, 256, 512, 1024, 2048}. Primary loss. Captures timbral quality across frequency scales.
- **Envelope reconstruction loss:** Removed. Since the ADSR gate is analytic, the output envelope is the ADSR envelope by construction — this loss is trivially satisfied and contributes no learning signal. If synthesis quality is poor (near-zero output), the MSS loss catches it.

**Optimizer:** AdamW, lr=1e-4, cosine decay, weight decay=1e-5. Batch size: 64 clips (2–4s each). ~500K steps on a single A100.

### 6b. Training curriculum

| Stage | Steps | Data | Key change |
|-------|-------|------|------------|
| 1 | 0–50K | Sine + sawtooth only | No text conditioning; verify backbone works |
| 2 | 50K–200K | All synthetic instruments | Add text with 30% null dropout (CFG training) |
| 3 | 200K–400K | 70% synthetic + 30% NSynth estimated | ADSR lr × 0.3; acquire real timbres |
| 4 | 400K+ | + augmentation, upsample edge cases | Extreme ADSR combinations (A<5ms, A>400ms, S=0) |

### 6c. Classifier-free guidance

Text prompt uses 30% null dropout during training. At inference, guidance scale 2.0–4.0:
```
output = model(text=null) + scale × (model(text=prompt) - model(text=null))
```
ADSR conditioning is **never** subject to classifier-free guidance in Option A — it is always applied at full strength via the analytic gate. In Option B, ADSR has its own separate guidance mechanism (see Section 4c).

---

## 7. Evaluation Metrics

### 7a. ADSR reconstruction accuracy

Extract the amplitude envelope from generated audio, fit best ADSR model, compare to conditioning input.

| Metric | Definition | Target |
|--------|------------|--------|
| **ATE** — Attack Time Error | \|fitted_A - target_A\| in ms | Median < 15ms |
| **DTE** — Decay Time Error | \|fitted_D - target_D\| in ms | Median < 30ms |
| **SLE** — Sustain Level Error | \|fitted_S - target_S\| (ratio) | Median < 0.05 |
| **RTE** — Release Time Error | \|fitted_R - target_R\| in ms | Median < 50ms |

### 7b. Timecode alignment accuracy

| Metric | Definition | Target |
|--------|------------|--------|
| **TAE** — Timecode Alignment Error | \|detected_onset_sample - intended_sample\| in ms | < 10ms (sub-frame at 30fps) |

TAE measurement depends on attack time:
- **Fast attack (A < 20ms):** Use `librosa.onset.onset_detect` on the output file. Sharp transient is detectable.
- **Slow attack (A ≥ 20ms):** Onset detection is unreliable — no sharp transient exists. Instead measure the sample at which the envelope first crosses 5% of peak amplitude and compare to intended onset. Report separately as TAE-slow.

Both variants should be reported. TAE on slow-attack sounds is a scheduling accuracy metric, not a perceptual sync metric (a 20ms offset on a 500ms attack is imperceptible).

### 7c. Perceptual quality

- **CLAP similarity:** cosine similarity between text prompt and generated audio in CLAP embedding space. Target > 0.35.
- **FAD (Fréchet Audio Distance):** distribution-level comparison vs. ground truth clips. Report per instrument family.
- **MOS (Mean Opinion Score):** human evaluation, 500-clip held-out set. Two axes: (1) "Does this sound like [text prompt]?", (2) "Does the shape match the described envelope?" Target > 3.5/5.0 on both.

### 7d. Option A vs Option B metric targets

ADSR metric targets differ because Option A's envelope is analytic (guaranteed) while Option B's is learned (approximate):

| Metric | Option A target | Option B target |
|--------|----------------|----------------|
| ATE | Median < 15ms | Median < 40ms |
| DTE | Median < 30ms | Median < 60ms |
| SLE | Median < 0.05 | Median < 0.10 |
| RTE | Median < 50ms | Median < 80ms |

Option B targets are looser but still require meaningful conditioning — if Option B's ATE approaches Option A's values, that is a strong result worth reporting.

### 7e. Required ablations

- **ADSR-sensitivity test:** Same prompt, A=10ms vs. A=500ms. Measure that ATE reflects the 490ms difference. If outputs are similar regardless of A, conditioning has failed.
- **Cross-instrument ADSR transfer:** Same (A, D, S, R) applied to piano, strings, percussion. CLAP scores should reflect different timbres; ADSR metrics should be similarly accurate across all three.
- **Velocity sensitivity:** Same prompt and ADSR, velocity=0.25 vs velocity=1.0. Output peak amplitude should scale proportionally.

---

## 8. Inference Pipeline

```
Stage 1: Input parsing
  - Parse JSON/YAML cue list
  - Validate each CueEvent (timecode or position format, A/D/S/R ranges)
  - Detect session mode: SMPTE mode (frame_rate present) or BPM mode (bpm present)
  - Parse global frame_rate / bpm / time_signature / session_start and output_duration

Stage 2: Timecode resolution
  - BPM mode only: for each cue, bpm_resolver converts BAR.BEAT.TICK → absolute ms → SMPTE timecode
  - For each cue: SMPTE timecode → sample offset
  - Sort cues by sample offset

Stage 2.5: ADSR inference (for null fields only)
  - Parse articulation terms from text_prompt (staccato, legato, pizzicato, etc.)
  - For each null ADSR field: articulation lookup → instrument-class default → global default
  - Resolve note_duration: explicit > next-cue gap heuristic > text-inferred > instrument default
    (Last cue has no next-cue gap: skip that step and fall through to instrument default)
  - Log inferred vs. explicit fields per cue (included in output report)
  - No null fields remain after this stage — all values are resolved before generation

Stage 3: Per-cue audio generation (parallelizable)
  - Cache text embeddings: if multiple cues share the same text_prompt, encode once and reuse
    (Reusing the same timbre_embedding for different ADSR values is correct — timbre is text-driven,
     ADSR enters through a separate pathway. This is not a bug.)
  - For each cue:
    a. text_prompt → CLAP encoder (or cache hit) → timbre_embedding
    b. (A, D, S, R, velocity, note_duration) → ADSREncoder → adsr_vector
    c. Compute analytic ADSR gate g(t) (Option A) OR run diffusion denoising loop (Option B)
       (velocity already applied inside model — do not apply again here)
    d. → clip_waveform [N_samples]
  - Per-cue random seed: each CueEvent may carry an optional seed int;
    if absent, derive deterministically from (timecode, sound, A, D, S, R, velocity)
    so results are reproducible without explicit seeds
  - Batch multiple cues through model if GPU memory allows

Stage 4: Timeline assembly
  - Allocate output buffer: zeros [total_output_samples × 2]
  - For each cue: write clip_waveform into buffer at cue's sample_offset
  - Where clips overlap: sum (mix)

Stage 5: Post-processing
  - Check for clipping; apply headroom limiter if needed
  - Optional dithering before quantization
  - Write WAV/FLAC at 24-bit (post-production standard)

Stage 6: Output
  - Return audio file
  - Return per-cue report: onset accuracy, duration accuracy
```

### Latency targets

**Option A (DDSP):**
- Per-cue generation: ~10–50ms per 2s clip on A100
- 20-cue list → 10-min output: < 2 seconds total

**Option B (ETTA DiT, 1.29B params):**
- Per-cue generation: ~2–5 seconds per clip (multiple diffusion steps)
- With ADSR-guided CFG (2× forward passes per step): ~4–10 seconds per clip
- 20-cue list: 80–200 seconds on a single A100
- Mitigation: apply ADSR guidance only on first 50% of denoising steps; batch cues; use fewer steps (25 vs 50) with quality tradeoff

Both options are offline pre-production use. Real-time streaming is out of scope.

---

## 9. Open Questions and Risks

### Risk: ADSR semantics break for non-musical sounds

ADSR originated in subtractive synthesis. "Explosion", "thunder", "crowd cheer" have semantically ambiguous decay/release stages. **Mitigation for v1 (Option A):** restrict scope to musical instrument notes and short discrete sound effects. **Mitigation for Option B:** the AF3 + estimator pipeline filters out low-confidence pseudo-labels — sounds where ADSR is semantically ambiguous will produce high estimator variance and get dropped. This is a feature, not a bug: the training data self-selects to sounds where ADSR is a meaningful descriptor.

### Risk: CLAP encoder doesn't generalize to instrument+note descriptions

"Piano note C4" is more specific than typical CLAP training data. Timbre embeddings may not reliably distinguish C4 from C5, or "muted trumpet" from "open trumpet." **Mitigation:** fine-tune CLAP text encoder (or use a music-text contrastive model trained on instrument descriptions) before freezing. Note: MERT is an audio encoder only — it cannot encode text prompts and is not applicable here.

### Risk: Envelope extraction for evaluation is unreliable on noisy sounds

ADSR fitting is ambiguous for percussive or inharmonic timbres. **Mitigation:** use synthetic test data (perfect ground truth) as the primary ADSR metric evaluation set. Report "ADSR metric applicability" as the fraction of clips where fitting converged.

### Risk: Drop-frame timecode arithmetic errors

29.97 DF is dominant in post-production. One-frame error = 33.4ms. **Mitigation:** dedicated SMPTE library, never hand-roll drop-frame arithmetic, unit tests for known edge-case timecodes.

Note: 23.976 fps (= 24000/1001) has the same rational-arithmetic requirement as 29.97. The non-drop-frame formula using integer fps will accumulate error at this frame rate. The SMPTE library must handle both correctly.

### Risk: Data licensing for pseudo-label sources

FreeSound clips carry individual per-file licenses (CC0, CC-BY, CC-BY-NC, etc.). BBC SFX has restricted commercial use. Before running the pseudo-label pipeline over these at scale, audit which license tiers are permissible for the intended use (research vs. commercial). Prefer CC0 sources for training data with no downstream restrictions. **Pre-Phase 5 checklist item.**

### Risk: Linear envelope curves sound mechanical

The plan specifies piecewise-linear ADSR throughout. Real synthesizers typically use exponential curves for A, D, and R — linear ramps can sound unnatural, especially for release tails longer than ~200ms. This is a deliberate v1 simplification. If linear envelopes sound mechanical during evaluation, replacing them with exponential curves is a straightforward swap in `envelope.py` without affecting any other component. Document the choice explicitly so it is not mistaken for an architectural constraint.

### Risk (Option B): Diffusion model ignores ADSR conditioning

Strong text priors may override ADSR conditioning when the model has already "seen" what a piano sounds like. **Mitigation:** ADSR-guided CFG at inference (primary mechanism) + ADSR perturbation augmentation during training (from Sketch2Sound). Sketch2Sound and Audio Palette both demonstrate that a lightweight adapter + CFG dropout on a frozen DiT backbone is sufficient to enforce explicit conditioning without an auxiliary loss.

### V2 path: inference-time ADSR guidance without fine-tuning (DITTO)

**DITTO** (Novack et al., ICML 2024) optimizes the initial noise latent at inference by backpropagating through the denoising loop toward any differentiable loss target — no model retraining needed. Applied to this system: define a differentiable envelope matching loss between generated audio and the target ADSR curve, then use DITTO to steer the noise latent per cue. DITTO-2 (ISMIR 2024) makes this 10–20× faster. This would allow ADSR-guided generation on top of an unmodified ETTA backbone with no adapter training — useful as a no-training baseline or for rapid prototyping of new ADSR conditioning strategies.

### Known limitation: ADSR is too coarse for nuanced sound design

Real acoustic instruments don't have four cleanly separable envelope stages. A piano struck hard differs from one struck soft in harmonic content, not just amplitude. Multi-breakpoint envelopes, per-partial ADSR, or velocity-sensitive curves are natural extensions for v2+.

### Known limitation: Polyphonic cues at the same timecode

Three CueEvents at the same timecode (e.g., a chord) produce three independently generated sounds that are summed. This misses sympathetic resonance and pedal effects. Acknowledged v1 limitation; document explicitly.

---

## 10. Implementation Phases

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 0: Infrastructure | 2 weeks | SMPTE parser + unit tests (incl. 23.976 and 29.97 DF edge cases), CueEvent schema with optional ADSR fields, adsr_defaults.py lookup table, adsr_inferer.py, BPM resolver (constant BPM + musical duration values), evaluation harness, data generation scripts with A+D < note_duration constraint |
| 1: DDSP Baseline (Option A) | 4 weeks | Working ADSR-DDSP model, trained on 500K synthetic clips, Option A ATE/DTE/SLE/RTE targets passing |
| 2: Inference Pipeline | 2 weeks | CueList → WAV pipeline end-to-end, per-cue seed, text embedding cache, 20-cue example working |
| 3: Real Instrument Fine-tuning | 3 weeks | NSynth fine-tuning, CLAP + FAD evaluation, MOS study |
| 4: ADSR Estimator + AF3 Labeling Pipeline | 3 weeks | Waveform-to-ADSR regressor trained on synthetic with log-scale MSE; MC dropout confidence with Platt calibration; AF3 sound type labels working; **data license audit completed**; human perceptual evaluation gate (~100–200 clips, pass/fail only) run before scaling |
| 5: Pseudo-label + Dataset Validation | 3 weeks | AF3 + estimator pipeline running over licensed unlabeled foley/drum libraries; confidence filtering; ~500K–1M labeled pairs; estimator pass rate > 80% confirmed before scaling |
| 6: Diffusion Upgrade (Option B) | 6 weeks | ETTA backbone + ADSR ControlNet adapter; ADSR-guided CFG inside denoising loop; Option B metric targets; comparative evaluation vs. Option A |

---

## 11. File Structure

```
timecode_audio/
  core/
    timecode.py          # SMPTE parse, drop-frame arithmetic, sample offset conversion
    bpm_resolver.py      # BAR.BEAT.TICK + BPM/time_sig → absolute ms → SMPTE; musical duration values → ms
    cue_event.py         # Pydantic CueEvent schema and validation (all ADSR fields optional)
    envelope.py          # Analytic piecewise-linear ADSR gate (no ML)
    adsr_defaults.py     # Articulation term lookup + instrument-class defaults + global defaults
    adsr_inferer.py      # Resolves null ADSR fields from text prompt using adsr_defaults
  model/
    adsr_encoder.py      # ADSR scalars → stage_position(t) per frame; also projects adsr_vector for Option B
    adsr_estimator.py    # Waveform-to-ADSR regressor: mel-spectrogram → (A, D, S, R); used for pseudo-labeling
    text_encoder.py      # CLAP/T5 wrapper (frozen weights); note: CLAP trained at 48kHz — resample inputs
    ddsp_synthesizer.py  # Option A: ADSR-extended DDSP backbone (operates at 16kHz internally — upsample output)
    etta_controlnet.py   # Option B: ETTA-DiT backbone + ADSR cross-attention ControlNet adapter
    losses.py            # MSS loss (Option A); OT-CFM + confidence-weighted (Option B)
  data/
    synthetic_gen.py     # Programmatic synth data generation
    nsynth_loader.py     # NSynth with estimated ADSR labels
    augmentation.py      # Timbre-transfer augmentation
    af3_labeler.py       # AF3 sound type caption generation + CLAP confidence filtering
    pseudo_label.py      # Full pipeline: unlabeled audio → (text_prompt, A, D, S, R, confidence)
  training/
    trainer.py           # Training loop, curriculum scheduler
    config.py            # Hyperparameters, curriculum stages
  inference/
    pipeline.py          # Full cue list → audio file pipeline
    mixer.py             # Timeline assembly, overlap summing, normalization
  eval/
    adsr_metrics.py           # ATE, DTE, SLE, RTE (separate targets for Option A vs B)
    timecode_metrics.py       # TAE (fast-attack via onset detection; slow-attack via 5% threshold)
    perceptual.py             # CLAP similarity, FAD
    adsr_estimator_eval.py    # Standalone evaluation of the ADSR estimator before pseudo-labeling at scale
  tests/
    test_timecode.py     # Drop-frame arithmetic unit tests (critical)
    test_bpm_resolver.py # BAR.BEAT.TICK conversion, musical durations, tempo edge cases (bar 1 beat 1, swing ticks)
    test_envelope.py     # ADSR gate correctness including zero-value edge cases
    test_adsr_inferer.py # Articulation term parsing, resolution hierarchy, conflict cases
    test_pipeline.py     # End-to-end smoke test: easy mode, precise mode, mixed mode
```

---

## 12. Prior Art Differentiation Summary

This system is differentiated from all prior work on three axes simultaneously:

**The three axes:**
1. Explicit four-parameter ADSR conditioning (not a drawn curve, not inferred, not RMS)
2. Per-cue SMPTE timecode anchoring (not relative timestamps within a single clip)
3. Spectral evolution across ADSR stages (not amplitude-only envelope control)

No existing system covers all three. The cue-list-to-timeline workflow (structured list of timecode + sound description → assembled audio) does not exist as a generative tool — DAWs, QLab, and Wwise all require pre-existing audio assets.

| Prior system | Why it's different |
|---|---|
| MPEG-4 SASL (1999) | Deterministic score-driven synthesis, not generative/neural |
| JASCO (Meta, 2024) | Per-timestamp symbolic conditioning, but vocabulary is chords/rhythm not ADSR; single clip not a cue list |
| Music ControlNet (2023) | Dynamics curve (continuous RMS drawn by user), not four discrete named ADSR parameters; no timecode |
| Sketch2Sound (Adobe, ICASSP 2025) | Time-varying loudness/brightness/pitch curves on a DiT — amplitude-only, no spectral evolution, no timecode, no ADSR parameterization; validates the adapter+CFG approach we use |
| Audio Palette (2024) | Loudness/pitch/timbre conditioning on DiT via multi-scale CFG — no ADSR decomposition, no timecode, no cue list; validates multi-signal CFG approach |
| Audio ControlNet (2026) | Loudness/event onset control, not ADSR decomposition |
| SynthCloner (2025) | ADSR disentanglement and re-application, but clip-level transfer with no timeline/cue list |
| Physics-Driven Diffusion (CVPR 2023) | Conditions on decay rate + frequency from impact physics — not user-specified, not four-parameter, no timecode |
| Stable Audio (Stability AI, ICML 2024) | Explicit scalar timing params (seconds_start, seconds_total) — proves scalars work in audio LDM, but not ADSR and not SMPTE |
| DDSP (2020) | Implicit internally-predicted envelope, not user-specified ADSR at cue positions |
| DITTO (ICML 2024) | Inference-time latent optimization toward any differentiable target — no ADSR parameterization, no timecode; useful as a v2 path for this system |
| DAWs (Pro Tools, Reaper) | Arrange pre-existing audio on a SMPTE timeline — not generative |
| QLab | Trigger pre-existing audio files at timecodes — not generative |
| Wwise/FMOD AHDSR | Event-driven (game state), not timecode-driven; not ML generative |
