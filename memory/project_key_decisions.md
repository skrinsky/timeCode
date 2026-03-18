---
name: project_key_decisions
description: Key architectural and design decisions made during planning — rationale included so future sessions don't re-litigate settled questions
type: project
---

**ADSR controls two separate things (not one):**
- Amplitude envelope → analytic gate g(t), guaranteed by construction
- Spectral evolution → stage_position(t) drives interpolation between params_peak and params_sustain
- Real sounds change timbre across ADSR stages (piano gets darker during decay, brass settles from attack transient). Applying a scalar gate uniformly across all harmonics is wrong.

**Release timbral evolution stays at sustain spectrum:**
stage_position holds at 0.0 during release — does not revert to attack character. Real instruments (piano, strings, drums) all get darker or stay the same during release, never brighter. Confirmed through discussion of piano escapement mechanics.

**No human annotators needed for training data:**
ADSR estimator is pre-trained on synthetic clips (known ADSR), then applied directly to unlabeled real audio. Domain gap is primarily timbral not envelope-geometric — estimator should generalize. Human evaluation gate (~100-200 clips, binary pass/fail) used before scaling, not for training.

**Option B injection: text via AdaLayerNorm (frozen), ADSR via cross-attention (trained):**
Using AdaLayerNorm for both would fuse signals and let the model ignore ADSR. Separate pathways prevent the frozen backbone's text prior from overriding ADSR conditioning.

**Frozen backbone decision tree:**
Start fully frozen. Run ADSR sensitivity test at 50K steps. Unfreeze last 4 layers only if ATE > 100ms. Use discriminative LR (1e-5 for backbone vs 1e-4 for adapter) + L2-to-init regularization. Hard ceiling: never unfreeze > 12 of 24 layers.

**ADSR inference from text (two-mode system):**
All ADSR fields are optional. If null, resolved via: articulation lookup → instrument-class default → global default. Explicit values always override. Articulation terms (staccato etc.) also flow to text encoder for timbral character independently. Resolution is logged per-cue in output report.

**Non-physical envelopes are a feature:**
The system places no restrictions on ADSR values relative to instrument class. Sustained snare, slow-attack piano, instant strings — all valid. This is the core creative value: decoupling timbre from envelope shape.

**ADSR-guided CFG for Option B:**
Applied inside denoising loop at each step on the flow field — not post-generation. Two forward passes per step (null ADSR and full ADSR). Can be limited to first 50% of steps to reduce compute cost.

**Linear vs exponential envelopes:**
V1 uses piecewise-linear. Real synths use exponential. Deliberate simplification — swap in envelope.py without affecting anything else if linear sounds mechanical during evaluation.

**Data licensing:**
FreeSound and BBC SFX have complex per-file licensing. CC0 sources preferred. License audit is a pre-Phase 5 checklist gate — do not pseudo-label at scale before auditing.
