# Timecode-Aligned Generative Audio: Research Landscape

## Core Problem

Generative audio models struggle to align output to timecode cues because:
- Output duration is non-deterministic
- Models have no native SMPTE/MTC awareness
- Inference latency is variable
- Length control is approximate even when prompted
- Generation is event-level, not parameter-level (no ADSR-at-timestamp control)

---

## 1. ML Papers: Temporally Grounded Audio Generation

### PicoAudio (arXiv:2407.02869, July 2024)
Uses "timestamp captions" as structured input and a timestamp matrix module injecting per-event temporal info into a diffusion model at 40ms resolution. An LLM parses free-form timing commands into the structured format.
- https://arxiv.org/abs/2407.02869

### PicoAudio2 (arXiv:2509.00683, September 2025)
Follow-up fusing coarse free-text + fine-grained timestamp matrix inside a Diffusion Transformer (DiT) using AdaLayerNorm for per-step adaptive guidance.
- https://arxiv.org/abs/2509.00683

### AudioTime Benchmark (arXiv:2407.02857, ICASSP 2025)
Dataset and evaluation benchmark for temporal control. Covers timestamps, duration, frequency, and ordering. Proposes STEAM (Strongly TEmporally-Aligned evaluation Metric). 4×5000 training / 4×500 test samples.
- https://arxiv.org/abs/2407.02857
- https://github.com/zeyuxie29/AudioTime

### FreeAudio (arXiv:2507.08557, ACM MM / ISMIR 2025)
Training-free approach. LLM plans non-overlapping time windows from timing prompts, applies Decoupling and Aggregating Attention Control + Contextual Latent Composition over a pretrained diffusion backbone. First method targeting long-form (multi-minute) timing-controlled T2A.
- https://arxiv.org/abs/2507.08557

### ControlAudio (arXiv:2510.08878, NeurIPS 2025)
Recasts controllable T2A as multi-task: pretrain DiT on text-only, fine-tune on text + timing + phoneme features. Targets both event timestamps and speech intelligibility simultaneously.
- https://arxiv.org/abs/2510.08878

### Audio ControlNet (arXiv:2602.04680, 2026)
ControlNet adapters on a pretrained T2A backbone. Controls loudness, pitch, and event roll at precise time locations. T2A-Editor inserts/removes audio events at user-specified timestamps via text instructions. Only 38M additional parameters.
- https://arxiv.org/abs/2602.04680
- https://github.com/juhayna-zh/AudioControlNet

### Stable Audio / Fast Timing-Conditioned Latent Audio Diffusion (arXiv:2402.04825, Stability AI 2024)
Introduced timing conditioning into mainstream audio diffusion. Conditions on text + start-time/duration embeddings. Foundation that FreeAudio benchmarks against.
- https://arxiv.org/abs/2402.04825

### AudioStory (arXiv:2508.20088, Tencent ARC 2025)
LLM decomposes narrative prompt into temporally ordered sub-tasks with timestamps, emotional tone markers, and character cues. Targets video dubbing, audio continuation, narrative synthesis — each with temporal offset metadata.
- https://arxiv.org/html/2508.20088
- https://github.com/TencentARC/AudioStory

---

## 2. Video-to-Audio with Temporal Synchronization

### TiVA: Time-Aligned Video-to-Audio Generation (ACM MM 2024)
Separates semantic matching from temporal alignment. Predicts an "audio layout" (temporal event structure) from video, uses that layout + semantic embeddings as dual conditions for latent diffusion.
- https://dl.acm.org/doi/10.1145/3664647.3681027
- https://tiva2024.github.io/TiVA.github.io/

### MMAudio (arXiv:2412.15322, CVPR 2025, Sony Research)
State-of-the-art public V2A model. Conditional synchronization module aligns video features at 24fps with audio latents frame-by-frame. Targets sub-25ms accuracy (human perception threshold for A/V misalignment). 157M params, 1.23s inference for 8s clip.
- https://arxiv.org/abs/2412.15322
- https://github.com/hkchengrex/MMAudio

### AV-Link (arXiv:2412.15191, ICCV 2025, Snap Research)
Time-aligned Rotary Position Embedding (RoPE) to align audio and video tokens across the temporal axis within a shared diffusion framework. Handles both V2A and A2V. Outperforms MovieGen's V2A component on sync benchmarks.
- https://arxiv.org/abs/2412.15191
- https://github.com/snap-research/AVLink

---

## 3. MPEG-4 Structured Audio — Historical Precedent

### MPEG-4 Structured Audio (ISO/IEC 14496-3, Part 5, 1999)
Developed at MIT Media Lab by Eric Scheirer under Barry Vercoe. **The closest existing standard to "ADSR parameters mapped to timecode cues."**

- **SAOL** (Structured Audio Orchestra Language): Describes synthesizers (derived from Csound/Music-N). Any synthesis algorithm, including ADSR envelope generators, is expressed here.
- **SASL** (Structured Audio Score Language): Drives SAOL instruments with timestamped parametric events — essentially a timecode-anchored ADSR/synthesis parameter stream. Higher temporal resolution than MIDI without MIDI's bandwidth limits.

The bitstream is made of time-stamped parametric events, each referencing an instrument in the orchestra chunk and providing synthesis parameters directly.
- https://en.wikipedia.org/wiki/MPEG-4_Structured_Audio
- https://sound.media.mit.edu/resources/mpeg4/sa-tech.html
- https://sound.media.mit.edu/resources/mpeg4/audio/general/aes106_4-StructuredAudio.pdf

---

## 4. DAW Plugins / Tools for SMPTE Sync

| Tool | Notes |
|------|-------|
| **TXL Timecode Plug-in** | Generates SMPTE LTC, MTC, Art-Net Timecode from DAW clock. VST3/AU. | https://txl20.com/txl-timecode-plug-in/ |
| **Cubase/Nuendo SMPTE Generator** | Built-in plugin outputting SMPTE LTC to sync external equipment. |
| **Ableton / Showsync M4L device** | Free Max for Live device outputting LTC SMPTE with warp-marker awareness. | https://help.ableton.com/hc/en-us/articles/360010120320-SMPTE-Timecode-FAQ |
| **El-Tee-See Two** | Free web app generating LTC SMPTE audio files. | https://elteesee.pehrhovey.net/ |
| **TimeCode Player / TimeCode Live** | Dedicated SMPTE playback and sync for show contexts. | https://timecodesync.com/player/ |

---

## 5. Game Audio Middleware

### FMOD Studio
Every event has its own timeline with markers and cues. Timeline parameter sheets drive per-frame automation of any synthesis parameter. Not natively SMPTE — designed for non-linear/interactive audio. Has been adapted for theater use in closer-to-timecode workflows.
- https://www.fmod.com/docs/2.03/studio/fmod-studio-concepts.html
- https://www.usitt-sound.org/wp-content/uploads/2020/09/FMOD_Adapted_for_Theater.pdf

### Wwise (Audiokinetic)
Integrates with Unreal Engine's Level Sequencer. AkAudioEvent sections display waveforms inside Sequencer — effectively a timecode-driven cue system. Music sync callbacks (PostTrigger) allow beat/bar-aligned events.
- https://documentation.help/Wwise-UE4-Integration/using__features__sequencer.html
- https://www.audiokinetic.com/en/blog/introducing-the-wwise-authoring-api/

---

## 6. Live Performance / Show Control

### QLab (Figure 53)
Dominant show control software for theater and live events. Accepts LTC and MTC as triggers. Central timecode source drives audio, video, lighting, and automation cues simultaneously. Can also output LTC (Timecode Cues).
- https://qlab.app/docs/v5/networking/using-timecode/
- https://qlab.app/docs/v5/networking/timecode-cues/

### MIDI Show Control (MSC)
Open standard linking show devices. Timecode drives MSC commands triggering cues across audio, lighting, pyrotechnics.
- https://en.wikipedia.org/wiki/Show_control
- https://theatrecrafts.com/pages/home/topics/concerts/controlling-lighting-and-sound-with-timecode

---

## 7. Music Information Retrieval (MIR)

### Audio-to-Score Alignment / Robust Audio Synchronization (ISMIR 2024)
Uses raw audio features for synchronization between recordings and symbolic scores. Applicable to driving a generative synth from a timecoded score.
- https://ismir2024program.ismir.net/poster_8.html

**Note:** The MIR community has extensive beat tracking and onset detection infrastructure but direct integration with generative synthesis remains sparse.

---

## Key Gaps (Open Research Space)

### The Core Gap (Verified Genuine)

**No existing system maps explicit ADSR parameters (Attack, Decay, Sustain, Release as four distinct named values) to specific timecode cue positions to drive generative audio synthesis.**

This gap was stress-tested against the full literature. The closest counterexamples all fall short:

| System | ADSR as discrete params? | Timecode/cue-keyed? | Generative/ML? | Closes the gap? |
|--------|--------------------------|---------------------|-----------------|-----------------|
| MPEG-4 SASL (1999) | Possible via SAOL | Yes (timestamps) | No | No |
| Music ControlNet (2023) | No (dynamics curve) | Partial (clip-relative) | Yes | No |
| Audio ControlNet (2026) | No (loudness/events) | Partial (seconds offset) | Yes | No |
| T-Foley / Stable-V2A | No (RMS envelope) | Partial (video-relative) | Yes | No |
| SynthCloner (2025) | Yes (A, D, S, R) | No (clip-level transfer) | Yes (transfer) | No |
| DDSP (2020) | No (implicit, predicted) | No | Yes | No |
| Wwise/FMOD | Partial (AHDSR) | No (event-driven) | No | No |

### Near-Misses Worth Knowing

**SynthCloner** (arXiv:2509.24286, 2025) — the closest ML paper to ADSR: disentangles synthesizer audio into content, timbre, and ADSR envelope as four separate codec paths. Allows explicit A/D/S/R control over output. But it is a timbre-transfer system — ADSR is a global attribute applied to a single conversion, not per-cue conditioning at timestamped positions. No timeline, no cue list.
- https://arxiv.org/abs/2509.24286

**JASCO / Meta AudioCraft** (arXiv:2406.10970, 2024) — the most architecturally close: accepts timestamped symbolic conditions (chord progressions, drum patterns) keyed to time positions. Local, per-position conditioning of generative synthesis. But the conditioning vocabulary is chords and rhythm, not ADSR parameters.
- https://arxiv.org/abs/2406.10970

**Music ControlNet** (arXiv:2311.07069, IEEE TASLP 2024) — controls dynamics as a time-varying curve. An amplitude envelope drawn over the whole generation — not discrete A/D/S/R values at cue points.
- https://arxiv.org/abs/2311.07069

**T-Foley / Stable-V2A / FOL·AI** — use RMS amplitude envelope as temporal conditioning for foley. Continuous amplitude curve ≠ four-parameter ADSR model.
- T-Foley: https://arxiv.org/abs/2401.09294
- Stable-V2A: https://arxiv.org/html/2412.15023v1

**DDSP** (arXiv:2001.04643, Magenta/Google) — time-varying amplitude + harmonic distributions frame-by-frame. Envelope is internally predicted, not user-specified ADSR values at cue positions.
- https://arxiv.org/abs/2001.04643

**Inverse synthesis tools (InverSynth, DiffMoog, DDX7)** — work in the *opposite* direction: audio in → ADSR parameters out (sound matching). Not generative in the cue-driven sense.
- InverSynth: https://arxiv.org/abs/1812.06349
- DiffMoog: https://arxiv.org/html/2401.12570v1

### Other Gaps

1. **ML papers use floating-point seconds, not SMPTE**: The entire PicoAudio/FreeAudio/ControlAudio family uses second-based timestamps (e.g., 2.4s–5.2s), not frame-rate-relative SMPTE HH:MM:SS:FF. Bridging to SMPTE requires only a frame-rate conversion layer — this is a shallow gap, not a deep one.

2. **Game middleware and ML research don't talk**: FMOD/Wwise have per-frame automation of synthesis parameters (including AHDSR envelopes) but are entirely disconnected from ML generation pipelines. No published system uses PicoAudio-style generation to fill FMOD event timelines.

3. **Generation is event-level, not parameter-level**: Current ML papers control *when a sound event occurs*, not *what the envelope shape parameters of that event are*. The precise boundary: Audio ControlNet and Music ControlNet do loudness-envelope-at-timestamps (an amplitude curve), but not ADSR-as-four-discrete-parameters at timecode cues.
