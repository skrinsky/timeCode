"""
Generate adapter training data from Stable Audio Open.

Generates audio clips with diverse prompts and natural envelopes,
saves them to data/adapter_training/ with a metadata.jsonl index.

Requires:
    - stable-audio-tools: pip install stable-audio-tools
    - HF_TOKEN env var set (model is gated)
    - Accepted Stability AI community license on HuggingFace

Usage:
    python generate_training_data.py
    python generate_training_data.py --n_clips 50000 --out_dir data/adapter_training
"""

import argparse
import json
import math
import os
import random
import sys
import types
from pathlib import Path

# k_diffusion (dep of stable-audio-tools) imports openai-clip which uses
# pkg_resources — removed in Python 3.12+. Shim it before the import chain fires.
try:
    import pkg_resources  # noqa: F401
except ImportError:
    import packaging as _packaging
    _mod = types.ModuleType("pkg_resources")
    _mod.packaging = _packaging
    sys.modules["pkg_resources"] = _mod

import soundfile as sf
import torch
from einops import rearrange

SAMPLE_RATE = 44100

# ---------------------------------------------------------------------------
# Prompt list — designed for maximum envelope diversity
# ---------------------------------------------------------------------------
# Each entry: (text_prompt, seconds_total_range)
# seconds_total_range controls note duration diversity

PROMPTS = [
    # --- Percussive / fast attack ---
    ("snare drum hit, dry, acoustic",              (0.5, 1.0)),
    ("kick drum, tight, punchy",                   (0.5, 1.0)),
    ("hi-hat closed, crisp",                       (0.3, 0.8)),
    ("hi-hat open, sustaining",                    (1.0, 2.0)),
    ("rimshot on snare drum",                      (0.5, 1.0)),
    ("wood block hit",                             (0.3, 0.8)),
    ("clap, dry room",                             (0.5, 1.0)),
    ("tambourine shake",                           (1.0, 2.0)),
    ("triangle hit, ringing",                      (2.0, 4.0)),
    ("cowbell hit",                                (0.5, 1.5)),
    ("tabla drum hit",                             (0.5, 1.5)),
    ("bongo drum hit",                             (0.5, 1.0)),
    ("conga drum hit",                             (0.5, 1.5)),

    # --- Piano / plucked (fast attack, long decay) ---
    ("grand piano note, single key, middle C",     (3.0, 6.0)),
    ("piano note, high register",                  (2.0, 5.0)),
    ("piano note, bass register, low C",           (4.0, 8.0)),
    ("acoustic guitar string pluck, single note",  (2.0, 4.0)),
    ("electric guitar note, clean, single note",   (2.0, 5.0)),
    ("acoustic guitar chord strum",                (3.0, 6.0)),
    ("harp string pluck, single note",             (3.0, 6.0)),
    ("pizzicato violin note",                      (1.0, 3.0)),
    ("pizzicato cello note",                       (1.5, 4.0)),
    ("banjo note, twang",                          (2.0, 4.0)),
    ("mandolin note",                              (1.0, 3.0)),
    ("harpsichord note",                           (2.0, 4.0)),
    ("xylophone note, single bar",                 (1.0, 3.0)),
    ("marimba note, resonant",                     (2.0, 5.0)),
    ("vibraphone note, sustained",                 (3.0, 7.0)),

    # --- Bowed strings (slow attack, sustained) ---
    ("violin playing a single long note, legato",  (4.0, 8.0)),
    ("cello playing a single long note, legato",   (4.0, 8.0)),
    ("viola playing a held note",                  (4.0, 8.0)),
    ("double bass playing a long note",            (4.0, 8.0)),
    ("string quartet chord, sustained",            (5.0, 10.0)),

    # --- Woodwinds (smooth attack, sustained) ---
    ("flute playing a single held note",           (3.0, 8.0)),
    ("clarinet playing a long note",               (3.0, 8.0)),
    ("oboe playing a sustained tone",              (3.0, 8.0)),
    ("bassoon playing a low sustained note",       (3.0, 8.0)),
    ("saxophone, tenor, long note",                (3.0, 8.0)),
    ("saxophone, alto, sustained note",            (3.0, 8.0)),
    ("recorder playing a single note",             (2.0, 6.0)),

    # --- Brass (medium attack, sustained) ---
    ("trumpet playing a single note, sustain",     (3.0, 7.0)),
    ("trombone playing a held note",               (3.0, 8.0)),
    ("French horn playing a sustained note",       (4.0, 8.0)),
    ("tuba playing a low note",                    (3.0, 8.0)),

    # --- Organ / pad (slow attack, long sustain) ---
    ("pipe organ chord, cathedral, sustained",     (6.0, 15.0)),
    ("hammond organ note, full drawbars",          (4.0, 10.0)),
    ("synthesizer pad, slow attack, lush",         (6.0, 15.0)),
    ("ambient synthesizer drone, sustained",       (8.0, 20.0)),
    ("choir sustaining a vowel, ah",               (5.0, 12.0)),
    ("cello section sustained chord",              (6.0, 12.0)),
    ("string orchestra sustained chord",           (6.0, 15.0)),

    # --- Voice / vocal ---
    ("single vocal note, soprano, sustained",      (3.0, 7.0)),
    ("vocal hum, single pitch, sustained",         (3.0, 8.0)),
    ("throat singing, drone",                      (5.0, 15.0)),

    # --- Synthesizer (diverse envelope shapes) ---
    ("synthesizer lead, staccato note",            (0.5, 1.5)),
    ("synthesizer bass note, short",               (0.5, 2.0)),
    ("synthesizer arpeggio, single note",          (1.0, 2.0)),
    ("FM synthesizer bell tone, decaying",         (3.0, 8.0)),
    ("synthesizer pluck, short attack",            (1.0, 4.0)),
    ("moog synthesizer note, smooth",              (3.0, 8.0)),

    # --- Foley / SFX (diverse natural envelopes) ---
    ("door creak, slow open",                      (2.0, 5.0)),
    ("door slam",                                  (0.5, 1.5)),
    ("footsteps on gravel, single step",           (0.5, 1.0)),
    ("footsteps on wood floor, single step",       (0.5, 1.0)),
    ("glass breaking",                             (1.0, 3.0)),
    ("keys jingling",                              (1.0, 3.0)),
    ("coin drop on hard surface",                  (1.0, 2.0)),
    ("pencil tapping on desk",                     (0.5, 1.0)),
    ("stapler click",                              (0.3, 0.8)),
    ("keyboard typing, single keypress",           (0.3, 0.8)),
    ("zipper, slow pull",                          (1.0, 3.0)),
    ("velcro ripping",                             (0.5, 1.5)),
    ("paper crumpling",                            (1.0, 3.0)),
    ("book dropping on table",                     (0.5, 1.5)),

    # --- Nature / environment (slow, sustained) ---
    ("rain on a window, light",                    (5.0, 15.0)),
    ("thunder rumble, distant",                    (3.0, 8.0)),
    ("wind through trees",                         (5.0, 15.0)),
    ("ocean waves, single wave",                   (4.0, 10.0)),
    ("fire crackling",                             (5.0, 15.0)),
    ("stream flowing, water",                      (5.0, 15.0)),
    ("bird chirp, single",                         (0.5, 2.0)),
    ("dog bark",                                   (0.5, 1.5)),
    ("cat meow",                                   (0.5, 2.0)),

    # --- Vehicle / mechanical ---
    ("car engine starting",                        (2.0, 5.0)),
    ("car engine idle, purring",                   (5.0, 15.0)),
    ("car door closing",                           (0.5, 1.5)),
    ("motorcycle engine rev",                      (2.0, 5.0)),
    ("train passing by",                           (5.0, 15.0)),
    ("airplane engine, distant",                   (5.0, 15.0)),
    ("helicopter rotor",                           (5.0, 15.0)),
    ("clock ticking",                              (3.0, 8.0)),
    ("machinery hum, industrial",                  (5.0, 15.0)),
]


def generate_clips(
    model,
    model_config: dict,
    prompts_with_durations: list[tuple[str, float]],
    device: torch.device,
    steps: int = 100,
    cfg_scale: float = 7.0,
) -> list[torch.Tensor]:
    """Generate a batch of clips. Returns list of [2, n_samples] tensors."""
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    sample_rate = model_config["sample_rate"]

    conditioning = [
        {"prompt": prompt, "seconds_start": 0, "seconds_total": dur}
        for prompt, dur in prompts_with_durations
    ]

    # Use the max clip duration in this batch as sample_size — not the 47s model max.
    # Round up to next multiple of 2048 so the VAE produces clean latent frames.
    max_dur_samples = max(int(dur * sample_rate) for _, dur in prompts_with_durations)
    sample_size = math.ceil(max_dur_samples / 2048) * 2048

    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        batch_size=len(conditioning),
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device,
    )
    # output: [B, 2, n_samples] (stereo)
    clips = []
    for i in range(output.shape[0]):
        clip = output[i].float()   # [2, n_samples]
        # Trim to actual duration
        n_samples = int(prompts_with_durations[i][1] * sample_rate)
        clip = clip[:, :n_samples]
        clips.append(clip)

    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clips",     type=int,   default=20_000)
    parser.add_argument("--out_dir",     default="data/adapter_training")
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--steps",       type=int,   default=100,
                        help="Diffusion steps per clip (lower=faster, less quality)")
    parser.add_argument("--cfg_scale",   type=float, default=7.0)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir   = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading Stable Audio Open...")
    try:
        from stable_audio_tools import get_pretrained_model
    except ImportError:
        print("ERROR: Install stable-audio-tools: pip install stable-audio-tools")
        sys.exit(1)

    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device)
    model.eval()
    sample_rate = model_config["sample_rate"]   # 44100
    print(f"Model loaded. Generating {args.n_clips} clips...")

    # Count existing clips to resume if interrupted
    existing = set()
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                rec = json.loads(line)
                existing.add(Path(rec["audio_path"]).stem)
    n_existing = len(existing)
    print(f"Found {n_existing} existing clips, generating {args.n_clips - n_existing} more.")

    meta_file = open(meta_path, "a")
    n_generated = n_existing

    try:
        while n_generated < args.n_clips:
            # Sample a batch of prompts
            batch_prompts = []
            for _ in range(min(args.batch_size, args.n_clips - n_generated)):
                prompt_text, dur_range = random.choice(PROMPTS)
                dur = random.uniform(dur_range[0], dur_range[1])
                batch_prompts.append((prompt_text, dur))

            with torch.no_grad():
                clips = generate_clips(
                    model, model_config, batch_prompts, device,
                    steps=args.steps, cfg_scale=args.cfg_scale,
                )

            for clip, (prompt_text, dur) in zip(clips, batch_prompts):
                clip_id   = f"clip_{n_generated:06d}"
                save_path = audio_dir / f"{clip_id}.wav"

                # Normalize and save
                peak = clip.abs().max().clamp(min=1e-8)
                clip_normalized = (clip / peak * 0.95).cpu().numpy().T  # [n_samples, 2]
                sf.write(str(save_path), clip_normalized, sample_rate)

                rec = {
                    "audio_path":   str(save_path.resolve()),
                    "text_prompt":  prompt_text,
                    "seconds_total": round(dur, 3),
                }
                meta_file.write(json.dumps(rec) + "\n")
                meta_file.flush()

                n_generated += 1
                if n_generated % 100 == 0:
                    print(f"  Generated {n_generated}/{args.n_clips} clips")

    finally:
        meta_file.close()

    print(f"Done. {n_generated} clips saved to {out_dir}/")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
