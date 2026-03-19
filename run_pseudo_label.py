"""
Generate ADSR pseudo-labels for NSynth using the trained ADSR estimator.

Reads NSynth audio + examples.json, runs ADSREstimator.predict_with_confidence(),
writes data/nsynth_estimated/metadata.jsonl.

Usage (on server after training estimator):
    python run_pseudo_label.py \
        --nsynth_dir /path/to/nsynth-train \
        --estimator_ckpt checkpoints_estimator/best.pt \
        --output_dir data/nsynth_estimated \
        --min_confidence 0.5 \
        --device cuda

NSynth pitched instrument families used (synth_lead and vocal excluded):
    piano, mallet, organ, guitar, bass, strings, reed, flute, brass
"""

from __future__ import annotations
import argparse
import json
import torch
import soundfile as sf
import torchaudio
from pathlib import Path
from tqdm import tqdm

from timecode_audio.model.adsr_estimator import ADSREstimator, PlattScaler

SAMPLE_RATE = 48000
NOTE_NAMES  = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# NSynth instrument families to include (exclude synth_lead=9, vocal=10)
INCLUDED_FAMILIES = {0, 1, 2, 3, 4, 5, 6, 7, 8}
FAMILY_NAMES = {
    0: "piano", 1: "mallet", 2: "organ", 3: "guitar", 4: "bass",
    5: "strings", 6: "reed", 7: "flute", 8: "brass",
}


def midi_to_hz(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def make_text_prompt(family: int, source: str, pitch: int) -> str:
    octave = pitch // 12 - 1
    note   = NOTE_NAMES[pitch % 12]
    name   = FAMILY_NAMES.get(family, "instrument")
    return f"{source} {name} note {note}{octave}"


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=-1)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(
            waveform.unsqueeze(0), sr, target_sr
        ).squeeze(0)
    return waveform   # [n_samples]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsynth_dir",       required=True,  help="Path to NSynth split dir (contains audio/ and examples.json)")
    parser.add_argument("--estimator_ckpt",   required=True,  help="Path to trained estimator checkpoint")
    parser.add_argument("--output_dir",       default="data/nsynth_estimated")
    parser.add_argument("--platt_scaler",     default=None,   help="Path to fitted PlattScaler JSON (optional)")
    parser.add_argument("--min_confidence",   type=float, default=0.5)
    parser.add_argument("--n_mc_passes",      type=int,   default=20)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--device",           default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Running on {device}")

    # Load estimator
    ckpt = torch.load(args.estimator_ckpt, map_location=device)
    model = ADSREstimator().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded estimator from step {ckpt.get('step', '?')}")

    # Load Platt scaler (optional)
    platt = None
    if args.platt_scaler:
        platt = PlattScaler.load(args.platt_scaler)
        print(f"Loaded Platt scaler (a={platt.a:.3f}, b={platt.b:.3f})")

    # Load NSynth metadata
    nsynth_dir = Path(args.nsynth_dir)
    examples_path = nsynth_dir / "examples.json"
    with open(examples_path) as f:
        examples = json.load(f)

    # Filter to pitched families only
    keys = [
        k for k, v in examples.items()
        if v.get("instrument_family", -1) in INCLUDED_FAMILIES
    ]
    print(f"NSynth: {len(examples)} total clips → {len(keys)} after family filter")

    # Output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    n_written = 0
    n_skipped = 0

    with open(meta_path, "w") as f_meta:
        # Process in batches
        for i in tqdm(range(0, len(keys), args.batch_size), desc="Pseudo-labeling"):
            batch_keys = keys[i : i + args.batch_size]

            # Load audio batch
            waveforms = []
            valid_keys = []
            for key in batch_keys:
                audio_path = nsynth_dir / "audio" / f"{key}.wav"
                if not audio_path.exists():
                    continue
                try:
                    w = load_audio(str(audio_path))
                    # Pad/trim to 4 seconds (NSynth notes are 4s)
                    target = SAMPLE_RATE * 4
                    if w.shape[0] < target:
                        w = torch.nn.functional.pad(w, (0, target - w.shape[0]))
                    else:
                        w = w[:target]
                    waveforms.append(w)
                    valid_keys.append(key)
                except Exception:
                    continue

            if not waveforms:
                continue

            audio_batch = torch.stack(waveforms).to(device)   # [B, n_samples]

            # MC dropout inference
            means, confidence = model.predict_with_confidence(
                audio_batch, platt_scaler=platt, n_passes=args.n_mc_passes
            )

            for j, key in enumerate(valid_keys):
                conf = float(confidence[j].item())
                if conf < args.min_confidence:
                    n_skipped += 1
                    continue

                meta = examples[key]
                pitch    = int(meta["pitch"])
                family   = int(meta.get("instrument_family", 0))
                source   = meta.get("instrument_source_str", "acoustic")

                record = {
                    "audio_path":      str(nsynth_dir / "audio" / f"{key}.wav"),
                    "text_prompt":     make_text_prompt(family, source, pitch),
                    "A_ms":            float(means["A"][j].item()),
                    "D_ms":            float(means["D"][j].item()),
                    "S_level":         float(means["S"][j].item()),
                    "R_ms":            float(means["R"][j].item()),
                    "pitch_midi":      pitch,
                    "instrument":      FAMILY_NAMES.get(family, "unknown"),
                    "source":          "nsynth_pseudo_labeled",
                    "adsr_confidence": conf,
                    "note_duration":   4000.0,   # NSynth notes are 4 seconds
                    "velocity":        meta.get("velocity", 127) / 127.0,
                    "sample_rate":     SAMPLE_RATE,
                }
                f_meta.write(json.dumps(record) + "\n")
                n_written += 1

    print(f"\nDone. {n_written} clips written, {n_skipped} skipped (below confidence {args.min_confidence})")
    print(f"Output: {meta_path}")


if __name__ == "__main__":
    main()
