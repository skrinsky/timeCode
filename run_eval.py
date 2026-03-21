"""
Stage 3 evaluation: ADSR reconstruction metrics on synthetic held-out set.

Loads checkpoints/final.pt, generates audio for a sample of the synthetic
validation set, and reports ATE/DTE/SLE/RTE against Option A targets.

Usage:
    python run_eval.py
    python run_eval.py --checkpoint checkpoints/best.pt --n_clips 500
"""

import argparse
import math
import torch
from torch.utils.data import DataLoader, random_split

from timecode_audio.data.synthetic_gen import SyntheticDataset
from timecode_audio.model.ddsp_synthesizer import DDSPSynthesizer
from timecode_audio.model.adsr_encoder import ADSREncoder, adsr_gate_samples
from timecode_audio.training.config import TrainingConfig
from timecode_audio.eval.adsr_metrics import compute_batch_adsr_errors, print_adsr_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--data_dir", default="data/synthetic")
    parser.add_argument("--n_clips", type=int, default=1000,
                        help="Number of val clips to evaluate")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Config matching Stage 3 (text_dim=512)
    config = TrainingConfig(text_dim=512)

    # Load models
    synthesizer = DDSPSynthesizer(
        n_harmonics=config.n_harmonics,
        n_noise_bands=config.n_noise_bands,
        hidden_dim=config.hidden_dim,
        text_dim=config.text_dim,
        sample_rate=config.sample_rate,
        frame_size=config.frame_size,
    ).to(device)

    adsr_encoder = ADSREncoder(adsr_dim=config.adsr_dim).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    synthesizer.load_state_dict(ckpt["synthesizer"])
    adsr_encoder.load_state_dict(ckpt["adsr_encoder"])
    synthesizer.eval()
    adsr_encoder.eval()
    print(f"Loaded checkpoint from step {ckpt['step']}")

    # Synthetic val set (same 2% split as trainer)
    dataset = SyntheticDataset(args.data_dir)
    val_size = max(1, int(len(dataset) * config.val_split))
    train_size = len(dataset) - val_size
    _, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    n_eval = min(args.n_clips, len(val_set))
    val_subset, _ = random_split(
        val_set, [n_eval, len(val_set) - n_eval],
        generator=torch.Generator().manual_seed(0),
    )
    loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating {n_eval} clips...")

    all_audio, all_A, all_D, all_S, all_R, all_dur = [], [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            f0_hz    = batch["f0_hz"].to(device)
            A        = batch["A"].to(device)
            D        = batch["D"].to(device)
            S        = batch["S"].to(device)
            R        = batch["R"].to(device)
            velocity = batch["velocity"].to(device)
            note_dur = batch["note_duration"].to(device)

            n_samples = batch["audio"].shape[1]
            n_frames  = math.ceil(n_samples / config.frame_size)

            stage_pos, _ = adsr_encoder(
                A, D, S, R, note_dur,
                n_frames=n_frames,
                frame_size=config.frame_size,
                sample_rate=config.sample_rate,
            )

            # Compute gate per clip
            gates = []
            for i in range(A.shape[0]):
                g = adsr_gate_samples(
                    A=float(A[i].item()),
                    D=float(D[i].item()),
                    S=float(S[i].item()),
                    R=float(R[i].item()),
                    note_duration_ms=float(note_dur[i].item()),
                    n_samples=n_samples,
                    sample_rate=config.sample_rate,
                )
                gates.append(g)
            gate = torch.stack(gates, dim=0).to(device)

            # No text conditioning for synthetic eval (null text)
            audio_pred = synthesizer(
                f0_hz=f0_hz,
                stage_pos=stage_pos,
                gate=gate,
                velocity=velocity,
                text_emb=None,
            )

            all_audio.append(audio_pred.cpu())
            all_A.append(A.cpu())
            all_D.append(D.cpu())
            all_S.append(S.cpu())
            all_R.append(R.cpu())
            all_dur.append(note_dur.cpu())

    audio_batch = torch.cat(all_audio, dim=0)
    A_batch     = torch.cat(all_A,    dim=0)
    D_batch     = torch.cat(all_D,    dim=0)
    S_batch     = torch.cat(all_S,    dim=0)
    R_batch     = torch.cat(all_R,    dim=0)
    dur_batch   = torch.cat(all_dur,  dim=0)

    errors = compute_batch_adsr_errors(
        audio_batch, A_batch, D_batch, S_batch, R_batch,
        sample_rate=config.sample_rate,
        note_durations=dur_batch,
    )
    print_adsr_report(errors, label="Stage 3 final (synthetic val set)")


if __name__ == "__main__":
    main()
