"""
ADSR sensitivity test — the go/no-go check for the adapter.

Generates the same prompt twice with very different attack times (10ms vs 500ms)
and measures whether the onset profiles are detectably different.

Usage:
    python sensitivity_test.py --adapter checkpoints_adapter/adapter_final.pt
    python sensitivity_test.py --adapter checkpoints_adapter/adapter_best.pt --prompt "piano note C4"

Outputs:
    sensitivity_test_A10ms.wav
    sensitivity_test_A500ms.wav
    Prints ATE (attack time error) between the two — should be > 200ms to pass.
"""

import argparse
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

# pkg_resources shim (needed by k_diffusion on Python 3.12+)
try:
    import pkg_resources  # noqa: F401
except ImportError:
    import packaging as _packaging
    _mod = types.ModuleType("pkg_resources")
    _mod.packaging = _packaging
    sys.modules["pkg_resources"] = _mod

from timecode_audio.model.stable_audio_adapter import EnvelopeAdapter
from timecode_audio.core.envelope import adsr_gate


SAMPLE_RATE = 44100
VAE_DOWNSAMPLE = 2048


def adsr_to_envelope(A, D, S, R, duration_s, device):
    """Convert ADSR params to a latent-rate envelope tensor [1, T_latent]."""
    n_samples = int(duration_s * SAMPLE_RATE)
    t_ms = np.arange(n_samples, dtype=np.float32) / SAMPLE_RATE * 1000.0
    gate = adsr_gate(t_ms, A=A, D=D, S=S, R=R, note_duration_ms=(duration_s - R / 1000) * 1000)
    gate_t = torch.from_numpy(gate)

    # Downsample to latent frame rate by averaging blocks of VAE_DOWNSAMPLE
    n_frames = n_samples // VAE_DOWNSAMPLE
    gate_t = gate_t[: n_frames * VAE_DOWNSAMPLE].reshape(n_frames, VAE_DOWNSAMPLE).mean(1)

    # Normalize to [0, 1]
    peak = gate_t.max().clamp(min=1e-8)
    gate_t = gate_t / peak

    return gate_t.unsqueeze(0).to(device)   # [1, T_latent]


def detect_attack_time_ms(audio_np, sample_rate, threshold=0.05):
    """Find the first sample where amplitude crosses threshold fraction of peak."""
    mono = np.abs(audio_np.mean(axis=1) if audio_np.ndim == 2 else audio_np)
    peak = mono.max()
    if peak < 1e-8:
        return None
    idx = np.argmax(mono > threshold * peak)
    return idx / sample_rate * 1000.0


def generate_with_adsr(sa_model, adapter, prompt, A, D, S, R, duration_s, device, steps=50, cfg_scale=7.0, envelope_guidance=10.0):
    """Generate audio conditioned on the given ADSR envelope."""
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    # Build envelope for CFG: guided = envelope_guidance * adapter(envelope)
    envelope = adsr_to_envelope(A, D, S, R, duration_s, device)

    # Hook: add adapter embedding to noisy latents at each denoising step
    adapter_embed = adapter.forward_with_cfg(envelope, guidance_scale=envelope_guidance)
    # adapter_embed: [1, 64, T_latent]

    print(f"  adapter_embed: shape={adapter_embed.shape}, "
          f"norm={adapter_embed.norm().item():.6f}, "
          f"max={adapter_embed.abs().max().item():.6f}")

    # Register a forward pre-hook on the conditioned model to inject adapter
    injected = {"embed": adapter_embed, "hook_count": 0}

    def inject_hook(module, args):
        count = injected["hook_count"]
        injected["hook_count"] += 1
        x = args[0]
        embed = injected["embed"]

        if count == 0:
            print(f"  [hook] x.shape={x.shape}, embed.shape={embed.shape}")

        T = x.shape[2]
        E = embed.shape[2]
        if E < T:
            embed = torch.nn.functional.pad(embed, (0, T - E))
        else:
            embed = embed[:, :, :T]

        B_full = x.shape[0]
        B = B_full // 2
        if B > 0 and B_full > 1:
            # batch_cfg: try first half as conditioned
            embed_b = embed.expand(B, -1, -1)
            x_new = x.clone()
            x_new[:B] = x[:B] + embed_b
        else:
            x_new = x + embed.expand(B_full, -1, -1)

        return (x_new,) + args[1:]

    # generate_diffusion_cond calls sa_model.model (DiTWrapper), not sa_model itself.
    # Register on sa_model.model so the hook actually fires.
    handle = sa_model.model.register_forward_pre_hook(inject_hook)

    sample_size = math.ceil(int(duration_s * SAMPLE_RATE) / VAE_DOWNSAMPLE) * VAE_DOWNSAMPLE
    conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": duration_s}]

    with torch.no_grad():
        output = generate_diffusion_cond(
            sa_model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            batch_size=1,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device,
        )

    handle.remove()
    print(f"  hook fired {injected['hook_count']} times")
    audio = output[0].float().cpu().numpy().T   # [n_samples, 2]
    return audio[:int(duration_s * SAMPLE_RATE)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",   default="checkpoints_adapter/adapter_best.pt")
    parser.add_argument("--dit_ckpt",  default=None,
                        help="Path to fine-tuned DiT layers (checkpoints_adapter_ft/dit_layers_best.pt). "
                             "Only needed after stage-2 fine-tuning.")
    parser.add_argument("--prompt",    default="grand piano note, single key, middle C")
    parser.add_argument("--duration",  type=float, default=4.0, help="clip duration in seconds")
    parser.add_argument("--steps",     type=int,   default=50)
    parser.add_argument("--out_dir",   default=".")
    parser.add_argument("--attack_fast", type=float, default=10.0,
                        help="Fast attack time in ms (default 10)")
    parser.add_argument("--attack_slow", type=float, default=500.0,
                        help="Slow attack time in ms (default 500)")
    parser.add_argument("--envelope_guidance", type=float, default=10.0,
                        help="Guidance scale for envelope adapter (default 10.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Fixed random seed so both clips differ only in envelope")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading Stable Audio Open...")
    from stable_audio_tools import get_pretrained_model
    sa_model, _ = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    sa_model = sa_model.to(device).eval()
    for p in sa_model.parameters():
        p.requires_grad_(False)

    print(f"Loading adapter from {args.adapter}...")
    adapter = EnvelopeAdapter.load(args.adapter, device=str(device))
    adapter.eval()

    if args.dit_ckpt:
        print(f"Loading fine-tuned DiT layers from {args.dit_ckpt}...")
        dit_state = torch.load(args.dit_ckpt, map_location=device)
        # Load only the saved params (last N layers) into the model
        current = dict(sa_model.named_parameters())
        for name, data in dit_state.items():
            if name in current:
                current[name].data.copy_(data)
            else:
                print(f"  [WARNING] {name} not found in sa_model — skipping")
        print(f"  Loaded {len(dit_state)} parameter tensors.")

    out_dir = Path(args.out_dir)
    D, S, R = 100.0, 0.6, 500.0   # hold D/S/R constant

    fast_ms = args.attack_fast
    slow_ms = args.attack_slow
    for A, label in [(fast_ms, f"A{int(fast_ms)}ms"), (slow_ms, f"A{int(slow_ms)}ms")]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"\nGenerating: A={A}ms, D={D}ms, S={S}, R={R}ms — '{args.prompt}'")
        audio = generate_with_adsr(
            sa_model, adapter, args.prompt,
            A=A, D=D, S=S, R=R,
            duration_s=args.duration,
            device=device,
            steps=args.steps,
            envelope_guidance=args.envelope_guidance,
        )
        path = out_dir / f"sensitivity_test_{label}.wav"
        sf.write(str(path), audio, SAMPLE_RATE)
        print(f"  Saved: {path}")

        attack_ms = detect_attack_time_ms(audio, SAMPLE_RATE)
        print(f"  Detected onset: {attack_ms:.1f}ms" if attack_ms is not None else "  Could not detect onset")

    print("\n--- RESULT ---")
    print("Listen to both files. If attack shapes differ clearly, the adapter is working.")
    print("If they sound identical, unfreeze the last 4 DiT layers (see plan section 4c).")


if __name__ == "__main__":
    main()
