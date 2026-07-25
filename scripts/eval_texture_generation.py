"""Generation probe — does the texture LoRA FUNCTION when sampled?

The discrimination score proves the adapter *learned* the artifact
distribution (better denoising of artifact latents). This script tests the
stronger claim: that *sampling* with the adapter actually produces
artifact-textured output. Entirely in latent space — no VAE, no audio, no
external cover engine.

Method: unconditionally sample N latent clips (plain Euler over the flow,
null conditioning, silence context — the training regime), adapter ON and
OFF, same seeds. Compare each batch's latent statistics against two
reference sets you provide: the artifact diet and clean audio latents.

    d(gen, ref) = distance over per-channel mean/std + per-channel
                  temporal-difference energy (latent-space "texture")

A functioning texture LoRA moves generations TOWARD the diet reference:
    d(ON, diet) < d(OFF, diet), ideally with d(*, clean) worsening or flat.

NOTE: this is a validation probe, not faithful ACE-Step inference (no CFG
mixing, no APG guidance, uniform Euler steps). Statistics comparisons are
still valid because ON and OFF share the sampler, seeds, and conditioning.

Usage:
    uv run python scripts/eval_texture_generation.py \
        --checkpoint-dir ./checkpoints --model sft \
        --adapter ./trained_adapters/glasser_texture/best \
        --artifact-dir ./preprocessed_tensors/glasser_diet_latents \
        --clean-dir ./preprocessed_tensors/clean_control_latents \
        [--n 16] [--frames 250] [--steps 30] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _set_stats(latents_list):
    """Per-channel mean/std + temporal-diff energy over a set of [T,64] latents."""
    import torch

    flat = torch.cat([x.float() for x in latents_list], dim=0)          # [sum_T, 64]
    diffs = torch.cat([x.float().diff(dim=0) for x in latents_list], dim=0)
    return {
        "mean": flat.mean(dim=0),                    # [64]
        "std": flat.std(dim=0),                      # [64]
        "diff_energy": diffs.pow(2).mean(dim=0).sqrt(),  # [64] latent "texture"
    }


def _dist(a: dict, b: dict) -> dict:
    return {k: float((a[k] - b[k]).pow(2).mean().sqrt()) for k in a}


def _load_ref(d: str):
    import torch
    files = sorted(Path(d).glob("*.pt"))
    if not files:
        sys.exit(f"no .pt files in {d}")
    return [torch.load(str(f), map_location="cpu", weights_only=False)["target_latents"]
            for f in files]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--model", default="sft", dest="variant")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--n", type=int, default=16, help="Latent clips to sample per cell")
    ap.add_argument("--frames", type=int, default=250, help="Latent frames (250 = 10 s)")
    ap.add_argument("--steps", type=int, default=30, help="Euler steps")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch

    from sidestep_engine._compat import configure_cuda_allocator
    configure_cuda_allocator()
    from sidestep_engine.models.gpu_utils import detect_gpu
    from sidestep_engine.models.loader import load_decoder_for_training, load_silence_latent

    gpu = detect_gpu(requested_device=args.device, requested_precision=args.precision)
    device = torch.device(gpu.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(gpu.precision, torch.float32)

    print(f"[probe] loading model ({args.variant}, {device}/{gpu.precision})", file=sys.stderr)
    model = load_decoder_for_training(
        args.checkpoint_dir, args.variant,
        device=str(device), precision=gpu.precision, offload_encoder=True,
    )
    from peft import PeftModel
    model.decoder = PeftModel.from_pretrained(model.decoder, args.adapter, is_trainable=False)
    model.decoder.eval()

    null_emb = getattr(model, "null_condition_emb", None)
    if null_emb is None:
        sys.exit("model has no null_condition_emb — cannot sample unconditionally")

    silence = load_silence_latent(
        args.checkpoint_dir, device="cpu", precision="fp32", variant=args.variant,
    ).float()
    T = args.frames
    if silence.shape[1] >= T:
        src = silence[:, :T, :]
    else:
        reps = -(-T // silence.shape[1])
        src = silence.repeat(1, reps, 1)[:, :T, :]
    context_1 = torch.cat([src, torch.ones(1, T, 64)], dim=-1)  # [1, T, 128]

    @torch.no_grad()
    def sample_batch(bsz: int, seed: int):
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(bsz, T, 64, generator=g).to(device, dtype=dtype)
        am = torch.ones(bsz, T, device=device, dtype=dtype)
        ehs = null_emb.to(device, dtype=dtype).expand(bsz, 1, -1)
        eam = torch.ones(bsz, 1, device=device, dtype=torch.bool)
        ctx = context_1.to(device, dtype=dtype).expand(bsz, -1, -1)

        ts = torch.linspace(1.0, 0.0, args.steps + 1)
        for i in range(args.steps):
            t_now, t_next = float(ts[i]), float(ts[i + 1])
            t = torch.full((bsz,), max(t_now, 1e-4), device=device, dtype=dtype)
            v = model.decoder(
                hidden_states=x, timestep=t, timestep_r=t,
                attention_mask=am, encoder_hidden_states=ehs,
                encoder_attention_mask=eam, context_latents=ctx,
            )[0]
            x = x - (t_now - t_next) * v  # dx/dt = v ; integrate t: 1 -> 0
        return [x[j].float().cpu() for j in range(bsz)]

    def sample_set(tag: str) -> list:
        out = []
        bi = 0
        while len(out) < args.n:
            b = min(args.batch_size, args.n - len(out))
            out.extend(sample_batch(b, args.seed * 7919 + bi))
            bi += 1
        print(f"[probe] sampled {len(out)} clips ({tag})", file=sys.stderr)
        return out

    gen_on = sample_set("adapter ON")
    with model.decoder.disable_adapter():
        gen_off = sample_set("adapter OFF")

    stats = {
        "diet": _set_stats(_load_ref(args.artifact_dir)),
        "clean": _set_stats(_load_ref(args.clean_dir)),
        "on": _set_stats(gen_on),
        "off": _set_stats(gen_off),
    }

    print("\n=== generation probe: latent-statistics distances ===")
    print(f"{'':14} {'-> diet':>12} {'-> clean':>12}")
    rows = {}
    for cell in ("off", "on"):
        d_diet = _dist(stats[cell], stats["diet"])
        d_clean = _dist(stats[cell], stats["clean"])
        rows[cell] = (d_diet, d_clean)
        for metric in d_diet:
            print(f"{cell:>4} {metric:<10} {d_diet[metric]:>12.4f} {d_clean[metric]:>12.4f}")

    print("\n  movement (OFF -> ON), negative toward-diet = functioning:")
    functioning = 0
    for metric in rows["on"][0]:
        delta_diet = rows["on"][0][metric] - rows["off"][0][metric]
        delta_clean = rows["on"][1][metric] - rows["off"][1][metric]
        toward = delta_diet < 0
        functioning += int(toward)
        print(f"    {metric:<12} d_diet {delta_diet:+.4f} {'[toward diet]' if toward else '[away]':<14} "
              f"d_clean {delta_clean:+.4f}")

    print("\n  verdict:")
    if functioning == 0:
        print("    [FAIL] generations do not move toward the artifact distribution — adapter not functioning as a texture direction.")
    elif functioning < 3:
        print("    [WARN] mixed movement — partial texture effect; try a later checkpoint or higher scale, and confirm with real covers.")
    else:
        print("    [OK] generations move toward the artifact distribution on all metrics — texture LoRA is FUNCTIONING. Confirm on real covers + ears.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
