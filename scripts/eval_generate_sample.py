"""Ear-verification probe for the eval-harness sampler.

Generates one audio clip with the real inference recipe (shift schedule +
APG-CFG + Euler) from a preprocessed sample's cached conditioning, decodes
through the VAE, and writes a WAV. If the output sounds like ACE-Step
output for the caption, the sampler is faithful enough to build the
harness on.

Usage:
    uv run python scripts/eval_generate_sample.py \
        --checkpoint-dir ./checkpoints --model base \
        --sample "./preprocessed_tensors/Joji/<file>.pt" \
        [--adapter path] [--steps 50] [--shift 1.0] [--guidance 7.0] \
        [--frames 750] [--seed 42] [--out eval_sample.wav]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--model", default="base", dest="variant")
    ap.add_argument("--sample", default=None, help="Path to a preprocessed .pt sample")
    ap.add_argument("--prompt", default=None,
                    help="Text prompt (bypasses --sample; encodes fresh conditioning)")
    ap.add_argument("--lyrics", default="[Instrumental]")
    ap.add_argument("--adapter", default=None, help="Optional PEFT adapter dir")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="APG guidance; 1.0 = real inference default (no CFG)")
    ap.add_argument("--frames", type=int, default=750, help="Latent frames (750 = 30 s)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--out", default="eval_sample.wav")
    args = ap.parse_args()

    import torch

    from sidestep_engine._compat import configure_cuda_allocator
    configure_cuda_allocator()
    from sidestep_engine.eval.generator import decode_to_wav, generate_clip, load_conditioning
    from sidestep_engine.models.gpu_utils import detect_gpu
    from sidestep_engine.models.loader import load_decoder_for_training, load_vae

    gpu = detect_gpu(requested_device=args.device, requested_precision=args.precision)
    device = torch.device(gpu.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(gpu.precision, torch.float32)

    if not args.sample and not args.prompt:
        sys.exit("provide --sample or --prompt")

    # Prompt mode needs the DiT encoder on-device; sample mode does not.
    print(f"[probe] loading model ({args.variant}, {device}/{gpu.precision})", file=sys.stderr)
    model = load_decoder_for_training(
        args.checkpoint_dir, args.variant,
        device=str(device), precision=gpu.precision,
        offload_encoder=not args.prompt,
    )

    if args.prompt:
        from sidestep_engine.eval.generator import build_conditioning_from_prompt
        cond = build_conditioning_from_prompt(
            model, args.checkpoint_dir,
            prompt=args.prompt, lyrics=args.lyrics, frames=args.frames,
            variant=args.variant, device=device, dtype=dtype,
        )
    else:
        cond = load_conditioning(args.sample, max_frames=args.frames)
    print(f"[probe] sample: {cond['name']}", file=sys.stderr)
    print(f"[probe] caption: {cond['caption'][:120]}", file=sys.stderr)
    if not args.prompt and (not cond["caption"] or len(cond["caption"].split()) < 3):
        print("[probe] WARNING: caption looks empty/filename-like — expect noise; "
              "use --prompt for a real test", file=sys.stderr)
    if args.adapter:
        from peft import PeftModel
        model.decoder = PeftModel.from_pretrained(model.decoder, args.adapter, is_trainable=False)
        model.decoder.eval()
        print(f"[probe] adapter attached: {args.adapter}", file=sys.stderr)

    t0 = time.perf_counter()
    latents = generate_clip(
        model, cond,
        infer_steps=args.steps, shift=args.shift, guidance_scale=args.guidance,
        seed=args.seed, device=device, dtype=dtype,
    )
    t_sample = time.perf_counter() - t0
    print(f"[probe] sampled {tuple(latents.shape)} in {t_sample:.1f}s "
          f"({args.steps} steps, shift={args.shift}, guidance={args.guidance})",
          file=sys.stderr)

    print("[probe] loading VAE for decode", file=sys.stderr)
    vae = load_vae(args.checkpoint_dir, device=str(device), precision=gpu.precision)
    out = decode_to_wav(latents, vae, args.out)
    print(f"[probe] OK -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
