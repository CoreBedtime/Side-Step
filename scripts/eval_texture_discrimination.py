"""Texture-LoRA discrimination score — "is it learning the artifact texture?"

Runs a 2x2 deterministic reconstruction-loss measurement:

                    adapter ON    adapter OFF
    artifact set        A             B
    clean set           C             D

- (B - A)  > 0            : the LoRA learned *something* about the artifact set
- (B - A) - (D - C) > 0   : it learned the artifact TEXTURE specifically,
                            not just generic drift (the discrimination score)
- gains concentrated at LOW t buckets: consistent with texture (fine detail)
  rather than structure.

Fully deterministic: fixed stratified timestep grid + seeded noise, identical
across all four cells, so the comparison is paired and low-variance.

Usage:
    uv run python scripts/eval_texture_discrimination.py \
        --checkpoint-dir ./checkpoints --model sft \
        --adapter ./trained_adapters/glasser_texture/best \
        --artifact-dir ./preprocessed_tensors/glasser_diet_val \
        --clean-dir ./preprocessed_tensors/clean_control \
        [--batches 24] [--batch-size 4] [--seed 42]

Both dirs must be Side-Step preprocessed tensor dirs. Use a held-out split
of the artifact set (or just the full set for a coarse signal). The clean
control set is a handful of real-music clips preprocessed the same way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Stratified timestep grid: low (detail) / mid / high (structure) buckets.
T_GRID = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85)
T_BUCKETS = {"low t (detail)": (0.0, 0.34), "mid t": (0.34, 0.67), "high t (structure)": (0.67, 1.0)}


def _masked_mse(pred, target, attention_mask):
    err = (pred - target) ** 2
    mask = attention_mask.unsqueeze(-1)
    per_sample = (err * mask).sum(dim=(-1, -2)) / (
        mask.sum(dim=(-1, -2)) * err.shape[-1]
    ).clamp(min=1e-8)
    return per_sample  # [B]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--model", default="sft", dest="variant")
    ap.add_argument("--adapter", required=True, help="PEFT adapter dir (e.g. <run>/best)")
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--batches", type=int, default=24, help="Max batches per dataset")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--null-cond", action="store_true",
                    help="Replace stored text conditioning with the model's "
                         "null_condition_emb (use for adapters trained with "
                         "--cfg-ratio 1.0 / unconditional texture LoRAs)")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from sidestep_engine._compat import configure_cuda_allocator
    configure_cuda_allocator()

    from sidestep_engine.models.gpu_utils import detect_gpu
    from sidestep_engine.models.loader import load_decoder_for_training
    from sidestep_engine.vendor.data_module import (
        PreprocessedTensorDataset,
        collate_preprocessed_batch,
    )

    gpu = detect_gpu(requested_device=args.device, requested_precision=args.precision)
    device = torch.device(gpu.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(gpu.precision, torch.float32)

    print(f"[eval] loading model ({args.variant}, {device}/{gpu.precision})", file=sys.stderr)
    model = load_decoder_for_training(
        args.checkpoint_dir, args.variant,
        device=str(device), precision=gpu.precision, offload_encoder=True,
    )

    from peft import PeftModel
    model.decoder = PeftModel.from_pretrained(model.decoder, args.adapter, is_trainable=False)
    model.decoder.eval()
    print(f"[eval] adapter attached from {args.adapter}", file=sys.stderr)

    def _loader(d: str) -> DataLoader:
        ds = PreprocessedTensorDataset(d, cache_in_ram=True)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate_preprocessed_batch, num_workers=0)

    loaders = {"artifact": _loader(args.artifact_dir), "clean": _loader(args.clean_dir)}

    @torch.no_grad()
    def run_pass(loader: DataLoader) -> dict:
        """Mean masked-MSE overall and per t-bucket, deterministic draws."""
        totals: dict = {k: [0.0, 0] for k in ("all", *T_BUCKETS)}
        for bi, batch in enumerate(loader):
            if bi >= args.batches:
                break
            batch.pop("conditioning_info", None)
            batch.pop("metadata", None)
            x0 = batch["target_latents"].to(device, dtype=dtype)
            am = batch["attention_mask"].to(device, dtype=dtype)
            ehs = batch["encoder_hidden_states"].to(device, dtype=dtype)
            eam = batch["encoder_attention_mask"].to(device, dtype=dtype)
            ctx = batch["context_latents"].to(device, dtype=dtype)
            bsz = x0.shape[0]

            if args.null_cond:
                null_emb = getattr(model, "null_condition_emb", None)
                if null_emb is None:
                    sys.exit("--null-cond: model has no null_condition_emb")
                ehs = null_emb.to(device, dtype=dtype).expand_as(ehs)

            # Deterministic noise per batch index (identical across cells)
            g = torch.Generator(device="cpu").manual_seed(args.seed * 100003 + bi)
            noise = torch.randn(x0.shape, generator=g).to(device, dtype=dtype)

            for t_val in T_GRID:
                t = torch.full((bsz,), t_val, device=device, dtype=dtype)
                t_ = t.view(-1, 1, 1)
                xt = t_ * noise + (1.0 - t_) * x0
                out = model.decoder(
                    hidden_states=xt, timestep=t, timestep_r=t,
                    attention_mask=am, encoder_hidden_states=ehs,
                    encoder_attention_mask=eam, context_latents=ctx,
                )
                per = _masked_mse(out[0].float(), (noise - x0).float(), am.float())
                v, n = float(per.mean()), bsz
                totals["all"][0] += v * n; totals["all"][1] += n
                for name, (lo, hi) in T_BUCKETS.items():
                    if lo <= t_val < hi:
                        totals[name][0] += v * n; totals[name][1] += n
        return {k: s / max(n, 1) for k, (s, n) in totals.items()}

    results: dict = {}
    for set_name, loader in loaders.items():
        results[("on", set_name)] = run_pass(loader)
        with model.decoder.disable_adapter():
            results[("off", set_name)] = run_pass(loader)
        print(f"[eval] {set_name} done", file=sys.stderr)

    A = results[("on", "artifact")]["all"];  B = results[("off", "artifact")]["all"]
    C = results[("on", "clean")]["all"];    D = results[("off", "clean")]["all"]
    gain_artifact, gain_clean = B - A, D - C
    disc = gain_artifact - gain_clean

    print("\n=== texture-LoRA discrimination report ===")
    print(f"  artifact set:  ON {A:.5f}   OFF {B:.5f}   gain {gain_artifact:+.5f} ({gain_artifact/max(B,1e-9)*100:+.1f}%)")
    print(f"  clean set:     ON {C:.5f}   OFF {D:.5f}   gain {gain_clean:+.5f} ({gain_clean/max(D,1e-9)*100:+.1f}%)")
    print(f"  DISCRIMINATION SCORE: {disc:+.5f}  ({disc/max(B,1e-9)*100:+.1f}% of base artifact loss)")
    print("\n  per t-bucket gains on the artifact set (texture should favor low t):")
    for bucket in T_BUCKETS:
        on_b = results[("on", "artifact")][bucket]; off_b = results[("off", "artifact")][bucket]
        print(f"    {bucket:22s} {off_b - on_b:+.5f} ({(off_b-on_b)/max(off_b,1e-9)*100:+.1f}%)")

    print("\n  verdict:")
    if gain_artifact <= 0:
        print("    [FAIL] adapter does not improve artifact reconstruction — not learning (or wrong checkpoint).")
    elif disc <= 0:
        print("    [WARN] adapter helps clean audio as much as artifact audio — generic drift, not texture-specific.")
    else:
        print("    [OK] adapter improves artifact reconstruction MORE than clean — texture-specific learning confirmed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
