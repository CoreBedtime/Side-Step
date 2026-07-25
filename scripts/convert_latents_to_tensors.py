"""Convert raw DiT latents (.npy) into Side-Step training tensors (.pt).

For texture/corruption LoRAs trained directly on latents: skips the entire
audio->VAE->latent preprocessing pipeline (no lossy re-encode, no captions,
no sidecars). Pair with ``--cfg-ratio 1.0`` for unconditional training —
the stored encoder_hidden_states is a shape-correct dummy that CFG dropout
replaces with the model's null_condition_emb on every step.

Context channel (``--context``):
    self     (default) cover-aligned: context = the clip's own latents.
             Training task == noFSQ cover inference: "given this content
             in context, reproduce it — with your texture."
    silence  standard text2music regime (what audio preprocessing does).
             Fallback if self-context training collapses (watch the
             discrimination score).

Usage:
    uv run python scripts/convert_latents_to_tensors.py \
        --input "C:/path/to/latents_npy" \
        --output ./preprocessed_tensors/glasser_diet_latents \
        [--context self] [--emb-dim 2048]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_latent(path: Path):
    import numpy as np
    import torch

    arr = np.load(str(path))
    t = torch.from_numpy(arr).float()
    while t.dim() > 2:
        t = t.squeeze(0)
    if t.dim() != 2:
        raise ValueError(f"{path.name}: expected 2-D latents, got {tuple(t.shape)}")
    # Auto-orient to [T, C] with C == 64
    if t.shape[0] == 64 and t.shape[1] != 64:
        t = t.transpose(0, 1)
    if t.shape[1] != 64:
        raise ValueError(f"{path.name}: neither dim is 64 channels: {tuple(t.shape)}")
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="Directory of .npy latent files")
    ap.add_argument("--output", required=True, help="Output dir for .pt tensors")
    ap.add_argument("--context", choices=("self", "silence"), default="self")
    ap.add_argument("--emb-dim", type=int, default=2048,
                    help="Text-embedding dim of the target variant (base/sft: 2048)")
    args = ap.parse_args()

    import torch

    in_dir, out_dir = Path(args.input), Path(args.output)
    files = sorted(in_dir.glob("*.npy"))
    if not files:
        sys.exit(f"no .npy files in {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for f in files:
        try:
            lat = _load_latent(f)  # [T, 64] float32
        except ValueError as exc:
            print(f"[skip] {exc}", file=sys.stderr)
            continue
        T = lat.shape[0]

        if args.context == "self":
            src = lat.unsqueeze(0)  # cover-aligned: own content as source
        else:
            src = torch.zeros(1, T, 64)  # silence-equivalent neutral context
        chunk_mask = torch.ones(1, T, 64)
        context = torch.cat([src, chunk_mask], dim=-1)[0]  # [T, 128]

        sample = {
            "target_latents": lat.to(torch.bfloat16),
            "attention_mask": torch.ones(T, dtype=torch.bfloat16),
            # Dummy conditioning — train with --cfg-ratio 1.0 so this is
            # replaced by null_condition_emb every step and never seen.
            "encoder_hidden_states": torch.zeros(1, args.emb_dim, dtype=torch.bfloat16),
            "encoder_attention_mask": torch.ones(1, dtype=torch.bool),
            "context_latents": context.to(torch.bfloat16),
            "metadata": {
                "filename": f.stem,
                "caption": "",
                "source": "raw-latents",
                "context_mode": args.context,
            },
        }
        torch.save(sample, out_dir / f"{f.stem}.pt")
        n_ok += 1

    print(f"wrote {n_ok}/{len(files)} tensors to {out_dir} "
          f"(context={args.context}, emb_dim={args.emb_dim})")
    print("train with:  --cfg-ratio 1.0  (unconditional; dummy embeddings are "
          "replaced by null_condition_emb every step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
