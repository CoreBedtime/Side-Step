"""Flow-matching inference sampler for eval-time generation.

Faithful subset of ``generate_audio`` from the official checkpoint code
(``modeling_acestep_v15_base.py``), verified against upstream commit
recorded in ``models/bundled_acestep/BUNDLED_ACESTEP_SOURCE.txt``:

- schedule: ``t = linspace(1, 0, steps+1)`` with the shift transform
  ``t = shift*t / (1 + (shift-1)*t)`` when ``shift != 1``
- CFG by batch-doubling against ``model.null_condition_emb``, guided by
  **APG** (projected guidance with momentum -0.75 and norm threshold 2.5,
  ``dims=[1]``) within a ``cfg_interval`` of t; plain conditional
  prediction outside the interval
- Euler update ``x <- x - v * (t_curr - t_next)`` (the official fallback
  when no solver is registered — which is always the case standalone,
  since the ``flow_matching_solvers`` shim ships an empty registry)

Documented deviations (validation-grade, not bit-exact inference):
- no cross-attention KV cache between steps (correct, just slower; the
  official loop reuses encoder KVs across denoising steps)
- no SDE / multi-eval solvers, no repaint, no cover-strength condition
  switching — eval generates plain text2music from cached conditioning
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from sidestep_engine.models.acestep_remote_imports import _ensure_acestep_remote_imports

__all__ = ["build_timesteps", "sample_latents"]


def build_timesteps(
    infer_steps: int,
    shift: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Official inference schedule: linspace(1, 0) + shift transform."""
    t = torch.linspace(1.0, 0.0, infer_steps + 1, device=device, dtype=dtype)
    if shift != 1.0:
        t = shift * t / (1 + (shift - 1) * t)
    return t


@torch.no_grad()
def sample_latents(
    model,
    cond: Dict[str, torch.Tensor],
    *,
    infer_steps: int = 50,
    shift: float = 1.0,
    guidance_scale: float = 1.0,
    cfg_interval: Tuple[float, float] = (0.0, 1.0),
    seed: int = 0,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate one latent clip ``[T, 64]`` from cached conditioning.

    Args:
        model: The full AceStep model (``decoder`` + ``null_condition_emb``).
        cond: Dict with ``encoder_hidden_states`` ``[1, L, D]``,
            ``encoder_attention_mask`` ``[1, L]``, ``context_latents``
            ``[1, T, 128]``, ``attention_mask`` ``[1, T]`` — i.e. exactly
            what preprocessing stores per sample (unsqueezed).
        infer_steps / shift: schedule params (base/sft: 50 / 1.0;
            turbo: 8 / 3.0).
        guidance_scale: APG guidance strength; <= 1.0 disables CFG
            entirely (single forward per step).  Default 1.0 — the REAL
            inference default for all models (acestep.cpp: "0 = auto
            (1.0 for all models)"); the modeling file's 7.0 signature
            default is not what production uses.
        cfg_interval: apply guidance only when ``start <= t <= end``
            (official ``cfg_interval_start/end``); conditional prediction
            is used outside.
        seed: CPU-generator seed for the initial noise (deterministic).

    Returns:
        Sampled latents ``[T, 64]`` (float32, CPU).
    """
    _ensure_acestep_remote_imports()
    from acestep.models.common.apg_guidance import MomentumBuffer, apg_forward

    ehs = cond["encoder_hidden_states"].to(device, dtype=dtype)
    eam = cond["encoder_attention_mask"].to(device, dtype=dtype)
    ctx = cond["context_latents"].to(device, dtype=dtype)
    am = cond["attention_mask"].to(device, dtype=dtype)
    T = ctx.shape[1]

    null_emb = getattr(model, "null_condition_emb", None)
    do_cfg = guidance_scale > 1.0 and null_emb is not None

    if do_cfg:
        # Official layout: [cond, uncond] stacked on the batch axis.
        ehs = torch.cat([ehs, null_emb.to(device, dtype=dtype).expand_as(ehs)], dim=0)
        eam = torch.cat([eam, eam], dim=0)
        ctx = torch.cat([ctx, ctx], dim=0)
        am = torch.cat([am, am], dim=0)

    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, T, 64, generator=g).to(device, dtype=dtype)

    momentum = MomentumBuffer()
    ts = build_timesteps(infer_steps, shift)

    for i in range(infer_steps):
        t_curr, t_next = float(ts[i]), float(ts[i + 1])
        x_in = torch.cat([x, x], dim=0) if do_cfg else x
        t_vec = torch.full((x_in.shape[0],), max(t_curr, 1e-5), device=device, dtype=dtype)
        v = model.decoder(
            hidden_states=x_in,
            timestep=t_vec,
            timestep_r=t_vec,
            attention_mask=am,
            encoder_hidden_states=ehs,
            encoder_attention_mask=eam,
            context_latents=ctx,
        )[0]
        if do_cfg:
            v_cond, v_uncond = v.chunk(2)
            if cfg_interval[0] <= t_curr <= cfg_interval[1]:
                v = apg_forward(
                    pred_cond=v_cond,
                    pred_uncond=v_uncond,
                    guidance_scale=guidance_scale,
                    momentum_buffer=momentum,
                    dims=[1],
                )
            else:
                v = v_cond
        x = x - v * (t_curr - t_next)

    return x[0].float().cpu()
