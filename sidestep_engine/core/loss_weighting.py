"""Timestep loss-weighting math for flow matching training.

Extracted from ``FixedLoRAModule.training_step`` into a pure function so
the math is unit-testable without a model, and so weighting composes
correctly with the ``x0_*`` loss family.

Composition rule (the "x0 compensation")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
With ``xt = t*x1 + (1-t)*x0`` the x0-reconstruction error equals
``t * (velocity error)``, so the per-sample x0 MSE is exactly
``t^2 *`` the velocity MSE — an *implicit* t^2 timestep weighting.
Stacking an explicit SNR weighting ``w(t)`` on top of that would apply
timestep weighting twice.  When ``x0_loss=True`` we therefore divide the
final velocity-space weight by ``t^2`` so that::

    w_applied * per_sample_x0_loss == w_velocity * per_sample_velocity_loss

i.e. the configured weighting acts exactly once, regardless of loss
parameterization.  (Exact for MSE; first-order for the huber variants.)

The compensation is deliberately applied *after* clamping and
normalization — compensating earlier would re-shape the weighting curve
instead of undoing the implicit t^2.

``min_snr`` uses the v-prediction form ``min(SNR, gamma) / (SNR + 1)``
from the Min-SNR-gamma paper (Hang et al. 2023).  The previous
``min(SNR, gamma) / SNR`` was the epsilon-prediction form, which
over-weights the low-noise end for velocity-style objectives.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = ["compute_timestep_weights"]


def compute_timestep_weights(
    t: torch.Tensor,
    *,
    loss_weighting: str,
    snr_gamma: float,
    t_bias: float,
    x0_loss: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute per-sample timestep weights for the flow matching loss.

    Args:
        t: Sampled timesteps ``[B]`` in (0, 1); t=1 is pure noise.
        loss_weighting: ``"none"``, ``"flow_snr"``, or ``"min_snr"``.
        snr_gamma: Clamp ceiling for the weighting curve.
        t_bias: Asymmetric low-t emphasis exponent (flow_snr only).
        x0_loss: True when the per-sample loss is computed on the
            reconstructed x0 (``x0_mse`` / ``x0_pseudo_huber``), which
            carries an implicit t^2 factor relative to the velocity loss.

    Returns:
        ``(weights, telemetry)`` where *weights* is the tensor to multiply
        per-sample losses by (``None`` for ``loss_weighting="none"``), and
        *telemetry* is the pre-normalization curve for TensorBoard
        (``None`` for ``"none"``).
    """
    if loss_weighting not in ("flow_snr", "min_snr"):
        return None, None

    t_f32 = t.float().clamp(min=1e-4, max=1.0 - 1e-4)

    if loss_weighting == "flow_snr":
        w = ((1.0 - t_f32) ** t_bias) / (t_f32 * (1.0 - t_f32))
        w = w.clamp(max=snr_gamma)
        telemetry = w.clone()
        w = w / w.mean().clamp(min=1e-8)  # normalize to preserve loss scale
    else:  # min_snr — v-prediction form
        snr = ((1.0 - t_f32) / t_f32) ** 2
        snr = snr.clamp(max=1e6)
        w = torch.clamp(snr, max=snr_gamma) / (snr + 1.0)
        telemetry = w.clone()

    if x0_loss:
        # Undo the implicit t^2 of the x0 parameterization so the explicit
        # weighting applies exactly once.  No re-normalization afterwards:
        # w/t^2 * (t^2 * L_velocity) == w * L_velocity by construction.
        w = w / (t_f32 * t_f32)

    return w, telemetry
