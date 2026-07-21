"""
Validation epoch runner for adapter training.

Runs a single no-grad pass over a held-out validation DataLoader,
returning the mean loss.  Stateless — the trainer orchestrates when
to call this and what to do with the result.

Determinism: the whole pass runs under a forked, fixed-seed RNG so
every epoch draws the *same* timesteps, noise, and CFG dropout mask.
Flow-matching loss is dominated by the variance of those draws; without
this, epoch-to-epoch val-loss differences are mostly sampling noise and
"save best" chases luck.  Forking also means validation no longer
perturbs the training RNG stream (training is bit-identical with or
without validation enabled).
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Iterator

import torch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Fixed offset added to the run seed for the validation RNG stream, so
# validation draws differ from the training warmup draws but are the
# same every epoch.
_VAL_SEED_OFFSET = 7919


@contextlib.contextmanager
def _deterministic_rng(device: torch.device, seed: int) -> Iterator[None]:
    """Fork RNG state, seed deterministically, restore state on exit."""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    fork_devices = [device] if device.type in ("cuda", "xpu") else []

    # torch.random.fork_rng does not capture MPS state; save it manually.
    mps_state = None
    if device.type == "mps" and hasattr(torch, "mps"):
        try:
            mps_state = torch.mps.get_rng_state()
        except Exception:  # pragma: no cover - defensive
            mps_state = None

    kwargs = {"devices": fork_devices}
    if fork_devices:
        kwargs["device_type"] = device.type
    try:
        with torch.random.fork_rng(**kwargs):
            torch.manual_seed(seed)
            yield
    finally:
        if mps_state is not None:
            try:
                torch.mps.set_rng_state(mps_state)
            except Exception:  # pragma: no cover - defensive
                pass


def run_validation_epoch(
    module: object,
    val_loader: "DataLoader",
    device: torch.device,
) -> float:
    """Run a no-grad validation pass and return mean loss.

    Args:
        module: A ``FixedLoRAModule`` (or any object with a
            ``training_step(batch) -> Tensor`` method).
        val_loader: Validation DataLoader.  If empty, returns
            ``float('inf')``.
        device: Target device (unused directly — module handles
            device transfers internally).

    Returns:
        Mean validation loss as a Python float, or ``float('inf')``
        if the loader yielded no valid batches.
    """
    total_loss = 0.0
    n_batches = 0

    was_training = getattr(module, "training", False)
    if hasattr(module, "model") and hasattr(module.model, "decoder"):
        module.model.decoder.eval()

    # Prevent training_step from updating mutable training state
    # (adaptive sampler, EMA, timestep buffer, metrics, loss history).
    _had_eval = getattr(module, "_eval_mode", False)
    module._eval_mode = True

    _run_seed = int(getattr(getattr(module, "training_config", None), "seed", 42) or 42)

    try:
        with _deterministic_rng(device, _run_seed + _VAL_SEED_OFFSET):
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        loss = module.training_step(batch)
                    except Exception:
                        logger.debug("[Validation] Skipping batch due to error", exc_info=True)
                        continue

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    total_loss += loss.item()
                    n_batches += 1
    finally:
        module._eval_mode = _had_eval
        if hasattr(module, "model") and hasattr(module.model, "decoder"):
            if was_training:
                module.model.decoder.train()

    if n_batches == 0:
        logger.warning("[Validation] No valid batches — returning inf")
        return float("inf")

    mean_loss = total_loss / n_batches
    return mean_loss
