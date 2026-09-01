"""Tests for introspection-based encoder offload.

``trainer_helpers.offload_non_decoder`` must walk the live model's
``named_children()`` instead of a hardcoded component name list -- the old
list carried ACE-Step 1.0-era names (``music_encoder``, ``vae``, ...) that
silently matched nothing on v1.5 checkpoints, making encoder offload a
complete no-op.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sidestep_engine.core.trainer_helpers import offload_non_decoder


class _FakeV15Model(nn.Module):
    """Mimics AceStepConditionGenerationModel's top-level layout."""

    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(2, 2)
        self.encoder = nn.Linear(2, 2)
        self.tokenizer = nn.Linear(2, 2)
        self.detokenizer = nn.Linear(2, 2)
        self.null_condition_emb = nn.Parameter(torch.zeros(1, 1, 2))


def _frozen(model: nn.Module) -> nn.Module:
    for p in model.parameters():
        p.requires_grad = False
    return model


def test_offloads_all_top_level_children_except_decoder():
    model = _frozen(_FakeV15Model())

    count = offload_non_decoder(model)

    assert count == 3  # encoder + tokenizer + detokenizer
    assert all(p.device.type == "cpu" for p in model.encoder.parameters())
    assert all(p.device.type == "cpu" for p in model.tokenizer.parameters())
    assert all(p.device.type == "cpu" for p in model.detokenizer.parameters())


def test_v15_module_names_are_matched():
    """Regression: v1.5 names must be found (the old hardcoded list missed
    them all, so the count was 0 and offload never fired)."""
    model = _frozen(_FakeV15Model())
    assert offload_non_decoder(model) > 0


def test_skips_modules_with_trainable_parameters():
    model = _frozen(_FakeV15Model())
    # Simulate an adapter injected outside the decoder -- must not offload.
    model.encoder.weight.requires_grad = True

    count = offload_non_decoder(model)

    assert count == 2  # tokenizer + detokenizer only
    assert model.encoder.weight.requires_grad


def test_meta_device_modules_counted_but_not_moved():
    """Modules left on meta by accelerate's hook-based offload must not
    crash ``.to()`` -- they count as already offloaded."""
    model = _frozen(_FakeV15Model())
    model.encoder.to_empty(device="meta")

    count = offload_non_decoder(model)

    assert count == 3
    assert next(model.encoder.parameters()).device.type == "meta"
