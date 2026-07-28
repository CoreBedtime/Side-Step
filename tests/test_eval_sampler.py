"""Tests for the eval-harness sampler schedule math."""

from __future__ import annotations

import torch

from sidestep_engine.core.timestep_sampling import TURBO_SHIFT3_TIMESTEPS
from sidestep_engine.eval.sampler import build_timesteps


class TestBuildTimesteps:
    def test_no_shift_is_linspace(self) -> None:
        ts = build_timesteps(10, shift=1.0)
        assert torch.allclose(ts, torch.linspace(1.0, 0.0, 11))

    def test_endpoints(self) -> None:
        ts = build_timesteps(30, shift=3.0)
        assert float(ts[0]) == 1.0 and float(ts[-1]) == 0.0

    def test_shift3_reproduces_turbo_schedule(self) -> None:
        """The mined shift transform must reproduce upstream's hardcoded
        SHIFT_TIMESTEPS[3.0] list exactly — cross-validation of the formula
        against the official constant."""
        ts = build_timesteps(8, shift=3.0)
        expected = torch.tensor(TURBO_SHIFT3_TIMESTEPS, dtype=ts.dtype)
        assert torch.allclose(ts[:-1], expected, atol=1e-6), (
            f"{ts[:-1].tolist()} != {expected.tolist()}"
        )

    def test_monotonically_decreasing(self) -> None:
        for shift in (1.0, 3.0, 5.0):
            ts = build_timesteps(50, shift=shift)
            assert (ts.diff() < 0).all(), f"non-monotone schedule at shift={shift}"
