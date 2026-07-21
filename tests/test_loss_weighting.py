"""Unit tests for timestep loss-weighting math and its compositions.

Covers:
- min_snr uses the v-prediction Min-SNR-gamma form
- flow_snr normalization and clamping
- x0-loss t^2 compensation (exact velocity-equivalence property)
- adaptive-sampler importance correction properties
- deterministic validation RNG forking
"""

from __future__ import annotations

import torch

from sidestep_engine.core.adaptive_timestep import AdaptiveTimestepSampler
from sidestep_engine.core.loss_weighting import compute_timestep_weights
from sidestep_engine.core.validation import _VAL_SEED_OFFSET, _deterministic_rng


def _t(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# compute_timestep_weights
# ---------------------------------------------------------------------------

class TestNone:
    def test_returns_none(self) -> None:
        w, tel = compute_timestep_weights(
            _t(0.1, 0.5, 0.9), loss_weighting="none", snr_gamma=5.0, t_bias=0.5,
        )
        assert w is None and tel is None


class TestMinSnr:
    def test_v_prediction_formula(self) -> None:
        t = _t(0.2, 0.5, 0.8)
        w, _ = compute_timestep_weights(
            t, loss_weighting="min_snr", snr_gamma=5.0, t_bias=0.5,
        )
        snr = ((1.0 - t) / t) ** 2
        expected = torch.clamp(snr, max=5.0) / (snr + 1.0)
        assert torch.allclose(w, expected, atol=1e-6)

    def test_high_noise_end_weight(self) -> None:
        # t -> 1 (pure noise): SNR -> 0, so weight -> 0/(0+1) = ~0.
        # (The old epsilon-form min(SNR,g)/SNR gave ~1 here.)
        w, _ = compute_timestep_weights(
            _t(0.999), loss_weighting="min_snr", snr_gamma=5.0, t_bias=0.5,
        )
        assert w.item() < 0.01

    def test_low_noise_end_clamped(self) -> None:
        # t -> 0 (data): SNR huge, weight -> gamma / (SNR+1) -> small,
        # never above gamma.
        w, _ = compute_timestep_weights(
            _t(0.01), loss_weighting="min_snr", snr_gamma=5.0, t_bias=0.5,
        )
        assert 0.0 < w.item() < 5.0


class TestFlowSnr:
    def test_mean_normalized(self) -> None:
        t = torch.rand(512).clamp(1e-3, 1 - 1e-3)
        w, _ = compute_timestep_weights(
            t, loss_weighting="flow_snr", snr_gamma=5.0, t_bias=0.5,
        )
        assert abs(w.mean().item() - 1.0) < 1e-5

    def test_telemetry_clamped_at_gamma(self) -> None:
        t = _t(0.001, 0.999, 0.5)
        _, tel = compute_timestep_weights(
            t, loss_weighting="flow_snr", snr_gamma=5.0, t_bias=0.5,
        )
        assert tel.max().item() <= 5.0 + 1e-6


class TestX0Compensation:
    def test_velocity_equivalence(self) -> None:
        """w_x0 * (t^2 * L_vel) must equal w_vel * L_vel exactly."""
        t = torch.rand(256).clamp(1e-3, 1 - 1e-3)
        loss_vel = torch.rand(256) + 0.1  # arbitrary positive per-sample losses
        for weighting in ("flow_snr", "min_snr"):
            w_vel, _ = compute_timestep_weights(
                t, loss_weighting=weighting, snr_gamma=5.0, t_bias=0.5, x0_loss=False,
            )
            w_x0, _ = compute_timestep_weights(
                t, loss_weighting=weighting, snr_gamma=5.0, t_bias=0.5, x0_loss=True,
            )
            t_c = t.clamp(1e-4, 1 - 1e-4)
            loss_x0 = (t_c ** 2) * loss_vel  # implicit t^2 of x0 parameterization
            assert torch.allclose(w_x0 * loss_x0, w_vel * loss_vel, rtol=1e-4), weighting

    def test_no_compensation_without_weighting(self) -> None:
        w, _ = compute_timestep_weights(
            _t(0.3), loss_weighting="none", snr_gamma=5.0, t_bias=0.5, x0_loss=True,
        )
        assert w is None  # x0 alone keeps its intentional implicit weighting


# ---------------------------------------------------------------------------
# Adaptive sampler importance correction
# ---------------------------------------------------------------------------

class TestImportanceWeights:
    def test_ratio_zero_is_identity(self) -> None:
        s = AdaptiveTimestepSampler(ratio=0.0)
        t = torch.rand(128).clamp(1e-3, 1 - 1e-3)
        iw = s.importance_weights(t)
        assert torch.allclose(iw, torch.ones_like(iw), atol=1e-5)

    def test_mean_one(self) -> None:
        s = AdaptiveTimestepSampler(ratio=0.5)
        s._bin_loss = torch.rand(s.n_bins) + 0.1
        t = torch.rand(256).clamp(1e-3, 1 - 1e-3)
        iw = s.importance_weights(t)
        assert abs(iw.mean().item() - 1.0) < 1e-5

    def test_oversampled_bins_downweighted(self) -> None:
        """Timesteps in a bin the sampler favors must get lower weight than
        the same timesteps would if the bin were not favored."""
        hot, cold = AdaptiveTimestepSampler(ratio=0.5), AdaptiveTimestepSampler(ratio=0.5)
        hot._bin_loss = torch.ones(hot.n_bins)
        hot._bin_loss[5] = 100.0  # bin [0.5, 0.6) heavily oversampled
        t_in_hot_bin = _t(0.52, 0.55, 0.58)
        # Compare unnormalized ratio via a mixed batch: hot-bin samples
        # should carry relatively less weight under the hot sampler.
        t = torch.cat([t_in_hot_bin, _t(0.12, 0.32, 0.82)])
        iw_hot = hot.importance_weights(t)
        iw_uniform = cold.importance_weights(t)
        rel_hot = iw_hot[:3].mean() / iw_hot[3:].mean()
        rel_uniform = iw_uniform[:3].mean() / iw_uniform[3:].mean()
        assert rel_hot < rel_uniform


# ---------------------------------------------------------------------------
# Deterministic validation RNG
# ---------------------------------------------------------------------------

class TestDeterministicRng:
    def test_same_draws_every_invocation(self) -> None:
        device = torch.device("cpu")
        with _deterministic_rng(device, 1234):
            a = torch.rand(16)
        with _deterministic_rng(device, 1234):
            b = torch.rand(16)
        assert torch.equal(a, b)

    def test_outer_stream_unperturbed(self) -> None:
        torch.manual_seed(0)
        _ = torch.rand(4)
        expected_next = torch.get_rng_state()
        torch.manual_seed(0)
        _ = torch.rand(4)
        with _deterministic_rng(torch.device("cpu"), 999):
            _ = torch.rand(1000)  # heavy use inside the fork
        assert torch.equal(torch.get_rng_state(), expected_next)

    def test_seed_offset_defined(self) -> None:
        assert isinstance(_VAL_SEED_OFFSET, int) and _VAL_SEED_OFFSET > 0
