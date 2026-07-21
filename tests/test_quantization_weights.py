"""Unit tests for sidestep_engine.quantization.weights (requires ``quantize`` extra)."""

from __future__ import annotations

import importlib.util
import unittest

import torch.nn as nn


def _optimum_quanto_available() -> bool:
    return importlib.util.find_spec("optimum.quanto") is not None


class TestNormalizeQtypeString(unittest.TestCase):
    """MPS qfloat8 → qint8 mapping (no optimum import required)."""

    def test_cpu_unchanged(self) -> None:
        from sidestep_engine.quantization.weights import normalize_qtype_string

        self.assertEqual(normalize_qtype_string("qfloat8", device="cuda:0"), "qfloat8")


@unittest.skipUnless(
    _optimum_quanto_available(),
    "optimum-quanto not installed (install side-step[quantize])",
)
class TestQuantoHelpers(unittest.TestCase):
    """Tests that load optimum-quanto."""

    def test_get_qtype_qfloat8(self) -> None:
        from sidestep_engine.quantization.weights import get_qtype

        q = get_qtype("qfloat8")
        self.assertIsNotNone(q)

    def test_tiny_module_quantize_smoke(self) -> None:
        from sidestep_engine.quantization.weights import apply_weight_quantization

        m = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        # CPU smoke: quantize + freeze should not raise
        apply_weight_quantization(m, "qint8", device_hint="cpu")

    def test_qlinear_contiguous_hooks_registered(self) -> None:
        """int4 mm requires contiguous A; we pre-hook QLinear after freeze."""
        from sidestep_engine.quantization.weights import apply_weight_quantization

        m = nn.Sequential(nn.Linear(8, 4))
        apply_weight_quantization(m, "qint8", device_hint="cpu")
        hooks = 0
        for mod in m.modules():
            if mod.__class__.__name__ == "QLinear" and getattr(mod, "_forward_pre_hooks", None):
                hooks += len(mod._forward_pre_hooks)
        self.assertGreater(hooks, 0)

    def test_torchao_key_rejected_before_quantize(self) -> None:
        """TorchAO-only keys break PEFT LoRA; we fail fast with a clear error."""
        from sidestep_engine.quantization.weights import apply_weight_quantization

        m = nn.Sequential(nn.Linear(4, 2))
        with self.assertRaises(ValueError) as ctx:
            apply_weight_quantization(m, "int8", device_hint="cpu")
        self.assertIn("qint8", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
