"""Tests for sidestep_engine.core.lora_resume_ladder.

Covers:
- ``_count_lora_keys`` / ``_remap_key`` helpers (pure functions).
- Primary strategy (``set_peft_model_state_dict``) with a real PEFT model.
- Auto-remap fallback variants.
- Hard-failure modes (no LoRA keys, no strategy matches).
"""

from __future__ import annotations

import unittest
from typing import Any, Dict

import torch
import torch.nn as nn


class _TinyMLP(nn.Module):
    """Minimal LoRA-injectable module for unit tests."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.k_proj(self.q_proj(x))


def _make_peft_model() -> nn.Module:
    """Build a small PEFT-wrapped model matching the production resume path."""
    from peft import LoraConfig, get_peft_model

    base = _TinyMLP()
    cfg = LoraConfig(
        r=4, lora_alpha=4, target_modules=["q_proj", "k_proj"], bias="none"
    )
    return get_peft_model(base, cfg)


def _capture_lora(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot LoRA tensors from *model*'s state_dict."""
    return {k: v.clone() for k, v in model.state_dict().items() if ".lora_" in k}


def _perturb_trainable(model: nn.Module) -> None:
    """Shift every trainable parameter so the next load has work to do."""
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(1.0)


class TestCountLoraKeys(unittest.TestCase):
    def test_recognises_lora_and_embedding_markers(self) -> None:
        from sidestep_engine.core.lora_resume_ladder import _count_lora_keys

        keys = [
            "base_model.model.q.lora_A.default.weight",
            "base_model.model.q.lora_B.default.weight",
            "embed.lora_embedding_A.default.weight",
            "other.weight",
        ]
        self.assertEqual(_count_lora_keys(keys), 3)

    def test_empty(self) -> None:
        from sidestep_engine.core.lora_resume_ladder import _count_lora_keys

        self.assertEqual(_count_lora_keys([]), 0)


class TestRemapKey(unittest.TestCase):
    def test_passthrough_unchanged(self) -> None:
        from sidestep_engine.core.lora_resume_ladder import _remap_key

        k = "base_model.model.q.lora_A.weight"
        self.assertEqual(_remap_key(k, "passthrough", "default"), k)

    def test_inject_adapter_inserts_name(self) -> None:
        from sidestep_engine.core.lora_resume_ladder import _remap_key

        k = "base_model.model.q.lora_A.weight"
        self.assertEqual(
            _remap_key(k, "inject_adapter", "default"),
            "base_model.model.q.lora_A.default.weight",
        )

    def test_add_base_prefix_only_when_missing(self) -> None:
        from sidestep_engine.core.lora_resume_ladder import _remap_key

        self.assertEqual(
            _remap_key("q.lora_A.weight", "add_base_prefix", "default"),
            "base_model.model.q.lora_A.weight",
        )
        # Idempotent when prefix is already present
        already = "base_model.model.q.lora_A.weight"
        self.assertEqual(_remap_key(already, "add_base_prefix", "default"), already)

    def test_prefix_and_inject(self) -> None:
        from sidestep_engine.core.lora_resume_ladder import _remap_key

        self.assertEqual(
            _remap_key("q.lora_A.weight", "prefix_and_inject", "default"),
            "base_model.model.q.lora_A.default.weight",
        )

    def test_inject_skips_non_lora_tensors(self) -> None:
        from sidestep_engine.core.lora_resume_ladder import _remap_key

        self.assertEqual(
            _remap_key("norm.weight", "inject_adapter", "default"),
            "norm.weight",
        )


class TestLadderPrimary(unittest.TestCase):
    """Primary strategy: ``set_peft_model_state_dict``."""

    def test_loads_matching_on_disk_format(self) -> None:
        from sidestep_engine.core.lora_resume_ladder import load_lora_resume_weights

        model = _make_peft_model()
        ref = _capture_lora(model)
        _perturb_trainable(model)

        # Mimic adapter_model.safetensors format: strip ``.default.``.
        on_disk = {k.replace(".default.", "."): v for k, v in ref.items()}

        result = load_lora_resume_weights(model, on_disk)

        self.assertEqual(result.strategy, "set_peft_model_state_dict")
        self.assertGreater(result.matched_lora_keys, 0)
        self.assertEqual(result.matched_lora_keys, result.total_lora_keys)
        for k, v in ref.items():
            torch.testing.assert_close(model.state_dict()[k], v)


class TestLadderAutoRemap(unittest.TestCase):
    """Fallback strategy: auto_remap variants."""

    def test_prefix_and_inject_recovers_when_primary_skipped(self) -> None:
        import sidestep_engine.core.lora_resume_ladder as mod

        model = _make_peft_model()
        ref = _capture_lora(model)
        _perturb_trainable(model)

        # Strip BOTH prefix and adapter name; a state_dict shape that only
        # the "prefix_and_inject" variant can remap onto the live model.
        stripped = {
            k.replace("base_model.model.", "").replace(".default.", "."): v
            for k, v in ref.items()
        }

        original_primary = mod._try_set_peft_state_dict
        mod._try_set_peft_state_dict = lambda *a, **kw: None  # simulate peft missing
        try:
            result = mod.load_lora_resume_weights(model, stripped)
        finally:
            mod._try_set_peft_state_dict = original_primary

        self.assertEqual(result.strategy, "auto_remap:prefix_and_inject")
        self.assertGreater(result.matched_lora_keys, 0)
        for k, v in ref.items():
            torch.testing.assert_close(model.state_dict()[k], v)


class TestLadderHardFailures(unittest.TestCase):
    """Regression guards: ladder must raise, never silently no-op."""

    def test_no_lora_keys_in_checkpoint_raises(self) -> None:
        from sidestep_engine.core.lora_resume_ladder import load_lora_resume_weights

        model = _make_peft_model()
        bogus: Dict[str, torch.Tensor] = {"base.weight": torch.zeros(4, 4)}
        with self.assertRaisesRegex(RuntimeError, "no LoRA keys"):
            load_lora_resume_weights(model, bogus)

    def test_all_lora_keys_mismatch_raises(self) -> None:
        from sidestep_engine.core.lora_resume_ladder import load_lora_resume_weights

        model = _make_peft_model()
        # LoRA-shaped keys that don't map to anything on the live model.
        bogus: Dict[str, torch.Tensor] = {
            "completely.other.path.lora_A.weight": torch.zeros(4, 8),
            "completely.other.path.lora_B.weight": torch.zeros(8, 4),
        }
        with self.assertRaisesRegex(RuntimeError, "no strategy matched"):
            load_lora_resume_weights(model, bogus)


if __name__ == "__main__":
    unittest.main()
