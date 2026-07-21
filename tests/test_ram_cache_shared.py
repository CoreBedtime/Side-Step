"""RAM-cache shared-memory tests for PreprocessedTensorDataset.

The cache must live in shared memory so spawned DataLoader workers map
the same physical pages instead of copying the cache per worker.
"""

from __future__ import annotations

from pathlib import Path

import torch

from sidestep_engine.vendor.data_module import (
    PreprocessedTensorDataset,
    _share_sample_tensors,
)


def _write_fake_samples(tmp_path: Path, n: int = 2) -> None:
    for i in range(n):
        sample = {
            "target_latents": torch.randn(10, 64),
            "attention_mask": torch.ones(10),
            "encoder_hidden_states": torch.randn(5, 32),
            "encoder_attention_mask": torch.ones(5),
            "context_latents": torch.randn(10, 8),
        }
        torch.save(sample, tmp_path / f"sample_{i}.pt")


class TestShareSampleTensors:
    def test_shares_nested_structures(self) -> None:
        obj = {
            "a": torch.randn(4),
            "nested": {"b": torch.randn(2)},
            "listed": [torch.randn(3), {"c": torch.randn(1)}],
            "scalar": 1.5,
            "text": "hi",
        }
        _share_sample_tensors(obj)
        assert obj["a"].is_shared()
        assert obj["nested"]["b"].is_shared()
        assert obj["listed"][0].is_shared()
        assert obj["listed"][1]["c"].is_shared()


class TestRamCacheShared:
    def test_cache_tensors_are_shared(self, tmp_path: Path) -> None:
        _write_fake_samples(tmp_path)
        ds = PreprocessedTensorDataset(str(tmp_path), cache_in_ram=True)
        assert ds._ram_cache is not None and len(ds._ram_cache) == 2
        for sample in ds._ram_cache:
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    assert value.is_shared(), f"{key} not in shared memory"

    def test_getitem_serves_from_shared_cache(self, tmp_path: Path) -> None:
        _write_fake_samples(tmp_path)
        ds = PreprocessedTensorDataset(str(tmp_path), cache_in_ram=True)
        item = ds[0]
        assert "target_latents" in item
        assert item["target_latents"].shape[-1] == 64

    def test_cache_disabled_still_works(self, tmp_path: Path) -> None:
        _write_fake_samples(tmp_path)
        ds = PreprocessedTensorDataset(str(tmp_path), cache_in_ram=False)
        assert ds._ram_cache is None
        item = ds[0]
        assert "target_latents" in item
