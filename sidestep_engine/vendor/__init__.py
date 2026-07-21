"""Vendored ACE-Step modules for standalone Side-Step operation.

The package intentionally exposes legacy re-export names lazily.  Importing a
single lightweight submodule such as ``sidestep_engine.vendor.configs`` should
not import torch, torchaudio, or other training-time dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    # Configs
    "LoRAConfig": "configs",
    "LoKRConfig": "configs",
    "TrainingConfig": "configs",
    # LoRA utilities
    "check_peft_available": "lora_utils",
    "inject_lora_into_dit": "lora_utils",
    "load_lora_weights": "lora_utils",
    "load_training_checkpoint": "lora_utils",
    "save_lora_weights": "lora_utils",
    "save_training_checkpoint": "lora_utils",
    # LoKR utilities
    "check_lycoris_available": "lokr_utils",
    "inject_lokr_into_dit": "lokr_utils",
    "load_lokr_weights": "lokr_utils",
    "save_lokr_training_checkpoint": "lokr_utils",
    "save_lokr_weights": "lokr_utils",
    # Data module
    "PreprocessedDataModule": "data_module",
    # Preprocessing utilities
    "load_audio_stereo": "preprocess_audio",
    "encode_lyrics": "preprocess_lyrics",
    "run_encoder": "preprocess_encoder",
    "build_context_latents": "preprocess_context",
    "encode_text": "preprocess_text",
    # Constants
    "DEFAULT_DIT_INSTRUCTION": "constants",
    "SFT_GEN_PROMPT": "constants",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load legacy re-exported symbols on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
