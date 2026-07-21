"""
ACE-Step Compatibility Check for Side-Step.

Side-Step bundles vendored copies of ACE-Step utilities for its corrected
(fixed) training loop, so a full ACE-Step installation is no longer required.

This module checks that critical vendored modules are importable.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version pin
# ---------------------------------------------------------------------------

TESTED_ACESTEP_COMMIT = "46116a6"
"""Short SHA of the upstream ``ace-step/ACE-Step-1.5`` commit that the
vendored files were last synced from."""

SIDESTEP_VERSION = "1.1.2-beta"
"""Current Side-Step release string."""


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

def _source_defines_symbol(module_name: str, symbol_name: str) -> tuple[bool, str]:
    """Return whether a module source file defines a symbol, without importing it."""
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        return False, "module not found"

    path = Path(spec.origin)
    if not path.is_file():
        return False, f"source file not found: {path}"

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        return False, f"could not inspect source: {exc}"

    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == symbol_name:
                return True, ""
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol_name:
                    return True, ""
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == symbol_name:
                return True, ""

    return False, f"{symbol_name} not found in {path.name}"


def configure_cuda_allocator() -> None:
    """Opt into PyTorch's expandable-segments CUDA allocator.

    Side-Step trains on variable-length audio latents, the worst case for
    the default caching allocator: blocks sized for one song can't be
    reused for the next, so reserved VRAM fragments and ratchets upward
    over a run (it looks like a memory leak in nvidia-smi).  Expandable
    segments (PyTorch >= 2.1) back allocations with growable virtual
    address ranges instead of fixed cudaMalloc segments, which largely
    eliminates that fragmentation.

    Must run before the first CUDA allocation.  Respects an existing
    user-provided ``PYTORCH_CUDA_ALLOC_CONF``.  On platforms where
    expandable segments are unsupported, PyTorch logs a warning and
    falls back to the default allocator — safe either way.
    """
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True",
    )


def check_compatibility() -> None:
    """Verify that critical symbols exist.

    Checks vendored modules (required). Non-fatal: prints warnings and
    continues.
    """
    warnings: list[str] = []

    # 1. Vendored modules (required for fixed training)
    checks = (
        ("sidestep_engine.vendor.data_module", "PreprocessedDataModule"),
        ("sidestep_engine.vendor.lora_utils", "inject_lora_into_dit"),
        ("sidestep_engine.vendor.configs", "TrainingConfig"),
    )
    for module_name, symbol_name in checks:
        ok, reason = _source_defines_symbol(module_name, symbol_name)
        if not ok:
            warnings.append(f"Cannot find vendored {module_name}.{symbol_name}: {reason}")

    if warnings:
        msg = (
            f"[Side-Step] Compatibility warning (vendored from ACE-Step "
            f"commit {TESTED_ACESTEP_COMMIT}):\n"
        )
        for w in warnings:
            msg += f"  - {w}\n"
        msg += (
            "  Side-Step's corrected training may not work.\n"
            "  Try reinstalling Side-Step or check for missing files."
        )
        logger.warning(msg)
        print(f"\n{msg}\n")
    else:
        logger.debug(
            "[Side-Step] Compatibility check passed (pin: %s)",
            TESTED_ACESTEP_COMMIT,
        )

    # Flash-attn availability check (wheels only built for Python 3.11)
    if sys.version_info[:2] != (3, 11):
        fa_msg = (
            f"[Side-Step] Python {sys.version_info[0]}.{sys.version_info[1]} detected. "
            "Flash Attention 2 wheels are only available for Python 3.11. "
            "Training will fall back to standard attention and may use more VRAM."
        )
        logger.warning(fa_msg)
        print(f"\n{fa_msg}\n")
