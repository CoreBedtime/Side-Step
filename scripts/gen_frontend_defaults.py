"""Regenerate frontend/js/defaults.json from the canonical schema.

The GUI fetches live defaults from ``/api/defaults`` at runtime;
``defaults.json`` is only the static fallback bundled with the frontend.
This script keeps that fallback in lockstep with the schema instead of
letting it drift by hand-editing.

Per the file's convention, platform-dependent values are written with
their **Linux** defaults (the API serves correct per-platform values at
runtime), and settings paths use forward slashes.

Usage:
    uv run python scripts/gen_frontend_defaults.py [--check]

``--check`` exits non-zero if the file is out of date (for CI / tests).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_PATH = PROJECT_ROOT / "frontend" / "js" / "defaults.json"

_COMMENT = (
    "GENERATED FILE - do not edit by hand. Regenerate with: "
    "uv run python scripts/gen_frontend_defaults.py. "
    "Static fallback for GUI field defaults; the /api/defaults endpoint "
    "(from the schema in sidestep_engine/core/schema.py) is the true source "
    "of truth and overrides this file at runtime. Platform-dependent values "
    "(num_workers) use Linux defaults here."
)


def build() -> dict:
    from sidestep_engine.training_defaults import get_gui_defaults

    out = {"_comment": _COMMENT}
    out.update(get_gui_defaults())

    # Linux-flavored statics regardless of the platform running this script.
    out["full-num-workers"] = "4"
    for key in (
        "settings-checkpoint-dir",
        "settings-adapters-dir",
        "settings-tensors-dir",
        "settings-audio-dir",
        "settings-exported-loras-dir",
    ):
        out[key] = out[key].replace("\\", "/")
    return out


def main() -> int:
    content = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        current = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
        if current != content:
            print(f"OUT OF DATE: {OUT_PATH} does not match the schema.")
            print("Regenerate with: uv run python scripts/gen_frontend_defaults.py")
            return 1
        print("defaults.json is up to date.")
        return 0
    OUT_PATH.write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote {OUT_PATH} ({len(build())} keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
