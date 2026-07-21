"""Parity tests: every config surface must agree with the canonical schema.

The schema (``sidestep_engine/core/schema.py``) is the single source of
truth for training options.  These tests lock the surfaces that are still
hand-written (dataclass defaults, static defaults.json, frontend element
IDs) to the schema so they cannot silently drift again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from sidestep_engine.core.schema import (
    SCHEMA,
    SCHEMA_BY_NAME,
    defaults_dict,
    gui_field_map,
    to_json_schema,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# CLI <-> schema
# ---------------------------------------------------------------------------

def _train_subparser() -> argparse.ArgumentParser:
    from sidestep_engine.cli.args import build_root_parser

    root = build_root_parser()
    for action in root._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices["train"]
    raise AssertionError("train subparser not found")


def test_cli_train_surface_matches_schema() -> None:
    """Every schema field with CLI flags appears on the train subcommand
    with the schema's flags, default, and choices."""
    train = _train_subparser()
    actions_by_dest = {a.dest: a for a in train._actions}

    for f in SCHEMA:
        if not f.cli:
            continue
        assert f.name in actions_by_dest, f"schema field {f.name} missing from CLI"
        a = actions_by_dest[f.name]
        # BooleanOptionalAction auto-generates --no-* companions; compare
        # only the schema-declared flags.
        declared = tuple(s for s in a.option_strings if not s.startswith("--no-"))
        assert declared == f.cli, (
            f"{f.name}: CLI flags {declared} != schema {f.cli}"
        )
        expected_default = (
            None if f.cli_default_none
            else list(f.default) if isinstance(f.default, tuple)
            else f.default
        )
        assert a.default == expected_default, (
            f"{f.name}: CLI default {a.default!r} != schema {expected_default!r}"
        )
        expected_choices = list(f.choices) if f.choices is not None else None
        actual_choices = list(a.choices) if a.choices is not None else None
        assert actual_choices == expected_choices, (
            f"{f.name}: CLI choices {actual_choices} != schema {expected_choices}"
        )


def test_cli_has_no_orphan_training_args() -> None:
    """Every non-suppressed argparse action on the train subcommand either
    comes from the schema or is on the documented hand-written list."""
    hand_written = {
        "help", "config",
        # Inline preprocessing group (preprocess params, not training config)
        "preprocess", "preprocess_only", "audio_dir", "dataset_json",
        "tensor_output", "max_duration", "normalize", "target_db", "target_lufs",
    }
    train = _train_subparser()
    for a in train._actions:
        if isinstance(a, argparse._HelpAction):
            continue
        assert a.dest in SCHEMA_BY_NAME or a.dest in hand_written, (
            f"CLI arg --{a.dest} is hand-written but not in the schema or "
            f"the documented hand-written list; add it to core/schema.py"
        )


# ---------------------------------------------------------------------------
# TrainingConfigV2 dataclass <-> schema
# ---------------------------------------------------------------------------

# schema name -> dataclass field name (where they differ)
_CONFIG_NAME_MAP = {
    "epochs": "max_epochs",
    "gradient_accumulation": "gradient_accumulation_steps",
    "save_every": "save_every_n_epochs",
}

# Documented, intentional divergences between the public schema default and
# the dataclass default.
_CONFIG_EXEMPT = {
    # Public default is "auto"; the dataclass stores the *resolved* value
    # (config_factory calls resolve_optimizer_type before construction).
    "optimizer_type",
    # Platform-dependent in the schema (2 on Windows); static 4 in dataclass.
    "num_workers",
    # Dataclass uses None to mean "unset"; public default is 0 (disabled).
    "max_latent_length",
}


def test_training_config_defaults_match_schema() -> None:
    from sidestep_engine.core.configs import TrainingConfigV2

    fields = TrainingConfigV2.__dataclass_fields__
    mismatches = []
    for f in SCHEMA:
        if not f.aggregate or f.name in _CONFIG_EXEMPT:
            continue
        cfg_name = _CONFIG_NAME_MAP.get(f.name, f.name)
        if cfg_name not in fields:
            continue  # adapter-config field or not represented on the dataclass
        dc_default = fields[cfg_name].default
        if dc_default != f.default:
            mismatches.append(f"{cfg_name}: dataclass={dc_default!r} schema={f.default!r}")
    assert not mismatches, (
        "TrainingConfigV2 defaults drifted from the schema:\n  " + "\n  ".join(mismatches)
    )


def test_adapter_config_defaults_match_schema() -> None:
    from sidestep_engine.core.configs import LoRAConfigV2

    fields = LoRAConfigV2.__dataclass_fields__
    checks = {"rank": "r", "alpha": "alpha", "dropout": "dropout",
              "attention_type": "attention_type", "target_mlp": "target_mlp"}
    for schema_name, dc_name in checks.items():
        assert fields[dc_name].default == SCHEMA_BY_NAME[schema_name].default, (
            f"LoRAConfigV2.{dc_name} default drifted from schema field {schema_name}"
        )


# ---------------------------------------------------------------------------
# training_defaults aggregate <-> schema (derivation sanity)
# ---------------------------------------------------------------------------

def test_training_defaults_derived_from_schema() -> None:
    from sidestep_engine.training_defaults import TRAINING_DEFAULTS, GUI_FIELD_MAP

    assert TRAINING_DEFAULTS == defaults_dict()
    assert GUI_FIELD_MAP == gui_field_map()


def test_gui_key_map_targets_exist() -> None:
    """Every GUI_KEY_MAP target must be a known schema field (or a documented
    non-schema passthrough)."""
    from sidestep_engine.training_defaults import GUI_KEY_MAP

    passthrough = {"target_modules", "self_target_modules", "cross_target_modules"}
    for gui_key, backend_name in GUI_KEY_MAP.items():
        assert backend_name in SCHEMA_BY_NAME or backend_name in passthrough, (
            f"GUI_KEY_MAP[{gui_key!r}] -> {backend_name!r} is not a schema field"
        )


# ---------------------------------------------------------------------------
# Frontend static fallback <-> schema
# ---------------------------------------------------------------------------

def test_frontend_defaults_json_up_to_date() -> None:
    """frontend/js/defaults.json must match what the generator produces."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_frontend_defaults", PROJECT_ROOT / "scripts" / "gen_frontend_defaults.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    expected = json.dumps(gen.build(), indent=2, ensure_ascii=False) + "\n"
    actual = (PROJECT_ROOT / "frontend" / "js" / "defaults.json").read_text(encoding="utf-8")
    assert actual == expected, (
        "frontend/js/defaults.json is stale. Regenerate with: "
        "uv run python scripts/gen_frontend_defaults.py"
    )


def test_gui_ids_exist_in_frontend() -> None:
    """Every schema gui_id must correspond to a real element in index.html."""
    index = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    missing = [
        f.gui_id for f in SCHEMA
        if f.gui_id and f'id="{f.gui_id}"' not in index
    ]
    assert not missing, f"schema gui_ids not found in frontend/index.html: {missing}"


# ---------------------------------------------------------------------------
# JSON schema endpoint payload
# ---------------------------------------------------------------------------

def test_json_schema_serializes() -> None:
    payload = to_json_schema()
    assert payload["version"] == 1
    names = [e["name"] for e in payload["fields"]]
    assert len(names) == len(set(names))
    assert "learning_rate" in names
    # help text must be de-escaped for non-argparse consumers
    for e in payload["fields"]:
        assert "%%" not in e["help"], f"{e['name']} help not de-escaped"
    # payload must be JSON-serializable end to end
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Presets reference only known fields
# ---------------------------------------------------------------------------

def test_builtin_presets_use_known_fields() -> None:
    """Built-in presets must not reference unknown training parameters."""
    from sidestep_engine.training_defaults import GUI_KEY_MAP

    presets_dir = PROJECT_ROOT / "presets"
    if not presets_dir.is_dir():
        pytest.skip("no presets directory")

    # Non-schema keys that presets/config dicts legitimately carry.
    extra_ok = {
        "name", "description", "_comment", "preset_name",
        "target_modules", "self_target_modules", "cross_target_modules",
        "target_modules_str", "self_target_modules_str", "cross_target_modules_str",
        "genre_ratio", "vram_profile", "shift", "num_inference_steps",
        "checkpoint_dir", "dataset_dir", "output_dir", "resume_from",
        "run_name", "log_dir", "chunk_duration", "crop_mode",
        "timestep_mu", "timestep_sigma",
    }
    unknown = []
    for path in sorted(presets_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        params = data.get("params", data)
        if not isinstance(params, dict):
            continue
        for key in params:
            backend = GUI_KEY_MAP.get(key, key)
            if backend not in SCHEMA_BY_NAME and backend not in extra_ok:
                unknown.append(f"{path.name}: {key}")
    assert not unknown, (
        "presets reference parameters unknown to the schema:\n  " + "\n  ".join(unknown)
    )
