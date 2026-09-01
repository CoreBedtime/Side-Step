"""
Argparse construction for Side-Step CLI.

Contains ``build_root_parser`` and all ``_add_*`` argument-group helpers.

Training-related arguments are **generated from the canonical schema**
(``sidestep_engine.core.schema``): flags, defaults, choices, and help all
come from one ``SchemaField`` record per option.  To add a training option,
add it to the schema — it appears here automatically.  Non-training
subcommands (captions, tags, export, …) remain hand-written.
"""

from __future__ import annotations

import argparse

from sidestep_engine.core.schema import (
    S_ADAPTER,
    S_ADVANCED,
    S_CKPT,
    S_DATA,
    S_DEVICE,
    S_LEVERS,
    S_LOG,
    S_LOHA,
    S_LOKR,
    S_LORA,
    S_MODEL,
    S_OFT,
    S_TRAINING,
    SchemaField,
    fields_for_section,
    render_help,
)

from sidestep_engine.core.constants import VARIANT_DIR_MAP

# Backward-compat alias (re-exported by cli/common.py).
from sidestep_engine.training_defaults import (  # noqa: F401
    DEFAULT_NUM_WORKERS as _DEFAULT_NUM_WORKERS,
)


# ===========================================================================
# Schema -> argparse rendering
# ===========================================================================

_PY_TYPES = {"int": int, "float": float, "str": str}


def _add_schema_argument(group: argparse._ArgumentGroup, f: SchemaField) -> None:
    """Add one schema field to an argparse group, preserving the historical
    hand-written surface (flags, dest, type, default, choices, help)."""
    assert f.cli, f"field {f.name} has no CLI flags"
    kwargs: dict = {"dest": f.name}
    if f.cli_action == "bool_optional":
        kwargs["action"] = argparse.BooleanOptionalAction
        kwargs["default"] = f.default
    elif f.cli_action == "store_true":
        kwargs["action"] = "store_true"
        kwargs["default"] = f.default
    else:
        if f.type in _PY_TYPES:
            kwargs["type"] = _PY_TYPES[f.type]
        if f.cli_default_none:
            kwargs["default"] = None
        elif isinstance(f.default, tuple):
            kwargs["default"] = list(f.default)
        else:
            kwargs["default"] = f.default
        if f.choices is not None:
            kwargs["choices"] = list(f.choices)
        if f.cli_nargs:
            kwargs["nargs"] = f.cli_nargs
        if f.cli_metavar:
            kwargs["metavar"] = f.cli_metavar
    kwargs["help"] = argparse.SUPPRESS if f.cli_suppressed else render_help(f)
    group.add_argument(*f.cli, **kwargs)


def _add_schema_section(
    parser: argparse.ArgumentParser, section: str,
) -> argparse._ArgumentGroup:
    """Create an argument group titled *section* and fill it from the schema."""
    group = parser.add_argument_group(section)
    for f in fields_for_section(section):
        _add_schema_argument(group, f)
    return group


# ===========================================================================
# Root parser
# ===========================================================================

def build_root_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with all subcommands."""

    formatter_class = argparse.HelpFormatter
    try:
        from sidestep_engine.ui.help_formatter import RichHelpFormatter
        formatter_class = RichHelpFormatter
    except ImportError:
        pass

    root = argparse.ArgumentParser(
        prog="sidestep",
        description="Side-Step -- LoRA/LoKR fine-tuning CLI",
        formatter_class=formatter_class,
    )

    root.add_argument(
        "--plain",
        action="store_true",
        default=False,
        help="Disable Rich output; use plain text (also set automatically when stdout is not a TTY)",
    )
    root.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Skip the confirmation prompt and start immediately",
    )
    # --gui kept as hidden root flag for backward compat (translated by deprecation shim)
    root.add_argument("--gui", action="store_true", default=False, help=argparse.SUPPRESS)
    root.add_argument("--port", type=int, default=8770, help=argparse.SUPPRESS)

    subparsers = root.add_subparsers(dest="subcommand")

    # -- train (was: fixed) --------------------------------------------------
    p_train = subparsers.add_parser(
        "train",
        help="Train an adapter (LoRA, DoRA, LoKR, LoHA, OFT)",
        formatter_class=formatter_class,
    )
    _add_common_training_args(p_train)
    _add_train_args(p_train)

    # -- preprocess (promoted to top-level) ----------------------------------
    p_preprocess = subparsers.add_parser(
        "preprocess",
        help="Preprocess audio into tensors (two-pass pipeline)",
        formatter_class=formatter_class,
    )
    _add_preprocess_subcommand_args(p_preprocess)

    # -- analyze (was: fisher) -----------------------------------------------
    p_analyze = subparsers.add_parser(
        "analyze",
        help="PP++ / Fisher analysis for adaptive LoRA rank assignment",
        formatter_class=formatter_class,
    )
    _add_model_args(p_analyze)
    _add_device_args(p_analyze)
    _add_fisher_args(p_analyze)

    # -- audio-analyze --------------------------------------------------------
    p_aa = subparsers.add_parser(
        "audio-analyze",
        help="Local offline audio analysis: extract BPM, key, and time signature",
        formatter_class=formatter_class,
    )
    _add_audio_analyze_args(p_aa)

    # -- captions ------------------------------------------------------------
    p_cap = subparsers.add_parser(
        "captions",
        help="Generate AI captions + fetch lyrics for audio sidecar files",
        formatter_class=formatter_class,
    )
    _add_captions_args(p_cap)

    # -- tags ----------------------------------------------------------------
    p_tags = subparsers.add_parser(
        "tags",
        help="Bulk sidecar tag operations (add, remove, list, clear trigger tags)",
        formatter_class=formatter_class,
    )
    _add_tags_args(p_tags)

    # -- dataset (was: build-dataset) ----------------------------------------
    p_dataset = subparsers.add_parser(
        "dataset",
        help="Build dataset.json from a folder of audio + sidecar metadata files",
        formatter_class=formatter_class,
    )
    p_dataset.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Root directory containing audio files (scanned recursively)",
    )
    p_dataset.add_argument(
        "--tag",
        type=str,
        default="",
        help="Custom trigger tag applied to all samples (default: none)",
    )
    p_dataset.add_argument(
        "--tag-position",
        type=str,
        default="prepend",
        choices=["prepend", "append", "replace"],
        help="Tag placement in prompts (default: prepend)",
    )
    p_dataset.add_argument(
        "--genre-ratio",
        type=int,
        default=0,
        help="Percentage of samples that use genre instead of caption (0-100, default: 0)",
    )
    p_dataset.add_argument(
        "--name",
        type=str,
        default="local_dataset",
        help="Dataset name in metadata block (default: local_dataset)",
    )
    p_dataset.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <input>/dataset.json)",
    )

    # -- convert-sidecars ----------------------------------------------------
    p_convert = subparsers.add_parser(
        "convert-sidecars",
        help="Convert per-file JSON or dataset.json metadata to TXT sidecars",
        formatter_class=formatter_class,
    )
    p_convert.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Directory with per-file .json sidecars, or path to a dataset.json",
    )
    p_convert.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Audio directory (for dataset.json mode; defaults to JSON parent dir)",
    )
    p_convert.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing .txt sidecars (default: skip files that already have one)",
    )
    p_convert.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Skip confirmation prompt",
    )

    # -- settings ------------------------------------------------------------
    p_settings = subparsers.add_parser(
        "settings",
        help="View or modify Side-Step persistent settings",
        formatter_class=formatter_class,
    )
    _add_settings_args(p_settings)

    # -- history -------------------------------------------------------------
    p_history = subparsers.add_parser(
        "history",
        help="List past training runs",
        formatter_class=formatter_class,
    )
    p_history.add_argument(
        "--limit", type=int, default=20,
        help="Maximum number of runs to show (default: 20)",
    )
    p_history.add_argument(
        "--json", action="store_true", default=False, dest="json_output",
        help="Output raw JSON instead of a table",
    )

    # -- export --------------------------------------------------------------
    p_export = subparsers.add_parser(
        "export",
        help="Export adapter to ComfyUI format",
        formatter_class=formatter_class,
    )
    p_export.add_argument(
        "adapter_dir",
        type=str,
        help="Path to adapter directory (e.g. output/my_lora/final)",
    )
    p_export.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output .safetensors file name (default: <adapter_dir_name>_comfyui.safetensors)",
    )
    p_export.add_argument(
        "--format", "-f",
        type=str,
        default="comfyui",
        choices=["comfyui"],
        dest="export_format",
        help="Export format (default: comfyui)",
    )
    p_export.add_argument(
        "--target", "-t",
        type=str,
        default="native",
        choices=["native", "generic"],
        help=(
            "ComfyUI target format (default: native). "
            "native = built-in ACE-Step 1.5 support (base_model.model prefix); "
            "generic = diffusion_model.decoder prefix (try if native fails)"
        ),
    )
    p_export.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Advanced: explicit key prefix override (ignores --target)",
    )
    p_export.add_argument(
        "--normalize-alpha",
        action="store_true",
        default=False,
        dest="normalize_alpha",
        help="Set alpha=rank so ComfyUI strength 1.0 = natural LoRA magnitude",
    )

    # -- gui -----------------------------------------------------------------
    p_gui = subparsers.add_parser(
        "gui",
        help="Launch the web GUI",
        formatter_class=formatter_class,
    )
    p_gui.add_argument(
        "--port", type=int, default=8770,
        help="GUI server port (default: 8770)",
    )

    return root


# ===========================================================================
# Argument groups
# ===========================================================================

def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add --model (was --model-variant) and --checkpoint-dir (schema-driven)."""
    _add_schema_section(parser, S_MODEL)


def _add_device_args(parser: argparse.ArgumentParser) -> None:
    """Add --device and --precision (schema-driven)."""
    _add_schema_section(parser, S_DEVICE)


def _add_common_training_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by training subcommands (schema-driven)."""
    _add_model_args(parser)
    _add_device_args(parser)

    # Schema-driven groups, in historical display order.
    _add_schema_section(parser, S_DATA)
    _add_schema_section(parser, S_TRAINING)
    _add_schema_section(parser, S_LEVERS)
    _add_schema_section(parser, S_ADAPTER)
    _add_schema_section(parser, S_LORA)
    _add_schema_section(parser, S_LOKR)
    _add_schema_section(parser, S_LOHA)
    _add_schema_section(parser, S_OFT)

    # -- Config file ---------------------------------------------------------
    g_cfg = parser.add_argument_group("Config file")
    g_cfg.add_argument("--config", type=str, default=None,
                       help="Load training config from JSON file. CLI args override JSON values.")

    _add_schema_section(parser, S_CKPT)
    _add_schema_section(parser, S_LOG)

    # -- Inline preprocessing (chained: preprocess then train) ---------------
    g_pre = parser.add_argument_group("Inline preprocessing")
    g_pre.add_argument("--preprocess", action="store_true", default=False,
                       help="Preprocess audio into tensors, then continue to training")
    g_pre.add_argument("--preprocess-only", action="store_true", default=False,
                       help="Run preprocessing and exit (do not train)")
    g_pre.add_argument("--audio-dir", type=str, default=None, help="Source audio directory (preprocessing)")
    g_pre.add_argument("--dataset-json", type=str, default=None, help="Labeled dataset JSON file (preprocessing)")
    g_pre.add_argument("--tensor-output", type=str, default=None, help="Output directory for .pt tensor files (preprocessing)")
    g_pre.add_argument("--max-duration", type=float, default=0, help="Max audio duration in seconds (0 = auto-detect from dataset, default: 0)")
    g_pre.add_argument("--normalize", type=str, default="none", choices=["none", "peak", "lufs"],
                        help="Audio normalization: none, peak (-1.0 dBFS), lufs (-14 LUFS). LUFS requires pyloudnorm (default: none)")
    g_pre.add_argument("--target-db", type=float, default=-1.0,
                        help="Peak normalization target in dBFS (used with --normalize peak, default: -1.0)")
    g_pre.add_argument("--target-lufs", type=float, default=-14.0,
                        help="LUFS normalization target (used with --normalize lufs, default: -14.0)")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the train subcommand (schema-driven)."""
    _add_schema_section(parser, S_ADVANCED)


def _add_preprocess_subcommand_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the top-level ``preprocess`` subcommand."""
    _add_model_args(parser)
    _add_device_args(parser)
    g = parser.add_argument_group("Preprocessing")
    g.add_argument("--audio-dir", "-i", "--input",
                   type=str, default=None, dest="audio_dir",
                   help="Source audio directory (scanned recursively)")
    g.add_argument("--dataset-json", type=str, default=None,
                   help="Labeled dataset JSON file (alternative to --audio-dir)")
    g.add_argument("--output", "-o", "--tensor-output",
                   type=str, default=None, dest="tensor_output",
                   help="Output directory for .pt tensor files")
    g.add_argument("--max-duration", type=float, default=0,
                   help="Max audio duration in seconds (0 = auto-detect, default: 0)")
    g.add_argument("--normalize", type=str, default="none",
                   choices=["none", "peak", "lufs"],
                   help="Audio normalization: none, peak, lufs (default: none)")
    g.add_argument("--target-db", type=float, default=-1.0,
                   help="Peak normalization target in dBFS (default: -1.0)")
    g.add_argument("--target-lufs", type=float, default=-14.0,
                   help="LUFS normalization target (default: -14.0)")


def _add_fisher_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ``analyze`` subcommand (Fisher + Spectral)."""
    g = parser.add_argument_group("PP++ / Fisher analysis")
    g.add_argument("--dataset-dir", "-d", type=str, required=True,
                   help="Directory containing preprocessed .pt files")
    g.add_argument("--rank", "-r", type=int, default=64,
                   help="Base LoRA rank (median target, default: 64)")
    g.add_argument("--rank-min", type=int, default=16,
                   help="Minimum adaptive rank (default: 16)")
    g.add_argument("--rank-max", type=int, default=128,
                   help="Maximum adaptive rank (default: 128)")
    g.add_argument("--timestep-focus", type=str, default="balanced",
                   help="Timestep focus: balanced (default), texture, structure, or low,high")
    g.add_argument("--runs", "--fisher-runs", type=int, default=None, dest="runs",
                   help="Number of estimation runs (default: auto from dataset size)")
    g.add_argument("--batches", "--fisher-batches", type=int, default=None, dest="batches",
                   help="Batches per run (default: auto from dataset size)")
    g.add_argument("--convergence-patience", type=int, default=5,
                   help="Early stop when ranking stable for N batches (default: 5)")
    g.add_argument("--output", type=str, default=None, dest="fisher_output",
                   help="Override output path for fisher_map.json")


def _add_audio_analyze_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ``audio-analyze`` subcommand."""
    g = parser.add_argument_group("Audio analysis")
    g.add_argument(
        "--input", "-i", type=str, required=True,
        help="Directory containing audio files to analyze (scanned recursively)",
    )
    g.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cuda, cpu (default: auto)",
    )
    g.add_argument(
        "--policy", type=str, default="fill_missing",
        choices=["fill_missing", "overwrite_all"],
        help="Merge policy for existing sidecar fields (default: fill_missing)",
    )
    g.add_argument(
        "--mode", type=str, default="standard",
        choices=["standard", "sas", "faf", "mid"],
        help=(
            "Analysis quality: standard (default, tempo + time signature, ~1s), "
            "sas (adds Demucs-separated key, ~20-30s). 'faf' and 'mid' are "
            "accepted for backwards compatibility and map to standard"
        ),
    )
    g.add_argument(
        "--tempo-centre", "--tempo-center", type=float, default=None,
        dest="tempo_centre",
        help=(
            "Centre of the tempo prior in BPM (default: 130). Lower it for "
            "libraries of predominantly slow material, where the default "
            "would wrongly double a genuine slow tempo"
        ),
    )
    g.add_argument(
        "--chunks", type=int, default=5,
        help="Number of analysis chunks for sas key detection (default: 5)",
    )


def _add_captions_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ``captions`` subcommand."""
    g = parser.add_argument_group("Captions")
    g.add_argument(
        "--input", "-i", type=str, required=True,
        help="Directory containing audio files to caption (scanned recursively)",
    )
    g.add_argument(
        "--provider", type=str, default=None,
        choices=["gemini", "openai", "local_8-10gb", "local_12gb", "local_16gb", "music_flamingo", "lyrics_only", "none"],
        help="Caption provider: gemini, openai, local_8-10gb, local_12gb, local_16gb, music_flamingo, lyrics_only, none (default: from settings, or gemini)",
    )
    g.add_argument(
        "--ai-model", "--model", type=str, default=None, dest="ai_model",
        help="Override the AI model name (e.g. gemini-2.5-flash, gpt-4o)",
    )
    g.add_argument(
        "--policy", type=str, default="fill_missing",
        choices=["fill_missing", "overwrite_caption", "overwrite_all"],
        help="Merge policy for existing sidecars (default: fill_missing)",
    )
    g.add_argument(
        "--lyrics-provider", type=str, default=None,
        choices=["genius", "transcriber_server", "music_flamingo", "none"],
        help="Lyrics provider: genius, transcriber_server, music_flamingo, none "
             "(default: genius when --lyrics is on)",
    )
    g.add_argument(
        "--lyrics", action=argparse.BooleanOptionalAction, default=True,
        help="Fetch lyrics (requires provider config). "
             "On by default; use --no-lyrics to skip",
    )
    g.add_argument(
        "--metadata-provider", type=str, default=None,
        choices=["music_flamingo", "none"],
        help="Metadata provider for structured fields: music_flamingo, none (default: none)",
    )
    g.add_argument(
        "--default-artist", type=str, default="",
        help="Default artist name for Genius lookups when filename has no artist",
    )
    g.add_argument(
        "--gemini-api-key", type=str, default=None,
        help="Gemini API key (overrides env/settings)",
    )
    g.add_argument(
        "--openai-api-key", type=str, default=None,
        help="OpenAI API key (overrides env/settings)",
    )
    g.add_argument(
        "--openai-base-url", type=str, default=None,
        help="Custom OpenAI-compatible base URL",
    )
    g.add_argument(
        "--genius-token", type=str, default=None,
        help="Genius API token (overrides env/settings)",
    )
    g.add_argument(
        "--music-flamingo-url", type=str, default=None,
        help="Music Flamingo server URL (overrides settings)",
    )
    g.add_argument(
        "--transcriber-server-url", type=str, default=None,
        help="Transcriber Server URL for lyrics (overrides settings)",
    )
    g.add_argument(
        "--hf-token", type=str, default=None,
        help="Hugging Face token for authenticated endpoints (overrides settings)",
    )
    g.add_argument(
        "--google-search", action="store_true", default=False,
        help="Enable Grounding with Google Search for Gemini captions "
             "(lets the model look up track info online; adds per-query cost)",
    )


def _add_tags_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ``tags`` subcommand."""
    _dir_parent = argparse.ArgumentParser(add_help=False)
    _dir_parent.add_argument(
        "directory", type=str, nargs="?", default=None,
        help="Directory of audio/sidecar files",
    )
    _dir_parent.add_argument("--input", "-i", type=str, default=None,
                             dest="input_compat", help=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="tags_action")

    p_add = sub.add_parser("add", parents=[_dir_parent],
                           help="Add a trigger tag to sidecar files")
    p_add.add_argument("--tag", "-t", type=str, required=True,
                       help="Trigger tag to add")
    p_add.add_argument("--position", type=str, default="prepend",
                       choices=["prepend", "append", "replace"],
                       help="Tag placement (default: prepend)")

    p_rm = sub.add_parser("remove", parents=[_dir_parent],
                          help="Remove a trigger tag from sidecar files")
    p_rm.add_argument("--tag", "-t", type=str, required=True,
                      help="Trigger tag to remove")

    sub.add_parser("clear", parents=[_dir_parent],
                   help="Clear all trigger tags from sidecar files")

    sub.add_parser("list", parents=[_dir_parent],
                   help="List trigger tags from sidecar files")


def _add_settings_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ``settings`` subcommand."""
    sub = parser.add_subparsers(dest="settings_action")

    sub.add_parser("show", help="Display current settings")

    p_set = sub.add_parser("set", help="Set a setting value")
    p_set.add_argument("key", type=str, help="Setting key (e.g. checkpoint_dir, gemini_api_key)")
    p_set.add_argument("value", type=str, help="New value")

    p_clear = sub.add_parser("clear", help="Clear a setting (reset to default)")
    p_clear.add_argument("key", type=str, help="Setting key to clear")

    sub.add_parser("path", help="Print the settings file path")

    p_defaults = sub.add_parser("defaults", help="View or modify training defaults")
    p_defaults.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), action="append",
                            dest="default_overrides",
                            help="Override a training default (e.g. --set lr 1e-4)")
