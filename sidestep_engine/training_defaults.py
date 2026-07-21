"""Canonical training defaults shared across entrypoints.

As of the schema refactor, the **single source of truth is
``sidestep_engine.core.schema``** — one ``SchemaField`` record per training
option (default, type, choices, range, help, CLI flags, GUI id).  This
module keeps the historical public API (``DEFAULT_*`` constants,
``TRAINING_DEFAULTS``, ``GUI_FIELD_MAP``, ``get_gui_defaults``) but derives
every value from the schema so nothing can drift.

To change a default, edit ``core/schema.py`` — CLI, Wizard, GUI, and
``/api/defaults`` all pick it up automatically.

GUI field-ID mapping
~~~~~~~~~~~~~~~~~~~~
The GUI uses HTML element IDs (``full-lr``, ``full-batch``, …) that
differ from backend parameter names.  ``GUI_FIELD_MAP`` (derived from the
schema's ``gui_id`` metadata) translates backend keys → GUI field IDs so
the server can emit defaults keyed the way the frontend expects.
``GUI_KEY_MAP`` is the reverse (GUI/JSON config key → backend parameter
name) used when *reading* config dicts produced by the frontend.
"""

from __future__ import annotations

import os

from sidestep_engine.core.schema import (
    SCHEMA_BY_NAME as _SCHEMA,
    defaults_dict as _schema_defaults_dict,
    gui_field_map as _schema_gui_field_map,
)


def _d(name: str):
    """Default value for schema field *name* (fails loudly on typos)."""
    return _SCHEMA[name].default


# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_LEARNING_RATE: float = _d("learning_rate")
DEFAULT_BATCH_SIZE: int = _d("batch_size")
DEFAULT_GRADIENT_ACCUMULATION: int = _d("gradient_accumulation")
DEFAULT_EPOCHS: int = _d("epochs")
DEFAULT_WARMUP_STEPS: int = _d("warmup_steps")
DEFAULT_WEIGHT_DECAY: float = _d("weight_decay")
DEFAULT_MAX_GRAD_NORM: float = _d("max_grad_norm")
DEFAULT_SEED: int = _d("seed")
DEFAULT_MAX_STEPS: int = _d("max_steps")
DEFAULT_DATASET_REPEATS: int = _d("dataset_repeats")

# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

DEFAULT_OPTIMIZER_TYPE: str = _d("optimizer_type")
"""Resolved at training-config build time to avoid importing torch at startup."""


def resolve_optimizer_type(
    optimizer_type: str | None,
    device_type: str | None = None,
) -> str:
    """Resolve the public ``auto`` optimizer default for the selected device.

    ``torch`` import/CUDA probing is intentionally kept out of this module so
    CLI help, the wizard menu, and GUI defaults stay fast.  Call this after
    device detection has already happened for an actual training run.
    """
    value = str(optimizer_type or DEFAULT_OPTIMIZER_TYPE).lower().strip()
    if value != "auto":
        return value
    return "adamw8bit" if str(device_type or "").lower() == "cuda" else "adamw"


DEFAULT_SCHEDULER_TYPE: str = _d("scheduler_type")
DEFAULT_SCHEDULER_FORMULA: str = _d("scheduler_formula")

# ---------------------------------------------------------------------------
# LoRA defaults
# ---------------------------------------------------------------------------

DEFAULT_RANK: int = _d("rank")
DEFAULT_ALPHA: int = _d("alpha")
DEFAULT_DROPOUT: float = _d("dropout")
DEFAULT_TARGET_MODULES: list = list(_d("target_modules"))
DEFAULT_ATTENTION_TYPE: str = _d("attention_type")
DEFAULT_TARGET_MLP: bool = _d("target_mlp")
DEFAULT_BIAS: str = _d("bias")

# ---------------------------------------------------------------------------
# LoKR defaults
# ---------------------------------------------------------------------------

DEFAULT_LOKR_LINEAR_DIM: int = _d("lokr_linear_dim")
DEFAULT_LOKR_LINEAR_ALPHA: int = _d("lokr_linear_alpha")
DEFAULT_LOKR_FACTOR: int = _d("lokr_factor")
DEFAULT_LOKR_DECOMPOSE_BOTH: bool = _d("lokr_decompose_both")
DEFAULT_LOKR_USE_TUCKER: bool = _d("lokr_use_tucker")
DEFAULT_LOKR_USE_SCALAR: bool = _d("lokr_use_scalar")
DEFAULT_LOKR_WEIGHT_DECOMPOSE: bool = _d("lokr_weight_decompose")

# ---------------------------------------------------------------------------
# LoHA defaults
# ---------------------------------------------------------------------------

DEFAULT_LOHA_LINEAR_DIM: int = _d("loha_linear_dim")
DEFAULT_LOHA_LINEAR_ALPHA: int = _d("loha_linear_alpha")
DEFAULT_LOHA_FACTOR: int = _d("loha_factor")
DEFAULT_LOHA_USE_TUCKER: bool = _d("loha_use_tucker")
DEFAULT_LOHA_USE_SCALAR: bool = _d("loha_use_scalar")

# ---------------------------------------------------------------------------
# OFT defaults
# ---------------------------------------------------------------------------

DEFAULT_OFT_BLOCK_SIZE: int = _d("oft_block_size")
DEFAULT_OFT_COFT: bool = _d("oft_coft")
DEFAULT_OFT_EPS: float = _d("oft_eps")

# ---------------------------------------------------------------------------
# VRAM / performance
# ---------------------------------------------------------------------------

DEFAULT_GRADIENT_CHECKPOINTING: bool = _d("gradient_checkpointing")
DEFAULT_GRADIENT_CHECKPOINTING_RATIO: float = _d("gradient_checkpointing_ratio")
DEFAULT_OFFLOAD_ENCODER: bool = _d("offload_encoder")

# ---------------------------------------------------------------------------
# Checkpointing / output
# ---------------------------------------------------------------------------

DEFAULT_SAVE_EVERY: int = _d("save_every")
DEFAULT_SAVE_BEST: bool = _d("save_best")
DEFAULT_SAVE_BEST_AFTER: int = _d("save_best_after")
DEFAULT_EARLY_STOP_PATIENCE: int = _d("early_stop_patience")
DEFAULT_STRICT_RESUME: bool = _d("strict_resume")
DEFAULT_TARGET_LOSS: float = _d("target_loss")
DEFAULT_TARGET_LOSS_FLOOR: float = _d("target_loss_floor")
DEFAULT_TARGET_LOSS_WARMUP: int = _d("target_loss_warmup")
DEFAULT_TARGET_LOSS_SMOOTHING: float = _d("target_loss_smoothing")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

DEFAULT_LOG_EVERY: int = _d("log_every")
DEFAULT_LOG_HEAVY_EVERY: int = _d("log_heavy_every")

# ---------------------------------------------------------------------------
# CFG / loss
# ---------------------------------------------------------------------------

DEFAULT_CFG_RATIO: float = _d("cfg_ratio")
DEFAULT_LOSS_WEIGHTING: str = _d("loss_weighting")
DEFAULT_SNR_GAMMA: float = _d("snr_gamma")
DEFAULT_LOSS_FN: str = _d("loss_fn")
DEFAULT_HUBER_DELTA: float = _d("huber_delta")
DEFAULT_CHANNEL_BALANCE: bool = _d("channel_balance")
DEFAULT_DYNAMIC_CHANNEL_BALANCE: bool = _d("dynamic_channel_balance")
DEFAULT_VAE_CHANNEL_PRIOR: bool = _d("vae_channel_prior")
DEFAULT_LATENT_NOISE: float = _d("latent_noise")
DEFAULT_T_BIAS: float = _d("t_bias")
DEFAULT_LEGACY_LOSS: bool = _d("legacy_loss")
DEFAULT_TIMESTEP_MODE: str = _d("timestep_mode")

# ---------------------------------------------------------------------------
# Chunking / cropping
# ---------------------------------------------------------------------------

DEFAULT_MAX_LATENT_LENGTH: int = _d("max_latent_length")
DEFAULT_CHUNK_DECAY_EVERY: int = _d("chunk_decay_every")

# ---------------------------------------------------------------------------
# DataLoader (platform-dependent)
# ---------------------------------------------------------------------------

DEFAULT_NUM_WORKERS: int = _d("num_workers")
DEFAULT_PREFETCH_FACTOR: int = _d("prefetch_factor")
DEFAULT_PIN_MEMORY: bool = _d("pin_memory")
DEFAULT_PERSISTENT_WORKERS: bool = _d("persistent_workers")

# ---------------------------------------------------------------------------
# "All the Levers" (experimental enhancements)
# ---------------------------------------------------------------------------

DEFAULT_EMA_DECAY: float = _d("ema_decay")
DEFAULT_EMA_START_STEP: int = _d("ema_start_step")
DEFAULT_VAL_SPLIT: float = _d("val_split")
DEFAULT_ADAPTIVE_TIMESTEP_RATIO: float = _d("adaptive_timestep_ratio")
DEFAULT_WARMUP_START_FACTOR: float = _d("warmup_start_factor")
DEFAULT_COSINE_ETA_MIN_RATIO: float = _d("cosine_eta_min_ratio")
DEFAULT_COSINE_RESTARTS_COUNT: int = _d("cosine_restarts_count")
DEFAULT_SAVE_BEST_EVERY_N_STEPS: int = _d("save_best_every_n_steps")

DEFAULT_LR_SCALE_SELF_ATTN: float = _d("lr_scale_self_attn")
DEFAULT_LR_SCALE_CROSS_ATTN: float = _d("lr_scale_cross_attn")
DEFAULT_LR_SCALE_MLP: float = _d("lr_scale_mlp")

# ---------------------------------------------------------------------------
# Model / device
# ---------------------------------------------------------------------------

DEFAULT_MODEL_VARIANT: str = _d("model_variant")
DEFAULT_ADAPTER_TYPE: str = _d("adapter_type")
DEFAULT_DEVICE: str = _d("device")
DEFAULT_PRECISION: str = _d("precision")

# Optimum-quanto (optional extra ``quantize``)
DEFAULT_WEIGHT_QUANTIZE: bool = _d("weight_quantize")
DEFAULT_WEIGHT_QTYPE: str = _d("weight_qtype")

# ---------------------------------------------------------------------------
# Aggregate dict — backend parameter names → default values.
# Derived from the schema (aggregate fields only).  Used by /api/defaults
# and review_summary _DEFAULTS.
# ---------------------------------------------------------------------------

TRAINING_DEFAULTS: dict = _schema_defaults_dict()

# ---------------------------------------------------------------------------
# GUI key mapping — frontend config key → backend parameter name.
# Keys not listed here are assumed to match the backend name exactly.
# (Hand-maintained: these are frontend aliases, not schema fields.)
# ---------------------------------------------------------------------------

GUI_KEY_MAP: dict = {
    "lr": "learning_rate",
    "learning-rate": "learning_rate",
    "batch-size": "batch_size",
    "gradient-accumulation": "gradient_accumulation",
    "save-every": "save_every",
    "grad_accum": "gradient_accumulation",
    "scheduler": "scheduler_type",
    "early_stop": "early_stop_patience",
    "target-loss": "target_loss",
    "target-loss-floor": "target_loss_floor",
    "target-loss-warmup": "target_loss_warmup",
    "target-loss-smoothing": "target_loss_smoothing",
    "projections": "target_modules",
    "self_projections": "self_target_modules",
    "cross_projections": "cross_target_modules",
}

# ---------------------------------------------------------------------------
# Backend parameter name → GUI field ID (derived from schema gui_id).
# Used by /api/defaults to emit defaults keyed the way the frontend expects.
# ---------------------------------------------------------------------------

GUI_FIELD_MAP: dict = _schema_gui_field_map()


# ---------------------------------------------------------------------------
# Float formatting hints — derived from schema ``gui_format`` so the GUI
# shows e.g. "3e-4" instead of "0.0003" and "1.0" instead of "1".
# ---------------------------------------------------------------------------

_SCI_NOTATION_FIELDS: set = {
    f.gui_id for f in _SCHEMA.values() if f.gui_id and f.gui_format == "sci"
}

_FORCE_DECIMAL_FIELDS: set = {
    f.gui_id for f in _SCHEMA.values() if f.gui_id and f.gui_format == "decimal1"
}


def _fmt_float(field_id: str, value: float) -> str:
    """Format a float to match the style defaults.json uses for *field_id*."""
    if field_id in _SCI_NOTATION_FIELDS and 0 < abs(value) < 0.01:
        # Compact scientific: "3e-4", "6e-5"
        s = f"{value:.0e}"           # e.g. "3e-04"
        # Strip leading zeros in exponent: "3e-04" → "3e-4"
        base, exp = s.split("e")
        return f"{base}e{int(exp)}"
    if field_id in _FORCE_DECIMAL_FIELDS:
        # Always show one decimal place: "1.0", "5.0"
        return f"{value:.1f}" if value == int(value) else f"{value:g}"
    return f"{value:g}"


def get_gui_defaults() -> dict:
    """Return ``TRAINING_DEFAULTS`` keyed by GUI field IDs.

    Values are converted to strings (matching HTML form value semantics)
    except booleans which stay as ``bool`` for checkbox binding.

    Also includes GUI-only keys (projections, settings paths, model-
    variant-dependent values) so the response is a complete superset
    of ``defaults.json``.
    """
    out: dict = {}
    for backend_key, value in TRAINING_DEFAULTS.items():
        field_id = GUI_FIELD_MAP.get(backend_key)
        if not field_id:
            continue
        if isinstance(value, bool):
            out[field_id] = value
        elif isinstance(value, float):
            out[field_id] = _fmt_float(field_id, value)
        else:
            out[field_id] = str(value)

    # -- Model-variant-dependent defaults (base is the default variant) -----
    out["full-shift"] = "1.0"
    out["full-inference-steps"] = "50"

    # -- UI presentation defaults (no backend equivalent) ------------------
    _projs = " ".join(DEFAULT_TARGET_MODULES)
    out["full-projections"] = _projs
    out["full-self-projections"] = _projs
    out["full-cross-projections"] = _projs
    out["full-chunk-duration"] = "0"
    out["full-max-latent-length"] = "0"
    out["full-crop-mode"] = "full"

    # -- Timestep defaults (from model config, not training params) --------
    out["full-timestep-mu"] = str(_d("timestep_mu"))      # "-0.4"
    out["full-timestep-sigma"] = str(_d("timestep_sigma"))  # "1.0"
    out["full-timestep-mode"] = DEFAULT_TIMESTEP_MODE

    # -- Empty-string defaults ---------------------------------------------
    out["full-resume-from"] = ""
    out["full-log-dir"] = ""

    # -- Settings path defaults (OS-native separators) ---------------------
    _sep = "\\" if os.name == "nt" else "/"
    out["settings-checkpoint-dir"] = f".{_sep}checkpoints"
    out["settings-adapters-dir"] = f".{_sep}trained_adapters"
    out["settings-tensors-dir"] = f".{_sep}preprocessed_tensors"
    out["settings-audio-dir"] = f".{_sep}my_audio"
    out["settings-exported-loras-dir"] = f".{_sep}exported_loras"

    return out
