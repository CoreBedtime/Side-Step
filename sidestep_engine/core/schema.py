"""Canonical training-option schema for Side-Step.

**This module is the single source of truth for every user-facing training
option**: its default value, type, allowed choices, numeric range, help text,
CLI flags, and GUI field ID.  Every other surface renders from this registry:

- ``training_defaults.py`` derives its ``DEFAULT_*`` constants,
  ``TRAINING_DEFAULTS`` aggregate, and ``GUI_FIELD_MAP`` from here.
- ``cli/args.py`` generates the ``train`` subcommand's argparse arguments
  from the ``cli`` metadata (see ``add_cli_argument`` there).
- ``gui/server.py`` serves the registry via ``GET /api/schema`` so the
  frontend can build forms without hand-maintaining copies.
- ``tests/test_schema_parity.py`` locks the remaining hand-written surfaces
  (dataclass defaults, static defaults.json) to this registry.

To add a new training option: add ONE ``SchemaField`` entry here, then use
the value in the trainer.  CLI flag, GUI default, and defaults aggregation
follow automatically.

IMPORTANT: this module must stay importable without torch (CLI --help and
GUI startup must not pay the torch import cost).  Keep it stdlib-only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterator, Optional, Tuple

__all__ = [
    "SchemaField",
    "SCHEMA",
    "SCHEMA_BY_NAME",
    "iter_fields",
    "fields_for_section",
    "defaults_dict",
    "gui_field_map",
    "render_help",
    "to_json_schema",
]


# ---------------------------------------------------------------------------
# Field record
# ---------------------------------------------------------------------------

_UNSET = object()


@dataclass(frozen=True)
class SchemaField:
    """One user-facing training option.

    Attributes:
        name: Backend parameter name (also the argparse ``dest``).
        type: One of ``"int"``, ``"float"``, ``"bool"``, ``"str"``,
            ``"str_list"``.
        default: The *effective* public default applied by config_factory /
            the GUI.  (The CLI may present ``None`` instead — see
            ``cli_default_none``.)
        section: CLI argument-group title this field belongs to.
        help: CLI help text.  The literal token ``{default}`` is replaced
            with the stringified default at render time (plain ``str()``
            conversion).  Literal ``%`` must be escaped as ``%%`` for
            argparse.
        choices: Allowed values (enum-style options), or ``None``.
        min / max: Inclusive numeric bounds (informational; enforced by
            ``TrainingConfigV2.__post_init__`` where applicable).
        cli: Option strings in order (first long flag is the canonical one),
            or ``None`` when the field has no CLI flag.
        cli_default_none: When True the argparse default is ``None`` so that
            "flag not provided" is distinguishable and resolved later
            (settings lookup, model config, or the schema default).
        cli_action: ``None`` (regular typed option), ``"bool_optional"``
            (``argparse.BooleanOptionalAction``), or ``"store_true"``.
        cli_metavar: Explicit metavar override.
        cli_nargs: nargs override (e.g. ``"+"`` for str_list fields).
        cli_suppressed: Hide from ``--help`` (``argparse.SUPPRESS``).
        aggregate: Include in the ``TRAINING_DEFAULTS`` dict (and therefore
            ``/api/defaults``).  Path-like / unset-by-default fields opt out.
        gui_id: HTML element ID in the GUI, or ``None`` when the GUI does
            not expose the field (or handles it specially).
        gui_format: Float formatting hint for the GUI: ``None`` (``%g``),
            ``"sci"`` (compact scientific, e.g. ``3e-4``), or ``"decimal1"``
            (always one decimal, e.g. ``1.0``).
    """

    name: str
    type: str
    default: Any
    section: str
    help: str
    choices: Optional[Tuple[Any, ...]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    cli: Optional[Tuple[str, ...]] = None
    cli_default_none: bool = False
    cli_action: Optional[str] = None
    cli_metavar: Optional[str] = None
    cli_nargs: Optional[str] = None
    cli_suppressed: bool = False
    aggregate: bool = True
    gui_id: Optional[str] = None
    gui_format: Optional[str] = None


def _f(**kw: Any) -> SchemaField:
    return SchemaField(**kw)


# ---------------------------------------------------------------------------
# Section titles (must match the argparse group titles in cli/args.py)
# ---------------------------------------------------------------------------

S_MODEL = "Model / paths"
S_DEVICE = "Device / platform"
S_DATA = "Data"
S_TRAINING = "Training"
S_LEVERS = "All the Levers (experimental)"
S_ADAPTER = "Adapter"
S_LORA = "LoRA (used when --adapter=lora)"
S_LOKR = "LoKR (used when --adapter=lokr)"
S_LOHA = "LoHA (used when --adapter=loha)"
S_OFT = "OFT [Experimental] (used when --adapter=oft)"
S_CKPT = "Checkpointing"
S_LOG = "Logging / TensorBoard"
S_ADVANCED = "Training (advanced)"

# Platform-dependent DataLoader defaults (Windows 'spawn' overhead).
_NUM_WORKERS_DEFAULT: int = 2 if sys.platform == "win32" else 4


# ---------------------------------------------------------------------------
# The registry.  Order matters: CLI arguments are emitted in declaration
# order within each section, exactly matching the historical hand-written
# parser layout.
# ---------------------------------------------------------------------------

SCHEMA: Tuple[SchemaField, ...] = (
    # -- Model / paths ------------------------------------------------------
    _f(name="checkpoint_dir", type="str", default="./checkpoints", section=S_MODEL,
       cli=("--checkpoint-dir", "-c"), cli_default_none=True, aggregate=False,
       help="Path to checkpoints root directory (auto-resolves from settings if omitted)"),
    _f(name="model_variant", type="str", default="base", section=S_MODEL,
       cli=("--model", "-M", "--model-variant"), cli_metavar="MODEL",
       gui_id="full-model-variant",
       help=("Model variant or subfolder name (default: {default}). "
             "Official ACE-Step 1.5: base, sft, turbo. "
             "Official XL: xl-base, xl-sft, xl-turbo (see VARIANT_DIR_MAP in core/constants). "
             "Custom fine-tunes: exact subdirectory name under checkpoint-dir.")),

    # -- Device / platform --------------------------------------------------
    _f(name="device", type="str", default="auto", section=S_DEVICE,
       cli=("--device",), gui_id="full-device",
       help="Device: auto, cuda, cuda:0, mps, xpu, cpu (default: {default})"),
    _f(name="precision", type="str", default="auto", section=S_DEVICE,
       choices=("auto", "bf16", "fp16", "fp32"),
       cli=("--precision",), gui_id="full-precision",
       help="Precision: auto, bf16, fp16, fp32 (default: {default})"),

    # -- Data ---------------------------------------------------------------
    _f(name="dataset_dir", type="str", default="", section=S_DATA,
       cli=("--dataset-dir", "-d"), cli_default_none=True, aggregate=False,
       help="Directory containing preprocessed .pt files"),
    _f(name="num_workers", type="int", default=_NUM_WORKERS_DEFAULT, section=S_DATA,
       cli=("--num-workers",), gui_id="full-num-workers", min=0,
       help=f"DataLoader workers (default: {_NUM_WORKERS_DEFAULT}; 0 on Windows)"),
    _f(name="pin_memory", type="bool", default=True, section=S_DATA,
       cli=("--pin-memory",), cli_action="bool_optional", gui_id="full-pin-memory",
       help="Pin memory for GPU transfer (default: True)"),
    _f(name="prefetch_factor", type="int", default=2, section=S_DATA,
       cli=("--prefetch-factor",), gui_id="full-prefetch-factor", min=0,
       help="DataLoader prefetch factor (default: {default}; 0 on Windows)"),
    _f(name="persistent_workers", type="bool", default=True, section=S_DATA,
       cli=("--persistent-workers",), cli_action="bool_optional",
       gui_id="full-persistent-workers",
       help="Keep workers alive between epochs (default: True; False on Windows)"),

    # -- Training hyperparams ----------------------------------------------
    _f(name="learning_rate", type="float", default=3e-4, section=S_TRAINING,
       cli=("--lr", "--learning-rate", "-l"), gui_id="full-lr", gui_format="sci",
       min=0.0, max=1.0,
       help="Initial learning rate (default: {default})"),
    _f(name="batch_size", type="int", default=1, section=S_TRAINING,
       cli=("--batch-size", "-b"), gui_id="full-batch", min=1,
       help="Training batch size (default: {default})"),
    _f(name="gradient_accumulation", type="int", default=4, section=S_TRAINING,
       cli=("--gradient-accumulation", "-g"), gui_id="full-grad-accum", min=1,
       help="Gradient accumulation steps (default: {default})"),
    _f(name="epochs", type="int", default=1000, section=S_TRAINING,
       cli=("--epochs", "-e"), gui_id="full-epochs", min=1,
       help="Maximum training epochs (default: {default})"),
    _f(name="warmup_steps", type="int", default=100, section=S_TRAINING,
       cli=("--warmup-steps",), gui_id="full-warmup", min=0,
       help="LR warmup steps (default: {default})"),
    _f(name="weight_decay", type="float", default=0.01, section=S_TRAINING,
       cli=("--weight-decay",), gui_id="full-weight-decay", min=0.0,
       help="AdamW weight decay (default: {default})"),
    _f(name="max_grad_norm", type="float", default=1.0, section=S_TRAINING,
       cli=("--max-grad-norm",), gui_id="full-max-grad-norm", gui_format="decimal1",
       min=0.0,
       help="Gradient clipping norm (default: {default})"),
    _f(name="seed", type="int", default=42, section=S_TRAINING,
       cli=("--seed", "-s"), gui_id="full-seed",
       help="Random seed (default: {default})"),
    _f(name="chunk_duration", type="int", default=0, section=S_TRAINING,
       cli=("--chunk-duration",), cli_default_none=True, aggregate=False,
       help=("Random latent chunk duration in seconds (default: disabled). "
             "Recommended: 60. Extracts a random window each iteration for data "
             "augmentation and VRAM savings. WARNING: values below 60s (e.g. 30) "
             "may reduce training quality for full-length inference")),
    _f(name="chunk_decay_every", type="int", default=10, section=S_TRAINING,
       cli=("--chunk-decay-every",), gui_id="full-chunk-decay-every", min=0,
       help="Epoch interval for halving chunk coverage histogram; 0 disables decay (default: {default})"),
    _f(name="max_latent_length", type="int", default=0, section=S_TRAINING,
       cli=("--max-latent-length",), cli_default_none=True,
       gui_id="full-max-latent-length", min=0,
       help="Random crop length in latent frames (0 = disabled). Takes precedence over --chunk-duration when > 0"),
    _f(name="max_steps", type="int", default=0, section=S_TRAINING,
       cli=("--max-steps", "-m"), gui_id="full-max-steps", min=0,
       help="Maximum optimizer steps; 0 = use epochs only (default: {default})"),
    _f(name="shift", type="float", default=None, section=S_TRAINING,
       cli=("--shift",), cli_default_none=True, cli_suppressed=True,
       aggregate=False, help=""),
    _f(name="num_inference_steps", type="int", default=None, section=S_TRAINING,
       cli=("--num-inference-steps",), cli_default_none=True, cli_suppressed=True,
       aggregate=False, help=""),
    _f(name="optimizer_type", type="str", default="auto", section=S_TRAINING,
       choices=("auto", "adamw", "adamw8bit", "adafactor", "prodigy"),
       cli=("--optimizer-type",), gui_id="full-optimizer",
       help="Optimizer (default: {default}; resolves to adamw8bit on CUDA, adamw otherwise)"),
    _f(name="scheduler_type", type="str", default="cosine", section=S_TRAINING,
       choices=("cosine", "cosine_restarts", "linear", "constant", "constant_with_warmup", "custom"),
       cli=("--scheduler-type",), gui_id="full-scheduler",
       help="LR scheduler (default: {default})"),
    _f(name="scheduler_formula", type="str", default="", section=S_TRAINING,
       cli=("--scheduler-formula",), gui_id="full-scheduler-formula",
       help="Custom LR formula (Python math expression). Only used with --scheduler-type custom"),
    _f(name="gradient_checkpointing", type="bool", default=True, section=S_TRAINING,
       cli=("--gradient-checkpointing",), cli_action="bool_optional",
       help=("Recompute activations to save VRAM (~40-60%% less, ~10-30%% slower). "
             "On by default; use --no-gradient-checkpointing to disable")),
    _f(name="gradient_checkpointing_ratio", type="float", default=1.0, section=S_TRAINING,
       cli=("--gradient-checkpointing-ratio",), gui_id="full-grad-ckpt-ratio",
       gui_format="decimal1", min=0.0, max=1.0,
       help=("Fraction of decoder layers to checkpoint (0.0=none, 0.5=half, 1.0=all). "
             "Only applies when --gradient-checkpointing is on (default: {default})")),
    _f(name="offload_encoder", type="bool", default=True, section=S_TRAINING,
       cli=("--offload-encoder",), cli_action="bool_optional",
       gui_id="full-offload-encoder",
       help=("Move encoder/VAE to CPU after setup (saves ~2-4GB VRAM). "
             "On by default; use --no-offload-encoder to disable")),
    _f(name="weight_quantize", type="bool", default=False, section=S_TRAINING,
       cli=("--weight-quantize",), cli_action="bool_optional",
       gui_id="full-weight-quantize",
       help=("Quantize frozen backbone weights with optimum-quanto after load "
             "(install: side-step[quantize]). Off by default")),
    _f(name="weight_qtype", type="str", default="qfloat8", section=S_TRAINING,
       cli=("--weight-qtype",), gui_id="full-weight-qtype",
       help=("optimum-quanto qtype (e.g. qfloat8, qint8); torchao keys (int8, float8) "
             "are not supported with LoRA — default: {default}")),

    # -- All the Levers (experimental) --------------------------------------
    _f(name="ema_decay", type="float", default=0.0, section=S_LEVERS,
       cli=("--ema-decay",), gui_id="full-ema-decay", min=0.0, max=1.0,
       help="EMA decay for adapter weights (0=off, 0.9999=typical, default: {default})"),
    _f(name="ema_start_step", type="int", default=2000, section=S_LEVERS,
       cli=("--ema-start-step",), gui_id="full-ema-start-step", min=0,
       help="Step at which EMA tracking begins (0=immediate, default: {default})"),
    _f(name="val_split", type="float", default=0.0, section=S_LEVERS,
       cli=("--val-split",), gui_id="full-val-split", min=0.0, max=0.5,
       help="Validation holdout fraction (0=off, 0.1=10%%, default: {default})"),
    _f(name="adaptive_timestep_ratio", type="float", default=0.0, section=S_LEVERS,
       cli=("--adaptive-timestep-ratio",), gui_id="full-adaptive-timestep",
       min=0.0, max=1.0,
       help="Adaptive timestep sampling ratio (0=off, 0.3=recommended, default: {default}). Base/SFT only"),
    _f(name="warmup_start_factor", type="float", default=0.1, section=S_LEVERS,
       cli=("--warmup-start-factor",), gui_id="full-warmup-start-factor",
       min=0.0, max=1.0,
       help="LR warmup starts at base_lr * this (default: {default})"),
    _f(name="cosine_eta_min_ratio", type="float", default=0.01, section=S_LEVERS,
       cli=("--cosine-eta-min-ratio",), gui_id="full-cosine-eta-min",
       min=0.0, max=1.0,
       help="Cosine scheduler decays LR to base_lr * this (default: {default})"),
    _f(name="cosine_restarts_count", type="int", default=4, section=S_LEVERS,
       cli=("--cosine-restarts-count",), gui_id="full-cosine-restarts", min=1,
       help="Number of cosine restart cycles (default: {default})"),
    _f(name="save_best_every_n_steps", type="int", default=0, section=S_LEVERS,
       cli=("--save-best-every-n-steps",), gui_id="full-save-best-every-n-steps",
       min=0,
       help="Step-level best-model check interval (0=epoch only, default: {default})"),
    _f(name="lr_scale_self_attn", type="float", default=1.0, section=S_LEVERS,
       cli=("--lr-scale-self-attn",), gui_id="full-lr-scale-self-attn",
       min=0.0, max=10.0,
       help="LR multiplier for self-attention params (default: {default})"),
    _f(name="lr_scale_cross_attn", type="float", default=1.0, section=S_LEVERS,
       cli=("--lr-scale-cross-attn",), gui_id="full-lr-scale-cross-attn",
       min=0.0, max=10.0,
       help="LR multiplier for cross-attention params (default: {default})"),
    _f(name="lr_scale_mlp", type="float", default=1.0, section=S_LEVERS,
       cli=("--lr-scale-mlp",), gui_id="full-lr-scale-mlp", min=0.0, max=10.0,
       help="LR multiplier for MLP/FFN params (default: {default})"),
    _f(name="timestep_mu", type="float", default=-0.4, section=S_LEVERS,
       cli=("--timestep-mu",), cli_default_none=True, aggregate=False,
       gui_id="full-timestep-mu",
       help="Override logit-normal timestep mean (default: from model config, typically -0.4)"),
    _f(name="timestep_sigma", type="float", default=1.0, section=S_LEVERS,
       cli=("--timestep-sigma",), cli_default_none=True, aggregate=False,
       gui_id="full-timestep-sigma",
       help="Override logit-normal timestep sigma (default: from model config, typically 1.0)"),

    # -- Adapter selection --------------------------------------------------
    _f(name="adapter_type", type="str", default="lora", section=S_ADAPTER,
       choices=("lora", "dora", "lokr", "loha", "oft"),
       cli=("--adapter", "-a", "--adapter-type"), gui_id="full-adapter-type",
       help="Adapter type: lora, dora, lokr, loha, or oft (default: {default})"),

    # -- LoRA ---------------------------------------------------------------
    _f(name="rank", type="int", default=64, section=S_LORA,
       cli=("--rank", "-r"), gui_id="full-rank", min=1, max=1024,
       help="LoRA rank (default: {default})"),
    _f(name="alpha", type="int", default=128, section=S_LORA,
       cli=("--alpha",), gui_id="full-alpha", min=1,
       help="LoRA alpha (default: {default})"),
    _f(name="dropout", type="float", default=0.1, section=S_LORA,
       cli=("--dropout",), gui_id="full-dropout", min=0.0, max=1.0,
       help="LoRA dropout (default: {default})"),
    _f(name="target_modules", type="str_list",
       default=("q_proj", "k_proj", "v_proj", "o_proj"), section=S_LORA,
       cli=("--target-modules",), cli_nargs="+", aggregate=False,
       help="Modules to apply adapter to"),
    _f(name="bias", type="str", default="none", section=S_LORA,
       choices=("none", "all", "lora_only"),
       cli=("--bias",), gui_id="full-bias",
       help="Bias training mode (default: {default})"),
    _f(name="attention_type", type="str", default="both", section=S_LORA,
       choices=("self", "cross", "both"),
       cli=("--attention-type",), gui_id="full-attention-type",
       help="Attention layers to target (default: {default})"),
    _f(name="self_target_modules", type="str_list", default=None, section=S_LORA,
       cli=("--self-target-modules",), cli_nargs="+", cli_default_none=True,
       aggregate=False,
       help="Projections for self-attention only (used when --attention-type=both)"),
    _f(name="cross_target_modules", type="str_list", default=None, section=S_LORA,
       cli=("--cross-target-modules",), cli_nargs="+", cli_default_none=True,
       aggregate=False,
       help="Projections for cross-attention only (used when --attention-type=both)"),
    _f(name="target_mlp", type="bool", default=True, section=S_LORA,
       cli=("--target-mlp",), cli_action="bool_optional", gui_id="full-target-mlp",
       help=("Target MLP/FFN layers (gate_proj, up_proj, down_proj). "
             "On by default; use --no-target-mlp to disable")),

    # -- LoKR ---------------------------------------------------------------
    _f(name="lokr_linear_dim", type="int", default=64, section=S_LOKR,
       cli=("--lokr-linear-dim",), gui_id="full-lokr-dim", min=1,
       help="LoKR linear dimension (default: {default})"),
    _f(name="lokr_linear_alpha", type="int", default=128, section=S_LOKR,
       cli=("--lokr-linear-alpha",), gui_id="full-lokr-alpha", min=1,
       help="LoKR linear alpha (default: {default})"),
    _f(name="lokr_factor", type="int", default=-1, section=S_LOKR,
       cli=("--lokr-factor",), gui_id="full-lokr-factor",
       help="LoKR factor; -1 for auto (default: {default})"),
    _f(name="lokr_decompose_both", type="bool", default=False, section=S_LOKR,
       cli=("--lokr-decompose-both",), cli_action="store_true",
       help="Decompose both Kronecker factors"),
    _f(name="lokr_use_tucker", type="bool", default=False, section=S_LOKR,
       cli=("--lokr-use-tucker",), cli_action="store_true",
       help="Use Tucker decomposition"),
    _f(name="lokr_use_scalar", type="bool", default=False, section=S_LOKR,
       cli=("--lokr-use-scalar",), cli_action="store_true",
       help="Use scalar scaling"),
    _f(name="lokr_weight_decompose", type="bool", default=False, section=S_LOKR,
       cli=("--lokr-weight-decompose",), cli_action="store_true",
       help="Enable DoRA-style weight decomposition"),

    # -- LoHA ---------------------------------------------------------------
    _f(name="loha_linear_dim", type="int", default=64, section=S_LOHA,
       cli=("--loha-linear-dim",), gui_id="full-loha-dim", min=1,
       help="LoHA linear dimension (default: {default})"),
    _f(name="loha_linear_alpha", type="int", default=128, section=S_LOHA,
       cli=("--loha-linear-alpha",), gui_id="full-loha-alpha", min=1,
       help="LoHA linear alpha (default: {default})"),
    _f(name="loha_factor", type="int", default=-1, section=S_LOHA,
       cli=("--loha-factor",), gui_id="full-loha-factor",
       help="LoHA factor; -1 for auto (default: {default})"),
    _f(name="loha_use_tucker", type="bool", default=False, section=S_LOHA,
       cli=("--loha-use-tucker",), cli_action="store_true",
       help="Use Tucker decomposition"),
    _f(name="loha_use_scalar", type="bool", default=False, section=S_LOHA,
       cli=("--loha-use-scalar",), cli_action="store_true",
       help="Use scalar scaling"),

    # -- OFT ----------------------------------------------------------------
    _f(name="oft_block_size", type="int", default=64, section=S_OFT,
       cli=("--oft-block-size",), gui_id="full-oft-block-size", min=1,
       help="OFT block size (default: {default})"),
    _f(name="oft_coft", type="bool", default=False, section=S_OFT,
       cli=("--oft-coft",), cli_action="store_true",
       help="Enable constrained OFT (Cayley projection)"),
    _f(name="oft_eps", type="float", default=6e-5, section=S_OFT,
       cli=("--oft-eps",), gui_id="full-oft-eps", gui_format="sci", min=0.0,
       help="OFT epsilon for numerical stability (default: {default})"),

    # -- Checkpointing ------------------------------------------------------
    _f(name="output_dir", type="str", default=None, section=S_CKPT,
       cli=("--output-dir", "-o"), cli_default_none=True, aggregate=False,
       help="Output directory for adapter weights"),
    _f(name="save_every", type="int", default=50, section=S_CKPT,
       cli=("--save-every",), gui_id="full-save-every", min=1,
       help="Save checkpoint every N epochs (default: {default})"),
    _f(name="resume_from", type="str", default=None, section=S_CKPT,
       cli=("--resume-from",), cli_default_none=True, aggregate=False,
       gui_id="full-resume-from",
       help="Path to checkpoint dir to resume from"),
    _f(name="strict_resume", type="bool", default=True, section=S_CKPT,
       cli=("--strict-resume",), cli_action="bool_optional",
       gui_id="full-strict-resume",
       help="Abort on config mismatch or failed state restore during resume (default: True)"),
    _f(name="run_name", type="str", default=None, section=S_CKPT,
       cli=("--run-name", "-n"), cli_default_none=True, aggregate=False,
       help="Name for this training run (used for output dir, TB logs). Auto-generated if omitted"),
    _f(name="save_best", type="bool", default=True, section=S_CKPT,
       cli=("--save-best",), cli_action="bool_optional", gui_id="full-save-best",
       help="Auto-save best model by smoothed loss (default: True)"),
    _f(name="save_best_after", type="int", default=200, section=S_CKPT,
       cli=("--save-best-after",), gui_id="full-save-best-after", min=0,
       help="Epoch to start best-model tracking (default: {default})"),
    _f(name="early_stop_patience", type="int", default=0, section=S_CKPT,
       cli=("--early-stop-patience",), gui_id="full-early-stop", min=0,
       help="Stop if no improvement for N epochs; 0=disabled (default: {default})"),
    _f(name="target_loss", type="float", default=0.0, section=S_CKPT,
       cli=("--target-loss",), gui_id="full-target-loss", min=0.0,
       help=("Target loss cruise control (0=disabled). Damps LR as the smoothed RAW "
             "(unweighted) loss approaches this value, so the target means the same "
             "thing under any loss weighting/balancing config. (default: {default})")),
    _f(name="target_loss_floor", type="float", default=0.01, section=S_CKPT,
       cli=("--target-loss-floor",), gui_id="full-target-loss-floor",
       min=0.0, max=1.0,
       help="Min LR multiplier at target loss; 0.01=1%% of scheduled LR (default: {default})"),
    _f(name="target_loss_warmup", type="int", default=50, section=S_CKPT,
       cli=("--target-loss-warmup",), gui_id="full-target-loss-warmup", min=0,
       help="Min steps before cruise control engages (default: {default})"),
    _f(name="target_loss_smoothing", type="float", default=0.98, section=S_CKPT,
       cli=("--target-loss-smoothing",), gui_id="full-target-loss-smoothing",
       min=0.0, max=1.0,
       help="EMA beta for loss smoothing in cruise control; higher=smoother (default: {default})"),

    # -- Logging / TensorBoard ---------------------------------------------
    _f(name="log_dir", type="str", default=None, section=S_LOG,
       cli=("--log-dir",), cli_default_none=True, aggregate=False,
       help="TensorBoard log directory (default: {output-dir}/runs)"),
    _f(name="log_every", type="int", default=10, section=S_LOG,
       cli=("--log-every",), gui_id="full-log-every", min=1,
       help="Log basic metrics every N steps (default: {default})"),
    _f(name="log_heavy_every", type="int", default=50, section=S_LOG,
       cli=("--log-heavy-every",), gui_id="full-log-heavy-every", min=0,
       help="Log per-layer gradient norms every N steps; 0 disables heavy logging (default: {default})"),

    # -- Training (advanced) — the `train` subcommand only ------------------
    _f(name="timestep_mode", type="str", default="continuous", section=S_ADVANCED,
       choices=("continuous", "discrete"),
       cli=("--timestep-mode",), gui_id="full-timestep-mode",
       help=("Timestep sampling: 'continuous' (logit-normal, recommended) or "
             "'discrete' (8-step turbo schedule). (default: {default})")),
    _f(name="cfg_ratio", type="float", default=0.15, section=S_ADVANCED,
       cli=("--cfg-ratio",), gui_id="full-cfg-dropout", min=0.0, max=1.0,
       help="CFG dropout probability (default: {default})"),
    _f(name="loss_weighting", type="str", default="none", section=S_ADVANCED,
       choices=("none", "min_snr", "flow_snr"),
       cli=("--loss-weighting",), gui_id="full-loss-weighting",
       help=("Loss weighting: 'none' (flat, reference objective), 'flow_snr' "
             "(U-shaped emphasis for rectified flow), or 'min_snr' "
             "(Min-SNR-gamma, v-prediction form). (default: {default})")),
    _f(name="snr_gamma", type="float", default=5.0, section=S_ADVANCED,
       cli=("--snr-gamma",), gui_id="full-snr-gamma", gui_format="decimal1",
       min=0.0,
       help="Gamma clamp for flow_snr/min_snr weighting (default: {default})"),
    _f(name="loss_fn", type="str", default="mse", section=S_ADVANCED,
       choices=("mse", "huber", "pseudo_huber", "x0_mse", "x0_pseudo_huber"),
       cli=("--loss-fn",), gui_id="full-loss-fn",
       help=("Loss function. x0_ prefix computes loss on reconstructed x0 "
             "(t² weighting). (default: {default})")),
    _f(name="huber_delta", type="float", default=1.0, section=S_ADVANCED,
       cli=("--huber-delta",), gui_id="full-huber-delta", min=0.0,
       help="Huber loss delta threshold (default: {default})"),
    _f(name="channel_balance", type="bool", default=True, section=S_ADVANCED,
       cli=("--channel-balance",), cli_action="bool_optional",
       gui_id="full-channel-balance",
       help="Per-channel fidelity balancing (default: {default})"),
    _f(name="dynamic_channel_balance", type="bool", default=False, section=S_ADVANCED,
       cli=("--dynamic-channel-balance",), cli_action="bool_optional",
       gui_id="full-dynamic-channel-balance",
       help="Dynamically rebalance channel weights from running loss (default: {default})"),
    _f(name="vae_channel_prior", type="bool", default=True, section=S_ADVANCED,
       cli=("--vae-channel-prior",), cli_action="bool_optional",
       gui_id="full-vae-channel-prior",
       help="Use VAE decoder channel importance in channel weights (default: {default})"),
    _f(name="latent_noise", type="float", default=0.0, section=S_ADVANCED,
       cli=("--latent-noise",), gui_id="full-latent-noise", min=0.0,
       help="Per-channel latent noise regularization scale, 0=off (default: {default})"),
    _f(name="t_bias", type="float", default=0.5, section=S_ADVANCED,
       cli=("--t-bias",), gui_id="full-t-bias", min=0.0, max=2.0,
       help="Asymmetric timestep emphasis toward low-t detail, 0=symmetric (default: {default})"),
    _f(name="legacy_loss", type="bool", default=False, section=S_ADVANCED,
       cli=("--legacy-loss",), cli_action="store_true", gui_id="full-legacy-loss",
       help="Revert all loss math to pre-flow-SNR behavior (flat MSE, no channel balancing)"),
    _f(name="ignore_fisher_map", type="bool", default=False, section=S_ADVANCED,
       cli=("--ignore-fisher-map",), cli_action="store_true", aggregate=False,
       help="Bypass auto-detection of fisher_map.json in --dataset-dir"),
    _f(name="dataset_repeats", type="int", default=1, section=S_ADVANCED,
       cli=("--dataset-repeats", "-R"), gui_id="full-dataset-repeats", min=1,
       help="Global dataset repetition multiplier (1 = no repetition, default: {default})"),
    _f(name="crop_mode", type="str", default=None, section=S_ADVANCED,
       cli=("--crop-mode",), cli_default_none=True, cli_metavar="MODE",
       aggregate=False, gui_id="full-crop-mode",
       help=("Crop/chunk mode hint for the trainer: full, seconds, or latent "
             "(wizard-aligned); pairs with chunk_duration / max_latent_length. "
             "Default: unset")),
)


SCHEMA_BY_NAME: dict = {f.name: f for f in SCHEMA}

# Fail fast on duplicate names (a duplicate would silently shadow a field).
assert len(SCHEMA_BY_NAME) == len(SCHEMA), "duplicate field name in SCHEMA"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_fields() -> Iterator[SchemaField]:
    """Iterate all fields in declaration order."""
    return iter(SCHEMA)


def fields_for_section(section: str) -> Tuple[SchemaField, ...]:
    """All fields belonging to *section*, in declaration order."""
    return tuple(f for f in SCHEMA if f.section == section)


def defaults_dict() -> dict:
    """Backend parameter name -> default, for aggregate fields only.

    This is the source of ``training_defaults.TRAINING_DEFAULTS``.
    """
    return {f.name: (list(f.default) if isinstance(f.default, tuple) else f.default)
            for f in SCHEMA if f.aggregate}


def gui_field_map() -> dict:
    """Backend parameter name -> GUI element ID, for GUI-exposed fields."""
    return {f.name: f.gui_id for f in SCHEMA if f.gui_id}


def render_help(f: SchemaField) -> str:
    """Render the CLI help string with the default substituted.

    Plain ``str()`` conversion matches the historical hand-written
    f-string formatting.  Uses ``replace`` (not ``str.format``) so literal
    braces in help text (e.g. ``{output-dir}/runs``) survive.
    """
    return f.help.replace("{default}", str(f.default))


_JSON_TYPES = {
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "str": "string",
    "str_list": "array",
}


def to_json_schema() -> dict:
    """Serialize the registry for the ``/api/schema`` endpoint.

    Returns a dict with a ``fields`` list preserving declaration order.
    Help text is de-escaped (``%%`` -> ``%``) for non-argparse consumers.
    """
    fields = []
    for f in SCHEMA:
        entry: dict = {
            "name": f.name,
            "type": _JSON_TYPES[f.type],
            "default": list(f.default) if isinstance(f.default, tuple) else f.default,
            "section": f.section,
            "help": render_help(f).replace("%%", "%"),
        }
        if f.choices is not None:
            entry["choices"] = list(f.choices)
        if f.min is not None:
            entry["min"] = f.min
        if f.max is not None:
            entry["max"] = f.max
        if f.cli and not f.cli_suppressed:
            entry["cli_flag"] = f.cli[0]
        if f.gui_id:
            entry["gui_id"] = f.gui_id
        if f.gui_format:
            entry["gui_format"] = f.gui_format
        fields.append(entry)
    return {"version": 1, "fields": fields}
