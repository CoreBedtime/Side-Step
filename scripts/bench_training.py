"""Benchmark real training-step throughput and peak VRAM.

Runs an actual short training session (model load + N optimizer steps)
through the exact production code path (config_factory -> loader ->
FixedLoRATrainer) and reports steps/s, per-step latency, and peak VRAM.
Use it to A/B any performance change with a number instead of vibes.

NOTE: this loads the model and trains on your GPU. Don't run it while
another training job is using the card.

Usage:
    uv run python scripts/bench_training.py \
        --checkpoint-dir ./checkpoints --model base \
        --dataset-dir ./preprocessed_tensors/my_set \
        --steps 30 --warmup 5 \
        --set optimizer_type=adamw8bit --set empty_cache_every=200

Any training option from the schema can be overridden with
``--set key=value`` (validated against core/schema.py).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_override(raw: str) -> tuple[str, object]:
    """Parse and validate a KEY=VALUE override against the schema."""
    from sidestep_engine.core.schema import SCHEMA_BY_NAME

    if "=" not in raw:
        raise SystemExit(f"--set expects KEY=VALUE (got {raw!r})")
    key, value = raw.split("=", 1)
    key = key.strip()
    f = SCHEMA_BY_NAME.get(key)
    if f is None:
        raise SystemExit(f"--set {key}: unknown training option (see core/schema.py)")
    if f.type == "int":
        return key, int(value)
    if f.type == "float":
        return key, float(value)
    if f.type == "bool":
        return key, value.strip().lower() in ("1", "true", "yes", "on")
    return key, value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--model", default="base", dest="model_variant")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--steps", type=int, default=30, help="Measured optimizer steps (default: 30)")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup steps excluded from timing (default: 5)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="Override any schema training option (repeatable)")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    args = ap.parse_args()

    from sidestep_engine.training_defaults import TRAINING_DEFAULTS

    overrides = dict(_parse_override(s) for s in args.set)

    tmp_out = tempfile.mkdtemp(prefix="sidestep_bench_")
    params: dict = {
        **TRAINING_DEFAULTS,
        "checkpoint_dir": args.checkpoint_dir,
        "model_variant": args.model_variant,
        "dataset_dir": args.dataset_dir,
        "output_dir": tmp_out,
        "device": args.device,
        "precision": args.precision,
        # Bench hygiene: no checkpoint/best-model I/O inside the timed window,
        # per-step yield cadence, bounded run length.
        "max_steps": args.warmup + args.steps,
        "epochs": 10_000,
        "log_every": 1,
        "log_heavy_every": 0,
        "save_best": False,
        "save_every": 10_000,
        "val_split": 0.0,
        "run_name": "bench",
        **overrides,
    }

    from sidestep_engine.cli.config_builder import build_configs_from_dict
    from sidestep_engine.models.loader import load_decoder_for_training

    adapter_cfg, train_cfg = build_configs_from_dict(params)

    print(f"[bench] loading model ({train_cfg.model_variant}, "
          f"{train_cfg.device}/{train_cfg.precision}) ...", file=sys.stderr)
    t_load0 = time.perf_counter()
    model = load_decoder_for_training(
        train_cfg.checkpoint_dir, train_cfg.model_variant,
        device=train_cfg.device, precision=train_cfg.precision,
        weight_quantize=train_cfg.weight_quantize,
        weight_qtype=train_cfg.weight_qtype,
        offload_encoder=train_cfg.offload_encoder,
    )
    load_s = time.perf_counter() - t_load0

    import torch
    from sidestep_engine.core.trainer import FixedLoRATrainer

    trainer = FixedLoRATrainer(model, adapter_cfg, train_cfg)

    cuda = torch.cuda.is_available()
    step_times: list[float] = []
    seen_steps = 0
    last_t: float | None = None

    print(f"[bench] running {args.warmup} warmup + {args.steps} measured steps ...",
          file=sys.stderr)
    for update in trainer.train():
        if update.kind == "fail":
            print(f"[bench] FAILED: {update.msg}", file=sys.stderr)
            return 1
        if update.kind != "step":
            continue
        now = time.perf_counter()
        seen_steps += 1
        if seen_steps == args.warmup:
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            last_t = time.perf_counter()
            continue
        if seen_steps > args.warmup and last_t is not None:
            step_times.append(now - last_t)
            last_t = now

    if not step_times:
        print("[bench] no measured steps (dataset too small for warmup+steps?)",
              file=sys.stderr)
        return 1

    result = {
        "model_load_s": round(load_s, 2),
        "measured_steps": len(step_times),
        "batch_size": train_cfg.batch_size,
        "grad_accum": train_cfg.gradient_accumulation_steps,
        "step_time_mean_s": round(statistics.mean(step_times), 4),
        "step_time_median_s": round(statistics.median(step_times), 4),
        "step_time_p90_s": round(sorted(step_times)[int(0.9 * len(step_times))], 4),
        "steps_per_s": round(1.0 / statistics.mean(step_times), 4),
        "samples_per_s": round(
            train_cfg.batch_size * train_cfg.gradient_accumulation_steps
            / statistics.mean(step_times), 3,
        ),
        "overrides": overrides,
    }
    if cuda:
        result["peak_vram_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        result["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)

    if args.json:
        print(json.dumps(result))
    else:
        print("\n=== bench_training results ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
