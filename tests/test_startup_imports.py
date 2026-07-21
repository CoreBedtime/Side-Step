from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from sidestep_engine.training_defaults import resolve_optimizer_type


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_fresh_python(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_training_defaults_import_does_not_import_torch() -> None:
    result = _run_fresh_python(
        """
        import sys
        import sidestep_engine.training_defaults as defaults

        assert defaults.DEFAULT_OPTIMIZER_TYPE == "auto"
        raise SystemExit(1 if "torch" in sys.modules else 0)
        """
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_cli_parser_build_does_not_import_torch() -> None:
    result = _run_fresh_python(
        """
        import sys
        from sidestep_engine.cli.common import build_root_parser

        build_root_parser()
        raise SystemExit(1 if "torch" in sys.modules else 0)
        """
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_compatibility_check_does_not_import_torch() -> None:
    result = _run_fresh_python(
        """
        import sys
        from sidestep_engine._compat import check_compatibility

        check_compatibility()
        raise SystemExit(1 if "torch" in sys.modules else 0)
        """
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_auto_optimizer_resolution() -> None:
    assert resolve_optimizer_type("auto", "cuda") == "adamw8bit"
    assert resolve_optimizer_type("auto", "mps") == "adamw"
    assert resolve_optimizer_type("auto", "xpu") == "adamw"
    assert resolve_optimizer_type("auto", "cpu") == "adamw"
    assert resolve_optimizer_type("prodigy", "cuda") == "prodigy"
