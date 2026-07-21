"""Tests for :mod:`sidestep_engine.models.acestep_remote_imports`."""

from __future__ import annotations

import importlib
import shutil
import sys
import types
from pathlib import Path

import pytest

from sidestep_engine.models import acestep_remote_imports as ar

# Capture real package location before tests patch ``ar.__file__``.
_REAL_AR_PARENT = Path(ar.__file__).resolve().parent


def _purge_acestep_from_modules() -> None:
    for key in list(sys.modules):
        if key == "acestep" or key.startswith("acestep."):
            del sys.modules[key]


def _real_bundled_tree() -> Path:
    return _REAL_AR_PARENT / "bundled_acestep"


@pytest.fixture(autouse=True)
def _restore_sys_path():
    """Avoid leaking ``sys.path`` inserts from :func:`ar._prepend_acestep_src_paths` across tests."""
    orig = sys.path.copy()
    yield
    sys.path[:] = orig


@pytest.fixture
def path_isolation(monkeypatch, tmp_path):
    """Point ``acestep_remote_imports.__file__`` at a fake repo so sibling ACE-Step paths do not exist."""
    fake_repo = tmp_path / "fakeroot"
    models_dir = fake_repo / "sidestep_engine" / "models"
    models_dir.mkdir(parents=True)
    fake_mod = models_dir / "acestep_remote_imports.py"
    fake_mod.write_text("# test stub\n", encoding="utf-8")
    monkeypatch.setattr(ar, "__file__", str(fake_mod))


class TestSuccessPaths:
    """Three success scenarios for :func:`ar._ensure_acestep_remote_imports`."""

    def test_bundled_only_imports_acept_config(self, path_isolation, monkeypatch, tmp_path):
        """With no external ACE-Step, bundled ``common`` provides ``AceStepConfig``."""
        monkeypatch.delenv("ACESTEP_SRC", raising=False)
        dest = tmp_path / "bundled_copy"
        shutil.copytree(_real_bundled_tree(), dest)
        monkeypatch.setattr(ar, "bundled_acestep_root", lambda: dest)

        _purge_acestep_from_modules()
        ar._ensure_acestep_remote_imports()

        from acestep.models.common.configuration_acestep_v15 import AceStepConfig

        cfg = AceStepConfig()
        assert cfg.model_type == "acestep"

    def test_acestep_src_takes_precedence_over_bundled(self, path_isolation, monkeypatch, tmp_path):
        """First candidate on the list (``ACESTEP_SRC``) wins when it provides ``common``."""
        upstream = tmp_path / "upstream"
        shutil.copytree(_real_bundled_tree(), upstream / "bundled_inner", dirs_exist_ok=True)
        shutil.move(str(upstream / "bundled_inner" / "acestep"), str(upstream / "acestep"))
        shutil.rmtree(upstream / "bundled_inner")
        cfg_file = upstream / "acestep" / "models" / "common" / "configuration_acestep_v15.py"
        text = cfg_file.read_text(encoding="utf-8")
        cfg_file.write_text(
            text + "\nAceStepConfig._SIDE_STEP_TEST_MARKER = 'upstream_wins'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("ACESTEP_SRC", str(upstream))
        dest = tmp_path / "lower_priority_bundled"
        shutil.copytree(_real_bundled_tree(), dest)
        monkeypatch.setattr(ar, "bundled_acestep_root", lambda: dest)

        _purge_acestep_from_modules()
        ar._ensure_acestep_remote_imports()

        from acestep.models.common.configuration_acestep_v15 import AceStepConfig

        assert getattr(AceStepConfig, "_SIDE_STEP_TEST_MARKER", None) == "upstream_wins"

    def test_apg_guidance_import_after_ensure(self, path_isolation, monkeypatch, tmp_path):
        """APG symbols resolve from bundled ``acestep.models.common.apg_guidance``."""
        monkeypatch.delenv("ACESTEP_SRC", raising=False)
        dest = tmp_path / "bundled_apg"
        shutil.copytree(_real_bundled_tree(), dest)
        monkeypatch.setattr(ar, "bundled_acestep_root", lambda: dest)

        _purge_acestep_from_modules()
        ar._ensure_acestep_remote_imports()

        from acestep.models.common.apg_guidance import apg_forward, cfg_forward

        assert callable(apg_forward)
        assert callable(cfg_forward)


class TestFailureAndRegressionPaths:
    """Three failure / regression scenarios."""

    def test_hollow_acestep_src_skipped_bundled_used(self, path_isolation, monkeypatch, tmp_path):
        """``ACESTEP_SRC`` with only ``acestep/__init__.py`` does not satisfy the predicate; bundled wins."""
        hollow = tmp_path / "hollow"
        (hollow / "acestep").mkdir(parents=True)
        (hollow / "acestep" / "__init__.py").write_text('"""hollow"""\n', encoding="utf-8")
        monkeypatch.setenv("ACESTEP_SRC", str(hollow))

        dest = tmp_path / "bundled_after_hollow"
        shutil.copytree(_real_bundled_tree(), dest)
        monkeypatch.setattr(ar, "bundled_acestep_root", lambda: dest)

        _purge_acestep_from_modules()
        ar._ensure_acestep_remote_imports()

        from acestep.models.common.configuration_acestep_v15 import AceStepConfig

        assert AceStepConfig().model_type == "acestep"

    def test_no_valid_tree_import_fails(self, path_isolation, monkeypatch, tmp_path):
        """When no candidate provides ``common``, importing ``AceStepConfig`` fails."""
        monkeypatch.delenv("ACESTEP_SRC", raising=False)
        empty = tmp_path / "empty_bundled"
        empty.mkdir()
        monkeypatch.setattr(ar, "bundled_acestep_root", lambda: empty)

        _purge_acestep_from_modules()
        ar._ensure_acestep_remote_imports()

        with pytest.raises(ImportError):
            importlib.import_module("acestep.models.common.configuration_acestep_v15")

    def test_broken_acestep_models_stub_repaired(self, path_isolation, monkeypatch, tmp_path):
        """Legacy ``acestep.models`` without ``__path__`` is fixed; common imports work."""
        monkeypatch.delenv("ACESTEP_SRC", raising=False)
        dest = tmp_path / "bundled_stub_test"
        shutil.copytree(_real_bundled_tree(), dest)
        monkeypatch.setattr(ar, "bundled_acestep_root", lambda: dest)

        _purge_acestep_from_modules()
        bad = types.ModuleType("acestep.models")
        sys.modules["acestep.models"] = bad

        ar._ensure_acestep_remote_imports()

        spec = ar._safe_find_spec("acestep.models.common.configuration_acestep_v15")
        assert spec is not None

        from acestep.models.common.configuration_acestep_v15 import AceStepConfig

        assert AceStepConfig().model_type == "acestep"
