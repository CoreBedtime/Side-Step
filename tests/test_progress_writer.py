"""Tests for sidestep_engine.core.progress_writer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestProgressWriter(unittest.TestCase):
    def test_maybe_write_emits_each_new_step_without_time_gate(self) -> None:
        """Fast successive steps must all be recorded (loss curve by step)."""
        from sidestep_engine.core.progress_writer import ProgressWriter

        with tempfile.TemporaryDirectory() as td:
            pw = ProgressWriter(td, interval=3600.0)
            pw.maybe_write(step=1, loss=0.5, epoch=1, max_epochs=2)
            pw.maybe_write(step=2, loss=0.4, epoch=1, max_epochs=2)
            pw.maybe_write(step=3, loss=0.3, epoch=1, max_epochs=2)
            pw.close()
            lines = (Path(td) / ".progress.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)
        for i, line in enumerate(lines, start=1):
            row = json.loads(line)
            self.assertEqual(row["step"], i)
            self.assertEqual(row["kind"], "step")

    def test_duplicate_step_skipped(self) -> None:
        from sidestep_engine.core.progress_writer import ProgressWriter

        with tempfile.TemporaryDirectory() as td:
            pw = ProgressWriter(td, interval=0.0)
            pw.maybe_write(step=5, loss=1.0)
            pw.maybe_write(step=5, loss=2.0)
            pw.close()
            lines = (Path(td) / ".progress.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["loss"], 1.0)


if __name__ == "__main__":
    unittest.main()
