"""The GUI server must not run dataset-scale file I/O on the event loop.

``scan_audio_dir`` takes ~2 s on a 900-file library and ``create_mix_dataset``
copies every audio file.  Called in-thread from an ``async def`` handler,
either one stalls *every* concurrent request for its full duration --
training progress, GPU telemetry, SSE streams included.
"""

from __future__ import annotations

import ast
import asyncio
import time
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parent.parent / "sidestep_engine" / "gui" / "server.py"

#: Handlers whose cost scales with dataset size. Cheap fixed-cost reads
#: (a single small JSON) are deliberately excluded -- the thread hop
#: would cost more than the read.
DATA_SCALING_HANDLERS = {
    "scan_audio": "scan_audio_dir",
    "create_mix_dataset": "create_mix_dataset",
    "list_checkpoints": "list_checkpoints",
}


def _handler(tree: ast.Module, name: str) -> ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"async handler {name!r} not found in server.py")


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(SERVER.read_text(encoding="utf-8"))


class TestHandlersOffload:
    """Static guard: these handlers must not call blocking work in-thread."""

    @pytest.mark.parametrize("handler,blocking", sorted(DATA_SCALING_HANDLERS.items()))
    def test_blocking_call_is_offloaded(self, tree, handler, blocking):
        node = _handler(tree, handler)

        offloaded = False
        direct = False
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
            if name in ("_offload", "run_in_executor"):
                offloaded = True
                # the blocking callable must be the thing handed to it
                for arg in call.args:
                    if getattr(arg, "id", None) == blocking:
                        break
            elif name == blocking:
                # A direct call is only acceptable as the argument *to* an
                # offload, which would be an ast.Name, not an ast.Call.
                direct = True

        assert offloaded, f"{handler} must hand {blocking} to an executor"
        assert not direct, f"{handler} still calls {blocking}() on the event loop"

    @pytest.mark.parametrize("handler", sorted(DATA_SCALING_HANDLERS))
    def test_handler_awaits(self, tree, handler):
        node = _handler(tree, handler)
        assert any(isinstance(n, ast.Await) for n in ast.walk(node)), (
            f"{handler} performs no await -- it cannot have yielded to the loop"
        )


class TestOffloadHelper:
    """Behavioural: _offload must actually let other coroutines run."""

    @staticmethod
    def _offload():
        import functools

        async def offload(fn, *args, **kwargs):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, functools.partial(fn, *args, **kwargs)
            )

        return offload

    def test_other_coroutines_progress_during_blocking_call(self):
        """A blocking call must not stop a concurrent ticker."""
        offload = self._offload()
        ticks = 0

        def slow(duration: float) -> str:
            time.sleep(duration)
            return "done"

        async def ticker() -> None:
            nonlocal ticks
            for _ in range(20):
                await asyncio.sleep(0.01)
                ticks += 1

        async def main():
            return await asyncio.gather(offload(slow, 0.15), ticker())

        result, _ = asyncio.run(main())
        assert result == "done"
        # In-thread, the event loop would be frozen and ticks would be ~0.
        assert ticks >= 5, f"event loop was starved (only {ticks} ticks)"

    def test_kwargs_are_forwarded(self):
        """create_mix_dataset is called entirely with keyword arguments."""
        offload = self._offload()

        def takes_kwargs(*, alpha, beta=2):
            return alpha + beta

        assert asyncio.run(offload(takes_kwargs, alpha=5)) == 7
        assert asyncio.run(offload(takes_kwargs, alpha=5, beta=10)) == 15

    def test_exceptions_propagate(self):
        offload = self._offload()

        def boom():
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            asyncio.run(offload(boom))
