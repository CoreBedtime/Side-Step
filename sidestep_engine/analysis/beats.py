"""Beat, tempo, and time-signature detection via the *Beat This!* tracker.

Replaces the librosa BPM/time-signature ensemble with a single transformer
forward pass (Foscarin, Schlüter & Widmer, ISMIR 2024).  One pass yields
beat *and* downbeat positions, so tempo comes from tracked beats rather
than from autocorrelation guessing, and the time signature falls out of the
downbeat spacing instead of an accent-pattern heuristic.

The tracker runs without DBN postprocessing (``dbn=False``), which is the
paper's central result -- so ``madmom`` is never required.

Two details are load-bearing and were established by measurement; see the
notes on :func:`_tempo_from_beats` and :func:`_octave_correct`.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Tracker configuration ─────────────────────────────────────────
_CHECKPOINT = "final0"
_FPS = 50.0  # beat_this emits beat times on a 20 ms frame grid

# ── Tempo prior ───────────────────────────────────────────────────
#
# The tracker's residual errors are near-exact half-tempo reads (a 140 BPM
# track tracked at 70).  A log-normal prior over tempo resolves the octave
# far better than acoustic evidence does: measured AUC 0.991, against 0.599
# for onset strength at inter-beat midpoints and 0.463 (worse than chance)
# for the tracker's own frame activations there -- the postprocessor has
# already suppressed those frames, so they carry no signal.
#
# The centre suits contemporary production (~124-155 BPM).  Libraries built
# around genuinely slow material want a lower centre, hence the override.
_TEMPO_CENTRE = 130.0
_TEMPO_SIGMA = 0.55
_PRIOR_THRESHOLD = 0.908
_MAX_BPM = 190.0

_MIN_BEATS = 8

# ── Model cache ───────────────────────────────────────────────────
# The tracker is loaded once and reused across a whole batch; reloading it
# per file dominated the runtime of the pipeline this replaces.
_tracker: Any = None
_tracker_device: Optional[str] = None
_tracker_lock = threading.Lock()


@dataclass
class BeatAnalysis:
    """Result of a single beat-tracking pass."""

    bpm: Optional[int] = None
    signature: Optional[str] = None
    beats_per_bar: Optional[int] = None
    confidence: Dict[str, str] = field(default_factory=dict)
    octave_corrected: bool = False
    n_beats: int = 0

    def as_sidecar_fields(self) -> Dict[str, str]:
        """The subset written to sidecar files (confidence is GUI-only)."""
        out: Dict[str, str] = {}
        if self.bpm is not None:
            out["bpm"] = str(self.bpm)
        if self.signature:
            out["signature"] = self.signature
        return out


def _resolve_device(device: str = "auto") -> str:
    """Resolve 'auto' to the best available device string."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "mps") and torch.mps.is_available():
            return "mps"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except ImportError:
        pass
    return "cpu"


def load_tracker(device: str = "auto") -> Any:
    """Load (or return the cached) Beat This! tracker.

    Args:
        device: Torch device string, or ``"auto"``.

    Returns:
        An ``Audio2Beats`` instance held for the process lifetime.

    Raises:
        ImportError: If ``beat-this`` is not installed.
    """
    global _tracker, _tracker_device

    resolved = _resolve_device(device)
    with _tracker_lock:
        if _tracker is not None and _tracker_device == resolved:
            return _tracker

        try:
            from beat_this.inference import Audio2Beats
        except ImportError as exc:
            raise ImportError(
                "beat-this is required for beat/tempo analysis. "
                "Install with: uv pip install 'side-step[audio_analyze]' "
                "(or pip install 'side-step[audio_analyze]')."
            ) from exc

        if _tracker is not None:
            unload_tracker(_locked=True)

        logger.info("Loading Beat This! tracker (%s) on %s", _CHECKPOINT, resolved)
        _tracker = Audio2Beats(
            checkpoint_path=_CHECKPOINT, device=resolved, dbn=False,
        )
        _tracker_device = resolved
        return _tracker


def unload_tracker(*, _locked: bool = False) -> None:
    """Release the cached tracker and its VRAM."""
    global _tracker, _tracker_device

    def _do() -> None:
        global _tracker, _tracker_device
        if _tracker is None:
            return
        _tracker = None
        _tracker_device = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    if _locked:
        _do()
    else:
        with _tracker_lock:
            _do()


def _tempo_from_beats(beats: np.ndarray) -> Optional[float]:
    """Tempo from the *mean* of inlier inter-beat intervals.

    Deliberately not the median.  Beat times land on a 20 ms frame grid, so
    the median snaps to a single grid cell -- a true 140 BPM track has a
    0.4286 s period whose intervals alternate 0.42/0.44, and taking the
    median reports 142.9 BPM.  Averaging the inliers cancels the
    quantization; this was worth roughly 20 percentage points of accuracy.

    Outlier intervals (dropped or doubled beats) are excluded first so a
    handful of tracking glitches cannot drag the mean.
    """
    beats = np.asarray(beats, dtype=float)
    if len(beats) < _MIN_BEATS:
        return None
    ibis = np.diff(beats)
    median = float(np.median(ibis))
    if median <= 0:
        return None
    inliers = ibis[np.abs(ibis - median) < 0.25 * median]
    period = float(np.mean(inliers)) if len(inliers) >= 4 else median
    return 60.0 / period if period > 0 else None


def _log_prior_gain(bpm: float, centre: float, sigma: float) -> float:
    """Log-prior advantage of ``2 * bpm`` over ``bpm``. Positive favours doubling."""

    def log_p(value: float) -> float:
        return -((np.log2(value / centre)) ** 2) / (2.0 * sigma ** 2)

    return float(log_p(bpm * 2.0) - log_p(bpm))


def _octave_correct(
    bpm: float,
    *,
    centre: float = _TEMPO_CENTRE,
    sigma: float = _TEMPO_SIGMA,
    threshold: float = _PRIOR_THRESHOLD,
    max_bpm: float = _MAX_BPM,
) -> tuple[float, bool]:
    """Resolve half-tempo tracking using a log-normal tempo prior.

    Returns ``(bpm, was_corrected)``.

    Note this only ever *doubles*.  The tracker does not report spuriously
    fast tempi in practice, and a symmetric rule would halve legitimately
    quick material.
    """
    if bpm is None or bpm <= 0:
        return bpm, False
    if bpm * 2.0 <= max_bpm and _log_prior_gain(bpm, centre, sigma) >= threshold:
        return bpm * 2.0, True
    return bpm, False


def _beats_per_bar(downbeats: np.ndarray, bpm: Optional[float]) -> Optional[int]:
    """Beats per bar, measured in absolute time against the *final* tempo.

    Working in seconds rather than in beat indices means octave correction
    is accounted for automatically: if the tempo was doubled, the same bar
    duration simply spans twice as many beats.  Counting tracked-beat
    indices instead would report a corrected 4/4 bar as 2/4.

    Bar lengths above 7 that are even are folded in half -- an 8-beat span
    is two 4/4 bars, not an 8/4 meter.
    """
    downbeats = np.asarray(downbeats, dtype=float)
    if len(downbeats) < 3 or not bpm or bpm <= 0:
        return None
    spans = np.diff(downbeats)
    spans = spans[spans > 0]
    if len(spans) == 0:
        return None
    median = float(np.median(spans))
    inliers = spans[np.abs(spans - median) < 0.25 * median]
    bar_seconds = float(np.mean(inliers)) if len(inliers) >= 2 else median
    beats_per_bar = int(round(bar_seconds * bpm / 60.0))
    while beats_per_bar > 7 and beats_per_bar % 2 == 0:
        beats_per_bar //= 2
    return beats_per_bar if 2 <= beats_per_bar <= 7 else None


def _signature_for(beats_per_bar: Optional[int]) -> Optional[str]:
    """Map a beats-per-bar count onto a time-signature label.

    Compound meter is not separable here: the tracker works at the beat
    level, so 6/8 and 3/4 present identically.  Three beats per bar is
    reported as 3/4, the far more common reading.
    """
    return {
        2: "2/4",
        3: "3/4",
        4: "4/4",
        5: "5/4",
        6: "6/8",
        7: "7/4",
    }.get(beats_per_bar or 0)


def _confidence(
    beats: np.ndarray,
    downbeats: np.ndarray,
    beats_per_bar: Optional[int],
    *,
    octave_corrected: bool,
) -> Dict[str, str]:
    """Grade tempo and signature from beat regularity and bar consistency."""
    conf = {"bpm": "low", "signature": "low"}
    beats = np.asarray(beats, dtype=float)
    if len(beats) < _MIN_BEATS:
        return conf

    ibis = np.diff(beats)
    mean_ibi = float(np.mean(ibis))
    cv = float(np.std(ibis) / (mean_ibi + 1e-10)) if mean_ibi > 0 else 1.0
    conf["bpm"] = "high" if cv < 0.05 else "medium" if cv < 0.15 else "low"
    if octave_corrected:
        # The reported tempo is an inference from a prior, not something the
        # tracker observed; never present that as high confidence.
        conf["bpm"] = "medium" if conf["bpm"] == "high" else conf["bpm"]

    downbeats = np.asarray(downbeats, dtype=float)
    if beats_per_bar and len(downbeats) >= 3:
        spans = np.diff(downbeats)
        spans = spans[spans > 0]
        if len(spans):
            median = float(np.median(spans))
            agree = float(np.mean(np.abs(spans - median) < 0.1 * median))
            conf["signature"] = (
                "high" if agree >= 0.85 else "medium" if agree >= 0.6 else "low"
            )
            if octave_corrected and conf["signature"] == "high":
                conf["signature"] = "medium"
    return conf


def analyze_beats(
    y: np.ndarray,
    sr: int,
    *,
    device: str = "auto",
    tempo_centre: float = _TEMPO_CENTRE,
) -> BeatAnalysis:
    """Detect tempo and time signature from an in-memory mono signal.

    Args:
        y: Mono audio samples.
        sr: Sample rate of *y*.
        device: Torch device string, or ``"auto"``.
        tempo_centre: Centre of the log-normal tempo prior, in BPM. Lower it
            for libraries built around slow material, where the default
            would wrongly double a genuinely slow tempo.

    Returns:
        A :class:`BeatAnalysis`.  Fields are ``None`` when the tracker found
        too few beats to speak with any confidence.
    """
    tracker = load_tracker(device)

    try:
        beats, downbeats = tracker(y, sr)
    except Exception as exc:
        logger.warning("Beat tracking failed: %s", exc)
        return BeatAnalysis()

    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    if len(beats) < _MIN_BEATS:
        logger.info("Beat tracking found only %d beats -- no result", len(beats))
        return BeatAnalysis(n_beats=len(beats))

    raw_bpm = _tempo_from_beats(beats)
    if raw_bpm is None:
        return BeatAnalysis(n_beats=len(beats))

    bpm, corrected = _octave_correct(raw_bpm, centre=tempo_centre)
    beats_per_bar = _beats_per_bar(downbeats, bpm)
    conf = _confidence(beats, downbeats, beats_per_bar, octave_corrected=corrected)

    result = BeatAnalysis(
        bpm=int(round(bpm)),
        signature=_signature_for(beats_per_bar),
        beats_per_bar=beats_per_bar,
        confidence=conf,
        octave_corrected=corrected,
        n_beats=len(beats),
    )
    logger.info(
        "Beats: bpm=%s%s signature=%s (%d beats, conf=%s)",
        result.bpm,
        f" (doubled from {raw_bpm:.1f})" if corrected else "",
        result.signature,
        result.n_beats,
        result.confidence,
    )
    return result
