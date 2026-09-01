"""
Local offline audio analysis for Side-Step.

Extracts BPM, time signature, and (optionally) musical key from an audio
file.  Two quality modes are available:

**standard** — ~1 s per file
    Beat This! tracker for tempo + time signature.  No stem separation,
    no key.  This is the default and covers most dataset work.

**sas** ("Smart/Slow As Sh*t") — ~20-30 s GPU
    Everything in *standard*, plus musical key from Demucs-separated
    harmonics with multi-chroma fusion, tonnetz disambiguation, tuning
    correction, ending resolution, and energy-gated chunked voting.

Tempo and time signature come from :mod:`sidestep_engine.analysis.beats`,
which replaced a librosa ensemble that was both slower and less accurate:
on 60 held-out BPM-tagged files the old F-A-F path scored 47/60 at 2.24
s/file against 49/60 at 0.83 s/file for the tracker.  It also returns a
real time signature, where F-A-F hardcoded ``"4/4"``.

Key detection returns ``(value, confidence)`` where confidence is
``"high"``, ``"medium"``, or ``"low"``.  Confidence is surfaced in the
GUI but **not** written to sidecar files.
"""

from __future__ import annotations

import gc
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Analysis modes ─────────────────────────────────────────────────
MODES = ("standard", "sas")
_DEFAULT_MODE = "standard"

#: Pre-overhaul mode names.  ``faf``/``mid`` both map to ``standard``;
#: note that ``mid`` used to produce a key and no longer does -- key now
#: requires ``sas``, which is the trade that makes the default fast.
_LEGACY_MODES = {"faf": "standard", "mid": "standard"}

_SAS_NUM_CHUNKS = 5
_SAS_CHUNK_SECONDS = 15  # seconds per analysis window

#: Sample rate for beat tracking. The tracker resamples internally, and
#: nothing downstream of it needs the native rate.
_BEAT_SR = 22050


def normalize_mode(mode: str) -> str:
    """Map a possibly-legacy mode name onto a current one."""
    resolved = _LEGACY_MODES.get(mode, mode)
    if resolved not in MODES:
        logger.warning(
            "Unknown analysis mode '%s', falling back to '%s'", mode, _DEFAULT_MODE,
        )
        return _DEFAULT_MODE
    if resolved != mode:
        logger.info("Analysis mode '%s' is legacy; using '%s'", mode, resolved)
    return resolved


def _select_chunks(
    y: np.ndarray,
    sr: int,
    n_chunks: int = _SAS_NUM_CHUNKS,
    chunk_sec: float = _SAS_CHUNK_SECONDS,
    min_gap_sec: float = 10.0,
    use_onset: bool = True,
) -> list[np.ndarray]:
    """Select the most informative audio chunks for S-A-S analysis.

    Energy-gated + spread: rank windows by onset density (or RMS),
    discard below-median, then greedily pick *n_chunks* that are
    maximally spread apart (at least *min_gap_sec* between centres).

    Args:
        y: Mono audio signal.
        sr: Sample rate.
        n_chunks: How many chunks to return.
        chunk_sec: Duration of each chunk in seconds.
        min_gap_sec: Minimum gap between chunk centres.
        use_onset: If True, rank by onset density; else by RMS.

    Returns:
        List of audio arrays (each ~chunk_sec long).
    """
    import librosa

    chunk_samples = int(chunk_sec * sr)
    hop_samples = chunk_samples // 2  # 50 % overlap for candidate windows
    if len(y) < chunk_samples:
        return [y]

    # Build candidate windows
    candidates: list[tuple[int, float]] = []  # (start_sample, score)
    for start in range(0, len(y) - chunk_samples + 1, hop_samples):
        window = y[start : start + chunk_samples]
        if use_onset:
            onset_env = librosa.onset.onset_strength(y=window, sr=sr)
            score = float(np.mean(onset_env))
        else:
            score = float(np.sqrt(np.mean(window ** 2)))  # RMS
        candidates.append((start, score))

    if not candidates:
        return [y]

    # Gate: discard below-median energy
    scores = np.array([s for _, s in candidates])
    median_score = float(np.median(scores))
    gated = [(start, score) for start, score in candidates if score >= median_score]
    if not gated:
        gated = candidates

    # Sort by score descending
    gated.sort(key=lambda x: x[1], reverse=True)

    # Greedy spread selection
    min_gap_samples = int(min_gap_sec * sr)
    selected_starts: list[int] = []
    selected_scores: list[float] = []

    for start, score in gated:
        centre = start + chunk_samples // 2
        too_close = any(
            abs(centre - (s + chunk_samples // 2)) < min_gap_samples
            for s in selected_starts
        )
        if not too_close:
            selected_starts.append(start)
            selected_scores.append(score)
            if len(selected_starts) >= n_chunks:
                break

    # If we didn't get enough, relax the gap constraint
    if len(selected_starts) < n_chunks:
        for start, score in gated:
            if start not in selected_starts:
                selected_starts.append(start)
                selected_scores.append(score)
                if len(selected_starts) >= n_chunks:
                    break

    # Return in chronological order
    selected_starts.sort()
    chunks = [y[s : s + chunk_samples] for s in selected_starts]
    return chunks


# ── Key profile families for multi-profile voting ────────────────
#
# Each family provides (major, minor) weight vectors over 12 pitch classes
# starting from C.  Using multiple families and voting across them
# significantly reduces key-detection errors.

_KEY_PROFILES = {
    "krumhansl": {
        "major": np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
        "minor": np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
    },
    "temperley": {
        "major": np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0,
                           2.0, 4.5, 2.0, 3.5, 1.5, 4.0]),
        "minor": np.array([5.0, 2.0, 3.5, 4.5, 2.0, 3.5,
                           2.0, 4.5, 3.5, 2.0, 1.5, 4.0]),
    },
    "albrecht": {
        "major": np.array([0.238, 0.006, 0.111, 0.006, 0.137, 0.094,
                           0.016, 0.214, 0.009, 0.080, 0.008, 0.081]),
        "minor": np.array([0.220, 0.006, 0.104, 0.123, 0.019, 0.103,
                           0.012, 0.214, 0.062, 0.022, 0.061, 0.052]),
    },
}

_PITCH_CLASSES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]


def _flush_vram() -> None:
    """Release GPU memory between pipeline stages."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch, 'mps') and torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
    except ImportError:
        pass


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


def _preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Trim silence and peak-normalize an audio signal.

    Args:
        y: Audio time series (mono).
        sr: Sample rate.

    Returns:
        Preprocessed audio array.
    """
    import librosa

    # Trim leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trimmed) < sr:  # less than 1 second after trim — keep original
        y_trimmed = y

    # Peak-normalize to 0 dBFS
    peak = np.max(np.abs(y_trimmed))
    if peak > 0:
        y_trimmed = y_trimmed / peak

    return y_trimmed


# ── Step 1: Demucs stem separation ─────────────────────────────────


def separate_stems(
    audio_path: Path,
    output_dir: Path,
    device: str = "auto",
) -> tuple[Path, Path]:
    """Run Demucs HTDemucs and return paths to drums and harmonics stems.

    Uses ``demucs.pretrained.get_model`` + ``demucs.apply.apply_model``
    (the v4.0.x low-level API).  The harmonics stem is created by
    summing the bass and other stems.  Vocals are discarded.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory to write stem WAV files into.
        device: Torch device string (auto, cuda, cpu).

    Returns:
        ``(drums_path, harmonics_path)`` as absolute Paths.
    """
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    device = _resolve_device(device)
    torch_device = torch.device(device)

    logger.info("Loading Demucs HTDemucs model on %s", device)
    model = get_model("htdemucs")
    model.to(torch_device)
    model.eval()

    # Load audio — torchaudio returns (channels, samples) and sample rate
    wav, sr = torchaudio.load(str(audio_path))

    # Resample to model's expected rate (44100 Hz) if needed
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        sr = model.samplerate

    # HTDemucs requires stereo (2-channel) input — duplicate mono if needed
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    # apply_model expects (batch, channels, samples)
    wav = wav.unsqueeze(0).to(torch_device)

    logger.info("Separating stems for %s", audio_path.name)
    with torch.no_grad():
        # apply_model returns (batch, num_sources, channels, samples)
        # apply_model signature: (model, mix, shifts, split, overlap,
        #   transition_power, progress, device, num_workers, segment)
        sources = apply_model(model, wav, device=torch_device)

    # model.sources == ['drums', 'bass', 'other', 'vocals']
    source_map = {name: i for i, name in enumerate(model.sources)}

    drums = sources[0, source_map["drums"]].cpu()
    bass = sources[0, source_map["bass"]].cpu()
    other = sources[0, source_map["other"]].cpu()

    # Merge bass + other → harmonics
    harmonics = bass + other

    # Write to WAV files
    drums_path = output_dir / "drums.wav"
    harmonics_path = output_dir / "harmonics.wav"

    torchaudio.save(str(drums_path), drums, sr)
    torchaudio.save(str(harmonics_path), harmonics, sr)

    # Free the model and tensors
    del model, sources, wav, drums, bass, other, harmonics
    _flush_vram()

    logger.info("Stems written: %s, %s", drums_path, harmonics_path)
    return drums_path, harmonics_path


# ── Step 4: Key detection (librosa + Krumhansl-Schmuckler) ────────


def _best_key_for_profile(
    chroma_avg: np.ndarray,
    major_profile: np.ndarray,
    minor_profile: np.ndarray,
) -> tuple[str, float]:
    """Find the best key match for a single profile family.

    Returns ``(key_label, correlation)``.
    """
    major_norm = major_profile / major_profile.sum()
    minor_norm = minor_profile / minor_profile.sum()

    best_corr = -2.0
    best_key = "C major"

    for shift in range(12):
        rotated = np.roll(chroma_avg, -shift)

        corr_maj = float(np.corrcoef(rotated, major_norm)[0, 1])
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = f"{_PITCH_CLASSES[shift]} major"

        corr_min = float(np.corrcoef(rotated, minor_norm)[0, 1])
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = f"{_PITCH_CLASSES[shift]} minor"

    return best_key, best_corr


def _key_votes_from_chroma(
    chroma_avg: np.ndarray,
    profiles: dict | None = None,
) -> list[tuple[str, float]]:
    """Vote on key from a single chroma vector using specified profiles.

    Returns list of ``(key_label, correlation)`` — one per profile family.
    If *profiles* is None, uses all ``_KEY_PROFILES``.
    """
    if profiles is None:
        profiles = _KEY_PROFILES

    results: list[tuple[str, float]] = []
    for name, pf in profiles.items():
        key_label, corr = _best_key_for_profile(
            chroma_avg, pf["major"], pf["minor"],
        )
        results.append((key_label, corr))
    return results


def _energy_weighted_chroma(
    chroma: np.ndarray,
    y_harmonic: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute an energy-weighted average chroma vector."""
    import librosa

    rms = librosa.feature.rms(y=y_harmonic, frame_length=2048, hop_length=512)
    rms_vec = rms[0]
    min_len = min(chroma.shape[1], len(rms_vec))
    chroma = chroma[:, :min_len]
    rms_vec = rms_vec[:min_len]

    weights = rms_vec / (rms_vec.sum() + 1e-10)
    chroma_avg = (chroma * weights[np.newaxis, :]).sum(axis=1)

    s = chroma_avg.sum()
    if s == 0:
        return None
    return chroma_avg / s


def detect_key(
    audio_path: Path,
    *,
    mode: str = _DEFAULT_MODE,
    n_chunks: int = _SAS_NUM_CHUNKS,
) -> tuple[Optional[str], str]:
    """Detect musical key with quality controlled by *mode*.

    - **standard**: 3-profile × energy-weighted ``chroma_cens`` × 8 s
      segment voting.
    - **sas**: standard + multi-chroma fusion (cens/cqt/stft) + tonnetz
      disambiguation + tuning correction + ending resolution +
      energy-gated chunked voting.

    Note that :func:`analyze_audio` only runs key detection in ``sas``;
    ``standard`` mode skips it entirely, which is what makes it fast.

    Returns ``(key, confidence)``.
    """
    try:
        import librosa
        from collections import Counter

        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        y = _preprocess_audio(y, sr)

        # Harmonic enhancement
        y_harmonic = librosa.effects.harmonic(y, margin=4.0)

        # ── sas: tuning correction ──
        tuning = 0.0
        if mode == "sas":
            try:
                tuning = float(librosa.estimate_tuning(y=y_harmonic, sr=sr))
                logger.debug("Tuning correction: %.3f bins", tuning)
            except Exception:
                tuning = 0.0

        # ── multi-profile voting ──
        all_votes: list[str] = []
        all_weights: list[float] = []

        # Determine which chroma types to use
        if mode == "sas":
            chroma_types = {
                "cens": lambda: librosa.feature.chroma_cens(
                    y=y_harmonic, sr=sr, tuning=tuning,
                ),
                "cqt": lambda: librosa.feature.chroma_cqt(
                    y=y_harmonic, sr=sr, tuning=tuning,
                ),
                "stft": lambda: librosa.feature.chroma_stft(
                    y=y_harmonic, sr=sr, tuning=tuning,
                ),
            }
        else:
            chroma_types = {
                "cens": lambda: librosa.feature.chroma_cens(
                    y=y_harmonic, sr=sr,
                ),
            }

        for chroma_name, chroma_fn in chroma_types.items():
            try:
                chroma = chroma_fn()
            except Exception:
                continue

            chroma_avg = _energy_weighted_chroma(chroma, y_harmonic)
            if chroma_avg is None:
                continue

            # Global multi-profile vote
            for key_label, corr in _key_votes_from_chroma(chroma_avg):
                all_votes.append(key_label)
                all_weights.append(1.0)

            # Segment-based voting
            rms = librosa.feature.rms(
                y=y_harmonic, frame_length=2048, hop_length=512,
            )
            rms_vec = rms[0]
            min_len = min(chroma.shape[1], len(rms_vec))
            chroma_s = chroma[:, :min_len]
            rms_s = rms_vec[:min_len]

            seg_frames = int(8.0 * sr / 512)
            n_segments = max(1, chroma_s.shape[1] // seg_frames)

            for seg_i in range(n_segments):
                start = seg_i * seg_frames
                end = min(start + seg_frames, chroma_s.shape[1])
                seg_chroma = chroma_s[:, start:end]
                seg_w = rms_s[start:end]

                w_sum = seg_w.sum()
                if w_sum < 1e-10:
                    continue

                seg_w_norm = seg_w / w_sum
                seg_avg = (seg_chroma * seg_w_norm[np.newaxis, :]).sum(axis=1)
                s = seg_avg.sum()
                if s < 1e-10:
                    continue
                seg_avg = seg_avg / s

                for key_label, _ in _key_votes_from_chroma(seg_avg):
                    all_votes.append(key_label)
                    all_weights.append(1.0)

        # ── sas-only extras ──
        if mode == "sas":
            # Tonnetz — weighted vote for major/minor disambiguation
            try:
                tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
                tonnetz_avg = np.mean(tonnetz, axis=1)
                # Dimensions 4-5 are major-third projections,
                # dimensions 2-3 are minor-third projections
                major_energy = float(np.sum(tonnetz_avg[4:6] ** 2))
                minor_energy = float(np.sum(tonnetz_avg[2:4] ** 2))
                # Weight factor: how strongly tonnetz leans major vs minor
                tonnetz_ratio = major_energy / (minor_energy + 1e-10)

                # Get the current leading vote to check if tonnetz agrees
                if all_votes:
                    temp_counts = Counter(all_votes)
                    leader = temp_counts.most_common(1)[0][0]
                    leader_is_major = "major" in leader
                    tonnetz_says_major = tonnetz_ratio > 1.0

                    if leader_is_major == tonnetz_says_major:
                        # Agreement — add weighted votes for the leader
                        all_votes.extend([leader] * 3)
                        all_weights.extend([1.5] * 3)
                    else:
                        # Disagreement — add votes for the relative key
                        # with moderate weight
                        alt_mode = "minor" if leader_is_major else "major"
                        # Find the best key of the other mode from chroma
                        chroma_cens = librosa.feature.chroma_cens(
                            y=y_harmonic, sr=sr, tuning=tuning,
                        )
                        ca = _energy_weighted_chroma(chroma_cens, y_harmonic)
                        if ca is not None:
                            for name, pf in _KEY_PROFILES.items():
                                prof = pf[alt_mode]
                                prof_norm = prof / prof.sum()
                                best_corr = -2.0
                                best_k = ""
                                for shift in range(12):
                                    rotated = np.roll(ca, -shift)
                                    c = float(np.corrcoef(rotated, prof_norm)[0, 1])
                                    if c > best_corr:
                                        best_corr = c
                                        best_k = f"{_PITCH_CLASSES[shift]} {alt_mode}"
                                if best_k:
                                    all_votes.append(best_k)
                                    all_weights.append(1.0)
            except Exception:
                pass

            # Ending resolution — last ~5 s weighted extra
            try:
                end_samples = min(int(5.0 * sr), len(y_harmonic))
                y_end = y_harmonic[-end_samples:]
                chroma_end = librosa.feature.chroma_cens(
                    y=y_end, sr=sr, tuning=tuning,
                )
                end_avg = np.mean(chroma_end, axis=1)
                s = end_avg.sum()
                if s > 1e-10:
                    end_avg = end_avg / s
                    for key_label, _ in _key_votes_from_chroma(end_avg):
                        all_votes.append(key_label)
                        all_weights.append(2.0)  # ending gets double weight
            except Exception:
                pass

            # Chunked voting (RMS-gated for harmonic content)
            chunks = _select_chunks(
                y_harmonic, sr, n_chunks=n_chunks, use_onset=False,
            )
            for chunk in chunks:
                try:
                    ch_chroma = librosa.feature.chroma_cens(
                        y=chunk, sr=sr, tuning=tuning,
                    )
                    ch_avg = _energy_weighted_chroma(ch_chroma, chunk)
                    if ch_avg is not None:
                        for key_label, _ in _key_votes_from_chroma(ch_avg):
                            all_votes.append(key_label)
                            all_weights.append(1.0)
                except Exception:
                    pass

        # ── Final vote ──
        if not all_votes:
            return None, "low"

        # Weighted majority vote
        weighted_counts: dict[str, float] = {}
        for vote, w in zip(all_votes, all_weights):
            weighted_counts[vote] = weighted_counts.get(vote, 0.0) + w

        best_key = max(weighted_counts, key=weighted_counts.get)
        total_weight = sum(all_weights)
        best_weight = weighted_counts[best_key]
        share = best_weight / total_weight

        if share >= 0.55:
            confidence = "high"
        elif share >= 0.35:
            confidence = "medium"
        else:
            confidence = "low"

        logger.info(
            "Key [%s]: %s (share=%.0f%%, votes=%d, conf=%s)",
            mode, best_key, share * 100, len(all_votes), confidence,
        )
        return best_key, confidence

    except Exception as exc:
        logger.warning("Key detection failed: %s", exc)
        return None, "low"


# ── Orchestrator ──────────────────────────────────────────────────


def analysis_fields_for(mode: str) -> tuple[str, ...]:
    """Sidecar fields a given mode is capable of producing.

    Callers use this to decide whether a sidecar is already complete
    *before* paying for analysis -- ``standard`` never yields a key, so
    waiting on one would re-analyse every file forever.
    """
    return ("bpm", "signature") if normalize_mode(mode) == "standard" else (
        "bpm", "signature", "key"
    )


def analyze_audio(
    audio_path: Path,
    *,
    device: str = "auto",
    mode: str = _DEFAULT_MODE,
    n_chunks: int = _SAS_NUM_CHUNKS,
    tempo_centre: float | None = None,
) -> Dict[str, Any]:
    """Run the audio analysis pipeline on a single file.

    **standard** — Beat This! tempo + time signature (~1 s).
    **sas** — adds Demucs-separated key detection (~20-30 s GPU).

    The audio is decoded once and the buffer reused; the previous
    implementation decoded each file three times, once per detector.

    Args:
        audio_path: Path to the input audio file.
        device: Torch device (auto, cuda, cpu, mps).
        mode: Quality tier — ``"standard"`` or ``"sas"``. Legacy ``faf``
            and ``mid`` are accepted and map to ``standard``.
        n_chunks: Number of analysis chunks for S-A-S key detection.
        tempo_centre: Override the log-normal tempo prior's centre, in
            BPM. Lower it for libraries of predominantly slow material,
            where the default would wrongly double a genuine slow tempo.

    Returns:
        Dict with ``bpm``, ``signature`` and (in ``sas``) ``key`` string
        values for sidecar writing, plus a ``confidence`` sub-dict with
        per-field levels (GUI-only, not persisted).
    """
    from sidestep_engine.analysis.beats import analyze_beats

    mode = normalize_mode(mode)
    result: Dict[str, Any] = {}
    confidence: Dict[str, str] = {}

    logger.info("Starting audio analysis [%s] for %s", mode, audio_path.name)

    # ── Tempo + time signature (both modes) ──
    import librosa

    y, sr = librosa.load(str(audio_path), sr=_BEAT_SR, mono=True)
    beat_kwargs: Dict[str, Any] = {"device": device}
    if tempo_centre is not None:
        beat_kwargs["tempo_centre"] = tempo_centre
    beats = analyze_beats(y, sr, **beat_kwargs)

    result.update(beats.as_sidecar_fields())
    confidence.update(
        {k: v for k, v in beats.confidence.items() if k in result}
    )
    del y

    # ── Key (sas only — needs Demucs to strip vocals from the chroma) ──
    if mode == "sas":
        try:
            import demucs  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Demucs is required for sas key detection. "
                "Install with: uv pip install 'side-step[audio_analyze]' "
                "(or pip install 'side-step[audio_analyze]'). "
                "Alternatively use mode standard (no key)."
            ) from exc

        tmp_dir = Path(tempfile.mkdtemp(prefix="sidestep_analysis_"))
        try:
            _, harmonics_path = separate_stems(audio_path, tmp_dir, device=device)
            key, key_conf = detect_key(
                harmonics_path, mode="sas", n_chunks=n_chunks,
            )
            if key:
                result["key"] = key
                confidence["key"] = key_conf
        finally:
            try:
                shutil.rmtree(tmp_dir)
            except OSError as exc:
                logger.debug("Could not clean temp dir %s: %s", tmp_dir, exc)

    result["confidence"] = confidence
    logger.info("Analysis complete for %s: %s", audio_path.name, result)
    return result


def analyze_audio_safe(
    audio_path: Path,
    *,
    device: str = "auto",
    mode: str = _DEFAULT_MODE,
) -> Dict[str, Any]:
    """Wrapper around :func:`analyze_audio` that catches exceptions.

    Returns a dict with ``status`` (``"ok"`` or ``"failed"``),
    the analysis ``fields``, and an optional ``error`` message.
    """
    try:
        fields = analyze_audio(audio_path, device=device, mode=mode)
        return {"status": "ok", "fields": fields}
    except Exception as exc:
        return {"status": "failed", "fields": {}, "error": str(exc)}


def unload_models() -> None:
    """Release cached analysis models (call when a batch finishes)."""
    from sidestep_engine.analysis.beats import unload_tracker

    unload_tracker()
    _flush_vram()
