"""Tests for the Beat This!-based tempo / time-signature analysis.

These cover the pure helpers only -- no model download, no GPU -- because
the load-bearing logic is the arithmetic around the tracker, not the
tracker itself.  Each case here corresponds to a failure measured against
real tagged audio during the overhaul.
"""

from __future__ import annotations

import numpy as np
import pytest

from sidestep_engine.analysis.beats import (
    BeatAnalysis,
    _beats_per_bar,
    _confidence,
    _octave_correct,
    _signature_for,
    _tempo_from_beats,
)


def _grid_beats(bpm: float, n: int = 200, fps: float = 50.0) -> np.ndarray:
    """Beat times snapped to the tracker's 20 ms frame grid."""
    period = 60.0 / bpm
    exact = np.arange(n) * period
    return np.round(exact * fps) / fps


class TestTempoFromBeats:
    """Frame quantization is the whole reason this uses a mean, not a median."""

    @pytest.mark.parametrize("bpm", [140.0, 128.0, 150.0, 174.0, 92.0])
    def test_recovers_tempo_through_frame_quantization(self, bpm):
        got = _tempo_from_beats(_grid_beats(bpm))
        assert got == pytest.approx(bpm, abs=0.5)

    def test_median_would_fail_where_mean_succeeds(self):
        """Guards the specific 140 -> 142.9 regression.

        A 140 BPM period of 0.4286 s lands on 0.42/0.44 grid cells; the
        median picks one cell and reports 142.9 BPM.
        """
        beats = _grid_beats(140.0)
        median_estimate = 60.0 / float(np.median(np.diff(beats)))
        assert abs(median_estimate - 140.0) > 2.0, "median should be wrong here"
        assert _tempo_from_beats(beats) == pytest.approx(140.0, abs=0.5)

    def test_outlier_intervals_do_not_drag_the_mean(self):
        beats = list(_grid_beats(120.0, n=60))
        del beats[20]  # a dropped beat leaves a double-length gap
        del beats[35]
        assert _tempo_from_beats(np.array(beats)) == pytest.approx(120.0, abs=1.0)

    def test_too_few_beats_returns_none(self):
        assert _tempo_from_beats(np.arange(4) * 0.5) is None
        assert _tempo_from_beats(np.array([])) is None


class TestOctaveCorrect:
    """The prior only ever doubles, and only well below the centre."""

    @pytest.mark.parametrize("half,expected", [(70.1, 140.2), (67.0, 134.0),
                                               (72.5, 145.0), (75.0, 150.0)])
    def test_doubles_half_tempo_reads(self, half, expected):
        got, corrected = _octave_correct(half)
        assert corrected is True
        assert got == pytest.approx(expected)

    @pytest.mark.parametrize("bpm", [83.0, 90.0, 95.0, 108.0, 128.0, 140.0, 174.0])
    def test_leaves_plausible_tempi_alone(self, bpm):
        got, corrected = _octave_correct(bpm)
        assert corrected is False
        assert got == pytest.approx(bpm)

    def test_never_exceeds_the_ceiling(self):
        # 96 * 2 = 192 > 190, so it must not double even if the prior likes it.
        got, corrected = _octave_correct(96.0)
        assert corrected is False
        assert got == pytest.approx(96.0)

    def test_lower_centre_suits_a_slow_library(self):
        """A genuinely slow track is doubled by default; the override stops it."""
        assert _octave_correct(75.0)[1] is True
        assert _octave_correct(75.0, centre=95.0)[1] is False

    def test_handles_degenerate_input(self):
        assert _octave_correct(0.0) == (0.0, False)
        assert _octave_correct(-5.0) == (-5.0, False)


class TestBeatsPerBar:
    """Measured in absolute time so octave correction is accounted for."""

    def test_four_four(self):
        downbeats = np.arange(20) * 2.0  # 2 s bars at 120 BPM -> 4 beats
        assert _beats_per_bar(downbeats, 120.0) == 4

    def test_three_four(self):
        downbeats = np.arange(20) * 1.5  # 1.5 s bars at 120 BPM -> 3 beats
        assert _beats_per_bar(downbeats, 120.0) == 3

    def test_corrected_tempo_yields_four_not_two(self):
        """The regression that made corrected 4/4 tracks report as 2/4.

        Bars are 1.71 s.  Against the half-tracked 70 BPM that is 2 beats;
        against the corrected 140 BPM it is the true 4.
        """
        downbeats = np.arange(20) * (4 * 60.0 / 140.0)
        assert _beats_per_bar(downbeats, 70.0) == 2
        assert _beats_per_bar(downbeats, 140.0) == 4

    def test_eight_beat_span_folds_to_four(self):
        """An 8-beat span is two 4/4 bars, not an 8/4 meter."""
        downbeats = np.arange(20) * (8 * 60.0 / 150.0)
        assert _beats_per_bar(downbeats, 150.0) == 4

    def test_rejects_implausible_and_missing(self):
        assert _beats_per_bar(np.array([0.0, 1.0]), 120.0) is None  # too few
        assert _beats_per_bar(np.arange(20) * 2.0, None) is None
        assert _beats_per_bar(np.arange(20) * 0.2, 120.0) is None  # < 2 beats/bar


class TestSignatureFor:
    @pytest.mark.parametrize("bpb,sig", [(2, "2/4"), (3, "3/4"), (4, "4/4"),
                                         (5, "5/4"), (6, "6/8"), (7, "7/4")])
    def test_known_meters(self, bpb, sig):
        assert _signature_for(bpb) == sig

    def test_unknown_returns_none(self):
        assert _signature_for(None) is None
        assert _signature_for(0) is None
        assert _signature_for(9) is None


class TestConfidence:
    def test_steady_beats_score_high(self):
        beats = _grid_beats(128.0, n=120)
        downbeats = beats[::4]
        conf = _confidence(beats, downbeats, 4, octave_corrected=False)
        assert conf["bpm"] == "high"
        assert conf["signature"] == "high"

    def test_octave_corrected_never_reports_high(self):
        """A doubled tempo is inferred from a prior, not observed."""
        beats = _grid_beats(70.0, n=120)
        downbeats = beats[::2]
        conf = _confidence(beats, downbeats, 4, octave_corrected=True)
        assert conf["bpm"] != "high"
        assert conf["signature"] != "high"

    def test_erratic_beats_score_low(self):
        rng = np.random.default_rng(0)
        beats = np.cumsum(rng.uniform(0.2, 0.9, size=120))
        conf = _confidence(beats, np.array([]), None, octave_corrected=False)
        assert conf["bpm"] == "low"
        assert conf["signature"] == "low"

    def test_too_few_beats_is_low(self):
        conf = _confidence(np.arange(3) * 0.5, np.array([]), None,
                           octave_corrected=False)
        assert conf == {"bpm": "low", "signature": "low"}


class TestBeatAnalysis:
    def test_sidecar_fields_exclude_confidence(self):
        r = BeatAnalysis(bpm=140, signature="4/4",
                         confidence={"bpm": "high", "signature": "high"})
        assert r.as_sidecar_fields() == {"bpm": "140", "signature": "4/4"}

    def test_empty_result_writes_nothing(self):
        assert BeatAnalysis().as_sidecar_fields() == {}

    def test_partial_result_writes_what_it_has(self):
        assert BeatAnalysis(bpm=95).as_sidecar_fields() == {"bpm": "95"}
