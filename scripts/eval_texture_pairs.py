"""Paired texture-corruption scorer for noFSQ cover experiments.

Given (source, cover, lora_scale) triples — same arrangement, cover generated
with the texture LoRA at a given scale — measures:

  - mrstft:    multi-resolution log-STFT distance to source (artifact meter;
               should INCREASE monotonically with scale)
  - hf_ratio:  high-frequency (>8 kHz) energy ratio delta vs source
  - flatness:  mean spectral-flatness delta vs source (noise-floor lift)
  - lat_mod:   envelope modulation energy near the DiT latent frame rate
               (a known codec/DiT signature band)
  - env_corr:  RMS-envelope Pearson correlation with source (arrangement
               preservation; should stay HIGH and FLAT across scales)

Verdicts: monotone dose-response of mrstft over scale (Spearman) = the LoRA
encodes a controllable texture direction; flat env_corr = content untouched.

Usage:
    uv run python scripts/eval_texture_pairs.py --manifest pairs.csv
        [--sr 24000] [--latent-rate 14.3]

pairs.csv columns (no header): source_path,cover_path,scale
Scale 0 rows (covers WITHOUT the LoRA) are the baseline — include some.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


def _load_mono(path: str, sr: int):
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    data, in_sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(data.mean(axis=1))
    if in_sr != sr:
        x = AF.resample(x, in_sr, sr)
    return x


def _metrics(src, cov, sr: int, latent_rate: float) -> dict:
    import torch

    n = min(src.shape[-1], cov.shape[-1])
    src, cov = src[:n], cov[:n]

    # -- multi-resolution log-STFT distance --------------------------------
    mr = 0.0
    for n_fft in (512, 1024, 2048):
        w = torch.hann_window(n_fft)
        S = torch.stft(src, n_fft, n_fft // 4, window=w, return_complex=True).abs()
        C = torch.stft(cov, n_fft, n_fft // 4, window=w, return_complex=True).abs()
        m = min(S.shape[-1], C.shape[-1])
        S, C = S[..., :m].clamp(min=1e-7), C[..., :m].clamp(min=1e-7)
        mr += float((S.log() - C.log()).abs().mean())
    mr /= 3.0

    # -- shared 1024-pt spectrogram for band metrics -----------------------
    n_fft = 1024
    w = torch.hann_window(n_fft)
    S = torch.stft(src, n_fft, n_fft // 4, window=w, return_complex=True).abs()
    C = torch.stft(cov, n_fft, n_fft // 4, window=w, return_complex=True).abs()
    m = min(S.shape[-1], C.shape[-1]); S, C = S[..., :m], C[..., :m]
    freqs_per_bin = sr / n_fft
    hf_bin = int(8000 / freqs_per_bin)

    def hf_ratio(X):
        return float(X[hf_bin:].pow(2).sum() / X.pow(2).sum().clamp(min=1e-12))

    def flatness(X):
        X = X.clamp(min=1e-7)
        return float((X.log().mean(dim=0).exp() / X.mean(dim=0)).log().mean())

    # -- envelope-domain metrics -------------------------------------------
    hop = int(sr * 0.010)  # 10 ms RMS envelope
    def envelope(x):
        frames = x[: (x.shape[-1] // hop) * hop].view(-1, hop)
        return frames.pow(2).mean(dim=1).sqrt()

    es, ec = envelope(src), envelope(cov)
    k = min(es.shape[0], ec.shape[0]); es, ec = es[:k], ec[:k]
    es_c, ec_c = es - es.mean(), ec - ec.mean()
    env_corr = float((es_c * ec_c).sum() /
                     (es_c.norm() * ec_c.norm()).clamp(min=1e-12))

    def latmod(env):
        # envelope modulation spectrum; energy near the latent frame rate
        env = env - env.mean()
        spec = torch.fft.rfft(env).abs()
        env_sr = 1.0 / 0.010  # 100 Hz envelope rate
        f = torch.fft.rfftfreq(env.shape[0], d=1.0 / env_sr)
        band = (f >= latent_rate * 0.8) & (f <= latent_rate * 1.2)
        ref = (f >= 2.0) & (f <= 40.0)
        return float(spec[band].pow(2).sum() / spec[ref].pow(2).sum().clamp(min=1e-12))

    return {
        "mrstft": mr,
        "hf_ratio_delta": hf_ratio(C) - hf_ratio(S),
        "flatness_delta": flatness(C) - flatness(S),
        "lat_mod_delta": latmod(ec) - latmod(es),
        "env_corr": env_corr,
    }


def _spearman(xs, ys) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True,
                    help="CSV: source_path,cover_path,scale (no header)")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--latent-rate", type=float, default=25.0,
                    help="DiT latent frame rate in Hz (signature band). "
                         "ACE-Step 1.5: 25 fps (250 frames per 10 s)")
    args = ap.parse_args()

    rows = [r for r in csv.reader(open(args.manifest, encoding="utf-8")) if r]
    by_scale: dict = defaultdict(list)
    for src_p, cov_p, scale in rows:
        m = _metrics(_load_mono(src_p, args.sr), _load_mono(cov_p, args.sr),
                     args.sr, args.latent_rate)
        by_scale[float(scale)].append(m)
        print(f"[pair] scale={float(scale):g} {Path(cov_p).name}: "
              f"mrstft={m['mrstft']:.4f} env_corr={m['env_corr']:.3f}", file=sys.stderr)

    scales = sorted(by_scale)
    print("\n=== texture dose-response report ===")
    print(f"{'scale':>7} {'n':>3} {'mrstft':>9} {'hf_delta':>9} {'flat_d':>8} "
          f"{'latmod_d':>9} {'env_corr':>9}")
    means = {}
    for s in scales:
        ms = by_scale[s]
        mean = {k: sum(m[k] for m in ms) / len(ms) for k in ms[0]}
        means[s] = mean
        print(f"{s:>7g} {len(ms):>3} {mean['mrstft']:>9.4f} {mean['hf_ratio_delta']:>+9.4f} "
              f"{mean['flatness_delta']:>+8.3f} {mean['lat_mod_delta']:>+9.4f} "
              f"{mean['env_corr']:>9.3f}")

    if len(scales) >= 3:
        rho = _spearman(scales, [means[s]["mrstft"] for s in scales])
        env_drop = means[scales[0]]["env_corr"] - means[scales[-1]]["env_corr"]
        print(f"\n  dose-response (Spearman scale vs mrstft): {rho:+.2f} "
              f"{'[OK monotone]' if rho > 0.8 else '[WARN not clearly monotone]'}")
        print(f"  arrangement drift (env_corr scale {scales[0]:g} -> {scales[-1]:g}): "
              f"{env_drop:+.3f} {'[OK content preserved]' if abs(env_drop) < 0.1 else '[WARN content shifting]'}")
    else:
        print("\n  (need >= 3 distinct scales for the dose-response verdict)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
