"""Eval harness: checkpoint-time sample generation and scoring.

Phase 1 (this package): deterministic latent generation from cached
conditioning + VAE decode to WAV artifacts.  Design doc:
``sidestep_documentation/Eval Harness Design.md``.

Inference semantics are mined from the official checkpoint code
(``checkpoints/acestep-v15-*/modeling_acestep_v15_*.py``); see
``sampler.py`` for the faithful subset and its documented deviations.
"""
