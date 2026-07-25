# Eval Harness — Design (banked 2026-07, pre-implementation)

Answers one question: **"is this LoRA getting better, and when should I stop?"**

## The four measurements

1. **Style capture** — embedding similarity between generated clips and the
   training-set centroid. Should rise, then plateau.
2. **Prompt adherence** — text–audio similarity between generated clips and
   their captions. Should hold; a slide = adapter is eating the conditioning.
3. **Memorization alarm** — max similarity between a generated clip and any
   *single* training clip. Spike toward one song = copying, not style.
4. **Ears** — WAVs playable in the Monitor. Metrics decide *when* to listen.

Stop signal: style climbs, adherence holds, then adherence slides — the knee
is the checkpoint. Harness *labels* the knee (advisory badge); never acts.

## Principles

- **Deterministic by construction.** Fixed eval set chosen at run start:
  K samples' cached conditioning tensors (no text encoders needed), fixed
  noise seeds, fixed step count, fixed clip duration (~10–20 s ⇒ stable
  shapes). Written to `eval_set.json`. Same dataset ⇒ same manifest ⇒
  cross-run A/B comparability.
- **Base-model anchor.** Generate the eval set once with adapters disabled
  (PEFT toggle); report all metrics as deltas vs. that reference.
- **Cheap / 8 GB-safe.** Generation reuses the training DiT in VRAM
  (no_grad, batch 1). VAE loads transiently for decode, freed after (CPU
  fallback). Scoring runs on CPU in a background thread while training
  resumes. Budget: seconds-to-a-minute per eval; cadence `eval_every` epochs
  + at best-saves + run end.
- **Non-interfering.** Runs between epochs in the training process, under
  forked fixed-seed RNG, EMA apply/restore around generation, decoder back
  to train mode, VRAM released. Training bit-identical with eval on/off.
- **Bounded artifacts.** WAVs + `metrics.jsonl` per eval in the run dir;
  keep last N + best, prune older WAVs. Progress-writer `kind="eval"`
  events; TensorBoard `eval/*` scalars.

## Scorers (plugin-shaped)

- **CLAP** (`transformers.ClapModel`, LAION music checkpoint, ~190 M params,
  zero new deps). CPU: ~1–3 s per 10 s clip. Covers ALL three metrics in v1;
  the only option for prompt adherence (joint text–audio space).
- **MERT** (m-a-p/MERT-v1-95M, optional extra `side-step[eval-mert]`, needs
  nnAudio; 24 kHz resample). Upgrades style + memorization precision
  (music-specific SSL features, per-frame). No text tower — cannot do
  adherence. Pin pooled layer range in the manifest for comparability.
- Division of labor: CLAP = adherence (+fallback for all), MERT = style +
  memorization when installed. Later plugin slots: FAD, aesthetics.
- **Scorers are keyed by LoRA PURPOSE, not one-size-fits-all.** CLAP/MERT
  are semantic embeddings trained to be INVARIANT to low-level texture —
  near-blind to texture/corruption LoRAs. Purpose `texture` uses
  reference-based metrics instead (paired covers make a ground truth exist:
  mrstft distance, HF/flatness deltas, latent-rate modulation energy,
  envelope-correlation for arrangement invariance, dose-response
  monotonicity over adapter scale). Prototyped in
  `scripts/eval_texture_pairs.py` + `scripts/eval_texture_discrimination.py`
  (2x2 adapter-on/off x artifact/clean loss differential). Also add an
  **external-judge hook** (user command: WAV paths in, scores out) so power
  users can plug in their own downstream models as metrics.
- Cheap bonus metric: librosa BPM of generated audio vs. caption BPM.
- One-time dataset embedding pass cached beside the tensors.

## Components

EvalSetBuilder (seeded selection; held-out split if val_split>0, else train
samples; captions from .pt metadata) → Sampler (minimal flow-matching
inference: Euler over variant shift schedule, CFG w/ null_condition_emb —
**mine the official code**: `checkpoints/acestep-v15-*/modeling_acestep_v15_*.py`
+ `apg_guidance.py`, VAE in `checkpoints/vae/`) → Decoder (lazy VAE) →
Scorer (CPU thread) → Reporter (jsonl + progress events + TB + GUI strip).

## Non-goals v1

No training on metrics (Goodhart), no auto-stop, no FAD, no lyric
intelligibility, no separate eval process.

## Phasing

1. **Generation plumbing**: sampler + VAE decode + WAV artifacts + Monitor
   audio players. Start with the sampler — highest-risk piece; verify one
   WAV by ear against real inference before building anything else.
2. **CLAP scoring**: three curves + base anchor + dataset-embedding cache.
3. **Advisory**: knee badge, History cross-run compare, BPM check.
   2.5: MERT plugin.

Schema fields: `eval_every`, `eval_clips`, `eval_duration`, `eval_steps`,
`eval_metrics` (one schema edit each, post-refactor).
