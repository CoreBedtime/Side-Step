"""Eval-clip generation: cached conditioning -> latents -> WAV artifacts.

No text encoders are ever loaded: conditioning comes from the preprocessed
``.pt`` tensors (the same ones training consumes).  The VAE is loaded
lazily for decode and should be freed by the caller when done.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from sidestep_engine.eval.sampler import sample_latents

logger = logging.getLogger(__name__)

__all__ = [
    "load_conditioning",
    "build_conditioning_from_prompt",
    "generate_clip",
    "decode_to_wav",
]


def build_conditioning_from_prompt(
    model,
    checkpoint_dir: str | Path,
    *,
    prompt: str,
    lyrics: str = "[Instrumental]",
    frames: int,
    variant: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Build fresh conditioning from a text prompt (no dataset needed).

    Loads the Qwen3 text encoder transiently, encodes prompt + lyrics,
    runs the DiT encoder (``model.encoder`` must be on *device*), and
    builds the standard text2music silence context.  Use this when the
    dataset's stored captions are absent or junk (filename-fallback
    sidecars) — unconditioned or filename-conditioned generation at high
    guidance produces noise, not music.
    """
    from sidestep_engine.data.preprocess_prompt import build_simple_prompt
    from sidestep_engine.models.loader import load_silence_latent, load_text_encoder
    from sidestep_engine.vendor.preprocess_encoder import run_encoder
    from sidestep_engine.vendor.preprocess_lyrics import encode_lyrics
    from sidestep_engine.vendor.preprocess_text import encode_text

    # Conditioning parity with real inference (verified against acestep.cpp
    # tensor dumps at cosine >= 0.999):
    # - caption goes through the SFT_GEN_PROMPT template (# Instruction /
    #   # Caption / # Metas) — the model never sees bare captions
    # - lyrics are TEMPLATED (# Languages / # Lyric ... <|endoftext|>) and
    #   embedded via the raw token table (encode_lyrics does embed_tokens)
    # - timbre falls back to a single SILENCE-LATENT frame, not zeros
    formatted = build_simple_prompt({
        "caption": prompt,
        "duration": round(frames / 25.0, 1),
    })
    language = "unknown"
    lyrics_formatted = f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"

    silence = load_silence_latent(
        checkpoint_dir, device="cpu", precision="fp32", variant=variant,
    )

    tokenizer, text_encoder = load_text_encoder(
        checkpoint_dir, device=str(device), precision="bf16",
    )
    try:
        text_hs, text_mask = encode_text(text_encoder, tokenizer, formatted, device, dtype)
        lyric_hs, lyric_mask = encode_lyrics(text_encoder, tokenizer, lyrics_formatted, device, dtype)
        ehs, eam = run_encoder(
            model, text_hs, text_mask, lyric_hs, lyric_mask, device, dtype,
            refer_audio_hidden_states_packed=silence[:, :1, :].to(device=device, dtype=dtype),
        )
    finally:
        text_encoder.to("cpu")
        del text_encoder
        if device.type == "cuda":
            torch.cuda.empty_cache().float()
    if silence.shape[1] >= frames:
        src = silence[:, :frames, :]
    else:
        reps = -(-frames // silence.shape[1])
        src = silence.repeat(1, reps, 1)[:, :frames, :]
    context = torch.cat([src, torch.ones(1, frames, 64)], dim=-1)

    return {
        "encoder_hidden_states": ehs.cpu(),
        "encoder_attention_mask": eam.cpu(),
        "context_latents": context,
        "attention_mask": torch.ones(1, frames),
        "caption": prompt,
        "name": "prompt",
    }


def load_conditioning(pt_path: str | Path, max_frames: Optional[int] = None) -> Dict[str, Any]:
    """Load one preprocessed sample's conditioning, optionally time-cropped.

    Returns a dict ready for :func:`sample_latents` (batch dim added),
    plus ``caption`` / ``name`` metadata for reporting.
    """
    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    ctx = data["context_latents"]
    am = data["attention_mask"]
    if max_frames is not None and ctx.shape[0] > max_frames:
        ctx = ctx[:max_frames]
        am = am[:max_frames]
    meta = data.get("metadata") or {}
    return {
        "encoder_hidden_states": data["encoder_hidden_states"].unsqueeze(0),
        "encoder_attention_mask": data["encoder_attention_mask"].unsqueeze(0),
        "context_latents": ctx.unsqueeze(0),
        "attention_mask": am.unsqueeze(0),
        "caption": meta.get("caption", ""),
        "name": Path(pt_path).stem,
    }


def generate_clip(
    model,
    cond: Dict[str, Any],
    *,
    infer_steps: int,
    shift: float,
    guidance_scale: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample one latent clip ``[T, 64]`` from a conditioning dict."""
    return sample_latents(
        model,
        {k: cond[k] for k in (
            "encoder_hidden_states", "encoder_attention_mask",
            "context_latents", "attention_mask",
        )},
        infer_steps=infer_steps,
        shift=shift,
        guidance_scale=guidance_scale,
        seed=seed,
        device=device,
        dtype=dtype,
    )


def decode_to_wav(latents: torch.Tensor, vae, out_path: str | Path) -> Path:
    """Decode ``[T, 64]`` latents to audio and write a WAV file.

    Inverse of preprocessing's ``tiled_vae_encode`` convention: stored
    latents are ``[T, 64]``; Oobleck wants ``[B, 64, T]``.
    """
    import soundfile as sf

    vae_device = next(vae.parameters()).device
    z = latents.unsqueeze(0).transpose(1, 2).to(vae_device, dtype=vae.dtype)
    with torch.inference_mode():
        audio = vae.decode(z).sample  # [B, C, S]
    wav = audio[0].float().cpu().clamp(-1.0, 1.0).transpose(0, 1).numpy()  # [S, C]
    sr = int(getattr(vae.config, "sampling_rate", 48000))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, sr)
    logger.info("[eval] wrote %s (%.1fs @ %d Hz)", out_path, wav.shape[0] / sr, sr)
    return out_path
