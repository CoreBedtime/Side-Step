"""
Convert plain-text lyric sidecars into Side-Step Option-A TXT sidecars.

This is intended for folders where sidecars were created as `<audio>.wav.txt`
containing only lyrics/transcripts (no `key: value` fields).

For each audio file in a directory:
- Reads `<audio>.txt` if present
- If it looks like a plain blob, rewrites it into Side-Step Option-A format
  with a fixed caption and the blob as the `lyrics:` block
- Creates a backup before overwriting
"""

from __future__ import annotations

import argparse
from pathlib import Path


_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aiff", ".aif"}
_FIELD_ORDER = ("caption", "genre", "bpm", "key", "signature", "is_instrumental")


def _looks_like_option_a(text: str) -> bool:
    """Heuristic: treat as Side-Step Option-A if it starts with `caption:`."""
    return text.lstrip().lower().startswith("caption:")


def _write_option_a(
    sidecar_path: Path,
    *,
    caption: str,
    lyrics: str,
    is_instrumental: bool,
) -> None:
    data = {
        "caption": caption,
        "genre": "",
        "bpm": "",
        "key": "",
        "signature": "",
        "is_instrumental": "true" if is_instrumental else "false",
        "lyrics": lyrics,
    }

    lines: list[str] = []
    for k in _FIELD_ORDER:
        lines.append(f"{k}: {data.get(k, '')}")

    # Lyrics block always last.
    lyr = data.get("lyrics", "")
    lines.append(f"lyrics:\n{lyr}" if lyr else "lyrics:")
    content = "\n".join(lines).rstrip("\n") + "\n"
    sidecar_path.write_text(content, encoding="utf-8")


def convert_directory(directory: Path, *, caption: str, overwrite: bool) -> dict[str, int]:
    """Convert sidecars in *directory*; returns stats."""
    directory = directory.resolve()
    audio_files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _AUDIO_EXTS)

    stats = {"audio_found": len(audio_files), "converted": 0, "skipped": 0, "missing_sidecar": 0}

    for audio_path in audio_files:
        sidecar_path = audio_path.with_suffix(".txt")
        if not sidecar_path.is_file():
            stats["missing_sidecar"] += 1
            continue

        text = sidecar_path.read_text(encoding="utf-8", errors="replace")

        if _looks_like_option_a(text) and not overwrite:
            stats["skipped"] += 1
            continue

        # Backup then overwrite.
        backup_path = sidecar_path.with_suffix(".txt.plain.bak")
        if not backup_path.exists():
            backup_path.write_text(text, encoding="utf-8")

        lyrics_blob = text.strip("\ufeff").strip()
        _write_option_a(
            sidecar_path,
            caption=caption,
            lyrics=lyrics_blob,
            is_instrumental=False,
        )
        stats["converted"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="Directory containing audio + sidecar .txt files")
    parser.add_argument(
        "--caption",
        default="acapella, dry vocals",
        help="Caption to write for all sidecars",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite even if sidecar already looks like Option-A",
    )
    args = parser.parse_args()

    stats = convert_directory(args.directory, caption=args.caption, overwrite=args.overwrite)
    print(
        "Converted sidecars.",
        f"audio_found={stats['audio_found']}",
        f"converted={stats['converted']}",
        f"skipped={stats['skipped']}",
        f"missing_sidecar={stats['missing_sidecar']}",
        sep="\n",
    )


if __name__ == "__main__":
    main()
