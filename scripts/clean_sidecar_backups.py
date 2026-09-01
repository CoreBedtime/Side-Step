#!/usr/bin/env python3
"""Remove accumulated sidecar backup files from a dataset directory.

Side-Step's ``write_sidecar`` copies the previous sidecar to ``.txt.bak``
on every write and nothing ever removes it, so a directory collects one
backup per sidecar per edit session.  Several one-off repair scripts left
their own variants behind as well (``.plain.bak``, ``.before_clean.bak``,
``.bak2``).

Dry-run by default -- it prints what it would delete and changes nothing.
Pass ``--delete`` to actually remove the files.

    python scripts/clean_sidecar_backups.py "D:/Shared/Training Data/Music"
    python scripts/clean_sidecar_backups.py "D:/..." --delete

Only files matching a known backup suffix are ever considered; ``.txt``
sidecars and audio files are never touched.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

#: Suffixes written by write_sidecar and by the one-off repair scripts.
#: Ordered longest-first so `.txt.plain.bak` is reported under its own
#: label rather than being swallowed by `.bak`.
BACKUP_SUFFIXES = (
    ".txt.before_clean.bak",
    ".txt.plain.bak",
    ".txt.bak2",
    ".txt.bak",
)

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def classify(path: Path) -> str | None:
    """Return the backup suffix *path* ends with, or None if it is not one."""
    name = path.name.lower()
    for suffix in BACKUP_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return None


def find_backups(root: Path) -> list[tuple[Path, str, int]]:
    """Collect (path, suffix, size) for every backup file under *root*."""
    found: list[tuple[Path, str, int]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = classify(path)
        if suffix is None:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        found.append((path, suffix, size))
    return found


def looks_like_dataset(root: Path) -> bool:
    """True if *root* contains audio somewhere -- a guard against typos.

    Deleting by glob in the wrong directory is the failure mode worth
    preventing, so refuse to operate on a tree with no audio in it.
    """
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            return True
    return False


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove sidecar backup files (.txt.bak and friends).",
    )
    parser.add_argument("root", help="Dataset directory to clean (searched recursively)")
    parser.add_argument(
        "--delete", action="store_true",
        help="Actually delete. Without this the script only reports.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip the 'directory contains audio' safety check.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"[FAIL] Not a directory: {root}", file=sys.stderr)
        return 1

    if not args.force and not looks_like_dataset(root):
        print(
            f"[FAIL] No audio files found under {root}.\n"
            "       This does not look like a dataset directory. "
            "Re-run with --force if you are sure.",
            file=sys.stderr,
        )
        return 1

    backups = find_backups(root)
    if not backups:
        print(f"[OK] No sidecar backups found under {root}")
        return 0

    by_suffix: Counter[str] = Counter()
    bytes_by_suffix: Counter[str] = Counter()
    for _, suffix, size in backups:
        by_suffix[suffix] += 1
        bytes_by_suffix[suffix] += size

    total_bytes = sum(bytes_by_suffix.values())
    verb = "Deleting" if args.delete else "Would delete"

    print(f"\n{verb} sidecar backups under {root}\n")
    print(f"  {'suffix':<26} {'files':>7}  {'size':>10}")
    print("  " + "-" * 46)
    for suffix, count in by_suffix.most_common():
        print(f"  {suffix:<26} {count:>7}  {human(bytes_by_suffix[suffix]):>10}")
    print("  " + "-" * 46)
    print(f"  {'total':<26} {len(backups):>7}  {human(total_bytes):>10}\n")

    if not args.delete:
        print("Dry run -- nothing was changed. Re-run with --delete to remove them.")
        return 0

    removed = 0
    failed = 0
    for path, _, _ in backups:
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            failed += 1
            print(f"  [WARN] {path}: {exc}", file=sys.stderr)

    print(f"[OK] Removed {removed} file(s), {human(total_bytes)} freed"
          + (f", {failed} failed" if failed else ""))
    if failed:
        return 1

    print(
        "\nNote: write_sidecar still creates a .txt.bak on every write, so "
        "these will accumulate again as you edit sidecars."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
