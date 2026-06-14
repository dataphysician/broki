from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import wave
from typing import Callable

from brainrot_guard.models import MediaItem, MediaKind


SUPPORTED: dict[str, tuple[MediaKind, str]] = {
    ".mp4": (MediaKind.VIDEO, "video/mp4"),
    ".mov": (MediaKind.VIDEO, "video/quicktime"),
    ".webm": (MediaKind.VIDEO, "video/webm"),
    ".mkv": (MediaKind.VIDEO, "video/x-matroska"),
    ".mp3": (MediaKind.AUDIO, "audio/mpeg"),
    ".wav": (MediaKind.AUDIO, "audio/wav"),
    ".flac": (MediaKind.AUDIO, "audio/flac"),
    ".png": (MediaKind.IMAGE, "image/png"),
    ".jpg": (MediaKind.IMAGE, "image/jpeg"),
    ".jpeg": (MediaKind.IMAGE, "image/jpeg"),
    ".webp": (MediaKind.IMAGE, "image/webp"),
    ".txt": (MediaKind.TEXT, "text/plain"),
    ".md": (MediaKind.TEXT, "text/markdown"),
}


def scan_media_folder(media_dir: Path) -> list[MediaItem]:
    root = media_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"media folder does not exist: {root}")
    items: list[MediaItem] = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        kind_mime = SUPPORTED.get(path.suffix.lower())
        if kind_mime is None or _looks_like_url_file(path):
            continue
        kind, mime = kind_mime
        items.append(MediaItem(id=_media_id(path), path=path, kind=kind, mime_type=mime, duration_ms=_duration_ms(path, kind)))
    return items


def _media_id(path: Path) -> str:
    stat = path.stat()
    return sha256(f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[:16]


def _looks_like_url_file(path: Path) -> bool:
    if path.suffix.lower() not in {".txt", ".md", ".url"}:
        return False
    sample = path.read_text(encoding="utf-8", errors="ignore")[:512].lower()
    return "youtube.com" in sample or "youtu.be" in sample


def _duration_ms(
    path: Path,
    kind: MediaKind,
    *,
    probe_runner: Callable[..., object] = subprocess.run,
) -> int | None:
    if kind in {MediaKind.IMAGE, MediaKind.TEXT}:
        return 3000
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav:
                rate = wav.getframerate()
                return round((wav.getnframes() / rate) * 1000) if rate > 0 else None
        except (OSError, EOFError, wave.Error):
            pass
    if kind in {MediaKind.VIDEO, MediaKind.AUDIO}:
        return _ffprobe_duration_ms(path, probe_runner=probe_runner)
    return None


def _ffprobe_duration_ms(path: Path, *, probe_runner: Callable[..., object]) -> int | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = probe_runner(cmd, capture_output=True, text=True, check=False)
    except (FileNotFoundError, OSError):
        return None
    if getattr(result, "returncode", 1) != 0:
        return None
    try:
        payload = json.loads(getattr(result, "stdout", "") or "{}")
        seconds = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if seconds <= 0:
        return None
    return round(seconds * 1000)
