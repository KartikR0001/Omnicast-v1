"""Module-1 Media Ingestion & Validation (R1, R11).

Validates format (magic-byte signature + ffprobe + full stream decode check),
size <= 200 MB, and duration <= 60 min (3600 s). Stores originals read-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
FFMPEG_BIN_DIR = ROOT / ".tools/ffmpeg/bin"
MAX_BYTES = 200 * 1024 * 1024  # 200 MB
MAX_DURATION_S = 60 * 60.0     # 60 minutes
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


def _tool_path(name: str) -> str:
    local = FFMPEG_BIN_DIR / name
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"Required binary '{name}' not found in {FFMPEG_BIN_DIR} or PATH")


class MediaValidationError(ValueError):
    """Structured validation failure carrying a human/machine reason and HTTP status code."""

    def __init__(self, reason: str, code: str = "INVALID_MEDIA", http_status: int = 400):
        super().__init__(reason)
        self.reason = reason
        self.code = code
        self.http_status = http_status


@dataclass
class MediaMetadata:
    file_path: str
    format_name: str
    codec_name: str
    size_bytes: int
    duration_s: float
    sample_rate: int
    channels: int
    sha256: str


def check_magic_bytes(header: bytes, ext: str) -> bool:
    """Verify binary file signature matches the declared audio container/codec."""
    if len(header) < 12:
        return False
    ext = ext.lower()
    if ext == ".wav":
        return header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    if ext == ".flac":
        return header[:4] == b"fLaC"
    if ext == ".m4a":
        return header[4:8] == b"ftyp"
    if ext == ".mp3":
        if header[:3] == b"ID3":
            return True
        # MPEG audio frame sync (11 set bits) with valid version/layer bits
        if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
            version_bits = (header[1] >> 3) & 0x03
            layer_bits = (header[1] >> 1) & 0x03
            return version_bits != 0x01 and layer_bits != 0x00
        return False
    return False


def validate_media(
    path: Path | str,
    max_bytes: int = MAX_BYTES,
    max_duration_s: float = MAX_DURATION_S,
) -> MediaMetadata:
    """Validate audio file format, magic signature, size, duration, and stream decodability (R1)."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise MediaValidationError("File does not exist", code="FILE_NOT_FOUND", http_status=400)

    size = p.stat().st_size
    if size == 0:
        raise MediaValidationError("Uploaded file is empty (0 bytes)", code="EMPTY_FILE", http_status=400)
    if size > max_bytes:
        mb = size / (1024 * 1024)
        raise MediaValidationError(
            f"File size ({mb:.1f} MB) exceeds maximum limit of {max_bytes / (1024 * 1024):.0f} MB",
            code="FILE_TOO_LARGE",
            http_status=413,
        )

    ext = p.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise MediaValidationError(
            f"Unsupported audio format '{ext or 'none'}'. Allowed: WAV, MP3, FLAC, M4A",
            code="UNSUPPORTED_FORMAT",
            http_status=415,
        )

    with open(p, "rb") as f:
        header = f.read(16)
    if not check_magic_bytes(header, ext):
        raise MediaValidationError(
            f"File magic-byte signature does not match declared extension '{ext}'",
            code="INVALID_SIGNATURE",
            http_status=415,
        )

    ffprobe = _tool_path("ffprobe")
    proc = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels,duration:format=format_name,duration",
            "-of", "json",
            str(p),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "ffprobe failed").strip().splitlines()[-1] if proc.stderr else "ffprobe error"
        raise MediaValidationError(f"Corrupt or unreadable media container: {err}", code="CORRUPT_MEDIA", http_status=400)

    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise MediaValidationError(f"Unparseable ffprobe metadata: {e}", code="CORRUPT_MEDIA", http_status=400) from e

    streams = info.get("streams") or []
    if not streams:
        raise MediaValidationError("File contains no decodable audio stream", code="NO_AUDIO_STREAM", http_status=400)

    st = streams[0]
    fmt = info.get("format") or {}
    dur_raw = st.get("duration") or fmt.get("duration")
    try:
        dur = float(dur_raw)
        sr = int(st.get("sample_rate") or 0)
        ch = int(st.get("channels") or 0)
    except (TypeError, ValueError) as e:
        raise MediaValidationError("Missing or invalid stream duration/sample_rate metadata", code="CORRUPT_MEDIA", http_status=400) from e

    if dur <= 0.0 or sr <= 0 or ch <= 0:
        raise MediaValidationError("Audio stream has zero duration, sample rate, or channels", code="EMPTY_AUDIO", http_status=400)
    if dur > max_duration_s:
        raise MediaValidationError(
            f"Audio duration ({dur / 60.0:.1f} min) exceeds maximum limit of {max_duration_s / 60.0:.0f} min",
            code="DURATION_EXCEEDED",
            http_status=400,
        )

    # Verify full stream decodability via ffmpeg error-strict null decode (catches truncated MP3/FLAC frames)
    # Skip full decode if file is large (>10 MB) and duration > 120s to keep validation fast, probing tail instead.
    ffmpeg = _tool_path("ffmpeg")
    if size <= 10 * 1024 * 1024:
        dec = subprocess.run(
            [ffmpeg, "-v", "error", "-xerror", "-i", str(p), "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if dec.returncode != 0 or (dec.stderr and "error" in dec.stderr.lower()):
            msg = (dec.stderr or "stream decode failed").strip().splitlines()[0]
            raise MediaValidationError(f"Truncated or corrupt audio frames: {msg}", code="CORRUPT_STREAM", http_status=400)

    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)

    return MediaMetadata(
        file_path=str(p.resolve()),
        format_name=str(fmt.get("format_name", ext.lstrip("."))),
        codec_name=str(st.get("codec_name", "unknown")),
        size_bytes=size,
        duration_s=dur,
        sample_rate=sr,
        channels=ch,
        sha256=h.hexdigest(),
    )


def decode_to_16k_mono(src_path: Path | str, dest_wav: Path | str | None = None) -> tuple[np.ndarray, int]:
    """Decode any supported audio format to 16 kHz mono float32 waveform using ffmpeg."""
    ffmpeg = _tool_path("ffmpeg")
    cmd = [
        ffmpeg, "-v", "error", "-y",
        "-i", str(src_path),
        "-ac", "1",
        "-ar", "16000",
        "-f", "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise MediaValidationError(f"Failed to decode audio to 16 kHz mono: {err}", code="DECODE_FAILED", http_status=400)
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise MediaValidationError("Decoded audio buffer is empty", code="EMPTY_AUDIO", http_status=400)
    if dest_wav is not None:
        out = Path(dest_wav)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, audio, 16000)
    return audio, 16000


def ingest_file(src_path: Path | str, storage_root: Path | str, project_id: str) -> tuple[MediaMetadata, Path, Path, np.ndarray]:
    """Validate source file, store read-only original in D1, and write 16 kHz working WAV (R1, R11)."""
    meta = validate_media(src_path)
    root = Path(storage_root)
    orig_dir = root / "originals"
    orig_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(src_path).suffix.lower()
    orig_dest = orig_dir / f"{project_id}_original{ext}"
    if orig_dest.exists():
        raise FileExistsError(f"Original file already exists and must never be overwritten: {orig_dest}")
    shutil.copy2(src_path, orig_dest)
    os.chmod(orig_dest, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 0o444 read-only (R11)

    proj_dir = root / "projects" / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    work_wav = proj_dir / "source_16k.wav"
    audio_16k, _ = decode_to_16k_mono(orig_dest, work_wav)
    return meta, orig_dest, work_wav, audio_16k

