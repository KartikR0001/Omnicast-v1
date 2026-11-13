# OMNICAST

A local AI studio that dubs long English audio into **Hindi** or **French**, checks every segment, and repairs only the bad ones. Runs on macOS and Windows.

- **Module-1 (dubbing pipeline):** upload check → speech-to-text (Whisper: MLX on Apple Silicon, faster-whisper on Windows/Linux) → translation (NLLB-200 600M) → text-to-speech (MMS-TTS) → rebuilt audio + subtitles (WAV + SRT)
- **Module-2 (quality and repair):** per-segment checks D1–D4 (round-trip CER with a separate MMS-CTC model, duration ratio, clipping/silence, empty translation) → rule-based repair (re-generate speech / re-translate / human review, max 2 attempts) → a repair is kept only if it is better, otherwise rolled back
- **Interfaces:** web review page (FastAPI) and a command-line tool

## Requirements

Works on **macOS (Apple Silicon)** and **Windows 10/11 (64-bit)**. Linux works too.

- Python 3.12 (installed automatically by [uv](https://docs.astral.sh/uv/))
- ffmpeg
- 16 GB RAM recommended, about 10 GB free disk space for the models (downloaded on the first run)

Speech-to-text engine (chosen automatically):
- **Apple Silicon Mac:** Whisper large-v3-turbo via MLX (fast, uses the Apple GPU)
- **Windows / Linux / Intel Mac:** Whisper "small" via faster-whisper (CPU; uses an NVIDIA GPU if CUDA is set up). You can pick a bigger model in `config/models.yaml` (`fallback_repo: medium`) for better accuracy at lower speed.

## Setup and run — macOS (Terminal)

```bash
xcode-select --install                       # gives git (skip if installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
# open a new Terminal, then:
git clone https://github.com/KartikR0001/Omnicast-v1.git
cd Omnicast-v1
uv sync --group ml
uv run python cli.py run samples/short_ocean_4s.wav --tgt hi          # first run downloads the models
HF_HUB_OFFLINE=1 uv run uvicorn app.main:app --port 8000
```

## Setup and run — Windows (PowerShell)

```powershell
winget install --id Git.Git -e
winget install --id Gyan.FFmpeg -e
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# close PowerShell and open a new one, then:
git clone https://github.com/KartikR0001/Omnicast-v1.git
cd Omnicast-v1
uv sync --group ml
uv run python cli.py run samples\short_ocean_4s.wav --tgt hi          # first run downloads the models
$env:HF_HUB_OFFLINE=1; uv run uvicorn app.main:app --port 8000
```

Open http://localhost:8000, upload a WAV/MP3/FLAC/M4A file (max 200 MB, 60 min), choose Hindi or French and **MMS-TTS**, then click **Validate & Start Pipeline Job**.

**Command line** (`\` instead of `/` in paths on Windows):
```bash
uv run python cli.py run path/to/audio.wav --tgt hi        # or --tgt fr
uv run python cli.py show <project_id>
uv run python cli.py regenerate <project_id> 0 --text "new translation"
```

Results are saved in `storage/` (created automatically, not tracked by git). `HF_HUB_OFFLINE=1` stops the app from contacting huggingface.co once the models are downloaded.

## Configuration

- `config/models.yaml`: model IDs and the TTS provider (`mms_tts` by default)
- `config/quality.yaml`: detector thresholds and the maximum number of repair attempts

The optional Chatterbox TTS (`--tts chatterbox`) needs its own Python environment at `.tools/envs/chatterbox`, which is not included here. The default MMS-TTS needs no extra setup.

## Demo

See **DEMO.md** for a step-by-step demo script and `samples/` for ready-to-use test clips.

## Licence note

The NLLB-200, MMS-TTS and MMS-1B-all model weights are released under CC-BY-NC-4.0 (non-commercial). This is an academic project.
