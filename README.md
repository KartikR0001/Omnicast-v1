# OMNICAST

A local AI studio that dubs long English audio into **Hindi** or **French**, checks every segment, and repairs only the bad ones.

- **Module-1 (dubbing pipeline):** upload check → speech-to-text (Whisper large-v3-turbo, MLX) → translation (NLLB-200 600M) → text-to-speech (MMS-TTS) → rebuilt audio + subtitles (WAV + SRT)
- **Module-2 (quality and repair):** per-segment checks D1–D4 (round-trip CER with a separate MMS-CTC model, duration ratio, clipping/silence, empty translation) → rule-based repair (re-generate speech / re-translate / human review, max 2 attempts) → a repair is kept only if it is better, otherwise rolled back
- **Interfaces:** web review page (FastAPI) and a command-line tool

## Requirements

- Apple Silicon Mac (M1 or later): the speech-to-text model uses MLX
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- ffmpeg on your PATH (`brew install ffmpeg`)
- About 10 GB of free disk space for the models (downloaded automatically on the first run)

## Setup

```bash
git clone <this-repo-url>
cd omnicast-app
uv sync --group ml
```

## Run

**Web review page**
```bash
uv run uvicorn app.main:app --port 8000
```
Open http://localhost:8000, upload a WAV/MP3/FLAC/M4A file (max 200 MB, 60 min) and choose Hindi or French.

**Command line**
```bash
uv run python cli.py run path/to/audio.wav --tgt hi        # or --tgt fr
uv run python cli.py show <project_id>
uv run python cli.py regenerate <project_id> 0 --text "new translation"
```

Results are saved in `storage/` (created automatically, not tracked by git).

After the first run has downloaded the models, add `HF_HUB_OFFLINE=1` in front of the commands so the app does not contact huggingface.co, for example:
```bash
HF_HUB_OFFLINE=1 uv run uvicorn app.main:app --port 8000
```

## Configuration

- `config/models.yaml`: model IDs and the TTS provider (`mms_tts` by default)
- `config/quality.yaml`: detector thresholds and the maximum number of repair attempts

The optional Chatterbox TTS (`--tts chatterbox`) needs its own Python environment at `.tools/envs/chatterbox`, which is not included here. The default MMS-TTS needs no extra setup.

## Demo

See **DEMO.md** for a step-by-step demo script and `samples/` for ready-to-use test clips.

## Licence note

The NLLB-200, MMS-TTS and MMS-1B-all model weights are released under CC-BY-NC-4.0 (non-commercial). This is an academic project.
