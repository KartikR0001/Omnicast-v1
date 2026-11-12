# OMNICAST — Demo guide for the Phase-1 review

Total demo time: about 10 minutes. Do steps 1–2 the day before.

## 1. Set up (once, the day before)

Needs an **Apple Silicon Mac** (M1 or later), Python 3.12, [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
brew install ffmpeg            # if not installed
cd omnicast-app
uv sync --group ml
```

## 2. Download the models before the review (important)

The first run downloads about 10 GB of models, which can take a long time on college Wi-Fi. Do it at home:

```bash
uv run python cli.py run samples/short_ocean_4s.wav --tgt hi
uv run python cli.py run samples/short_ocean_4s.wav --tgt fr
```

Both should end with status `COMPLETED`. After this, the app works **without internet**.

## 3. On the review day

Start the web app (offline mode, so nothing is downloaded during the demo):

```bash
cd omnicast-app
HF_HUB_OFFLINE=1 uv run uvicorn app.main:app --port 8000
```

Open http://localhost:8000. Keep this terminal visible: it shows each stage as the job runs.

## 4. Demo script

| # | Do this | Say this |
|---|---|---|
| 1 | Show the home page | "OMNICAST dubs English audio into Hindi or French on this laptop, with no cloud APIs. Module-1 does the dubbing, Module-2 checks and repairs each segment." |
| 2 | Choose `samples/fake_audio.wav` and click **Validate & Start Pipeline Job** | "First, the upload check: this is a text file renamed to .wav. It is rejected with a clear reason (invalid signature, HTTP 415)." |
| 3 | Choose `samples/short_satellite_7s.wav`, target **Hindi**, click **Validate & Start Pipeline Job** | "The upload returns at once (HTTP 202) and the job runs in the background: speech-to-text, translation, speech synthesis, quality checks, rebuild." |
| 4 | Click **Refresh List**, then **Open Studio →** on the finished project | "Each segment shows the English text, the Hindi text, both audio players, and the quality metrics: D1 error rate, D2 length ratio, D3 clipping and silence." |
| 5 | Play source and Hindi audio | "D1 transcribes our generated speech with a different model family (MMS-CTC), so the checker is not grading itself." |
| 6 | Edit the Hindi text of a segment, click **Save Text**, then **Regenerate Segment #0** | "A reviewer can correct a translation and regenerate just that segment. A new version (v2) is created; v1 is kept, so nothing is ever overwritten." |
| 7 | Show the **Module-2 Repair Audit Trail** under the segment | "Every repair attempt is logged: action, reason, and whether it was accepted or rolled back." |
| 8 | Click **Download Dubbed WAV** and **Download Aligned SRT** (top of the page) | "Finally the full dubbed audio and timed subtitles are exported." |
| 9 | *(optional, if time)* Upload `samples/long_librispeech_65s.flac`, target **French** | "A longer, multi-segment file. In our dev runs, one segment here failed the D1 check and was repaired automatically, then verified." It takes a few minutes, so start it early or skip it. |

Command-line alternative (if the browser has problems):
```bash
HF_HUB_OFFLINE=1 uv run python cli.py run samples/short_satellite_7s.wav --tgt hi
uv run python cli.py show <project_id printed above>
```

## 5. Questions the guide may ask

- **Why two ASR models?** Whisper does the transcription; MMS-CTC (a different family) checks the output, so the system does not grade itself.
- **What if a repair makes things worse?** It is rolled back: a new version is kept only if it has fewer problems and no worse error rate. After 2 failed attempts the segment goes to human review.
- **Why MMS-TTS and not voice cloning?** The cloning model we tried (Chatterbox) looped on French and was slow; voice cloning is planned for Phase-2.
- **Is anything sent to the cloud?** No. All models run locally; with `HF_HUB_OFFLINE=1` the app makes no network connections at all.

## 6. If something goes wrong

| Problem | Fix |
|---|---|
| `Required binary 'ffprobe' not found` | `brew install ffmpeg`, then open a new terminal |
| Job stuck or very slow on the first run | It is downloading models — run step 2 beforehand |
| Error about huggingface / offline | You started with `HF_HUB_OFFLINE=1` before the models were downloaded — run step 2 without it first |
| `Address already in use` | Use another port: `--port 8001` and open http://localhost:8001 |
| Not an Apple Silicon Mac | The speech-to-text model needs MLX (Apple Silicon); run the demo on an M-series Mac |
| Want a clean start | Stop the server and delete the `storage/` folder |
