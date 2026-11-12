# OMNICAST demo dataset

A small, ready-made set of English audio for demonstrating OMNICAST to the project guide and evaluators.
All audio comes from the **validation / dev-clean** splits of public datasets (never test data).

## Valid files (`valid/`): should be accepted and dubbed

| # | File | Length | What it shows |
|---|---|---|---|
| 01 | `01_short_4s.wav` | 4 s | Quickest end-to-end run (one segment) |
| 02 | `02_short_7s.wav` | 7 s | Standard demo clip; compare with the reference Hindi/French in `demo_set.csv` |
| 03 | `03_numbers_9s.wav` | 9 s | Numbers in speech ("70 runs in 41 balls") |
| 04 | `04_names_5s.wav` | 5 s | Proper names (George W. Bush) |
| 05 | `05_long_sentence_13s.wav` | 13 s | Longer, complex sentence |
| 06 | `06_story_25s.flac` | 25 s | Story excerpt, a few segments |
| 07 | `07_speech_52s.flac` | 52 s | Different speaker, several segments |
| 08 | `08_story_65s_repair_case.flac` | 65 s | In our French dev run, one segment failed the D1 check and was repaired automatically |
| 09 | `09_long_113s.flac` | 113 s | Long-form segmentation over about 2 minutes (takes a few minutes) |
| 10 | `10_same_as_02.mp3` | 7 s | MP3 upload works (same audio as 02) |
| 11 | `11_same_as_03.m4a` | 9 s | M4A (AAC) upload works (same audio as 03) |
| 12 | `12_stereo_44khz.wav` | 7 s | Stereo 44.1 kHz input is converted to 16 kHz mono automatically |

## Invalid files (`invalid/`): should be rejected with a clear reason

| File | What it is | Expected result |
|---|---|---|
| `fake_audio.wav` | Text file renamed to .wav | INVALID_SIGNATURE (HTTP 415) |
| `empty.wav` | 0-byte file | EMPTY_FILE (HTTP 400) |
| `corrupt.mp3` | MP3 header with no audio frames | CORRUPT_MEDIA (HTTP 400) |
| `unsupported_format.ogg` | Real audio in an unsupported format | UNSUPPORTED_FORMAT (HTTP 415) |

These results were checked by running the project's upload validator (`pipeline/ingest.py`, `validate_media()`) on every file.

## `demo_set.csv`

One row per file: format, duration, speaker, source, licence, what to show, and the **reference English transcript**.
For the FLEURS clips (01–05, 10–12) it also has the **reference Hindi and French translations**, so you can compare them with OMNICAST's output.

## Suggested demo order (about 10 minutes)

1. `invalid/fake_audio.wav`, then `invalid/unsupported_format.ogg`: the upload checks.
2. `valid/02_short_7s.wav` → Hindi: full pipeline; open the review page; compare with the reference Hindi.
3. Edit one translation and click **Regenerate Segment**: a new version (v2) is created and v1 is kept.
4. `valid/10_same_as_02.mp3` → French: other format, other language.
5. *(If time)* `valid/08_story_65s_repair_case.flac` → French: a multi-segment job with the repair log.

## Sources and licences

- FLEURS (Conneau et al., 2023), en_us validation split: CC BY 4.0
- LibriSpeech-Long dev-clean (derived from LibriSpeech, Panayotov et al., 2015): CC BY 4.0
