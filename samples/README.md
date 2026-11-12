# Demo samples

| File | What it is | Use in the demo |
|---|---|---|
| `short_ocean_4s.wav` | 4 s English sentence (FLEURS en_us validation, id 1531) | Quick end-to-end run |
| `short_satellite_7s.wav` | 7 s English sentence (FLEURS en_us validation, id 1548) | Second quick run / other language |
| `long_librispeech_65s.flac` | 65 s audiobook excerpt (LibriSpeech-Long dev-clean, 1272/128104/0001) | Longer run with several segments; in our dev runs one French segment was flagged and repaired automatically |
| `fake_audio.wav` | A text file renamed to `.wav` | Shows the upload check rejecting a fake file |

A larger demo dataset (12 valid clips in WAV/FLAC/MP3/M4A + 4 invalid files, with reference transcripts and translations) is in **`demo_set/`**; see `demo_set/README.md`.

Sources and licences: FLEURS (Conneau et al., 2023), CC BY 4.0; LibriSpeech-Long (from LibriSpeech, Panayotov et al., 2015), CC BY 4.0. Only validation/dev-clean clips are included, never test data.
