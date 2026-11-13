# Stress tests

These files test how OMNICAST behaves on **difficult audio**. They were made from clean validation/dev clips by
adding effects with ffmpeg (noise, speed, volume, echo, phone filter, music-like tones) or by joining clips.
They are clearly artificial: use them to explore robustness, not to report accuracy.

| File | What it tests | What to expect |
|---|---|---|
| `01_background_noise_moderate.wav` | Moderate background noise | Should still work; small ASR errors possible |
| `02_background_noise_heavy.wav` | Heavy background noise | ASR may mishear words, and the translation follows those mistakes |
| `03_very_quiet.wav` | Very quiet recording (-30 dB) | Usually still transcribed; may lose words |
| `04_too_loud_distorted.wav` | Very loud, distorted input | Check whether ASR still understands it |
| `05_fast_speech_1.5x.wav` | Fast speech (1.5x) | Harder for ASR; dubbed audio may be much longer than the source (watch D2) |
| `06_slow_speech_0.7x.wav` | Slow speech (0.7x) | Should work; dubbed audio may be shorter than the source |
| `07_phone_quality.wav` | Telephone quality (8 kHz) | Accepted and converted to 16 kHz; slightly lower ASR quality |
| `08_background_music.wav` | Speech over a constant chord | Should work; tones may confuse ASR slightly |
| `09_echo_room.wav` | Echo / reverberant room | Small ASR errors possible |
| `10_long_silences_between_sentences.wav` | Three sentences with 4 s silences | Segmenter should cut at the silences (about 3 segments) |
| `11_two_speakers.flac` | Two different speakers | Both parts dubbed (in one synthetic voice) |
| `12_long_form_8min.flac` | About 8 minutes, one speaker | Many segments; takes several minutes (longer on Windows) |
| `13_low_quality_mp3_32kbps.mp3` | Heavily compressed MP3 | Accepted; slight quality loss |

Files 01–09 and 13 contain the same sentence (FLEURS validation 00084), so you can compare how each effect changes
the transcript and translation against the clean version (`demo_set/valid/05_long_sentence_13s.wav`).
Reference English (and Hindi/French where available) for every file is in `samples/test_sets.csv`.
