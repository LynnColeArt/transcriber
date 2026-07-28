# transcriber

`transcriber` is a local Go command for turning common audio and video files into speaker-labeled text, subtitle, or JSON transcripts:

```sh
transcriber <file in> <file out>
```

It is designed to be called from anywhere on your system. Media decoding is handled by `ffmpeg`, vocal stem isolation by Demucs, transcription by `whisper.cpp` with Whisper large-v3, and speaker detection by pyannote Community-1.

## Features

- Accepts common media containers and codecs supported by `ffmpeg`, including MP3, MP4, M4A, MOV, WAV, FLAC, AAC, and OGG.
- Isolates vocal stems with Demucs by default when it is installed, reducing background music before transcription and diarization.
- Offers explicit Demucs vocal isolation and ffmpeg speech cleanup for difficult recordings.
- Can use Silero VAD for recordings with substantial non-speech regions.
- Supports vocabulary prompts and configurable rolling text context.
- Detects and labels an unknown number of speakers locally with pyannote Community-1 by default.
- Writes speaker-labeled plain text by default, with explicit JSON, SRT, and VTT output modes.
- Runs locally with no hosted transcription API.
- Provides `transcriber setup` for one-command model installation and `transcriber doctor` for dependency checks.

## Pipeline

1. Decode the selected audio stream from the input file with `ffmpeg`.
2. Preprocess the decoded audio:
   - `auto`: use Demucs vocal isolation when available, otherwise warn and normalize only. This is the default.
   - `demucs`: require Demucs vocal stem separation.
   - `denoise`: use ffmpeg speech cleanup filters.
   - `none`: normalize only; retained as an explicit alias.
3. Normalize to mono 16 kHz WAV for Whisper.
4. Transcribe with `whisper.cpp` using `large-v3` and reset rolling text context by default.
5. Infer the number of speakers, run local speaker diarization, and align speaker intervals to Whisper token timestamps.
6. Write speaker-labeled plain text by default, or the explicitly requested structured/subtitle format.

## Quick start

Install the system prerequisites. On Ubuntu:

```sh
sudo apt install build-essential cmake ffmpeg git pipx
```

Accept the access conditions for [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and create a read token. The setup command prompts for it securely when needed. For unattended setup, export it first:

```sh
export HF_TOKEN="your-token"
```

Build the CLI and run the model installer:

```sh
make install
export PATH="$HOME/.local/bin:$PATH"
transcriber setup
```

`transcriber setup` installs or refreshes the local helpers and prepares all default models:

- Whisper large-v3
- Silero VAD 6.2
- Demucs `htdemucs`
- pyannote Community-1

The command is safe to rerun and reuses cached downloads. Expect several gigabytes of model and Python dependency data. When `nvcc` is available, whisper.cpp is built with CUDA automatically; set `WHISPER_ACCELERATOR=cpu` or `WHISPER_ACCELERATOR=cuda` to override detection.

Verify everything and transcribe:

```sh
transcriber doctor
transcriber recording.mp3 recording.txt
```

`make setup` is a shortcut for `make install` followed by `transcriber setup`.

### Manual setup

The individual installers remain available for troubleshooting or partial installations:

```sh
scripts/setup-whispercpp.sh
scripts/setup-demucs.sh
scripts/setup-diarization.sh
```

After the models are cached, inference runs locally. Set `HF_HUB_OFFLINE=1` to prohibit later network access. The diarization helper disables pyannote inference telemetry by default.

## Usage

```sh
transcriber setup
transcriber recording.mp3 recording.txt
transcriber -format srt call.mp4 call.srt
transcriber -format json podcast.wav podcast.json
transcriber -format vtt -language en lecture.m4a lecture.vtt
transcriber -prompt "SOX9, Enhancer 13" lecture.m4a lecture.txt
transcriber -vad meeting.wav meeting.txt
transcriber -diarize=false solo-narration.wav narration.txt
```

Check dependencies:

```sh
transcriber doctor
```

Useful flags:

```text
-preprocess auto|demucs|denoise|none
-format text|json|srt|vtt|auto
-model /path/to/ggml-large-v3.bin
-whisper-bin /path/to/whisper-cli
-language auto|CODE
-prompt "names, acronyms, specialized terms"
-max-context 0
-audio-stream 0
-vad
-vad-model /path/to/ggml-silero-v6.2.0.bin
-diarize=false
-diarizer /path/to/transcriber-diarize
-diarization-model pyannote/speaker-diarization-community-1
-diarization-device auto|cpu|cuda
-num-speakers 2
-min-speakers 2
-max-speakers 6
-threads 8
-keep-temp
-verbose
```

Environment overrides:

```text
WHISPER_MODEL
WHISPER_BIN
WHISPER_LANGUAGE
WHISPER_VAD_MODEL
FFMPEG_BIN
DEMUCS_BIN
DEMUCS_MODEL
TRANSCRIBER_PREPROCESS
TRANSCRIBER_FORMAT
TRANSCRIBER_SETUP
DIARIZER_BIN
DIARIZATION_MODEL
DIARIZATION_DEVICE
HF_TOKEN
```

## Notes

Whisper large-v3 gives strong transcription quality, but it can be slow on CPU. For faster drafts, use a smaller or quantized `whisper.cpp` model and pass it with `-model`.

The default `-max-context 0` prevents prior decoded text from contaminating later windows. If a recording benefits from conversational continuity, try `-max-context 64` or restore whisper.cpp's full rolling context with `-max-context -1`. Measure this against representative recordings: context behavior is audio-dependent.

VAD is intentionally opt-in. It is most helpful for meetings, surveillance audio, and other recordings with long silent regions; dense edited speech may not benefit.

Demucs provides actual vocal stem separation and is selected by the default `auto` mode when installed. Use `-preprocess none` to skip it for pristine speech-only audio. The ffmpeg denoise mode is useful for simpler noise cleanup, but it is not equivalent to separating vocals from music.

Diarization runs by default and infers the speaker count without a hint. `-num-speakers`, `-min-speakers`, and `-max-speakers` are optional constraints for unusual recordings, not required inputs. Use `-diarize=false` to skip speaker detection for a faster single-speaker transcription.

Diarization answers “who spoke when” with anonymous labels such as `SPEAKER_00`. It does not infer a person's real name. Overlapping speakers can be marked by the diarizer, but a single mixed audio track may still prevent Whisper from transcribing both voices correctly.

With `-format json`, JSON contains the complete Whisper data plus top-level `diarization` metadata, per-token `speaker` fields, and per-segment `speaker_turns`. Text and subtitle formats prefix each turn with its speaker label. `-format auto` preserves the older extension-driven behavior when desired.
