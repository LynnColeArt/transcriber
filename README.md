# transcriber

`transcriber` is a local Go command for turning common audio and video files into text, subtitle, or JSON transcripts:

```sh
transcriber <file in> <file out>
```

It is designed to be called from anywhere on your system. Media decoding is handled by `ffmpeg`, optional vocal stem isolation is handled by Demucs, and transcription is handled by `whisper.cpp` with Whisper large-v3 by default.

## Features

- Accepts common media containers and codecs supported by `ffmpeg`, including MP3, MP4, M4A, MOV, WAV, FLAC, AAC, and OGG.
- Isolates vocals with Demucs `htdemucs` when available.
- Falls back to ffmpeg-based denoise and speech cleanup when Demucs is unavailable.
- Writes `.txt`, `.srt`, `.vtt`, or `.json` based on the output filename.
- Runs locally with no hosted transcription API.
- Provides `transcriber doctor` for dependency checks.

## Pipeline

1. Decode the first audio stream from the input file with `ffmpeg`.
2. Preprocess the decoded audio:
   - `auto`: use Demucs vocal isolation when installed, otherwise use ffmpeg denoise.
   - `demucs`: require Demucs vocal stem separation.
   - `denoise`: use ffmpeg speech cleanup filters.
   - `none`: normalize audio only.
3. Normalize to mono 16 kHz WAV for Whisper.
4. Transcribe with `whisper.cpp`.
5. Copy the generated transcript to the requested output path.

## Install

Build and install the Go command:

```sh
make install
```

This installs `transcriber` to `~/.local/bin`.

Make sure `~/.local/bin` is on your shell `PATH`:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

Install `ffmpeg` with your system package manager. On Ubuntu:

```sh
sudo apt install ffmpeg
```

Install `whisper.cpp` and download the default large-v3 model:

```sh
scripts/setup-whispercpp.sh
```

Install Demucs and prime the default stem-separation model:

```sh
scripts/setup-demucs.sh
```

## Usage

```sh
transcriber recording.mp3 recording.txt
transcriber call.mp4 call.srt
transcriber -preprocess demucs podcast.wav podcast.json
transcriber -language en lecture.m4a lecture.vtt
```

Check dependencies:

```sh
transcriber doctor
```

Useful flags:

```text
-preprocess auto|demucs|denoise|none
-model /path/to/ggml-large-v3.bin
-whisper-bin /path/to/whisper-cli
-threads 8
-keep-temp
-verbose
```

Environment overrides:

```text
WHISPER_MODEL
WHISPER_BIN
FFMPEG_BIN
DEMUCS_BIN
DEMUCS_MODEL
TRANSCRIBER_PREPROCESS
```

## Notes

Whisper large-v3 gives strong transcription quality, but it can be slow on CPU. For faster drafts, use a smaller or quantized `whisper.cpp` model and pass it with `-model`.

Demucs provides actual vocal stem separation. The ffmpeg denoise fallback is useful for simpler noise cleanup, but it is not equivalent to separating vocals from music.
