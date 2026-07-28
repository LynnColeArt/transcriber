#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

missing=()
for command_name in git cmake ffmpeg pipx cc c++; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    missing+=("$command_name")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "Missing setup prerequisites: ${missing[*]}"
  echo "On Ubuntu, install them with:"
  echo "  sudo apt install build-essential cmake ffmpeg git pipx"
  exit 1
fi

echo "[1/3] Setting up Whisper large-v3 and Silero VAD..."
"$SCRIPT_DIR/setup-whispercpp.sh"

echo "[2/3] Setting up Demucs and its vocal-separation model..."
"$SCRIPT_DIR/setup-demucs.sh"

echo "[3/3] Setting up pyannote Community-1 speaker diarization..."
"$SCRIPT_DIR/setup-diarization.sh"

echo
echo "All transcription models are ready."
echo "Run 'transcriber doctor' to inspect the installation."
