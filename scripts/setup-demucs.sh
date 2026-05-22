#!/usr/bin/env bash
set -euo pipefail

if ! command -v pipx >/dev/null 2>&1; then
  echo "pipx is required. On Ubuntu: sudo apt install pipx"
  exit 1
fi

if ! command -v demucs >/dev/null 2>&1; then
  pipx install demucs
fi

# Demucs 4 with recent torchaudio may need torchcodec for writing separated WAVs.
pipx inject demucs torchcodec >/dev/null

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

ffmpeg -hide_banner -nostdin -y \
  -f lavfi -i sine=frequency=440:duration=1 \
  -ac 2 -ar 44100 "$tmpdir/prime.wav" >/dev/null 2>&1

demucs -n htdemucs --two-stems vocals -o "$tmpdir/out" "$tmpdir/prime.wav"

echo "Demucs and htdemucs are ready."
