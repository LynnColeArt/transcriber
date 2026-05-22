#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${WHISPERCPP_DIR:-$HOME/.local/share/whisper.cpp}"
MODEL="${WHISPER_MODEL_NAME:-large-v3}"
JOBS="${JOBS:-$(nproc)}"

mkdir -p "$(dirname "$INSTALL_DIR")"

if [[ ! -d "$INSTALL_DIR/.git" ]]; then
  git clone https://github.com/ggerganov/whisper.cpp "$INSTALL_DIR"
fi

cmake -S "$INSTALL_DIR" -B "$INSTALL_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$INSTALL_DIR/build" -j "$JOBS"

"$INSTALL_DIR/models/download-ggml-model.sh" "$MODEL"

BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
ln -sf "$INSTALL_DIR/build/bin/whisper-cli" "$BIN_DIR/whisper-cli"

cat <<EOF
whisper.cpp is ready.

Binary:
  $BIN_DIR/whisper-cli

Model:
  $INSTALL_DIR/models/ggml-$MODEL.bin

Add this to your shell if needed:
  export PATH="\$HOME/.local/bin:\$PATH"
  export WHISPER_MODEL="$INSTALL_DIR/models/ggml-$MODEL.bin"
EOF
