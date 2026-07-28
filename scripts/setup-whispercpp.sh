#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${WHISPERCPP_DIR:-$HOME/.local/share/whisper.cpp}"
MODEL="${WHISPER_MODEL_NAME:-large-v3}"
JOBS="${JOBS:-$(nproc)}"
ACCELERATOR="${WHISPER_ACCELERATOR:-auto}"

mkdir -p "$(dirname "$INSTALL_DIR")"

if [[ ! -d "$INSTALL_DIR/.git" ]]; then
  git clone https://github.com/ggml-org/whisper.cpp "$INSTALL_DIR"
fi

cmake_args=(
  -S "$INSTALL_DIR"
  -B "$INSTALL_DIR/build"
  -DCMAKE_BUILD_TYPE=Release
)

case "$ACCELERATOR" in
  auto)
    if command -v nvcc >/dev/null 2>&1; then
      cmake_args+=(-DGGML_CUDA=ON)
      ACCELERATOR="cuda"
    else
      ACCELERATOR="cpu"
    fi
    ;;
  cuda)
    if ! command -v nvcc >/dev/null 2>&1; then
      echo "WHISPER_ACCELERATOR=cuda requested, but nvcc was not found."
      exit 1
    fi
    cmake_args+=(-DGGML_CUDA=ON)
    ;;
  cpu)
    cmake_args+=(-DGGML_CUDA=OFF)
    ;;
  *)
    echo "WHISPER_ACCELERATOR must be auto, cuda, or cpu."
    exit 1
    ;;
esac

cmake "${cmake_args[@]}"
cmake --build "$INSTALL_DIR/build" -j "$JOBS"

"$INSTALL_DIR/models/download-ggml-model.sh" "$MODEL"
"$INSTALL_DIR/models/download-vad-model.sh" silero-v6.2.0

BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
ln -sf "$INSTALL_DIR/build/bin/whisper-cli" "$BIN_DIR/whisper-cli"

cat <<EOF
whisper.cpp is ready.

Binary:
  $BIN_DIR/whisper-cli

Model:
  $INSTALL_DIR/models/ggml-$MODEL.bin

VAD model:
  $INSTALL_DIR/models/ggml-silero-v6.2.0.bin

Accelerator:
  $ACCELERATOR

Add this to your shell if needed:
  export PATH="\$HOME/.local/bin:\$PATH"
  export WHISPER_MODEL="$INSTALL_DIR/models/ggml-$MODEL.bin"
EOF
