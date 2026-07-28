#!/usr/bin/env bash
set -euo pipefail

if ! command -v pipx >/dev/null 2>&1; then
  echo "pipx is required. On Ubuntu: sudo apt install pipx"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${TRANSCRIBER_DIARIZER_PACKAGE:-}" ]]; then
  PACKAGE_DIR="$(cd -- "$TRANSCRIBER_DIARIZER_PACKAGE" && pwd)"
elif [[ -f "$SCRIPT_DIR/diarizer/pyproject.toml" ]]; then
  PACKAGE_DIR="$SCRIPT_DIR/diarizer"
else
  PACKAGE_DIR="$(cd -- "$SCRIPT_DIR/../diarizer" && pwd)"
fi

pipx install --force "$PACKAGE_DIR"
PIPX_BIN_DIR="$(pipx environment --value PIPX_BIN_DIR)"
DIARIZER_COMMAND="$PIPX_BIN_DIR/transcriber-diarize"

cat <<'EOF'
Community-1 requires one-time Hugging Face access:
  1. Visit https://huggingface.co/pyannote/speaker-diarization-community-1
     and accept the model's access conditions.
  2. Log in with `hf auth login`, or provide a read token when prompted.

For non-interactive setup, export the token first:
       export HF_TOKEN="your-token"
EOF

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_TOKEN:-}" && -t 0 ]]; then
  read -r -s -p "Hugging Face read token (press Enter to use cached login): " token
  echo
  if [[ -n "$token" ]]; then
    export HF_TOKEN="$token"
  fi
fi

"$DIARIZER_COMMAND" --download-only

cat <<'EOF'
The local diarization helper and Community-1 model are ready.

Then run:
  transcriber meeting.wav meeting.txt

After the model is cached, inference runs locally. To force offline mode:
  export HF_HUB_OFFLINE=1
EOF
