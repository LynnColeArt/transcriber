#!/usr/bin/env bash
set -euo pipefail

if ! command -v pipx >/dev/null 2>&1; then
  echo "pipx is required. On Ubuntu: sudo apt install pipx"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "$SCRIPT_DIR/../diarizer" && pwd)"

pipx install --force "$PACKAGE_DIR"

cat <<'EOF'
The local diarization helper is installed.

Before the first run:
  1. Visit https://huggingface.co/pyannote/speaker-diarization-community-1
     and accept the model's access conditions.
  2. Create a Hugging Face read token.
  3. Export it for the initial model download:
       export HF_TOKEN="your-token"

Then run:
  transcriber meeting.wav meeting.txt

After the model is cached, inference runs locally. To force offline mode:
  export HF_HUB_OFFLINE=1
EOF
