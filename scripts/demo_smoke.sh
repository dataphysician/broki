#!/usr/bin/env bash
set -euo pipefail

workdir="$(mktemp -d /tmp/brainrot-guard-smoke-XXXXXX)"
mkdir -p "$workdir/media"
printf 'Counting by twos\n2 4 6 8\n' > "$workdir/media/counting.txt"
printf '\x89PNG\r\n\x1a\n' > "$workdir/media/frame.png"
printf 'ID3' > "$workdir/media/sound.mp3"
printf 'demo video' > "$workdir/media/clip.mp4"

PYTHONPATH=backend python -m brainrot_guard demo \
  --media-dir "$workdir/media" \
  --db-path "$workdir/state.sqlite3" \
  --artifacts-dir "$workdir/artifacts" \
  --duration-ms 3000
