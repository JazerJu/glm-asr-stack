#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/glm-asr-stack-$(date +%Y%m%d-%H%M%S).tar.gz}"
base="$(basename "$ROOT")"
parent="$(dirname "$ROOT")"
# Models are intentionally excluded; download or mount them at deploy time.
tar -C "$parent" -czf "$OUT" \
  --exclude="$base/.git" \
  --exclude="$base/**/*.pyc" \
  --exclude="$base/**/__pycache__" \
  --exclude="$base/*.tar.gz" \
  --exclude="$base/runtime/work_dir/*" \
  --exclude="$base/runtime/output/*" \
  --exclude="$base/runtime/tasks/*" \
  --exclude="$base/runtime/live_sessions/*" \
  --exclude="$base/models/glm-asr-bf16" \
  --exclude="$base/models/qwen3-forcealigner-0.6b" \
  "$base"
echo "$OUT"
ls -lh "$OUT"
