#!/usr/bin/env bash
# collect_target_proofs.sh — one-shot operator script that runs the four
# environment-bound readiness proofs on the target GPU machine and saves
# the JSON outputs to .sisyphus/evidence/.
#
# Run with --dry-run to print the commands without executing them.
# Run --help for usage.
set -euo pipefail

DB_PATH="/tmp/broki-target.sqlite3"
MEDIA_DIR=""
ARTIFACTS_DIR="/tmp/broki-artifacts"
EVIDENCE_DIR=".sisyphus/evidence"
FSAVERAGE5_DIR="${BROKI_FSAVERAGE5_DIR:-}"
TRIBE_CKPT="${BROKI_TRIBE_CKPT:-}"
VLM_PROVIDER="gemini"
DRY_RUN=0

usage() {
  cat <<'EOF'
collect_target_proofs.sh — capture BROKI target-machine readiness proofs

Usage:
  bash backend/scripts/collect_target_proofs.sh [options]

Options:
  --db-path PATH         SQLite DB for the seeded caregiver calibration (default: /tmp/broki-target.sqlite3)
  --media-dir PATH       Local media folder (required)
  --artifacts-dir PATH   Where analyze/browser writes artifacts (default: /tmp/broki-artifacts)
  --evidence-dir PATH    Where proof JSONs are written (default: .sisyphus/evidence)
  --fsaverage5-dir PATH  fsaverage5 directory (defaults to $BROKI_FSAVERAGE5_DIR)
  --tribe-ckpt PATH      TRIBE v2 checkpoint (defaults to $BROKI_TRIBE_CKPT)
  --vlm-provider NAME    gemini, xai, or minicpm (default: gemini)
  --dry-run              Print commands with [DRY-RUN] prefix; do not execute
  --help                 Show this help and exit

Environment:
  Operator must export the API key for the selected --vlm-provider
  (e.g. GEMINI_API_KEY, XAI_API_KEY, or MINICPM_*) before running.

The script writes the following JSON files to --evidence-dir:
  hardware.json, tribe-plotbrain.json, vlm-probe.json, analysis.json, browser.json
The final validate-readiness invocation is printed but NOT executed.
EOF
}

# ---------- arg parse ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --db-path)        DB_PATH="$2"; shift 2 ;;
    --media-dir)      MEDIA_DIR="$2"; shift 2 ;;
    --artifacts-dir)  ARTIFACTS_DIR="$2"; shift 2 ;;
    --evidence-dir)   EVIDENCE_DIR="$2"; shift 2 ;;
    --fsaverage5-dir) FSAVERAGE5_DIR="$2"; shift 2 ;;
    --tribe-ckpt)     TRIBE_CKPT="$2"; shift 2 ;;
    --vlm-provider)   VLM_PROVIDER="$2"; shift 2 ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --help|-h)        usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MEDIA_DIR" ]]; then
  echo "error: --media-dir is required" >&2
  usage
  exit 2
fi

_run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] $*"
  else
    eval "$@"
  fi
}

_echo() {
  echo "$@"
}

_echo "== collect_target_proofs =="
_echo "DB_PATH=$DB_PATH"
_echo "MEDIA_DIR=$MEDIA_DIR"
_echo "ARTIFACTS_DIR=$ARTIFACTS_DIR"
_echo "EVIDENCE_DIR=$EVIDENCE_DIR"
_echo "FSAVERAGE5_DIR=${FSAVERAGE5_DIR:-<unset>}"
_echo "TRIBE_CKPT=${TRIBE_CKPT:-<unset>}"
_echo "VLM_PROVIDER=$VLM_PROVIDER"
_echo ""

_echo "[1/7] mkdir evidence dir"
_run mkdir -p "$EVIDENCE_DIR"

_echo ""
_echo "[2/7] seed caregiver calibration"
_run env PYTHONPATH=backend python -m brainrot_guard seed-calibration --db-path "$DB_PATH" --min-per-label 3

_echo ""
_echo "[3/7] hardware proof (validate-hardware --force)"
_run env PYTHONPATH=backend python -m brainrot_guard validate-hardware --force \| tee "$EVIDENCE_DIR/hardware.json"

_echo ""
_echo "[4/7] TRIBE / PlotBrain smoke-render proof (validate-live --force --smoke-render)"
_run env BRAINROT_GUARD_LIVE_VALIDATE=1 BRAINROT_GUARD_PLOTBRAIN_SMOKE=1 BRAINROT_GUARD_FSAVERAGE5_DIR=\""$FSAVERAGE5_DIR"\" BRAINROT_GUARD_TRIBE_CKPT=\""$TRIBE_CKPT"\" PYTHONPATH=backend python -m brainrot_guard validate-live --force --smoke-render \| tee "$EVIDENCE_DIR/tribe-plotbrain.json"

_echo ""
_echo "[5/7] credentialed VLM probe (validate-vlm-live --force --probe)"
_echo "    operator must export $VLM_PROVIDER's API key env var (e.g. GEMINI_API_KEY)"
_run env VLM_PROVIDER=\""$VLM_PROVIDER"\" BRAINROT_GUARD_VLM_LIVE_VALIDATE=1 BRAINROT_GUARD_VLM_PROBE=1 PYTHONPATH=backend python -m brainrot_guard validate-vlm-live --force --probe \| tee "$EVIDENCE_DIR/vlm-probe.json"

_echo ""
_echo "[6/7] representative analysis proof (validate-analysis)"
_run env BRAINROT_GUARD_ENABLE_LIVE_ANALYSIS=1 PYTHONPATH=backend python -m brainrot_guard validate-analysis --media-dir "$MEDIA_DIR" --db-path "$DB_PATH" --artifacts-dir "$ARTIFACTS_DIR" --limit 1 \| tee "$EVIDENCE_DIR/analysis.json"

_echo ""
echo "[7/7] browser visual proof (validate-browser)"
_run env PYTHONPATH=backend python -m brainrot_guard validate-browser --db-path "$DB_PATH" --media-dir "$MEDIA_DIR" --artifacts-dir "$ARTIFACTS_DIR" --screenshot-path /tmp/broki-review.png \| tee "$EVIDENCE_DIR/browser.json"

_echo ""
_echo "== final readiness report (do not auto-execute) =="
_echo "Once the five proof JSONs are saved, run:"
_echo "  PYTHONPATH=backend python -m brainrot_guard validate-readiness \\"
_echo "    --db-path \"$DB_PATH\" \\"
_echo "    --media-dir \"$MEDIA_DIR\" \\"
_echo "    --artifacts-dir \"$ARTIFACTS_DIR\" \\"
_echo "    --evidence-dir \"$EVIDENCE_DIR\" \\"
_echo "    --force-live --smoke-render --force-vlm --probe-vlm --force-hardware"
