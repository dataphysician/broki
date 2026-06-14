#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DB_PATH=$(mktemp /tmp/broki-smoke-XXXXXX.sqlite3)
EVIDENCE_DIR=$(mktemp -d /tmp/broki-evidence-XXXXXX)
export PYTHONPATH=backend

echo "[smoke_proofs] DB_PATH=$DB_PATH"
echo "[smoke_proofs] EVIDENCE_DIR=$EVIDENCE_DIR"

echo "[smoke_proofs] step 1/6: seed caregiver calibration"
python -m brainrot_guard seed-calibration --db-path "$DB_PATH" --min-per-label 3

echo "[smoke_proofs] step 2/6: validate-calibration ready check"
CALIB_OUT=$(python -m brainrot_guard validate-calibration --db-path "$DB_PATH")
echo "$CALIB_OUT"
if ! echo "$CALIB_OUT" | grep -q '"ready": true'; then
  echo "[smoke_proofs] FAIL: validate-calibration did not report ready=true" >&2
  rm -rf "$DB_PATH" "$EVIDENCE_DIR"
  exit 1
fi

echo "[smoke_proofs] step 3/6: write stub analysis.json"
cat > "$EVIDENCE_DIR/analysis.json" <<'JSON'
{
  "ready": true,
  "media_count": 1,
  "analyzed_count": 1,
  "results": [
    {
      "status": "ready",
      "segment_count": 1,
      "npz_artifact_count": 1,
      "frame_artifact_count": 1,
      "frame_provenance_ready_count": 1,
      "scalar_heatmap_bar_count": 0,
      "artifact_integrity": "ready",
      "frame_manifest_status": "ready",
      "engagement_gate_crossed": true,
      "risk_gate_crossed": false,
      "warning_decision": "proceed"
    }
  ]
}
JSON

echo "[smoke_proofs] step 4/6: write stub browser.json"
cat > "$EVIDENCE_DIR/browser.json" <<'JSON'
{
  "ready": true,
  "browser_status": "ready",
  "visible_media_stage": true,
  "visible_brain_frame": true,
  "visible_timeline_playhead": true,
  "playhead_moved": true,
  "brain_frame_synced": true,
  "active_timeline_frame_synced": true,
  "visible_feedback_controls": true,
  "visible_skip_controls": true,
  "visible_auto_close_controls": true,
  "visible_proxy_label": true,
  "screenshot_bytes": 1,
  "missing_visible_checks": []
}
JSON

echo "[smoke_proofs] step 5/6: validate-readiness with stubbed proofs"
READINESS_OUT=$(python -m brainrot_guard validate-readiness \
  --db-path "$DB_PATH" \
  --evidence-dir "$EVIDENCE_DIR")
echo "$READINESS_OUT"

if ! echo "$READINESS_OUT" | grep -q '"caregiver_calibration_dataset".*"ready": true'; then
  echo "[smoke_proofs] FAIL: caregiver_calibration_dataset not ready" >&2
  rm -rf "$DB_PATH" "$EVIDENCE_DIR"
  exit 1
fi
if ! echo "$READINESS_OUT" | grep -q '"representative_live_analysis".*"ready": true'; then
  echo "[smoke_proofs] FAIL: representative_live_analysis not ready" >&2
  rm -rf "$DB_PATH" "$EVIDENCE_DIR"
  exit 1
fi
if ! echo "$READINESS_OUT" | grep -q '"browser_visual_proof".*"ready": true'; then
  echo "[smoke_proofs] FAIL: browser_visual_proof not ready" >&2
  rm -rf "$DB_PATH" "$EVIDENCE_DIR"
  exit 1
fi

echo "[smoke_proofs] step 6/6: cleanup"
rm -rf "$DB_PATH" "$EVIDENCE_DIR"

echo "[smoke_proofs] OK"
exit 0
