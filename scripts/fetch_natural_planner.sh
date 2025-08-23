#!/usr/bin/env bash
# Clone google-deepmind/natural-plan into
# benchmarks/agentic_planner/eval/ and apply your local patches.
# Re-run safe / idempotent.
#
# Usage:
#   bash scripts/fetch_natural_planner.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_URL="https://github.com/google-deepmind/natural-plan.git"
PATCH_DIR="$ROOT/benchmarks/agentic_planner/PATCHES"
PIN_FILE="$PATCH_DIR/UPSTREAM_COMMIT.txt"
DEST="$ROOT/benchmarks/agentic_planner/eval"   

if [[ -f "$PIN_FILE" && -s "$PIN_FILE" ]]; then
  PIN_COMMIT="$(tr -d ' \t\r\n' < "$PIN_FILE")"
else
  echo "[warn] $PIN_FILE missing or empty; defaulting to 'main'"
  PIN_COMMIT="main"
fi

# Ensure DEST exists and is suitable
mkdir -p "$DEST"
if [[ -d "$DEST/.git" ]]; then
  echo "[info] existing git checkout found at $DEST; fetching updates"
  git -C "$DEST" remote set-url origin "$REPO_URL" || true
  git -C "$DEST" fetch --all --tags --prune
elif [[ -z "$(ls -A "$DEST")" ]]; then
  echo "[clone] $REPO_URL -> $DEST"
  git clone "$REPO_URL" "$DEST"
else
  echo "[error] $DEST exists and is not empty, but not a git repo."
  echo "        Move/clean it or remove it, then re-run."
  exit 2
fi

echo "[checkout] $PIN_COMMIT"
git -C "$DEST" checkout --quiet "$PIN_COMMIT"

# Apply local patches (if any)
shopt -s nullglob
PATCHES=("$PATCH_DIR"/*.patch)
if (( ${#PATCHES[@]} )); then
  echo "[patch] applying ${#PATCHES[@]} patch(es)"
  for p in "${PATCHES[@]}"; do
    echo "  - $(basename "$p")"
    # --forward: skip if already applied
    (cd "$DEST" && patch -p1 --forward --batch < "$p") || {
      echo "[error] failed to apply $(basename "$p")"; exit 1;
    }
  done
else
  echo "[patch] no patches found in $PATCH_DIR"
fi

SHORT_SHA="$(git -C "$DEST" rev-parse --short HEAD)"
echo "[ok] natural-plan ready at $DEST (commit: $SHORT_SHA)"

# Helpful echo lines for how to run
echo
echo "Run examples:"
echo "  PYTHONPATH=\"$DEST\":\$PYTHONPATH \\"
echo "    python \"$DEST/evaluate_meeting_planning.py\" \\"
echo "      --data_path \"$DEST/data/meeting_planning.json\""
echo "  python \"$DEST/evaluate_trip_planning.py\" \\"
echo "      --data_path \"$DEST/data/trip_planning.json\""
echo "  python \"$DEST/evaluate_calendar_scheduling.py\" \\"
echo "      --data_path \"$DEST/data/calendar_scheduling.json\""
