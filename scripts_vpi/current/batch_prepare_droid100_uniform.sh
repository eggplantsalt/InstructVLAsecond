#!/usr/bin/env bash
set -euo pipefail

cd /storage/v-xiangxizheng/zy_workspace/InstructVLA

EPISODES=("$@")
if [ ${#EPISODES[@]} -eq 0 ]; then
  EPISODES=(0 1 2 3 4 6)
fi

OUT_ROOT="outputs_vpi/formal_droid100/uniform_debug"
LOG_DIR="outputs_vpi/logs/formal/batch_logs"
NUM_FRAMES="${NUM_FRAMES:-30}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-uniform}"
FORCE="${FORCE:-0}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

echo "Episodes: ${EPISODES[*]}"
echo "OUT_ROOT: $OUT_ROOT"
echo "NUM_FRAMES: $NUM_FRAMES"
echo "SAMPLE_STRATEGY: $SAMPLE_STRATEGY"
echo "FORCE: $FORCE"

for EP in "${EPISODES[@]}"; do
  EP_PAD=$(printf "%06d" "$EP")
  EP_DIR="$OUT_ROOT/episode_${EP_PAD}"
  LOG="$LOG_DIR/prepare_episode_${EP_PAD}.log"

  if [ "$FORCE" != "1" ] && [ -f "$EP_DIR/episode_meta.json" ]; then
    echo "[SKIP] episode $EP already prepared: $EP_DIR"
    continue
  fi

  echo
  echo "=== Prepare episode $EP ==="

  python scripts_vpi/current/prepare_droid100_episode.py \
    --episode "$EP" \
    --num_frames "$NUM_FRAMES" \
    --sample_strategy "$SAMPLE_STRATEGY" \
    --out_root "$OUT_ROOT" \
    2>&1 | tee "$LOG"
done

echo
echo "[DONE] Prepared episodes: ${EPISODES[*]}"
