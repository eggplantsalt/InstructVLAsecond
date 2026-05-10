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

is_prepared_complete() {
  local ep_dir="$1"
  local num_frames="$2"

  [ -f "$ep_dir/episode_meta.json" ] || return 1
  [ -f "$ep_dir/actions.npy" ] || return 1
  [ -f "$ep_dir/states.npy" ] || return 1
  [ -f "$ep_dir/instruction.txt" ] || return 1
  [ -d "$ep_dir/frames" ] || return 1

  local n
  n=$(find "$ep_dir/frames" -maxdepth 1 -type f -name "frame_*.png" | wc -l)
  [ "$n" -eq "$num_frames" ] || return 1

  return 0
}


echo "Episodes: ${EPISODES[*]}"
echo "OUT_ROOT: $OUT_ROOT"
echo "NUM_FRAMES: $NUM_FRAMES"
echo "SAMPLE_STRATEGY: $SAMPLE_STRATEGY"
echo "FORCE: $FORCE"

for EP in "${EPISODES[@]}"; do
  EP_PAD=$(printf "%06d" "$EP")
  EP_DIR="$OUT_ROOT/episode_${EP_PAD}"
  LOG="$LOG_DIR/prepare_episode_${EP_PAD}.log"

  if [ "$FORCE" != "1" ] && is_prepared_complete "$EP_DIR" "$NUM_FRAMES"; then
    echo "[SKIP] episode $EP already prepared and complete: $EP_DIR"
    continue
  fi

  if [ "$FORCE" != "1" ] && [ -e "$EP_DIR" ]; then
    echo "[WARN] episode $EP prepared dir exists but is incomplete; removing and rebuilding: $EP_DIR"
    rm -rf "$EP_DIR"
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
