#!/usr/bin/env bash
set -euo pipefail

cd /storage/v-xiangxizheng/zy_workspace/InstructVLA

EPISODES=("$@")
if [ ${#EPISODES[@]} -eq 0 ]; then
  EPISODES=(0 1 2 3 4 6)
fi

EP_ROOT="outputs_vpi/formal_droid100/uniform_debug"
OUT_ROOT="outputs_vpi/formal_droid100/visual_text_suite"
LOG_DIR="outputs_vpi/logs/formal/batch_logs"
NUM_FRAMES="${NUM_FRAMES:-30}"
FORCE="${FORCE:-0}"

RANDOM_TEXTS=(
  "XQJ 729 BLUE TRIANGLE"
  "ALPHA 42 GREEN STAR"
  "ZETA MONKEY 913"
)

mkdir -p "$OUT_ROOT" "$LOG_DIR"

is_eval_complete() {
  local out_dir="$1"
  local num_frames="$2"

  [ -f "$out_dir/summary.json" ] || return 1
  [ -f "$out_dir/main_conditions/summary.json" ] || return 1
  [ -f "$out_dir/main_conditions/condition_summary.csv" ] || return 1
  [ -f "$out_dir/extra_wrong_random/summary.json" ] || return 1
  [ -f "$out_dir/extra_wrong_random/condition_summary.csv" ] || return 1

  local dirs=(
    "smallBottom_correctText"
    "smallBottom_wrongText_00"
    "smallBottom_wrongText_01"
    "smallBottom_wrongText_02"
    "smallBottom_wrongText_03"
    "smallBottom_wrongText_04"
    "smallBottom_randomText_00"
    "smallBottom_randomText_01"
    "smallBottom_randomText_02"
  )

  local d n
  for d in "${dirs[@]}"; do
    [ -d "$out_dir/images/$d" ] || return 1
    n=$(find "$out_dir/images/$d" -maxdepth 1 -type f -name "frame_*.png" | wc -l)
    [ "$n" -eq "$num_frames" ] || return 1
  done

  return 0
}


echo "Episodes: ${EPISODES[*]}"
echo "EP_ROOT: $EP_ROOT"
echo "OUT_ROOT: $OUT_ROOT"
echo "NUM_FRAMES: $NUM_FRAMES"
echo "FORCE: $FORCE"

for EP in "${EPISODES[@]}"; do
  EP_PAD=$(printf "%06d" "$EP")
  EP_DIR="$EP_ROOT/episode_${EP_PAD}"
  OUT_DIR="$OUT_ROOT/episode_${EP_PAD}"
  LOG="$LOG_DIR/visual_text_suite_episode_${EP_PAD}.log"

  if [ ! -f "$EP_DIR/episode_meta.json" ]; then
    echo "[ERROR] Missing prepared episode: $EP_DIR"
    echo "Run: bash scripts_vpi/current/batch_prepare_droid100_uniform.sh $EP"
    exit 1
  fi

  if [ "$FORCE" != "1" ] && is_eval_complete "$OUT_DIR" "$NUM_FRAMES"; then
    echo "[SKIP] episode $EP already evaluated and complete: $OUT_DIR"
    continue
  fi

  if [ "$FORCE" != "1" ] && [ -e "$OUT_DIR" ]; then
    echo "[WARN] episode $EP output exists but is incomplete; removing and rerunning: $OUT_DIR"
    rm -rf "$OUT_DIR"
  fi

  case "$EP" in
    0)
      WRONG_LANGUAGE="Move the cup to the sink"
      WRONG_TEXTS=(
        "Move the cup to the sink"
        "Put the marker on the table"
        "Pick up the cup"
        "Open the drawer"
        "Put the sponge in the bowl"
      )
      ;;
    1)
      WRONG_LANGUAGE="Put the candy bar on the right side of the first shelf"
      WRONG_TEXTS=(
        "Put the candy bar on the right side of the first shelf"
        "Put the candy bar on the table"
        "Place the candy bar inside the sink"
        "Pick up the marker"
        "Open the drawer"
      )
      ;;
    2)
      WRONG_LANGUAGE="Put one green sachet on the table"
      WRONG_TEXTS=(
        "Put one green sachet on the table"
        "Put one green sachet inside the sink"
        "Pick up the cup"
        "Open the drawer"
        "Put the marker in the pot"
      )
      ;;
    3)
      WRONG_LANGUAGE="Place the pack of doritos on the table"
      WRONG_TEXTS=(
        "Place the pack of doritos on the table"
        "Put the pack of doritos on the shelf"
        "Pick up the cup"
        "Open the drawer"
        "Put the marker in the pot"
      )
      ;;
    4)
      WRONG_LANGUAGE="Move the sharpie into the bowl"
      WRONG_TEXTS=(
        "Move the sharpie into the bowl"
        "Move the sharpie to the sink"
        "Pick up the cup"
        "Open the drawer"
        "Put the candy bar on the shelf"
      )
      ;;
    6)
      WRONG_LANGUAGE="Put the pen back into the bowl"
      WRONG_TEXTS=(
        "Put the pen back into the bowl"
        "Place the pen inside the sink"
        "Pick up the cup"
        "Open the drawer"
        "Put the marker in the pot"
      )
      ;;
    *)
      WRONG_LANGUAGE="Open the drawer"
      WRONG_TEXTS=(
        "Open the drawer"
        "Pick up the cup"
        "Put the object on the table"
        "Move the object to the sink"
        "Put the marker in the pot"
      )
      ;;
  esac

  echo
  echo "=== Run visual text suite for episode $EP ==="
  echo "EP_DIR: $EP_DIR"
  echo "OUT_DIR: $OUT_DIR"
  echo "WRONG_LANGUAGE: $WRONG_LANGUAGE"
  printf "WRONG_TEXTS:\n"
  printf "  - %s\n" "${WRONG_TEXTS[@]}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts_vpi/current/run_droid_visual_text_suite.py \
    --episode_dir "$EP_DIR" \
    --out_dir "$OUT_DIR" \
    --num_frames "$NUM_FRAMES" \
    --wrong_language "$WRONG_LANGUAGE" \
    --wrong_texts "${WRONG_TEXTS[@]}" \
    --random_texts "${RANDOM_TEXTS[@]}" \
    2>&1 | tee "$LOG"
done

echo
echo "[DONE] Evaluated episodes: ${EPISODES[*]}"
