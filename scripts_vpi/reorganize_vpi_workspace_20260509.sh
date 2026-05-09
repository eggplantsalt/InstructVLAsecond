#!/usr/bin/env bash
set -euo pipefail

cd /storage/v-xiangxizheng/zy_workspace/InstructVLA

echo "=== Create target directories ==="

mkdir -p scripts_vpi/current
mkdir -p scripts_vpi/archive_20260509/bridge_old
mkdir -p scripts_vpi/archive_20260509/droid_pilot_old
mkdir -p scripts_vpi/archive_20260509/debug_old
mkdir -p scripts_vpi/archive_20260509/backups
mkdir -p scripts_vpi/archive_20260509/pycache

mkdir -p outputs_vpi/formal_droid100
mkdir -p outputs_vpi/logs/formal
mkdir -p outputs_vpi/logs/archived_old
mkdir -p outputs_vpi/archive_pilot_20260509/bridge_old
mkdir -p outputs_vpi/archive_pilot_20260509/droid_pilot_old
mkdir -p outputs_vpi/archive_pilot_20260509/debug_old

echo
echo "=== Move current active scripts ==="

for f in \
  scripts_vpi/prepare_droid100_episode.py \
  scripts_vpi/run_droid_visual_text_suite.py \
  scripts_vpi/batch_prepare_droid100_uniform.sh \
  scripts_vpi/batch_run_droid_visual_text_suite.sh
do
  if [ -e "$f" ]; then
    echo "[MOVE current] $f -> scripts_vpi/current/"
    mv "$f" scripts_vpi/current/
  fi
done

echo
echo "=== Keep utility files in scripts_vpi root ==="
echo "[KEEP] scripts_vpi/__init__.py"
echo "[KEEP] scripts_vpi/instructvla_v100_utils.py"

echo
echo "=== Move script backups and pycache ==="

for f in scripts_vpi/*.bak_*; do
  if [ -e "$f" ]; then
    echo "[MOVE backup] $f -> scripts_vpi/archive_20260509/backups/"
    mv "$f" scripts_vpi/archive_20260509/backups/
  fi
done

if [ -d scripts_vpi/__pycache__ ]; then
  echo "[MOVE pycache] scripts_vpi/__pycache__ -> scripts_vpi/archive_20260509/pycache/"
  mv scripts_vpi/__pycache__ scripts_vpi/archive_20260509/pycache/__pycache__
fi

echo
echo "=== Move old Bridge scripts ==="

for f in \
  scripts_vpi/read_bridge_one_episode.py \
  scripts_vpi/run_clean_bridge_action.py \
  scripts_vpi/run_clean_bridge_stream.py \
  scripts_vpi/save_bridge_full_episode.py \
  scripts_vpi/run_clean_bridge_alignment_diag.py \
  scripts_vpi/run_clean_bridge_one_step_eval.py \
  scripts_vpi/run_overlay_same_language_eval.py \
  scripts_vpi/run_visual_centric_pair_eval.py \
  scripts_vpi/run_no_image_text_visual_prompt_control.py \
  scripts_vpi/run_wrong_language_default_control.py \
  scripts_vpi/run_reasoning_mode_pair_eval.py \
  scripts_vpi/make_visual_centric_frames.py \
  scripts_vpi/make_visual_centric_frames_bigtext.py \
  scripts_vpi/make_visual_centric_frames_bigtext_v2.py
do
  if [ -e "$f" ]; then
    echo "[MOVE bridge old] $f -> scripts_vpi/archive_20260509/bridge_old/"
    mv "$f" scripts_vpi/archive_20260509/bridge_old/
  fi
done

echo
echo "=== Move old DROID pilot scripts ==="

for f in \
  scripts_vpi/probe_droid100_lerobot.py \
  scripts_vpi/extract_droid100_lerobot_episode.py \
  scripts_vpi/run_droid_single_clean.py \
  scripts_vpi/run_droid_clean_stream.py \
  scripts_vpi/run_droid_all_conditions_pair_eval.py \
  scripts_vpi/run_droid_artifact_controls.py \
  scripts_vpi/run_droid_small_bottom_suite.py \
  scripts_vpi/run_droid_ep0_multi_wrong_random.py
do
  if [ -e "$f" ]; then
    echo "[MOVE droid pilot old] $f -> scripts_vpi/archive_20260509/droid_pilot_old/"
    mv "$f" scripts_vpi/archive_20260509/droid_pilot_old/
  fi
done

echo
echo "=== Move debug/smoke scripts ==="

for f in \
  scripts_vpi/test_instructvla_chat.py \
  scripts_vpi/test_instructvla_chat_manual.py \
  scripts_vpi/ask_image_instructvla.py \
  scripts_vpi/inspect_vlait_annotations.py \
  scripts_vpi/debug_preprocessed_visual_image.py \
  scripts_vpi/check_visual_centric_instruction_ignored.py \
  scripts_vpi/check_visual_centric_instruction_ignored_final.py \
  scripts_vpi/smoke_reasoning_predict_action.py
do
  if [ -e "$f" ]; then
    echo "[MOVE debug old] $f -> scripts_vpi/archive_20260509/debug_old/"
    mv "$f" scripts_vpi/archive_20260509/debug_old/
  fi
done

echo
echo "=== Move formal DROID outputs ==="

if [ -d outputs_vpi/droid100_uniform_debug ]; then
  echo "[MOVE formal] outputs_vpi/droid100_uniform_debug -> outputs_vpi/formal_droid100/uniform_debug"
  mv outputs_vpi/droid100_uniform_debug outputs_vpi/formal_droid100/uniform_debug
fi

if [ -d outputs_vpi/droid100_visual_text_suite ]; then
  echo "[MOVE formal] outputs_vpi/droid100_visual_text_suite -> outputs_vpi/formal_droid100/visual_text_suite"
  mv outputs_vpi/droid100_visual_text_suite outputs_vpi/formal_droid100/visual_text_suite
fi

echo
echo "=== Move formal logs ==="

for f in \
  outputs_vpi/prepare_droid100_uniform_episode_000000.log \
  outputs_vpi/prepare_droid100_uniform_episode_000001.log \
  outputs_vpi/droid100_visual_text_suite_episode_000000.log \
  outputs_vpi/droid100_visual_text_suite_episode_000001.log
do
  if [ -e "$f" ]; then
    echo "[MOVE formal log] $f -> outputs_vpi/logs/formal/"
    mv "$f" outputs_vpi/logs/formal/
  fi
done

if [ -d outputs_vpi/batch_logs ]; then
  echo "[MOVE formal batch logs] outputs_vpi/batch_logs -> outputs_vpi/logs/formal/batch_logs"
  mv outputs_vpi/batch_logs outputs_vpi/logs/formal/batch_logs
fi

echo
echo "=== Move old archived logs ==="

if [ -d outputs_vpi/_archived_logs_20260509 ]; then
  echo "[MOVE archived logs] outputs_vpi/_archived_logs_20260509 -> outputs_vpi/logs/archived_old/"
  mv outputs_vpi/_archived_logs_20260509 outputs_vpi/logs/archived_old/_archived_logs_20260509
fi

for f in outputs_vpi/*.log; do
  if [ -e "$f" ]; then
    echo "[MOVE remaining root log] $f -> outputs_vpi/logs/archived_old/"
    mv "$f" outputs_vpi/logs/archived_old/
  fi
done

echo
echo "=== Move old Bridge outputs ==="

for d in \
  outputs_vpi/bridge_debug \
  outputs_vpi/bridge_full_episode \
  outputs_vpi/bridge_full_episode_visual_centric \
  outputs_vpi/clean_action_debug \
  outputs_vpi/clean_alignment_diag \
  outputs_vpi/clean_one_step_eval \
  outputs_vpi/clean_stream_debug \
  outputs_vpi/overlay_same_language_eval \
  outputs_vpi/preprocess_debug \
  outputs_vpi/prompt_only_pair_eval \
  outputs_vpi/reasoning_mode_pair_eval \
  outputs_vpi/visual_centric_pair_eval \
  outputs_vpi/wrong_language_default_control
do
  if [ -d "$d" ]; then
    echo "[MOVE bridge output] $d -> outputs_vpi/archive_pilot_20260509/bridge_old/"
    mv "$d" outputs_vpi/archive_pilot_20260509/bridge_old/
  fi
done

echo
echo "=== Move old DROID pilot outputs ==="

for d in \
  outputs_vpi/droid100_all_conditions \
  outputs_vpi/droid100_artifact_controls \
  outputs_vpi/droid100_clean_single \
  outputs_vpi/droid100_clean_stream \
  outputs_vpi/droid100_debug \
  outputs_vpi/droid100_ep0_multi_wrong_random \
  outputs_vpi/droid100_small_bottom_suite
do
  if [ -d "$d" ]; then
    echo "[MOVE droid pilot output] $d -> outputs_vpi/archive_pilot_20260509/droid_pilot_old/"
    mv "$d" outputs_vpi/archive_pilot_20260509/droid_pilot_old/
  fi
done

echo
echo "=== Update batch scripts paths after moving to scripts_vpi/current ==="

python - <<'PY'
from pathlib import Path

batch_files = [
    Path("scripts_vpi/current/batch_prepare_droid100_uniform.sh"),
    Path("scripts_vpi/current/batch_run_droid_visual_text_suite.sh"),
]

repls = [
    ("scripts_vpi/prepare_droid100_episode.py", "scripts_vpi/current/prepare_droid100_episode.py"),
    ("scripts_vpi/run_droid_visual_text_suite.py", "scripts_vpi/current/run_droid_visual_text_suite.py"),
    ('OUT_ROOT="outputs_vpi/droid100_uniform_debug"', 'OUT_ROOT="outputs_vpi/formal_droid100/uniform_debug"'),
    ('EP_ROOT="outputs_vpi/droid100_uniform_debug"', 'EP_ROOT="outputs_vpi/formal_droid100/uniform_debug"'),
    ('OUT_ROOT="outputs_vpi/droid100_visual_text_suite"', 'OUT_ROOT="outputs_vpi/formal_droid100/visual_text_suite"'),
    ('LOG_DIR="outputs_vpi/batch_logs"', 'LOG_DIR="outputs_vpi/logs/formal/batch_logs"'),
]

for p in batch_files:
    if not p.exists():
        continue
    text = p.read_text()
    for old, new in repls:
        text = text.replace(old, new)
    p.write_text(text)
    print("[UPDATED]", p)
PY

chmod +x scripts_vpi/current/*.sh

echo
echo "[DONE] Reorganization complete."
