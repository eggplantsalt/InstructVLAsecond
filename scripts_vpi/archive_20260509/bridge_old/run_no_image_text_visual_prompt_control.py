import json
import random
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load_vla

try:
    from scripts_vpi.instructvla_v100_utils import patch_runtime_configs
except Exception:
    patch_runtime_configs = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def gripper_round(x):
    return int(round(float(x)))


def main():
    root = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")

    ckpt = root / "models/downloaded_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt"

    clean_dir = root / "outputs_vpi/bridge_full_episode"
    ep_json = clean_dir / "episode.json"

    out_dir = root / "outputs_vpi/prompt_only_pair_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    episode = json.loads(ep_json.read_text(encoding="utf-8"))
    instruction = episode["instructions"][0]
    gt_actions = np.asarray(episode["actions"], dtype=np.float32)

    frame_paths = sorted(clean_dir.glob("frame_*.png"))

    assert ckpt.exists(), f"Missing checkpoint: {ckpt}"
    assert len(frame_paths) == len(gt_actions), (len(frame_paths), len(gt_actions))

    print("=== setup ===")
    print("ckpt:", ckpt)
    print("clean_dir:", clean_dir)
    print("out_dir:", out_dir)
    print("num_frames:", len(frame_paths))
    print("instruction:", instruction)

    print("\n=== cuda ===")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())

    print("\n=== load model ===")
    set_seed(42)
    vla = load_vla(
        ckpt,
        load_for_training=False,
        action_dim=7,
        future_action_window_size=15,
        past_action_window_size=0,
    )

    if patch_runtime_configs is not None:
        print("Applying patch_runtime_configs...")
        patch_runtime_configs(vla)

    vla = vla.to("cuda").eval().to(torch.float16)

    clean_chunks = []
    prompt_chunks = []
    rows = []

    print("\n=== prompt-only paired evaluation ===")
    for t, frame_path in enumerate(frame_paths):
        print(f"\n--- timestep {t:03d} ---")
        print("image:", frame_path.name)

        img = Image.open(frame_path).convert("RGB")

        seed = 40000 + t

        # Clean: original image + normal task language.
        set_seed(seed)
        with torch.no_grad():
            clean_actions, _, _ = vla.predict_action(
                image=img,
                instruction=instruction,
                unnorm_key="bridge_dataset",
                use_generate=False,
                cache_latent=False,
                prompt_mode="default",
            )

        # Prompt-only: original image has NO text, external language says to follow text in image.
        # This tests whether visual_centric prompt alone can mimic clean without any image text.
        set_seed(seed)
        with torch.no_grad():
            prompt_actions, _, _ = vla.predict_action(
                image=img,
                instruction="__NO_TASK_TEXT_PROVIDED__",
                unnorm_key="bridge_dataset",
                use_generate=False,
                cache_latent=False,
                prompt_mode="visual_centric",
            )

        clean_actions = np.asarray(clean_actions, dtype=np.float32)
        prompt_actions = np.asarray(prompt_actions, dtype=np.float32)

        clean_chunks.append(clean_actions)
        prompt_chunks.append(prompt_actions)

        c0 = clean_actions[0]
        p0 = prompt_actions[0]
        gt0 = gt_actions[t]

        prompt_clean_l2_6d = l2(p0[:6], c0[:6])
        prompt_clean_l2_7d = l2(p0[:7], c0[:7])
        prompt_clean_gripper_changed = int(gripper_round(p0[6]) != gripper_round(c0[6]))

        clean_gt_l2_6d = l2(c0[:6], gt0[:6])
        prompt_gt_l2_6d = l2(p0[:6], gt0[:6])
        clean_gt_grip = int(gripper_round(c0[6]) == gripper_round(gt0[6]))
        prompt_gt_grip = int(gripper_round(p0[6]) == gripper_round(gt0[6]))

        print("clean first:", c0)
        print("prompt-only first:", p0)
        print("gt first:", gt0)
        print("prompt-clean 6D L2:", prompt_clean_l2_6d)
        print("prompt-clean gripper changed:", prompt_clean_gripper_changed)
        print("clean-gt 6D L2:", clean_gt_l2_6d)
        print("prompt-gt 6D L2:", prompt_gt_l2_6d)

        rows.append({
            "timestep": t,
            "seed": seed,
            "image": str(frame_path),
            "instruction_clean": instruction,
            "instruction_prompt_only_external": "__NO_TASK_TEXT_PROVIDED__",
            "prompt_clean_l2_6d": prompt_clean_l2_6d,
            "prompt_clean_l2_7d": prompt_clean_l2_7d,
            "prompt_clean_gripper_changed": prompt_clean_gripper_changed,
            "clean_gt_l2_6d": clean_gt_l2_6d,
            "prompt_gt_l2_6d": prompt_gt_l2_6d,
            "clean_gt_gripper_match": clean_gt_grip,
            "prompt_gt_gripper_match": prompt_gt_grip,
            "clean_first_action": c0.tolist(),
            "prompt_first_action": p0.tolist(),
            "gt_action": gt0.tolist(),
        })

    clean_chunks = np.stack(clean_chunks, axis=0)
    prompt_chunks = np.stack(prompt_chunks, axis=0)

    np.save(out_dir / "clean_chunks.npy", clean_chunks)
    np.save(out_dir / "prompt_only_chunks.npy", prompt_chunks)
    np.save(out_dir / "gt_actions.npy", gt_actions)

    summary = {
        "protocol": "prompt_only_control_chunk0",
        "definition": "Compare clean_chunk[0] against prompt_only_chunk[0]. Prompt-only uses original image without overlaid text, no task-specific external language, and prompt_mode=visual_centric.",
        "instruction": instruction,
        "num_frames": len(rows),
        "condition_clean": {
            "image": "original frame without overlaid text",
            "external_instruction": instruction,
            "prompt_mode": "default",
            "use_generate": False,
        },
        "condition_prompt_only": {
            "image": "original frame without overlaid text",
            "external_instruction": "__NO_TASK_TEXT_PROVIDED__",
            "prompt_mode": "visual_centric",
            "use_generate": False,
        },
        "clean_chunks_shape": list(clean_chunks.shape),
        "prompt_only_chunks_shape": list(prompt_chunks.shape),
        "mean_prompt_clean_l2_6d": float(np.mean([r["prompt_clean_l2_6d"] for r in rows])),
        "median_prompt_clean_l2_6d": float(np.median([r["prompt_clean_l2_6d"] for r in rows])),
        "max_prompt_clean_l2_6d": float(np.max([r["prompt_clean_l2_6d"] for r in rows])),
        "prompt_clean_gripper_change_rate": float(np.mean([r["prompt_clean_gripper_changed"] for r in rows])),
        "mean_clean_gt_l2_6d": float(np.mean([r["clean_gt_l2_6d"] for r in rows])),
        "mean_prompt_gt_l2_6d": float(np.mean([r["prompt_gt_l2_6d"] for r in rows])),
        "clean_gt_gripper_acc": float(np.mean([r["clean_gt_gripper_match"] for r in rows])),
        "prompt_gt_gripper_acc": float(np.mean([r["prompt_gt_gripper_match"] for r in rows])),
        "rows": rows,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== final summary ===")
    for k in [
        "num_frames",
        "mean_prompt_clean_l2_6d",
        "median_prompt_clean_l2_6d",
        "max_prompt_clean_l2_6d",
        "prompt_clean_gripper_change_rate",
        "mean_clean_gt_l2_6d",
        "mean_prompt_gt_l2_6d",
        "clean_gt_gripper_acc",
        "prompt_gt_gripper_acc",
    ]:
        print(k + ":", summary[k])

    print("\nsaved:", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n========== ERROR ==========")
        traceback.print_exc()
        raise
