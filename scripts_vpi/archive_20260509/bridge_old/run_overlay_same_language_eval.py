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
    overlay_dir = root / "outputs_vpi/bridge_full_episode_visual_centric"
    ep_json = clean_dir / "episode.json"

    out_dir = root / "outputs_vpi/overlay_same_language_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    episode = json.loads(ep_json.read_text(encoding="utf-8"))
    instruction = episode["instructions"][0]
    gt_actions = np.asarray(episode["actions"], dtype=np.float32)

    clean_frames = sorted(clean_dir.glob("frame_*.png"))
    overlay_frames = sorted(overlay_dir.glob("frame_*.png"))

    assert ckpt.exists(), f"Missing checkpoint: {ckpt}"
    assert len(clean_frames) == len(overlay_frames) == len(gt_actions)

    print("=== setup ===")
    print("num_frames:", len(clean_frames))
    print("instruction:", instruction)
    print("clean_dir:", clean_dir)
    print("overlay_dir:", overlay_dir)
    print("out_dir:", out_dir)

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
    overlay_chunks = []
    rows = []

    print("\n=== paired eval: clean vs overlay_same_language ===")
    for t, (clean_path, overlay_path) in enumerate(zip(clean_frames, overlay_frames)):
        print(f"\n--- timestep {t:03d} ---")

        clean_img = Image.open(clean_path).convert("RGB")
        overlay_img = Image.open(overlay_path).convert("RGB")

        seed = 20000 + t

        # Clean: original image + normal task language.
        set_seed(seed)
        with torch.no_grad():
            clean_actions, _, _ = vla.predict_action(
                image=clean_img,
                instruction=instruction,
                unnorm_key="bridge_dataset",
                use_generate=False,
                cache_latent=False,
                prompt_mode="default",
            )

        # Overlay same-language: image has same task text, and language is still normal task.
        set_seed(seed)
        with torch.no_grad():
            overlay_actions, _, _ = vla.predict_action(
                image=overlay_img,
                instruction=instruction,
                unnorm_key="bridge_dataset",
                use_generate=False,
                cache_latent=False,
                prompt_mode="default",
            )

        clean_actions = np.asarray(clean_actions, dtype=np.float32)
        overlay_actions = np.asarray(overlay_actions, dtype=np.float32)

        clean_chunks.append(clean_actions)
        overlay_chunks.append(overlay_actions)

        c0 = clean_actions[0]
        o0 = overlay_actions[0]
        gt0 = gt_actions[t]

        overlay_clean_l2_6d = l2(o0[:6], c0[:6])
        overlay_clean_l2_7d = l2(o0[:7], c0[:7])
        overlay_clean_gripper_changed = int(gripper_round(o0[6]) != gripper_round(c0[6]))

        clean_gt_l2_6d = l2(c0[:6], gt0[:6])
        overlay_gt_l2_6d = l2(o0[:6], gt0[:6])
        clean_gt_grip = int(gripper_round(c0[6]) == gripper_round(gt0[6]))
        overlay_gt_grip = int(gripper_round(o0[6]) == gripper_round(gt0[6]))

        print("clean first:", c0)
        print("overlay first:", o0)
        print("gt first:", gt0)
        print("overlay-clean 6D L2:", overlay_clean_l2_6d)
        print("overlay-clean gripper changed:", overlay_clean_gripper_changed)
        print("clean-gt 6D L2:", clean_gt_l2_6d)
        print("overlay-gt 6D L2:", overlay_gt_l2_6d)

        rows.append({
            "timestep": t,
            "seed": seed,
            "clean_image": str(clean_path),
            "overlay_image": str(overlay_path),
            "instruction": instruction,
            "prompt_mode": "default",
            "overlay_clean_l2_6d": overlay_clean_l2_6d,
            "overlay_clean_l2_7d": overlay_clean_l2_7d,
            "overlay_clean_gripper_changed": overlay_clean_gripper_changed,
            "clean_gt_l2_6d": clean_gt_l2_6d,
            "overlay_gt_l2_6d": overlay_gt_l2_6d,
            "clean_gt_gripper_match": clean_gt_grip,
            "overlay_gt_gripper_match": overlay_gt_grip,
            "clean_first_action": c0.tolist(),
            "overlay_first_action": o0.tolist(),
            "gt_action": gt0.tolist(),
        })

    clean_chunks = np.stack(clean_chunks, axis=0)
    overlay_chunks = np.stack(overlay_chunks, axis=0)

    np.save(out_dir / "clean_chunks.npy", clean_chunks)
    np.save(out_dir / "overlay_chunks.npy", overlay_chunks)
    np.save(out_dir / "gt_actions.npy", gt_actions)

    summary = {
        "instruction": instruction,
        "num_frames": len(rows),
        "condition_clean": {
            "image": "original frame",
            "external_instruction": instruction,
            "prompt_mode": "default",
            "use_generate": False,
        },
        "condition_overlay_same_language": {
            "image": "same frame with overlaid identical task text",
            "external_instruction": instruction,
            "prompt_mode": "default",
            "use_generate": False,
        },
        "clean_chunks_shape": list(clean_chunks.shape),
        "overlay_chunks_shape": list(overlay_chunks.shape),
        "mean_overlay_clean_l2_6d": float(np.mean([r["overlay_clean_l2_6d"] for r in rows])),
        "median_overlay_clean_l2_6d": float(np.median([r["overlay_clean_l2_6d"] for r in rows])),
        "max_overlay_clean_l2_6d": float(np.max([r["overlay_clean_l2_6d"] for r in rows])),
        "overlay_clean_gripper_change_rate": float(np.mean([r["overlay_clean_gripper_changed"] for r in rows])),
        "mean_clean_gt_l2_6d": float(np.mean([r["clean_gt_l2_6d"] for r in rows])),
        "mean_overlay_gt_l2_6d": float(np.mean([r["overlay_gt_l2_6d"] for r in rows])),
        "clean_gt_gripper_acc": float(np.mean([r["clean_gt_gripper_match"] for r in rows])),
        "overlay_gt_gripper_acc": float(np.mean([r["overlay_gt_gripper_match"] for r in rows])),
        "rows": rows,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== final summary ===")
    for k in [
        "num_frames",
        "mean_overlay_clean_l2_6d",
        "median_overlay_clean_l2_6d",
        "max_overlay_clean_l2_6d",
        "overlay_clean_gripper_change_rate",
        "mean_clean_gt_l2_6d",
        "mean_overlay_gt_l2_6d",
        "clean_gt_gripper_acc",
        "overlay_gt_gripper_acc",
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
