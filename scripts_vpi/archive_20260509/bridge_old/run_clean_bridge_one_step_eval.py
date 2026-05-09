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
    ep_dir = root / "outputs_vpi/bridge_full_episode"
    ep_json = ep_dir / "episode.json"

    out_dir = root / "outputs_vpi/clean_one_step_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    episode = json.loads(ep_json.read_text(encoding="utf-8"))
    instruction = episode["instructions"][0]
    gt_actions = np.asarray(episode["actions"], dtype=np.float32)
    frame_paths = sorted(ep_dir.glob("frame_*.png"))

    assert ckpt.exists(), f"Missing checkpoint: {ckpt}"
    assert len(frame_paths) == len(gt_actions), (len(frame_paths), len(gt_actions))

    print("=== setup ===")
    print("num_frames:", len(frame_paths))
    print("instruction:", instruction)
    print("ckpt:", ckpt)
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

    pred_chunks = []
    pred_first_actions = []
    rows = []

    print("\n=== one-step teacher-forced evaluation ===")
    for t, frame_path in enumerate(frame_paths):
        print(f"\n--- timestep {t:03d} ---")
        image = Image.open(frame_path).convert("RGB")

        seed = 30000 + t
        set_seed(seed)

        with torch.no_grad():
            pred_chunk, pred_norm, meta = vla.predict_action(
                image=image,
                instruction=instruction,
                unnorm_key="bridge_dataset",
                use_generate=False,
                cache_latent=False,
                prompt_mode="default",
            )

        pred_chunk = np.asarray(pred_chunk, dtype=np.float32)
        pred_action = pred_chunk[0]        # 关键：只取 action chunk 的第一个 action
        gt_action = gt_actions[t]

        pred_chunks.append(pred_chunk)
        pred_first_actions.append(pred_action)

        l2_6d = l2(pred_action[:6], gt_action[:6])
        l2_7d = l2(pred_action[:7], gt_action[:7])
        grip_match = int(gripper_round(pred_action[6]) == gripper_round(gt_action[6]))

        print("pred_chunk shape:", pred_chunk.shape)
        print("pred_action = pred_chunk[0]:", pred_action)
        print("gt_action:", gt_action)
        print("one_step_l2_6d:", l2_6d)
        print("one_step_l2_7d:", l2_7d)
        print("gripper_match:", grip_match)

        rows.append({
            "timestep": t,
            "seed": seed,
            "image": str(frame_path),
            "instruction": instruction,
            "pred_chunk_shape": list(pred_chunk.shape),
            "pred_action_chunk_index_used": 0,
            "pred_action": pred_action.tolist(),
            "gt_action": gt_action.tolist(),
            "one_step_l2_6d": l2_6d,
            "one_step_l2_7d": l2_7d,
            "pred_gripper": float(pred_action[6]),
            "gt_gripper": float(gt_action[6]),
            "gripper_match": grip_match,
        })

    pred_chunks = np.stack(pred_chunks, axis=0)
    pred_first_actions = np.stack(pred_first_actions, axis=0)

    np.save(out_dir / "pred_chunks.npy", pred_chunks)
    np.save(out_dir / "pred_first_actions.npy", pred_first_actions)
    np.save(out_dir / "gt_actions.npy", gt_actions)

    summary = {
        "protocol": "one_step_teacher_forced",
        "definition": "For each frame_t, run predict_action and compare pred_chunk_t[0] with gt_action_t.",
        "instruction": instruction,
        "num_frames": len(rows),
        "pred_chunks_shape": list(pred_chunks.shape),
        "pred_first_actions_shape": list(pred_first_actions.shape),
        "gt_actions_shape": list(gt_actions.shape),
        "mean_l2_6d": float(np.mean([r["one_step_l2_6d"] for r in rows])),
        "median_l2_6d": float(np.median([r["one_step_l2_6d"] for r in rows])),
        "max_l2_6d": float(np.max([r["one_step_l2_6d"] for r in rows])),
        "mean_l2_7d": float(np.mean([r["one_step_l2_7d"] for r in rows])),
        "gripper_acc": float(np.mean([r["gripper_match"] for r in rows])),
        "rows": rows,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== final summary ===")
    for k in [
        "num_frames",
        "pred_chunks_shape",
        "pred_first_actions_shape",
        "gt_actions_shape",
        "mean_l2_6d",
        "median_l2_6d",
        "max_l2_6d",
        "mean_l2_7d",
        "gripper_acc",
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
