import json
import csv
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


def l2(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    root = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")
    ckpt = root / "models/downloaded_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt"

    episode_dir = root / "outputs_vpi/droid100_debug/episode_000000"
    meta = json.loads((episode_dir / "episode_meta.json").read_text(encoding="utf-8"))
    gt_actions = np.load(episode_dir / "actions.npy")
    gt_states = np.load(episode_dir / "states.npy")

    instruction = meta["instruction"]
    frame_paths = [Path(p) for p in meta["frame_paths"]]
    num_frames = min(len(frame_paths), len(gt_actions), 30)

    out_dir = root / "outputs_vpi/droid100_clean_stream"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== setup ===")
    print("ckpt:", ckpt, ckpt.exists())
    print("instruction:", instruction)
    print("num_frames:", num_frames)
    print("gt_actions shape:", gt_actions.shape)
    print("gt_states shape:", gt_states.shape)
    print("out_dir:", out_dir)

    print("\n=== load model ===")
    vla = load_vla(
        ckpt,
        load_for_training=False,
        action_dim=7,
        future_action_window_size=15,
    )

    if patch_runtime_configs is not None:
        print("Applying patch_runtime_configs...")
        patch_runtime_configs(vla)

    vla = vla.to("cuda").eval().to(torch.float16)

    print("\n=== norm stats ===")
    if hasattr(vla, "norm_stats"):
        print("norm_stats keys:", list(vla.norm_stats.keys()))

    pred_chunks = []
    pred_norm_chunks = []
    rows = []

    print("\n=== clean stream inference ===")
    for t in range(num_frames):
        image_path = frame_paths[t]
        image = Image.open(image_path).convert("RGB")

        # 固定每一帧 seed，后续 visual 条件也用同样 seed，减少 sampling noise。
        set_seed(1000 + t)

        with torch.no_grad():
            actions, normalized_actions, meta_feature = vla.predict_action(
                image=image,
                instruction=instruction,
                unnorm_key="bridge_dataset",
                use_generate=False,
                cache_latent=False,
                prompt_mode="default",
            )

        actions = np.asarray(actions, dtype=np.float32)
        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)

        pred_chunks.append(actions)
        pred_norm_chunks.append(normalized_actions)

        gt0 = gt_actions[t]
        pred0 = actions[0]
        norm0 = normalized_actions[0]

        row = {
            "timestep": t,
            "image_path": str(image_path),
            "pred_first_bridge": pred0.tolist(),
            "pred_first_normalized": norm0.tolist(),
            "gt_action": gt0.tolist(),
            "sanity_bridge_vs_gt_l2_6d": l2(pred0[:6], gt0[:6]),
            "sanity_bridge_vs_gt_l2_7d": l2(pred0, gt0),
            "pred_gripper": float(pred0[6]),
            "gt_gripper": float(gt0[6]),
            "normalized_6d_norm": l2(norm0[:6], np.zeros(6, dtype=np.float32)),
        }
        rows.append(row)

        print(
            f"t={t:02d}",
            "norm_first[:6]=", np.round(norm0[:6], 4),
            "bridge_first[:6]=", np.round(pred0[:6], 4),
            "gt[:6]=", np.round(gt0[:6], 4),
            "sanity_l2_6d=", round(row["sanity_bridge_vs_gt_l2_6d"], 4),
            "grip pred/gt=", round(float(pred0[6]), 4), round(float(gt0[6]), 4),
        )

    pred_chunks = np.stack(pred_chunks, axis=0)
    pred_norm_chunks = np.stack(pred_norm_chunks, axis=0)

    np.save(out_dir / "clean_pred_chunks_bridge_unnorm.npy", pred_chunks)
    np.save(out_dir / "clean_pred_chunks_normalized.npy", pred_norm_chunks)
    np.save(out_dir / "gt_actions.npy", gt_actions[:num_frames])
    np.save(out_dir / "gt_states.npy", gt_states[:num_frames])

    summary = {
        "dataset": "lerobot/droid_100",
        "episode": meta["episode"],
        "instruction": instruction,
        "num_frames": num_frames,
        "unnorm_key": "bridge_dataset",
        "prompt_mode": "default",
        "use_generate": False,
        "cache_latent": False,
        "pred_chunks_bridge_shape": list(pred_chunks.shape),
        "pred_chunks_normalized_shape": list(pred_norm_chunks.shape),
        "gt_actions_shape": list(gt_actions[:num_frames].shape),
        "mean_sanity_bridge_vs_gt_l2_6d": float(np.mean([r["sanity_bridge_vs_gt_l2_6d"] for r in rows])),
        "mean_sanity_bridge_vs_gt_l2_7d": float(np.mean([r["sanity_bridge_vs_gt_l2_7d"] for r in rows])),
        "rows": rows,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = out_dir / "per_frame_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestep",
                "sanity_bridge_vs_gt_l2_6d",
                "sanity_bridge_vs_gt_l2_7d",
                "pred_gripper",
                "gt_gripper",
                "normalized_6d_norm",
                "image_path",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "timestep": r["timestep"],
                "sanity_bridge_vs_gt_l2_6d": r["sanity_bridge_vs_gt_l2_6d"],
                "sanity_bridge_vs_gt_l2_7d": r["sanity_bridge_vs_gt_l2_7d"],
                "pred_gripper": r["pred_gripper"],
                "gt_gripper": r["gt_gripper"],
                "normalized_6d_norm": r["normalized_6d_norm"],
                "image_path": r["image_path"],
            })

    print("\n=== final summary ===")
    print("pred_chunks_bridge_shape:", pred_chunks.shape)
    print("pred_chunks_normalized_shape:", pred_norm_chunks.shape)
    print("mean sanity bridge-vs-gt L2 6D:", summary["mean_sanity_bridge_vs_gt_l2_6d"])
    print("mean sanity bridge-vs-gt L2 7D:", summary["mean_sanity_bridge_vs_gt_l2_7d"])
    print("saved:", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n========== ERROR ==========")
        traceback.print_exc()
        raise
