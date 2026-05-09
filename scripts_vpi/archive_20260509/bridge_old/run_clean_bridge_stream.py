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


def main():
    root = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")

    ckpt = root / "models/downloaded_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt"

    frame_dir = root / "outputs_vpi/bridge_debug"
    gt_json = frame_dir / "first_10_steps.json"

    instruction = "put small spoon from basket to tray"

    out_dir = root / "outputs_vpi/clean_stream_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== paths ===")
    print("ckpt:", ckpt, ckpt.exists())
    print("frame_dir:", frame_dir, frame_dir.exists())
    print("gt_json:", gt_json, gt_json.exists())
    print("out_dir:", out_dir)

    assert ckpt.exists(), f"Missing checkpoint: {ckpt}"
    assert gt_json.exists(), f"Missing gt json: {gt_json}"

    gt_data = json.loads(gt_json.read_text(encoding="utf-8"))
    gt_actions = np.array(gt_data["actions"], dtype=np.float32)
    instructions = gt_data.get("instructions", [])

    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    assert len(frame_paths) > 0, f"No frames found in {frame_dir}"

    # We only compare frames that have ground-truth actions saved.
    num_frames = min(len(frame_paths), len(gt_actions))
    frame_paths = frame_paths[:num_frames]

    print("\n=== data ===")
    print("num saved frames:", len(frame_paths))
    print("gt_actions shape:", gt_actions.shape)
    print("instruction:", instruction)
    if instructions:
        print("dataset instruction example:", instructions[0])

    print("\n=== cuda ===")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())

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

    all_results = []
    all_pred_actions = []

    print("\n=== stream inference ===")
    for t, image_path in enumerate(frame_paths):
        print(f"\n--- timestep {t} / {num_frames - 1}: {image_path.name} ---")
        image = Image.open(image_path).convert("RGB")

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
        all_pred_actions.append(actions)

        # First action of this chunk should correspond roughly to gt action at timestep t.
        first_l2 = l2(actions[0], gt_actions[t])

        # Chunk comparison: compare pred chunk with available gt future actions.
        remaining_gt = gt_actions[t:]
        k = min(len(actions), len(remaining_gt))
        chunk_l2_per_step = np.linalg.norm(actions[:k] - remaining_gt[:k], axis=1)
        chunk_mean_l2 = float(chunk_l2_per_step.mean())

        # Gripper agreement for first action, using dimension 6.
        pred_gripper = float(actions[0, 6])
        gt_gripper = float(gt_actions[t, 6])
        gripper_match = int(round(pred_gripper) == round(gt_gripper))

        print("pred chunk shape:", actions.shape)
        print("pred first action:", actions[0])
        print("gt action:", gt_actions[t])
        print("first_action_l2:", first_l2)
        print("chunk_compare_steps:", k)
        print("chunk_mean_l2:", chunk_mean_l2)
        print("pred_gripper:", pred_gripper, "gt_gripper:", gt_gripper, "match:", gripper_match)

        result = {
            "timestep": t,
            "image_path": str(image_path),
            "instruction": instruction,
            "pred_chunk_shape": list(actions.shape),
            "gt_action": gt_actions[t].tolist(),
            "pred_first_action": actions[0].tolist(),
            "normalized_first_action": normalized_actions[0].tolist(),
            "first_action_l2": first_l2,
            "chunk_compare_steps": int(k),
            "chunk_mean_l2": chunk_mean_l2,
            "chunk_l2_per_step": chunk_l2_per_step.tolist(),
            "pred_gripper": pred_gripper,
            "gt_gripper": gt_gripper,
            "gripper_match": gripper_match,
        }
        all_results.append(result)

    all_pred_actions = np.stack(all_pred_actions, axis=0)

    np.save(out_dir / "pred_action_chunks.npy", all_pred_actions)
    np.save(out_dir / "gt_actions.npy", gt_actions)

    summary = {
        "instruction": instruction,
        "num_frames": num_frames,
        "use_generate": False,
        "prompt_mode": "default",
        "cache_latent": False,
        "pred_action_chunks_shape": list(all_pred_actions.shape),
        "gt_actions_shape": list(gt_actions.shape),
        "mean_first_action_l2": float(np.mean([r["first_action_l2"] for r in all_results])),
        "mean_chunk_l2": float(np.mean([r["chunk_mean_l2"] for r in all_results])),
        "gripper_match_rate": float(np.mean([r["gripper_match"] for r in all_results])),
        "results": all_results,
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
                "first_action_l2",
                "chunk_compare_steps",
                "chunk_mean_l2",
                "pred_gripper",
                "gt_gripper",
                "gripper_match",
                "image_path",
            ],
        )
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "timestep": r["timestep"],
                "first_action_l2": r["first_action_l2"],
                "chunk_compare_steps": r["chunk_compare_steps"],
                "chunk_mean_l2": r["chunk_mean_l2"],
                "pred_gripper": r["pred_gripper"],
                "gt_gripper": r["gt_gripper"],
                "gripper_match": r["gripper_match"],
                "image_path": r["image_path"],
            })

    print("\n=== final summary ===")
    print("num_frames:", summary["num_frames"])
    print("pred_action_chunks_shape:", summary["pred_action_chunks_shape"])
    print("mean_first_action_l2:", summary["mean_first_action_l2"])
    print("mean_chunk_l2:", summary["mean_chunk_l2"])
    print("gripper_match_rate:", summary["gripper_match_rate"])
    print("saved:", out_dir)
    print("csv:", csv_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n========== ERROR ==========")
        traceback.print_exc()
        raise
