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


WRONG_INSTRUCTIONS = [
    "open the bottom drawer",
    "move the cup to the sink",
]


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

    out_dir = root / "outputs_vpi/wrong_language_default_control"
    out_dir.mkdir(parents=True, exist_ok=True)

    episode = json.loads(ep_json.read_text(encoding="utf-8"))
    correct_instruction = episode["instructions"][0]
    gt_actions = np.asarray(episode["actions"], dtype=np.float32)

    frame_paths = sorted(clean_dir.glob("frame_*.png"))

    assert ckpt.exists(), f"Missing checkpoint: {ckpt}"
    assert len(frame_paths) == len(gt_actions), (len(frame_paths), len(gt_actions))

    print("=== setup ===")
    print("ckpt:", ckpt)
    print("clean_dir:", clean_dir)
    print("out_dir:", out_dir)
    print("num_frames:", len(frame_paths))
    print("correct_instruction:", correct_instruction)
    print("wrong_instructions:", WRONG_INSTRUCTIONS)

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

    all_results = {}

    for wrong_instruction in WRONG_INSTRUCTIONS:
        tag = wrong_instruction.replace(" ", "_").replace("/", "_")
        print(f"\n\n==============================")
        print(f"WRONG INSTRUCTION: {wrong_instruction}")
        print(f"==============================")

        clean_chunks = []
        wrong_chunks = []
        rows = []

        for t, frame_path in enumerate(frame_paths):
            print(f"\n--- timestep {t:03d} ---")
            print("image:", frame_path.name)

            img = Image.open(frame_path).convert("RGB")

            seed = 50000 + t

            # Clean: original image + correct task language + default prompt.
            set_seed(seed)
            with torch.no_grad():
                clean_actions, _, _ = vla.predict_action(
                    image=img,
                    instruction=correct_instruction,
                    unnorm_key="bridge_dataset",
                    use_generate=False,
                    cache_latent=False,
                    prompt_mode="default",
                )

            # Wrong-language: same original image + wrong task language + default prompt.
            # In default mode, instruction IS inserted into:
            # "What action should the robot take to {instruction}?"
            set_seed(seed)
            with torch.no_grad():
                wrong_actions, _, _ = vla.predict_action(
                    image=img,
                    instruction=wrong_instruction,
                    unnorm_key="bridge_dataset",
                    use_generate=False,
                    cache_latent=False,
                    prompt_mode="default",
                )

            clean_actions = np.asarray(clean_actions, dtype=np.float32)
            wrong_actions = np.asarray(wrong_actions, dtype=np.float32)

            clean_chunks.append(clean_actions)
            wrong_chunks.append(wrong_actions)

            c0 = clean_actions[0]
            w0 = wrong_actions[0]
            gt0 = gt_actions[t]

            wrong_clean_l2_6d = l2(w0[:6], c0[:6])
            wrong_clean_l2_7d = l2(w0[:7], c0[:7])
            wrong_clean_gripper_changed = int(gripper_round(w0[6]) != gripper_round(c0[6]))

            clean_gt_l2_6d = l2(c0[:6], gt0[:6])
            wrong_gt_l2_6d = l2(w0[:6], gt0[:6])
            clean_gt_grip = int(gripper_round(c0[6]) == gripper_round(gt0[6]))
            wrong_gt_grip = int(gripper_round(w0[6]) == gripper_round(gt0[6]))

            print("clean first:", c0)
            print("wrong first:", w0)
            print("gt first:", gt0)
            print("wrong-clean 6D L2:", wrong_clean_l2_6d)
            print("wrong-clean gripper changed:", wrong_clean_gripper_changed)
            print("clean-gt 6D L2:", clean_gt_l2_6d)
            print("wrong-gt 6D L2:", wrong_gt_l2_6d)

            rows.append({
                "timestep": t,
                "seed": seed,
                "image": str(frame_path),
                "correct_instruction": correct_instruction,
                "wrong_instruction": wrong_instruction,
                "wrong_clean_l2_6d": wrong_clean_l2_6d,
                "wrong_clean_l2_7d": wrong_clean_l2_7d,
                "wrong_clean_gripper_changed": wrong_clean_gripper_changed,
                "clean_gt_l2_6d": clean_gt_l2_6d,
                "wrong_gt_l2_6d": wrong_gt_l2_6d,
                "clean_gt_gripper_match": clean_gt_grip,
                "wrong_gt_gripper_match": wrong_gt_grip,
                "clean_first_action": c0.tolist(),
                "wrong_first_action": w0.tolist(),
                "gt_action": gt0.tolist(),
            })

        clean_chunks = np.stack(clean_chunks, axis=0)
        wrong_chunks = np.stack(wrong_chunks, axis=0)

        np.save(out_dir / f"clean_chunks__{tag}.npy", clean_chunks)
        np.save(out_dir / f"wrong_chunks__{tag}.npy", wrong_chunks)

        summary = {
            "protocol": "wrong_language_default_control_chunk0",
            "definition": "Compare clean_chunk[0] against wrong_language_chunk[0]. Both use original image and prompt_mode=default; only external task language changes.",
            "correct_instruction": correct_instruction,
            "wrong_instruction": wrong_instruction,
            "num_frames": len(rows),
            "condition_clean": {
                "image": "original frame without overlaid text",
                "external_instruction": correct_instruction,
                "prompt_mode": "default",
                "use_generate": False,
            },
            "condition_wrong_language": {
                "image": "original frame without overlaid text",
                "external_instruction": wrong_instruction,
                "prompt_mode": "default",
                "use_generate": False,
            },
            "clean_chunks_shape": list(clean_chunks.shape),
            "wrong_chunks_shape": list(wrong_chunks.shape),
            "mean_wrong_clean_l2_6d": float(np.mean([r["wrong_clean_l2_6d"] for r in rows])),
            "median_wrong_clean_l2_6d": float(np.median([r["wrong_clean_l2_6d"] for r in rows])),
            "max_wrong_clean_l2_6d": float(np.max([r["wrong_clean_l2_6d"] for r in rows])),
            "wrong_clean_gripper_change_rate": float(np.mean([r["wrong_clean_gripper_changed"] for r in rows])),
            "mean_clean_gt_l2_6d": float(np.mean([r["clean_gt_l2_6d"] for r in rows])),
            "mean_wrong_gt_l2_6d": float(np.mean([r["wrong_gt_l2_6d"] for r in rows])),
            "clean_gt_gripper_acc": float(np.mean([r["clean_gt_gripper_match"] for r in rows])),
            "wrong_gt_gripper_acc": float(np.mean([r["wrong_gt_gripper_match"] for r in rows])),
            "rows": rows,
        }

        (out_dir / f"summary__{tag}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        all_results[wrong_instruction] = summary

        print("\n=== summary for wrong instruction ===")
        for k in [
            "num_frames",
            "mean_wrong_clean_l2_6d",
            "median_wrong_clean_l2_6d",
            "max_wrong_clean_l2_6d",
            "wrong_clean_gripper_change_rate",
            "mean_clean_gt_l2_6d",
            "mean_wrong_gt_l2_6d",
            "clean_gt_gripper_acc",
            "wrong_gt_gripper_acc",
        ]:
            print(k + ":", summary[k])

    compact = {
        wrong: {
            "mean_wrong_clean_l2_6d": s["mean_wrong_clean_l2_6d"],
            "median_wrong_clean_l2_6d": s["median_wrong_clean_l2_6d"],
            "max_wrong_clean_l2_6d": s["max_wrong_clean_l2_6d"],
            "wrong_clean_gripper_change_rate": s["wrong_clean_gripper_change_rate"],
            "mean_clean_gt_l2_6d": s["mean_clean_gt_l2_6d"],
            "mean_wrong_gt_l2_6d": s["mean_wrong_gt_l2_6d"],
            "clean_gt_gripper_acc": s["clean_gt_gripper_acc"],
            "wrong_gt_gripper_acc": s["wrong_gt_gripper_acc"],
        }
        for wrong, s in all_results.items()
    }

    (out_dir / "summary_all.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n\n=== final compact summary ===")
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    print("\nsaved:", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n========== ERROR ==========")
        traceback.print_exc()
        raise
