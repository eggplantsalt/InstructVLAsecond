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
    visual_dir = root / "outputs_vpi/bridge_full_episode_visual_centric"
    ep_json = clean_dir / "episode.json"

    out_dir = root / "outputs_vpi/visual_centric_pair_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    episode = json.loads(ep_json.read_text(encoding="utf-8"))
    instruction = episode["instructions"][0]
    gt_actions = np.asarray(episode["actions"], dtype=np.float32)

    clean_frames = sorted(clean_dir.glob("frame_*.png"))
    visual_frames = sorted(visual_dir.glob("frame_*.png"))

    assert ckpt.exists(), f"Missing checkpoint: {ckpt}"
    assert len(clean_frames) == len(visual_frames), (len(clean_frames), len(visual_frames))
    assert len(clean_frames) == len(gt_actions), (len(clean_frames), len(gt_actions))

    print("=== paths ===")
    print("ckpt:", ckpt, ckpt.exists())
    print("clean_dir:", clean_dir)
    print("visual_dir:", visual_dir)
    print("out_dir:", out_dir)
    print("num_frames:", len(clean_frames))
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
    visual_chunks = []
    rows = []

    print("\n=== paired evaluation ===")
    for t, (clean_path, visual_path) in enumerate(zip(clean_frames, visual_frames)):
        print(f"\n--- timestep {t:03d} ---")
        print("clean:", clean_path.name)
        print("visual:", visual_path.name)

        clean_img = Image.open(clean_path).convert("RGB")
        visual_img = Image.open(visual_path).convert("RGB")

        seed = 10000 + t

        # Clean baseline.
        set_seed(seed)
        with torch.no_grad():
            clean_actions, clean_norm, clean_meta = vla.predict_action(
                image=clean_img,
                instruction=instruction,
                unnorm_key="bridge_dataset",
                use_generate=False,
                cache_latent=False,
                prompt_mode="default",
            )

        # Visual centric condition. Do NOT pass original task through language.
        # The actual task appears only as overlaid image text.
        set_seed(seed)
        with torch.no_grad():
            visual_actions, visual_norm, visual_meta = vla.predict_action(
                image=visual_img,
                instruction="look at the image and follow the instruction written in the image",
                unnorm_key="bridge_dataset",
                use_generate=False,
                cache_latent=False,
                prompt_mode="visual_centric",
            )

        clean_actions = np.asarray(clean_actions, dtype=np.float32)
        visual_actions = np.asarray(visual_actions, dtype=np.float32)

        clean_chunks.append(clean_actions)
        visual_chunks.append(visual_actions)

        c0 = clean_actions[0]
        v0 = visual_actions[0]
        gt0 = gt_actions[t]

        visual_clean_l2_6d = l2(v0[:6], c0[:6])
        visual_clean_l2_7d = l2(v0[:7], c0[:7])
        visual_clean_gripper_changed = int(gripper_round(v0[6]) != gripper_round(c0[6]))

        clean_gt_l2_6d = l2(c0[:6], gt0[:6])
        visual_gt_l2_6d = l2(v0[:6], gt0[:6])

        clean_gt_grip = int(gripper_round(c0[6]) == gripper_round(gt0[6]))
        visual_gt_grip = int(gripper_round(v0[6]) == gripper_round(gt0[6]))

        print("clean first:", c0)
        print("visual first:", v0)
        print("gt first:", gt0)
        print("visual-clean 6D L2:", visual_clean_l2_6d)
        print("visual-clean gripper changed:", visual_clean_gripper_changed)
        print("clean-gt 6D L2:", clean_gt_l2_6d)
        print("visual-gt 6D L2:", visual_gt_l2_6d)

        rows.append({
            "timestep": t,
            "seed": seed,
            "clean_image": str(clean_path),
            "visual_image": str(visual_path),
            "instruction_clean": instruction,
            "instruction_visual_external": "follow the instruction shown in the image",
            "visual_clean_l2_6d": visual_clean_l2_6d,
            "visual_clean_l2_7d": visual_clean_l2_7d,
            "visual_clean_gripper_changed": visual_clean_gripper_changed,
            "clean_gt_l2_6d": clean_gt_l2_6d,
            "visual_gt_l2_6d": visual_gt_l2_6d,
            "clean_gt_gripper_match": clean_gt_grip,
            "visual_gt_gripper_match": visual_gt_grip,
            "clean_first_action": c0.tolist(),
            "visual_first_action": v0.tolist(),
            "gt_action": gt0.tolist(),
        })

    clean_chunks = np.stack(clean_chunks, axis=0)
    visual_chunks = np.stack(visual_chunks, axis=0)

    np.save(out_dir / "clean_chunks.npy", clean_chunks)
    np.save(out_dir / "visual_chunks.npy", visual_chunks)
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
        "condition_visual_centric": {
            "image": "same frame with overlaid task text",
            "external_instruction": "follow the instruction shown in the image",
            "prompt_mode": "visual_centric",
            "use_generate": False,
        },
        "clean_chunks_shape": list(clean_chunks.shape),
        "visual_chunks_shape": list(visual_chunks.shape),
        "mean_visual_clean_l2_6d": float(np.mean([r["visual_clean_l2_6d"] for r in rows])),
        "median_visual_clean_l2_6d": float(np.median([r["visual_clean_l2_6d"] for r in rows])),
        "max_visual_clean_l2_6d": float(np.max([r["visual_clean_l2_6d"] for r in rows])),
        "visual_clean_gripper_change_rate": float(np.mean([r["visual_clean_gripper_changed"] for r in rows])),
        "mean_clean_gt_l2_6d": float(np.mean([r["clean_gt_l2_6d"] for r in rows])),
        "mean_visual_gt_l2_6d": float(np.mean([r["visual_gt_l2_6d"] for r in rows])),
        "clean_gt_gripper_acc": float(np.mean([r["clean_gt_gripper_match"] for r in rows])),
        "visual_gt_gripper_acc": float(np.mean([r["visual_gt_gripper_match"] for r in rows])),
        "rows": rows,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== final summary ===")
    for k in [
        "num_frames",
        "mean_visual_clean_l2_6d",
        "median_visual_clean_l2_6d",
        "max_visual_clean_l2_6d",
        "visual_clean_gripper_change_rate",
        "mean_clean_gt_l2_6d",
        "mean_visual_gt_l2_6d",
        "clean_gt_gripper_acc",
        "visual_gt_gripper_acc",
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
