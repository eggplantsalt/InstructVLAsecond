import argparse
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
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
        )
    )


def gripper_round(x):
    return int(round(float(x)))


def load_model(ckpt: Path):
    kwargs = dict(
        load_for_training=False,
        action_dim=7,
        future_action_window_size=15,
        past_action_window_size=0,
    )

    try:
        model = load_vla(ckpt, stage="stage2", **kwargs)
    except TypeError:
        model = load_vla(ckpt, **kwargs)
        model.stage = "stage2"

    return model


def predict_one(vla, image, instruction, prompt_mode, seed):
    # Very important:
    # use_generate=True branch may reuse last_response every 20 runs.
    # For paired evaluation, reset it so each condition does fresh reasoning.
    vla.last_response = None
    vla.latent = None
    vla.run_index = 0

    set_seed(seed)

    with torch.no_grad():
        actions, norm_actions, meta = vla.predict_action(
            image=image,
            instruction=instruction,
            unnorm_key="bridge_dataset",
            use_generate=True,
            cache_latent=False,
            prompt_mode=prompt_mode,
        )

    actions = np.asarray(actions, dtype=np.float32)
    response = getattr(vla, "last_response", None)
    response = "" if response is None else str(response)

    return actions, response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_frames", type=int, default=28)
    args = parser.parse_args()

    root = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")

    ckpt = root / "models/downloaded_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt"

    clean_dir = root / "outputs_vpi/bridge_full_episode"
    visual_dir = root / "outputs_vpi/bridge_full_episode_visual_centric"
    ep_json = clean_dir / "episode.json"

    out_dir = root / "outputs_vpi/reasoning_mode_pair_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    episode = json.loads(ep_json.read_text(encoding="utf-8"))
    instruction = episode["instructions"][0]
    gt_actions = np.asarray(episode["actions"], dtype=np.float32)

    clean_frames = sorted(clean_dir.glob("frame_*.png"))
    visual_frames = sorted(visual_dir.glob("frame_*.png"))

    n = min(len(clean_frames), len(visual_frames), len(gt_actions), args.max_frames)

    assert ckpt.exists(), f"Missing checkpoint: {ckpt}"
    assert n > 0, "No frames found"
    assert len(clean_frames) >= n, len(clean_frames)
    assert len(visual_frames) >= n, len(visual_frames)

    src = (root / "vla/instructvla_eagle_dual_sys_v2_meta_query_v2.py").read_text()
    visual_mode = "visual_centric" if "visual_centric" in src else "image_text_primary"

    print("=== setup ===")
    print("ckpt:", ckpt)
    print("clean_dir:", clean_dir)
    print("visual_dir:", visual_dir)
    print("out_dir:", out_dir)
    print("num_frames:", n)
    print("instruction:", instruction)
    print("visual_mode:", visual_mode)

    print("\n=== cuda ===")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())

    print("\n=== load model ===")
    set_seed(42)
    vla = load_model(ckpt)

    if patch_runtime_configs is not None:
        print("Applying patch_runtime_configs...")
        patch_runtime_configs(vla)

    vla = vla.to("cuda").eval().to(torch.float16)

    clean_chunks = []
    visual_chunks = []
    prompt_only_chunks = []
    rows = []

    print("\n=== reasoning-mode paired evaluation ===")

    for t in range(n):
        print(f"\n--- timestep {t:03d} ---")

        clean_img = Image.open(clean_frames[t]).convert("RGB")
        visual_img = Image.open(visual_frames[t]).convert("RGB")
        gt0 = gt_actions[t]

        seed = 60000 + t

        # 1. Clean reasoning:
        # original image + true task language + default prompt.
        clean_actions, clean_resp = predict_one(
            vla=vla,
            image=clean_img,
            instruction=instruction,
            prompt_mode="default",
            seed=seed,
        )

        # 2. Visual-centric reasoning:
        # image with overlaid task text + no task-specific external language.
        visual_actions, visual_resp = predict_one(
            vla=vla,
            image=visual_img,
            instruction="__NO_TASK_TEXT_PROVIDED__",
            prompt_mode=visual_mode,
            seed=seed,
        )

        # 3. Prompt-only / no-image-text reasoning:
        # original image without text + no task-specific external language.
        prompt_actions, prompt_resp = predict_one(
            vla=vla,
            image=clean_img,
            instruction="__NO_TASK_TEXT_PROVIDED__",
            prompt_mode=visual_mode,
            seed=seed,
        )

        clean_chunks.append(clean_actions)
        visual_chunks.append(visual_actions)
        prompt_only_chunks.append(prompt_actions)

        c0 = clean_actions[0]
        v0 = visual_actions[0]
        p0 = prompt_actions[0]

        row = {
            "timestep": t,
            "seed": seed,
            "clean_image": str(clean_frames[t]),
            "visual_image": str(visual_frames[t]),
            "instruction": instruction,
            "visual_mode": visual_mode,
            "use_generate": True,

            "visual_clean_l2_6d": l2(v0[:6], c0[:6]),
            "visual_clean_l2_7d": l2(v0[:7], c0[:7]),
            "visual_clean_gripper_changed": int(
                gripper_round(v0[6]) != gripper_round(c0[6])
            ),

            "prompt_clean_l2_6d": l2(p0[:6], c0[:6]),
            "prompt_clean_l2_7d": l2(p0[:7], c0[:7]),
            "prompt_clean_gripper_changed": int(
                gripper_round(p0[6]) != gripper_round(c0[6])
            ),

            "clean_gt_l2_6d": l2(c0[:6], gt0[:6]),
            "visual_gt_l2_6d": l2(v0[:6], gt0[:6]),
            "prompt_gt_l2_6d": l2(p0[:6], gt0[:6]),

            "clean_gt_gripper_match": int(
                gripper_round(c0[6]) == gripper_round(gt0[6])
            ),
            "visual_gt_gripper_match": int(
                gripper_round(v0[6]) == gripper_round(gt0[6])
            ),
            "prompt_gt_gripper_match": int(
                gripper_round(p0[6]) == gripper_round(gt0[6])
            ),

            "clean_first_action": c0.tolist(),
            "visual_first_action": v0.tolist(),
            "prompt_only_first_action": p0.tolist(),
            "gt_action": gt0.tolist(),

            "clean_reasoning_response": clean_resp,
            "visual_reasoning_response": visual_resp,
            "prompt_only_reasoning_response": prompt_resp,
        }

        rows.append(row)

        print("clean first:", c0)
        print("visual first:", v0)
        print("prompt-only first:", p0)
        print("gt first:", gt0)
        print("visual-clean 6D L2:", row["visual_clean_l2_6d"])
        print("prompt-clean 6D L2:", row["prompt_clean_l2_6d"])
        print("visual-clean gripper changed:", row["visual_clean_gripper_changed"])
        print("prompt-clean gripper changed:", row["prompt_clean_gripper_changed"])
        print("clean response:", clean_resp)
        print("visual response:", visual_resp)
        print("prompt-only response:", prompt_resp)

    clean_chunks = np.stack(clean_chunks, axis=0)
    visual_chunks = np.stack(visual_chunks, axis=0)
    prompt_only_chunks = np.stack(prompt_only_chunks, axis=0)

    np.save(out_dir / "clean_reason_chunks.npy", clean_chunks)
    np.save(out_dir / "visual_reason_chunks.npy", visual_chunks)
    np.save(out_dir / "prompt_only_reason_chunks.npy", prompt_only_chunks)
    np.save(out_dir / "gt_actions.npy", gt_actions[:n])

    summary = {
        "protocol": "reasoning_mode_chunk0_pair_eval",
        "definition": (
            "use_generate=True for clean, visual text, and prompt-only. "
            "Compare condition_chunk[0] against clean_reason_chunk[0]."
        ),
        "num_frames": n,
        "instruction": instruction,
        "visual_mode": visual_mode,
        "use_generate": True,

        "condition_clean_reason": {
            "image": "original frame without overlaid text",
            "external_instruction": instruction,
            "prompt_mode": "default",
            "use_generate": True,
        },
        "condition_visual_reason": {
            "image": "frame with overlaid task text",
            "external_instruction": "__NO_TASK_TEXT_PROVIDED__",
            "prompt_mode": visual_mode,
            "use_generate": True,
        },
        "condition_prompt_only_reason": {
            "image": "original frame without overlaid text",
            "external_instruction": "__NO_TASK_TEXT_PROVIDED__",
            "prompt_mode": visual_mode,
            "use_generate": True,
        },

        "clean_chunks_shape": list(clean_chunks.shape),
        "visual_chunks_shape": list(visual_chunks.shape),
        "prompt_only_chunks_shape": list(prompt_only_chunks.shape),

        "mean_visual_clean_l2_6d": float(np.mean([r["visual_clean_l2_6d"] for r in rows])),
        "median_visual_clean_l2_6d": float(np.median([r["visual_clean_l2_6d"] for r in rows])),
        "max_visual_clean_l2_6d": float(np.max([r["visual_clean_l2_6d"] for r in rows])),
        "visual_clean_gripper_change_rate": float(np.mean([r["visual_clean_gripper_changed"] for r in rows])),

        "mean_prompt_clean_l2_6d": float(np.mean([r["prompt_clean_l2_6d"] for r in rows])),
        "median_prompt_clean_l2_6d": float(np.median([r["prompt_clean_l2_6d"] for r in rows])),
        "max_prompt_clean_l2_6d": float(np.max([r["prompt_clean_l2_6d"] for r in rows])),
        "prompt_clean_gripper_change_rate": float(np.mean([r["prompt_clean_gripper_changed"] for r in rows])),

        "mean_clean_gt_l2_6d": float(np.mean([r["clean_gt_l2_6d"] for r in rows])),
        "mean_visual_gt_l2_6d": float(np.mean([r["visual_gt_l2_6d"] for r in rows])),
        "mean_prompt_gt_l2_6d": float(np.mean([r["prompt_gt_l2_6d"] for r in rows])),

        "clean_gt_gripper_acc": float(np.mean([r["clean_gt_gripper_match"] for r in rows])),
        "visual_gt_gripper_acc": float(np.mean([r["visual_gt_gripper_match"] for r in rows])),
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
        "mean_visual_clean_l2_6d",
        "median_visual_clean_l2_6d",
        "max_visual_clean_l2_6d",
        "visual_clean_gripper_change_rate",
        "mean_prompt_clean_l2_6d",
        "median_prompt_clean_l2_6d",
        "max_prompt_clean_l2_6d",
        "prompt_clean_gripper_change_rate",
        "mean_clean_gt_l2_6d",
        "mean_visual_gt_l2_6d",
        "mean_prompt_gt_l2_6d",
        "clean_gt_gripper_acc",
        "visual_gt_gripper_acc",
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
