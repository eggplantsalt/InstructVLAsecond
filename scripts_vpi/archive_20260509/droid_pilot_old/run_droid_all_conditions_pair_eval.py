import json
import csv
import random
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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


def grip_round(x):
    return int(round(float(x)))


def load_font(size=16):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = w if not cur else cur + " " + w
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def make_overlay_image(image_path, text, out_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # DROID 图像是 320x180，文字不能太小，也不能遮挡太多。
    # 这里用顶部白色半透明条，约占 30% 高度。
    banner_h = max(46, int(h * 0.30))

    overlay = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    draw.rectangle([0, 0, w, banner_h], fill=(255, 255, 255, 230))
    draw.rectangle([0, 0, w - 1, banner_h - 1], outline=(0, 0, 0, 255), width=2)

    font = load_font(size=15)
    lines = wrap_text(draw, text, font, max_width=w - 12)

    y = 6
    for line in lines[:3]:
        draw.text((6, y), line, fill=(0, 0, 0, 255), font=font)
        y += 17

    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(out_path)
    return out_path


def run_predict(vla, image_path, instruction, prompt_mode, seed):
    set_seed(seed)
    image = Image.open(image_path).convert("RGB")

    # 清掉可能的 cache，避免跨条件污染。
    for attr, val in [
        ("last_response", None),
        ("latent", None),
        ("run_index", 0),
    ]:
        if hasattr(vla, attr):
            try:
                setattr(vla, attr, val)
            except Exception:
                pass

    with torch.no_grad():
        actions, normalized_actions, meta_feature = vla.predict_action(
            image=image,
            instruction=instruction,
            unnorm_key="bridge_dataset",
            use_generate=False,
            cache_latent=False,
            prompt_mode=prompt_mode,
        )

    return (
        np.asarray(actions, dtype=np.float32),
        np.asarray(normalized_actions, dtype=np.float32),
    )


def summarize_condition(name, cond_bridge, cond_norm, clean_bridge, clean_norm):
    rows = []
    n = cond_bridge.shape[0]

    for t in range(n):
        cb = clean_bridge[t, 0]
        cn = clean_norm[t, 0]
        xb = cond_bridge[t, 0]
        xn = cond_norm[t, 0]

        rows.append({
            "timestep": t,
            "bridge_l2_6d": l2(xb[:6], cb[:6]),
            "bridge_l2_7d": l2(xb, cb),
            "normalized_l2_6d": l2(xn[:6], cn[:6]),
            "normalized_l2_7d": l2(xn, cn),
            "gripper_changed": int(grip_round(xb[6]) != grip_round(cb[6])),
            "clean_gripper": float(cb[6]),
            "condition_gripper": float(xb[6]),
        })

    summary = {
        "condition": name,
        "num_frames": n,
        "mean_bridge_l2_6d": float(np.mean([r["bridge_l2_6d"] for r in rows])),
        "median_bridge_l2_6d": float(np.median([r["bridge_l2_6d"] for r in rows])),
        "max_bridge_l2_6d": float(np.max([r["bridge_l2_6d"] for r in rows])),
        "mean_normalized_l2_6d": float(np.mean([r["normalized_l2_6d"] for r in rows])),
        "median_normalized_l2_6d": float(np.median([r["normalized_l2_6d"] for r in rows])),
        "max_normalized_l2_6d": float(np.max([r["normalized_l2_6d"] for r in rows])),
        "gripper_change_rate": float(np.mean([r["gripper_changed"] for r in rows])),
        "rows": rows,
    }
    return summary


def main():
    root = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")
    ckpt = root / "models/downloaded_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt"

    episode_dir = root / "outputs_vpi/droid100_debug/episode_000000"
    clean_dir = root / "outputs_vpi/droid100_clean_stream"

    meta = json.loads((episode_dir / "episode_meta.json").read_text(encoding="utf-8"))
    frame_paths = [Path(p) for p in meta["frame_paths"]]
    instruction = meta["instruction"]

    clean_bridge = np.load(clean_dir / "clean_pred_chunks_bridge_unnorm.npy")
    clean_norm = np.load(clean_dir / "clean_pred_chunks_normalized.npy")

    num_frames = min(len(frame_paths), clean_bridge.shape[0], 30)

    wrong_instruction = "Move the cup to the sink"
    no_task_instruction = "__NO_TASK_TEXT_PROVIDED__"

    out_dir = root / "outputs_vpi/droid100_all_conditions"
    overlay_correct_dir = out_dir / "overlay_correct"
    overlay_wrong_dir = out_dir / "overlay_wrong"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== setup ===")
    print("ckpt:", ckpt, ckpt.exists())
    print("instruction:", instruction)
    print("wrong_instruction:", wrong_instruction)
    print("num_frames:", num_frames)
    print("clean_bridge shape:", clean_bridge.shape)
    print("clean_norm shape:", clean_norm.shape)
    print("out_dir:", out_dir)

    print("\n=== make overlay images ===")
    overlay_correct_paths = []
    overlay_wrong_paths = []

    for t in range(num_frames):
        src = frame_paths[t]
        p_correct = overlay_correct_dir / f"frame_{t:03d}.png"
        p_wrong = overlay_wrong_dir / f"frame_{t:03d}.png"

        make_overlay_image(src, instruction, p_correct)
        make_overlay_image(src, wrong_instruction, p_wrong)

        overlay_correct_paths.append(p_correct)
        overlay_wrong_paths.append(p_wrong)

    print("overlay_correct first:", overlay_correct_paths[0])
    print("overlay_wrong first:", overlay_wrong_paths[0])

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

    if hasattr(vla, "norm_stats"):
        print("norm_stats keys:", list(vla.norm_stats.keys()))

    conditions = {
        "overlay_same": {
            "image_paths": overlay_correct_paths,
            "instruction": instruction,
            "prompt_mode": "default",
        },
        "visual_centric": {
            "image_paths": overlay_correct_paths,
            "instruction": no_task_instruction,
            "prompt_mode": "visual_centric",
        },
        "prompt_only": {
            "image_paths": frame_paths[:num_frames],
            "instruction": no_task_instruction,
            "prompt_mode": "visual_centric",
        },
        "wrong_language": {
            "image_paths": frame_paths[:num_frames],
            "instruction": wrong_instruction,
            "prompt_mode": "default",
        },
        "wrong_image_text": {
            "image_paths": overlay_wrong_paths,
            "instruction": no_task_instruction,
            "prompt_mode": "visual_centric",
        },
    }

    all_summaries = {}

    print("\n=== run conditions ===")
    for cname, cfg in conditions.items():
        print(f"\n--- condition: {cname} ---")
        cond_bridge_chunks = []
        cond_norm_chunks = []

        for t in range(num_frames):
            # 必须和 clean stream 一样用 seed=1000+t
            seed = 1000 + t
            actions, norm_actions = run_predict(
                vla=vla,
                image_path=cfg["image_paths"][t],
                instruction=cfg["instruction"],
                prompt_mode=cfg["prompt_mode"],
                seed=seed,
            )

            cond_bridge_chunks.append(actions)
            cond_norm_chunks.append(norm_actions)

            b_l2 = l2(actions[0, :6], clean_bridge[t, 0, :6])
            n_l2 = l2(norm_actions[0, :6], clean_norm[t, 0, :6])
            gchg = int(grip_round(actions[0, 6]) != grip_round(clean_bridge[t, 0, 6]))

            print(
                f"{cname} t={t:02d}",
                "norm_l2_6d=", round(n_l2, 4),
                "bridge_l2_6d=", round(b_l2, 4),
                "grip_changed=", gchg,
            )

        cond_bridge_chunks = np.stack(cond_bridge_chunks, axis=0)
        cond_norm_chunks = np.stack(cond_norm_chunks, axis=0)

        np.save(out_dir / f"{cname}_pred_chunks_bridge_unnorm.npy", cond_bridge_chunks)
        np.save(out_dir / f"{cname}_pred_chunks_normalized.npy", cond_norm_chunks)

        summary = summarize_condition(
            name=cname,
            cond_bridge=cond_bridge_chunks,
            cond_norm=cond_norm_chunks,
            clean_bridge=clean_bridge[:num_frames],
            clean_norm=clean_norm[:num_frames],
        )
        all_summaries[cname] = summary

        print(
            f"[SUMMARY] {cname}:",
            "mean_norm_l2_6d=", round(summary["mean_normalized_l2_6d"], 4),
            "median_norm_l2_6d=", round(summary["median_normalized_l2_6d"], 4),
            "max_norm_l2_6d=", round(summary["max_normalized_l2_6d"], 4),
            "grip_change_rate=", round(summary["gripper_change_rate"], 4),
        )

    final = {
        "dataset": "lerobot/droid_100",
        "episode": meta["episode"],
        "instruction": instruction,
        "wrong_instruction": wrong_instruction,
        "no_task_instruction": no_task_instruction,
        "num_frames": num_frames,
        "clean_source_dir": str(clean_dir),
        "conditions": all_summaries,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = out_dir / "condition_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "mean_normalized_l2_6d",
                "median_normalized_l2_6d",
                "max_normalized_l2_6d",
                "mean_bridge_l2_6d",
                "median_bridge_l2_6d",
                "max_bridge_l2_6d",
                "gripper_change_rate",
            ],
        )
        writer.writeheader()
        for cname, s in all_summaries.items():
            writer.writerow({
                "condition": cname,
                "mean_normalized_l2_6d": s["mean_normalized_l2_6d"],
                "median_normalized_l2_6d": s["median_normalized_l2_6d"],
                "max_normalized_l2_6d": s["max_normalized_l2_6d"],
                "mean_bridge_l2_6d": s["mean_bridge_l2_6d"],
                "median_bridge_l2_6d": s["median_bridge_l2_6d"],
                "max_bridge_l2_6d": s["max_bridge_l2_6d"],
                "gripper_change_rate": s["gripper_change_rate"],
            })

    print("\n=== final condition summary ===")
    for cname, s in all_summaries.items():
        print(
            cname,
            "mean_norm_l2_6d=", round(s["mean_normalized_l2_6d"], 4),
            "median_norm_l2_6d=", round(s["median_normalized_l2_6d"], 4),
            "max_norm_l2_6d=", round(s["max_normalized_l2_6d"], 4),
            "grip_change_rate=", round(s["gripper_change_rate"], 4),
        )

    print("\nSaved:", out_dir)
    print("summary:", out_dir / "summary.json")
    print("csv:", csv_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n========== ERROR ==========")
        traceback.print_exc()
        raise
