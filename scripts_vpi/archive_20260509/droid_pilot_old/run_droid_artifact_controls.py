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


def load_font(size=15):
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


def make_top_banner(image_path, out_path, text=None):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    banner_h = max(46, int(h * 0.30))

    overlay = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    draw.rectangle([0, 0, w, banner_h], fill=(255, 255, 255, 230))
    draw.rectangle([0, 0, w - 1, banner_h - 1], outline=(0, 0, 0, 255), width=2)

    if text:
        font = load_font(size=15)
        lines = wrap_text(draw, text, font, max_width=w - 12)
        y = 6
        for line in lines[:3]:
            draw.text((6, y), line, fill=(0, 0, 0, 255), font=font)
            y += 17

    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(out_path)
    return out_path


def make_small_bottom_text(image_path, out_path, text):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    overlay = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    # 尽量低遮挡：底部小条，只占约 14% 高度。
    banner_h = max(24, int(h * 0.14))
    y0 = h - banner_h

    draw.rectangle([0, y0, w, h], fill=(255, 255, 255, 210))
    draw.rectangle([0, y0, w - 1, h - 1], outline=(0, 0, 0, 255), width=1)

    font = load_font(size=10)
    lines = wrap_text(draw, text, font, max_width=w - 10)

    y = y0 + 4
    for line in lines[:2]:
        draw.text((5, y), line, fill=(0, 0, 0, 255), font=font)
        y += 11

    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(out_path)
    return out_path


def run_predict(vla, image_path, instruction, seed):
    set_seed(seed)
    image = Image.open(image_path).convert("RGB")

    # 避免跨条件 cache 污染。
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
            prompt_mode="default",
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

    return {
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


def main():
    root = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")
    ckpt = root / "models/downloaded_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt"

    episode_dir = root / "outputs_vpi/droid100_debug/episode_000000"
    clean_dir = root / "outputs_vpi/droid100_clean_stream"
    previous_dir = root / "outputs_vpi/droid100_all_conditions"

    meta = json.loads((episode_dir / "episode_meta.json").read_text(encoding="utf-8"))
    frame_paths = [Path(p) for p in meta["frame_paths"]]
    instruction = meta["instruction"]

    clean_bridge = np.load(clean_dir / "clean_pred_chunks_bridge_unnorm.npy")
    clean_norm = np.load(clean_dir / "clean_pred_chunks_normalized.npy")

    num_frames = min(len(frame_paths), clean_bridge.shape[0], 30)

    out_dir = root / "outputs_vpi/droid100_artifact_controls"
    out_dir.mkdir(parents=True, exist_ok=True)

    random_text = "XQJ 729 BLUE TRIANGLE"
    blank_dir = out_dir / "blank_banner"
    random_dir = out_dir / "random_text"
    small_dir = out_dir / "small_bottom_text"

    print("=== setup ===")
    print("ckpt:", ckpt, ckpt.exists())
    print("instruction:", instruction)
    print("random_text:", random_text)
    print("num_frames:", num_frames)
    print("clean_bridge shape:", clean_bridge.shape)
    print("clean_norm shape:", clean_norm.shape)
    print("previous_all_conditions:", previous_dir)
    print("out_dir:", out_dir)

    print("\n=== make artifact-control images ===")
    blank_paths = []
    random_paths = []
    small_paths = []

    for t in range(num_frames):
        src = frame_paths[t]

        p_blank = blank_dir / f"frame_{t:03d}.png"
        p_random = random_dir / f"frame_{t:03d}.png"
        p_small = small_dir / f"frame_{t:03d}.png"

        make_top_banner(src, p_blank, text=None)
        make_top_banner(src, p_random, text=random_text)
        make_small_bottom_text(src, p_small, text=instruction)

        blank_paths.append(p_blank)
        random_paths.append(p_random)
        small_paths.append(p_small)

    print("blank first:", blank_paths[0])
    print("random first:", random_paths[0])
    print("small first:", small_paths[0])

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
        "blank_banner": blank_paths,
        "random_text": random_paths,
        "small_bottom_text": small_paths,
    }

    all_summaries = {}

    print("\n=== run artifact controls ===")
    for cname, paths in conditions.items():
        print(f"\n--- condition: {cname} ---")
        cond_bridge_chunks = []
        cond_norm_chunks = []

        for t in range(num_frames):
            seed = 1000 + t
            actions, norm_actions = run_predict(
                vla=vla,
                image_path=paths[t],
                instruction=instruction,
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

    # 读取上一轮 overlay_same / visual_centric / wrong_image_text，方便之后解释。
    previous_summary = None
    prev_path = previous_dir / "summary.json"
    if prev_path.exists():
        previous_summary = json.loads(prev_path.read_text(encoding="utf-8"))

    final = {
        "dataset": "lerobot/droid_100",
        "episode": meta["episode"],
        "instruction": instruction,
        "random_text": random_text,
        "num_frames": num_frames,
        "clean_source_dir": str(clean_dir),
        "previous_all_conditions_summary": str(prev_path) if prev_path.exists() else None,
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

    print("\n=== final artifact-control summary ===")
    for cname, s in all_summaries.items():
        print(
            cname,
            "mean_norm_l2_6d=", round(s["mean_normalized_l2_6d"], 4),
            "median_norm_l2_6d=", round(s["median_normalized_l2_6d"], 4),
            "max_norm_l2_6d=", round(s["max_normalized_l2_6d"], 4),
            "mean_bridge_l2_6d=", round(s["mean_bridge_l2_6d"], 4),
            "grip_change_rate=", round(s["gripper_change_rate"], 4),
        )

    if previous_summary is not None:
        print("\n=== previous key conditions for comparison ===")
        for cname in ["overlay_same", "visual_centric", "wrong_image_text", "prompt_only", "wrong_language"]:
            if cname in previous_summary["conditions"]:
                s = previous_summary["conditions"][cname]
                print(
                    cname,
                    "mean_norm_l2_6d=", round(s["mean_normalized_l2_6d"], 4),
                    "mean_bridge_l2_6d=", round(s["mean_bridge_l2_6d"], 4),
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
