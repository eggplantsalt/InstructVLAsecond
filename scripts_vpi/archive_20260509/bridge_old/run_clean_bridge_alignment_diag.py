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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cont_l2(a, b):
    return float(np.linalg.norm(np.asarray(a[:6], dtype=np.float32) - np.asarray(b[:6], dtype=np.float32)))


def grip_match(a, b):
    return int(round(float(a[6])) == round(float(b[6])))


def summarize_one_step(pred_chunks, gt):
    pred0 = pred_chunks[:, 0, :]
    n = min(len(pred0), len(gt))
    cont = [cont_l2(pred0[t], gt[t]) for t in range(n)]
    grip = [grip_match(pred0[t], gt[t]) for t in range(n)]
    return {
        "n": n,
        "mean_cont_l2": float(np.mean(cont)),
        "gripper_acc": float(np.mean(grip)),
        "per_timestep": [
            {
                "t": t,
                "cont_l2": cont[t],
                "pred_g": float(pred0[t, 6]),
                "gt_g": float(gt[t, 6]),
                "grip_match": grip[t],
            }
            for t in range(n)
        ],
    }


def summarize_horizon(pred_chunks, gt, max_h=15):
    out = []
    T = len(pred_chunks)
    for h in range(max_h + 1):
        cont = []
        grip = []
        for t in range(T):
            j = t + h
            if j < len(gt):
                cont.append(cont_l2(pred_chunks[t, h], gt[j]))
                grip.append(grip_match(pred_chunks[t, h], gt[j]))
        if cont:
            out.append({
                "horizon": h,
                "pairs": len(cont),
                "mean_cont_l2": float(np.mean(cont)),
                "gripper_acc": float(np.mean(grip)),
            })
    return out


def summarize_deployment_like(pred_chunks, gt, k):
    cont = []
    grip = []
    details = []

    T = len(gt)
    for start in range(0, T, k):
        if start >= len(pred_chunks):
            break

        chunk = pred_chunks[start]
        steps = min(k, len(chunk), T - start)

        for i in range(steps):
            p = chunk[i]
            g = gt[start + i]
            c = cont_l2(p, g)
            m = grip_match(p, g)
            cont.append(c)
            grip.append(m)
            details.append({
                "frame_used": start,
                "chunk_index": i,
                "gt_timestep": start + i,
                "cont_l2": c,
                "pred_g": float(p[6]),
                "gt_g": float(g[6]),
                "grip_match": m,
            })

    return {
        "use_length": k,
        "pairs": len(cont),
        "mean_cont_l2": float(np.mean(cont)) if cont else None,
        "gripper_acc": float(np.mean(grip)) if grip else None,
        "details": details,
    }


def main():
    root = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")
    ckpt = root / "models/downloaded_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt"
    ep_dir = root / "outputs_vpi/bridge_full_episode"
    ep_json = ep_dir / "episode.json"

    out_dir = root / "outputs_vpi/clean_alignment_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    episode = json.loads(ep_json.read_text(encoding="utf-8"))
    gt = np.asarray(episode["actions"], dtype=np.float32)
    instruction = episode["instructions"][0]
    frame_paths = sorted(ep_dir.glob("frame_*.png"))

    print("ckpt:", ckpt, ckpt.exists())
    print("episode:", ep_json, ep_json.exists())
    print("num frames:", len(frame_paths))
    print("gt shape:", gt.shape)
    print("instruction:", instruction)
    print("gt gripper:", gt[:, 6].tolist())

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
        patch_runtime_configs(vla)

    vla = vla.to("cuda").eval().to(torch.float16)

    pred_chunks = []

    print("\n=== predict all frames ===")
    for t, p in enumerate(frame_paths):
        print(f"predict {t:03d}: {p.name}")
        set_seed(42 + t)
        img = Image.open(p).convert("RGB")

        with torch.no_grad():
            actions, normalized_actions, meta_feature = vla.predict_action(
                image=img,
                instruction=instruction,
                unnorm_key="bridge_dataset",
                use_generate=False,
                cache_latent=False,
                prompt_mode="default",
            )

        pred_chunks.append(np.asarray(actions, dtype=np.float32))

    pred_chunks = np.stack(pred_chunks, axis=0)
    np.save(out_dir / "pred_chunks.npy", pred_chunks)
    np.save(out_dir / "gt_actions.npy", gt)

    summary = {
        "instruction": instruction,
        "pred_chunks_shape": list(pred_chunks.shape),
        "gt_shape": list(gt.shape),
        "gt_gripper": gt[:, 6].tolist(),
        "one_step": summarize_one_step(pred_chunks, gt),
        "horizon": summarize_horizon(pred_chunks, gt, max_h=15),
        "deployment_like": [
            summarize_deployment_like(pred_chunks, gt, k)
            for k in [1, 2, 3, 4, 5, 8, 16]
        ],
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== one-step ===")
    print(json.dumps(summary["one_step"], indent=2)[:3000])

    print("\n=== horizon summary ===")
    for r in summary["horizon"]:
        print(r)

    print("\n=== deployment-like summary ===")
    for r in summary["deployment_like"]:
        print({
            "use_length": r["use_length"],
            "pairs": r["pairs"],
            "mean_cont_l2": r["mean_cont_l2"],
            "gripper_acc": r["gripper_acc"],
        })

    print("\nsaved:", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n========== ERROR ==========")
        traceback.print_exc()
        raise
