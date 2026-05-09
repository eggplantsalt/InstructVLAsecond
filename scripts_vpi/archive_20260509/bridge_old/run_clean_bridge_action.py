import json
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load_vla

# Optional: reuse the V100 runtime patch we previously wrote for VLMEvalKit/VQA.
try:
    from scripts_vpi.instructvla_v100_utils import patch_runtime_configs
except Exception:
    patch_runtime_configs = None


def main():
    root = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")

    ckpt = root / "models/downloaded_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt"

    image_path = root / "outputs_vpi/bridge_debug/frame_000.png"
    gt_json = root / "outputs_vpi/bridge_debug/first_10_steps.json"

    instruction = "put small spoon from basket to tray"

    out_dir = root / "outputs_vpi/clean_action_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== paths ===")
    print("ckpt:", ckpt, ckpt.exists())
    print("image:", image_path, image_path.exists())
    print("gt_json:", gt_json, gt_json.exists())
    print("out_dir:", out_dir)

    assert ckpt.exists(), f"Missing checkpoint: {ckpt}"
    assert image_path.exists(), f"Missing image: {image_path}"
    assert gt_json.exists(), f"Missing gt json: {gt_json}"

    print("\n=== cuda ===")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())

    print("\n=== load gt actions ===")
    gt_data = json.loads(gt_json.read_text(encoding="utf-8"))
    gt_actions = np.array(gt_data["actions"], dtype=np.float32)
    print("gt_actions shape:", gt_actions.shape)
    print("gt first action:", gt_actions[0])

    print("\n=== load model ===")
    vla = load_vla(
        ckpt,
        load_for_training=False,
        action_dim=7,
        future_action_window_size=15,
        # Use the model/default config for past_action_window_size unless explicitly needed.
    )

    if patch_runtime_configs is not None:
        print("Applying patch_runtime_configs...")
        patch_runtime_configs(vla)

    vla = vla.to("cuda").eval().to(torch.float16)

    print("\n=== run predict_action: clean baseline ===")
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

    print("\n=== outputs ===")
    print("pred actions shape:", actions.shape)
    print("normalized actions shape:", normalized_actions.shape)
    print("meta_feature shape:", tuple(meta_feature.shape))
    print("pred first action:", actions[0])
    print("normalized first action:", normalized_actions[0])

    n = min(len(actions), len(gt_actions))
    diff = actions[:n] - gt_actions[:n]
    l2_per_step = np.linalg.norm(diff, axis=1)

    print("\n=== comparison ===")
    print("compare steps:", n)
    print("first-step L2:", float(l2_per_step[0]))
    print("mean L2 over compared steps:", float(l2_per_step.mean()))
    print("l2_per_step:", l2_per_step.tolist())

    np.save(out_dir / "pred_actions.npy", actions)
    np.save(out_dir / "pred_normalized_actions.npy", normalized_actions)
    np.save(out_dir / "gt_actions_first10.npy", gt_actions)

    summary = {
        "image_path": str(image_path),
        "instruction": instruction,
        "unnorm_key": "bridge_dataset",
        "use_generate": False,
        "prompt_mode": "default",
        "pred_actions_shape": list(actions.shape),
        "normalized_actions_shape": list(normalized_actions.shape),
        "gt_actions_shape": list(gt_actions.shape),
        "pred_first_action": actions[0].tolist(),
        "normalized_first_action": normalized_actions[0].tolist(),
        "gt_first_action": gt_actions[0].tolist(),
        "compare_steps": int(n),
        "first_step_l2": float(l2_per_step[0]),
        "mean_l2": float(l2_per_step.mean()),
        "l2_per_step": l2_per_step.tolist(),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved outputs to:", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n========== ERROR ==========")
        traceback.print_exc()
        raise
