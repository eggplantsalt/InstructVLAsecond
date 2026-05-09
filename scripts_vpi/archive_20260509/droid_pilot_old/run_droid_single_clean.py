import json
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

    episode_dir = root / "outputs_vpi/droid100_debug/episode_000000"
    meta_path = episode_dir / "episode_meta.json"
    actions_path = episode_dir / "actions.npy"
    states_path = episode_dir / "states.npy"

    out_dir = root / "outputs_vpi/droid100_clean_single"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== paths ===")
    print("ckpt:", ckpt, ckpt.exists())
    print("episode_dir:", episode_dir, episode_dir.exists())
    print("meta:", meta_path, meta_path.exists())
    print("actions:", actions_path, actions_path.exists())
    print("states:", states_path, states_path.exists())
    print("out_dir:", out_dir)

    assert ckpt.exists(), f"Missing checkpoint: {ckpt}"
    assert meta_path.exists(), f"Missing meta: {meta_path}"
    assert actions_path.exists(), f"Missing actions: {actions_path}"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    instruction = meta["instruction"]
    image_path = Path(meta["frame_paths"][0])

    gt_actions = np.load(actions_path)
    gt_states = np.load(states_path)

    print("\n=== sample ===")
    print("instruction:", instruction)
    print("image_path:", image_path, image_path.exists())
    print("gt_actions shape:", gt_actions.shape)
    print("gt_states shape:", gt_states.shape)
    print("gt first action:", gt_actions[0])
    print("gt first state:", gt_states[0])

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

    # 尝试打印模型里可能存在的 norm stats key，方便判断有没有 droid 相关 key。
    print("\n=== possible norm stats keys ===")
    for attr in ["norm_stats", "unnorm_stats", "action_stats", "dataset_statistics"]:
        obj = getattr(vla, attr, None)
        if isinstance(obj, dict):
            print(attr, "keys:", list(obj.keys()))
        elif obj is not None:
            print(attr, "type:", type(obj))

    image = Image.open(image_path).convert("RGB")

    # InstructVLA checkpoint 大概率没有 droid 的 unnorm stats。
    # 所以这里先尝试 droid 相关 key；失败后退回 bridge_dataset。
    # 对后续 visual-vs-clean 成对比较来说，关键是所有条件使用同一个 unnorm_key。
    candidate_unnorm_keys = [
        "droid",
        "droid_dataset",
        "droid_100",
        "bridge_dataset",
        None,
    ]

    last_err = None
    used_key = None
    result = None

    print("\n=== predict_action attempts ===")
    for key in candidate_unnorm_keys:
        try:
            print("trying unnorm_key:", key)
            with torch.no_grad():
                if key is None:
                    actions, normalized_actions, meta_feature = vla.predict_action(
                        image=image,
                        instruction=instruction,
                        use_generate=False,
                        cache_latent=False,
                        prompt_mode="default",
                    )
                else:
                    actions, normalized_actions, meta_feature = vla.predict_action(
                        image=image,
                        instruction=instruction,
                        unnorm_key=key,
                        use_generate=False,
                        cache_latent=False,
                        prompt_mode="default",
                    )
            used_key = key
            result = (actions, normalized_actions, meta_feature)
            print("[OK] succeeded with unnorm_key:", key)
            break
        except Exception as e:
            print("[FAIL]", key, repr(e))
            last_err = e

    if result is None:
        print("\nAll predict_action attempts failed.")
        raise last_err

    actions, normalized_actions, meta_feature = result
    actions = np.asarray(actions, dtype=np.float32)
    normalized_actions = np.asarray(normalized_actions, dtype=np.float32)

    print("\n=== output ===")
    print("used_unnorm_key:", used_key)
    print("pred actions shape:", actions.shape)
    print("normalized actions shape:", normalized_actions.shape)
    print("meta_feature shape:", tuple(meta_feature.shape) if hasattr(meta_feature, "shape") else type(meta_feature))
    print("pred first action:", actions[0])
    print("normalized first action:", normalized_actions[0])

    # 只做 sanity check，不把这个当成核心结论。
    gt0 = gt_actions[0]
    print("\n=== rough gt comparison, sanity only ===")
    print("gt first action:", gt0)
    print("full 7d L2 pred_first vs gt_first:", l2(actions[0], gt0))
    print("continuous 6d L2 pred_first vs gt_first:", l2(actions[0][:6], gt0[:6]))
    print("gripper pred/gt:", float(actions[0][6]), float(gt0[6]))

    np.save(out_dir / "pred_actions.npy", actions)
    np.save(out_dir / "pred_normalized_actions.npy", normalized_actions)
    np.save(out_dir / "gt_actions.npy", gt_actions)

    summary = {
        "dataset": "lerobot/droid_100",
        "episode": meta["episode"],
        "instruction": instruction,
        "image_path": str(image_path),
        "used_unnorm_key": used_key,
        "pred_actions_shape": list(actions.shape),
        "normalized_actions_shape": list(normalized_actions.shape),
        "gt_actions_shape": list(gt_actions.shape),
        "pred_first_action": actions[0].tolist(),
        "normalized_first_action": normalized_actions[0].tolist(),
        "gt_first_action": gt0.tolist(),
        "full_7d_l2_pred_gt_first": l2(actions[0], gt0),
        "continuous_6d_l2_pred_gt_first": l2(actions[0][:6], gt0[:6]),
        "pred_gripper": float(actions[0][6]),
        "gt_gripper": float(gt0[6]),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== saved ===")
    print("out_dir:", out_dir)
    print("summary:", out_dir / "summary.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n========== ERROR ==========")
        traceback.print_exc()
        raise
