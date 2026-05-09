import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load_vla

try:
    from scripts_vpi.instructvla_v100_utils import patch_runtime_configs
except Exception:
    patch_runtime_configs = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_once(vla, img, instr, use_generate, seed):
    set_seed(seed)
    vla.last_response = None
    vla.latent = None
    vla.run_index = 0

    with torch.no_grad():
        actions, _, _ = vla.predict_action(
            image=img,
            instruction=instr,
            unnorm_key="bridge_dataset",
            use_generate=use_generate,
            cache_latent=False,
            prompt_mode="visual_centric",
        )

    actions = np.asarray(actions, dtype=np.float32)
    response = str(getattr(vla, "last_response", ""))
    return actions, response


def print_pairwise(outs, responses=None):
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            d0_6d = float(np.linalg.norm(outs[i][0, :6] - outs[j][0, :6]))
            d0_7d = float(np.linalg.norm(outs[i][0, :7] - outs[j][0, :7]))
            dall = float(np.linalg.norm(outs[i] - outs[j]))
            msg = (
                f"{i} vs {j}: "
                f"first_6d_l2={d0_6d:.12f}, "
                f"first_7d_l2={d0_7d:.12f}, "
                f"full_chunk_l2={dall:.12f}"
            )
            if responses is not None:
                msg += f", same_response={responses[i] == responses[j]}"
            print(msg)


def main():
    root = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")
    ckpt = root / "models/downloaded_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt"

    # 用原图，不用带字图。这样可以排除从图片中文字读到任务的可能。
    img_path = root / "outputs_vpi/bridge_full_episode/frame_000.png"
    img = Image.open(img_path).convert("RGB")

    instructions = [
        "put small spoon from basket to tray",
        "__RANDOM_SENTINEL_DO_NOT_USE_TASK_TEXT_987654__",
        "open the bottom drawer and ignore the spoon completely",
    ]

    print("=== setup ===")
    print("ckpt:", ckpt)
    print("image:", img_path)
    print("prompt_mode: visual_centric")
    print("instructions:")
    for i, x in enumerate(instructions):
        print(f"  {i}: {x}")

    set_seed(123)
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

    print("\n\n==============================")
    print("A. use_generate=False")
    print("==============================")

    outs = []
    for instr in instructions:
        actions, response = run_once(
            vla=vla,
            img=img,
            instr=instr,
            use_generate=False,
            seed=999,
        )
        outs.append(actions)

        print("\n--- instruction arg ---")
        print(instr)
        print("first action:", actions[0])
        print("response:", response)

    print("\n=== pairwise diffs, use_generate=False ===")
    print_pairwise(outs)

    print("\n\n==============================")
    print("B. use_generate=True")
    print("==============================")

    outs_reason = []
    responses = []
    for instr in instructions:
        actions, response = run_once(
            vla=vla,
            img=img,
            instr=instr,
            use_generate=True,
            seed=999,
        )
        outs_reason.append(actions)
        responses.append(response)

        print("\n--- instruction arg ---")
        print(instr)
        print("response:", response)
        print("first action:", actions[0])

    print("\n=== pairwise diffs, use_generate=True ===")
    print_pairwise(outs_reason, responses)

    print("\n\n=== interpretation ===")
    print("If all pairwise L2 values are 0.0 or numerically tiny,")
    print("then instruction arg is ignored under prompt_mode='visual_centric'.")
    print("If use_generate=True also has same_response=True,")
    print("then reasoning text also does not depend on instruction arg.")


if __name__ == "__main__":
    main()
