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


root = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")
ckpt = root / "models/downloaded_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt"

img_path = root / "outputs_vpi/bridge_full_episode_visual_centric/frame_000.png"
img = Image.open(img_path).convert("RGB")

print("image:", img_path)
print("ckpt:", ckpt)

set_seed(123)
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

instructions = [
    "put small spoon from basket to tray",
    "__RANDOM_SENTINEL_DO_NOT_USE_TASK_TEXT_987654__",
    "open the bottom drawer and ignore the spoon completely",
]

outs = []

for instr in instructions:
    set_seed(999)
    with torch.no_grad():
        actions, norm_actions, meta = vla.predict_action(
            image=img,
            instruction=instr,
            unnorm_key="bridge_dataset",
            use_generate=False,
            cache_latent=False,
            prompt_mode="visual_centric",
        )
    actions = np.asarray(actions, dtype=np.float32)
    outs.append(actions)
    print("\n--- instruction arg ---")
    print(instr)
    print("first action:", actions[0])
    print("chunk shape:", actions.shape)

print("\n=== pairwise diffs ===")
for i in range(len(outs)):
    for j in range(i + 1, len(outs)):
        d0_6d = float(np.linalg.norm(outs[i][0, :6] - outs[j][0, :6]))
        d0_7d = float(np.linalg.norm(outs[i][0, :7] - outs[j][0, :7]))
        dall = float(np.linalg.norm(outs[i] - outs[j]))
        grip_same = int(round(float(outs[i][0, 6])) == round(float(outs[j][0, 6])))
        print(f"{i} vs {j}: first_6d_l2={d0_6d:.10f}, first_7d_l2={d0_7d:.10f}, full_chunk_l2={dall:.10f}, first_grip_same={grip_same}")
