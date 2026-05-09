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


def load_model(ckpt):
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


root = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")
ckpt = root / "models/downloaded_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt"

clean_img = root / "outputs_vpi/bridge_full_episode/frame_000.png"
visual_img = root / "outputs_vpi/bridge_full_episode_visual_centric/frame_000.png"

episode = __import__("json").loads((root / "outputs_vpi/bridge_full_episode/episode.json").read_text())
instruction = episode["instructions"][0]

src = (root / "vla/instructvla_eagle_dual_sys_v2_meta_query_v2.py").read_text()
if 'visual_centric' in src:
    visual_mode = "visual_centric"
else:
    visual_mode = "image_text_primary"

print("ckpt:", ckpt)
print("instruction:", instruction)
print("visual_mode:", visual_mode)

set_seed(42)
vla = load_model(ckpt)

if patch_runtime_configs is not None:
    print("Applying patch_runtime_configs...")
    patch_runtime_configs(vla)

vla = vla.to("cuda").eval().to(torch.float16)

conditions = [
    ("clean_reason", clean_img, instruction, "default"),
    ("visual_reason", visual_img, "__NO_TASK_TEXT_PROVIDED__", visual_mode),
    ("prompt_only_reason", clean_img, "__NO_TASK_TEXT_PROVIDED__", visual_mode),
]

for name, img_path, instr, mode in conditions:
    print("\n==============================")
    print("condition:", name)
    print("img:", img_path)
    print("instruction_arg:", instr)
    print("prompt_mode:", mode)

    image = Image.open(img_path).convert("RGB")

    # 强制每个 condition 都重新 reasoning，避免复用 last_response。
    vla.last_response = None
    vla.latent = None
    vla.run_index = 0

    set_seed(123)
    with torch.no_grad():
        actions, norm_actions, meta = vla.predict_action(
            image=image,
            instruction=instr,
            unnorm_key="bridge_dataset",
            use_generate=True,
            cache_latent=False,
            prompt_mode=mode,
        )

    actions = np.asarray(actions, dtype=np.float32)
    print("actions shape:", actions.shape)
    print("first action:", actions[0])
    print("last_response:", getattr(vla, "last_response", None))
