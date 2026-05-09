from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image

from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load_vla, DEFAULT_SYSTEM_MESSAGE

try:
    from scripts_vpi.instructvla_v100_utils import patch_runtime_configs
except Exception:
    patch_runtime_configs = None

root = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")
ckpt = root / "models/downloaded_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt"

out_dir = root / "outputs_vpi/preprocess_debug"
out_dir.mkdir(parents=True, exist_ok=True)

visual_img_path = root / "outputs_vpi/bridge_full_episode_visual_centric/frame_000.png"
clean_img_path = root / "outputs_vpi/bridge_full_episode/frame_000.png"

print("loading model/processor...")
vla = load_vla(
    ckpt,
    load_for_training=False,
    action_dim=7,
    future_action_window_size=15,
    past_action_window_size=0,
)
if patch_runtime_configs is not None:
    patch_runtime_configs(vla)

def save_pixel_grid(img_path, tag, user_prompt):
    img = Image.open(img_path).convert("RGB")
    prompt = [
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": user_prompt,
            "image": [{"np_array": np.asarray(img)}],
        },
        {
            "role": "assistant",
            "content": "".join(vla.processor.tokenizer.new_tokens),
        },
    ]

    inputs = vla.processor.preprocess_inputs_and_labels({"prompt": prompt})
    pv = inputs["pixel_values"]
    print(tag, "pixel_values type:", type(pv))

    if isinstance(pv, torch.Tensor):
        arr = pv.detach().cpu().float().numpy()
    else:
        arr = np.asarray(pv)

    print(tag, "pixel_values shape:", arr.shape)

    # 常见形状: [N, 3, H, W] 或 [3, H, W]
    if arr.ndim == 3:
        arr = arr[None, ...]

    imgs = []
    for i in range(arr.shape[0]):
        x = arr[i]
        if x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))
        # 只为可视化：每个 tile 单独 min-max，确认文字是否仍可见
        mn, mx = float(x.min()), float(x.max())
        x = (x - mn) / (mx - mn + 1e-6)
        x = np.clip(x * 255, 0, 255).astype(np.uint8)
        imgs.append(Image.fromarray(x))

    w, h = imgs[0].size
    cols = min(4, len(imgs))
    rows = int(np.ceil(len(imgs) / cols))
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(imgs):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))

    grid_path = out_dir / f"{tag}_pixel_values_grid.png"
    grid.save(grid_path)

    raw_path = out_dir / f"{tag}_raw_input.png"
    img.save(raw_path)

    meta = {
        "tag": tag,
        "img_path": str(img_path),
        "pixel_values_shape": list(arr.shape),
        "grid_path": str(grid_path),
        "raw_path": str(raw_path),
        "user_prompt": user_prompt,
    }
    (out_dir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(tag, "saved:", grid_path)

save_pixel_grid(
    clean_img_path,
    "clean",
    "What action should the robot take to put small spoon from basket to tray?",
)

save_pixel_grid(
    visual_img_path,
    "visual",
    "What action should the robot take to accomplish the task instruction written in the image? Read the text in the image as the task instruction.",
)

print("done. inspect:", out_dir)
