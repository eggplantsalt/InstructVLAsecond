import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds

builder_dir = Path("datasets/VLA_Instruction_Tuning/bridge_dataset/1.0.0")
builder = tfds.builder_from_directory(builder_dir=str(builder_dir))
episode = next(iter(builder.as_dataset(split="train[:1]")))
steps = list(episode["steps"])

out_dir = Path("outputs_vpi/bridge_full_episode")
out_dir.mkdir(parents=True, exist_ok=True)

actions = []
instructions = []

for i, step in enumerate(steps):
    img = step["observation"]["image_0"].numpy()
    Image.fromarray(img).save(out_dir / f"frame_{i:03d}.png")
    actions.append(step["action"].numpy().tolist())
    instructions.append(step["language_instruction"].numpy().decode("utf-8"))

meta = {}
for k, v in episode["episode_metadata"].items():
    vv = v.numpy()
    if isinstance(vv, bytes):
        vv = vv.decode("utf-8")
    elif hasattr(vv, "item"):
        vv = vv.item()
    meta[k] = vv

(out_dir / "episode.json").write_text(
    json.dumps({
        "num_steps": len(steps),
        "metadata": meta,
        "instructions": instructions,
        "actions": actions,
    }, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("saved:", out_dir)
print("num_steps:", len(steps))
print("instruction:", instructions[0])
print("gripper:", [a[6] for a in actions])
