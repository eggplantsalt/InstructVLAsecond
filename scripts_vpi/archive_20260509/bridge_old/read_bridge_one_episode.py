import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json
import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds

# Avoid TF trying to grab all GPU memory.
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Could not set memory growth:", repr(e))

builder_dir = Path("datasets/VLA_Instruction_Tuning/bridge_dataset/1.0.0")
print("BUILDER_DIR:", builder_dir.resolve())
print("EXISTS:", builder_dir.exists())
print("FILES:")
for p in sorted(builder_dir.glob("*")):
    print(" ", p.name, round(p.stat().st_size / 1024 / 1024, 2), "MB")

print("\nLoading TFDS builder...")
builder = tfds.builder_from_directory(builder_dir=str(builder_dir))

print("BUILDER NAME:", builder.name)
print("VERSION:", builder.version)
print("SPLITS:", builder.info.splits)

# Try to read only the first episode.
print("\nReading first episode...")
ds = builder.as_dataset(split="train[:1]")
episode = next(iter(ds))

print("\n=== episode metadata ===")
for k, v in episode["episode_metadata"].items():
    try:
        if hasattr(v, "numpy"):
            vv = v.numpy()
            if isinstance(vv, bytes):
                vv = vv.decode("utf-8")
            print(k, "=", vv)
        else:
            print(k, "=", v)
    except Exception as e:
        print(k, "ERROR:", repr(e))

steps = list(episode["steps"])
print("\nNUM_STEPS:", len(steps))

first = steps[0]
print("\n=== first step top keys ===")
print(first.keys())

print("\n=== first step observation keys ===")
print(first["observation"].keys())

print("\n=== first step available fields ===")
for k, v in first.items():
    if isinstance(v, dict):
        print(k, "DICT_KEYS", list(v.keys()))
    else:
        try:
            print(k, "shape=", v.shape, "dtype=", v.dtype)
        except Exception:
            print(k, type(v))

# Language
print("\n=== language instruction ===")
try:
    instr = first["language_instruction"].numpy().decode("utf-8")
    print(instr)
except Exception as e:
    print("language_instruction ERROR:", repr(e))
    instr = None

# Image
print("\n=== image ===")
try:
    img = first["observation"]["image_0"].numpy()
    print("image_0 shape:", img.shape, "dtype:", img.dtype, "min:", img.min(), "max:", img.max())
    out_dir = Path("outputs_vpi/bridge_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_dir / "first_frame.png")
    print("saved:", out_dir / "first_frame.png")
except Exception as e:
    print("image_0 ERROR:", repr(e))

# Action
print("\n=== action ===")
try:
    action = first["action"].numpy()
    print("action shape:", action.shape, "dtype:", action.dtype)
    print("action:", action)
except Exception as e:
    print("action ERROR:", repr(e))

# Save first few frames and actions.
try:
    out_dir = Path("outputs_vpi/bridge_debug")
    actions = []
    instructions = []
    for i, step in enumerate(steps[:10]):
        img_i = step["observation"]["image_0"].numpy()
        Image.fromarray(img_i).save(out_dir / f"frame_{i:03d}.png")
        actions.append(step["action"].numpy().tolist())
        instructions.append(step["language_instruction"].numpy().decode("utf-8"))

    with open(out_dir / "first_10_steps.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_steps": len(steps),
                "instructions": instructions,
                "actions": actions,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("\nsaved first 10 frames/actions to:", out_dir)
except Exception as e:
    print("save debug ERROR:", repr(e))
