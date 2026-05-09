import argparse
from pathlib import Path

import torch
from PIL import Image
import numpy as np

from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load_vla


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_path", default="./asset/teaser.png")
    parser.add_argument("--question", default="Can you describe the main idea of this image?")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print("Loading model from:", args.model_path)
    print("Using dtype:", dtype)

    model = load_vla(args.model_path, stage="stage2").eval().to(dtype).cuda()

    # Hard-disable Qwen2 sliding-window attention for V100 / non-flash-attn backends.
    # Without this, Qwen2 may build a [seq, seq] mask while KV length becomes larger,
    # causing 288 vs 576 attention-mask mismatch.
    for module in model.modules():
        for cfg_name in ["config", "generation_config"]:
            cfg = getattr(module, cfg_name, None)
            if cfg is None:
                continue
            if hasattr(cfg, "use_cache"):
                cfg.use_cache = False
            if hasattr(cfg, "use_sliding_window"):
                cfg.use_sliding_window = False
            if hasattr(cfg, "sliding_window"):
                cfg.sliding_window = None
            if hasattr(cfg, "max_window_layers"):
                cfg.max_window_layers = 0
            if hasattr(cfg, "attn_implementation"):
                cfg.attn_implementation = "sdpa"
            if hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = "sdpa"

    # Also patch common nested model configs explicitly.
    candidates = [
        getattr(model, "vlm", None),
        getattr(getattr(model, "vlm", None), "language_model", None),
        getattr(getattr(getattr(model, "vlm", None), "language_model", None), "base_model", None),
        getattr(getattr(getattr(getattr(model, "vlm", None), "language_model", None), "base_model", None), "model", None),
    ]
    for obj in candidates:
        if obj is None:
            continue
        for cfg_name in ["config", "generation_config"]:
            cfg = getattr(obj, cfg_name, None)
            if cfg is None:
                continue
            if hasattr(cfg, "use_cache"):
                cfg.use_cache = False
            if hasattr(cfg, "use_sliding_window"):
                cfg.use_sliding_window = False
            if hasattr(cfg, "sliding_window"):
                cfg.sliding_window = None
            if hasattr(cfg, "max_window_layers"):
                cfg.max_window_layers = 0
            if hasattr(cfg, "attn_implementation"):
                cfg.attn_implementation = "sdpa"
            if hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = "sdpa"

    print("[DEBUG] finished disabling cache/sliding-window configs")

    # Disable KV cache to avoid Qwen2 attention-mask mismatch on V100 / non-flash-attn backends.
    for module in model.modules():
        if hasattr(module, "config") and hasattr(module.config, "use_cache"):
            module.config.use_cache = False
        if hasattr(module, "generation_config") and hasattr(module.generation_config, "use_cache"):
            module.generation_config.use_cache = False

    image = Image.open(args.image_path).convert("RGB")

    messages = [
        {"content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": args.question,
            "image": [{"np_array": np.asarray(image)}],
        },
    ]

    inputs = model.processor.prepare_input(dict(prompt=messages))

    with torch.no_grad():
        with torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            output = model.vlm.generate(
                input_ids=inputs["input_ids"].cuda(),
                attention_mask=inputs["attention_mask"].cuda(),
                pixel_values=inputs["pixel_values"].cuda(),
                max_new_tokens=200,
                output_hidden_states=False,
                use_cache=False,
                do_sample=False,
            )

    response = model.processor.tokenizer.decode(output[0])
    print("========== RESPONSE ==========")
    print(response)


if __name__ == "__main__":
    main()
