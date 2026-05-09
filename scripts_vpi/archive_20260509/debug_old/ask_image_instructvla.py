
import argparse
import json

import torch

from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load_vla
from scripts_vpi.instructvla_v100_utils import ask_image, get_torch_dtype, patch_runtime_configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json_output", default=None)
    args = parser.parse_args()

    dtype = get_torch_dtype(args.dtype)

    print("Loading model from:", args.model_path)
    print("Using dtype:", dtype)

    model = load_vla(args.model_path, stage="stage2").eval().to(dtype).cuda()
    patch_runtime_configs(model)

    result = ask_image(
        model=model,
        image_path=args.image_path,
        question=args.question,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )

    print("========== RESPONSE ==========")
    print(result["response"])

    if args.json_output is not None:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("Saved JSON output to:", args.json_output)


if __name__ == "__main__":
    main()
