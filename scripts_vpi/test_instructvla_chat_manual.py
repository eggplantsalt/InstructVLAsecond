
import argparse
import torch
from PIL import Image
import numpy as np

from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load_vla


def patch_runtime_configs(model):
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


def build_multimodal_inputs(vlm, input_ids, pixel_values):
    assert vlm.img_context_token_id is not None, "img_context_token_id is None"

    vit_embeds = vlm.extract_feature(pixel_values)

    input_embeds = vlm.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape

    flat_embeds = input_embeds.reshape(B * N, C)
    flat_ids = input_ids.reshape(B * N)

    selected = flat_ids == vlm.img_context_token_id
    num_selected = int(selected.sum().item())

    flat_vit = vit_embeds.reshape(-1, C).to(
        flat_embeds.device,
        dtype=flat_embeds.dtype,
    )

    print("[DEBUG] input_ids shape:", tuple(input_ids.shape))
    print("[DEBUG] input_embeds shape:", tuple(input_embeds.shape))
    print("[DEBUG] vit_embeds shape:", tuple(vit_embeds.shape))
    print("[DEBUG] selected image tokens:", num_selected)
    print("[DEBUG] flattened vit tokens:", flat_vit.shape[0])

    if num_selected != flat_vit.shape[0]:
        raise RuntimeError(
            f"Image token mismatch: selected={num_selected}, vit_tokens={flat_vit.shape[0]}"
        )

    flat_embeds[selected] = flat_vit
    input_embeds = flat_embeds.reshape(B, N, C)
    return input_embeds


@torch.no_grad()
def manual_greedy_generate(vlm, input_ids, attention_mask, pixel_values, tokenizer, max_new_tokens=80):
    input_embeds = build_multimodal_inputs(vlm, input_ids, pixel_values)

    cur_embeds = input_embeds
    cur_attention_mask = attention_mask
    generated_new_tokens = []

    eos_ids = set()
    for x in [
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    ]:
        if x is not None:
            eos_ids.add(int(x))

    print("[DEBUG] eos_ids:", eos_ids)

    embed_layer = vlm.language_model.get_input_embeddings()

    for step in range(max_new_tokens):
        outputs = vlm.language_model(
            inputs_embeds=cur_embeds,
            attention_mask=cur_attention_mask,
            use_cache=False,
            return_dict=True,
        )

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        token_id = int(next_token[0, 0].item())
        generated_new_tokens.append(token_id)

        if step < 5:
            print(f"[DEBUG] step={step}, token_id={token_id}, token={tokenizer.decode([token_id])!r}")

        if token_id in eos_ids:
            print("[DEBUG] hit eos token, stop")
            break

        next_embed = embed_layer(next_token).to(cur_embeds.dtype)

        cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)

        one_mask = torch.ones(
            (cur_attention_mask.shape[0], 1),
            dtype=cur_attention_mask.dtype,
            device=cur_attention_mask.device,
        )
        cur_attention_mask = torch.cat([cur_attention_mask, one_mask], dim=1)

    return generated_new_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_path", default="./asset/teaser.png")
    parser.add_argument("--question", default="Can you describe the main idea of this image?")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=80)
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
    patch_runtime_configs(model)
    print("[DEBUG] runtime configs patched")

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

    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    pixel_values = inputs["pixel_values"].cuda()

    with torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
        new_token_ids = manual_greedy_generate(
            vlm=model.vlm,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            tokenizer=model.processor.tokenizer,
            max_new_tokens=args.max_new_tokens,
        )

    response = model.processor.tokenizer.decode(new_token_ids, skip_special_tokens=False)

    print("========== RESPONSE TOKENS ==========")
    print(new_token_ids)
    print("========== RESPONSE ==========")
    print(response)


if __name__ == "__main__":
    main()
