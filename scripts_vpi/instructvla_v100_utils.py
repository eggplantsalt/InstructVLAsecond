
"""
V100-compatible InstructVLA visual-language inference utilities.

Why this exists:
- The official InstructVLA / Eagle2 generate() path uses HuggingFace generation with
  inputs_embeds + attention_mask.
- On V100, FlashAttention2 is unavailable or unsuitable.
- Replacing flash_attention_2 with sdpa/eager can trigger attention-mask shape mismatch
  during generate(), e.g. 288 vs 576.
- This utility bypasses model.vlm.generate() and implements a small manual greedy
  decoding loop with use_cache=False.

Scope:
- Use this for visual-language chat / OCR / visual prompt diagnostic experiments.
- Do NOT assume this changes or fixes action prediction. Action pipelines should be
  handled separately unless they explicitly call model.vlm.generate().
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Dict, Any

import torch
from PIL import Image
import numpy as np


def get_torch_dtype(dtype: str) -> torch.dtype:
    """Convert a string dtype name to torch dtype."""
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def patch_runtime_configs(model, attn_implementation: str = "sdpa") -> None:
    """
    Runtime patch for V100 / non-flash-attn inference.

    It disables:
    - KV cache
    - sliding-window attention
    - stale flash-attn config paths

    This patch is intentionally conservative and only affects the currently loaded
    Python model object.
    """
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
                cfg.attn_implementation = attn_implementation
            if hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = attn_implementation

    # Patch common nested objects explicitly. Some PEFT/X-LoRA wrappers keep configs
    # at nested levels that are easy to miss.
    candidates = [
        getattr(model, "vlm", None),
        getattr(getattr(model, "vlm", None), "language_model", None),
        getattr(getattr(getattr(model, "vlm", None), "language_model", None), "base_model", None),
        getattr(
            getattr(getattr(getattr(model, "vlm", None), "language_model", None), "base_model", None),
            "model",
            None,
        ),
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
                cfg.attn_implementation = attn_implementation
            if hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = attn_implementation


def build_multimodal_inputs(vlm, input_ids: torch.Tensor, pixel_values: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    """
    Build multimodal input embeddings for Eagle/InstructVLA.

    This mirrors the key logic inside Eagle2's generate():
    1. Get language token embeddings.
    2. Extract visual features from pixel_values.
    3. Replace image-context-token embeddings with visual embeddings.

    Returns:
        input_embeds: Tensor with shape [B, N, C]
    """
    if getattr(vlm, "img_context_token_id", None) is None:
        raise RuntimeError("vlm.img_context_token_id is None. The model may not be initialized correctly.")

    vit_embeds = vlm.extract_feature(pixel_values)

    input_embeds = vlm.language_model.get_input_embeddings()(input_ids)
    batch_size, seq_len, hidden_dim = input_embeds.shape

    flat_embeds = input_embeds.reshape(batch_size * seq_len, hidden_dim)
    flat_ids = input_ids.reshape(batch_size * seq_len)

    selected = flat_ids == vlm.img_context_token_id
    selected_count = int(selected.sum().item())

    flat_vit = vit_embeds.reshape(-1, hidden_dim).to(
        flat_embeds.device,
        dtype=flat_embeds.dtype,
    )

    if verbose:
        print("[DEBUG] input_ids shape:", tuple(input_ids.shape))
        print("[DEBUG] input_embeds shape:", tuple(input_embeds.shape))
        print("[DEBUG] vit_embeds shape:", tuple(vit_embeds.shape))
        print("[DEBUG] selected image tokens:", selected_count)
        print("[DEBUG] flattened vit tokens:", flat_vit.shape[0])

    if selected_count == 0:
        raise RuntimeError("No image context tokens found in input_ids.")

    if selected_count != flat_vit.shape[0]:
        raise RuntimeError(
            f"Image token mismatch: selected={selected_count}, vit_tokens={flat_vit.shape[0]}"
        )

    flat_embeds[selected] = flat_vit
    return flat_embeds.reshape(batch_size, seq_len, hidden_dim)


@torch.no_grad()
def manual_greedy_generate(
    vlm,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    tokenizer,
    max_new_tokens: int = 128,
    verbose: bool = False,
) -> Tuple[Sequence[int], str]:
    """
    Manual greedy autoregressive generation for InstructVLA VLM chat.

    It avoids HuggingFace generate(), disables KV cache, and manually extends:
    - inputs_embeds
    - attention_mask

    Returns:
        new_token_ids, decoded_text
    """
    input_embeds = build_multimodal_inputs(
        vlm=vlm,
        input_ids=input_ids,
        pixel_values=pixel_values,
        verbose=verbose,
    )

    cur_embeds = input_embeds
    cur_attention_mask = attention_mask

    embed_layer = vlm.language_model.get_input_embeddings()

    eos_ids = set()
    for token_id in [
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    ]:
        if token_id is not None:
            eos_ids.add(int(token_id))

    if verbose:
        print("[DEBUG] eos_ids:", eos_ids)

    new_token_ids = []

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

        new_token_ids.append(token_id)

        if verbose and step < 10:
            print(
                f"[DEBUG] step={step}, token_id={token_id}, "
                f"token={tokenizer.decode([token_id])!r}"
            )

        if token_id in eos_ids:
            if verbose:
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

    decoded = tokenizer.decode(new_token_ids, skip_special_tokens=False)
    return new_token_ids, decoded


def make_messages(image: Image.Image, question: str, system_prompt: str = "You are a helpful assistant.") -> list:
    """Create an InstructVLA-compatible message list."""
    return [
        {"content": system_prompt},
        {
            "role": "user",
            "content": question,
            "image": [{"np_array": np.asarray(image)}],
        },
    ]


@torch.no_grad()
def ask_image(
    model,
    image_path: str,
    question: str,
    dtype: torch.dtype = torch.float16,
    max_new_tokens: int = 128,
    system_prompt: str = "You are a helpful assistant.",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Ask an image question using V100-compatible manual greedy generation.

    Example:
        result = ask_image(
            model=model,
            image_path="data_vpi/attack/sample.png",
            question="What text is written in the image?",
        )
        print(result["response"])
    """
    patch_runtime_configs(model)

    image = Image.open(image_path).convert("RGB")
    messages = make_messages(image=image, question=question, system_prompt=system_prompt)

    inputs = model.processor.prepare_input(dict(prompt=messages))

    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    pixel_values = inputs["pixel_values"].cuda()

    with torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
        token_ids, response = manual_greedy_generate(
            vlm=model.vlm,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            tokenizer=model.processor.tokenizer,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
        )

    return {
        "response": response,
        "token_ids": list(token_ids),
        "image_path": image_path,
        "question": question,
    }
