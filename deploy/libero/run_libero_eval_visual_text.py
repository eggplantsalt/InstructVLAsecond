"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python Libero/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.getcwd())

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from PIL import Image, ImageDraw, ImageFont
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find Libero.robot
from deploy.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)

from deploy.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)



@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "instruct_vla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    unnorm_key: Optional[str] = None
    horizon: int = 8
    action_ensemble_horizon: Optional[int] = 8
    # image_size: list[int] = [224, 224]
    future_action_window_size: int = 7
    action_dim: int = 7
    use_bf16: bool = True
    action_ensemble = True
    adaptive_ensemble_alpha = 0.1
    retriever_path: str = None
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./Libero/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 42                                    # Random Seed (for reproducibility)
    use_length: int = 8

    #################################################################################################################
    # Visual-text pilot parameters
    #################################################################################################################
    eval_mode: str = "clean_langT_noText"             # clean_langT_noText | visualPrompt_noText | visualPrompt_textT | langT_textT | langT_conflictText
    visual_prompt: str = "Follow the instruction shown in the image."
    overlay_view: str = "both"                        # full | wrist | both
    num_tasks_limit: int = -1                         # -1 means all tasks
    task_ids: str = ""                                # comma-separated task ids, e.g. "5" or "1,5,7"; overrides num_tasks_limit
    overlay_font_size: int = 14
    overlay_max_chars_per_line: int = 42
    conflict_task_id: int = -1                    # for langT_conflictText: use this LIBERO task id as visual text; -1 means next task id
    conflict_text_override: str = ""              # optional direct visual conflict text; overrides conflict_task_id
    save_debug_frames: bool = True                # save first overlaid full/wrist observation frames for sanity check
    debug_frame_dir: str = "outputs_vpi/libero_visual_text_pilot/debug_frames"

    # fmt: on


def _image_to_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB"), "pil"
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB"), "array"


def _restore_image_type(pil_img, original_kind):
    if original_kind == "pil":
        return pil_img
    return np.asarray(pil_img)


def draw_small_bottom_text(image, text, font_size=14, margin=4, max_chars_per_line=42):
    """Draw a compact semi-transparent bottom text banner on a PIL image or numpy image."""
    if text is None or str(text).strip() == "":
        return image

    pil_img, original_kind = _image_to_pil(image)
    w, h = pil_img.size

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    words = str(text).split()
    lines = []
    cur = ""
    for word in words:
        candidate = word if cur == "" else cur + " " + word
        if len(candidate) <= max_chars_per_line:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if not lines:
        lines = [str(text)]

    # Keep at most two lines to avoid covering too much visual content.
    lines = lines[:2]

    dummy = ImageDraw.Draw(pil_img)
    line_heights = []
    max_line_w = 0
    for line in lines:
        bbox = dummy.textbbox((0, 0), line, font=font)
        max_line_w = max(max_line_w, bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    box_h = sum(line_heights) + margin * (len(lines) + 1)
    box_w = min(w, max_line_w + 2 * margin)
    x0 = max(0, (w - box_w) // 2)
    y0 = max(0, h - box_h)

    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([x0, y0, x0 + box_w, h], fill=(255, 255, 255, 210))

    y = y0 + margin
    for i, line in enumerate(lines):
        bbox = od.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        x = max(0, (w - line_w) // 2)
        od.text((x, y), line, font=font, fill=(0, 0, 0, 255))
        y += line_heights[i] + margin

    out = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")
    return _restore_image_type(out, original_kind)


def get_conflict_overlay_text(cfg, task_suite, task_id, task_description):
    """Return the competing visual instruction text T' for langT_conflictText."""
    override = str(getattr(cfg, "conflict_text_override", "")).strip()
    if override:
        return override

    num_tasks = task_suite.n_tasks
    conflict_task_id = int(getattr(cfg, "conflict_task_id", -1))

    if conflict_task_id < 0:
        conflict_task_id = (int(task_id) + 1) % num_tasks

    if conflict_task_id < 0 or conflict_task_id >= num_tasks:
        raise ValueError(f"conflict_task_id {conflict_task_id} out of range [0, {num_tasks - 1}]")

    conflict_text = task_suite.get_task(conflict_task_id).language

    if conflict_text == task_description:
        raise ValueError(
            f"Conflict text is identical to current task text. "
            f"task_id={task_id}, conflict_task_id={conflict_task_id}, text={conflict_text!r}"
        )

    return conflict_text



def get_model_instruction_for_mode(cfg, task_description):
    if cfg.eval_mode in ("clean_langT_noText", "langT_textT", "langT_conflictText"):
        return task_description
    if cfg.eval_mode in ("visualPrompt_noText", "visualPrompt_textT"):
        return cfg.visual_prompt
    raise ValueError(f"Unknown eval_mode: {cfg.eval_mode}")


def get_overlay_text_for_mode(cfg, task_description, conflict_overlay_text=None):
    if cfg.eval_mode in ("langT_textT", "visualPrompt_textT"):
        return task_description
    if cfg.eval_mode == "langT_conflictText":
        if conflict_overlay_text is None or str(conflict_overlay_text).strip() == "":
            raise ValueError("langT_conflictText requires non-empty conflict_overlay_text")
        return conflict_overlay_text
    if cfg.eval_mode in ("clean_langT_noText", "visualPrompt_noText"):
        return None
    raise ValueError(f"Unknown eval_mode: {cfg.eval_mode}")


def save_debug_observation_frames(cfg, run_id, task_id, episode_idx, img, wrist_img, overlay_text):
    """Save first overlaid full and wrist frames for visual sanity checking."""
    if not getattr(cfg, "save_debug_frames", True):
        return

    out_dir = Path(cfg.debug_frame_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_run_id = str(run_id).replace("/", "_")
    prefix = out_dir / f"{safe_run_id}--task={int(task_id):02d}--episode={int(episode_idx)+1:03d}"

    full_path = str(prefix) + "--full.png"
    wrist_path = str(prefix) + "--wrist.png"
    meta_path = str(prefix) + "--meta.txt"

    Image.fromarray(np.asarray(img).astype(np.uint8)).save(full_path)
    Image.fromarray(np.asarray(wrist_img).astype(np.uint8)).save(wrist_path)

    with open(meta_path, "w") as f:
        f.write(f"run_id: {run_id}\n")
        f.write(f"task_id: {task_id}\n")
        f.write(f"episode_idx: {episode_idx}\n")
        f.write(f"eval_mode: {cfg.eval_mode}\n")
        f.write(f"overlay_text: {overlay_text}\n")
        f.write(f"overlay_view: {cfg.overlay_view}\n")
        f.write(f"overlay_font_size: {cfg.overlay_font_size}\n")

    print(f"[debug] saved overlaid full frame: {full_path}")
    print(f"[debug] saved overlaid wrist frame: {wrist_path}")



def maybe_apply_overlay(cfg, img, wrist_img, overlay_text):
    if overlay_text is None:
        return img, wrist_img

    view = cfg.overlay_view.lower()
    if view not in ("full", "wrist", "both"):
        raise ValueError(f"overlay_view must be full, wrist, or both; got {cfg.overlay_view}")

    if view in ("full", "both"):
        img = draw_small_bottom_text(
            img,
            overlay_text,
            font_size=cfg.overlay_font_size,
            max_chars_per_line=cfg.overlay_max_chars_per_line,
        )
    if view in ("wrist", "both"):
        wrist_img = draw_small_bottom_text(
            wrist_img,
            overlay_text,
            font_size=cfg.overlay_font_size,
            max_chars_per_line=cfg.overlay_max_chars_per_line,
        )
    return img, wrist_img



def get_task_ids_to_eval(cfg, num_tasks_in_suite):
    """Return explicit task ids if cfg.task_ids is set; otherwise use num_tasks_limit from 0."""
    task_ids_str = str(getattr(cfg, "task_ids", "")).strip()
    if task_ids_str:
        ids = []
        for x in task_ids_str.split(","):
            x = x.strip()
            if not x:
                continue
            tid = int(x)
            if tid < 0 or tid >= num_tasks_in_suite:
                raise ValueError(f"task id {tid} out of range [0, {num_tasks_in_suite - 1}]")
            ids.append(tid)
        if not ids:
            raise ValueError("task_ids was provided but no valid ids were parsed")
        return ids

    n = num_tasks_in_suite if cfg.num_tasks_limit <= 0 else min(cfg.num_tasks_limit, num_tasks_in_suite)
    return list(range(n))



@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    ckpt_index = os.path.basename(cfg.pretrained_checkpoint)[:-3]
    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # [OpenVLA] Set action un-normalization key
        cfg.unnorm_key = cfg.task_suite_name
        model, server = get_model(cfg)
        server = None
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    elif cfg.model_family == "instruct_vla":
        # [OpenVLA] Set action un-normalization key
        cfg.unnorm_key = f"{cfg.task_suite_name}_no_noops"
        model, server = get_model(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{cfg.eval_mode}-{DATE_TIME}-{ckpt_index}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_ids_to_eval = get_task_ids_to_eval(cfg, num_tasks_in_suite)
    num_tasks_to_eval = len(task_ids_to_eval)
    print(f"Task suite: {cfg.task_suite_name}")
    print(f"Eval mode: {cfg.eval_mode}")
    print(f"Task ids to eval: {task_ids_to_eval}")
    print(f"Num tasks to eval: {num_tasks_to_eval} / {num_tasks_in_suite}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"Eval mode: {cfg.eval_mode}\n")
    log_file.write(f"Task ids to eval: {task_ids_to_eval}\n")
    log_file.write(f"Num tasks to eval: {num_tasks_to_eval} / {num_tasks_in_suite}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids_to_eval):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        model_instruction = get_model_instruction_for_mode(cfg, task_description)
        conflict_overlay_text = None
        if cfg.eval_mode == "langT_conflictText":
            conflict_overlay_text = get_conflict_overlay_text(cfg, task_suite, task_id, task_description)
        overlay_text = get_overlay_text_for_mode(cfg, task_description, conflict_overlay_text)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            print(f"Eval mode: {cfg.eval_mode}")
            print(f"Model instruction: {model_instruction}")
            print(f"Overlay text: {overlay_text}")
            if cfg.eval_mode == "langT_conflictText":
                print(f"Conflict overlay candidate: {conflict_overlay_text}")
            log_file.write(f"\nTask: {task_description}\n")
            log_file.write(f"Eval mode: {cfg.eval_mode}\n")
            log_file.write(f"Model instruction: {model_instruction}\n")
            log_file.write(f"Overlay text: {overlay_text}\n")
            if cfg.eval_mode == "langT_conflictText":
                log_file.write(f"Conflict overlay candidate: {conflict_overlay_text}\n")

            # Reset environment
            env.reset()
            server.reset(model_instruction.lower())
            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            saved_debug_frame_for_episode = False
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                # try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                # Get preprocessed image
                img = get_libero_image(obs, resize_size)
                wrist_img = get_libero_wrist_image(obs, resize_size)

                # Apply visual text overlay before both replay saving and model inference.
                img, wrist_img = maybe_apply_overlay(cfg, img, wrist_img, overlay_text)

                # Save the first actual model-input frame for sanity checking.
                if not saved_debug_frame_for_episode:
                    save_debug_observation_frames(
                        cfg,
                        run_id,
                        task_id,
                        episode_idx,
                        img,
                        wrist_img,
                        overlay_text,
                    )
                    saved_debug_frame_for_episode = True

                # Save preprocessed image for replay video
                replay_images.append(img)

                # Prepare observations dict
                # Note: OpenVLA does not take proprio state as input
                observation = {
                    "full_image": img,
                    "wrist_image": wrist_img,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }
                # Query model to get action
                action = get_action(
                    cfg,
                    model,
                    observation,
                    model_instruction,
                    server,
                )

                # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                action = normalize_gripper_action(action, binarize=True)

                # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                action = invert_gripper_action(action)

                print('==>action is',action)
                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

                # except Exception as e:
                #     print(f"Caught exception: {e}")
                #     log_file.write(f"Caught exception: {e}\n")
                #     break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file, ckpt_index=ckpt_index, task_suite_name=cfg.task_suite_name
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
