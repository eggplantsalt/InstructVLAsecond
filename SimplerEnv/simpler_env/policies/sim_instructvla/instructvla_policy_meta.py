"""SimplerEnv 中 InstructVLA 的 policy 对接实现（meta 版本）。

本文件是 SimplerEnv 与 InstructVLA 模型之间的关键桥接层，核心职责：
1) 加载 InstructVLA checkpoint
2) 将环境观测图像转换成模型输入
3) 调用 predict_action 获取原始动作
4) 把原始动作转换成环境执行格式（平移/旋转/夹爪）
"""
from collections import deque
from typing import Optional, Sequence
import os
# from PIL import Image
from PIL import Image, ImageDraw, ImageFont

#新加的函数
def render_text_on_image(
    image: Image.Image,
    text: str,
    font_size: int = 18,
    margin: int = 8,
    max_chars_per_line: int = 48,
) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = w if cur == "" else cur + " " + w
        if len(test) <= max_chars_per_line:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    if len(lines) == 0:
        return img

    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    text_block_h = sum(line_heights) + margin * (len(lines) + 1)
    text_block_w = min(max(line_widths) + 2 * margin, img.width)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [(0, 0), (text_block_w, text_block_h)],
        fill=(0, 0, 0, 180)
    )

    y = margin
    for i, line in enumerate(lines):
        overlay_draw.text((margin, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_heights[i] + margin

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return img

import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor

from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load_vla
import tensorflow as tf
from .adaptive_ensemble import AdaptiveEnsembler
from copy import deepcopy
import pickle

def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # 兼容单图与批量图输入，统一成 4D 张量。
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # 根据裁剪面积比例推导裁剪后的高宽比例。
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # 构建中心裁剪框坐标。
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # 执行中心裁剪并缩放回固定分辨率。
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # 若原始输入是单图，则还原成 3D。
    if expanded_dims:
        image = image[0]

    return image


class InstructVLAInference:
    """SimplerEnv 侧 InstructVLA 推理控制器。"""

    def __init__(
        self,
        saved_model_path: str = 'TBD',
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 1,
        action_ensemble_horizon: Optional[int] = None,
        image_size: list[int] = [224, 224],
        future_action_window_size: int = 15,
        action_dim: int = 7,
        action_model_type: str = "DiT-B",
        action_scale: float = 1.0,
        use_bf16: bool = False,
        action_ensemble = True,
        adaptive_ensemble_alpha = 0.1,
    ) -> None:
        # 关闭 tokenizer 并行日志，减少控制台噪声。
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            # widowx_bridge 默认使用 bridge 数据统计做动作反归一化。
            unnorm_key = "bridge_dataset" if unnorm_key is None else unnorm_key
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                # Set 7 for widowx_bridge to fix the window size of motion scale between each frame. see appendix in our paper for details
                action_ensemble_horizon = 7
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            # google_robot 默认使用 fractal 统计。
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                # Set 2 for google_robot to fix the window size of motion scale between each frame. see appendix in our paper for details
                action_ensemble_horizon = 2
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
                # 加载 InstructVLA（推理模式）。
        self.vla = load_vla(
          saved_model_path,
          load_for_training=False, 
          future_action_window_size=future_action_window_size,
          past_action_window_size=horizon,
          action_dim=action_dim,
        )

        if use_bf16:
            self.vla.vlm = self.vla.vlm.to(torch.bfloat16)
        else:
            self.vla.vlm = self.vla.vlm.to(torch.float16)
        # 切到 CUDA + eval，关闭训练行为。
        self.vla = self.vla.to("cuda").eval()

        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self.cognition_features_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            # 可选：动作历史集成，减小时间抖动。
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        self.num_cognition_features_history = 0
        self.action_step = 0
        self.cached_action = None

    def _add_cognition_features_to_history(self, cognition_feature) -> None:
        """把当前认知特征写入历史缓存。"""
        self.cognition_features_history.append(cognition_feature)
        self.num_cognition_features_history = min(self.num_cognition_features_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        """任务切换时重置内部状态，避免跨任务污染。

        这个函数会把与上一条任务相关的缓存、计数器、夹爪粘滞状态、
        以及 VLA 推理上下文全部清空，让下一次 `step` 从干净状态开始。
        """
        # 更新当前任务文本；后续 `step` 会用这个描述作为语言条件输入。
        self.task_description = task_description
        # 清空图像历史队列（长度受 horizon 限制），避免上一任务视觉上下文残留。
        self.image_history.clear()
        # 清空认知特征历史，防止旧任务的隐状态影响新任务动作。
        self.cognition_features_history.clear()
        # 如果启用了动作集成器，同时重置其内部时间窗口缓存。
        if self.action_ensemble:
            self.action_ensembler.reset()
        # 历史有效帧计数归零；后续会在 step 中重新累计。
        self.num_image_history = 0
        # 认知特征有效帧计数归零。
        self.num_cognition_features_history = 0
        # 关闭 sticky gripper 状态机，表示当前不处于“持续重复夹爪动作”阶段。
        self.sticky_action_is_on = False
        # sticky 重复计数器清零。
        self.gripper_action_repeat = 0
        # sticky 夹爪动作值复位为 0。
        self.sticky_gripper_action = 0.0
        # 上一时刻夹爪动作置空；下一次会按“首次观测”逻辑处理。
        self.previous_gripper_action = None
        # 清理 VLA 侧上一轮文本响应缓存。
        self.vla.last_response = None
        # 重置 VLA 内部运行步号（通常用于多步推理/追踪）。
        self.vla.run_index = 0
        # 清理 VLA 潜变量缓存，避免跨任务复用旧 latent。
        self.vla.latent = None
        # 本策略动作步号归零。
        self.action_step = 0
        # 已缓存动作清空；下一步必须重新调用模型预测。
        self.cached_action = None

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None,center_crop: Optional[bool] = False, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                # 指令变化则重置缓存。
                self.reset(task_description)

        assert image.dtype == np.uint8
        
        if center_crop:
            batch_size = 1
            crop_scale = 0.9

            # 先转 Tensor，再做裁剪缩放。
            image = tf.convert_to_tensor(image)
            orig_dtype = image.dtype

            # 归一化到 [0,1] 便于几何变换。
            image = tf.image.convert_image_dtype(image, tf.float32)

            # 中心裁剪并缩放。
            image = crop_and_resize(image, crop_scale, batch_size)

            # 转回原始 dtype。
            image = tf.clip_by_value(image, 0, 1)
            image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

            # 转 PIL，匹配 vla.predict_action 输入格式。
            image: Image.Image = Image.fromarray(image.numpy())
            image = image.convert("RGB")
        else:
            image: Image.Image = Image.fromarray(image)
            image = image.convert("RGB")

        render_instruction_on_image = True
        image_text = self.task_description if self.task_description is not None else ""

        if render_instruction_on_image and image_text:
            image = render_text_on_image(image, image_text)
            if not hasattr(self, "_debug_saved_image"):
                image.save("debug_instruction_overlay.jpg")
                self._debug_saved_image = True
                print(f"[DEBUG] saved overlay image with text: {image_text}")

        # 调用模型主推理入口，得到:
        # raw_actions: 反归一化动作
        # normalized_actions: 归一化动作
        # cognition_features_current: 当前语言-视觉隐变量
        raw_actions, normalized_actions, cognition_features_current = self.vla.predict_action(image=image, 
                                                                        instruction=self.task_description,
                                                                        unnorm_key=self.unnorm_key,
                                                                        do_sample=False, 
                                                                        prompt_mode="image_text_primary",
                                                                        )
            # self.cached_action = raw_actions
            # self.action_step = 0
        if self.action_ensemble:
            # 对当前动作进行时间一致性融合。
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # 把模型输出动作映射到 SimplerEnv 控制格式。
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)

        # 欧拉角增量 -> 轴角形式（环境侧使用轴角执行旋转）。
        roll, pitch, yaw = action_rotation_delta
        axes, angles = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = axes * angles
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            # google_robot 的夹爪逻辑采用 sticky 策略抑制高频开合抖动。
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            # bridge 环境使用 {-1, +1} 控制夹爪开合。
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
        # SimplerEnv 里默认不由策略主动终止 episode。
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """可视化辅助：缩放图像。"""
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        """可视化一段轨迹中的动作曲线与关键帧。"""
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # 构建“图像条 + 多维动作曲线”的拼图布局。
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # 绘制每一维动作随时间变化曲线。
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)