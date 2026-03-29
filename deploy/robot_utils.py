"""机器人策略评测工具集。

该文件负责把“环境观测 -> 模型推理 -> 动作后处理”这条链路统一封装，
让不同模型族（OpenVLA / InstructVLA）在评测脚本里使用同一套调用接口。
"""

import os
import random
import time

import numpy as np
import torch

from deploy.openvla_utils import (
    get_vla,
    get_vla_action,
)
from deploy.instructvla_utils import (
    InstructVLAServer
)

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})



def set_seed_everywhere(seed: int):
    """统一设置随机种子，保证评测可复现。

    这里同时覆盖 Python、NumPy、PyTorch 与 cuDNN 的随机性行为。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg, wrap_diffusion_policy_for_droid=False):
    """按模型家族加载评测模型，并返回可选的服务封装对象。

    返回值:
    - model: 实际被调用推理的模型对象
    - cronus_server: InstructVLA 的服务层封装（OpenVLA 时为 None）
    """
    # OpenVLA: 直接返回 HuggingFace 风格模型对象。
    if cfg.model_family == "openvla":
        cronus_server = None
        model = get_vla(cfg)
    # InstructVLA: 通过 InstructVLAServer 统一封装预处理、动作块缓存和集成策略。
    elif cfg.model_family == "instruct_vla":
        cronus_server = InstructVLAServer(cfg)
        model = cronus_server.vla
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model, cronus_server


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    # 当前两类模型均按 224 输入尺度训练，评测时保持一致可减少分布偏移。
    if cfg.model_family == "openvla" or "instruct_vla" in cfg.model_family:
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_action(cfg, model, obs, task_label, server, processor=None):
    """统一动作查询入口。

    参数说明:
    - obs: 环境观测，至少包含图像与可选状态信息
    - task_label: 任务文本指令
    - server: InstructVLA 场景下的服务封装
    """
    if cfg.model_family == "openvla":
        # OpenVLA 走 openvla_utils 的标准推理路径。
        action = get_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
        )
        assert action.shape == (ACTION_DIM,)
    elif "instruct_vla" in cfg.model_family:
        # InstructVLA 走 server 封装，内部会处理多视角图像、动作 chunk 与可选 ensemble。
        action = server.get_vla_action(
            model, cfg, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
        )
        # assert action.shape == (ACTION_DIM,), 'action shape is wrong'
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # 仅对末维夹爪动作做线性归一化，其它维度保持不变。
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # 二值化后可减少夹爪振荡（开/合决策更稳定）。
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    # 数据集定义与环境执行定义相反时，翻转符号对齐执行语义。
    action[..., -1] = action[..., -1] * -1.0
    return action
