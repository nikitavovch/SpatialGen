import os
import re
import math

import torch
from packaging import version
import cv2
import json
import logging
import random
from pathlib import Path

from accelerate.state import PartialState
from accelerate.logging import MultiProcessAdapter

from src.utils.typing import *


def parse_version(ver: str):
    return version.parse(ver)


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def load_module_weights(path, module_name=None, ignore_modules=None, map_location=None) -> Tuple[dict, int, int]:
    if module_name is not None and ignore_modules is not None:
        raise ValueError("module_name and ignore_modules cannot be both set")
    if map_location is None:
        map_location = get_device()

    ckpt = torch.load(path, map_location=map_location)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    state_dict_to_load = state_dict

    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any([k.startswith(ignore_module + ".") for ignore_module in ignore_modules])
            if ignore:
                # print(f'ignore k: {k}')
                continue
            state_dict_to_load[k] = v

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf"^{module_name}\.(.*)$", k)
            if m is None:
                continue
            # print(f'load k: {k}, m: {m.group(1)}')
            # state_dict_to_load[m.group(1)] = v
            state_dict_to_load[k] = v

    return state_dict_to_load


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars, *args, **kwargs):
        if isinstance(vars, list):
            return [wrapper(x, *args, **kwargs) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x, *args, **kwargs) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v, *args, **kwargs) for k, v in vars.items()}
        else:
            return func(vars, *args, **kwargs)

    return wrapper


@make_recursive_func
def todevice(vars, device="cuda"):
    if isinstance(vars, torch.Tensor):
        return vars.to(device)
    elif isinstance(vars, str):
        return vars
    elif isinstance(vars, bool):
        return vars
    elif isinstance(vars, float):
        return vars
    elif isinstance(vars, int):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def colorize_single_channel_image(image, color_map=cv2.COLORMAP_JET):
    """
    return numpy data
    """
    image: Float[Tensor, "1 Ht Wt"]
    image = image.squeeze()
    assert len(image.shape) == 2

    image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    if torch.is_tensor(image):
        image = image.cpu().numpy()

    image = image.astype(np.uint8)

    image = cv2.applyColorMap(image, color_map)

    return image


def load_pretrain_stable_diffusion(
    new_model: torch.nn.Module, finetune_from: str = "/seaweedfs/training/experiments/zhenqing/cache/models--runwayml--stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin"
):
    print(f"Attempting to load state from {finetune_from}")
    old_state = torch.load(finetune_from, map_location="cpu")
    # if "state_dict" in old_state:
    #     old_state = old_state["state_dict"]

    in_filters_load = old_state["conv_in.weight"]
    new_state = new_model.state_dict()
    if "conv_in.weight" in new_state:
        in_filters_current = new_state["conv_in.weight"]
        in_shape = in_filters_current.shape
        ## because the model adopts additional inputs as conditions.
        if in_shape != in_filters_load.shape:
            input_keys = [
                "conv_in.weight",
            ]
            for input_key in input_keys:
                if input_key not in old_state or input_key not in new_state:
                    continue
                input_weight = new_state[input_key]
                if input_weight.size() != old_state[input_key].size():
                    print(f"Manual init: {input_key}")
                    input_weight.zero_()
                    input_weight[:, :4, :, :].copy_(old_state[input_key])
                old_state[input_key] = torch.nn.parameter.Parameter(input_weight)

    new_model.load_state_dict(old_state, strict=False)


def read_json(filepath):
    with open(filepath, "r") as fp:
        return json.load(fp)


def readlines(filepath):
    """Reads in a text file and returns lines in a list."""
    with open(filepath, "r") as f:
        lines = f.read().splitlines()
    # Remove empty lines
    lines = [line for line in lines if line.strip()]
    return lines


def load_from_jsonl(filename: Path):
    assert filename.suffix == ".jsonl"
    if not filename.exists():
        return None

    data = []
    with open(filename, encoding="utf-8") as f:
        for row in f:
            data.append(json.loads(row))
    return data

# print model info, learnable parameters, non-learnable parameters, total parameters, model size, all in billion
def print_model_info(model):
    print("=" * 20)
    # print model class name
    print("model name: ", type(model).__name__)
    print(
        "learnable parameters(M): ",
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
    )
    print(
        "non-learnable parameters(M): ",
        sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6,
    )
    print("total parameters(M): ", sum(p.numel() for p in model.parameters()) / 1e6)
    print(
        "model size(MB): ",
        sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024,
    )

def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def print_memory(msg):
    mem = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"{msg}: {mem:.2f} MB")
    

def setup_logger(name: str, level: str = None, path: str = None) -> MultiProcessAdapter:
    """
    Returns a `logging.Logger` for `name` that can handle multiprocessing with additional FileHandler.
    """
    if PartialState._shared_state == {}: PartialState()
    if level is None:
        level = "DEBUG" if os.getenv("DEBUG", False) else "INFO"
    name = name.split('.')[0]
    if path is None:
        os.makedirs("logs", exist_ok=True)
        path = f"logs/{name}"

    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f"{path}.log")
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(file_handler)

    return MultiProcessAdapter(logger, {})

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min
def seed_everything(seed=None):
    seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        raise ValueError(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")

    print(f"seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def worker_init_fn(worker_id):
    # Get a unique seed for this worker
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    
    # Set seeds for all RNGs
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    # For CUDA (multi-GPU)
    if torch.cuda.is_available():
        # Get the current GPU rank (requires distributed setup or manual tracking)
        gpu_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        torch.cuda.manual_seed_all(worker_seed + gpu_rank)
        
from diffusers.utils import is_xformers_available

def enable_flash_attn_if_avail(model):
    """Enable flash attention if available."""
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            print(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        model.enable_xformers_memory_efficient_attention()
        print("xformers is enabled for memory efficient attention.")
    