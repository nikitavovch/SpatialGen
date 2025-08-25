from typing import *
from PIL.Image import Image as PILImage
from numpy import ndarray
from torch import Tensor
from wandb import Image as WandbImage

from PIL import Image
import numpy as np
import torch
from einops import rearrange
import wandb
import matplotlib.pyplot as plt


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def wandb_mvimage_log(outputs: Dict[str, Tensor], max_num: int = 4, max_view: int = 8) -> List[WandbImage]:
    """Organize multi-view images in Dict `outputs` for wandb logging.

    Only process values in Dict `outputs` that have keys containing the word "images",
    which should be in the shape of (B, V, 3, H, W).
    """
    formatted_images = []
    for k in outputs.keys():
        if "images" in k and outputs[k] is not None:  # (B, V, 3, H, W)
            assert outputs[k].ndim == 5
            num, view = outputs[k].shape[:2]
            num, view = min(num, max_num), min(view, max_view)
            mvimages = rearrange(outputs[k][:num, :view], "b v c h w -> c (b h) (v w)")
            formatted_images.append(
                wandb.Image(
                    tensor_to_image(mvimages.detach()),
                    caption=k
                )
            )

    return formatted_images


def tensor_to_image(tensor: Tensor, return_pil: bool = False) -> Union[ndarray, PILImage]:
    if tensor.ndim == 4:  # (B, C, H, W)
        tensor = rearrange(tensor, "b c h w -> c h (b w)")
    assert tensor.ndim == 3  # (C, H, W)

    assert tensor.shape[0] in [1, 3]  # grayscale, RGB (not consider RGBA here)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    image = (tensor.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)  # (H, W, C)
    if return_pil:
        image = Image.fromarray(image)
    return image


def load_image(image_path: str, rgba: bool = False, imagenet_norm: bool = False) -> Tensor:
    image = Image.open(image_path)
    tensor_image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.  # (C, H, W) in [0, 1]

    if not rgba and tensor_image.shape[0] == 4:
        mask = tensor_image[3:4]
        tensor_image = tensor_image[:3] * mask + (1. - mask)  # white background

    if imagenet_norm:
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor_image.dtype, device=tensor_image.device).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor_image.dtype, device=tensor_image.device).view(3, 1, 1)
        tensor_image = (tensor_image - mean) / std

    return tensor_image  # (C, H, W)

  
def apply_depth_to_colormap(depth_map: np.ndarray, cmap: str = "viridis", near_plane: float = 0.00, far_plane: float = 15.0):
    """
    depth_map: (H, W) is the depth map.
    """
    near_plane = max(near_plane, depth_map.min())
    far_plane = min(far_plane, depth_map.max())
    depth_map = (depth_map - near_plane) / (far_plane - near_plane + 1e-6)
    depth_map = np.clip(depth_map, 0.0, 1.0)
    depth_map = plt.get_cmap(cmap)(depth_map)
    return depth_map[..., :3]

def convert_depth_to_colormap(depths: Tensor) -> Tensor:
    """
    depths: (B, N, H, W) is the depth map. [0, 1]
    Returns: (B, N, 3, H, W) is the depth map in color.
    """
    depths = depths.permute(0, 2, 3, 1).float().cpu()
    depths = depths.clip(0, 1)
    depths = depths.mean(dim=-1)
    color_depths = apply_depth_to_colormap(depths, cmap="viridis")
    return torch.from_numpy(color_depths).permute(0, 3, 1, 2)

from numpy import typing as npt
import cv2
def colorize_depth(depth: Union[npt.NDArray, Image.Image], depth_min: int = -1, depth_max: int = -1,
                   output_type: str = "np") -> npt.NDArray:
    if isinstance(depth, Image.Image):
        depth = np.array(depth)

    depth = 1.0 / (depth + 1e-6)
    invalid_mask = (depth <= 0) | (depth >= 1e6) | np.isnan(depth) | np.isinf(depth)
    if depth_min < 0 or depth_max < 0:
        depth_min = np.percentile(depth[~invalid_mask], 5)
        depth_max = np.percentile(depth[~invalid_mask], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min + 1e-10)
    depth_scaled_uint8 = np.uint8(np.clip(depth_scaled * 255, 0, 255))
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_JET)
    depth_color[invalid_mask, :] = 0
    depth_color = depth_color[..., ::-1]

    if output_type == "pil":
        depth_color = Image.fromarray(depth_color.astype(np.uint8)).convert("RGB")
    return depth_color

import torchvision
def save_color_depth_image(depth_imgs, cmap="viridis", output_path:str='depths.png'):
    """
    Args:
        depth_imgs: (B, H, W) is the depth map.
        cmap: colormap to use.
        output_path: path to save the image.
    """
    depth_imgs = apply_depth_to_colormap(depth_imgs, cmap=cmap)  # BxT_out,H,W,3
    depth_imgs_torch = torch.from_numpy(depth_imgs).permute(0, 3, 1, 2)
    depth_imgs_torch = torchvision.utils.make_grid(depth_imgs_torch)
    torchvision.utils.save_image(depth_imgs_torch, output_path)
    return depth_imgs