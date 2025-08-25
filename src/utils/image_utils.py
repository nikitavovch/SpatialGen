#
# Copyright (C) 2024, Chuan FANG
# Personal page, https://fangchuan.github.io/
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
import PIL


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def calculate_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).contiguous().view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_psnr_numpy(img1, img2):
    mse = np.mean((img1 - img2) ** 2, axis=(1, 2, 3))
    return 20 * np.log10(1.0 / np.sqrt(mse))


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def crop_image_to_target_ratio(image: PIL.Image, target_aspect_ratio: float = 4.0 / 3.0):
    """Crops an image to satisfy a target aspect ratio."""

    actual_aspect_ratio = image.width / image.height

    if actual_aspect_ratio > target_aspect_ratio:
        # we should crop width
        new_width = image.height * target_aspect_ratio

        left = (image.width - new_width) / 2
        top = 0
        right = (image.width + new_width) / 2
        bottom = image.height

        # Crop the center of the image
        image = image.crop((left, top, right, bottom))

    elif actual_aspect_ratio < target_aspect_ratio:
        # we should crop height
        new_height = image.width / target_aspect_ratio

        left = 0
        top = (image.height - new_height) / 2
        right = image.width
        bottom = (image.height + new_height) / 2

        # Crop the center of the image
        image = image.crop((left, top, right, bottom))

    return image
