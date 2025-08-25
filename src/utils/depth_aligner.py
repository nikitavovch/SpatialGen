import cv2
import numpy as np
import torch

from src.utils.typing import *


class DepthAligner:
    def __init__(self) -> None:
        pass

    def _align_scale_shift_numpy(self, pred: np.array, target: np.array):
        mask = (target > 0) & (pred < 199)
        target_mask = target[mask]
        pred_mask = pred[mask]
        if np.sum(mask) > 10:
            scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
            if scale < 0:
                scale = np.median(target[mask]) / (np.median(pred[mask]) + 1e-8)
                shift = 0
        else:
            scale = 1
            shift = 0
        return scale, shift

    def _align_scale_shift_torch(
        self, predicted_depth: Float[Tensor, "H W"], rendered_depth: Float[Tensor, "H W"], mask: Float[Tensor, "H W"], fuse: bool = True
    ):
        """
        Optimize a scale and shift parameter in the least squares sense, such that rendered_depth and predicted_depth match.
        Formally, solves the following objective:

        min     || (d * a + b) - d_hat ||
        a, b

        where d = 1 / predicted_depth, d_hat = 1 / rendered_depth

        :param rendered_depth: torch.Tensor (H, W)
        :param predicted_depth:  torch.Tensor (H, W)
        :param mask: torch.Tensor (H, W) - 1: valid points of rendered_depth, 0: invalid points of rendered_depth (ignore)
        :param fuse: whether to fuse shifted/scaled predicted_depth with the rendered_depth

        :return: scale/shift corrected depth
        """
        if mask.sum() == 0:
            return predicted_depth

        valid_depth_mask = (rendered_depth > 0.002) & (predicted_depth > 0.002)  # ignore points that are too close
        valid_depth_mask = valid_depth_mask & (rendered_depth < 0.998) & (predicted_depth < 0.998)  # ignore points that are too far
        mask = mask & valid_depth_mask
        rendered_disparity = 1 / rendered_depth[mask].unsqueeze(-1)
        predicted_disparity = 1 / predicted_depth[mask].unsqueeze(-1)
        # breakpoint()
        X = torch.cat([predicted_disparity, torch.ones_like(predicted_disparity)], dim=1)
        XTX_inv = (X.T @ X).inverse()
        XTY = X.T @ rendered_disparity
        AB = XTX_inv @ XTY
        # breakpoint()

        fixed_disparity = (1 / predicted_depth) * AB[0] + AB[1]
        fixed_depth = 1 / fixed_disparity
        print(f"[DepthAligner::_align_scale_shift_torch] scale: {AB[0]}, shift: {AB[1]}")

        if fuse:
            fused_depth = torch.where(mask, rendered_depth, fixed_depth)
            return fused_depth
        else:
            return fixed_depth

    def __call__(self, refer_dpt, new_dpt, refer_mask):
        if np.sum(refer_mask > 0.5) < 1.0:
            return new_dpt
        # get areas need to be aligned
        render_dpt_valid = refer_dpt[refer_mask]
        new_dpt_valid = new_dpt[refer_mask]
        # rectify
        # scale, shift = self._align_scale_shift_numpy(pred=new_dpt_valid, target=render_dpt_valid)
        # print(f"[DepthAligner] scale: {scale}, shift: {shift}")
        # aligned_dpt = new_dpt * scale + shift

        predicted_depth = torch.from_numpy(new_dpt).float()
        rendered_depth = torch.from_numpy(refer_dpt).float()
        mask = torch.from_numpy(refer_mask).bool()
        aligned_dpt = self._align_scale_shift_torch(predicted_depth=predicted_depth, rendered_depth=rendered_depth, mask=mask, fuse=True)
        aligned_dpt = aligned_dpt.cpu().numpy().squeeze()
        return aligned_dpt


class SmoothDepthAligner:
    def __init__(self) -> None:
        self.coarse_align = DepthAligner()

    def _coarse_alignment(self, refer_dpt, new_dpt, refer_mask):
        # determine the scale and shift of new_dpt to coarsely align it to refer_dpt
        aligned_dpt = self.coarse_align(refer_dpt, new_dpt, refer_mask)
        return aligned_dpt

    def _refine_movements(self, refer_dpt, new_dpt, refer_msk):
        """
        Follow https://arxiv.org/pdf/2311.13384
        """
        # Determine the adjustment of un-inpainted area
        refer_msk = refer_msk > 0.5
        H, W = refer_msk.shape[0:2]
        U = np.arange(W)[None, :].repeat(H, axis=0)
        V = np.arange(H)[:, None].repeat(W, axis=1)
        # on kept areas
        valid_refer_dpt = refer_dpt[refer_msk]
        valid_new_dpt = new_dpt[refer_msk]
        visiable_dpt_adjustment = valid_refer_dpt - valid_new_dpt
        # iterative refinement
        complete_adjust = np.zeros_like(new_dpt)
        for i in range(100):
            complete_adjust[refer_msk] = visiable_dpt_adjustment
            complete_adjust = cv2.blur(complete_adjust, (15, 15))
        # complete_adjust[~refer_msk] = keep_adjust_dpt
        new_dpt = new_dpt + complete_adjust
        return new_dpt

    def align_depth(self, reference_dpt, new_dpt, reference_msk):
        if np.sum(reference_msk > 0.5) < 1.0:
            print("[SmoothDepthAligner] No reference depth map, skip alignment.")
            return new_dpt
        reference_msk = reference_msk > 0.5
        new_dpt = self._coarse_alignment(reference_dpt, new_dpt, reference_msk)
        # new_dpt = self._refine_movements(reference_dpt, new_dpt, reference_msk)
        return new_dpt
