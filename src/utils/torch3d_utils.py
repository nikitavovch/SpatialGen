import torch
import torch.nn as nn
import numpy as np
import open3d as o3d

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    # PointsRenderer,
    PerspectiveCameras,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.utils import cameras_from_opencv_projection
from torchvision.utils import save_image

from src.utils.typing import *


# A renderer class should be initialized with a
# function for rasterization and a function for compositing.
# The rasterizer should:
#     - transform inputs from world -> screen space
#     - rasterize inputs
#     - return fragments
# The compositor can take fragments as input along with any other properties of
# the scene and generate images.

# E.g. rasterize inputs and then shade
#
# fragments = self.rasterize(point_clouds)
# images = self.compositor(fragments, point_clouds)
# return images


class CustomPointsRenderer(nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.

    The points are rendered with with varying alpha (weights) values depending on
    the distance of the pixel center to the true point in the xy plane. The purpose
    of this is to soften the hard decision boundary, for differentiability.
    See Section 3.2 of "SynSin: End-to-end View Synthesis from a Single Image"
    (https://arxiv.org/pdf/1912.08804.pdf) for more details.
    """

    def __init__(self, rasterizer, compositor, min_depth: float=0.01) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor
        self.min_depth = min_depth

    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds: Pointclouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)
        
        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments.zbuf


def torch3d_rasterize_points(
    cv_cam_poses_c2w: Float[Tensor, "B 4 4"],
    in_pointcloud: Float[Tensor, "N 6"],
    intrinsic: Float[Tensor, "3 3"],
    image_width: int = 256,
    image_height: int = 256,
    point_radius: float = 0.01,
    device: str = "cuda",
) -> Float[Tensor, "B 3 H W"]:

    # Initialize a camera.
    cv_cam_poses_w2c = torch.inverse(cv_cam_poses_c2w)
    Rs = cv_cam_poses_w2c[:, :3, :3].to(device)
    Ts = cv_cam_poses_w2c[:, :3, 3].to(device)
    img_size = torch.tensor([image_height, image_width])[None, :].repeat(cv_cam_poses_w2c.shape[0], 1).to(device)
    K = torch.tensor([[intrinsic[0, 0], 0, intrinsic[0, 2], 0], [0, intrinsic[1, 1], intrinsic[1, 2], 0], [0, 0, 1, 0], [0, 0, 0, 1]])[None, :, :]
    K = K.repeat(cv_cam_poses_w2c.shape[0], 1, 1).to(device)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    raster_settings = PointsRasterizationSettings(image_size=[image_height, image_width], 
                                                  radius=point_radius, 
                                                  points_per_pixel=10,
                                                  max_points_per_bin=1000000)

    torch3d_pointcloud = Pointclouds(points=[in_pointcloud[:, :3]], features=[in_pointcloud[:, 3:]])

    rendered_rgbs = []
    rendered_dpths = []
    for idx in range(Rs.shape[0]):
        R = Rs[idx : idx + 1]
        tvec = Ts[idx : idx + 1]
        k = K[idx : idx + 1]
        image_size = img_size[idx : idx + 1]
        camera = cameras_from_opencv_projection(R=R, tvec=tvec, camera_matrix=k, image_size=image_size)

        # Create a points renderer by compositing points using an alpha compositor (nearer points
        # are weighted more heavily). See [1] for an explanation.
        rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
        renderer = CustomPointsRenderer(rasterizer=rasterizer, 
                                        compositor=AlphaCompositor(background_color=(0., 0., 0.)))
        rendered_rgb, rendered_dpth = renderer(torch3d_pointcloud)
        rendered_rgbs.append(rendered_rgb)
        rendered_dpths.append(rendered_dpth)

    rendered_rgbs = torch.cat(rendered_rgbs, dim=0)
    rendered_dpths = torch.cat(rendered_dpths, dim=0)
    # breakpoint()
    # visualize the rendered rgb and depth
    # from matplotlib import pyplot as plt
    # fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    # rgb_0 = rendered_rgbs[1].cpu().numpy()
    # axs[0].imshow(rgb_0)
    # axs[0].axis("off")
    # axs[0].set_title("Rendered RGB")
    # depth_0 = rendered_dpths[1, :,:, 0].cpu().numpy()
    # axs[1].imshow(depth_0)
    # axs[1].axis("off")
    # axs[1].set_title("Rendered Depth")
    # plt.savefig("rendered_rgb_depth.png")
    torch.cuda.empty_cache()
    return rendered_rgbs.permute(0, 3, 1, 2), rendered_dpths[:, :, :, 0:1].permute(0, 3, 1, 2)


def render_rgb_and_depth_from_ply(
    input_pointcloud: Float[Tensor, "N 6"],
    cv_cam_poses_c2w: Float[Tensor, "B 4 4"],
    intrinsic: Float[Tensor, "B 3 3"],
    image_width: int = 256,
    image_height: int = 256,
    point_radius: float = 0.01,
    device: str = "cuda",
):
    rendered_tar_rgbs, rendered_tar_depths = torch3d_rasterize_points(
        cv_cam_poses_c2w=cv_cam_poses_c2w,
        in_pointcloud=input_pointcloud,
        intrinsic=intrinsic,
        image_height=image_height,
        image_width=image_width,
        point_radius=point_radius,
        device=device,
    )
    rendered_tar_depths: Float[Tensor, "B H W 1"] = rendered_tar_depths.permute(0, 2, 3, 1)
    rendered_masks = (rendered_tar_rgbs.mean(1, keepdim=True) > 0).float().clamp(0, 1)
    rendered_masks: Float[Tensor, "B H W 1"] = rendered_masks.permute(0, 2, 3, 1)
    rendered_tar_rgbs: Float[Tensor, "B H W 3"] = rendered_tar_rgbs.permute(0, 2, 3, 1)
    return rendered_tar_rgbs, rendered_tar_depths, rendered_masks