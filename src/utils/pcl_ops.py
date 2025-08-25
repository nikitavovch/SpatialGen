import os
import os.path as osp

import random
from copy import deepcopy
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import torchvision
from einops import rearrange
from torchvision.utils import save_image
from torch.nn import functional as F
import cv2

from src.utils.typing import *
# from recons.gs.basic import edge_filter

def descale_depth(depth, min_depth, max_depth):
    depth = depth * (max_depth - min_depth) + min_depth
    return depth


def convert_z_to_distance(
    depth_map: Float[Tensor, "1 H W"],
    image_height: int = 256,
    image_width: int = 256,
    focal_length: float = 1.0,
) -> Float[Tensor, "1 H W"]:
    """
    Convert distance map to depth map
    params:
        distance_map: [1, 1, H, W]
        focal_length: noormlized focal length
    """

    vs, us = torch.meshgrid(
        torch.linspace(-1, 1, image_height),
        torch.linspace(-1, 1, image_width),
        indexing="ij",
    )
    us = us.reshape(1, image_height, image_width)
    vs = vs.reshape(1, image_height, image_width)
    depth_cos = torch.cos(torch.atan2(torch.sqrt(us * us + vs * vs), torch.tensor(focal_length))).to(depth_map.device)
    distance = depth_map / depth_cos

    return distance

def convert_distance_to_z(distance_map: Float[Tensor, "1 H W"], 
                          image_height: int = 256,
                          image_width: int = 256,
                          focal_length: float = 1.0) -> Float[Tensor, "1 H W"]:
    """
    Convert distance map to depth map
    params:
        distance_map: [1, 1, H, W]
        focal_length: noormlized focal length
    """

    vs, us = torch.meshgrid(torch.linspace(-1, 1, image_height), torch.linspace(-1, 1, image_width), indexing="ij")
    us = us.reshape(1, image_height, image_width)
    vs = vs.reshape(1, image_height, image_width)
    depth_cos = torch.cos(torch.atan2(torch.sqrt(us * us + vs * vs), torch.tensor(focal_length)))
    depth = distance_map * depth_cos

    return depth


def convert_distance_to_z(distance_map: np.ndarray, 
                          image_height: int = 768,
                          image_width: int = 1024,
                          focal_length: float = 1.0) -> np.ndarray:
    """
    Convert distance map to depth map, in this case, the image shape is not rectangular
    params:
        distance_map: [H, W]
        focal_length: NOT normlized focal length !!!
    """
    
    xs = np.linspace((-0.5 * image_width) + 0.5, (0.5 * image_width) - 0.5, image_width)
    xs = xs.reshape(1, image_width).repeat(image_height, 0).astype(np.float32)[:, :, None]
    ys = np.linspace((-0.5 * image_height) + 0.5, (0.5 * image_height) - 0.5, image_height)
    ys = ys.reshape(image_height, 1).repeat(image_width, 1).astype(np.float32)[:, :, None]
    zs = np.full([image_height, image_width, 1], focal_length, np.float32)
    img_plane = np.concatenate([xs, ys, zs], 2)

    depth_map = distance_map / np.linalg.norm(img_plane, 2, axis=2) * focal_length

    return depth_map  # [H, W]

def batch_convert_z_to_distance(
    depth_maps: Float[Tensor, "B 1 H W"],
    image_height: int = 256,
    image_width: int = 256,
    focal_length: float = 1.0,
) -> Float[Tensor, "B 1 H W"]:
    """
    Convert distance map to depth map
    params:
        distance_map: [1, 1, H, W]
        focal_length: noormlized focal length
    """
    bsz = depth_maps.shape[0]
    vs, us = torch.meshgrid(
        torch.linspace(-1, 1, image_height),
        torch.linspace(-1, 1, image_width),
        indexing="ij",
    )
    us = us.reshape(1, image_height, image_width)
    vs = vs.reshape(1, image_height, image_width)
    depth_cos = torch.cos(torch.atan2(torch.sqrt(us * us + vs * vs), torch.tensor(focal_length))).to(depth_maps.device)
    distance = depth_maps / depth_cos

    return distance

def descale_depth(depth, min_depth, max_depth):
    depth = depth * (max_depth - min_depth) + min_depth
    return depth


def rgbd_to_pointcloud(
    depth_image: np.array,
    c2w_pose: np.array,
    depth_scale: float = 1.0,
    intrinsic_mat: np.array = None,
    rgb_image: np.array = None,
    normal_image: np.array = None,
) -> o3d.geometry.PointCloud:
    """
    Convert RGBD image to point cloud
    params:
        depth_image: np.array, depth image
        c2w_pose: np.array, camera to world pose
        depth_scale: float, depth scale
        fl_x: float, focal length in x axis
        fl_y: float, focal length in y axis
        rgb_image: np.array, rgb image
    """

    depth_image = (depth_image * depth_scale).astype(np.float32)
    if len(depth_image.shape) == 2:
        depth_image = np.expand_dims(depth_image, axis=2)
    H, W, C = depth_image.shape
    n_pts = H * W
    # Get camera intrinsic
    if intrinsic_mat is not None:
        K = intrinsic_mat
    else:
        hfov = 90.0 * np.pi / 180.0
        fl_x = W / 2.0 / np.tan((hfov / 2.0))
        K = np.array(
            [
                [fl_x, 0.0, W / 2.0],
                [0.0, fl_x, H / 2.0],
                [
                    0.0,
                    0.0,
                    1,
                ],
            ]
        )

    depth_map = depth_image.reshape(1, H, W)

    # pts_x = np.linspace(0, W - 1, W)
    # pts_y = np.linspace(0, H - 1, H)
    pts_x = np.linspace(0, W, W)
    pts_y = np.linspace(0, H, H)
    
    pts_xx, pts_yy = np.meshgrid(pts_x, pts_y)

    pts = np.stack((pts_xx, pts_yy, np.ones_like(pts_xx)), axis=0)
    pts = np.linalg.inv(K) @ (pts * depth_map).reshape(3, n_pts)
    # filter out invalid points with large gradient
    points_grad = np.zeros((H, W, 1))
    points_map = pts.T.reshape(H, W, 3)
    points_grad_x = points_map[2:, 1:-1] - points_map[:-2, 1:-1]
    points_grad_x = np.linalg.norm(points_grad_x.reshape(-1, 3), axis=-1)
    points_grad_y = points_map[1:-1, 2:] - points_map[1:-1, :-2]
    points_grad_y = np.linalg.norm(points_grad_y.reshape(-1, 3), axis=-1)
    # print(f"points_grad_x: {points_grad_x.shape}, points_grad_y: {points_grad_y.shape}")
    grad = np.sqrt(points_grad_x**2 + points_grad_y**2)
    # print(f"grad: {grad.shape}")
    points_grad[1:-1, 1:-1, 0] = grad.reshape(H - 2, W - 2)
    grad_thresh = grad.mean() * 2

    # invalid_mask = points_grad.mean(axis=2) > grad_thresh
    # invalid_mask = invalid_mask.reshape(n_pts)
    invalid_mask = np.zeros(n_pts, dtype=bool)

    if rgb_image is not None:
        color = (rgb_image[:, :, :3] / 255.0).clip(0.0, 1.0)
    else:
        color = np.zeros_like(pts[:3].T)

    if normal_image is not None:
        # convert normal to [-1, 1]
        normal = np.clip((normal_image + 0.5) / 255.0, 0.0, 1.0) * 2 - 1
        normal = normal / (np.linalg.norm(normal, axis=2)[:, :, np.newaxis] + 1e-6)

        points, colors, normals = (
            np.transpose(pts)[:, :3],
            color.reshape(n_pts, 3),
            normal.reshape(n_pts, 3),
        )
    else:
        points, colors = np.transpose(pts)[:, :3], color.reshape(n_pts, 3)
        normals = np.zeros_like(points)

    points = points[~invalid_mask]
    colors = colors[~invalid_mask]
    normals = normals[~invalid_mask]
    o3d_ply = o3d.geometry.PointCloud()
    o3d_ply.points = o3d.utility.Vector3dVector(points)
    o3d_ply.colors = o3d.utility.Vector3dVector(colors)
    o3d_ply.normals = o3d.utility.Vector3dVector(normals)

    return o3d_ply.transform(c2w_pose)

def batch_rgbd_to_pointcloud(rgb_images: Float[Tensor, "B N H W 3"],
                             depth_maps: Float[Tensor, "B N H W 1"],
                             c2w_poses: Float[Tensor, "B N 4 4"],
                             intrinsic_mat: Float[Tensor, "B 3 3"],):
    bsz, n_views = depth_maps.shape[:2]
    H, W = depth_maps.shape[-3:-1]
    batch_intrinsic_mat = intrinsic_mat[:, None, ...]

    pts_x = torch.linspace(0, W - 1, W, device=depth_maps.device)
    pts_y = torch.linspace(0, H - 1, H, device=depth_maps.device)
    pts_xx, pts_yy = torch.meshgrid(pts_x, pts_y, indexing="ij")

    pts = torch.stack((pts_xx, pts_yy, torch.ones_like(pts_xx)), dim=-1)
    pts = pts[None, None, ...].repeat(bsz, n_views, 1, 1, 1)
    pts: Float[Tensor, "B N 3 Np"] = torch.inverse(batch_intrinsic_mat) @ (pts * depth_maps).reshape(bsz, n_views, 3, H * W)
    # pts = rearrange(pts, "B N c Np -> B N Np c")
    # torch.bmm(c2w_poses[:, :, :3, :3], pts) + c2w_poses[:, :, :3, 3, None]
    trans_pts = torch.einsum("BNdc, BNcp -> BNdp", c2w_poses[:, :, :3, :3], pts)
    trans_pts = trans_pts + c2w_poses[:, :, :3, 3][:, :, :, None]
    trans_pts = rearrange(trans_pts, "B N c Np -> B (N Np) c", Np=H * W)
    colors = rearrange(rgb_images, "B N H W c -> B (N H W) c")
    
    return torch.concat([trans_pts, colors], dim=-1)
    
def batch_project_points(input_points: Float[Tensor, "B 3 Np"], 
                         output_c2w_poses: Float[Tensor, "B N 4 4"], 
                         intrinsic: Float[Tensor, "B 3 3"], 
                         output_ref_depths: Float[Tensor, "B N H W 1"]):
    """
    Project points to output views, and filter out invalid points by the ref_depths maps
    params:
        input_points: [B, 3, Np], input points
        output_c2w_poses: [B, N, 4, 4], output poses
        intrinsic: [B, 3, 3], intrinsic matrix
        output_ref_depths: [B, N, H, W, 1], output reference depth maps, used to filter out invalid points caused by occlusion
    """
    bsz, n_views = output_c2w_poses.shape[:2]
    H, W = output_ref_depths.shape[-2:]
    
    output_w2c_poses = torch.inverse(output_c2w_poses)
    batch_intrinsics = intrinsic[:, None, ...].repeat(1, n_views, 1, 1)
    output_ref_depths = rearrange(output_ref_depths, "B N H W c -> B N c H W")
    
    input_points: Float[Tensor, "B 3 Np"] = input_points[:, None, ...].repeat(1, n_views, 1, 1)
    trans_pts = torch.einsum("BNdc, BNcp -> BNdp", output_w2c_poses[:, :, :3, :3], input_points)
    trans_pts = trans_pts + output_w2c_poses[:, :, :3, 3][:, :, :, None]
    
    projections: Float[Tensor, "B N 3 Np"] = torch.einsum("BNcd, BNdp -> BNcp", batch_intrinsics, trans_pts)
    # projections = rearrange(projections, "B N c (H W) -> B N H W c")
    # valid_pts = projections[:, :, 2] > 0
    # projections = projections / projections[:, :, 2:3, :]
    return projections
    
    

def nei_delta(input, pad=2):
    if not type(input) is torch.Tensor:
        input = torch.from_numpy(input.astype(np.float32))
    if len(input.shape) < 3:
        input = input[:, :, None]
    h, w, c = input.shape
    # reshape
    input = input.permute(2, 0, 1)[None]
    input = F.pad(input, pad=(pad, pad, pad, pad), mode="replicate")
    kernel = 2 * pad + 1
    input = F.unfold(input, [kernel, kernel], padding=0)
    input = input.reshape(c, -1, h, w).permute(2, 3, 0, 1).squeeze()  # hw(3)*25
    return torch.amax(input, dim=-1), torch.amin(input, dim=-1), input


def edge_filter(metric_dpt, valid_mask=None, times=0.1):
    if valid_mask is None:
        valid_mask = np.zeros_like(metric_dpt, dtype=bool)
        valid_mask[metric_dpt > 0] = True
    _max = np.percentile(metric_dpt[valid_mask], 95)
    _min = np.percentile(metric_dpt[valid_mask], 5)
    _range = _max - _min
    nei_max, nei_min, _ = nei_delta(metric_dpt)
    delta = (nei_max - nei_min).numpy()
    edge = delta > times * _range
    return edge

def save_pointcloud(
    input_images: Float[Tensor, "Bt 3 H W"],
    input_depths: Float[Tensor, "Bt 1 H W"],
    c2w_poses: Float[Tensor, "Bt 4 4"],
    min_depth: Float[Tensor, "B "],
    max_depth: Float[Tensor, "B "],
    scene_scale: Float[Tensor, "B "],
    use_metric_depth: bool = False,
    output_folder: str = "./debug_output",
    intrinsic_mat: np.ndarray = None,
    is_gt: bool = False,
):
    """
    save input and output point cloud
    input_images: [BxT, C, H, W], range [-1, 1]
    input_depths: [BxT, 1, H, W], range [-1, 1]
    output_images: [BxT, C, H, W], range [-1, 1]
    output_depths: [BxT, 1, H, W], range [-1, 1]

    """
    os.makedirs(output_folder, exist_ok=True)

    prefix_str = "gt_" if is_gt else "pred_"

    input_rgbs = (input_images + 1.0) / 2.0
    inputs = torchvision.utils.make_grid(input_rgbs, nrow=1)
    torchvision.utils.save_image(inputs, os.path.join(output_folder, f"{prefix_str}rgbs.png"))

    input_rgbs = input_rgbs.clone().permute(0, 2, 3, 1).cpu().numpy() * 255.0
    if not use_metric_depth:
        input_depths = descale_depth((input_depths.clone() + 1.0) / 2.0, min_depth, max_depth)
    else:
        input_depths = descale_metric_log_normalization((input_depths.clone() + 1.0) / 2.0, min_depth, max_depth)
    # input_depths[input_depths <= 0.01] = 0.0
    # input_depths[input_depths >= 12.0] = 0.0
    
    if not use_metric_depth:
        denormalized_depths = input_depths * scene_scale
        denormalized_depths = denormalized_depths.permute(0, 2, 3, 1).cpu().numpy()
    else:
        denormalized_depths = input_depths.permute(0, 2, 3, 1).cpu().numpy()

    

    input_rgb_ply = o3d.geometry.PointCloud()
    img_height, img_width = input_rgbs.shape[1], input_rgbs.shape[2]
    cam_intrinsic = np.array(
        [
            [img_width / 2.0, 0, img_width / 2.0],
            [0, img_height / 2.0, img_height / 2.0],
            [0, 0, 1],
        ]
    )
    # create a camera
    camera = o3d.camera.PinholeCameraIntrinsic()
    camera.set_intrinsics(
        img_width,
        img_height,
        cam_intrinsic[0, 0],
        cam_intrinsic[1, 1],
        cam_intrinsic[0, 2],
        cam_intrinsic[1, 2],
    )
    save_data_dict = {}
    rgb_views = {}
    depth_views = {}
    pose_views = {}
    point_views = {}
    color_views = {}
    for id, (rgb, depth, c2w_pose) in enumerate(zip(input_rgbs, denormalized_depths, c2w_poses)):
        Image.fromarray((depth[:, :, 0] * 1000.0).astype(np.uint16)).save(os.path.join(output_folder, f"{prefix_str}in_depth_{id}.png"))
        
        edge_mask = edge_filter(depth, times=0.2)
        Image.fromarray((edge_mask * 255).astype(np.uint8)).save(os.path.join(output_folder, f"{prefix_str}in_depth_{id}_edge.png"))
        if len(edge_mask.shape) == 2:
            edge_mask = edge_mask[:, :, np.newaxis]
        depth[edge_mask] = 0.0
        
        # recover depth to pointcloud
        o3d_recon_ply = rgbd_to_pointcloud(
            rgb_image=rgb,
            depth_image=depth[:, :, 0],
            c2w_pose=c2w_pose.detach().cpu().numpy(),
            depth_scale=1.0,
            intrinsic_mat=intrinsic_mat,
        )
        input_rgb_ply += o3d_recon_ply
        # draw camera frame in the point cloud
        T = c2w_pose.cpu().numpy().astype(np.float32)
        cam_lines = o3d.geometry.LineSet.create_camera_visualization(intrinsic=camera, extrinsic=np.linalg.inv(T), scale=0.05)
        cam_lines.paint_uniform_color([1, 0, 0])
        o3d.io.write_line_set(os.path.join(output_folder, f"input_cam_{id}.ply"), cam_lines)
        rgb_views[f"input_rgb_{id}"] = rgb
        depth_views[f"input_depth_{id}"] = depth[:, :, 0]
        pose_views[f"input_pose_{id}"] = c2w_pose.detach().cpu().numpy()
        point_views[f"input_points_{id}"] = np.asarray(o3d_recon_ply.points)
        color_views[f"input_colors_{id}"] = np.asarray(o3d_recon_ply.colors)

    # save data into a npz file
    save_data_dict.update(rgb_views)
    save_data_dict.update(depth_views)
    save_data_dict.update(pose_views)
    save_data_dict.update(point_views)
    save_data_dict.update(color_views)
    save_data_dict["intrinisc_mat"] = intrinsic_mat
    np.savez(os.path.join(output_folder, f"{prefix_str}data.npz"), **save_data_dict)

    o3d.io.write_point_cloud(os.path.join(output_folder, f"{prefix_str}pointcloud.ply"), input_rgb_ply)
    return True
    
def save_input_output_pointcloud(
    input_images: Float[Tensor, "Bt 3 H W"],
    input_depths: Float[Tensor, "Bt 1 H W"],
    output_images: Float[Tensor, "Bt 3 H W"],
    output_depths: Float[Tensor, "Bt 1 H W"],
    poses_input: Float[Tensor, "Bt 4 4"],
    poses_output: Float[Tensor, "Bt 4 4"],
    min_depth: Float[Tensor, "B "],
    max_depth: Float[Tensor, "B "],
    scene_scale: Float[Tensor, "B "],
    use_metric_depth: bool = True,
    use_scene_coord_map: bool = False,
    output_folder: str = "./debug_output",
    intrinsic_mat: np.ndarray = None,
    input_depth_conf_maps: Float[Tensor, "Bt 1 H W"] = None,
    output_depth_conf_maps: Float[Tensor, "Bt 1 H W"] = None,
    is_gt: bool = False,
):
    """
    save input and output point cloud
    input_images: [BxT, C, H, W], range [-1, 1]
    input_depths: [BxT, 1, H, W], range [-1, 1]
    output_images: [BxT, C, H, W], range [-1, 1]
    output_depths: [BxT, 1, H, W], range [-1, 1]

    """
    os.makedirs(output_folder, exist_ok=True)

    prefix_str = "gt_" if is_gt else "pred_"

    input_rgbs = (input_images + 1.0) / 2.0
    inputs = torchvision.utils.make_grid(input_rgbs, nrow=1)
    torchvision.utils.save_image(inputs, os.path.join(output_folder, "input_rgbs.png"))
    output_rgbs = (output_images + 1.0) / 2.0
    outputs = torchvision.utils.make_grid(output_rgbs, nrow=1)
    torchvision.utils.save_image(outputs, os.path.join(output_folder, f"{prefix_str}tar_rgbs.png"))

    input_rgbs = input_rgbs.clone().permute(0, 2, 3, 1).cpu().numpy() * 255.0
    output_rgbs = output_rgbs.clone().permute(0, 2, 3, 1).cpu().numpy() * 255.0
    
    if not use_scene_coord_map:
        if not use_metric_depth:
            input_depths = descale_depth((input_depths.clone() + 1.0) / 2.0, min_depth, max_depth)
            output_depths = descale_depth((output_depths.clone() + 1.0) / 2.0, min_depth, max_depth)
        else:
            input_depths = descale_metric_log_normalization((input_depths.clone() + 1.0) / 2.0, min_depth, max_depth)
            output_depths = descale_metric_log_normalization((output_depths.clone() + 1.0) / 2.0, min_depth, max_depth)
        input_depths[input_depths <= 0.05] = 0.0
        input_depths[input_depths >= 12.0] = 0.0
        output_depths[output_depths <= 0.05] = 0.0
        output_depths[output_depths >= 12.0] = 0.0
            
        if not use_metric_depth:
            filtered_input_depths = input_depths * scene_scale
            filtered_output_depths = output_depths * scene_scale
            filtered_input_depths = filtered_input_depths.permute(0, 2, 3, 1).cpu().numpy()
            filtered_output_depths = filtered_output_depths.permute(0, 2, 3, 1).cpu().numpy()
        else:
            filtered_input_depths = input_depths.permute(0, 2, 3, 1).cpu().numpy()
            filtered_output_depths = output_depths.permute(0, 2, 3, 1).cpu().numpy()
    else:
        input_depths = descale_depth((input_depths.clone() + 1.0) / 2.0, min_depth, max_depth) * scene_scale
        output_depths = descale_depth((output_depths.clone() + 1.0) / 2.0, min_depth, max_depth) * scene_scale
        filtered_input_depths = input_depths.permute(0, 2, 3, 1).cpu().numpy()
        filtered_output_depths = output_depths.permute(0, 2, 3, 1).cpu().numpy()

    

    input_rgb_ply = o3d.geometry.PointCloud()
    total_rgb_ply = o3d.geometry.PointCloud()
    img_height, img_width = input_rgbs.shape[1], input_rgbs.shape[2]
    cam_intrinsic = np.array(
        [
            [img_width / 2.0, 0, img_width / 2.0],
            [0, img_height / 2.0, img_height / 2.0],
            [0, 0, 1],
        ]
    )
    # create a camera
    camera = o3d.camera.PinholeCameraIntrinsic()
    camera.set_intrinsics(
        img_width,
        img_height,
        cam_intrinsic[0, 0],
        cam_intrinsic[1, 1],
        cam_intrinsic[0, 2],
        cam_intrinsic[1, 2],
    )
    save_data_dict = {}
    in_rgbs, input_depths, input_poses = [], [], []
    target_rgbs, target_depths, target_poses = [], [], []
    input_points, input_colors = [], []
    target_points, target_colors = [], []
    for id, (rgb, depth, c2w_pose) in enumerate(zip(input_rgbs, filtered_input_depths, poses_input)):
        
        if not use_scene_coord_map:
            edge_mask = edge_filter(depth, times=0.2)
            if len(edge_mask.shape) == 2:
                edge_mask = edge_mask[:, :, np.newaxis]
            depth[edge_mask] = 0.0
            
            # recover depth to pointcloud
            o3d_recon_ply = rgbd_to_pointcloud(
                rgb_image=rgb,
                depth_image=depth[:, :, 0],
                c2w_pose=c2w_pose.detach().cpu().numpy(),
                depth_scale=1.0,
                intrinsic_mat=intrinsic_mat,
            )
        else:
            if input_depth_conf_maps is not None:
                invalid_conf_mask = input_depth_conf_maps[id].permute(1, 2, 0).cpu().numpy() < 5.0
                depth = depth * (1 - invalid_conf_mask.astype(np.float32))
            scene_coord_maps = deepcopy(depth).reshape(-1, 3)
            # calculate the depth map 
            w2c_pose = np.linalg.inv(c2w_pose.detach().cpu().numpy())
            depth = (w2c_pose[:3, :3] @ scene_coord_maps.T + w2c_pose[:3, 3:4]).T.reshape(img_height, img_width, 3)
            depth = depth[:, :, 2:3]
            
            o3d_recon_ply = o3d.geometry.PointCloud()
            o3d_recon_ply.points = o3d.utility.Vector3dVector(scene_coord_maps)
            o3d_recon_ply.colors = o3d.utility.Vector3dVector((rgb/255.0).reshape(-1, 3))
            
        input_rgb_ply += o3d_recon_ply
        # draw camera frame in the point cloud
        T = c2w_pose.cpu().numpy().astype(np.float32)
        cam_lines = o3d.geometry.LineSet.create_camera_visualization(intrinsic=camera, extrinsic=np.linalg.inv(T), scale=0.05)
        cam_lines.paint_uniform_color([1, 0, 0])
        o3d.io.write_line_set(os.path.join(output_folder, f"input_cam_{id}.ply"), cam_lines)
        
        in_rgbs.append(rgb)
        input_depths.append(depth[:, :, 0])
        input_poses.append(c2w_pose.detach().cpu().numpy())
        input_points.append(np.asarray(o3d_recon_ply.points))
        input_colors.append(np.asarray(o3d_recon_ply.colors))

    total_rgb_ply += input_rgb_ply
    for id, (rgb, depth, c2w_pose) in enumerate(zip(output_rgbs, filtered_output_depths, poses_output)):
        if not use_scene_coord_map:
            edge_mask = edge_filter(depth, times=0.2)
            if len(edge_mask.shape) == 2:
                edge_mask = edge_mask[:, :, np.newaxis]
            depth[edge_mask] = 0.0
            # recover depth to pointcloud
            o3d_recon_ply = rgbd_to_pointcloud(
                rgb_image=rgb,
                depth_image=depth[:, :, 0],
                c2w_pose=c2w_pose.detach().cpu().numpy(),
                depth_scale=1.0,
                intrinsic_mat=intrinsic_mat,
            )
        else:
            if output_depth_conf_maps is not None:
                invalid_conf_mask = output_depth_conf_maps[id].permute(1, 2, 0).cpu().numpy() < 5.0
                depth = depth * (1 - invalid_conf_mask.astype(np.float32))
            scene_coord_maps = deepcopy(depth).reshape(-1, 3)
            # calculate the depth map 
            w2c_pose = np.linalg.inv(c2w_pose.detach().cpu().numpy())
            depth = (w2c_pose[:3, :3] @ scene_coord_maps.T + w2c_pose[:3, 3:4]).T.reshape(img_height, img_width, 3)
            depth = depth[:, :, 2:3]
            
            o3d_recon_ply = o3d.geometry.PointCloud()
            o3d_recon_ply.points = o3d.utility.Vector3dVector(scene_coord_maps)
            o3d_recon_ply.colors = o3d.utility.Vector3dVector((rgb/255.0).reshape(-1, 3))

        total_rgb_ply += o3d_recon_ply
        T = c2w_pose.cpu().numpy().astype(np.float32)
        cam_lines = o3d.geometry.LineSet.create_camera_visualization(intrinsic=camera, extrinsic=np.linalg.inv(T), scale=0.05)
        cam_lines.paint_uniform_color([0, 0, 1])
        o3d.io.write_line_set(os.path.join(output_folder, f"output_cam_{id}.ply"), cam_lines)
        
        target_rgbs.append(rgb)
        target_depths.append(depth[:, :, 0])
        target_poses.append(c2w_pose.detach().cpu().numpy())
        target_points.append(np.asarray(o3d_recon_ply.points))
        target_colors.append(np.asarray(o3d_recon_ply.colors))

    # update all pose to relative pose w.r.t the first view
    ref_pose = deepcopy(input_poses[0])
    for i in range(len(input_poses)):
        input_poses[i] = np.linalg.inv(ref_pose) @ input_poses[i]
    for i in range(len(target_poses)):
        target_poses[i] = np.linalg.inv(ref_pose) @ target_poses[i]
    # save data into a npz file
    save_data_dict['input_rgbs'] = in_rgbs
    save_data_dict['input_depths'] = input_depths
    save_data_dict['input_poses'] = input_poses
    save_data_dict['input_points'] = input_points
    save_data_dict['input_colors'] = input_colors
    save_data_dict['target_rgbs'] = target_rgbs
    save_data_dict['target_depths'] = target_depths
    save_data_dict['target_poses'] = target_poses
    save_data_dict['target_points'] = target_points
    save_data_dict['target_colors'] = target_colors
    save_data_dict["intrinsic"] = intrinsic_mat

    o3d.io.write_point_cloud(os.path.join(output_folder, f"{prefix_str}input_rgb.ply"), input_rgb_ply)
    o3d.io.write_point_cloud(os.path.join(output_folder, f"{prefix_str}total_rgb.ply"), total_rgb_ply)
    
    return save_data_dict

def save_input_output_pointcloud_with_sem(
    input_images: Float[Tensor, "Bt 3 H W"],
    input_depths: Float[Tensor, "Bt 1 H W"],
    input_semantics: Float[Tensor, "Bt 3 H W"],
    output_images: Float[Tensor, "Bt 3 H W"],
    output_depths: Float[Tensor, "Bt 1 H W"],
    output_semantics: Float[Tensor, "Bt 3 H W"],
    poses_input: Float[Tensor, "Bt 4 4"],
    poses_output: Float[Tensor, "Bt 4 4"],
    min_depth: Float[Tensor, "B "],
    max_depth: Float[Tensor, "B "],
    scene_scale: Float[Tensor, "B "],
    intrinsic_mat: np.ndarray = None,
    output_folder: str = "./debug_output",
    is_gt: bool = False,
):
    """
    save input and output point cloud
    input_images: [BxT, C, H, W], range [-1, 1]
    input_depths: [BxT, 1, H, W], range [-1, 1]
    output_images: [BxT, C, H, W], range [-1, 1]
    output_depths: [BxT, 1, H, W], range [-1, 1]

    """
    os.makedirs(output_folder, exist_ok=True)

    prefix_str = "gt_" if is_gt else "pred_"

    input_rgbs = (input_images + 1.0) / 2.0
    inputs = torchvision.utils.make_grid(input_rgbs, nrow=1)
    torchvision.utils.save_image(inputs, os.path.join(output_folder, "input_rgbs.png"))
    output_rgbs = (output_images + 1.0) / 2.0
    outputs = torchvision.utils.make_grid(output_rgbs, nrow=1)
    torchvision.utils.save_image(outputs, os.path.join(output_folder, f"{prefix_str}tar_rgbs.png"))

    input_rgbs = input_rgbs.permute(0, 2, 3, 1).cpu().numpy() * 255.0
    output_rgbs = output_rgbs.permute(0, 2, 3, 1).cpu().numpy() * 255.0
    input_depths = descale_depth((input_depths + 1.0) / 2.0, min_depth, max_depth).permute(0, 2, 3, 1).cpu().numpy()
    output_depths = descale_depth((output_depths + 1.0) / 2.0, min_depth, max_depth).permute(0, 2, 3, 1).cpu().numpy()
    if input_semantics is not None and output_semantics is not None:
        input_sems = ((input_semantics + 1.0) / 2.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0
        output_sems = ((output_semantics + 1.0) / 2.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0

    input_rgb_ply = o3d.geometry.PointCloud()
    input_sem_ply = o3d.geometry.PointCloud()
    total_rgb_ply = o3d.geometry.PointCloud()
    total_sem_ply = o3d.geometry.PointCloud()
    img_height, img_width = input_rgbs.shape[1], input_rgbs.shape[2]
    R_cv_gl = np.eye(3)
    cam_intrinsic = np.array(
        [
            [img_width / 2.0, 0, img_width / 2.0],
            [0, img_height / 2.0, img_height / 2.0],
            [0, 0, 1],
        ]
    )
    # create a camera
    camera = o3d.camera.PinholeCameraIntrinsic()
    camera.set_intrinsics(
        img_width,
        img_height,
        cam_intrinsic[0, 0],
        cam_intrinsic[1, 1],
        cam_intrinsic[0, 2],
        cam_intrinsic[1, 2],
    )

    for id, (rgb, depth, semantic, c2w_pose) in enumerate(zip(input_rgbs, input_depths, input_sems, poses_input)):
        Image.fromarray((depth[:, :, 0] * 1000.0).astype(np.uint16)).save(os.path.join(output_folder, f"pred_in_depth_{id}.png"))
        # recover depth to pointcloud
        o3d_recon_ply = rgbd_to_pointcloud(
            rgb_image=rgb,
            depth_image=depth[:, :, 0],
            c2w_pose=c2w_pose.detach().cpu().numpy(),
            # depth_scale=scene_scale.cpu().numpy(),
            depth_scale=1.0,
            intrinsic_mat=intrinsic_mat,
        )
        o3d_sem_ply = rgbd_to_pointcloud(
            rgb_image=semantic,
            depth_image=depth[:, :, 0],
            c2w_pose=c2w_pose.detach().cpu().numpy(),
            # depth_scale=scene_scale.cpu().numpy(),
            depth_scale=1.0,
            intrinsic_mat=intrinsic_mat,
        )
        # draw camera frame in the point cloud
        T = c2w_pose.cpu().numpy().astype(np.float32)
        T[:3, :3] = T[:3, :3] @ R_cv_gl
        cam_lines = o3d.geometry.LineSet.create_camera_visualization(intrinsic=camera, extrinsic=np.linalg.inv(T), scale=0.05)
        cam_lines.paint_uniform_color([1, 0, 0])
        o3d.io.write_line_set(os.path.join(output_folder, f"input_cam_{id}.ply"), cam_lines)
        input_rgb_ply += o3d_recon_ply
        input_sem_ply += o3d_sem_ply

    total_rgb_ply += input_rgb_ply
    total_sem_ply += input_sem_ply
    for id, (rgb, depth, semantic, c2w_pose) in enumerate(zip(output_rgbs, output_depths, output_sems, poses_output)):
        Image.fromarray((depth[:, :, 0] * 1000.0).astype(np.uint16)).save(os.path.join(output_folder, f"pred_tar_depth_{id}.png"))
        # recover depth to pointcloud
        o3d_recon_ply = rgbd_to_pointcloud(
            rgb_image=rgb,
            depth_image=depth[:, :, 0],
            c2w_pose=c2w_pose.detach().cpu().numpy(),
            # depth_scale=scene_scale.cpu().numpy(),
            depth_scale=1.0,
            intrinsic_mat=intrinsic_mat,
        )
        o3d_sem_ply = rgbd_to_pointcloud(
            rgb_image=semantic,
            depth_image=depth[:, :, 0],
            c2w_pose=c2w_pose.detach().cpu().numpy(),
            # depth_scale=scene_scale.cpu().numpy(),
            depth_scale=1.0,
            intrinsic_mat=intrinsic_mat,
        )
        T = c2w_pose.cpu().numpy().astype(np.float32)
        T[:3, :3] = T[:3, :3] @ R_cv_gl
        cam_lines = o3d.geometry.LineSet.create_camera_visualization(intrinsic=camera, extrinsic=np.linalg.inv(T), scale=0.05)
        cam_lines.paint_uniform_color([0, 0, 1])
        o3d.io.write_line_set(os.path.join(output_folder, f"output_cam_{id}.ply"), cam_lines)
        total_rgb_ply += o3d_recon_ply
        total_sem_ply += o3d_sem_ply
    o3d.io.write_point_cloud(os.path.join(output_folder, f"{prefix_str}input_rgb.ply"), input_rgb_ply)
    o3d.io.write_point_cloud(os.path.join(output_folder, f"{prefix_str}total_rgb.ply"), total_rgb_ply)
    o3d.io.write_point_cloud(os.path.join(output_folder, f"{prefix_str}input_sem.ply"), input_sem_ply)
    o3d.io.write_point_cloud(os.path.join(output_folder, f"{prefix_str}total_sem.ply"), total_sem_ply)


def eval_cross_view_consistency(
    input_rgbs: Float[Tensor, "Bt 3 H W"],
    input_depths: Float[Tensor, "Bt 1 H W"],
    output_rgbs: Float[Tensor, "Bt 3 H W"],
    output_depths: Float[Tensor, "Bt 1 H W"],
    poses_input: Float[Tensor, "Bt 4 4"],
    poses_output: Float[Tensor, "Bt 4 4"],
    min_depth: Float[Tensor, "B "],
    max_depth: Float[Tensor, "B "],
    scene_scale: Float[Tensor, "B "],
    output_folder: str = "./debug_output",
) -> Float[Tensor, "Bt"]:

    input_rgbs = ((input_rgbs + 1.0) / 2.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0
    output_rgbs = ((output_rgbs + 1.0) / 2.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0
    input_depths = descale_depth((input_depths + 1.0) / 2.0, min_depth, max_depth).permute(0, 2, 3, 1).cpu().numpy()
    output_depths = descale_depth((output_depths + 1.0) / 2.0, min_depth, max_depth).permute(0, 2, 3, 1).cpu().numpy()

    input_rgb_ply = o3d.geometry.PointCloud()
    img_height, img_width = input_rgbs.shape[1], input_rgbs.shape[2]
    for id, (rgb, depth, c2w_pose) in enumerate(zip(input_rgbs, input_depths, poses_input)):
        # recover depth to pointcloud
        o3d_recon_ply = rgbd_to_pointcloud(
            rgb_image=rgb,
            depth_image=depth[:, :, 0],
            c2w_pose=c2w_pose.detach().cpu().numpy(),
            depth_scale=scene_scale.cpu().numpy(),
        )
        input_rgb_ply += o3d_recon_ply

    # evaluate reprojection error
    input_points = np.asarray(input_rgb_ply.points).T  # 3xN
    input_colors = np.asarray(input_rgb_ply.colors).T  # 3xN
    K = np.array(
        [
            [img_width / 2.0, 0, img_width // 2],
            [0, img_height / 2.0, img_height // 2],
            [0, 0, 1],
        ]
    )
    avg_reproj_error = []
    for idx, (output_rgb, c2w_pose) in enumerate(zip(output_rgbs, poses_output)):
        w2c_pose = np.linalg.inv(c2w_pose.detach().cpu().numpy())
        projected_points = (np.dot(w2c_pose[:3, :3], input_points).T + w2c_pose[:3, 3].T).T
        tmp = np.dot(K, projected_points)
        projections = tmp[:, tmp[2] > 0]
        projections_colors = input_colors[:, tmp[2] > 0]
        projections = projections / projections[2]
        valid_px = np.logical_and(projections[0] >= 0, projections[0] < img_width)
        print(f"valid_px: {valid_px.shape}")
        valid_py = np.logical_and(projections[1] >= 0, projections[1] < img_height)
        valid_p = np.logical_and(valid_px, valid_py)
        print(f"valid_p: {valid_p.shape}")
        projections = projections[:, valid_p]
        print(f"projections: {projections.shape}")
        projections_colors = projections_colors[:, valid_p]
        print(f"projections_colors: {projections_colors.shape}")
        if projections.shape[1] > 0:
            visual_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            import copy

            visual_img[projections[1].astype(int), projections[0].astype(int)] = projections_colors.T * 255
            Image.fromarray(visual_img).save(os.path.join(output_folder, f"output_rgb_{idx}_reproj.png"))
            # calculate our reprojection error, L1 loss
            output_rgb = output_rgb / 255.0
            reproj_error = output_rgb[projections[1].astype(int), projections[0].astype(int)] - projections_colors.T
            reproj_err_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            reproj_err_img[projections[1].astype(int), projections[0].astype(int)] = np.abs(reproj_error) * 255
            Image.fromarray(reproj_err_img).save(os.path.join(output_folder, f"output_rgb_{idx}_reproj_err.png"))
            avg_reproj_error.append(np.mean(np.abs(reproj_error)))

    return avg_reproj_error / len(avg_reproj_error)

def check_depth_consistency(reproj_pt2ds, gt_depth_map, reproj_z, z_thresh, n_in_views: int = 3, n_pts_per_view: int = 512*512):
    H, W = gt_depth_map.shape[-2:]
    grid = torch.clone(reproj_pt2ds[:2, :]).transpose(1, 0).view(n_in_views, n_pts_per_view, 1, 2)
    grid[..., 0] = (grid[..., 0] / float(W - 1)) * 2 - 1.0  # normalize to [-1, 1]
    grid[..., 1] = (grid[..., 1] / float(H - 1)) * 2 - 1.0  # normalize to [-1, 1]
    gt_depth_map = gt_depth_map[None, None, :, :].repeat(n_in_views, 1, 1, 1)
    gt_z_sample = F.grid_sample(gt_depth_map, grid, mode="nearest", align_corners=True, padding_mode="zeros")
    gt_z_sample = gt_z_sample.squeeze(1).squeeze(-1).reshape(-1)
    z_diff = torch.abs(reproj_z - gt_z_sample)
    valid_z_mask = z_diff < z_thresh
    return valid_z_mask

def descale_metric_log_normalization(depths: Float[Tensor, "N 1 H W"], min_value: float = 0.05, max_value: float = 12.593009948730469, pose_scale: float = 1.0) -> Float[Tensor, "N 1 H W"]:
    """
    descale metric depth to original value
    params:
        depths: [N, 1, H, W]
        min_value: min value of depth maps
        max_value: max value of depth maps
    Returns:
        descaled_depths: [N, 1, H, W]
    """
    descaled_depths = torch.exp(depths * torch.log(max_value / min_value)) * min_value
    return descaled_depths / pose_scale
    
def cross_view_point_rendering(
    input_rgbs: Float[Tensor, "Ni 3 H W"],
    input_depths: Float[Tensor, "Ni 3 H W"],
    output_depths: Float[Tensor, "No 3 H W"],
    input_rays_od: Float[Tensor, "Ni 6 H W"],
    poses_output: Float[Tensor, "No 4 4"],
    intrinsic: Float[Tensor, "3 3"],
    z_thresh: float = 0.01,
    min_depth: float = 0.05,
    max_depth: float = 12.593009948730469,
    pose_scale: float = 1.0,
    scene_scale: float = 0.0794091328499913,
    use_metric_depth: bool = False,
    depth_perturb_std_max: float = 0.1,
    do_depth_perturb: bool = True,
) -> Float[Tensor, "No 3 H W"]:
    """
    Rendering input RGBD to output views.
    Here we take the output depth maps as the reference to filter out the invalid points caused by occlusion.
    params:
    input_rgbs: [Ni, 3, H, W], input rgb images. Normalized to [-1, 1]
    input_depths: [Ni, 1, H, W], input depth maps. Normalized to [-1, 1]
    output_depths: [No, 1, H, W], gt output depth maps. Normalized to [-1, 1]
    input_rays_od: [Ni, 6, H, W], input rays in the world coordinate system.
    poses_output: [No, 4, 4], camera poses of the output views.
    intrinsic: [3, 3], intrinsic matrix.
    z_thresh: float, z threshold for depth consistency check.
    min_depth: float, min depth value.
    max_depth: float, max depth value.
    use_metric_depth: bool, whether to use metric depth.
    """

    # recover the rgb-d to pointcloud
    input_rgbs = (input_rgbs + 1.0) / 2.0

    input_depths = (input_depths[:,0:1,:,:] + 1.0) / 2.0
    output_depths = (output_depths[:,0:1,:,:] + 1.0) / 2.0
    
    if use_metric_depth:
        input_depths = descale_metric_log_normalization(input_depths, min_depth, max_depth, pose_scale=pose_scale)
        output_depths = descale_metric_log_normalization(output_depths, min_depth, max_depth, pose_scale=pose_scale)
    else:
        input_depths = descale_depth(input_depths, min_depth, max_depth) * scene_scale
        output_depths = descale_depth(output_depths, min_depth, max_depth) * scene_scale
    # perturb the input depth maps to mimic the monocular depth estimation behavior
    depth_perturb_std_max = depth_perturb_std_max if use_metric_depth else depth_perturb_std_max * 0.1
    if do_depth_perturb:
        depth_perturb_std = random.uniform(0.0, depth_perturb_std_max)
        input_depths = torch.normal(mean=input_depths, std=depth_perturb_std)
        
    for idx in range(input_depths.shape[0]):
        depth_np = input_depths[idx].clone().permute(1, 2, 0).cpu().numpy()
        edge_mask = edge_filter(depth_np, times=0.1)
        if len(edge_mask.shape) == 2:
            edge_mask = edge_mask[:, :, np.newaxis]
        depth_np[edge_mask] = 0.0
        input_depths[idx] = torch.from_numpy(depth_np).permute(2, 0, 1).to(input_depths.device)
        
    H, W = input_rgbs.shape[-2:]
    n_in_views, n_out_views = input_rgbs.shape[0], output_depths.shape[0]
    n_pts_per_view = H * W
    # recover the input point cloud
    input_ray_os, input_ray_ds = input_rays_od.chunk(2, dim=1)
    focal_x = float(intrinsic[0, 0].cpu().numpy())
    normalized_focal_len = 2.0 * focal_x / W
    input_distances: Float[Tensor, "Ni 1 H W"] = batch_convert_z_to_distance(input_depths, image_height=H, image_width=W, focal_length=normalized_focal_len)
    in_points: Float[Tensor, "Ni 3 H W"] = input_ray_os + input_ray_ds * input_distances
    input_colors: Float[Tensor, "3 Np"] = rearrange(input_rgbs, "Ni c H W -> c (Ni H W)", Ni=n_in_views, H=H, W=W)
    input_points: Float[Tensor, "3 Np"] = rearrange(in_points, "Ni c H W -> c (Ni H W)", Ni=n_in_views, H=H, W=W)
    
    # project to the output view
    w2c_output_poses = torch.inverse(poses_output)
    batch_input_points = input_points.unsqueeze(0).repeat(n_out_views, 1, 1)
    batch_intrinsics = intrinsic.unsqueeze(0).repeat(n_out_views, 1, 1)
    pts_reproj = torch.bmm(w2c_output_poses[:, :3, :3], batch_input_points) + w2c_output_poses[:, :3, 3, None]
    pts_reproj: Float[Tensor, "No 3 Np"] = torch.bmm(batch_intrinsics, pts_reproj)
    # print(f"pts_reproj shape: {pts_reproj.shape}")

    zs_reproj: Float[Tensor, "No Np"] = pts_reproj[:, 2]
    pts_reproj: Float[Tensor, "No 3 Np"] = pts_reproj / zs_reproj.unsqueeze(1)

    valid_z_mask: Float[Tensor, "No Np"] = zs_reproj > 1e-4
    valid_x_mask: Float[Tensor, "No Np"] = (pts_reproj[:, 0] >= 0.0) & (pts_reproj[:, 0] <= float(W - 1))
    valid_y_mask: Float[Tensor, "No Np"] = (pts_reproj[:, 1] >= 0.0) & (pts_reproj[:, 1] <= float(H - 1))
    
    valid_masks: Float[Tensor, "No Np"] = valid_x_mask & valid_y_mask & valid_z_mask
    # get final projected image
    rendered_imgs = []
    for out_idx in range(n_out_views):
        # mask on this view
        mask = valid_masks[out_idx]
        # gt depth map on this view
        gt_depth = output_depths[out_idx, 0]
        # projected 2d points on this view
        reproj_pt2ds = pts_reproj[out_idx, :, :]
        # projected z values on this view
        z_reproj = zs_reproj[out_idx, :]

        # check z consistency w.r.t. the ground truth depth map
        consist_z_mask = check_depth_consistency(reproj_pt2ds=reproj_pt2ds,
                                                gt_depth_map=gt_depth,
                                                reproj_z=z_reproj,
                                                z_thresh=z_thresh,
                                                n_in_views=n_in_views,
                                                n_pts_per_view=n_pts_per_view)
        # print(f"mask>0: {mask.sum()}, consist_z_mask>0: {consist_z_mask.sum()}")
        mask = mask & consist_z_mask
        valid_pts_reproj = reproj_pt2ds[:, mask]
        rendered_img = torch.zeros((3, H, W), dtype=torch.float32)
        
        if valid_pts_reproj.shape[1] > 0:
            valid_pt2d_xs = valid_pts_reproj[0].int()
            valid_pt2d_ys = valid_pts_reproj[1].int()
            # print(valid_pt2d_xs.shape, valid_pt2d_ys.shape)
            local_win_xs = torch.cat([(valid_pt2d_xs - 1).clamp(0, W-1), 
                                   valid_pt2d_xs, 
                                   (valid_pt2d_xs + 1).clamp(0, W-1)], dim=0) 
            local_win_ys = torch.cat([(valid_pt2d_ys - 1).clamp(0, H-1),
                                      valid_pt2d_ys, 
                                      (valid_pt2d_ys + 1).clamp(0, H-1)], dim=0)
            # print(local_win_xs.shape, local_win_ys.shape)
            rendered_img[:, local_win_ys, local_win_xs] = input_colors[:, mask].repeat(1, 3)           
            # rendered_img[:, valid_pt2d_ys, valid_pt2d_xs] = input_colors[:, mask]
            # blending the pixels around local window
            # for dx in range(-1, 2):
            #     for dy in range(-1, 2):
            #         valid_pt_xs = (valid_pt2d_xs + dx).clamp(0, W-1)
            #         valid_pt_ys = (valid_pt2d_ys + dy).clamp(0, H-1)
            #         rendered_img[:, valid_pt_ys, valid_pt_xs] = input_colors[:, mask]

        rendered_imgs.append(rendered_img.unsqueeze(0))

    rendered_imgs = torch.cat(rendered_imgs, dim=0)
    # save_image(rendered_imgs, "rendered_imgs.png")
    return rendered_imgs

def cross_viewpoint_rendering(
    input_rgbs: Float[Tensor, "B Ni 3 H W"],
    input_depths: Float[Tensor, "B Ni 3 H W"],
    poses_output: Float[Tensor, "No 4 4"],
    intrinsic: Float[Tensor, "B 3 3"],
    min_depth: Float[Tensor, "B "],
    max_depth: Float[Tensor, "B "],
    scene_scale: Float[Tensor, "B "],
    use_scene_coord_map: bool = False,
) -> Float[Tensor, "No 3 H W"]:
    """
    Rendering input RGBD to output views.
    TODO: take the output depth maps as the reference to filter out the invalid points caused by occlusion.
    params:
    input_rgbs: [B, Ni, 3, H, W], input rgb images. Normalized to [-1, 1]
    input_depths: [B, Ni, 3, H, W], input depth maps. Normalized to [-1, 1]
    poses_output: [B, No, 4, 4], camera poses of the output views.
    intrinsic: [B, 3, 3], intrinsic matrix.
    z_thresh: float, z threshold for depth consistency check.
    min_depth: float, min depth value.
    max_depth: float, max depth value.
    use_scene_coord_map: bool, whether to use scene coordinate map.
    """
    assert use_scene_coord_map == True, "do not use_scene_coord_map is not supported yet."
    
    bsz, n_in_views, _, H, W = input_rgbs.shape
    n_out_views = poses_output.shape[1]
    n_pixels = H * W
    # recover the rgb-d to pointcloud
    input_rgbs = (input_rgbs + 1.0) / 2.0

    input_depths = input_depths * 0.5 + 0.5
    input_depths = descale_depth(input_depths, min_depth[:, None, None, None, None], max_depth[:, None, None, None, None]) * scene_scale[:, None, None, None, None]
        
        
    batch_input_points: Float[Tensor, "B 3 Np"] = rearrange(input_depths, "B Ni C H W -> B C (Ni H W)", Ni=n_in_views, H=H, W=W)
    batch_input_colors: Float[Tensor, "B 3 Np"] = rearrange(input_rgbs, "B Ni C H W -> B C (Ni H W)", Ni=n_in_views, H=H, W=W)
    
    batch_input_points: Float[Tensor, "B No 3 Np"] = batch_input_points.unsqueeze(1).repeat(1, n_out_views, 1, 1)
    batch_input_colors: Float[Tensor, "B No 3 Np"] = batch_input_colors.unsqueeze(1).repeat(1, n_out_views, 1, 1)
    
    # project to the output view
    w2c_output_poses: Float[Tensor, "B No 4 4"] = torch.inverse(poses_output.float()).to(input_rgbs.dtype)
    
    batch_intrinsics = intrinsic.unsqueeze(1).repeat(1, n_out_views, 1, 1)
    pts_reproj = torch.matmul(w2c_output_poses[:, :, :3, :3], batch_input_points) + w2c_output_poses[:, :, :3, 3:4]
    pts_reproj: Float[Tensor, "B No 3 Np"] = torch.matmul(batch_intrinsics, pts_reproj)

    zs_reproj: Float[Tensor, "B No Np"] = pts_reproj[:, :, 2]
    pts_reproj: Float[Tensor, "B No 3 Np"] = pts_reproj / zs_reproj.unsqueeze(2)

    valid_z_mask: Float[Tensor, "B No Np"] = zs_reproj > 1e-4
    valid_x_mask: Float[Tensor, "B No Np"] = (pts_reproj[:, :, 0] >= 0.0) & (pts_reproj[:, :, 0] <= float(W - 1))
    valid_y_mask: Float[Tensor, "B No Np"] = (pts_reproj[:, :, 1] >= 0.0) & (pts_reproj[:, :, 1] <= float(H - 1))
    
    valid_masks: Float[Tensor, "B No Np"] = valid_x_mask & valid_y_mask & valid_z_mask
    
    pts_reproj: Float[Tensor, "B No Np 3"] = pts_reproj.permute(0, 1, 3, 2)
    # valid_pts_reproj = pts_reproj[valid_masks, :]
    # print(f"valid_pts_reproj shape: {valid_pts_reproj.shape}")
    # valid_pt2d_xs = valid_pts_reproj[:, 0].int()
    # valid_pt2d_ys = valid_pts_reproj[:, 1].int()
    # print(f"valid_pt2d_xs shape: {valid_pt2d_xs.shape}, valid_pt2d_ys shape: {valid_pt2d_ys.shape}")
    # rendered_imgs = torch.zeros((bsz, n_out_views, H, W, 3)).to(input_rgbs)
    # batch_input_colors = batch_input_colors.permute(0, 1, 3, 2)
    # rendered_imgs[:, :, valid_pt2d_ys, valid_pt2d_xs] = batch_input_colors[valid_masks]
    
    batch_input_colors: Float[Tensor, "B No Np 3"] = batch_input_colors.permute(0, 1, 3, 2)
    # get final projected image
    rendered_imgs = []
    for b in range(bsz):
        for out_view_idx in range(n_out_views):
            # mask on this view
            mask = valid_masks[b, out_view_idx]
            # projected 2d points on this view
            reproj_pt2ds = pts_reproj[b, out_view_idx, :, :]
            valid_pts_reproj = reproj_pt2ds[mask, :]

            rendered_img = torch.zeros((H, W, 3)).to(input_rgbs)
            
            if valid_pts_reproj.shape[0] > 0:
                valid_pt2d_xs = valid_pts_reproj[:, 0].int()
                valid_pt2d_ys = valid_pts_reproj[:, 1].int()
                # print(valid_pt2d_xs.shape, valid_pt2d_ys.shape)
                local_win_xs = torch.cat([(valid_pt2d_xs - 2).clamp(0, W-1),
                                        #   (valid_pt2d_xs - 1).clamp(0, W-1), 
                                          valid_pt2d_xs, 
                                        #   (valid_pt2d_xs + 1).clamp(0, W-1),
                                          (valid_pt2d_xs + 2).clamp(0, W-1)], dim=0) 
                local_win_ys = torch.cat([(valid_pt2d_ys - 2).clamp(0, H-1),
                                        #   (valid_pt2d_ys - 1).clamp(0, H-1),
                                          valid_pt2d_ys, 
                                        #   (valid_pt2d_ys + 1).clamp(0, H-1),
                                          (valid_pt2d_ys + 2).clamp(0, H-1)], dim=0)
                rendered_img[local_win_ys, local_win_xs, :] = batch_input_colors[b, out_view_idx, mask, :].repeat(3, 1)           

            rendered_imgs.append(rendered_img.unsqueeze(0))

    rendered_imgs: Float[Tensor, "BNv H W 3"] = torch.cat(rendered_imgs, dim=0)
    # save_image(rendered_imgs.permute(0, 3, 1, 2), "rendered_imgs.png", nrow=1)
    # print(f"rendered_imgs shape: {rendered_imgs.shape}")
    rendered_imgs = rearrange(rendered_imgs, "(B No) H W C-> B No C H W", B=bsz, No=n_out_views)
    return rendered_imgs

from src.utils.torch3d_utils import torch3d_rasterize_points
def cross_viewpoint_rendering_pt3d(
    input_rgbs: Float[Tensor, "B Ni 3 H W"],
    input_depths: Float[Tensor, "B Ni 3 H W"],
    poses_output: Float[Tensor, "No 4 4"],
    intrinsic: Float[Tensor, "B 3 3"],
    min_depth: Float[Tensor, "B "],
    max_depth: Float[Tensor, "B "],
    scene_scale: Float[Tensor, "B "],
    use_scene_coord_map: bool = False,
) -> Float[Tensor, "No 3 H W"]:
    """
    Rendering input RGBD to output views.
    TODO: take the output depth maps as the reference to filter out the invalid points caused by occlusion.
    params:
    input_rgbs: [B, Ni, 3, H, W], input rgb images. Normalized to [-1, 1]
    input_depths: [B, Ni, 3, H, W], input depth maps. Normalized to [-1, 1]
    poses_output: [B, No, 4, 4], camera poses of the output views.
    intrinsic: [B, 3, 3], intrinsic matrix.
    z_thresh: float, z threshold for depth consistency check.
    min_depth: float, min depth value.
    max_depth: float, max depth value.
    use_scene_coord_map: bool, whether to use scene coordinate map.
    """
    assert use_scene_coord_map == True, "do not use_scene_coord_map is not supported yet."
    
    bsz, n_in_views, _, H, W = input_rgbs.shape
    n_out_views = poses_output.shape[1]
    # recover the rgb-d to pointcloud
    input_rgbs = (input_rgbs + 1.0) / 2.0

    input_depths = input_depths * 0.5 + 0.5
    input_depths = descale_depth(input_depths, min_depth[:, None, None, None, None], max_depth[:, None, None, None, None]) * scene_scale[:, None, None, None, None]
        
        
    batch_input_points: Float[Tensor, "B Np 3"] = rearrange(input_depths, "B Ni C H W -> B (Ni H W) C", Ni=n_in_views, H=H, W=W)
    batch_input_colors: Float[Tensor, "B Np 3"] = rearrange(input_rgbs, "B Ni C H W -> B (Ni H W) C", Ni=n_in_views, H=H, W=W)
    
    batch_pointcloud: Float[Tensor, "B Np 6"] = torch.cat([batch_input_points, batch_input_colors], dim=-1)
    
    rendered_imgs = []
    for b in range(bsz):
        pointcloud: Float[Tensor, "Np 6"] = batch_pointcloud[b].float()
        tar_c2w_poses: Float[Tensor, "Np 4 4"] = poses_output[b].float()
        intrinisc_mat = intrinsic[b].float()
        projected_tar_imgs, projected_tar_depths = torch3d_rasterize_points(
                        cv_cam_poses_c2w=tar_c2w_poses,
                        in_pointcloud=pointcloud,
                        intrinsic=intrinisc_mat,
                        image_width=W,
                        image_height=H,
                        point_radius=0.01,
                        device=input_rgbs.device,
                    )
        rendered_img: Float[Tensor, "Np 3 H W"] = projected_tar_imgs
        rendered_imgs.append(rendered_img)
        
    rendered_imgs: Float[Tensor, "BNp 3 H W"] = torch.cat(rendered_imgs, dim=0)
    # save_image(rendered_imgs, "rendered_imgs.png", nrow=1)
    # print(f"rendered_imgs shape: {rendered_imgs.shape}")
    rendered_imgs = rearrange(rendered_imgs, "(B No) C H W-> B No C H W", B=bsz, No=n_out_views)
    return rendered_imgs

def cross_view_point_projecting(
    input_rgbs: Float[Tensor, "Ni 3 H W"],
    input_depths: Float[Tensor, "Ni 3 H W"],
    input_rays_od: Float[Tensor, "Ni 6 H W"],
    poses_output: Float[Tensor, "No 4 4"],
    intrinsic: Float[Tensor, "3 3"],
    min_depth: float = 0.05,
    max_depth: float = 12.593009948730469,
    scene_scale: float = 0.0794091328499913,
    use_metric_depth: bool = False,
) -> Float[Tensor, "No 3 H W"]:
    """
    Rendering input RGBD to output views.
    Here we donot have the gt output depth maps, thus cannot filter out the invalid points caused by occlusion.
    params:
    input_rgbs: [Ni, 3, H, W], input rgb images. Normalized to [-1, 1]
    input_depths: [Ni, 1, H, W], input depth maps. Normalized to [-1, 1]
    input_rays_od: [Ni, 6, H, W], input rays in the world coordinate system.
    poses_output: [No, 4, 4], camera poses of the output views.
    intrinsic: [3, 3], intrinsic matrix.
    min_depth: float, min depth value.
    max_depth: float, max depth value.
    use_metric_depth: bool, whether to use metric depth.
    """

    # recover the rgb-d to pointcloud
    input_rgbs = (input_rgbs + 1.0) / 2.0

    input_depths = (input_depths[:,0:1,:,:] + 1.0) / 2.0
    
    if use_metric_depth:
        input_depths = descale_metric_log_normalization(input_depths, min_depth, max_depth)
    else:
        input_depths = descale_depth(input_depths, min_depth, max_depth) * scene_scale

    for idx in range(input_depths.shape[0]):
        depth_np = input_depths[idx].clone().permute(1, 2, 0).cpu().numpy()
        edge_mask = edge_filter(depth_np, times=0.1)
        if len(edge_mask.shape) == 2:
            edge_mask = edge_mask[:, :, np.newaxis]
        depth_np[edge_mask] = 0.0
        input_depths[idx] = torch.from_numpy(depth_np).permute(2, 0, 1).to(input_depths.device)
    
    H, W = input_rgbs.shape[-2:]
    n_in_views, n_out_views = input_rgbs.shape[0], poses_output.shape[0]
    n_pts_per_view = H * W
    # recover the input point cloud
    input_ray_os, input_ray_ds = input_rays_od.chunk(2, dim=1)
    focal_x = float(intrinsic[0, 0].cpu().numpy())
    normalized_focal_len = 2.0 * focal_x / W
    input_distances: Float[Tensor, "Ni 1 H W"] = batch_convert_z_to_distance(input_depths, image_height=H, image_width=W, focal_length=normalized_focal_len)
    in_points: Float[Tensor, "Ni 3 H W"] = input_ray_os + input_ray_ds * input_distances
    input_colors: Float[Tensor, "3 Np"] = rearrange(input_rgbs, "Ni c H W -> c (Ni H W)", Ni=n_in_views, H=H, W=W)
    input_points: Float[Tensor, "3 Np"] = rearrange(in_points, "Ni c H W -> c (Ni H W)", Ni=n_in_views, H=H, W=W)
    
    input_pointcloud = torch.cat([input_points, input_colors], dim=0).permute(1,0)
    from utils.torch3d_utils import render_rgb_and_depth_from_ply
    rendered_tar_rgbs, rendered_tar_depths, rendered_tar_masks = render_rgb_and_depth_from_ply(
        input_pointcloud=input_pointcloud.to('cuda'),
        cv_cam_poses_c2w=poses_output.to('cuda'),
        intrinsic=intrinsic.to('cuda'),
        image_width=W,
        image_height=H,
        point_radius=0.02,
        device='cuda'
    )
    rendered_tar_rgbs = rendered_tar_rgbs.permute(0, 3, 1, 2)
    rendered_tar_depths = rendered_tar_depths.permute(0, 3, 1, 2)
    rendered_tar_masks = rendered_tar_masks.permute(0, 3, 1, 2)
    # rendered_tar_rgbs = torchvision.utils.make_grid(rendered_tar_rgbs, nrow=1)
    # save_image(rendered_tar_rgbs, 'rendered_tar_rgbs.png')
    # save_image(rendered_tar_depths, 'rendered_tar_depths.png')
    # save_image(rendered_tar_masks, 'rendered_tar_masks.png')
    # # project to the output view
    # w2c_output_poses = torch.inverse(poses_output)
    # batch_input_points = input_points.unsqueeze(0).repeat(n_out_views, 1, 1)
    # batch_intrinsics = intrinsic.unsqueeze(0).repeat(n_out_views, 1, 1)
    # pts_reproj = torch.bmm(w2c_output_poses[:, :3, :3], batch_input_points) + w2c_output_poses[:, :3, 3, None]
    # pts_reproj: Float[Tensor, "No 3 Np"] = torch.bmm(batch_intrinsics, pts_reproj)
    # # print(f"pts_reproj shape: {pts_reproj.shape}")

    # zs_reproj: Float[Tensor, "No Np"] = pts_reproj[:, 2]
    # pts_reproj: Float[Tensor, "No 3 Np"] = pts_reproj / zs_reproj.unsqueeze(1)

    # valid_z_mask: Float[Tensor, "No Np"] = zs_reproj > 1e-4
    # valid_x_mask: Float[Tensor, "No Np"] = (pts_reproj[:, 0] >= 0.0) & (pts_reproj[:, 0] <= float(W - 1))
    # valid_y_mask: Float[Tensor, "No Np"] = (pts_reproj[:, 1] >= 0.0) & (pts_reproj[:, 1] <= float(H - 1))
    
    # valid_masks: Float[Tensor, "No Np"] = valid_x_mask & valid_y_mask & valid_z_mask

    # # get final projected image
    # rendered_imgs = []
    # for out_idx in range(n_out_views):
    #     # mask on this view
    #     mask = valid_masks[out_idx]
    #     # projected 2d points on this view
    #     reproj_pt2ds = pts_reproj[out_idx, :, :]
    #     # projected z values on this view
    #     z_reproj = zs_reproj[out_idx, :]

    #     valid_pts_reproj = reproj_pt2ds[:, mask]
    #     rendered_img = torch.zeros((3, H, W), dtype=torch.float32)
        
    #     if valid_pts_reproj.shape[1] > 0:
    #         valid_pt2d_xs = valid_pts_reproj[0].int()
    #         valid_pt2d_ys = valid_pts_reproj[1].int()
    #         # print(valid_pt2d_xs.shape, valid_pt2d_ys.shape)
    #         local_win_xs = torch.cat([(valid_pt2d_xs - 1).clamp(0, W-1), 
    #                                valid_pt2d_xs, 
    #                                (valid_pt2d_xs + 1).clamp(0, W-1)], dim=0) 
    #         local_win_ys = torch.cat([(valid_pt2d_ys - 1).clamp(0, H-1),
    #                                   valid_pt2d_ys, 
    #                                   (valid_pt2d_ys + 1).clamp(0, H-1)], dim=0)
    #         rendered_img[:, local_win_ys, local_win_xs] = input_colors[:, mask].repeat(1, 3)           

    #     rendered_imgs.append(rendered_img.unsqueeze(0))

    # rendered_imgs = torch.cat(rendered_imgs, dim=0)
    # save_image(rendered_imgs, "cross_view_point_projecting.png")
    return rendered_tar_rgbs


def process_depth(
    ref_depth,
    ref_image,
    src_depths,
    src_images,
    ref_P_c2w,
    src_Ps_c2w,
    ref_K,
    src_Ks,
    z_thresh=0.1,
    n_consistent_thresh=3,
    ref_idx=0,
):
    """
    process depth map for depth consistency check

    """
    n_src_imgs = src_depths.shape[0]
    h, w = ref_depth.shape[-2:]
    n_pts = h * w

    src_Ks = src_Ks.cuda()
    src_Ps_c2w = src_Ps_c2w.cuda()
    ref_K = ref_K.cuda()
    ref_P_c2w = ref_P_c2w.cuda()
    ref_depth = ref_depth.cuda()

    ref_K_inv = torch.inverse(ref_K)
    src_Ks_inv = torch.inverse(src_Ks)
    # ref_P_inv = torch.inverse(ref_P_c2w)

    pts_x = np.linspace(0, w - 1, w)
    pts_y = np.linspace(0, h - 1, h)
    pts_xx, pts_yy = np.meshgrid(pts_x, pts_y)

    pts = torch.from_numpy(np.stack((pts_xx, pts_yy, np.ones_like(pts_xx)), axis=0)).float().cuda()
    pts = ref_P_c2w[:3, :3] @ (ref_K_inv @ (pts * ref_depth).view(3, n_pts)) + ref_P_c2w[:3, 3, None]
    original_pts = pts.clone().cpu().numpy().transpose(1, 0)
    # cam_pcl = o3d.geometry.PointCloud()
    # cam_pcl.points = o3d.utility.Vector3dVector(pts.cpu().numpy().T)
    # cam_pcl.colors = o3d.utility.Vector3dVector(ref_image.view(-1, 3).cpu().numpy())
    # o3d.io.write_point_cloud(f"{ref_idx}_cam_pcl.ply", cam_pcl)

    n_valid = 0.0
    pts_sample_all = []
    valid_per_src_all = []
    idx_start = 0
    idx_end = n_src_imgs
    src_Ps_batch = torch.inverse(src_Ps_c2w[idx_start:idx_end])  # w2c
    src_Ks_batch = src_Ks[idx_start:idx_end]
    src_Ks_inv_batch = src_Ks_inv[idx_start:idx_end]
    src_depths_batch = src_depths[idx_start:idx_end].cuda()

    n_batch_imgs = idx_end - idx_start
    pts_reproj = torch.bmm(src_Ps_batch[:, :3, :3], pts.unsqueeze(0).repeat(n_batch_imgs, 1, 1)) + src_Ps_batch[:, :3, 3, None]
    pts_reproj = torch.bmm(src_Ks_batch, pts_reproj)
    # logger.info(f"pts_reproj shape: {pts_reproj.shape}")

    z_reproj = pts_reproj[:, 2]
    pts_reproj = pts_reproj / z_reproj.unsqueeze(1)
    # logger.info(f"pts_reproj shape: {pts_reproj.shape}")

    valid_z = z_reproj > 1e-4
    valid_x = (pts_reproj[:, 0] >= 0.0) & (pts_reproj[:, 0] <= float(w - 1))
    valid_y = (pts_reproj[:, 1] >= 0.0) & (pts_reproj[:, 1] <= float(h - 1))

    grid = torch.clone(pts_reproj[:, :2]).transpose(2, 1).view(n_batch_imgs, n_pts, 1, 2)
    grid[..., 0] = (grid[..., 0] / float(w - 1)) * 2 - 1.0  # normalize to [-1, 1]
    grid[..., 1] = (grid[..., 1] / float(h - 1)) * 2 - 1.0  # normalize to [-1, 1]
    z_sample = F.grid_sample(src_depths_batch, grid, mode="nearest", align_corners=True, padding_mode="zeros")
    # logger.info(f"z_sample shape: {z_sample.shape}")
    z_sample = z_sample.squeeze(1).squeeze(-1)
    # logger.info(f"z_sample shape: {z_sample.shape}")

    z_diff = torch.abs(z_reproj - z_sample)
    valid_disp = z_diff < z_thresh
    # logger.info(f"valid_disp shape: {valid_disp.shape}")

    valid_per_src = valid_disp & valid_x & valid_y & valid_z
    n_valid += torch.sum(valid_per_src.int(), dim=0)

    # back project sampled pts for later averaging
    pts_sample = torch.bmm(src_Ks_inv_batch, pts_reproj * z_sample.unsqueeze(1))
    # logger.info(f"pts_sample shape: {pts_sample.shape}")
    pts_sample = torch.bmm(
        src_Ps_batch[:, :3, :3].transpose(2, 1),
        pts_sample - src_Ps_batch[:, :3, 3, None],
    )
    # logger.info(f"pts_sample shape: {pts_sample.shape}")
    pts_sample_all.append(pts_sample)
    valid_per_src_all.append(valid_per_src)
    pts_sample_all = torch.cat(pts_sample_all, dim=0)
    valid_per_src_all = torch.cat(valid_per_src_all, dim=0)

    valid = n_valid >= n_consistent_thresh
    logger.info(f"valid points: {len(valid[valid==True])}")
    # average sampled points amongst consistent views
    pts_avg = pts
    for i in range(n_src_imgs):
        pts_sample_i = pts_sample_all[i]
        invalid_idx = torch.isnan(pts_sample_i)  # filter out NaNs from div/0 due to grid sample zero padding
        pts_sample_i[invalid_idx] = 0.0
        valid_i = valid_per_src_all[i] & ~torch.any(invalid_idx, dim=0)
        pts_avg += pts_sample_i * valid_i.float().unsqueeze(0)
    pts_avg = pts_avg / (n_valid + 1).float().unsqueeze(0).expand(3, n_pts)

    pts_filtered = pts_avg.transpose(1, 0)[valid].cpu().numpy()
    valid = valid.view(ref_depth.shape[-2:])
    rgb_filtered = ref_image[valid].view(-1, 3).cpu().numpy()
    original_rgb = ref_image.view(-1, 3).cpu().numpy()
    return original_pts, original_rgb, pts_filtered, rgb_filtered, valid.cpu().numpy()


def fuse_scene_pcl(
    input_images: Float[Tensor, "Bt 3 H W"],
    input_depths: Float[Tensor, "Bt 1 H W"],
    output_images: Float[Tensor, "Bt 3 H W"],
    output_depths: Float[Tensor, "Bt 1 H W"],
    poses_input: Float[Tensor, "Bt 4 4"],
    poses_output: Float[Tensor, "Bt 4 4"],
    min_depth: Float[Tensor, "B "],
    max_depth: Float[Tensor, "B "],
    output_folder: str = "./debug_output",
    intrinsic_mat: Float[Tensor, "4 4"] = None,
    is_gt: bool = False,
):
    """
    Fuse scee point cloud from input and output views, with muilt-view depth consistency constraint.
    input_images: [BxT, C, H, W], range [-1, 1], input view rgb images
    input_depths: [BxT, 1, H, W], range [-1, 1], input view depth images
    output_images: [BxT, C, H, W], range [-1, 1], target view rgb images
    output_depths: [BxT, 1, H, W], range [-1, 1], target view depth images
    poses_input: [BxT, 4, 4], input view camera poses
    poses_output: [BxT, 4, 4], target view camera poses
    min_depth: [B, 1], minimum depth value
    max_depth: [B, 1], maximum depth value
    output_folder: str, output folder
    intrinsic_mat: [4, 4], intrinsic matrix

    """
    os.makedirs(output_folder, exist_ok=True)

    prefix_str = "gt_" if is_gt else "pred_"
    logger.info(f"fuse scene point cloud for {prefix_str} images")

    input_rgbs = (input_images + 1.0) / 2.0  # BxT, 3, H, W
    inputs = torchvision.utils.make_grid(input_rgbs, nrow=1)
    torchvision.utils.save_image(inputs, os.path.join(output_folder, "input_rgbs.png"))
    output_rgbs = (output_images + 1.0) / 2.0  # BxT, 3, H, W
    outputs = torchvision.utils.make_grid(output_rgbs, nrow=1)
    torchvision.utils.save_image(outputs, os.path.join(output_folder, f"{prefix_str}tar_rgbs.png"))

    input_depths = descale_depth((input_depths + 1.0) / 2.0, min_depth, max_depth)  # * scene_scale  # BxT, 1, H, W
    output_depths = descale_depth((output_depths + 1.0) / 2.0, min_depth, max_depth)  # * scene_scale  # BxT, 1, H, W

    img_height, img_width = input_rgbs.shape[2], input_rgbs.shape[3]

    cam_intrinsic = np.array(
        [
            [img_width / 2.0, 0, img_width / 2.0],
            [0, img_height / 2.0, img_height / 2.0],
            [0, 0, 1],
        ]
    )
    # create a camera
    camera = o3d.camera.PinholeCameraIntrinsic()
    camera.set_intrinsics(
        img_width,
        img_height,
        cam_intrinsic[0, 0],
        cam_intrinsic[1, 1],
        cam_intrinsic[0, 2],
        cam_intrinsic[1, 2],
    )

    # pred_depths = torch.cat([input_depths, output_depths], dim=0)
    # pred_rgbs = torch.cat([input_rgbs, output_rgbs], dim=0)
    pred_depths = input_depths
    pred_rgbs = input_rgbs
    pred_rgbs = pred_rgbs.permute(0, 2, 3, 1)
    # all_poses = torch.cat([poses_input, poses_output], dim=0)
    all_poses = poses_input

    if intrinsic_mat is not None:
        K = intrinsic_mat.unsqueeze(0).repeat(all_poses.shape[0], 1, 1).to(all_poses.device)
    else:
        hfov = 90.0 * np.pi / 180.0
        fl_x = img_width / 2.0 / np.tan((hfov / 2.0))
        K = torch.tensor(
            [
                [fl_x, 0.0, img_width / 2.0],
                [0.0, fl_x, img_height / 2.0],
                [
                    0.0,
                    0.0,
                    1,
                ],
            ],
            dtype=torch.float32,
        )
        K = K.unsqueeze(0).repeat(all_poses.shape[0], 1, 1).to(all_poses.device)

    n_imgs = pred_depths.shape[0]
    original_pts = []
    original_rgb = []
    fused_pts = []
    fused_rgb = []
    all_idx = torch.arange(n_imgs)
    all_valid = []
    for ref_idx in range(n_imgs):
        src_idx = all_idx != ref_idx
        ori_pts, ori_rgb, pts, rgb, valid = process_depth(
            ref_depth=pred_depths[ref_idx],
            ref_image=pred_rgbs[ref_idx],
            src_depths=pred_depths[src_idx],
            src_images=pred_rgbs[src_idx],
            ref_P_c2w=all_poses[ref_idx],
            src_Ps_c2w=all_poses[src_idx],
            ref_K=K[ref_idx],
            src_Ks=K[src_idx],
            z_thresh=0.1,
            n_consistent_thresh=1,
            ref_idx=ref_idx,
        )
        # save depth image
        depth_img = pred_depths[ref_idx].permute(1, 2, 0).cpu().numpy().squeeze()
        Image.fromarray((depth_img * 1000).astype(np.uint16)).save(os.path.join(output_folder, f"{prefix_str}_depth_{ref_idx}.png"))
        # ref_frame_c2w_pose = all_poses[ref_idx].cpu().numpy().astype(np.float32)
        # cam_lines = o3d.geometry.LineSet.create_camera_visualization(
        #     intrinsic=camera, extrinsic=np.linalg.inv(ref_frame_c2w_pose), scale=0.05
        # )
        # cam_lines.paint_uniform_color([0, 0, 1])
        # o3d.io.write_line_set(os.path.join(output_folder, f"cam_{ref_idx}.ply"), cam_lines)

        fused_pts.append(pts)
        fused_rgb.append(rgb)
        original_pts.append(ori_pts)
        original_rgb.append(ori_rgb)
        all_valid.append(valid)
    fused_pts = np.concatenate(fused_pts, axis=0)
    fused_rgb = np.concatenate(fused_rgb, axis=0)
    all_valid = np.stack(all_valid, axis=0)

    pcd_filt = o3d.geometry.PointCloud()
    pcd_filt.points = o3d.utility.Vector3dVector(fused_pts)
    pcd_filt.colors = o3d.utility.Vector3dVector(fused_rgb)
    # pcd_filt = pcd_filt.voxel_down_sample(0.05)
    pcd_filepath = os.path.join(output_folder, f"{prefix_str}filter_reconstructions.ply")
    o3d.io.write_point_cloud(pcd_filepath, pcd_filt)

    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(np.concatenate(original_pts, axis=0))
    pcd_original.colors = o3d.utility.Vector3dVector(np.concatenate(original_rgb, axis=0))
    pcd_filepath = os.path.join(output_folder, f"{prefix_str}original_reconstructions.ply")
    o3d.io.write_point_cloud(pcd_filepath, pcd_original)
    return fused_pts, fused_rgb, all_valid

def are_points_collinear(points, tolerance=1e-6):
    """
    检查一组点是否在一条直线上。
    :param points: 点的列表，每个点是一个 3D 坐标 (x, y, z)。
    :param tolerance: 共线性判断的容差。
    :return: True 如果点共线，否则 False。
    """
    if len(points) < 3:
        return True  # 少于 3 个点默认共线

    # 取前两个点确定一条直线
    p0 = points[0]
    p1 = points[1]

    # 计算直线的方向向量
    direction = p1 - p0
    length = np.linalg.norm(direction)
    if length < tolerance:
        return True  # 两点距离太近，共线
    # 计算所有点到直线的距离
    # 点到直线的距离公式：|(p - p0) × direction| / |direction|
    vectors = np.array(points) - p0  # 所有点相对于 p0 的向量
    cross_products = np.cross(vectors, direction)  # 计算叉积
    distances = np.linalg.norm(cross_products, axis=1) / length # 计算距离

    # 检查所有距离是否小于容差
    return np.all(distances < tolerance)

def complete_depth_map(depth_map, max_depth=None, inpaint_radius=15, depth_scale=1000):
    """
    Complete a depth map by filling in zero values using inpainting.
    
    Args:
        depth_map: Input depth map (2D numpy array) with 0 values to be filled
        max_depth: Optional maximum depth value for normalization
        inpaint_radius: Radius for inpainting algorithm
        
    Returns:
        Completed depth map with zeros filled in
    """
    # Normalize depth map to 0-255 range for inpainting
    if max_depth is None:
        max_depth = np.max(depth_map[depth_map > 0])
    
    normalized = np.zeros_like(depth_map, dtype=np.float32)
    mask = (depth_map == 0).astype(np.uint8) * 255  # Mask where depth==0
    if np.all(mask == 0):
        return depth_map
    
    # Avoid division by zero if all values are zero
    if max_depth > 0:
        normalized = (depth_map / max_depth * depth_scale).astype(np.uint16)
    else:
        normalized = depth_map.astype(np.uint16)
    
    # Perform inpainting
    inpainted = cv2.inpaint(normalized, mask, inpaint_radius, cv2.INPAINT_TELEA)
    
    # Convert back to original depth range
    completed_depth = inpainted.astype(np.float32) / depth_scale * max_depth
    
    # Preserve original non-zero values (inpainting might slightly modify them)
    completed_depth[depth_map > 0] = depth_map[depth_map > 0]
    
    return completed_depth
