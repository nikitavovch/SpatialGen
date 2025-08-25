
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
import torch

from src.utils.typing import *
from src.utils.cam_ops import get_ray_directions, get_plucker_rays

def interp_poses(w2c_poses: list[np.ndarray], num_frames: int = 24) -> list[np.ndarray]:
    """interpolate two camera poses"""
    v_rotation_in = np.zeros([0, 4])
    v_pos_x_in = []
    v_pos_y_in = []
    v_pos_z_in = []
    for i, pose in enumerate(w2c_poses):
        v_rotation_in = np.append(v_rotation_in, [Rotation.from_matrix(pose[:3, :3]).as_quat()], axis=0)
        v_pos_x_in.append(pose[0, 3])
        v_pos_y_in.append(pose[1, 3])
        v_pos_z_in.append(pose[2, 3])

    in_times = np.arange(0, len(v_rotation_in)).tolist()
    out_times = np.linspace(0, len(v_rotation_in) - 1, num_frames).tolist()
    v_rotation_in = Rotation.from_quat(v_rotation_in)
    slerp = Slerp(in_times, v_rotation_in)
    v_interp_rotation = slerp(out_times)
    fx = interp1d(in_times, np.array(v_pos_x_in), kind="linear")
    fy = interp1d(in_times, np.array(v_pos_y_in), kind="linear")
    fz = interp1d(in_times, np.array(v_pos_z_in), kind="linear")
    v_interp_xs = fx(out_times)
    v_interp_ys = fy(out_times)
    v_interp_zs = fz(out_times)

    target_poses = []
    for idx in range(len(out_times)):
        rot_matrix = v_interp_rotation[idx].as_matrix()
        trans = np.array([v_interp_xs[idx], v_interp_ys[idx], v_interp_zs[idx]])
        T_c2w = np.eye(4)
        T_c2w[:3, :3] = rot_matrix
        T_c2w[:3, 3] = trans

        target_poses.append(T_c2w)
    return target_poses

def interpolate_camera_trajectory(
    input_view_poses: Float[Tensor, "T 4 4"],
    target_view_poses: Float[Tensor, "T 4 4"],
    fps: int = 25,
    image_width: int = 256,
    image_height: int = 256,
    focal_length: float = 128.0,
):
    """
    Interpolate camera trajectory between input and target view poses.
    Args:
        input_view_poses (torch.Tensor): input view poses (T x 4 x 4)
        target_view_poses (torch.Tensor): target view poses (T x 4 x 4)
        fps (int): frames per second
        image_width (int): image width
        image_height (int): image height
        focal_length (float): focal length
    Returns:
        dict: interpolated data
    """
    device = input_view_poses.device
    num_in_views = input_view_poses.shape[0]
    num_target_views = target_view_poses.shape[0]
    num_total_views = num_in_views + num_target_views

    def interp_seq_poses(w2c_poses: Float[Tensor, "N 4 4"], num_frames: int = 24) -> Float[Tensor, "N 4 4"]:
        """interpolate a sequence camera poses"""
        w2c_poses_np = w2c_poses.cpu().numpy()
        v_rotation_in = np.zeros([0, 4])
        v_pos_x_in = []
        v_pos_y_in = []
        v_pos_z_in = []
        for i, pose in enumerate(w2c_poses_np):
            v_rotation_in = np.append(v_rotation_in, [Rotation.from_matrix(pose[:3, :3]).as_quat()], axis=0)
            v_pos_x_in.append(pose[0, 3])
            v_pos_y_in.append(pose[1, 3])
            v_pos_z_in.append(pose[2, 3])

        in_times = np.arange(0, len(v_rotation_in)).tolist()
        out_times = np.linspace(0, len(v_rotation_in) - 1, num_frames).tolist()
        v_rotation_in = Rotation.from_quat(v_rotation_in)
        slerp = Slerp(in_times, v_rotation_in)
        v_interp_rotation = slerp(out_times)
        fx = interp1d(in_times, np.array(v_pos_x_in), kind="linear")
        fy = interp1d(in_times, np.array(v_pos_y_in), kind="linear")
        fz = interp1d(in_times, np.array(v_pos_z_in), kind="linear")
        v_interp_xs = fx(out_times)
        v_interp_ys = fy(out_times)
        v_interp_zs = fz(out_times)

        target_poses = []
        for idx in range(len(out_times)):
            rot_matrix = v_interp_rotation[idx].as_matrix()
            trans = np.array([v_interp_xs[idx], v_interp_ys[idx], v_interp_zs[idx]])
            T_c2w = np.eye(4)
            T_c2w[:3, :3] = rot_matrix
            T_c2w[:3, 3] = trans

            target_poses.append(torch.from_numpy(T_c2w).float())

        return torch.stack(target_poses, dim=0)

    # interpolate c2w poses
    total_views = torch.cat([input_view_poses, target_view_poses], dim=0)
    # sort by distance to the reference frame
    # total_views = total_views[torch.norm(total_views[:, :3, 3] - input_view_poses[0, :3, 3], dim=1).argsort()]
    
    num_inter_frames = num_total_views * fps
    interpolated_poses = interp_seq_poses(total_views, num_inter_frames)
    interpolated_poses = interpolated_poses.to(device)
    poses = torch.cat([input_view_poses, interpolated_poses], dim=0)

    relative_poses = torch.inverse(poses[0:1]).repeat(poses.shape[0], 1, 1) @ poses
    directions = get_ray_directions(H=image_height, W=image_width, focal=focal_length).to(device)
    cano_ray_dirs: Float[Tensor, "B H W 3"] = directions[None, :, :, :].repeat(relative_poses.shape[0], 1, 1, 1)
    # always use plucker ray
    rays = get_plucker_rays(cano_ray_dirs, relative_poses, keepdim=True)
    rays = rays.permute(0, 3, 1, 2)

    cond_Ts: Float[Tensor, "N 4 4"] = relative_poses[:num_in_views]
    target_Ts: Float[Tensor, "N 4 4"] = relative_poses[num_in_views:]

    input_plucker_rays: Float[Tensor, "N 6 H W"] = rays[:num_in_views]
    target_plucker_rays: Float[Tensor, "N 6 H W"] = rays[num_in_views:]

    data = {}
    # data["room_uid"] = room_uid
    data["pose_out"] = target_Ts
    data["pose_in"] = cond_Ts
    data["plucker_rays_input"] = input_plucker_rays
    data["plucker_rays_target"] = target_plucker_rays
    return data

"""
Code borrowed from

https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/camera_utils.py
"""

import numpy as np
import scipy


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def generate_ellipse_path_z(
    poses: np.ndarray,
    n_frames: int = 120,
    # const_speed: bool = True,
    variation: float = 0.0,
    phase: float = 0.0,
    height: float = 0.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=height (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], height])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                variation
                * (
                    z_low[2]
                    + (z_high - z_low)[2]
                    * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                )
                + height,
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:
    #     # Resample theta angles so that the velocity is closer to constant.
    #     lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    #     theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    #     positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])


def generate_ellipse_path_y(
    poses: np.ndarray,
    n_frames: int = 120,
    # const_speed: bool = True,
    variation: float = 0.0,
    phase: float = 0.0,
    height: float = 0.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at y=height (in middle of zero-mean capture pattern).
    offset = np.array([center[0], height, center[2]])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    y_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    y_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-z.
        # Optionally also interpolate in y to change camera height along path.
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                variation
                * (
                    y_low[1]
                    + (y_high - y_low)[1]
                    * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                )
                + height,
                low[2] + (high - low)[2] * (np.sin(theta) * 0.5 + 0.5),
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:
    #     # Resample theta angles so that the velocity is closer to constant.
    #     lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    #     theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    #     positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])


def generate_interpolated_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
):
    """Creates a smooth spline path between input keyframe camera poses.

    Spline is calculated with poses in format (position, lookat-point, up-point).

    Args:
      poses: (n, 3, 4) array of input pose keyframes.
      n_interp: returned path will have n_interp * (n - 1) total poses.
      spline_degree: polynomial degree of B-spline.
      smoothness: parameter for spline smoothing, 0 forces exact interpolation.
      rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
      Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(
        points, n_interp * (points.shape[0] - 1), k=spline_degree, s=smoothness
    )
    return points_to_poses(new_points)
