# Copyright (c) 2023-2024, Chuan Fang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys

sys.path.append(".")
sys.path.append("..")

import json
import os.path as osp
import collections
import math

import numpy as np
import torch
import trimesh


import cv2
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation
from icecream import ic


from src.utils.typing import *
from src.utils.filters import SpatialGradient
from src.utils.cam_ops import (
    get_ray_directions,
    get_rays,
    opengl_to_opencv,
    unproject_depth,
)
from src.utils.equilib import cube2equi, equi2pers
from src.utils.misc import read_json, readlines
from src.utils.image_utils import crop_image_to_target_ratio
from src.utils.colmap_utils import qvec2rotmat

from src.data.view_sampler import ViewSampler
from src.utils.layout_utils import (
    parse_obj_bbox_from_meta,
    DEFAULT_UNKNOWN_SEM2COLOR,
    compute_camera_inside_bbox,
    trimesh_to_p3dmesh,
    SDCMeshRenderer,
    parse_spatiallm_obj_bbox_from_meta,
)
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection


def get_koolai_room_ids(data_dir: str, split_filepath: str, invalid_split_filepath: str = None):
    # load valid room ids
    with open(split_filepath, "r") as f:
        uids = f.readlines()
        room_uids = [uid.strip() for uid in uids]

    # load invalid room ids
    if invalid_split_filepath is not None:
        with open(invalid_split_filepath, "r") as f:
            invalid_room_uids = f.readlines()
            invalid_room_uids = [uid.strip() for uid in invalid_room_uids]
    else:
        invalid_room_uids = []

    valid_room_uids = [osp.join(data_dir, uid) for uid in room_uids if uid not in invalid_room_uids]
    return valid_room_uids


def get_spatiallm_room_ids(data_dir: str):
    valid_scene_filepath = os.path.join(data_dir, "new_spatiallm_testing_scenes.txt")
    with open(valid_scene_filepath, "r") as f:
        room_uids = f.readlines()
        room_uids = [uid.strip() for uid in room_uids]

    valid_room_uids = [osp.join(data_dir, uid) for uid in room_uids]
    return valid_room_uids


TO_DIFFUSION_TENSOR = torchvision.transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)


class ExampleDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        split_filepath: str,
        image_height: int = 256,
        image_width: int = 256,
        T_in: int = 3,
        total_view: int = 8,
        validation: bool = False,
        use_normal: bool = False,
        use_semantic: bool = False,
        use_metric_depth: bool = False,
        use_scene_coord_map: bool = False,
        use_layout_prior: bool = False,
        use_layout_prior_from_p3d: bool = False,
        return_metric_data: bool = True,
    ):
        """This is a ExampleDataset class that supports SpatialGen, SpatialLM datasets.
        It is designed to be used for training and evaluation of the Room3D method
        SpatialGen Dataset Folder structure:
            data_dir:
            ├── scene_id:
                |-- perspective:
                    |-- room_id:
                        |-- rgb:      # extract perspective image from panorama
                        |-- depth:
                        |-- normal:
                        |-- semantic:
                        |-- layout_semantic:
                        |-- layout_depth:

                            ...
                        |-- cameras.json # cameras metadata
        Args:
            data_dir (str): _description_
            scannet_data_dir (str): _description_
            scannetpp_data_dir (str): _description_
            split_filepath (str): _description_
            sequential (_type_): _description_
            invalid_split_filepath (str, optional): _description_. Defaults to None.
            image_height (int, optional): _description_. Defaults to 256.
            image_width (int, optional): _description_. Defaults to 256.
            T_in (int, optional): number of input views. Defaults to 3.
            total_view (int, optional): _description_. Defaults to 8.
            validation (bool, optional): _description_. Defaults to False.
            sampler_type (str, optional): _description_. Defaults to "random".
            fixuse_plucker_ray (bool, optional): _description_. Defaults to True.
            use_metric_depth (bool, optional): whether to use metric depth. Defaults to False.
            return_metric_data (bool, optional): _description_. Defaults to False.
        """

        self.data_dir = data_dir
        # dataset name: "spatialgen", "spatiallm", "hypersim", "structured3d"
        self.load_dataset = dataset_name
        self.image_width = image_width
        self.image_height = image_height

        self.samples = []

        self.T_in = T_in if T_in is not None else 1
        self.num_sample_views = total_view

        self.use_normal = use_normal
        self.use_semantic = use_semantic
        self.return_metric_data = return_metric_data
        assert not (
            use_metric_depth and use_scene_coord_map
        ), "use_metric_depth and use_scene_coord_map cannot be used together"
        self.use_metric_depth = use_metric_depth
        self.use_scene_coord_map = use_scene_coord_map

        self.use_layout_prior = use_layout_prior
        self.use_layout_prior_from_p3d = use_layout_prior_from_p3d
        ic(
            self.use_metric_depth,
            self.use_layout_prior,
            self.use_scene_coord_map,
        )

        self.yaw_increase_angle = 60.0
        self.is_validation = validation
        if "spatialgen" == self.load_dataset:
            # get spatialgen rooms
            samples = get_koolai_room_ids(data_dir=data_dir, split_filepath=split_filepath)
            self.samples += samples
        elif "spatiallm" == self.load_dataset:
            self.samples = get_spatiallm_room_ids(data_dir=data_dir)

        if self.use_layout_prior_from_p3d:
            self.cuda_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            self.p3d_renderer = SDCMeshRenderer(
                cameras=None, image_size=(self.image_width, self.image_height), device=self.cuda_device
            )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_pil_image(file_path: str) -> Float[Tensor, "1 C H W"]:
        """Load image , Pillow version"""
        img = Image.open(file_path)
        img = np.array(img)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis].astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def resize_and_crop_image(
        img: Image,
        height=None,
        width=None,
        value_scale_factor=1.0,
        resampling_mode=Image.BILINEAR,
        disable_warning=False,
        target_aspect_ratio=None,
    ):
        """ " Reads an image file using PIL, then optionally resizes the image,
        with selective resampling, scales values, and returns the image as a
        tensor

        Args:
            filepath: path to the image.
            height, width: resolution to resize the image to. Both must not be
                None for scaling to take place.
            value_scale_factor: value to scale image values with, default is 1.0
            resampling_mode: resampling method when resizing using PIL. Default
                is PIL.Image.BILINEAR
            target_aspect_ratio: if not None, will crop the image to match this
            aspect ratio. Default is None

        Returns:
            img: tensor with (optionally) scaled and resized image data.

        """

        if target_aspect_ratio:
            img = crop_image_to_target_ratio(img, target_aspect_ratio)

        # resize if both width and height are not none.
        if height is not None and width is not None:
            img_width, img_height = img.size
            # do we really need to resize? If not, skip.
            if (img_width, img_height) != (width, height):
                # warn if it doesn't make sense.
                if (width > img_width or height > img_height) and not disable_warning:
                    print(
                        f"WARNING: target size ({width}, {height}) has a "
                        f"dimension larger than input size ({img_width}, "
                        f"{img_height})."
                    )
                img = img.resize((width, height), resample=resampling_mode)

        img = torchvision.transforms.functional.to_tensor(img).float() * value_scale_factor

        return img

    def depth_scale_shift_normalization(
        self,
        depths: Float[Tensor, "N 1 H W"],
        valid_mask: Float[Tensor, "N 1 H W"] = None,
        min: float = None,
        max: float = None,
        move_invalid_to_far_plane: bool = True,
    ) -> Tuple[Float[Tensor, "N 1 H W"], float, float]:
        """
        scale depth to [-1, 1], use the same scale for all depth maps
        params:
            depths: [N, 1, H, W]
        Returns:
            normalized_depths: [N, 1, H, W]
            max_value: max value of depth maps
        """

        if valid_mask is None:
            valid_mask = torch.ones_like(depths).bool()
        valid_mask = valid_mask & (depths > 0)

        # # normalize to [-1, 1]
        if min is not None or max is not None:
            min_value, max_value = torch.tensor([min], dtype=torch.float32), torch.tensor([max], dtype=torch.float32)
        else:
            min_value, max_value = depths.min(), depths.max()
        # print(f"depth_min: {min_value}, depth_max: {max_value}")
        normalized_depths = (depths - min_value) / (max_value - min_value) * 2.0 - 1.0
        normalized_depths = torch.clip(normalized_depths, -1.0, 1.0)

        normalized_depths[~valid_mask] = 1.0 if move_invalid_to_far_plane else -1.0
        return normalized_depths, min_value, max_value

    # resize image size, return the image in [-1, 1]
    def resize_image(
        self, image: Float[Tensor, "1 3 H W"], image_type: str = "rgb", depth_scale: float = 1000.0
    ) -> Float[Tensor, "1 3 H W"]:
        interpolation = cv2.INTER_NEAREST if image_type in ["depth", "normal"] else cv2.INTER_LINEAR
        if image.shape[-2:] != (self.image_height, self.image_width):
            if image_type == "depth":
                pers_img = torch.from_numpy(
                    cv2.resize(
                        image[0].permute(1, 2, 0).cpu().numpy(),
                        (self.image_height, self.image_width),
                        interpolation=interpolation,
                    )
                )[None, :, :]
            else:
                pers_img = torch.from_numpy(
                    cv2.resize(
                        image[0].permute(1, 2, 0).cpu().numpy(),
                        (self.image_height, self.image_width),
                        interpolation=interpolation,
                    )
                ).permute(2, 0, 1)
        else:
            pers_img = image[0]

        if image_type in ["rgb", "semantic"]:
            # normalize rgb to [-1, 1]
            pers_img = pers_img.float() / 255.0
            pers_img = pers_img.clip(0.0, 1.0) * 2.0 - 1.0
        elif image_type in ["depth"]:
            pers_img = pers_img.float() / depth_scale
        elif image_type in ["normal"]:
            # convert normal to [-1, 1]
            normal_img = pers_img.permute(1, 2, 0).cpu().numpy()
            normal = np.clip((normal_img + 0.5) / 255.0, 0.0, 1.0) * 2 - 1
            normal = normal / (np.linalg.norm(normal, axis=2)[:, :, np.newaxis] + 1e-6)
            # save normal in camera space, flip to make +z upward
            pers_img = torch.from_numpy(normal).permute(2, 0, 1).float()  # [3, 256, 256]
        return pers_img

    def _load_nonsquare_rgb(self, image: np.ndarray):
        image_pil = Image.fromarray(image)
        image = self.resize_and_crop_image(
            image_pil,
            height=self.image_height,
            width=self.image_width,
            resampling_mode=Image.BILINEAR,
            disable_warning=False,
            target_aspect_ratio=self.image_width / self.image_height,
        )
        image = image.unsqueeze(0)
        return image

    def _load_nonsquare_depth(self, depth_map: np.ndarray, depth_scale: float = 1.0):
        depth_pil = Image.fromarray(depth_map)
        depth = self.resize_and_crop_image(
            depth_pil,
            height=self.image_height,
            width=self.image_width,
            resampling_mode=Image.NEAREST,
            disable_warning=False,
            target_aspect_ratio=self.image_width / self.image_height,
        )

        depth = depth[None, :, :] / depth_scale
        return depth

    def calc_scale_mat(
        self, raw_poses: Float[Tensor, "N 4 4"], depth_range: float = 1.5, offset_center: bool = True
    ) -> Tuple[Float[Tensor, "4 4"], float]:
        """
        raw_poses: [N, 4, 4], camera to world poses
        depth_range: maximum depth range of each camera
        """
        c2w_poses = raw_poses
        min_vertices = c2w_poses[:, :3, 3].min(dim=0).values
        max_vertices = c2w_poses[:, :3, 3].max(dim=0).values

        if offset_center:
            center = (min_vertices + max_vertices) / 2.0
        else:
            center = torch.zeros(3, dtype=torch.float32)

        # convert the scene to [-1, 1] unit cube
        # scale = 2. / (torch.max(max_vertices - min_vertices) + 2 * depth_range)
        # TODO: use above scale, but for now, use the depth scale
        positions_scale = torch.max(max_vertices - min_vertices)
        if depth_range is not None:
            # scale = 2.0 / positions_scale if positions_scale > (2 * depth_range) else 2.0 / (2 * depth_range)
            scale = 2.0 / (2 * depth_range)
        else:
            scale = 2.0 / positions_scale

        # we should normalized to unit cube
        scale_mat = torch.eye(4, dtype=torch.float32)
        scale_mat[:3, 3] = -center
        scale_mat[:3] *= scale

        return scale_mat, scale

    def prepare_ray_directions(
        self, num_sample_views: int = 16, focal_length: float = 128.0
    ) -> Float[Tensor, "B H W 3"]:
        # get ray directions by intrinsics
        directions = get_ray_directions(
            H=self.image_height,
            W=self.image_width,
            focal=focal_length,
        )
        directions: Float[Tensor, "B H W 3"] = directions[None, :, :, :].repeat(num_sample_views, 1, 1, 1)
        return directions

    def get_spatiallm_item(self, sample_room_path: str | None = None) -> Dict:
        """
        Get item from [SpatialLM Testset](https://huggingface.co/datasets/manycore-research/SpatialLM-Testset/tree/main).
        Args:
            sample_room_path: str|None=None, room path

        Returns:
            Dict: spatialLM item
        """
        assert self.use_scene_coord_map, "get_spatiallm_item only supports use_scene_coord_map=True"

        room_uid = sample_room_path.split("/")[-1]
        rgb_dir = sample_room_path
        scene_layout_meta_file = os.path.join(sample_room_path, "room_layout.json")
        scene_cams_meta_file = os.path.join(sample_room_path, "cameras.json")
        scene_layout_mesh_file = os.path.join(sample_room_path, "layout_bbox.ply")
        valid_frames_file = os.path.join(sample_room_path, "valid_frames.txt")
        assert os.path.exists(scene_layout_meta_file), f"Layout metadata file {scene_layout_meta_file} does not exist."
        assert os.path.exists(scene_cams_meta_file), f"Camera metadata file {scene_cams_meta_file} does not exist."
        assert os.path.exists(scene_layout_mesh_file), f"Layout mesh file {scene_layout_mesh_file} does not exist."
        assert os.path.exists(valid_frames_file), f"Valid frames file {valid_frames_file} does not exist."

        # load camera metadata
        camera_data_dict = read_json(scene_cams_meta_file)
        # load layout metadata
        layout_data_dict = read_json(scene_layout_meta_file)

        # visualize the camera poses
        ori_width = camera_data_dict["width"]
        ori_height = camera_data_dict["height"]
        if ori_width != 1280 or ori_height != 720:
            raise NotImplementedError(
                f"Room {room_uid} has resolution {ori_width}x{ori_height}, Only support 1280x720 resolution"
            )
        intrinsic_mat = torch.tensor(camera_data_dict["intrinsic"], dtype=torch.float32).reshape(3, 3)
        # update intrinsic matrix for the target image size, since crop only change the principal point,
        # we need to update the focal length according to the size ratio of the short side
        focal_scale = self.image_height / float(ori_height)
        intrinsic_mat[0, 0] *= focal_scale
        intrinsic_mat[1, 1] *= focal_scale
        intrinsic_mat[0, 2] = self.image_width / 2.0
        intrinsic_mat[1, 2] = self.image_height / 2.0

        # use the sequential sampling
        num_sample_views = self.num_sample_views
        T_in = self.T_in
        T_out = num_sample_views - T_in
        valid_frame_ids = readlines(valid_frames_file)[:360]
        step = len(valid_frame_ids) // num_sample_views
        sampled_camera_keys = valid_frame_ids[::step][:num_sample_views]
        cam_meta_data = camera_data_dict["cameras"]

        rgbs = []
        poses = []
        for cam_key in sampled_camera_keys:
            # load meta data
            c2w_pose = torch.from_numpy(np.array(cam_meta_data[cam_key]).reshape(4, 4).astype(np.float32)).float()
            poses.append(c2w_pose)

            # load rgb
            rgb_filepath = os.path.join(rgb_dir, cam_key + ".jpg")
            rgb = np.array(Image.open(rgb_filepath).convert("RGB")).astype(np.uint8)
            rgb: Float[Tensor, "1 3 H W"] = self._load_nonsquare_rgb(rgb)  # (1, 3, H, W)

            # convert rgb to [-1, 1]
            normalized_rgb = rgb * 2.0 - 1.0
            rgbs.append(normalized_rgb)

        poses: Float[Tensor, "N 4 4"] = torch.stack(poses, dim=0)
        rgbs: Float[Tensor, "N 3 H W"] = torch.cat(rgbs, dim=0)

        metric_poses = poses.clone()

        obj_layout_dict, obj_mesh_list = parse_spatiallm_obj_bbox_from_meta(
            layout_data_dict, sample_room_path, return_mesh=True
        )
        render_layout_semantics = []
        render_layout_depths = []
        for i in range(self.num_sample_views):
            cam_in_obj_mask = compute_camera_inside_bbox(metric_poses[i], obj_layout_dict)[0]
            pose_device = self.cuda_device
            #  extract wall mesh
            wall_mesh = obj_mesh_list[-1]
            filtered_mesh_list = [obj_mesh_list[_id] for _id, _x in enumerate(cam_in_obj_mask) if not _x]

            # filtered_mesh_list.append(wall_mesh)
            concat_mesh = trimesh.util.concatenate(filtered_mesh_list)
            obj_mesh, face_color = trimesh_to_p3dmesh(concat_mesh)
            obj_mesh = obj_mesh.to(pose_device)
            face_color = face_color.to(pose_device)

            wall_mesh, wall_face_color = trimesh_to_p3dmesh(wall_mesh)
            wall_mesh = wall_mesh.to(pose_device)
            wall_face_color = wall_face_color.to(pose_device)
            w2c = metric_poses[i : i + 1].inverse()

            p3d_camera = cameras_from_opencv_projection(
                w2c[..., :3, :3],
                w2c[..., :3, 3],
                camera_matrix=intrinsic_mat[None,],
                image_size=torch.tensor([[self.image_height, self.image_width]]),
            ).to(pose_device)

            layout_rener_imgs = self.p3d_renderer(obj_mesh, cameras=p3d_camera, faces_color=face_color)
            wall_render_imgs = self.p3d_renderer(wall_mesh, cameras=p3d_camera, faces_color=wall_face_color)
            wall_render_sem_img = (wall_render_imgs["render_segment"].cpu()[0].numpy() * 255).astype(np.uint8)
            wall_render_dep_img = wall_render_imgs["depth"].cpu()[0].numpy().astype(np.float32)
            render_sem_img = (layout_rener_imgs["render_segment"].cpu()[0].numpy() * 255).astype(np.uint8)

            # fill the background area with wall_rendering
            bg_area = np.all(render_sem_img == 0, axis=-1)
            render_sem_img[bg_area] = wall_render_sem_img[bg_area]

            # fill the background area with wall_rendering
            render_depth_img = layout_rener_imgs["depth"].cpu()[0].numpy()
            render_depth_img[render_depth_img == -1.0] = wall_render_dep_img[render_depth_img == -1.0]
            # filter out the occluded pixels (when the farther object is occluded by the closer walls)
            occluded_area = np.logical_and(
                render_depth_img > (wall_render_dep_img + 0.5), wall_render_dep_img > 0
            ).squeeze()
            render_depth_img[occluded_area] = wall_render_dep_img[occluded_area]
            render_sem_img[occluded_area] = wall_render_sem_img[occluded_area]

            render_layout_semantics.append(TO_DIFFUSION_TENSOR(render_sem_img))
            render_layout_depths.append(torch.from_numpy(render_depth_img).permute(2, 0, 1).float())

            # # save semantic images
            # if True:
            #     merged_image = Image.new('RGB', (self.image_width, self.image_height * 2))
            #     merged_image.paste(Image.fromarray(render_sem_img.astype(np.uint8)), (0, 0))
            #     merged_image.paste(Image.fromarray(colorize_depth(render_depth_img[:,:,0]).astype(np.uint8)).convert('RGB'), (0, 1 * self.image_height))
            #     merged_image.save(f"{room_uid.replace('/', '_')}_{i}_pyt3d.png")

        render_semantics: Float[Tensor, "N 3 H W"] = torch.stack(render_layout_semantics, dim=0)
        render_depths: Float[Tensor, "N 1 H W"] = torch.stack(render_layout_depths, dim=0)

        # always choose the first view as the reference view
        relative_poses = torch.inverse(poses[0:1]).repeat(poses.shape[0], 1, 1) @ poses
        # update poses to relative poses
        poses = relative_poses

        layout_scene_coord_maps = unproject_depth(depth_map=render_depths, C2W=poses, K=intrinsic_mat)
        # normalize scene coord maps to [-1, 1]
        coord_max, coord_min = layout_scene_coord_maps.max(), layout_scene_coord_maps.min()
        # max_depth is scene range
        scene_range = torch.abs(coord_max - coord_min)
        layout_scene_coord_maps: Float[Tensor, "N 3 H W"] = (
            layout_scene_coord_maps - coord_min
        ) / scene_range * 2.0 - 1.0
        layout_scene_coord_maps = layout_scene_coord_maps.clamp(-1.0, 1.0)

        # normalize poses to unit cube[-1,1] w.r.t current sample views
        curr_scale_mat, curr_scene_scale = self.calc_scale_mat(poses, depth_range=scene_range, offset_center=False)
        for pose_idx in range(poses.shape[0]):
            # scale pose_c2w
            subview_pose = curr_scale_mat @ poses[pose_idx]
            R_c2w = (subview_pose[:3, :3]).numpy()
            q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
            q_c2w = trimesh.transformations.unit_vector(q_c2w)
            R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
            subview_pose[:3, :3] = torch.from_numpy(R_c2w)
            poses[pose_idx] = subview_pose

        fl_x, fl_y, cx, cy = (
            intrinsic_mat[0, 0],
            intrinsic_mat[1, 1],
            intrinsic_mat[0, 2],
            intrinsic_mat[1, 2],
        )
        directions = get_ray_directions(
            H=self.image_height,
            W=self.image_width,
            focal=[fl_x, fl_y],
            principal=[cx, cy],
        )
        canonical_ray_directions: Float[Tensor, "B H W 3"] = directions[None, :, :, :].repeat(num_sample_views, 1, 1, 1)
        rays_o, rays_d = get_rays(canonical_ray_directions, relative_poses, keepdim=True)
        rays_od = torch.cat([rays_o, rays_d], dim=-1)
        rays_od = rays_od.permute(0, 3, 1, 2)  # B, 6, H, W
        # plucker rays
        rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  # B, H, W, 3
        plucker_rays = torch.cat([rays_dxo, rays_d], dim=-1)  # B, H, W, 6
        plucker_rays = plucker_rays.permute(0, 3, 1, 2)  # B, 6, H, W

        # source views
        input_images: Float[Tensor, "N 3 H W"] = rgbs[:T_in]

        # atarget views
        target_images: Float[Tensor, "N 3 H W"] = rgbs[T_in:num_sample_views]

        depth_class = torch.tensor([1, 0, 0, 0]).float()
        depth_task_embeddings = torch.stack([depth_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        color_class = torch.tensor([0, 1, 0, 0]).float()
        color_task_embeddings = torch.stack([color_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_normal:
            normal_class = torch.tensor([0, 0, 1, 0]).float()
            normal_task_embeddings = torch.stack([normal_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_semantic:
            semantic_class = torch.tensor([0, 0, 0, 1]).float()
            semantic_task_embeddings = torch.stack([semantic_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_layout_prior or self.use_layout_prior_from_p3d:
            layout_sem_class = torch.tensor([0, 0, 0, 10]).float()
            layout_sem_task_embeddings = torch.stack([layout_sem_class] * self.num_sample_views, dim=0)
            layout_depth_class = torch.tensor([0, 0, 0, 101]).float()
            layout_depth_task_embeddings = torch.stack([layout_depth_class] * self.num_sample_views, dim=0)

        cond_Ts: Float[Tensor, "N 4 4"] = poses[:T_in]
        target_Ts: Float[Tensor, "N 4 4"] = poses[T_in : self.num_sample_views]

        input_plucker_rays: Float[Tensor, "N 6 H W"] = plucker_rays[:T_in]
        target_plucker_rays: Float[Tensor, "N 6 H W"] = plucker_rays[T_in : self.num_sample_views]

        input_rays: Float[Tensor, "N 6 H W"] = rays_od[:T_in]
        target_rays: Float[Tensor, "N 6 H W"] = rays_od[T_in : self.num_sample_views]

        # shuffled_indices = torch.randperm(self.num_sample_views)
        shuffled_indices = torch.arange(self.num_sample_views)
        input_indices = shuffled_indices[:T_in]
        target_indices = shuffled_indices[T_in:]

        data = {}
        data["dataset"] = "spatiallm"
        data["room_uid"] = room_uid
        data["image_input"] = input_images
        data["image_target"] = target_images

        if self.use_normal:
            data["normal_task_embeddings"] = normal_task_embeddings

        if self.use_semantic:
            data["semantic_task_embeddings"] = semantic_task_embeddings

        if self.use_layout_prior or self.use_layout_prior_from_p3d:
            data["semantic_layout_input"] = render_semantics[:T_in]
            data["semantic_layout_target"] = render_semantics[T_in : self.num_sample_views]
            data["depth_layout_input"] = layout_scene_coord_maps[:T_in]
            data["depth_layout_target"] = layout_scene_coord_maps[T_in : self.num_sample_views]
            data["layout_sem_task_embeddings"] = layout_sem_task_embeddings
            data["layout_depth_task_embeddings"] = layout_depth_task_embeddings

        data["pose_out"] = target_Ts
        data["pose_in"] = cond_Ts
        data["plucker_rays_input"] = input_plucker_rays
        data["plucker_rays_target"] = target_plucker_rays
        data["rays_input"] = input_rays
        data["rays_target"] = target_rays
        data["color_task_embeddings"] = color_task_embeddings
        data["depth_task_embeddings"] = depth_task_embeddings
        data["depth_min"] = coord_min
        data["depth_max"] = coord_max
        data["scene_scale"] = curr_scene_scale
        data["input_indices"] = input_indices
        data["output_indices"] = target_indices
        if self.return_metric_data:
            data["pose_metric_input"] = metric_poses[:T_in]
            data["pose_metric_target"] = metric_poses[T_in : self.num_sample_views]
            if self.use_layout_prior or self.use_layout_prior_from_p3d:
                data["layout_depth_metric_input"] = render_depths[:T_in]
                data["layout_depth_metric_target"] = render_depths[T_in : self.num_sample_views]
        data["intrinsic"] = intrinsic_mat

        return data

    def get_spatialgen_item(
        self,
        sample_room_path: str | None = None,
        depth_scale: float = 1000.0,
    ) -> Dict:

        room_uid = sample_room_path.split("/")[-1]
        rgb_dir = os.path.join(sample_room_path, "rgb")
        depth_dir = os.path.join(sample_room_path, "depth")
        normal_dir = os.path.join(sample_room_path, "normal")
        semantic_dir = os.path.join(sample_room_path, "semantic")
        layout_semantic_dir = os.path.join(sample_room_path, "layout_semantic")
        layout_depth_dir = os.path.join(sample_room_path, "layout_depth")
        cam_meta_filepath = os.path.join(sample_room_path, "cameras.json")

        # load camera poses
        cameras_meta = read_json(cam_meta_filepath)
        total_valid_cam_keys = list(cameras_meta["cameras"].keys())

        # random the num of input views
        num_sample_views = self.num_sample_views
        T_in = self.T_in
        T_out = self.num_sample_views - T_in

        # use the sequential sampling
        start_index = np.random.randint(0, len(total_valid_cam_keys)) if not self.is_validation else 0
        valid_frame_ids = np.roll(total_valid_cam_keys, -start_index)
        if self.is_validation:
            valid_frame_ids = total_valid_cam_keys
        sample_keys = valid_frame_ids[:num_sample_views]

        # c2w poses, rgbs, background colors
        poses, rgbs = [], []
        depths, normals, semantics = [], [], []
        render_semantics, render_depths = [], []

        intrinsic_mat = torch.tensor(cameras_meta["intrinsic"], dtype=torch.float32).reshape(3, 3)
        ori_img_height, ori_img_width = cameras_meta["height"], cameras_meta["width"]
        # resize intrinsic matrix w.r.t the hight
        if self.image_height != ori_img_height or self.image_width != ori_img_width:
            intrinsic_mat[0, 0] *= self.image_height / ori_img_height
            intrinsic_mat[1, 1] *= self.image_height / ori_img_height
            intrinsic_mat[0, 2] = self.image_width / 2.0
            intrinsic_mat[1, 2] = self.image_height / 2.0
        focal_len = float(intrinsic_mat[0, 0])

        for idx, view_key in enumerate(sample_keys):

            # load c2w pose
            c2w_pose = torch.from_numpy(np.array(cameras_meta["cameras"][view_key]).reshape(4, 4)).float()
            # load perspective images
            frame_name = f"frame_{view_key}"
            rgb_image_filepath = os.path.join(rgb_dir, f"{frame_name}.jpg")
            depth_image_filepath = os.path.join(depth_dir, f"{frame_name}.png")
            normal_image_filepath = os.path.join(normal_dir, f"{frame_name}.jpg")
            semantic_image_filepath = os.path.join(semantic_dir, f"{frame_name}.jpg")

            rgb_img: Float[Tensor, "1 3 H W"] = self._load_pil_image(rgb_image_filepath)
            depth_img: Float[Tensor, "1 1 H W"] = self._load_pil_image(depth_image_filepath)
            if self.use_normal:
                normal_img: Float[Tensor, "1 3 H W"] = self._load_pil_image(normal_image_filepath)
            if self.use_semantic:
                semantic_img: Float[Tensor, "1 3 H W"] = self._load_pil_image(semantic_image_filepath)
            if self.use_layout_prior or self.use_layout_prior_from_p3d:
                layout_semantic_image_filepath = os.path.join(layout_semantic_dir, f"{frame_name}.jpg")
                layout_depth_image_filepath = os.path.join(layout_depth_dir, f"{frame_name}.png")
                layout_semantic_img: Float[Tensor, "1 3 H W"] = self._load_pil_image(layout_semantic_image_filepath)
                layout_depth_img: Float[Tensor, "1 1 H W"] = self._load_pil_image(layout_depth_image_filepath)

            # scale rgb to [-1, 1]
            subview_rgbs = [self.resize_image(rgb_img, image_type="rgb")]
            subview_depths = [self.resize_image(depth_img, image_type="depth", depth_scale=depth_scale)]
            if self.use_normal:
                subview_normals = [self.resize_image(normal_img, image_type="normal")]
            if self.use_semantic:
                subview_semantics = [self.resize_image(semantic_img, image_type="semantic")]
            if self.use_layout_prior or self.use_layout_prior_from_p3d:
                subview_layout_semantics = [self.resize_image(layout_semantic_img, image_type="semantic")]
                subview_layout_depths = [
                    self.resize_image(layout_depth_img, image_type="depth", depth_scale=depth_scale)
                ]
                # complete depth_img with layout_depth
                subview_depths[0][subview_depths[0] < 1e-3] = subview_layout_depths[0][subview_depths[0] < 1e-3]
                # cut off the depth
                subview_depths[0][subview_depths[0] > 12.5] = 12.5
                subview_layout_depths[0][subview_layout_depths[0] > 12.5] = 12.5
                render_semantics += subview_layout_semantics
                render_depths += subview_layout_depths

            subview_poses = [c2w_pose]

            poses += subview_poses
            rgbs += subview_rgbs
            depths += subview_depths
            if self.use_normal:
                normals += subview_normals
            if self.use_semantic:
                semantics += subview_semantics

        poses: Float[Tensor, "N 4 4"] = torch.stack(poses, dim=0)
        rgbs: Float[Tensor, "N 3 H W"] = torch.stack(rgbs, dim=0)
        depths: Float[Tensor, "N 1 H W"] = torch.stack(depths, dim=0)
        if self.use_normal:
            normals: Float[Tensor, "N 3 H W"] = torch.stack(normals, dim=0)
        if self.use_semantic:
            semantics: Float[Tensor, "N 3 H W"] = torch.stack(semantics, dim=0)
        if self.use_layout_prior or self.use_layout_prior_from_p3d:
            render_semantics = torch.stack(render_semantics, dim=0)
            render_depths = torch.stack(render_depths, dim=0)

        metric_poses = poses.clone()
        metric_depths = depths.clone()

        # always choose the first view as the reference view
        relative_poses = torch.inverse(poses[0:1]).repeat(poses.shape[0], 1, 1) @ poses
        # update poses to relative poses
        poses = relative_poses

        if self.use_scene_coord_map:
            # project depth to point cloud
            scene_coord_maps = unproject_depth(depth_map=depths, C2W=poses, K=intrinsic_mat)
            # normalize scene coord maps to [-1, 1]
            coord_max, coord_min = scene_coord_maps.max(), scene_coord_maps.min()
            # max_depth is scene range
            scene_range = torch.abs(coord_max - coord_min)
            scene_coord_maps: Float[Tensor, "N 3 H W"] = (scene_coord_maps - coord_min) / scene_range * 2.0 - 1.0
            scene_coord_maps = scene_coord_maps.clamp(-1.0, 1.0)

            if self.use_layout_prior or self.use_layout_prior_from_p3d:
                layout_scene_coord_maps = unproject_depth(depth_map=render_depths, C2W=poses, K=intrinsic_mat)
                # normalize scene coord maps to [-1, 1]
                layout_scene_coord_maps: Float[Tensor, "N 3 H W"] = (
                    layout_scene_coord_maps - coord_min
                ) / scene_range * 2.0 - 1.0
                layout_scene_coord_maps = layout_scene_coord_maps.clamp(-1.0, 1.0)
        else:
            # normalize depth to [-1, 1]
            normalized_depths, min_depth, max_depth = self.depth_scale_shift_normalization(depths)
            if self.use_layout_prior or self.use_layout_prior_from_p3d:
                assert self.use_metric_depth, "use_metric_depth should be True when use layout_depths!!!"

        if not self.use_metric_depth:
            # normalize poses to unit cube[-1,1] w.r.t current sample views
            curr_scale_mat, curr_scene_scale = self.calc_scale_mat(poses, depth_range=scene_range, offset_center=False)
            for pose_idx in range(poses.shape[0]):
                # scale pose_c2w
                subview_pose = curr_scale_mat @ poses[pose_idx]
                R_c2w = (subview_pose[:3, :3]).numpy()
                q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
                q_c2w = trimesh.transformations.unit_vector(q_c2w)
                R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
                subview_pose[:3, :3] = torch.from_numpy(R_c2w)
                poses[pose_idx] = subview_pose
        else:
            assert not self.use_scene_coord_map
            curr_scene_scale = 1

        canonical_ray_directions = self.prepare_ray_directions(
            num_sample_views=num_sample_views, focal_length=focal_len
        )
        # if not self.use_plucker_ray:
        rays_o, rays_d = get_rays(canonical_ray_directions, poses, keepdim=True)
        rays_od = torch.cat([rays_o, rays_d], dim=-1)
        rays_od = rays_od.permute(0, 3, 1, 2)  # B, 6, H, W

        # plucker rays
        rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  # B, H, W, 3
        plucker_rays = torch.cat([rays_dxo, rays_d], dim=-1)  # B, H, W, 6
        plucker_rays = plucker_rays.permute(0, 3, 1, 2)  # B, 6, H, W

        # source views
        input_images: Float[Tensor, "N 3 H W"] = rgbs[:T_in]

        # target views
        target_images: Float[Tensor, "N 3 H W"] = rgbs[T_in : self.num_sample_views]

        if not self.use_scene_coord_map:
            input_depths: Float[Tensor, "N 3 H W"] = normalized_depths[:T_in].repeat(1, 3, 1, 1)
            target_depths: Float[Tensor, "N 3 H W"] = normalized_depths[T_in : self.num_sample_views].repeat(1, 3, 1, 1)
        else:
            input_depths: Float[Tensor, "N 3 H W"] = scene_coord_maps[:T_in]
            target_depths: Float[Tensor, "N 3 H W"] = scene_coord_maps[T_in : self.num_sample_views]

        if self.use_normal:
            input_normals: Float[Tensor, "N 3 H W"] = normals[:T_in]
            target_normals: Float[Tensor, "N 3 H W"] = normals[T_in : self.num_sample_views]
        if self.use_semantic:
            input_semantics: Float[Tensor, "N 3 H W"] = semantics[:T_in]
            target_semantics: Float[Tensor, "N 3 H W"] = semantics[T_in : self.num_sample_views]

        depth_class = torch.tensor([1, 0, 0, 0]).float()
        depth_task_embeddings = torch.stack([depth_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        color_class = torch.tensor([0, 1, 0, 0]).float()
        color_task_embeddings = torch.stack([color_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_normal:
            normal_class = torch.tensor([0, 0, 1, 0]).float()
            normal_task_embeddings = torch.stack([normal_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_semantic:
            semantic_class = torch.tensor([0, 0, 0, 1]).float()
            semantic_task_embeddings = torch.stack([semantic_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_layout_prior or self.use_layout_prior_from_p3d:
            layout_sem_class = torch.tensor([0, 0, 0, 10]).float()
            layout_sem_task_embeddings = torch.stack([layout_sem_class] * self.num_sample_views, dim=0)
            layout_depth_class = torch.tensor([0, 0, 0, 101]).float()
            layout_depth_task_embeddings = torch.stack([layout_depth_class] * self.num_sample_views, dim=0)

        cond_Ts: Float[Tensor, "N 4 4"] = poses[:T_in]
        target_Ts: Float[Tensor, "N 4 4"] = poses[T_in : self.num_sample_views]

        input_plucker_rays: Float[Tensor, "N 6 H W"] = plucker_rays[:T_in]
        target_plucker_rays: Float[Tensor, "N 6 H W"] = plucker_rays[T_in : self.num_sample_views]

        input_rays: Float[Tensor, "N 6 H W"] = rays_od[:T_in]
        target_rays: Float[Tensor, "N 6 H W"] = rays_od[T_in : self.num_sample_views]

        shuffled_indices = torch.arange(self.num_sample_views)
        input_indices = shuffled_indices[:T_in]
        target_indices = shuffled_indices[T_in:]

        data = {}
        data["dataset"] = "spatialgen"
        data["room_uid"] = room_uid
        data["image_input"] = input_images
        data["image_target"] = target_images
        data["depth_input"] = input_depths
        data["depth_target"] = target_depths
        if self.use_normal:
            data["normal_input"] = input_normals
            data["normal_target"] = target_normals
            data["normal_task_embeddings"] = normal_task_embeddings

        if self.use_semantic:
            data["semantic_input"] = input_semantics
            data["semantic_target"] = target_semantics
            data["semantic_task_embeddings"] = semantic_task_embeddings

        if self.use_layout_prior or self.use_layout_prior_from_p3d:
            data["semantic_layout_input"] = render_semantics[:T_in]
            data["semantic_layout_target"] = render_semantics[T_in : self.num_sample_views]
            if not self.use_scene_coord_map:
                normalized_render_depths = normalized_render_depths.repeat(1, 3, 1, 1)
                data["depth_layout_input"] = normalized_render_depths[:T_in]
                data["depth_layout_target"] = normalized_render_depths[T_in : self.num_sample_views]
            else:
                data["depth_layout_input"] = layout_scene_coord_maps[:T_in]
                data["depth_layout_target"] = layout_scene_coord_maps[T_in : self.num_sample_views]
            data["layout_sem_task_embeddings"] = layout_sem_task_embeddings
            data["layout_depth_task_embeddings"] = layout_depth_task_embeddings

        data["pose_out"] = target_Ts
        data["pose_in"] = cond_Ts
        data["plucker_rays_input"] = input_plucker_rays
        data["plucker_rays_target"] = target_plucker_rays
        data["rays_input"] = input_rays
        data["rays_target"] = target_rays
        data["color_task_embeddings"] = color_task_embeddings
        data["depth_task_embeddings"] = depth_task_embeddings
        if not self.use_scene_coord_map:
            data["depth_min"] = min_depth
            data["depth_max"] = max_depth
        else:
            data["depth_min"] = coord_min
            data["depth_max"] = coord_max
        data["scene_scale"] = curr_scene_scale
        data["input_indices"] = input_indices
        data["output_indices"] = target_indices
        if self.return_metric_data:
            data["pose_metric_input"] = metric_poses[:T_in]
            data["pose_metric_target"] = metric_poses[T_in : self.num_sample_views]
            data["depth_metric_input"] = metric_depths[:T_in]
            data["depth_metric_target"] = metric_depths[T_in : self.num_sample_views]
            data["layout_depth_metric_input"] = render_depths[:T_in]
            data["layout_depth_metric_target"] = render_depths[T_in : self.num_sample_views]
        data["intrinsic"] = intrinsic_mat
        # if self.koolai_prompt_dir is not None:
        #     scene_uid = room_uid.split("/")[1]
        #     room_id = sample_room_path.split("/")[-1]
        #     uid = f"{scene_uid}_{room_id}"
        #     prompt_npz_data = np.load(f"{self.koolai_prompt_dir}/{uid}.npz")
        #     prompt_embed = torch.from_numpy(prompt_npz_data["caption_feature"])[0]
        #     prompt_attention_mask = torch.from_numpy(prompt_npz_data["attention_mask"])[0]
        #     data["prompt_embed"] = prompt_embed
        #     data["prompt_attention_mask"] = prompt_attention_mask

        return data

    def inner_get_item(self, index: int) -> Dict:
        sample_room_path = self.samples[index]
        data = {}
        if self.load_dataset == "spatialgen":
            data = self.get_spatialgen_item(sample_room_path, depth_scale=1000.0)
        elif self.load_dataset == "spatiallm":
            # SpatialLMM-Testset dataset
            data = self.get_spatiallm_item(sample_room_path)
        else:
            raise ValueError(f"Error: unknown dataset type {self.load_dataset}")

        return data

    def __getitem__(self, index):

        sample_room_path = self.samples[index]
        try:
            return self.inner_get_item(index)
        except Exception as e:
            print(f"[DEBUG-DATASET] Error when loading {sample_room_path}")
            print(f"[DEBUG-DATASET] Error: {e}")
            raise e

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.image_height, "width": self.image_width})
        return batch
