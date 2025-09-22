import sys
sys.path.append(".")
sys.path.append("..")
import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import trimesh
from PIL import Image
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection


from src.utils.flux_utils import parse_layout_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wireframe = trimesh.load_mesh("./assets/wireframe.ply")

raster_settings = RasterizationSettings(
    image_size=(512, 512),
    blur_radius=1e-5,
    perspective_correct=True,
    clip_barycentric_coords=True,
    z_clip_value=1e-5,
)


def main(root: Path, scene_key: str, camera_keys: List[int]):
    """
    Render an image using the P3FD model and input data.
    """

    # layout meta data file
    layout_meta_filepath = root / scene_key / "room_layout.json"
    with open(layout_meta_filepath, "r") as f:
        layout_data = json.load(f)
    # camera meta data file
    camera_meta_filepath = root / scene_key / "cameras.json"
    with open(camera_meta_filepath, "r") as f:
        camera_data = json.load(f)

    # parse layout data
    meshes, colors = parse_layout_data(layout_data, wireframe)

    vertices = [
        torch.tensor(mesh.vertices, dtype=torch.float32).to(device) for mesh in meshes
    ]
    faces = [torch.tensor(mesh.faces, dtype=torch.int32).to(device) for mesh in meshes]

    mesh = Meshes(verts=vertices, faces=faces)

    height, width = camera_data["height"], camera_data["width"]
    intrinsic = np.array(camera_data["intrinsic"], dtype=np.float32)[np.newaxis]
    intrinsic = torch.from_numpy(intrinsic)

    renderer = MeshRasterizer(raster_settings=raster_settings)
    renderer.to(device)

    os.makedirs(root / scene_key / "condition", exist_ok=True)

    for camera_key in camera_keys:
        save_condition_img_path = root / scene_key / "condition" / f"frame_{camera_key}.jpg"

        if os.path.exists(save_condition_img_path):
            try:
                image = Image.open(save_condition_img_path)
                image.verify()
                print(f"Skipping {save_condition_img_path} as it already exists and is valid.")
                continue
            except Exception as e:
                print(f"Failed to open {save_condition_img_path}: {e}")

        camera = camera_data["cameras"][str(camera_key)]
        pose = np.array(camera, dtype=np.float32)[np.newaxis, ...]
        pose = torch.from_numpy(pose)
        pose = pose.inverse()

        p3d_camera = cameras_from_opencv_projection(
            pose[..., :3, :3],
            pose[..., :3, 3],
            camera_matrix=intrinsic,
            image_size=torch.tensor([[height, width]]),
        ).to(device)

        fragments = renderer(mesh, cameras=p3d_camera)

        # depth map (BxHxW)
        depth = fragments.zbuf.squeeze(-1)
        # set invalid depth to infinity
        depth[depth == -1.0] = torch.inf
        min_depth, mesh_ids = torch.min(depth, dim=0)
        visible_mesh_ids = torch.unique(mesh_ids[torch.isfinite(min_depth)])

        # face id (BxHxW)
        pix_to_face = fragments.pix_to_face.squeeze(-1)
        # convert face idx to mesh idx
        for i in range(pix_to_face.shape[0]):
            pix_to_face[i, pix_to_face[i] != -1] = i

        # HxW
        mesh_id_map = torch.full_like(pix_to_face[0], -1)
        near_layout_id_map = torch.argmin(depth[:3], dim=0)

        min_depth, near_object_id_map = torch.min(depth[3:], dim=0)
        near_object_id_map = near_object_id_map + 3  # shift index
        near_object_id_map[~torch.isfinite(min_depth)] = -1

        for visible_id in visible_mesh_ids:
            if visible_id in [0, 1, 2]:
                # wall, floor, ceiling
                mesh_id_map[near_layout_id_map == visible_id] = visible_id
            else:
                mesh_id_map[near_object_id_map == visible_id] = visible_id

        # HxWx3
        condition = torch.zeros_like(mesh_id_map, dtype=torch.uint8)
        condition = condition.unsqueeze(-1).repeat(1, 1, 3)

        # color semantic with corresponding mesh color
        for mesh_id in visible_mesh_ids:
            condition[mesh_id_map == mesh_id] = torch.tensor(
                colors[mesh_id], dtype=torch.uint8, device=device
            )

        condition = Image.fromarray(condition.cpu().numpy())
        condition.save(save_condition_img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./example_data/scenes",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    dataset_root_dir = args.dataset_dir
    
    scene_folder_lst = [s for s in os.listdir(dataset_root_dir) if os.path.isdir(os.path.join(dataset_root_dir, s))]
    print(f"Found {len(scene_folder_lst)} under {dataset_root_dir}")

    total_num = len(scene_folder_lst)

    for index, scene_name in enumerate(scene_folder_lst):
        scene_img_folderpath = os.path.join(dataset_root_dir, scene_name, 'rgb')
        scene_imgs_lst = [img for img in os.listdir(scene_img_folderpath) if img.endswith(".jpg")]
        camera_keys = np.arange(len(scene_imgs_lst))
        print(
            f"Processing [{index:0{len(str(total_num))}d}/{total_num}]: {scene_name} with {len(scene_imgs_lst)} images"
        )
        try:
            main(root=Path(dataset_root_dir), 
                 scene_key=scene_name, 
                 camera_keys=camera_keys)
        except Exception as e:
            print(f"Failing {scene_name} due to {e}")
    