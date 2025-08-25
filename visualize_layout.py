import os
import sys
sys.path.append(".")
sys.path.append("..")

import trimesh
import numpy as np
import open3d as o3d

from src.utils.misc import read_json
from src.utils.layout_utils import (
    color2labels_dict,
    convert_oriented_box_to_trimesh_fmt, 
    parse_closed_room_from_meta
)

def visualize_spatialgen_data(scene_data_dir: str, vis_output_dir: str = None):
    """
    Test the layout data by loading it and printing the contents.
    """
    scene_layout_meta_file = os.path.join(scene_data_dir, "room_layout.json")
    scene_cams_meta_file = os.path.join(scene_data_dir, "cameras.json")
    if not os.path.exists(scene_layout_meta_file):
        print(f"Layout metadata file {scene_layout_meta_file} does not exist.")
        return
    if not os.path.exists(scene_cams_meta_file):
        print(f"Camera metadata file {scene_cams_meta_file} does not exist.")
        return

    if vis_output_dir is not None:
        os.makedirs(vis_output_dir, exist_ok=True)

    # load camera metadata
    camera_data_dict = read_json(scene_cams_meta_file)
    # load layout metadata
    layout_data_dict = read_json(scene_layout_meta_file)

    # visualize the camera poses
    img_width = camera_data_dict["width"]
    img_height = camera_data_dict["height"]
    cam_intrinsic = np.array(camera_data_dict["intrinsic"]).reshape(3, 3)

    camera = o3d.camera.PinholeCameraIntrinsic()
    camera.set_intrinsics(
        img_width,
        img_height,
        cam_intrinsic[0, 0],
        cam_intrinsic[1, 1],
        cam_intrinsic[0, 2],
        cam_intrinsic[1, 2],
    )
    cam_meta_data = camera_data_dict["cameras"]
    for cam_key, c2w_pose in cam_meta_data.items():
        c2w_pose = np.array(c2w_pose).reshape(4, 4).astype(np.float32)
        pose_w2c = np.linalg.inv(c2w_pose)
        # draw camera frame in the point cloud
        cam_lines = o3d.geometry.LineSet.create_camera_visualization(intrinsic=camera, extrinsic=pose_w2c, scale=0.05)
        cam_lines.paint_uniform_color([1, 0, 0])
        o3d.io.write_line_set(os.path.join(vis_output_dir, f"cam_{cam_key}.ply"), cam_lines)

    # visualize the layout
    obj_bbox_meshs = []

    bbox_infos = layout_data_dict["bboxes"]
    for idx, bbox_info in enumerate(bbox_infos):
        bbox_world_mesh = convert_oriented_box_to_trimesh_fmt(bbox_info, color_to_labels=color2labels_dict)
        obj_bbox_meshs.append(bbox_world_mesh)

    return_mesh_list = obj_bbox_meshs
    object_bbox_mesh = trimesh.util.concatenate(obj_bbox_meshs)
    wall_mesh, _, _ = parse_closed_room_from_meta(layout_data_dict, scene_data_dir)
    object_bbox_mesh = trimesh.util.concatenate([object_bbox_mesh, wall_mesh])
    return_mesh_list.append(wall_mesh)
    object_bbox_mesh.export(vis_output_dir + "/layout_bbox.ply")

    return return_mesh_list


if __name__ == "__main__":
    
    data_root_dir = "./example_data/scenes"
    scene_data_dirs = [os.path.join(data_root_dir, d) for d in os.listdir(data_root_dir) if d.startswith("scene_")]
    
    for scene_data_dir in scene_data_dirs:
        vis_output_dir = scene_data_dir
        # save layout_bbox.ply and camera poses in vis_output_dir
        visualize_spatialgen_data(scene_data_dir, vis_output_dir)