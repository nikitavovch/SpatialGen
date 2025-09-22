import sys
sys.path.append('.')
sys.path.append('..')

from typing import Any, Dict

import numpy as np
import shapely
import trimesh

from src.utils.colormaps import CEILING_COLOR, COLOR2LABELS, FLOOR_COLOR, WALL_COLOR

colormaps = {v: k for k, v in COLOR2LABELS.items()}


def construct_floorplan_polygon(
    floor_corners: np.ndarray, epsilon=0.005, wall_thickness=0.24
):
    floorplan = shapely.geometry.Polygon(floor_corners)
    floorplan = shapely.simplify(floorplan, tolerance=epsilon, preserve_topology=True)

    floorplan = floorplan.buffer(
        wall_thickness / 2, cap_style="flat", join_style="mitre"
    )
    floorplan = shapely.simplify(floorplan, tolerance=epsilon, preserve_topology=True)

    if not floorplan.exterior.is_ccw:
        # print("Reversing polygon orientation to ensure CCW order.")
        floorplan = floorplan.reverse()

    inner_walls = floorplan.buffer(
        -wall_thickness / 2, cap_style="flat", join_style="mitre"
    )

    if not inner_walls.exterior.is_ccw:
        # print("Reversing polygon orientation to ensure CCW order.")
        inner_walls = inner_walls.reverse()

    inner_coords = np.array(inner_walls.exterior.coords)[:-1]

    return inner_coords


def convert_wall_to_trimesh(
    vertices: np.ndarray,
) -> trimesh.Trimesh:
    """
    create a quad polygen from vertices and normal
    params:
        quad_vertices: 4x3 np array, 4 vertices of the quad
        normal: 3x1 np array, normal of the quad
        camera_center: 3x1 np array, camera center
        camera_rotation: 3x3 np array, camera rotation
    """
    normal = np.cross(
        vertices[1] - vertices[0],
        vertices[2] - vertices[1],
    )
    normal = normal / np.linalg.norm(normal)

    faces = np.array([[0, 1, 2], [2, 3, 0]])
    normals = np.zeros_like(faces).astype(np.float32)
    normals[:, :] = normal

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        face_normals=normals,
        process=False,
    )
    return mesh


def convert_room_to_trimesh(room_metadata: Dict[str, Any]):
    assert "floor" in room_metadata and "ceil" in room_metadata

    layout_corners = (
        np.array(
            [
                [corner["start"]["x"], corner["start"]["y"]]
                for corner in room_metadata["floor"]
            ]
        )
        / 1000.0
    )
    wall_height = (
        np.mean(np.array([corner["start"]["z"] for corner in room_metadata["ceil"]]))
        / 1000.0
    )

    layout_corners = construct_floorplan_polygon(layout_corners)

    wall_meshes = []
    for i in range(len(layout_corners)):
        corner_i = layout_corners[i]
        corner_j = layout_corners[(i + 1) % len(layout_corners)]

        wall = np.stack([corner_i, corner_j, corner_j, corner_i])
        wall = np.concatenate([wall, np.zeros_like(wall[:, :1])], axis=1)
        wall[2:, -1] = wall_height
        wall_mesh = convert_wall_to_trimesh(wall)

        wall_meshes.append(wall_mesh)
    wall_mesh = trimesh.util.concatenate(wall_meshes)

    # construct floor and ceiling polygons
    layout_polygon = shapely.Polygon(layout_corners)
    layout_vertices, layout_faces = trimesh.creation.triangulate_polygon(layout_polygon)
    floor_vertices = np.concatenate(
        [layout_vertices, np.zeros((layout_vertices.shape[0], 1))], axis=1
    )
    floor_mesh = trimesh.Trimesh(
        vertices=floor_vertices,
        faces=layout_faces,
        face_normals=np.array([[0, 0, 1]]).repeat(layout_faces.shape[0], axis=0),
    )

    ceiling_vertices = np.concatenate(
        [layout_vertices, wall_height * np.ones((layout_vertices.shape[0], 1))], axis=1
    )
    ceiling_mesh = trimesh.Trimesh(
        vertices=ceiling_vertices,
        faces=layout_faces,
        face_normals=np.array([[0, 0, -1]]).repeat(layout_faces.shape[0], axis=0),
    )

    mesh_list = [wall_mesh, floor_mesh, ceiling_mesh]
    color_list = [WALL_COLOR, FLOOR_COLOR, CEILING_COLOR]

    return mesh_list, color_list


def convert_object_to_trimesh(
    layout_data: Dict[str, Any], wireframe: trimesh.Trimesh, min_size=0.1
):
    meshes = []
    colors = []
    for bbox in layout_data["bboxes"]:
        color = colormaps.get(bbox["class"], None)

        if color is None:
            print(f"Skip unknown object ({bbox['class']})")
            continue

        size = np.abs(bbox["size"])

        if np.all(size < min_size):
            print(
                f"Skip tiny object ({bbox['class']}) with size {np.round(bbox['size'], 2)}"
            )
            continue

        transform = np.array(bbox["transform"]).reshape(4, 4)

        mesh = wireframe.copy()
        mesh.apply_scale(size)
        mesh.apply_transform(transform)

        meshes.append(mesh)
        colors.append(color)
    return meshes, colors


def parse_layout_data(
    layout_data: Dict[str, Any], wireframe: trimesh.Trimesh, min_size=0.1
):
    # room mesh: wall, floor, ceiling
    room_meshes, room_colors = convert_room_to_trimesh(layout_data)

    # object mesh
    object_meshes, object_colors = convert_object_to_trimesh(
        layout_data, wireframe, min_size
    )

    meshes = room_meshes + object_meshes
    colors = room_colors + object_colors

    return meshes, colors
