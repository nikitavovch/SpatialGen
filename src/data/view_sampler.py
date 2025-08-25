import math

import torch
import numpy as np
from shapely.geometry import Polygon, Point
import trimesh
from matplotlib import pyplot as plt

from src.utils.typing import *
from src.utils.rotation_ops import matrix_to_euler_angles_torch, matrix_to_euler_angles_np, make_rotation_by_up_and_eye
from src.utils.layout_utils import room_meta_to_polygon, largest_rectangle_in_polygon, shrink_or_swell_polygon, create_oriented_bboxes, convert_oriented_box_to_trimesh_fmt
from src.utils.pcl_ops import are_points_collinear

class ViewSampler:
    def __init__(
        self,
        room_metadata: Dict[str, Any],
        num_sample_views: int = 8,
        yaw_interval_thresh: float = 60.0,
        distance_interval_thresh: float = 1.0,
        trajectory_type: str = "line",
        is_validation: bool = False
    ):
        self._num_sample_views = num_sample_views
        self._yaw_interval_thresh = yaw_interval_thresh
        self._distance_interval_thresh = distance_interval_thresh
        self._trajectory_type = trajectory_type
        self.is_validation = is_validation

        self.room_polygon, self.room_width, self.room_length, self.ori_room_polygon = self.parse_room_polygon(room_metadata)
        self.obj_bbox_polygons = self.parse_object_bbox_polygon(room_metadata, self.room_polygon)
        

    @staticmethod
    def parse_room_polygon(
        room_metadata: Dict[str, Any], scale_factor: float = 0.001, shrink_factor: float = 0.20, simplify_tol: float = 0.01
    ) -> Tuple[Polygon, float, float]:
        # parse original room layout
        ori_room_polygon = room_meta_to_polygon(room_meta_dict=room_metadata, SCALE=scale_factor)
        if ori_room_polygon.geom_type == "MultiPolygon":
            # take the polygon with the largest area
            areas = [poly.area for poly in ori_room_polygon.geoms]
            max_indice = np.argmax(areas)
            ori_room_polygon = ori_room_polygon.geoms[max_indice]
        # # show room layout polygon
        # from matplotlib import pyplot as plt
        # plt.figure()
        # xs, ys = ori_room_polygon.exterior.xy
        # plt.plot(xs, ys)

        # parse the largest rectangle embeded the room layout
        # largest_rect, shrinked_room_polygon, area = largest_rectangle_in_polygon(ori_room_polygon, buffer=0.5)
        # largest_rect_xs, largest_rect_ys = largest_rect.exterior.xy
        # plt.plot(largest_rect_xs, largest_rect_ys, "g-")

        # shrink the largest rectangle
        shrinked_room_polygon = shrink_or_swell_polygon(ori_room_polygon, shrink_factor=shrink_factor, swell=False)
        if shrinked_room_polygon.geom_type == "MultiPolygon":
            # take the polygon with the largest area
            areas = [poly.area for poly in shrinked_room_polygon.geoms]
            max_indice = np.argmax(areas)
            shrinked_room_polygon = shrinked_room_polygon.geoms[max_indice]
        # shrinked_room_polygon = shrinked_room_polygon.simplify(tolerance=simplify_tol, preserve_topology=True)
        shrinked_xs, shrinked_ys = shrinked_room_polygon.exterior.xy
        # plt.plot(shrinked_xs, shrinked_ys)
        # plt.savefig("roomlayout_polygons.png")

        # min max corners
        min_xs, min_ys = min(shrinked_xs), min(shrinked_ys)
        max_xs, max_ys = max(shrinked_xs), max(shrinked_ys)
        width = max_xs - min_xs
        length = max_ys - min_ys
        # print(f"layout width: {width}, height: {height}")

        return shrinked_room_polygon, width, length, ori_room_polygon
    
    @staticmethod
    def parse_object_bbox_polygon(room_metadata: Dict[str, Any], room_polygon: Polygon ) -> List[Polygon]:
        # parse room_meta_dict to object bbox polygons
        object_bbox_polygons = []
        
        cam0_dict = room_metadata["cameras"]["0"]
        if "bboxes" not in cam0_dict:
            return object_bbox_polygons
        bbox_info_in_cam0 = cam0_dict["bboxes"]
        T_enu_cv = np.array(room_metadata["T_enu_cv"]).reshape(4, 4)
        T_cv_enu = np.linalg.inv(T_enu_cv)
        T_w2c0 = np.array(cam0_dict["camera_transform"]).reshape(4, 4)
        T_c02w = np.linalg.inv(T_w2c0)
        
        # obbs = create_oriented_bboxes(bbox_info_in_cam0, scale=1.0)
        # obbs.apply_transform(T_cv_enu)
        # obbs.apply_transform(T_c02w)
        # obbs.export('object_bboxs_world.ply')
        for bbox_info in bbox_info_in_cam0:
            box_sizes = np.array(bbox_info['size'])
            if np.any(box_sizes > 10) or np.any(box_sizes < 0.1):
                # print(f"skip bbox too large or too tiny! box_sizes: {box_sizes}")
                continue
            transform_matrix = np.array(bbox_info["transform"]).reshape(4, 4)
            T_cambox2w = T_cv_enu @ transform_matrix
            T_box2w = T_c02w @ T_cambox2w
            bbox_info["transform"] = T_box2w.flatten().tolist()
            if not room_polygon.contains(Point(T_box2w[:2, 3])):
                # print(f"object bbox center is outside the room!")
                continue
            bbox_world_mesh = convert_oriented_box_to_trimesh_fmt(bbox_info)
            bbox_bottom_corners = trimesh.bounds.corners(bbox_world_mesh.bounding_box_oriented.bounds)[:4, :2]
            # print(f'bbox_bottom_corners: {bbox_bottom_corners}')
            bbox_polygon = Polygon(bbox_bottom_corners)
            object_bbox_polygons.append(bbox_polygon)
        
        # show bbox polygons
        # from matplotlib import pyplot as plt
        # plt.figure()
        # for bbox_polygon in object_bbox_polygons:
        #     xs, ys = bbox_polygon.exterior.xy
        #     plt.plot(xs, ys)
        # # save figure
        # plt.savefig("object_bbox_polygons.png")
        return object_bbox_polygons
        
    def is_point_inside_object_bbox(self, point: Point) -> bool:
        for obj_bbox_polygon in self.obj_bbox_polygons:
            distance_2d = point.distance(obj_bbox_polygon)
            if obj_bbox_polygon.contains(point) or distance_2d < 0.4:
                return True
        return False

    def sample(self, valid_frames: List[str], cameras_dict: Dict[str, Any], intrinsics: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """
        Sample input and target views from a room.
        Args:
            valid_frames: valid frame indices
            cameras_dict: camera-to-world extrinsics, shape (num_views, 4, 4)
            intrinsics: camera intrinsics, shape (3, 3)
            kwargs: additional arguments
        Returns:
            sample_views: indices for input and target views, shape (num_sample_views,)
        """
        if intrinsics is None:
            intrinsics = np.array([[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]]).repeat(len(valid_frames), axis=0)

        if self._trajectory_type == "line":
            return self.sample_straight_line(valid_frames, cameras_dict, intrinsics, **kwargs)
        elif self._trajectory_type == "panoramic":
            return self.sample_panoramic(valid_frames, cameras_dict, intrinsics, **kwargs)
        elif self._trajectory_type == "spiral":
            return self.sample_spiral(valid_frames, cameras_dict, intrinsics, **kwargs)
        elif self._trajectory_type == "randomwalk":
            return self.sample_randomwalk(valid_frames, cameras_dict, intrinsics, **kwargs)
        elif self._trajectory_type == "evenBins":
            return self.sample_even_bins(valid_frames, cameras_dict, intrinsics, **kwargs)
        else:
            raise NotImplementedError(f"Unknown trajectory type: {self._trajectory_type}")

    def get_valid_camera_keys(self, valid_frames: List[str], cameras_dict: Dict[str, Any], room_path: str) -> List[str]:

        cam_height = 1.2  # camera height, TODO: get from mean of all cameras
        # 1. load room layout, get the boundary of the room
        self.room_polygon: Polygon

        # # show room layout and object bbox polygons
        # plt.figure()
        # xs, ys = self.room_polygon.exterior.xy
        # plt.plot(xs, ys)
        # for obj_bbox_polygon in self.obj_bbox_polygons:
        #     xs, ys = obj_bbox_polygon.exterior.xy
        #     plt.plot(xs, ys)

        # load all cameras
        valid_camera_ids = []
        for cam_id_str in valid_frames:
            c2w_pose = self._load_koolai_camera_pose(cameras_dict, int(cam_id_str))
            cam_pos_x = c2w_pose[0, 3]
            cam_pos_y = c2w_pose[1, 3]
            point_2d = Point(cam_pos_x, cam_pos_y)

            # if the camera is inside the room, and the camera is not too close to the object bbox
            if self.room_polygon.contains(point_2d) and not self.is_point_inside_object_bbox(point_2d):
                valid_camera_ids.append(cam_id_str)
                # plt.plot(cam_pos_x, cam_pos_y, "r*")
        # # save figure 
        # fig_path = os.path.join(room_path, "room_and_object_bbox_polygons.png")
        # plt.savefig("room_and_object_bbox_polygons.png")   
        
        if len(valid_camera_ids) == 0:
            valid_camera_ids = valid_frames
            print("No valid cameras inside")
        return valid_camera_ids

                             
    @staticmethod
    def _load_koolai_camera_pose(camera_meta_dict: Dict, idx: int) -> Float[Tensor, "4 4"]:
        """load c2w pose from camera meta

        Args:
            camera_meta_dict (Dict): camera meta json
            idx (int): camera idx

        Returns:
            torch.Tensor: pose
        """
        cam_meta = camera_meta_dict[str(idx)]

        # w2c pose
        pose = np.array(cam_meta["camera_transform"]).reshape(4, 4)
        c2w_pose = np.linalg.inv(pose)
        c2w_pose = torch.from_numpy(c2w_pose).float()
        return c2w_pose

    def sample_straight_line(self, valid_frames: List[str], cameras_dict: Dict[str, Any], intrinsics: np.ndarray, **kwargs) -> Dict[str, Any]:
        if 'b_make_sampled_views_consecutive' in kwargs:
            b_make_sampled_views_consecutive = kwargs['b_make_sampled_views_consecutive']
        else:
            b_make_sampled_views_consecutive = False
        num_sample_views = self._num_sample_views
        num_valid_views = len(valid_frames)
        if num_valid_views < num_sample_views:
            # padding the valid frames
            valid_frames = valid_frames * (num_sample_views // num_valid_views + 1)
                    
        valid_frames_tmp = valid_frames
        cam_height = 1.2  # camera height, TODO: get from mean of all cameras
        # 1. load room layout, get the boundary of the room
        self.room_polygon: Polygon
        room_centroid = np.array([self.room_polygon.centroid.x, self.room_polygon.centroid.y, cam_height]).astype(np.float32)
        
        ret_dict = {}
        ret_keys = None
        ret_yaws = []
        
        if not self.is_validation:
            # traverse the valid_frames_tmp, get num_input_views context views that lie on the same line
            for i in range(len(valid_frames_tmp) - num_sample_views + 1):
                sample_view_keys = valid_frames_tmp[i : i + num_sample_views]
                sample_c2w_cams = [self._load_koolai_camera_pose(cameras_dict, key) for key in sample_view_keys]
                # 获取当前连续的 8 个相机的位置
                candidate_positions = [c2w[:3, 3].numpy() for c2w in sample_c2w_cams]

                # 检查是否共线
                if are_points_collinear(candidate_positions):
                    # 返回满足条件的相机 ID
                    ret_keys = sample_view_keys
                else:
                    continue
                
        if ret_keys is None or self.is_validation:
            ret_keys = valid_frames[:num_sample_views]
        # 20241118_3FO4K5FWU0T5_perspective_room_1151
        # ret_keys[:11] = ['35', '45', '62', '72', '90', '102', '103', '104', '118', '138', '140']
        # print(f"ret_keys: {ret_keys}")
        # compute relative rotation w.r.t the room centroid
        sample_c2w_cams = [self._load_koolai_camera_pose(cameras_dict, key) for key in ret_keys]
        sample_c2w_cams = torch.stack(sample_c2w_cams, dim=0)
        dis_to_centroids = sample_c2w_cams[:, :3, 3] - room_centroid
        dirs_to_centroid = dis_to_centroids / (np.linalg.norm(dis_to_centroids, axis=-1, keepdims=True) + 1e-6)

        # # show room layout and object bbox polygons
        # plt.figure()
        # xs, ys = self.ori_room_polygon.exterior.xy
        # plt.plot(xs, ys)
        # for obj_bbox_polygon in self.obj_bbox_polygons:
        #     xs, ys = obj_bbox_polygon.exterior.xy
        #     plt.plot(xs, ys)
            
        # calculate the orientation between camera direction and centroid
        for idx, c2w_pose in enumerate(sample_c2w_cams[0:1]):
            dir_to_centroid = dirs_to_centroid[idx]
            c2w_R = c2w_pose[:3, :3]
            cam_pos_x = c2w_pose[0, 3]
            cam_pos_y = c2w_pose[1, 3]

            # construct new rotation matrix to make the camera direct to the centroid
            new_c2w_R = make_rotation_by_up_and_eye(up=np.array([0.0, 0.0, 1.0]), eye=dir_to_centroid)
            # rotation from c2w_R to new_c2w_R
            delta_R = np.dot(new_c2w_R.T, c2w_R)
            delta_angle_deg = matrix_to_euler_angles_np(delta_R) * 180 / np.pi
            delta_yaw_deg = -delta_angle_deg[2]
            ret_yaws.append(float(delta_yaw_deg))
            # print(f"camera {ret_keys[idx]} rotate angle: {delta_yaw_deg}")
            # plt.plot(cam_pos_x, cam_pos_y, "r*")
        # save figure 
        # plt.savefig("room_and_object_bbox_polygons.png")        
    
        # compute relative rotation w.r.t the first view
        relative_rotations = sample_c2w_cams[0:1, :3, :3].transpose(1, 2) @ sample_c2w_cams[:, :3, :3]
        # rotation angles from c_i to c_0
        relative_rotation_degs = matrix_to_euler_angles_torch(relative_rotations) * 180.0 / math.pi
        relative_yaws = relative_rotation_degs[:, 2].cpu().numpy().tolist()
        ret_yaws = [ret_yaws[0] + relative_yaw for relative_yaw in relative_yaws]

        # compute relative distance w.r.t the first view
        # relative_distances = sample_c2w_cams[:, :3, 3] - sample_c2w_cams[0:1, :3, 3]
        # relative_distances = torch.norm(relative_distances, dim=-1)
        # _, indices = torch.sort(relative_distances, descending=False)
        # new_indices = indices.cpu().numpy()
        new_indices = np.arange(len(ret_keys))
        # new_indices = np.roll(new_indices, 1)  #  make the first view as the reference view

        ret_dict["sample_keys"] = [ret_keys[i] for i in new_indices]
        ret_dict["sample_yaws"] = [ret_yaws[i] for i in new_indices]
            
        ret_dict["sample_rolls"] = [float(0.0)] * num_sample_views
        ret_dict["sample_pitches"] = [float(0.0)] * num_sample_views
        return ret_dict

    def sample_panoramic(self, valid_frames: List[str], cameras_dict: Dict[str, Any], intrinsics: np.ndarray, **kwargs) -> Dict[str, Any]:
        if 'b_make_sampled_views_consecutive' in kwargs:
            b_make_sampled_views_consecutive = kwargs['b_make_sampled_views_consecutive']
        else:
            b_make_sampled_views_consecutive = False
        num_sample_views = self._num_sample_views

        cam_height = 1.2  # camera height, TODO: get from mean of all cameras
        # 1. load room layout, get the boundary of the room
        self.room_polygon: Polygon
        room_centroid = np.array([self.room_polygon.centroid.x, self.room_polygon.centroid.y, cam_height]).astype(np.float32)
        dist_to_centroid_thresh = 0.3  # ratio of distance to the centroid of the room
        dist_thresh = dist_to_centroid_thresh * min(self.room_width, self.room_length)

        if len(self.obj_bbox_polygons) == 0:
            # load all cameras
            valid_camera_ids = []
            for cam_id_str in valid_frames:
                c2w_pose = self._load_koolai_camera_pose(cameras_dict, int(cam_id_str))
                cam_pos_x = c2w_pose[0, 3]
                cam_pos_y = c2w_pose[1, 3]
                # plt.plot(cam_pos_x, cam_pos_y, "b*")
                point_2d = Point(cam_pos_x, cam_pos_y)

                if self.room_polygon.contains(point_2d):
                    # if the camera is inside the room
                    distance_2d = point_2d.distance(Point(room_centroid[:2]))
                    # if the camera is inside the room, and the camera is not too far from the centroid
                    if distance_2d <= dist_thresh:
                        valid_camera_ids.append(cam_id_str)

            if len(valid_camera_ids) == 0:
                valid_camera_ids = valid_frames
                print("No valid cameras inside the room.")
        else:
            #  the input valid_frames are the cameras that are not too close to the object bbox
            valid_camera_ids = valid_frames
        len_valid_cameras = len(valid_camera_ids)

        ret_dict = {}
        # take panorama on a single viewpoint
        # ret_keys = [valid_camera_ids[np.random.randint(0, len_valid_cameras)]] * num_sample_views
        key_indices = np.random.choice(len_valid_cameras, min(num_sample_views, len_valid_cameras), replace=False)
        ret_keys = [valid_camera_ids[idx] for idx in key_indices for _ in range(8)]  # repeat inteleavely
        ret_keys = ret_keys[:num_sample_views]
        # ret_keys[:16] = ['64', '64', '64', '64', '64', '64', '64', '64', '91', '91', '91', '91', '91', '91', '91', '91']
        # print(f"ret_keys: {ret_keys}")

        # compute the yaw angle for the first view, make it direct to the centroid
        init_yaw_degree = 0.0
        sample_c2w_cam0 = self._load_koolai_camera_pose(cameras_dict, ret_keys[0])
        dis_to_centroid = sample_c2w_cam0[:3, 3] - room_centroid
        dir_to_centroid = dis_to_centroid / (np.linalg.norm(dis_to_centroid) + 1e-6)

        # calculate the orientation between camera direction and centroid
        c2w_R = sample_c2w_cam0[:3, :3]

        # construct new rotation matrix to make the camera direct to the centroid
        new_c2w_R = make_rotation_by_up_and_eye(up=np.array([0.0, 0.0, 1.0]), eye=dir_to_centroid)
        # rotation from c2w_R to new_c2w_R
        delta_R = np.dot(new_c2w_R.T, c2w_R)
        delta_angle_deg = matrix_to_euler_angles_np(delta_R) * 180 / np.pi
        init_yaw_degree = -delta_angle_deg[2]

        # randomly shuffle the views
        ret_dict["sample_keys"] = ret_keys

        prev_yaw_degree = init_yaw_degree
        ret_dict["sample_yaws"] = [float(prev_yaw_degree)]        
        for i in range(1, num_sample_views):
            # increase the yaw degree by 60 degree
            # subview_yaw_degree = prev_yaw_degree + np.random.normal(loc=self._yaw_interval_thresh, scale=5.0)  # [45, 75]
            # increase the yaw by 45 degree
            subview_yaw_degree = prev_yaw_degree + 45.0
            ret_dict["sample_yaws"].append(float(subview_yaw_degree))
            prev_yaw_degree = subview_yaw_degree

        ret_dict["sample_rolls"] = [float(0.0)] * num_sample_views
        ret_dict["sample_pitches"] = [float(0.0)] * num_sample_views
        return ret_dict

    def sample_spiral(self, valid_frames: List[str], cameras_dict: Dict[str, Any], intrinsics: np.ndarray, **kwargs) -> Dict[str, Any]:
        if 'b_make_sampled_views_consecutive' in kwargs:
            b_make_sampled_views_consecutive = kwargs['b_make_sampled_views_consecutive']
        else:
            b_make_sampled_views_consecutive = False
        num_sample_views = self._num_sample_views
        num_valid_views = len(valid_frames)
        if num_valid_views < num_sample_views:
            # padding the valid frames
            valid_frames = valid_frames * (num_sample_views // num_valid_views + 1)

        cam_height = 1.2  # camera height, TODO: get from mean of all cameras
        dist_to_centroid_max_thresh = 0.5  # ratio of distance to the centroid of the room
        dist_to_centroid_min_thresh = 0.2  # ratio of distance to the centroid of the room
        # 1. load room layout, get the boundary of the room
        self.room_polygon: Polygon
        room_centroid = np.array([self.room_polygon.centroid.x, self.room_polygon.centroid.y, cam_height]).astype(np.float32)
        distance_max = dist_to_centroid_max_thresh * min(self.room_width, self.room_length)
        distance_min = dist_to_centroid_min_thresh * min(self.room_width, self.room_length)
        
        ret_dict = {}
        ret_keys = None
        ret_yaws = []

        # # show room layout and object bbox polygons
        # plt.figure()
        # xs, ys = self.ori_room_polygon.exterior.xy
        # plt.plot(xs, ys)
        # for obj_bbox_polygon in self.obj_bbox_polygons:
        #     xs, ys = obj_bbox_polygon.exterior.xy
        #     plt.plot(xs, ys)
        # if len(self.obj_bbox_polygons) > 0:
        #     random_obj_bbox_polygon = self.obj_bbox_polygons[np.random.randint(0, len(self.obj_bbox_polygons))]
        # else:
        #     random_obj_bbox_polygon = None
        random_obj_bbox_polygon = None
        
        min_distances_to_object = {}
        # load all cameras
        valid_camera_ids = []
        if not self.is_validation:
            for cam_id_str in valid_frames:
                c2w_pose = self._load_koolai_camera_pose(cameras_dict, int(cam_id_str))
                cam_pos_x = c2w_pose[0, 3]
                cam_pos_y = c2w_pose[1, 3]
                # plt.plot(cam_pos_x, cam_pos_y, "b*")
                point_2d = Point(cam_pos_x, cam_pos_y)

                if random_obj_bbox_polygon is not None:
                    distance_2_obj = point_2d.distance(random_obj_bbox_polygon)
                    min_distances_to_object[cam_id_str] = distance_2_obj
                    valid_camera_ids.append(cam_id_str)
                else:
                    # if the camera is inside the room, and the camera is not too far from the centroid
                    if self.room_polygon.contains(point_2d):
                        distance_2d = point_2d.distance(Point(room_centroid[:2]))
                        if distance_2d <= distance_max:
                            valid_camera_ids.append(cam_id_str)
            
            if random_obj_bbox_polygon is not None:
                # sort the cameras by the distance to the object
                valid_camera_ids = sorted(valid_camera_ids, key=lambda x: min_distances_to_object[x])
                valid_camera_ids = valid_camera_ids[:num_sample_views]
                room_centroid = np.array([random_obj_bbox_polygon.centroid.x, random_obj_bbox_polygon.centroid.y, cam_height]).astype(np.float32)
                # valid_camera_poses = [self._load_koolai_camera_pose(cameras_dict, int(cam_id_str)) for cam_id_str in valid_camera_ids]
                # for cam_pose in valid_camera_poses:
                #     cam_pos_x = cam_pose[0, 3]
                #     cam_pos_y = cam_pose[1, 3]
                #     plt.plot(cam_pos_x, cam_pos_y, "r*")

        # # save figure 
        # plt.savefig("room_and_object_bbox_polygons.png")
        if len(valid_camera_ids) == 0 or self.is_validation:
            valid_camera_ids = valid_frames
        #  evenly sample the expected number of views from the valid cameras
        len_valid_cameras = len(valid_camera_ids)
        if len_valid_cameras < num_sample_views:
            # padding valid_camera_ids
            valid_camera_ids = valid_camera_ids * (num_sample_views // len_valid_cameras + 1)
        ret_keys = valid_camera_ids[:num_sample_views]
        # ret_keys[:16] = ['37', '39', '62', '65', '90', '94', '102', '118', '109', '123', '138', '139', '140', '153', '155', '156', ]
        # print(f"ret_keys: {ret_keys}")

        valid_camera_poses = [self._load_koolai_camera_pose(cameras_dict, int(cam_id_str)) for cam_id_str in ret_keys]
        valid_camera_poses = np.stack(valid_camera_poses, axis=0)

        dis_to_centroids = valid_camera_poses[:, :3, 3] - room_centroid
        dirs_to_centroid = dis_to_centroids / (np.linalg.norm(dis_to_centroids, axis=-1, keepdims=True) + 1e-6)

        # calculate the orientation between camera direction and centroid
        for idx, c2w_pose in enumerate(valid_camera_poses):
            dir_to_centroid = dirs_to_centroid[idx]
            c2w_R = c2w_pose[:3, :3]

            # construct new rotation matrix to make the camera direct to the centroid
            new_c2w_R = make_rotation_by_up_and_eye(up=np.array([0.0, 0.0, 1.0]), eye=dir_to_centroid)
            # rotation from c2w_R to new_c2w_R
            delta_R = np.dot(new_c2w_R.T, c2w_R)
            delta_angle_deg = matrix_to_euler_angles_np(delta_R) * 180 / np.pi
            delta_yaw_deg = -delta_angle_deg[2]
            ret_yaws.append(float(delta_yaw_deg))
            # print(f"camera {ret_keys[idx]} rotate angle: {delta_yaw_deg}")

        ret_dict["sample_keys"] = ret_keys
        ret_dict["sample_yaws"] = ret_yaws
        # ret_dict["sample_rolls"] = ((np.random.rand(num_sample_views) - 0.5) * 2 * 15).tolist()     # [-30, 30]
        ret_dict["sample_rolls"] = [float(0.0)] * num_sample_views
        # ret_dict["sample_pitches"] = (np.random.rand(num_sample_views) * -30).tolist()   # [-30, 0]
        ret_dict["sample_pitches"] = [float(0.0)] * num_sample_views

        return ret_dict
    
    def sample_randomwalk(self, valid_frames: List[str], cameras_dict: Dict[str, Any], intrinsics: np.ndarray, **kwargs) -> Dict[str, Any]:
        # always make the sampled views consecutive
        b_make_sampled_views_consecutive = True
        num_sample_views = self._num_sample_views
        cam_height = 1.2  # camera height, TODO: get from mean of all cameras
        # 1. load room layout, get the boundary of the room
        self.room_polygon: Polygon
        room_centroid = np.array([self.room_polygon.centroid.x, self.room_polygon.centroid.y, cam_height]).astype(np.float32)
        dist_to_centroid_thresh = 0.7  # ratio of distance to the centroid of the room
        dist_thresh = dist_to_centroid_thresh * min(self.room_width, self.room_length)

        # if not self.is_validation:
        if len(self.obj_bbox_polygons) == 0:
            # load all cameras
            valid_camera_ids = []
            for cam_id_str in valid_frames:
                c2w_pose = self._load_koolai_camera_pose(cameras_dict, int(cam_id_str))
                cam_pos_x = c2w_pose[0, 3]
                cam_pos_y = c2w_pose[1, 3]
                # plt.plot(cam_pos_x, cam_pos_y, "b*")
                point_2d = Point(cam_pos_x, cam_pos_y)

                if self.room_polygon.contains(point_2d):
                    # if the camera is inside the room
                    distance_2d = point_2d.distance(Point(room_centroid[:2]))
                    # if the camera is inside the room, and the camera is not too far from the centroid
                    if distance_2d <= dist_thresh:
                        valid_camera_ids.append(cam_id_str)

            if len(valid_camera_ids) == 0:
                valid_camera_ids = valid_frames
                room_uid = kwargs.get("room_uid", None)
                if room_uid is not None:
                    print(f"Room {room_uid} has no valid cameras inside the room.")
                # else:
                #     print("No valid cameras inside the room.")
        else:
            #  the input valid_frames are the cameras that are not too close to the object bbox
            valid_camera_ids = valid_frames

        len_valid_cameras = len(valid_camera_ids)

        ret_dict = {}
        # easy mode: take N consecutive images
        if len_valid_cameras < num_sample_views:
            # padding the views
            valid_camera_ids = list(valid_camera_ids) * (num_sample_views // len_valid_cameras + 1)
        ret_keys = valid_camera_ids[:num_sample_views]
        # ret_keys[:16] = ['37', '40', '63', '70', '92', '93', '102', '116', '109', '123', '137', '138', '140', '153', '155', '156', ]
        # print(f"ret_keys: {ret_keys}")
        # compute the yaw angle for the first view, make it direct to the centroid
        init_yaw_degree = 0.0
        sample_c2w_cam0 = self._load_koolai_camera_pose(cameras_dict, ret_keys[0])
        dis_to_centroid = sample_c2w_cam0[:3, 3] - room_centroid
        dir_to_centroid = dis_to_centroid / (np.linalg.norm(dis_to_centroid) + 1e-6)

        # calculate the orientation between camera direction and centroid
        c2w_R = sample_c2w_cam0[:3, :3]

        # construct new rotation matrix to make the camera direct to the centroid
        new_c2w_R = make_rotation_by_up_and_eye(up=np.array([0.0, 0.0, 1.0]), eye=dir_to_centroid)
        # rotation from c2w_R to new_c2w_R
        delta_R = np.dot(new_c2w_R.T, c2w_R)
        delta_angle_deg = matrix_to_euler_angles_np(delta_R) * 180 / np.pi
        init_yaw_degree = -delta_angle_deg[2]

        # randomly shuffle the views
        ret_dict["sample_keys"] = ret_keys

        prev_yaw_degree = init_yaw_degree
        ret_dict["sample_yaws"] = [float(prev_yaw_degree)]        
        for i in range(1, num_sample_views):
            # increase the yaw degree by 60 degree
            subview_yaw_degree = prev_yaw_degree + np.random.normal(loc=self._yaw_interval_thresh, scale=5.0)  # [45, 75]
            ret_dict["sample_yaws"].append(float(subview_yaw_degree))
            prev_yaw_degree = subview_yaw_degree

        ret_dict["sample_rolls"] = [float(0.0)] * num_sample_views
        # ret_dict["sample_pitches"] = ((np.random.rand(num_sample_views) - 0.5) * 2 * 30).tolist() # [-30, 30]
        ret_dict["sample_pitches"] = [float(0.0)] * num_sample_views # [-30, 30]
        return ret_dict

    def sample_even_bins(self, valid_frames: List[str], cameras_dict: Dict[str, Any], intrinsics: np.ndarray, **kwargs) -> List[str]:
        raise NotImplementedError("Even bins sampling is not implemented yet.")
