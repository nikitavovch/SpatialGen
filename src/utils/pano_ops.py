import os
import cv2
import numpy as np
import open3d as o3d
import torch

from src.utils.typing import *

def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * np.pi


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * np.pi

def get_unit_spherical_map():
    h = 512
    w = 1024

    coorx, coory = np.meshgrid(np.arange(w), np.arange(h))
    us = np_coorx2u(coorx, w)
    vs = np_coory2v(coory, h)

    X = np.expand_dims(np.cos(vs) * np.sin(us), 2)
    Y = np.expand_dims(np.sin(vs), 2)
    Z = np.expand_dims(np.cos(vs) * np.cos(us), 2)
    unit_map = np.concatenate([X, Z, Y], axis=2)

    return unit_map

def torch_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * torch.tensor(np.pi, dtype=torch.float32)

def torch_coory2v(coory, coorH=512):
    return ((coory + 0.5) / coorH - 0.5) * torch.tensor(np.pi, dtype=torch.float32)

def get_unit_spherical_map_torch():
    h = 512
    w = 1024

    coory, coorx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    us = torch_coorx2u(coorx, w)
    vs = torch_coory2v(coory, h)

    X = torch.unsqueeze(torch.cos(vs) * torch.sin(us), 2)
    Y = torch.unsqueeze(torch.cos(vs) * torch.cos(us), 2)
    Z = torch.unsqueeze(torch.sin(vs), 2)
    unit_map = torch.cat([X, Y, Z], dim=2) * -1

    return unit_map


def vis_color_pointcloud(depth_img:np.ndarray, rgb_img:np.ndarray=None, depth_scale:float=1.0, saved_color_pcl_filepath:str='./pointcloud.ply')->o3d.geometry.PointCloud:
    """
    :param rgb_img: rgb panorama image 
    :param depth_img: depth panorama image 
    :param saved_color_pcl_filepath: saved color point cloud filepath
    :return: o3d.geometry.PointCloud
    """

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw([inlier_cloud, outlier_cloud])

    assert depth_img.shape[:2] == (512, 1024)

    if np.isnan(depth_img.any()) or len(depth_img[depth_img > 0]) == 0:
        print('empyt depth image')
        exit(-1)

    if rgb_img is not None: 
        assert rgb_img.shape[:2] == (512, 1024)
        
        if rgb_img.shape[2] == 4:
            rgb_img = rgb_img[:, :, :3]
        if np.isnan(rgb_img.any()) or len(rgb_img[rgb_img > 0]) == 0:
            print('empyt rgb image')
            exit(-1)
        color = np.clip(rgb_img, 0.0, 255.0) / 255.0

    depth_img = np.expand_dims((depth_img / depth_scale), axis=2)
    pointcloud = depth_img * get_unit_spherical_map()
    pointcloud = pointcloud.reshape(-1, 3)
    o3d_pointcloud = o3d.geometry.PointCloud()
    o3d_pointcloud.points = o3d.utility.Vector3dVector(pointcloud)
    o3d_pointcloud.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3)) if rgb_img is not None else o3d.utility.Vector3dVector(np.random.rand(pointcloud.shape[0], 3))
    # must constrain normals pointing towards camera
    o3d_pointcloud.estimate_normals()
    o3d_pointcloud.orient_normals_towards_camera_location(camera_location=(0, 0, 0))
    # remove outliers
    # cl, ind = o3d_pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # display_inlier_outlier(o3d_pointcloud, ind)
    o3d.io.write_point_cloud(saved_color_pcl_filepath, o3d_pointcloud)
    return pointcloud

def unitxyz2uv(xyz, equ_w:int, equ_h:int, normlized:bool= False):
    # reverse of get_unit_spherical_map_torch
    x, y, z = torch.split(-xyz, 1, dim=-1)
    phi = torch.arctan2(x, y)
    c = torch.sqrt(x ** 2 + y ** 2)
    theta = torch.arctan2(z, c)

    # longitude and latitude to equirectangular coordinate
    if normlized:
        u = (phi / (2 * np.pi) + 0.5)
        v = (theta / np.pi + 0.5)
    else:
        u = (phi / (2 * np.pi) + 0.5) * equ_w - 0.5
        v = (theta / np.pi + 0.5) * equ_h - 0.5
    return [u, v]
    
def fibonacci_spiral_samples_on_unit_sphere(nb_samples, mode=0):
    shift = 1.0 if mode == 0 else nb_samples * torch.randn()

    # ga = np.pi * (3.0 - np.sqrt(5.0))
    ga = torch.tensor(np.pi * (3.0 - np.sqrt(5.0)), dtype=torch.float32)
    offset = 2.0 / nb_samples

    # ss = np.zeros((nb_samples, 3))
    ss = torch.zeros((nb_samples, 3), dtype=torch.float32)
    j = 0
    for i in range(nb_samples):
        phi = ga * ((i + shift) % nb_samples)
        # cos_phi = np.cos(phi)
        cos_phi = torch.cos(phi)
        # sin_phi = np.sin(phi)
        sin_phi = torch.sin(phi)
        cos_theta = ((i + 0.5) * offset) - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        # ss[j, :] = np.array([sin_phi * sin_theta, cos_theta, cos_phi * sin_theta])
        ss[j, :] = torch.tensor([sin_phi * sin_theta, cos_theta, cos_phi * sin_theta], dtype=torch.float32)
        j += 1
    return ss

def get_pano_fibonacci_mask(nb_samples:int=60000, width:int=1024, height:int=512):
    sampled_point = fibonacci_spiral_samples_on_unit_sphere(nb_samples=nb_samples)
    mask = torch.zeros((height, width), dtype=torch.float32)
    for sampled_point_i in range(sampled_point.shape[0]):
        xyz = sampled_point[sampled_point_i, :]
        uv = unitxyz2uv(xyz, equ_w=width, equ_h=height)
        mask[int(uv[1]), int(uv[0])] = 1
    # mask = np.where(mask > 0.5, True, False)
    mask = torch.where(mask > 0.5, torch.tensor(True), torch.tensor(False))
    return mask

def cvt_to_spherical_pointcloud(depth_img:Float[Tensor, "B Hi Wi 1"], 
                                rgb_img:Float[Tensor, "B Hi Wi 3"]=None, 
                                depth_scale:float=1.0, 
                                num_sample_points:int=100000,
                                sampling_mask: Float[Tensor, "Hi Wi"]=None,
                                saved_color_pcl_filepath:str=None)->o3d.geometry.PointCloud:
    """
    :param rgb_img: rgb panorama image 
    :param depth_img: depth panorama image 
    :param depth_scale: depth scale
    :param saved_color_pcl_filepath: saved color point cloud filepath
    :return: o3d.geometry.PointCloud
    """
    batch_size, H, W, _ = depth_img.shape
    assert (H, W) == (512, 1024)

    if sampling_mask is None:
        sampling_mask = get_pano_fibonacci_mask(nb_samples=num_sample_points).to(depth_img.device)
    
    if torch.isnan(depth_img.any()) or len(depth_img[depth_img > 0]) == 0:
        print('empyt depth image')
        return torch.zeros_like(rgb_img).reshape(batch_size, -1, 3)

    depth_img = depth_img / depth_scale
    unit_coord_map = get_unit_spherical_map_torch()
    pointcloud = depth_img * unit_coord_map[None, :, :, :].to(depth_img.device)
    sampling_mask = sampling_mask[None, :, :].expand(batch_size, -1, -1)
    pointcloud = pointcloud[sampling_mask]
    pointcloud = pointcloud.reshape(batch_size, -1, 3)
    
    if rgb_img is not None: 
        assert (rgb_img.shape[1], rgb_img.shape[2]) == (512, 1024)
        
        if rgb_img.shape[3] == 4:
            rgb_img = rgb_img[:, :, :, :3]
        color = rgb_img[sampling_mask]
        color = color.reshape(batch_size, -1, 3)
        pointcloud = torch.concat([pointcloud, color], dim=2)
    
    if saved_color_pcl_filepath is not None:
        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(pointcloud[:,:, :3].reshape(-1, 3).detach().cpu().numpy())
        o3d_pointcloud.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3).detach().cpu().numpy()) if rgb_img is not None else o3d.utility.Vector3dVector(np.random.rand(pointcloud.shape[0], 3))
        # must constrain normals pointing towards camera
        o3d_pointcloud.estimate_normals()
        o3d_pointcloud.orient_normals_towards_camera_location(camera_location=(0, 0, 0))
        o3d.io.write_point_cloud(saved_color_pcl_filepath, o3d_pointcloud)
    return pointcloud
