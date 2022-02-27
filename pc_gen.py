import os
import sys
import time
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import open3d as o3d
osp = os.path
np.random.seed(0)
script_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(script_dir, "build"))
import libndt


# random sample point
def create_random_pc():

    count = 10

    center_list = [
        [x, 0, 0]
        for x in np.linspace(0, 9.0, count)]

    center_list.extend([
        [5, y, 0]
        for y in np.linspace(-5, 5, count)])

    sample_count = 50

    sigma = [0.2, 0.2, 0.1]

    pc = []

    for center in center_list:
        pc.append(np.concatenate(
            [[np.random.normal(c, s, sample_count).astype(np.float32)]
             for s, c in zip(sigma, center)]
            ).T)
    pc = np.concatenate(pc)
    pc_list = pc.tolist()

    pc = np.array(pc_list).astype(np.float32).reshape((-1, 3))

    return pc.astype(np.float32)


def load_pc_from_pcd(path, voxel_filter=False):
    pcd = o3d.io.read_point_cloud(path)
    pcd = pcd.voxel_down_sample(voxel_size=0.2)
    return np.asarray(pcd.points).astype(np.float32), pcd


def create_random_fixed_pc():
    sample_count = 100
    center = [0.5, 0.5, 0.5]
    sigma = [0.3, 0.3, 0.1]

    pc = []

    pc.append(np.concatenate(
        [[np.random.normal(c, s, sample_count).astype(np.float32)]
         for s, c in zip(sigma, center)]
        ).T)
    pc = np.concatenate(pc)
    pc_list = pc.tolist()

    pc = np.array(pc_list).astype(np.float32).reshape((-1, 3))

    return pc.astype(np.float32)



# fixed point
def create_fixed_pc():
    # return np.array([[1, 0, 0]])
    return np.array([[x, 0, 0] for x in list(
        np.arange(-7.5, 7.5, 0.1))]).astype(np.float32)


def create_grid(c):
    value = 0.4
    ax_count = 4
    grid = []
    for x in np.linspace(-value, value, ax_count):
        for y in np.linspace(-value, value, ax_count):
            for z in np.linspace(-value/2, value/2, ax_count):
                grid.append([x+c[0], y+c[1], z+c[2]])
    return grid


def create_grid2(c):
    value = 0.4
    ax_count = 4
    grid = []
    for x in np.linspace(-value/2, value/2, ax_count):
        for y in np.linspace(-value, value, ax_count):
            for z in np.linspace(-value, value, ax_count):
                grid.append([x+c[0], y+c[1], z+c[2]])
    return grid


def create_grid3(c):
    value = 0.4
    ax_count = 4
    grid = []
    for x in np.linspace(-value, value, ax_count):
        for y in np.linspace(-value/2, value/2, ax_count):
            for z in np.linspace(-value, value, ax_count):
                grid.append([x+c[0], y+c[1], z+c[2]])
    return grid


# grid point
def create_fixed_grid_pc():
    refs = []

    for x in np.arange(0.5, 3.5, 1.0):
        for y in np.arange(0.5, 3.5, 1.0):
            refs.extend(create_grid([x, y, -0.5]))

    for z in np.arange(0.5, 3.5, 1.0):
        for y in np.arange(0.5, 3.5, 1.0):
            refs.extend(create_grid2([-0.5, y, z]))

    for z in np.arange(0.5, 3.5, 1.0):
        for x in np.arange(0.5, 3.5, 1.0):
            refs.extend(create_grid3([x, 3.5, z]))

    return np.array(refs, dtype=np.float32)


# triangle reference pointcloud.
def create_triangle_pc():
    return np.array([
        [-2/3, -2/3, 0],
        [1/3, -2/3, 0],
        [1/3, 4/3, 0]
    ], dtype=np.float32)
