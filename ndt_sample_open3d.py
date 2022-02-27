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

import pc_gen
import common

osp = os.path
np.random.seed(0)
script_dir = osp.dirname(osp.abspath(__file__))

sys.path.append(osp.join(script_dir, "build"))
import libndt

# leaf_size = 0.08  # for bunny
# leaf_size = 5.0  # for car
leaf_size = 2.0  # for room


def visualize_pc(ax, pc, label, marker='.', markersize=10):
    ax.scatter3D(pc[:, 0], pc[:, 1], pc[:, 2], label=label,
                 marker=marker, s=markersize)


def main():
    # definition
    # pc: point cloud

    # init ndt lib
    ndt = libndt.NDT()
    ndt.set_leaf_size(leaf_size)

    # prepare reference pc
    # sample_data_path = osp.join(script_dir, "data", "bunny.pcd")
    # sample_data_path = osp.join(script_dir, "data", "downsample_map.pcd")
    sample_data_path = osp.join(script_dir, "data", "room_scan1.pcd")
    reference_pc, ref_pcd = pc_gen.load_pc_from_pcd(sample_data_path)

    # prepare scan pc
    # scan_pc = convert_pc([0.4, -0.4, 0.0, 0.0, 0., 3.0], reference_pc)
    # scan_pc = scan_pc[np.mod(np.arange(scan_pc.shape[0]), 2) == 0]

    # sample_data_path = osp.join(script_dir, "data", "scan_sample.pcd")
    sample_data_path = osp.join(script_dir, "data", "room_scan1.pcd")
    # sample_data_path = osp.join(script_dir, "data", "room_scan2.pcd")
    scan_pc, scan_pcd = pc_gen.load_pc_from_pcd(sample_data_path)

    scan_transform = [0.2, 0.0, 0, 0, 0, 0] # xyz rpy deg

    scan_pc = common.convert_pc(scan_transform, reference_pc, degrees=True)
    scan_pcd = common.convert_pcd(scan_pcd, scan_transform, degrees=True)

    # create map
    t = time.time()
    ndt.create_map(reference_pc)
    print("create_map: {}".format(time.time() - t))

    # Run registration and get result transform.
    max_iteration_count = 100
    t = time.time()
    transform = ndt.registration(scan_pc, max_iteration_count)

    print(transform)

    print("registration: {}".format(time.time() - t))

    registerd_pc = common.convert_pc(transform, scan_pc)
    registered_pcd = common.convert_pcd(scan_pcd, transform)

    scan_pcd = common.set_color_pcd([255, 0, 0], scan_pcd)
    registered_pcd = common.set_color_pcd([0, 0, 255], registered_pcd)
    ref_pcd = common.set_color_pcd([0, 255, 0], ref_pcd)

    o3d.visualization.draw_geometries([ref_pcd, scan_pcd, registered_pcd])
    o3d.visualization.draw_geometries([ref_pcd, registered_pcd])


if __name__ == "__main__":
    main()
