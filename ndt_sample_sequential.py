import os
import sys
import time
import math
import copy
import yaml
import numpy as np
from pprint import pprint
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
import time


def visualize_pc(ax, pc, label, marker='.', markersize=10):
    ax.scatter3D(pc[:, 0], pc[:, 1], pc[:, 2], label=label,
                 marker=marker, s=markersize)


def main():
    # definition
    # pc: point cloud

    setting_file_path = sys.argv[1]

    if not osp.exists(setting_file_path):
        raise OSError(-1, "{} not found".format(setting_file_path))
    with open(setting_file_path) as f:
        setting = yaml.safe_load(f)

    pprint(setting)

    # init ndt lib

    from glob import glob

    files = sorted(glob("../data/scans/*.pcd"))

    prev = files[0]

    pose = [0, 0, 0,
            0, 0, 0]

    pose_log = []
    pose_log.append(pose)

    counter = 0

    # load reference pc
    reference_pc = pc_gen.load_pc_from_pcd(
        osp.join(script_dir, "./data/downsample_map.pcd"))

    with open("result.log", 'w') as f:
        f.write("x,y,z,roll,pitch,yaw\n")
        for p in pose_log:
            f.write("{},{},{},{},{},{}\n".format(*p))

    for f in files[1:]:
        ndt = libndt.NDT()
        ndt.set_leaf_size(setting["leaf_size"])

        print("map: {}".format(prev))
        print("scan: {}".format(f))

        # load scan pc
        scan_pc = pc_gen.load_pc_from_pcd(
            osp.join(script_dir, f), True)

        # transform scan pc by initial pose
        scan_transform = pose
        # scan_pc = common.convert_pc(scan_transform, scan_pc)

        # create map
        t = time.time()
        ndt.create_map(reference_pc)
        print("create_map: {}".format(time.time() - t))

        # Run registration and get result transform.
        max_iteration_count = 2
        t = time.time()

        print(np.array([pose]))

        transform = ndt.registration(scan_pc, max_iteration_count, np.array(
            [pose], dtype=np.float32))
        pose = transform
        pose_log.append(pose)

        prev = f
        # reference_pc = scan_pc

        with open("result.log", 'a') as f:
            f.write("{},{},{},{},{},{}\n".format(*pose))

        print(transform)

        print("registration: {}".format(time.time() - t))
        counter += 1
        # if counter > 2:
        # break

if __name__ == "__main__":
    main()
