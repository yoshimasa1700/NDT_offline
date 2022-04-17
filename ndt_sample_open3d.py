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
    ndt = libndt.NDT()
    ndt.set_leaf_size(setting["leaf_size"])

    # load reference pc
    reference_pc = pc_gen.load_pc_from_pcd(
        osp.join(script_dir, setting["map_data"]))

    # load scan pc
    scan_pc = pc_gen.load_pc_from_pcd(
        osp.join(script_dir, setting["scan_data"]))

    # transform scan pc by initial pose
    scan_transform = setting["init_pose"]
    scan_pc = common.convert_pc(scan_transform, scan_pc)

    # create map
    t = time.time()
    ndt.create_map(reference_pc)
    print("create_map: {}".format(time.time() - t))

    # Run registration and get result transform.
    max_iteration_count = 10
    t = time.time()
    transform = ndt.registration(scan_pc, max_iteration_count)

    print(transform)

    print("registration: {}".format(time.time() - t))

    registerd_pc = common.convert_pc(transform, scan_pc)

    scan_pcd = common.set_color_pcd(
        [255, 0, 0], pc_gen.convert_np2o3d(scan_pc))
    registered_pcd = common.set_color_pcd(
        [0, 0, 255], pc_gen.convert_np2o3d(registerd_pc))
    ref_pcd = common.set_color_pcd(
        [0, 255, 0], pc_gen.convert_np2o3d(reference_pc))

    map_data = ndt.get_map()
    center_points = np.array([g[0] for g in map_data])
    covs = np.array([g[1] for g in map_data])

    map_viz = []

    for cov, cp in zip(covs, center_points):
        std_dev = list(map(math.sqrt, cov[0: 3]))
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.vertices = o3d.utility.Vector3dVector(
            np.asarray(mesh_sphere.vertices) * np.array(std_dev) + np.array(cp))
        map_viz.append(mesh_sphere)

    map_viz.append(ref_pcd)

    map_viz.append(registered_pcd)

    not_found_list = ndt.get_not_found_list()

    found_pcd = common.set_color_pcd(
        [255, 0, 0],
        pc_gen.convert_np2o3d(np.array(ndt.get_found_list())))
    # map_viz.append(found_pcd)

    not_found_pcd = common.set_color_pcd(
        [127, 0, 255],
        pc_gen.convert_np2o3d(np.array(ndt.get_not_found_list())))
    # map_viz.append(not_found_pcd)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # for mv in map_viz:
    #     vis.add_geometry(mv)

    # count = 0

    # while True:
    #     # Get intermidiate values.
    #     jacobian = np.array(ndt.get_jacobian_list()).reshape((-1, 6))
    #     hessian = np.array(ndt.get_hessian_list()).reshape((-1, 6, 6))
    #     jacobians_sum = np.array(ndt.get_jacobian_sum_list()).reshape((-1, 6))
    #     hessian_sum = np.array(ndt.get_hessian_sum_list()).reshape((-1, 6, 6))
    #     update = np.array(ndt.get_update_list()).reshape((-1, 6))
    #     score = ndt.get_score_list()
    #     map_data = ndt.get_map()

    #     idx = count % (update.shape[0] + 1)
    #     count += 1

    #     if idx != 0:
    #         registered_pcd = common.convert_pcd(scan_pcd, update[idx-1])
    #         print("idx: {} ,score: {}".format(idx, score[idx-1]))
    #         print("idx: {} ,jacobian: {}".format(idx, jacobians_sum[idx-1]))
    #         print("idx: {} ,hessian: {}".format(idx, hessian_sum[idx-1]))
    #         print("idx: {} ,tf: {}".format(idx, update[idx-1]))
    #     else:
    #         registered_pcd = scan_pcd

    #     vis.update_geometry(registered_pcd)
    #     vis.poll_events()
    #     vis.update_renderer()

    #     time.sleep(1)

    # o3d.visualization.draw_geometries(map_viz)
    # o3d.visualization.draw_geometries_with_animation_callback(
    # map_viz, animation)

    map_viz.append(scan_pcd)

    o3d.visualization.draw_geometries(map_viz)


if __name__ == "__main__":
    main()
