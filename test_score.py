import os
import sys
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import open3d as o3d
osp = os.path
np.random.seed(0)
script_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(script_dir, "build"))
import libndt

leaf_size = 0.05  # for bunny
# leaf_size = 2.0  # for car


def visualize_pc(ax, pc, label, marker='.', markersize=1):
    ax.scatter3D(pc[:, 0], pc[:, 1], pc[:, 2], label=label,
                 marker=marker, s=markersize)


# random sample point
def create_random_pc():
    sample_count = 50
    mu = 0
    sigma = 1
    return np.random.normal(mu, sigma, (sample_count, 3)).astype(np.float32)


def load_pc_from_pcd(path, voxel_filter=False):
    pcd = o3d.io.read_point_cloud(path)
    # pcd = pcd.voxel_down_sample(voxel_size=2.0)
    return np.asarray(pcd.points).astype(np.float32), pcd


# fixed point
def create_fixed_pc():
    # return np.array([[1, 0, 0]])
    return np.array([[x, 0, 0] for x in list(
        np.arange(-7.5, 7.5, 0.1))]).astype(np.float32)


# grid point
def create_fixed_grid_pc():
    value = 0.1
    refs = []
    for x in np.linspace(-value, value, 2):
        for y in np.linspace(-value, value, 2):
            for z in np.linspace(-value, value, 2):
                refs.append([x, y, z])
    return np.array(refs, dtype=np.float32)


# triangle reference pointcloud.
def create_triangle_pc():
    return np.array([
        [-2/3, -2/3, 0],
        [1/3, -2/3, 0],
        [1/3, 4/3, 0]
    ], dtype=np.float32)


# Transform scan_pc by calculated transform
def convert_pc(transform, pc):
    euler = np.array(transform[3: 6])
    rot = Rotation.from_euler('zyx', euler, degrees=True)
    trans = np.array(transform[0: 3])
    return np.apply_along_axis(
        lambda x: rot.apply(x) + trans,
        1, pc).astype(np.float32)


def visualize_result(ndt, reference_pc, scan_pc, registerd_pc):
    # Get intermidiate values.
    # jacobian = np.array(ndt.get_jacobian_list()).reshape((-1, 6))
    # hessian = np.array(ndt.get_hessian_list()).reshape((-1, 6, 6))
    # jacobians_sum = np.array(ndt.get_jacobian_sum_list()).reshape((-1, 6))
    # update = np.array(ndt.get_update_list()).reshape((-1, 6))

    # Start plot result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    visualize_pc(ax, reference_pc, "reference")
    visualize_pc(ax, scan_pc, "scan")
    visualize_pc(ax, registerd_pc, "registered")

    sample_count = len(reference_pc)
    # iter_count = int(jacobian.shape[0] / sample_count)
    # print("iteration count: {}".format(iter_count))

    # for ic in range(iter_count):
    #     # plot jacobian
    #     ax.quiver(scan_pc[:, 0], scan_pc[:, 1], scan_pc[:, 2],
    #               jacobian[ic*sample_count:(ic+1)*sample_count, 0],
    #               jacobian[ic*sample_count:(ic+1)*sample_count, 1],
    #               jacobian[ic*sample_count:(ic+1)*sample_count, 2], color="orange"
    #     )

    #     # plot hessian
    #     ax.quiver(scan_pc[:, 0], scan_pc[:, 1], scan_pc[:, 2],
    #               hessian[ic*sample_count:(ic+1)*sample_count, 0, 0],
    #               hessian[ic*sample_count:(ic+1)*sample_count, 1, 1],
    #               hessian[ic*sample_count:(ic+1)*sample_count, 2, 2], color="yellow"
    #     )

    # plot summed jacobian
    # ax.quiver(0, 0, 0,
    #           jacobians_sum[0, 0] / sample_count,
    #           jacobians_sum[0, 1] / sample_count,
    #           jacobians_sum[0, 2] / sample_count,
    #           color="red")

    # plot update
    # ax.quiver(0, 0, 0,
    #           update[0, 0],
    #           update[0, 1],
    #           update[0, 2],
    #           color="blue")

    map_data = ndt.get_map()

    center_points = np.array([g[0] for g in map_data])
    visualize_pc(ax, center_points, "grid center", marker='^', markersize=20)

    # plot settings.
    ax.legend()
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ticks = np.arange(-0.4, 0.4, leaf_size)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    l = 0.2

    ax.set_xlim(-l/2, l/2)
    ax.set_ylim(0, l)
    ax.set_zlim(-l, l)

    plt.show()


def main():
    # definition
    # pc: point cloud

    # init ndt lib
    ndt = libndt.NDT()
    ndt.set_leaf_size(leaf_size)

    trans = np.array([[1.0, 0, 0, 0, 0, 0]], dtype=np.float32)

    x_list = np.arange(-7.5, 7.5, 0.1)

    res = [ndt.calc_score(np.array([[x, 0, 0]], dtype=np.float32), trans)
           for x in x_list]

    scores = [score for score, _, _ in res]
    jacobians = [j[0] for _, j, _ in res]
    hessians = [h[0] for _, _, h in res]

    plt.plot(x_list, scores, label="score")
    plt.plot(x_list, jacobians, label="jacobian")
    plt.plot(x_list, hessians, label="hessian")
    plt.xlabel('X axis')
    plt.ylabel('Score')
    plt.grid()
    plt.legend()
    plt.title('input and score')

    plt.show()


if __name__ == "__main__":
    main()
