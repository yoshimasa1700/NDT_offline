import os
import sys
import time
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


# leaf_size = 0.08  # for bunny
# leaf_size = 5.0  # for car
leaf_size = 1.0  # for room


def visualize_pc(ax, pc, label, marker='.', markersize=5):
    ax.scatter3D(pc[:, 0], pc[:, 1], pc[:, 2], label=label,
                 marker=marker, s=markersize)


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

    dest = copy.deepcopy(pc)

    return np.apply_along_axis(
        lambda x: rot.apply(x) + trans,
        1, dest).astype(np.float32)


def visualize_elipse(ax, rx, ry, rz, cx, cy, cz):
    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v)) + cx
    y = ry * np.outer(np.sin(u), np.sin(v)) + cy
    z = rz * np.outer(np.ones_like(u), np.cos(v)) + cz

    # Plot:
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='g', alpha=0.5)


def visualize_result(ndt, reference_pc, scan_pc, registerd_pc):
    # Get intermidiate values.
    jacobian = np.array(ndt.get_jacobian_list()).reshape((-1, 6))
    hessian = np.array(ndt.get_hessian_list()).reshape((-1, 6, 6))
    jacobians_sum = np.array(ndt.get_jacobian_sum_list()).reshape((-1, 6))
    hessian_sum = np.array(ndt.get_hessian_sum_list()).reshape((-1, 6, 6))
    update = np.array(ndt.get_update_list()).reshape((-1, 6))
    score = ndt.get_score_list()
    map_data = ndt.get_map()

    # Start plot result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # def plot(data):
    #     ax.cla()

    #     idx = data % (update.shape[0] + 1)
    #     # print("idx: {} ,score: {}".format(idx, score[idx]))
    #     # print("idx: {} ,jacobian: {}".format(idx, jacobians_sum[idx]))
    #     # print("idx: {} ,hessian: {}".format(idx, hessian_sum[idx]))
    #     # print("idx: {} ,tf: {}".format(idx, update[idx]))

    #     if idx != 0:
    #         registerd_pc = convert_pc(update[idx-1], scan_pc)
    #     else:
    #         registerd_pc = scan_pc

    #     visualize_pc(ax, reference_pc, "reference")
    #     visualize_pc(ax, scan_pc, "scan")
    #     visualize_pc(ax, registerd_pc, "registered")

    #     center_points = np.array([g[0] for g in map_data])
    #     # visualize_pc(ax, center_points, "grid center", marker='^', markersize=20)

    #     covs = np.array([g[1] for g in map_data])
    #     # for cov, cp in zip(covs, center_points):
    #     #     visualize_elipse(ax, cov[0], cov[1], cov[2], cp[0], cp[1], cp[2])

    #     l = 10

    #     ax.set_xlim(0, 10)
    #     ax.set_ylim(-3, 3)
    #     ax.set_zlim(-3, 3)

    #     ax.legend()
    #     ax.set_xlabel('X axis')
    #     ax.set_ylabel('Y axis')
    #     ax.set_zlabel('Z axis')
    #     if idx != 0:
    #         ax.set_title("idx: {} ,score: {}".format(idx, score[idx-1]))
    #     else:
    #         ax.set_title("orig point cloud")

    # ani = animation.FuncAnimation(fig, plot, interval=500, frames=update.shape[0]+1)
    # ani.save("ndt_sample.gif", writer="imagemagick")

    visualize_pc(ax, reference_pc, "reference")
    # visualize_pc(ax, scan_pc, "scan")
    # visualize_pc(ax, registerd_pc, "registered")

    sample_count = len(reference_pc)
    iter_count = int(update.shape[0])
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

    # plot settings.
    ax.legend()
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ticks = np.arange(-0.4, 0.4, leaf_size)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    center_points = np.array([g[0] for g in map_data])
    # visualize_pc(ax, center_points, "grid center", marker='^', markersize=20)

    covs = np.array([g[1] for g in map_data])
    for cov, cp in zip(covs, center_points):
        visualize_elipse(ax, cov[0], cov[1], cov[2], cp[0], cp[1], cp[2])

    l = 10

    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)

    # ax.set_xlim(0, l)
    # ax.set_ylim(0, l)
    # ax.set_zlim(0, l)

    plt.show()


def main():
    # definition
    # pc: point cloud

    # init ndt lib
    ndt = libndt.NDT()
    ndt.set_leaf_size(leaf_size)

    # prepare reference pc
    # sample_data_path = osp.join(script_dir, "data", "bunny.pcd")
    # sample_data_path = osp.join(script_dir, "data", "downsample_map.pcd")
    # sample_data_path = osp.join(script_dir, "data", "room_scan1.pcd")
    # reference_pc, ref_pcd = load_pc_from_pcd(sample_data_path)
    reference_pc = create_random_pc()
    # reference_pc = create_fixed_grid_pc()

    # prepare scan pc
    scan_pc = convert_pc([0.1, 0.2, 0.2, 0., 0., 0], reference_pc)
    # sample_data_path = osp.join(script_dir, "data", "scan_sample.pcd")
    # sample_data_path = osp.join(script_dir, "data", "room_scan2.pcd")
    # scan_pc, scan_pcd = load_pc_from_pcd(sample_data_path)

    # create map
    t = time.time()
    ndt.create_map(reference_pc)
    print("create_map: {}".format(time.time() - t))

    # Run registration and get result transform.
    max_iteration_count = 100
    t = time.time()
    transform = ndt.registration(scan_pc, max_iteration_count)
    print("registration: {}".format(time.time() - t))

    registerd_pc = convert_pc(transform, scan_pc)

    # euler = np.array(transform[3: 6])
    # rot = Rotation.from_euler('zyx', euler, degrees=True).as_dcm()
    # trans = np.array(transform[0: 3])
    # registered_pcd = copy.deepcopy(scan_pcd)
    # registered_pcd.rotate(rot)
    # registered_pcd.translate(trans)

    # s_color = np.array([[255, 0, 0] for _ in range(len(scan_pcd.points))])
    # scan_pcd.colors = o3d.utility.Vector3dVector(s_color)
    # r_color = np.array([[0, 0, 255] for _ in range(len(registered_pcd.points))])
    # registered_pcd.colors = o3d.utility.Vector3dVector(r_color)
    # # o3d.visualization.draw_geometries([ref_pcd, scan_pcd, registered_pcd])
    # o3d.visualization.draw_geometries([ref_pcd, registered_pcd])

    # visualize result
    visualize_result(ndt, reference_pc, scan_pc, registerd_pc)


if __name__ == "__main__":
    main()
