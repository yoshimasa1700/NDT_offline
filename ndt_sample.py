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

def visualize_pc(ax, pc, label):
    ax.scatter3D(pc[:, 0], pc[:, 1], pc[:, 2], label=label, marker='.')

mu = 0
sigma = 1

# create reference point cloud

## random sample point
# sample_count = 50
# reference_pc = np.random.normal(mu, sigma, (sample_count, 3)).astype(np.float32)

pcd = o3d.io.read_point_cloud(osp.join(script_dir, "data", "bunny.pcd"))
# pcd = o3d.io.read_point_cloud(osp.join(osp.dirname(osp.abspath(__file__)), "data", "downsample_map.pcd"))
reference_pc = np.asarray(pcd.points).astype(np.float32)

## fixed point
# reference_pc = np.array([[1, 0, 0]])
# reference_pc = np.array([[x, 0, 0] for x in list(np.arange(-7.5, 7.5, 0.1))]).astype(np.float32)


## grid point
# value = 0.1
# refs = []
# for x in np.linspace(-value, value, 2):
#     for y in np.linspace(-value, value, 2):
#         for z in np.linspace(-value, value, 2):
#             refs.append([x, y, z])
# reference_pc = np.array(refs, dtype=np.float32)
sample_count = reference_pc.shape[0]


## triangle reference pointcloud.
# reference_pc = np.array([
#     [-2/3,-2/3,0],
#     [1/3,-2/3,0],
#     [1/3,4/3,0]
# ], dtype=np.float32)

print(reference_pc.shape)

# create map
ndt = libndt.NDT()

leaf_size = 0.05  # for bunny
ndt.set_leaf_size(leaf_size)
# ndt.set_leaf_size(2.0) # for car

t = time.time()
ndt.create_map(reference_pc)
print("create_map: {}".format(time.time() - t))

# create scan point cloud
euler = np.array([ 0.0, 0.0, 0.1])
rot = Rotation.from_euler('zyx', euler, degrees=True)
trans = np.array([ 0.006, 0.006, 0.0])
scan_pc = np.apply_along_axis(lambda x: rot.apply(x) + trans, 1, reference_pc).astype(np.float32)

# pcd2 = o3d.io.read_point_cloud(osp.join(osp.dirname(osp.abspath(__file__)), "data", "scan_sample.pcd"))

# pcd2 = pcd2.voxel_down_sample(voxel_size=0.1)

# print(len(pcd2.points))

# scan_pc = np.asarray(pcd2.points).astype(np.float32)

# Run registration and get result transform.
max_iteration_count = 100
t = time.time()
transform = ndt.registration(scan_pc, max_iteration_count)
print("registration: {}".format(time.time() - t))


# reference_pc = np.asarray(pcd.voxel_down_sample(voxel_size=10.0).points).astype(np.float32)
# scan_pc = np.asarray(pcd2.voxel_down_sample(voxel_size=10.0).points).astype(np.float32)


# Transform scan_pc by calculated transform
euler_ans = np.array(transform[3: 6])
rot_ans = Rotation.from_euler('zyx', euler_ans, degrees=True)
trans_ans = np.array(transform[0: 3])
transform_affine = np.eye(4)
ans_pc = np.apply_along_axis(
    lambda x: rot_ans.apply(x) + trans_ans,
    1, scan_pc)

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
visualize_pc(ax, ans_pc, "registered")

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

# plot settings.
ax.legend()
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

ticks=np.arange(-0.4, 0.4, leaf_size)

ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

l = 0.2

ax.set_xlim(-l/2, l/2)
ax.set_ylim(0, l)
ax.set_zlim(-l, l)

plt.show()
