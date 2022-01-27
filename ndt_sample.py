import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
osp = os.path
np.random.seed(0)
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "build"))
import libndt

def visualize_pc(ax, pc, label):
    ax.scatter3D(pc[:, 0], pc[:, 1], pc[:, 2], label=label)

mu = 0
sigma = 1

# create reference point cloud
sample_count = 30
reference_pc = np.random.normal(mu, sigma, (sample_count, 3)).astype(np.float32)
# reference_pc = np.array([[1, 1, 1]])

value = 0.1

# refs = []

# for x in np.linspace(-value, value, 2):
#     for y in np.linspace(-value, value, 2):
#         for z in np.linspace(-value, value, 2):
#             refs.append([x, y, z])

# reference_pc = np.array(refs, dtype=np.float32)

# reference_pc = np.array([[-0.1, 0.1, -0.1]], dtype=np.float32)

# print(reference_pc)


# reference_pc[:, 2] = np.zeros(sample_count)

# reference_pc = np.array([
#     [-2/3,-2/3,0],
#     [1/3,-2/3,0],
#     [1/3,4/3,0]
# ], dtype=np.float32)

# create map
ndt = libndt.NDT()
ndt.create_map(reference_pc)

# create scan point cloud
euler = np.array([ 0.0, 0, 0.0])
rot = Rotation.from_euler('zyx', euler, degrees=True)
trans = np.array([ 1.0, 0.0, 0.0])
scan_pc = np.apply_along_axis(lambda x: rot.apply(x) + trans, 1, reference_pc).astype(np.float32)

print(scan_pc.dtype)

# print(scan_pc)

# scan_pc = np.array([[ 0.9, 8, 5],
#                     [ 0.8, 8, 5],
#                     [ 0.7, 8, 5],
#                     [ 0.6, 8, 5]
# ], dtype=np.float32)
# reference_pc = np.array([[ 1, 8, 5]], dtype=np.float32)

print(scan_pc)

# registration and get result transform.

transform = ndt.registration(scan_pc, 1)

# print(transform)

jacobians = ndt.get_jacobian_list()
jacobians = np.array(jacobians).reshape((-1, 6))


print(jacobians)

jacobians_sum = np.array(ndt.get_jacobian_list2()).reshape((-1, 6))

update = np.array(ndt.get_update_list()).reshape((-1, 6))

# print(jacobians[:, 0:3])

jacobian_part = jacobians[:, 0:3]


euler_ans = np.array(transform[3: 6])
rot_ans = Rotation.from_euler('zyx', euler, degrees=True)
trans_ans = np.array(transform[0: 3])

transform_affine = np.eye(4)
ans_pc = np.apply_along_axis(
    lambda x: rot_ans.apply(x) + trans_ans,
    1, scan_pc)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


visualize_pc(ax, reference_pc, "reference")
visualize_pc(ax, scan_pc, "scan")
visualize_pc(ax, ans_pc, "registered")


iter_count = int(jacobian_part.shape[0] / sample_count)


for ic in range(iter_count):
    ax.quiver(scan_pc[:, 0], scan_pc[:, 1], scan_pc[:, 2],
              jacobian_part[ic*sample_count:(ic+1)*sample_count, 0],
              jacobian_part[ic*sample_count:(ic+1)*sample_count, 1],
              jacobian_part[ic*sample_count:(ic+1)*sample_count, 2], color="orange"
    )


ax.quiver(0, 0, 0,
          jacobians_sum[0, 0] / sample_count,
          jacobians_sum[0, 1] / sample_count,
          jacobians_sum[0, 2] / sample_count,
          color="red")

ax.quiver(0, 0, 0,
          update[0, 0],
          update[0, 1],
          update[0, 2],
          color="blue")


ax.legend()

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)

plt.show()
