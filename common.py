import copy
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d


# Transform scan_pc by calculated transform
def convert_pc(transform, pc, degrees=False):
    euler = np.array(transform[3: 6])
    rot = Rotation.from_euler('xyz', euler, degrees=degrees)
    trans = np.array(transform[0: 3])

    dest = copy.deepcopy(pc)

    return np.apply_along_axis(
        lambda x: rot.apply(x) + trans,
        1, dest).astype(np.float32)


def convert_pcd(pcd, transform, degrees=False):
    out = copy.deepcopy(pcd)

    out = out.rotate(Rotation.from_euler('xyz', transform[3: 6],
                                         degrees=degrees).as_dcm())
    out = out.translate(np.array(transform[0: 3]))

    return out


def set_color_pcd(color, pcd):
    color_arr = np.array([color for _ in range(len(pcd.points))])
    pcd.colors = o3d.utility.Vector3dVector(color_arr)

    return pcd
