import numpy as np
import math
from typing import Sequence, Tuple


def _generate_from_views(views: Sequence[Sequence[float]], radius: float, has_pytorch3d: bool) -> Tuple[np.ndarray, np.ndarray]:
    if has_pytorch3d:
        from pytorch3d.renderer import look_at_view_transform

        R_list = []
        T_list = []
        for view in views:
            R, T = look_at_view_transform(dist=radius, elev=view[0], azim=view[1])
            R_list.append(R[0].numpy())
            T_list.append(T[0].numpy())

        camera_positions = []
        camera_rotations = []
        for r, t in zip(R_list, T_list):
            r_mat = r.T
            t_vec = -r @ t
            camera_positions.append(t_vec)
            camera_rotations.append(r_mat)

        return np.array(camera_positions), np.array(camera_rotations)

    camera_positions = []
    R_matrices = []
    for view in views:
        elev = math.radians(view[0])
        azim = math.radians(view[1])

        y = math.sin(elev)
        radius_at_y = math.cos(elev)
        x = math.sin(azim) * radius_at_y
        z = math.cos(azim) * radius_at_y

        pos = np.array([x * radius, y * radius, z * radius])
        camera_positions.append(pos)

        forward = -pos
        forward = forward / np.linalg.norm(forward)

        up = np.array([0, 1, 0])
        if np.abs(np.dot(forward, up)) > 0.999:
            up = np.array([1, 0, 0])

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        real_up = np.cross(right, forward)

        R = np.stack([right, real_up, -forward], axis=-1)
        R_matrices.append(R)

    return np.array(camera_positions), np.array(R_matrices)


def generate_cameras_from_views(views: Sequence[Sequence[float]], radius: float = 2.2) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from pytorch3d.renderer import look_at_view_transform
        import torch
        has_pytorch3d = True
    except ImportError:
        has_pytorch3d = False

    return _generate_from_views(views, radius, has_pytorch3d)

def generate_sphere_cameras(K: int = 20, radius: float = 2.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成均匀分布的 K 个摄像机位置和对应的旋转矩阵。
    修复了 PyTorch3D 渲染时坐标系不匹配导致物体偏离视野或消失的问题。
    """
    # 检查是否安装了 PyTorch3D
    try:
        from pytorch3d.renderer import look_at_view_transform
        import torch
        has_pytorch3d = True
    except ImportError:
        has_pytorch3d = False

    if K == 10:
        # 优化后的 20 视角仰角 (避免绝对垂直俯视)
        # views = [
        #     [15, 45], [15, 135], [15, 225], [15, 315],
        #     [45, 0], [45, 120], [45, 240],
        #     [0,0],
        #     [-40, 0], [-40, 90], [-40, 180], [-40, 270],
        #     [-50, 45], [-50,135], [-50, 225], [-50, 315]
        # ]
        views = [
            [10, 60], [10, 300],
            [40, 0], [40, 120], [40, 240],
            [-20, 45], [-20, 180], [-20, 315],
            [-40, 135], [-40, 225]]
    else:
        # 使用 Fibonacci sphere 生成仰角 (elev) 和方位角 (azim)
        views = []
        phi = math.pi * (3. - math.sqrt(5.))

        # 限制 y 的范围，避开极点 (垂直俯视/仰视)
        y_max = 0.6
        y_min = -0.4

        for i in range(K):
            y = y_max - (i / float(K - 1)) * (y_max - y_min)
            # 通过 y 坐标反推仰角 (Elevation)
            elev = math.degrees(math.asin(y))

            theta = phi * i
            # theta 即为方位角 (Azimuth)
            azim = math.degrees(theta)

            views.append([elev, azim])

    return _generate_from_views(views, radius, has_pytorch3d)
