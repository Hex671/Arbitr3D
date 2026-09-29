import numpy as np
import open3d as o3d
from typing import Tuple

def compute_min_distance(pts_A: np.ndarray, pts_B: np.ndarray) -> float:
    """计算两个点云之间的最短距离"""
    if len(pts_A) == 0 or len(pts_B) == 0:
        return float('inf')

    # 随机降采样加速计算
    if len(pts_A) > 1000:
        pts_A = pts_A[np.random.choice(len(pts_A), 1000, replace=False)]
    if len(pts_B) > 1000:
        pts_B = pts_B[np.random.choice(len(pts_B), 1000, replace=False)]

    # 使用 KDTree 计算最近距离
    pcd_B = o3d.geometry.PointCloud()
    pcd_B.points = o3d.utility.Vector3dVector(pts_B)
    kdtree = o3d.geometry.KDTreeFlann(pcd_B)

    min_dist = float('inf')
    for pt in pts_A:
        [_, idx, dist_sq] = kdtree.search_knn_vector_3d(pt, 1)
        if dist_sq[0] < min_dist:
            min_dist = dist_sq[0]

    return float(np.sqrt(min_dist))

def compute_relative_direction(pts_A: np.ndarray, pts_B: np.ndarray) -> str:
    """
    计算 A 相对于 B 的方位 (A is [direction] of B)
    假设坐标系：Z 向上 (Up), Y 向前 (Front), X 向右 (Right)
    (基于常规物体坐标系，具体可根据轴调整)
    """
    if len(pts_A) == 0 or len(pts_B) == 0:
        return "unknown"

    center_A = np.mean(pts_A, axis=0)
    center_B = np.mean(pts_B, axis=0)

    diff = center_A - center_B

    # 找到主导方向
    abs_diff = np.abs(diff)
    dom_axis = np.argmax(abs_diff)

    if dom_axis == 2: # Z 轴
        if diff[2] > 0:
            return "above"
        else:
            return "below"
    elif dom_axis == 1: # Y 轴
        if diff[1] > 0:
            return "in front of"
        else:
            return "behind"
    else: # X 轴
        if diff[0] > 0:
            return "to the right of"
        else:
            return "to the left of"

def map_distance_to_text(dist: float) -> str:
    """将物理距离映射为文本"""
    if dist < 0.05:
        return "非常接近/接触 (Touching/Very close)"
    elif dist < 0.15:
        return "较近 (Close)"
    elif dist < 0.40:
        return "中等距离 (Medium distance)"
    elif dist < 0.80:
        return "较远 (Far)"
    else:
        return "非常远 (Very far)"
