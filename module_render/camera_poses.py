import numpy as np
import math
from typing import Tuple

def generate_sphere_cameras(K: int = 20, radius: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于斐波那契网格生成均匀分布的 K 个摄像机位置（外参平移向量）
    和对应的旋转矩阵（使相机看向原点）。
    返回: 相机位置(K, 3), 旋转矩阵(K, 3, 3)
    """
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(K):
        y = 1 - (i / float(K - 1)) * 2  # y goes from 1 to -1
        radius_at_y = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y

        point = np.array([x * radius, y * radius, z * radius])
        points.append(point)

    camera_positions = np.array(points)
    
    # 构造 LookAt 矩阵 (假设看向原点 [0,0,0], up 向量为 [0,1,0])
    R_matrices = []
    for pos in camera_positions:
        forward = -pos
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 1, 0])
        # 处理相机正好在正上方或正下方的情况
        if np.abs(np.dot(forward, up)) > 0.999:
            up = np.array([1, 0, 0])
            
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        real_up = np.cross(right, forward)
        
        # 旋转矩阵 R 的列分别是 right, real_up, -forward 
        R = np.stack([right, real_up, -forward], axis=-1)
        R_matrices.append(R)
        
    return camera_positions, np.array(R_matrices)
