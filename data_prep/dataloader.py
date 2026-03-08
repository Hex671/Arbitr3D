import numpy as np
import open3d as o3d
import os

class PointCloudDataset:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_point_cloud(self, file_path: str) -> o3d.geometry.PointCloud:
        """
        加载 .ply 或 .obj 格式的点云
        """
        full_path = os.path.join(self.data_dir, file_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Point cloud file not found: {full_path}")
        pcd = o3d.io.read_point_cloud(full_path)
        return pcd

    def normalize_pc(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        点云归一化：将对象平移至坐标原点，并缩放至单位球内。
        这确保了后续渲染时固定机位的一致性。
        """
        points = np.asarray(pcd.points)
        
        # 平移至原点
        centroid = np.mean(points, axis=0)
        points -= centroid
        
        # 缩放至单位球
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points /= max_dist
            
        normalized_pcd = o3d.geometry.PointCloud()
        normalized_pcd.points = o3d.utility.Vector3dVector(points)
        
        # 继承原点云的其他属性
        if pcd.has_colors():
            normalized_pcd.colors = pcd.colors
        if pcd.has_normals():
            normalized_pcd.normals = pcd.normals
            
        return normalized_pcd

    def extract_color(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        提取点云的 RGB 颜色特征
        如果点云没有颜色，则生成默认颜色
        """
        if pcd.has_colors():
            return np.asarray(pcd.colors)
        else:
            # 返回默认的浅灰色
            num_points = np.asarray(pcd.points).shape[0]
            return np.ones((num_points, 3)) * 0.7
