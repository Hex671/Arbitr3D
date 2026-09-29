import numpy as np
import open3d as o3d
import math
from typing import Dict, Any

class GeometricFeatureExtractor:
    def __init__(self, full_pc_xyz: np.ndarray):
        """
        初始化特征提取器
        :param full_pc_xyz: (N, 3) 完整的归一化3D点云坐标，用于计算全局属性
        """
        self.full_pc_xyz = full_pc_xyz
        self.total_points = len(full_pc_xyz)

        # 计算全局边界框，用于归一化
        self.global_min_bounds = np.min(full_pc_xyz, axis=0)
        self.global_max_bounds = np.max(full_pc_xyz, axis=0)
        self.global_extents = self.global_max_bounds - self.global_min_bounds

        # 近似全局体积 (AABB)
        self.global_volume = np.prod(self.global_extents)

        # 预计算整个点云的法线，用于特征提取时的参照 (可选，但为了加速，针对子掩码动态计算更好)

    def compute_features(self, mask_pts_3d: np.ndarray) -> Dict[str, Any]:
        """
        计算给定 3D 点集的 4 项几何特征
        :param mask_pts_3d: (M, 3) 属于该掩码的 3D 点云坐标
        """
        if len(mask_pts_3d) < 4:
            return self._empty_features()

        # 1. 主曲率 (表面弯曲度)
        # 通过 PCA 计算：最小特征值占总特征值的比例
        pca = o3d.geometry.PointCloud()
        pca.points = o3d.utility.Vector3dVector(mask_pts_3d)
        mean, covariance = pca.compute_mean_and_covariance()
        eigenvalues, _ = np.linalg.eigh(covariance)
        eigenvalues = np.sort(eigenvalues)
        principal_curvature = float(eigenvalues[0] / (np.sum(eigenvalues) + 1e-8))

        # 2. 全局归一化高度坐标 (Normalized Z / Height)
        # 本项目坐标系: Y 轴为"上"方向 (index=1)，与相机系统、概率融合一致
        up_axis = 1
        mean_height = np.mean(mask_pts_3d[:, up_axis])
        normalized_z = float((mean_height - self.global_min_bounds[up_axis]) / (self.global_extents[up_axis] + 1e-8))

        # 3. 点数占比 (Point Ratio)
        # 该掩码映射到的 3D 点数占总点数的比例，反映部件的相对大小
        point_ratio = float(len(mask_pts_3d) / self.total_points)

        # 4. 法向量朝向 (Normal Vector)
        # 使用 Open3D 估计法线
        pca.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=15))
        normals = np.asarray(pca.normals)
        # 取平均法线
        mean_normal = np.mean(normals, axis=0)
        mean_normal_norm = np.linalg.norm(mean_normal) + 1e-8
        mean_normal = mean_normal / mean_normal_norm

        # 与竖直轴 (Y轴, index=1) 的夹角 [0, 90]，取绝对值消除法线正反向的影响
        nv_angle_with_up_rad = np.arccos(np.clip(np.abs(mean_normal[up_axis]), 0.0, 1.0))
        nv_angle_z = float(np.degrees(nv_angle_with_up_rad))
        nv_angle_xy = 90.0 - nv_angle_z

        # 5. 空间质心和相对包围盒尺寸 (Spatial Centroid and Relative Bounding Box Dimensions)
        # 计算掩码局部的 AABB 包围盒及其中心
        mask_min_bounds = np.min(mask_pts_3d, axis=0)
        mask_max_bounds = np.max(mask_pts_3d, axis=0)
        mask_extents = mask_max_bounds - mask_min_bounds
        mask_centroid = (mask_min_bounds + mask_max_bounds) / 2.0

        # 归一化质心坐标 (0~1)，相对于全局包围盒
        normalized_centroid = (mask_centroid - self.global_min_bounds) / (self.global_extents + 1e-8)

        # 相对尺寸比例 (掩码尺寸 / 全局尺寸)
        relative_dimensions = mask_extents / (self.global_extents + 1e-8)

        # 明确坐标系约定：X=0(宽), Y=1(高/竖直轴), Z=2(深)
        return {
            "principal_curvature": round(principal_curvature, 4),
            "normalized_z": round(normalized_z, 4), # 保留旧键名以兼容旧代码，实际是高度 (Y轴)
            "point_ratio": round(point_ratio, 4),
            "normal_vector": {
                "angle_with_z": round(nv_angle_z, 2),
                "angle_with_xy": round(nv_angle_xy, 2)
            },
            "spatial_centroid": {
                "x_width": round(float(normalized_centroid[0]), 3),
                "y_height": round(float(normalized_centroid[1]), 3),
                "z_depth": round(float(normalized_centroid[2]), 3)
            },
            "relative_dimensions": {
                "x_width_ratio": round(float(relative_dimensions[0]), 3),
                "y_height_ratio": round(float(relative_dimensions[1]), 3),
                "z_depth_ratio": round(float(relative_dimensions[2]), 3)
            }
        }

    def _empty_features(self) -> Dict[str, Any]:
        return {
            "principal_curvature": 0.0,
            "normalized_z": 0.0,
            "point_ratio": 0.0,
            "normal_vector": {
                "angle_with_z": 0.0,
                "angle_with_xy": 0.0
            },
            "spatial_centroid": {
                "x_width": 0.0,
                "y_height": 0.0,
                "z_depth": 0.0
            },
            "relative_dimensions": {
                "x_width_ratio": 0.0,
                "y_height_ratio": 0.0,
                "z_depth_ratio": 0.0
            }
        }
