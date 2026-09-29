import numpy as np
import open3d as o3d
from datatypes import PointProbabilityField

class GraphOptimizer:
    def __init__(self, k_neighbors: int = 30, spatial_weight: float = 1.0,
                 normal_dot_threshold: float = 0.0):
        self.k_neighbors = k_neighbors
        self.spatial_weight = spatial_weight
        self.normal_dot_threshold = normal_dot_threshold  # 法向量点积阈值，≥ 此值视为同表面

    def _orient_normals_outward(self, pcd):
        """
        基于质心的法向量外朝向定向。
        将每个点的法向量翻转为指向远离物体质心的方向，
        使薄板正反两面的法向量天然相反。

        相比 orient_normals_consistent_tangent_plane，此方法不通过近邻图传播，
        不会因薄结构两侧互为近邻而把对面法向量翻转到同一方向。
        """
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        centroid = np.mean(points, axis=0)
        outward_dirs = points - centroid  # 每个点到质心的外指向量

        # 法向量与外指向量的点积 < 0 说明法向量指向质心内部，需要翻转
        dots = np.sum(normals * outward_dirs, axis=1)
        flip_mask = dots < 0
        normals[flip_mask] = -normals[flip_mask]

        pcd.normals = o3d.utility.Vector3dVector(normals)
        print(f"    [GraphOptimizer] 法向量外朝向定向完成，翻转了 {np.sum(flip_mask)}/{len(flip_mask)} 个点")
        return np.asarray(pcd.normals)

    def apply_graph_cuts(self, prob_field: PointProbabilityField) -> np.ndarray:
        """
        同表面感知的 KNN 概率平滑。
        利用质心外朝向法向量，通过点积判断邻居是否在同一表面，
        仅对同表面邻居进行概率平均，避免极薄物体两侧的语义互相污染。

        薄板正面法向量 ≈ (0,0,+1)，反面 ≈ (0,0,-1)
        → 同面点积 ≈ +1，对面点积 ≈ -1，阈值 0.0 即可分开。
        """
        pcd = prob_field.point_cloud
        points = np.asarray(pcd.points)
        probs = prob_field.prob_matrix.copy()
        num_points = points.shape[0]

        # 确保点云有法向量
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        # 用质心外朝向替代 orient_normals_consistent_tangent_plane
        normals = self._orient_normals_outward(pcd)

        # 构建 KDTree
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        smoothed_probs = np.zeros_like(probs)

        for i in range(num_points):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], self.k_neighbors)

            if k > 0:
                idx = np.asarray(idx)
                # 法向量点积过滤：同表面点积 ≈ +1，对面点积 ≈ -1
                dots = normals[idx] @ normals[i]
                same_surface = dots >= self.normal_dot_threshold
                valid_idx = idx[same_surface]

                if len(valid_idx) > 0:
                    smoothed_probs[i] = np.mean(probs[valid_idx], axis=0)
                else:
                    smoothed_probs[i] = probs[i]
            else:
                smoothed_probs[i] = probs[i]

        # Blend original and smoothed
        alpha = 0.7
        final_probs = alpha * probs + (1 - alpha) * smoothed_probs

        return final_probs

    def extract_final_labels(self, smoothed_prob_matrix: np.ndarray) -> np.ndarray:
        """
        对平滑后的概率场执行 Argmax，得到每个点的最终硬标签
        """
        return np.argmax(smoothed_prob_matrix, axis=1)
