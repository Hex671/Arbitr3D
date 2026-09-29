import numpy as np
import cv2
from typing import List, Dict
from datatypes import SoftMask2D

class SoftProjector:
    def __init__(self, depth_threshold: float = 0.05, erode_pixels: int = 0):
        self.depth_threshold = depth_threshold
        self.erode_pixels = erode_pixels

    def project_with_depth_check(
        self,
        soft_masks_across_views: List[List[SoftMask2D]],
        depth_maps: List[np.ndarray],
        index_maps: List[np.ndarray],
        point_cloud_xyz: np.ndarray
    ) -> List[Dict]:
        """
        基于深度的 2D-to-3D 软映射
        返回: 投影数据列表，每个元素包含 {point_indices: [], probabilities: {}}
        """
        projected_data = []

        for view_idx, (soft_masks, depth_map, index_map) in enumerate(zip(soft_masks_across_views, depth_maps, index_maps)):
            # index_map shape: (H, W, 1) or (H, W)
            if index_map.ndim == 3:
                index_map = index_map[:, :, 0]
            if depth_map.ndim == 3:
                depth_map = depth_map[:, :, 0]

            for sm in soft_masks:
                mask = sm.mask
                # 掩码边缘腐蚀：去除边界像素，只保留内部置信区域
                if self.erode_pixels > 0:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (2 * self.erode_pixels + 1, 2 * self.erode_pixels + 1))
                    mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                # Find 3D point indices in this mask
                pixel_indices = np.where(mask)
                point_indices_in_mask = index_map[pixel_indices]

                # Filter out background (-1)
                valid_mask = point_indices_in_mask != -1
                valid_point_indices = point_indices_in_mask[valid_mask]
                valid_pixel_y = pixel_indices[0][valid_mask]
                valid_pixel_x = pixel_indices[1][valid_mask]

                if len(valid_point_indices) == 0:
                    continue

                # --- 可选：基于 3D 空间连通性的聚类，去掉跨物体错误投影 ---
                mask_points_3d = point_cloud_xyz[valid_point_indices]
                n_pts = len(mask_points_3d)
                if n_pts >= 3:
                    span = mask_points_3d.max(axis=0) - mask_points_3d.min(axis=0)
                    diag = float(np.linalg.norm(span)) + 1e-9
                    # eps 随点云尺度自适应；min_samples 随点数缩放，避免小掩码被整段判成噪声后丢弃
                    eps = max(1e-5, 0.025 * diag)
                    min_samples = int(min(50, max(3, n_pts // 25)))
                    from sklearn.cluster import DBSCAN

                    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(mask_points_3d)
                    labels = clustering.labels_
                    unique_labels = set(labels)
                    valid_clusters = [l for l in unique_labels if l != -1]

                    if len(valid_clusters) > 1:
                        valid_mask_cluster = labels != -1
                        valid_point_indices = valid_point_indices[valid_mask_cluster]
                    elif len(valid_clusters) == 0:
                        # 尺度/参数不匹配时不再整掩码丢弃，保留全部有效投影点参与投票
                        pass
                # -----------------------------------------------------------------

                projected_data.append({
                    'point_indices': valid_point_indices.tolist(),
                    'probabilities': sm.probabilities
                })

        return projected_data
