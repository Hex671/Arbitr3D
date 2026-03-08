from typing import List, Dict
import numpy as np

class Projector2DTo3D:
    def __init__(self):
        pass

    def back_project_2d_to_3d(self, masks_across_views: List[List[np.ndarray]], 
                              index_maps: List[np.ndarray],
                              glip_detections: List[List[Dict]]) -> List[Dict]:
        """
        利用 Index Map，将 2D 掩码转化为 3D 空间中的点索引集合（局部点簇）。
        
        输入:
            masks_across_views: K 个视角，每个视角有 M 个 mask
            index_maps: K 个视角的 index map (H, W, 1)，记录每个像素对应的点云真实 index
            glip_detections: 对应的 GLIP 预测结果，包含 label 和 score
            
        输出:
            local_clusters: 包含反投影出的点云索引列表及其类别置信度的字典列表。
            [{'point_indices': [1, 5, 100, ...], 'glip_score': {'chair back': 0.85}}, ...]
        """
        local_clusters = []
        
        for view_idx, (view_masks, view_detections) in enumerate(zip(masks_across_views, glip_detections)):
            idx_map = index_maps[view_idx].squeeze(-1) # (H, W)
            
            for mask, det in zip(view_masks, view_detections):
                # 提取 mask 覆盖区域内的有效 point indices
                valid_pixels = mask & (idx_map != -1)
                point_indices = idx_map[valid_pixels].tolist()
                
                # 去重
                point_indices = list(set(point_indices))
                
                if len(point_indices) > 0:
                    cluster = {
                        'point_indices': point_indices,
                        'glip_score': {det['label']: det['score']}
                    }
                    local_clusters.append(cluster)
                    
        return local_clusters
