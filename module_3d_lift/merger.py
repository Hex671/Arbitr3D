from typing import List, Set
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datatypes import Entity3D

class MacroMerger:
    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold

    def calculate_3d_iou(self, cluster_a_indices: Set[int], cluster_b_indices: Set[int]) -> float:
        intersection = len(cluster_a_indices.intersection(cluster_b_indices))
        if intersection == 0:
            return 0.0
        union = len(cluster_a_indices.union(cluster_b_indices))
        return intersection / float(union)

    def macro_merge(self, local_clusters: List[dict]) -> List[Entity3D]:
        """
        宏观合并。按包含的点数降序排列，计算 3D IoU。
        若 IoU > 阈值，则认定它们是同一个 3D Entity，并合并它们携带的 GLIP 分数。
        """
        # 按点数降序排列
        local_clusters_sorted = sorted(local_clusters, key=lambda x: len(x['point_indices']), reverse=True)
        
        entities = []
        entity_counter = 1
        
        for cluster in local_clusters_sorted:
            cluster_indices = set(cluster['point_indices'])
            merged = False
            
            # 尝试与现有 entity 合并
            for entity in entities:
                entity_indices = set(entity.point_indices)
                iou = self.calculate_3d_iou(cluster_indices, entity_indices)
                
                if iou > self.iou_threshold:
                    # 执行合并：合并点索引，叠加 GLIP scores
                    entity_indices.update(cluster_indices)
                    entity.point_indices = list(entity_indices)
                    entity.update_glip_scores(cluster['glip_score'])
                    merged = True
                    break
            
            if not merged:
                # 创建新 Entity
                new_entity = Entity3D(entity_id=entity_counter)
                new_entity.point_indices = list(cluster_indices)
                new_entity.update_glip_scores(cluster['glip_score'])
                entities.append(new_entity)
                entity_counter += 1
                
        return entities
