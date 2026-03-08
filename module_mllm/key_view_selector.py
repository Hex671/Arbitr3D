from typing import List
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datatypes import Entity3D

class KeyViewSelector:
    def __init__(self, target_views: int = 4):
        self.target_views = target_views

    def greedy_view_selection(self, entities: List[Entity3D], index_maps: List[np.ndarray]) -> List[int]:
        """
        使用贪心算法选出 4~6 个能完整、无遮挡地展现 M 个实体的关键机位索引。
        """
        K = len(index_maps)
        entity_ids = [e.entity_id for e in entities]
        
        # 统计每个视角中，各个 entity 的可见像素面积
        view_entity_visibility = []
        for view_idx in range(K):
            idx_map = index_maps[view_idx].squeeze(-1)
            visibility = {eid: 0 for eid in entity_ids}
            
            pt_to_eid = {}
            for e in entities:
                for pt in e.point_indices:
                    pt_to_eid[pt] = e.entity_id
                    
            valid_pixels = idx_map[idx_map != -1]
            for pt in valid_pixels:
                if pt in pt_to_eid:
                    visibility[pt_to_eid[pt]] += 1
            
            view_entity_visibility.append(visibility)
            
        selected_views = []
        covered_entities = set()
        
        for _ in range(self.target_views):
            best_view = -1
            max_new_coverage = -1
            
            for view_idx in range(K):
                if view_idx in selected_views:
                    continue
                    
                new_coverage = 0
                for eid, count in view_entity_visibility[view_idx].items():
                    if eid not in covered_entities and count > 10:
                        new_coverage += count
                        
                if new_coverage > max_new_coverage:
                    max_new_coverage = new_coverage
                    best_view = view_idx
                    
            if best_view != -1:
                selected_views.append(best_view)
                for eid, count in view_entity_visibility[best_view].items():
                    if count > 10:
                        covered_entities.add(eid)
            else:
                break
                
        # 如果贪心选不够 target_views，随便补齐剩下的
        idx = 0
        while len(selected_views) < self.target_views and idx < K:
            if idx not in selected_views:
                selected_views.append(idx)
            idx += 1
            
        return selected_views
