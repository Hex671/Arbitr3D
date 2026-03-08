from typing import List, Dict
import numpy as np
import sys
import os
import open3d as o3d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datatypes import Entity3D

class ConflictResolver:
    def __init__(self):
        pass

    def resolve_overlaps(self, entities: List[Entity3D]) -> List[Entity3D]:
        """
        微观硬隔离：找出共享点，执行细粒度优先原则。
        如果点 p 同时属于大体积 EA 和小体积 EB，强制将其判给 EB。
        """
        # 建立 点 -> 所属 Entities 映射
        point_to_entities = {}
        for entity in entities:
            for pt in entity.point_indices:
                if pt not in point_to_entities:
                    point_to_entities[pt] = []
                point_to_entities[pt].append(entity.entity_id)
                
        # 统计每个 entity 的体积（点数）
        entity_volumes = {e.entity_id: len(e.point_indices) for e in entities}
        
        # 清空原始实体的点列表，准备重新分配
        new_point_indices_map = {e.entity_id: [] for e in entities}
        
        for pt, e_ids in point_to_entities.items():
            if len(e_ids) == 1:
                new_point_indices_map[e_ids[0]].append(pt)
            else:
                # 冲突发生，细粒度优先（选体积最小的 Entity）
                best_e_id = min(e_ids, key=lambda eid: entity_volumes[eid])
                new_point_indices_map[best_e_id].append(pt)
                
        # 更新实体的点列表
        for entity in entities:
            entity.point_indices = new_point_indices_map[entity.entity_id]
            
        # 过滤掉空的 entity
        resolved_entities = [e for e in entities if len(e.point_indices) > 0]
        return resolved_entities

    def graph_cut_smoothing(self, entities: List[Entity3D], point_cloud: o3d.geometry.PointCloud) -> List[Entity3D]:
        """
        基于 3D 空间 KNN 建立邻接图，执行多数投票，消除“飞地”和锯齿。
        这部分使用 Open3D 的 KDTreeFlann 实现简单的多数投票平滑。
        """
        if not entities or len(np.asarray(point_cloud.points)) == 0:
            return entities
            
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
        num_points = np.asarray(point_cloud.points).shape[0]
        
        # 记录每个点当前的 labels，-1 表示背景
        point_labels = np.ones(num_points, dtype=np.int32) * -1
        
        for e in entities:
            for pt in e.point_indices:
                point_labels[pt] = e.entity_id
                
        smoothed_labels = np.copy(point_labels)
        
        # 对每个点进行 KNN 多数投票
        k = 10 # 邻居数量
        for i in range(num_points):
            [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud.points[i], k)
            
            neighbors_labels = point_labels[idx]
            # 过滤掉背景投票
            valid_labels = neighbors_labels[neighbors_labels != -1]
            
            if len(valid_labels) > 0:
                # 多数投票
                counts = np.bincount(valid_labels)
                majority_label = np.argmax(counts)
                smoothed_labels[i] = majority_label
                
        # 将平滑后的标签写回 Entities
        new_indices_map = {e.entity_id: [] for e in entities}
        for pt, label in enumerate(smoothed_labels):
            if label != -1 and label in new_indices_map:
                new_indices_map[label].append(pt)
                
        for e in entities:
            e.point_indices = new_indices_map[e.entity_id]
            
        return [e for e in entities if len(e.point_indices) > 0]
