from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

@dataclass
class Entity3D:
    """
    经过阶段 2 融合与隔离后的绝对物理部件 (3D实体提议)
    """
    entity_id: int
    point_indices: List[int] = field(default_factory=list)
    glip_scores: Dict[str, float] = field(default_factory=dict)
    
    def add_point_index(self, index: int):
        self.point_indices.append(index)
        
    def update_glip_scores(self, scores: Dict[str, float]):
        for cls_name, score in scores.items():
            if cls_name not in self.glip_scores:
                self.glip_scores[cls_name] = score
            else:
                self.glip_scores[cls_name] += score
