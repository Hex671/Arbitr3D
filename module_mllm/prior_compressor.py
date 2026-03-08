from typing import List, Dict
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datatypes import Entity3D

class PriorCompressor:
    def __init__(self):
        pass

    def aggregate_glip_scores(self, entities: List[Entity3D]) -> Dict[str, Dict[str, float]]:
        """
        累加并归一化每个 Entity3D 的 glip_scores，生成浓缩先验字典。
        返回格式: {"1": {"chair back": 0.8, "chair seat": 0.2}, ...}
        """
        prior_dict = {}
        for entity in entities:
            scores = entity.glip_scores
            if not scores:
                prior_dict[str(entity.entity_id)] = {}
                continue
                
            # 归一化
            total_score = sum(scores.values())
            normalized_scores = {k: round(v / total_score, 3) for k, v in scores.items()}
            
            # 按分数降序排列
            sorted_scores = dict(sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True))
            prior_dict[str(entity.entity_id)] = sorted_scores
            
        return prior_dict
