import numpy as np
import json
import os
import copy
from typing import List, Dict, Tuple
from datatypes import SoftMask2D
from module_3d_lift.geometric_features import GeometricFeatureExtractor


class InstanceKnowledgeUpdater:
    def __init__(self, pc_xyz: np.ndarray, index_maps: List[np.ndarray], mllm_client, model_name: str = "gpt-4o"):
        """
        初始化知识库更新器
        """
        self.pc_xyz = pc_xyz
        self.index_maps = index_maps
        self.mllm_client = mllm_client
        self.model_name = model_name

    def _extract_mask_points(self, mask: np.ndarray, view_idx: int) -> np.ndarray:
        mask_pixels = np.where(mask > 0)
        index_map = self.index_maps[view_idx]
        if index_map.ndim == 3:
            index_map = index_map[:, :, 0]
        pc_indices = index_map[mask_pixels[0], mask_pixels[1]]
        valid_indices = pc_indices[pc_indices >= 0]
        if len(valid_indices) == 0:
            return np.array([])
        return self.pc_xyz[valid_indices]

    def build_updated_knowledge(self, soft_masks_across_views: List[List[SoftMask2D]], semantics_across_views: List[List[str]]) -> dict:
        """
        构建更新版的实例知识库。
        邻接/空间拓扑关系已移至 TopologyGraphBuilder 处理，
        此处仅保留高置信度点统计和掩码信息汇总。
        """
        print("    [Knowledge Update] Building updated instance knowledge...")

        # 1. 赋予全局唯一 ID，并提取 3D 点云
        all_masks_info = [] # [{"global_id": "view_0_mask_1", "view_idx": 0, "mask_idx": 1, "semantic": "leg", "pts_3d": ndarray}]

        for v_idx, (soft_masks, sems) in enumerate(zip(soft_masks_across_views, semantics_across_views)):
            for m_idx, (sm, sem) in enumerate(zip(soft_masks, sems)):
                pts_3d = self._extract_mask_points(sm.mask, v_idx)
                if len(pts_3d) < 5: # 点数太少忽略
                    continue
                all_masks_info.append({
                    "global_id": f"view_{v_idx}_mask_{m_idx}",
                    "view_idx": v_idx,
                    "mask_idx": m_idx,
                    "semantic": sem,
                    "pts_3d": pts_3d
                })

        updated_knowledge = {
            "highly_confident_points": {} # point_idx -> semantic
        }

        # 计算 100% 投票的高置信度点
        point_votes = {}
        for info in all_masks_info:
            sem = info["semantic"]
            mask_pixels = np.where(soft_masks_across_views[info["view_idx"]][info["mask_idx"]].mask > 0)
            idx_map = self.index_maps[info["view_idx"]]
            if idx_map.ndim == 3: idx_map = idx_map[:, :, 0]
            valid_indices = idx_map[mask_pixels[0], mask_pixels[1]]
            valid_indices = valid_indices[valid_indices >= 0]

            for idx in valid_indices:
                if idx not in point_votes:
                    point_votes[idx] = {}
                point_votes[idx][sem] = point_votes[idx].get(sem, 0) + 1

        confident_pts_count = 0
        for idx, votes_dict in point_votes.items():
            total_votes = sum(votes_dict.values())
            if total_votes > 3 and len(votes_dict) == 1:
                sem = list(votes_dict.keys())[0]
                if sem not in ["background", "unlabeled"]:
                    updated_knowledge["highly_confident_points"][int(idx)] = sem
                    confident_pts_count += 1
        print(f"    [Knowledge Update] Found {confident_pts_count} highly confident points.")

        return updated_knowledge, all_masks_info

    def get_reasonable_adjacencies_from_mllm(self, category: str, parts: List[str]) -> dict:
        print("    [Knowledge Update] Calling MLLM to get reasonable adjacencies...")
        prompt = (
            f"You are a 3D anatomy expert. The object category is '{category}'.\n"
            f"The predefined parts are: {parts}.\n"
            f"Please define which parts are geometrically and functionally ADJACENT (touching or connected) to each other.\n"
            f"Output a JSON object where each key is a part name, and its value is a list of part names it connects to.\n"
            "Example:\n"
            "{\n"
            "  \"seat\": [\"back\", \"leg\", \"arm\"],\n"
            "  \"back\": [\"seat\", \"arm\"]\n"
            "}\n"
            "Do not output markdown code blocks or explanations, only the raw JSON."
        )

        try:
            res = self.mllm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )
            text = (res.choices[0].message.content or "").strip()
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                text = json_match.group(1)

            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                return json.loads(text[start_idx:end_idx+1])
            return json.loads(text)
        except Exception as e:
            print(f"Failed to get reasonable adjacencies: {e}")
            return {}
