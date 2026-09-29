import base64
import copy
import json
import re
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np

DEFAULT_POSE_CHECK_VIEWS = ([10, 0], [10, 90], [10, 180], [10, 270])


def pose_is_abnormal(pose_assessment: Optional[Dict[str, Any]]) -> bool:
    return bool(pose_assessment) and not bool(pose_assessment.get("is_pose_reasonable", True))


def filter_unified_knowledge_for_pose(
    unified_knowledge: Optional[dict],
    object_category: str,
    view_id_str: str,
    pose_assessment: Optional[Dict[str, Any]] = None,
) -> dict:
    if not unified_knowledge:
        return {}

    target_knowledge = unified_knowledge
    if object_category in unified_knowledge:
        target_knowledge = unified_knowledge[object_category]

    abnormal = pose_is_abnormal(pose_assessment)
    filtered_unified_knowledge = {}
    for part_name, part_info in target_knowledge.items():
        if isinstance(part_info, dict):
            filtered_part_info = {}
            for key, value in part_info.items():
                if key == "view_appearance":
                    continue
                if abnormal and key in {"z_height_range", "spatial_location"}:
                    continue
                filtered_part_info[key] = copy.deepcopy(value)
            if (not abnormal) and "view_appearance" in part_info and view_id_str in part_info["view_appearance"]:
                filtered_part_info["current_view_appearance"] = copy.deepcopy(part_info["view_appearance"][view_id_str])
            filtered_unified_knowledge[part_name] = filtered_part_info
        else:
            filtered_unified_knowledge[part_name] = copy.deepcopy(part_info)
    return filtered_unified_knowledge


def filter_text_instance_knowledge_for_pose(
    text_view_instance_knowledge: Optional[dict],
    pose_assessment: Optional[Dict[str, Any]] = None,
) -> dict:
    if not text_view_instance_knowledge:
        return {}
    abnormal = pose_is_abnormal(pose_assessment)
    if not abnormal:
        return copy.deepcopy(text_view_instance_knowledge)

    filtered = {}
    for mask_key, feat in text_view_instance_knowledge.items():
        if isinstance(feat, dict):
            filtered[mask_key] = {
                k: copy.deepcopy(v)
                for k, v in feat.items()
                if k not in {"spatial_centroid", "normalized_z"}
            }
        else:
            filtered[mask_key] = copy.deepcopy(feat)
    return filtered


def adapt_global_topology_text_for_pose(
    global_topology_text: str,
    pose_assessment: Optional[Dict[str, Any]] = None,
) -> str:
    if not global_topology_text:
        return global_topology_text
    if not pose_is_abnormal(pose_assessment):
        return global_topology_text
    return global_topology_text.replace(
        "Y=高度(0=底,1=顶)",
        "Y=当前摆放下的相对高度(0=当前最低,1=当前最高)",
    )


# =====================================================================
# Ablation: disable_height_prior
# 用于 PartObjaverse-Tiny 等数据集上的"去高度先验"消融。
# 与 pose-aware 的 filter_*_for_pose 函数语义一致，但触发条件是 ablation
# 配置 disable_height_prior=True，独立于姿态预检结论。
# =====================================================================
def filter_unified_knowledge_for_no_height(
    unified_knowledge: Optional[dict],
    object_category: str,
    disable_height_prior: bool,
) -> dict:
    """剔除 unified_knowledge 中各部件的 z_height_range / y_height_range 字段。

    保持顶层结构（含 {category: {...}} 和 {part: {...}} 两种）。
    disable_height_prior=False 时返回深拷贝，避免下游误改原对象。
    """
    if not unified_knowledge:
        return {}
    if not disable_height_prior:
        return copy.deepcopy(unified_knowledge)

    out = copy.deepcopy(unified_knowledge)

    def _strip_part_dict(part_dict: dict) -> None:
        for _part_name, part_info in part_dict.items():
            if isinstance(part_info, dict):
                part_info.pop("z_height_range", None)
                part_info.pop("y_height_range", None)

    if (
        object_category
        and object_category in out
        and isinstance(out[object_category], dict)
    ):
        _strip_part_dict(out[object_category])
    else:
        _strip_part_dict(out)
    return out


def filter_text_instance_knowledge_for_no_height(
    text_view_instance_knowledge: Optional[dict],
    disable_height_prior: bool,
) -> dict:
    """剔除每个 mask 文本知识中的 normalized_z 与 spatial_centroid.y_height。"""
    if not text_view_instance_knowledge:
        return {}
    if not disable_height_prior:
        return copy.deepcopy(text_view_instance_knowledge)

    out: Dict[str, Any] = {}
    for mask_key, feat in text_view_instance_knowledge.items():
        if isinstance(feat, dict):
            new_feat: Dict[str, Any] = {}
            for k, v in feat.items():
                if k == "normalized_z":
                    continue
                if k == "spatial_centroid" and isinstance(v, dict):
                    # spatial_centroid.y_height 与 normalized_z 表达同一信息，一并剔除
                    new_feat[k] = {
                        kk: copy.deepcopy(vv)
                        for kk, vv in v.items()
                        if kk != "y_height"
                    }
                else:
                    new_feat[k] = copy.deepcopy(v)
            out[mask_key] = new_feat
        else:
            out[mask_key] = copy.deepcopy(feat)
    return out


def strip_height_from_global_topology_text(
    global_topology_text: str,
    disable_height_prior: bool,
) -> str:
    """从 global_topology_text 中移除显式的高度数值与 Y 轴说明。

    保留拓扑结构（接触关系、空间分布模式），仅去掉 ``h=[a~b]`` 与
    ``Y=高度(0=底,1=顶)`` 这类直接的高度提示。
    """
    if not disable_height_prior or not global_topology_text:
        return global_topology_text

    import re as _re
    out = global_topology_text
    # 1) 头部 Y 轴说明
    out = out.replace(
        "Y=高度(0=底,1=顶)；",
        "（高度先验已禁用）",
    )
    # 2) 部件层次行内的 h=[a~b]，分别匹配 ", h=[..]" 和 "h=[..], "
    out = _re.sub(r",\s*h=\[[\-\d\.]+~[\-\d\.]+\]", "", out)
    out = _re.sub(r"h=\[[\-\d\.]+~[\-\d\.]+\],\s*", "", out)
    # 3) 显式的"从顶到底"措辞
    out = out.replace("（从顶到底）", "")
    return out


class PosePlausibilityChecker:
    def __init__(self, api_key: str, base_url: str = None, model_name: str = "gpt-4o"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            self.client = None

    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', img_bgr)
        return base64.b64encode(buffer).decode('utf-8')

    def _fallback_result(self, views: Sequence[Sequence[float]], reason: str) -> Dict[str, Any]:
        return {
            "is_pose_reasonable": True,
            "pose_type": "normal",
            "reasoning": reason,
            "checked_views": [list(v) for v in views],
            "raw_response": "",
            "source": "fallback",
        }

    def _parse_response(self, response_text: str, views: Sequence[Sequence[float]]) -> Dict[str, Any]:
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            clean_text = re.sub(r'^```(?:json)?\s*', '', clean_text)
            clean_text = re.sub(r'\s*```$', '', clean_text)

        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in pose assessment response")

        data = json.loads(match.group(0))
        pose_flag = data.get("is_pose_reasonable")
        if isinstance(pose_flag, str):
            pose_flag = pose_flag.strip().lower() in {"true", "1", "yes", "normal", "reasonable"}
        elif not isinstance(pose_flag, bool):
            pose_type_str = str(data.get("pose_type", "")).strip().lower()
            if pose_type_str in {"abnormal", "odd", "strange", "unreasonable", "singular"}:
                pose_flag = False
            else:
                pose_flag = True

        reasoning = str(data.get("reasoning", "")).strip()
        pose_type = str(data.get("pose_type", "normal" if pose_flag else "abnormal")).strip() or ("normal" if pose_flag else "abnormal")

        return {
            "is_pose_reasonable": bool(pose_flag),
            "pose_type": pose_type,
            "reasoning": reasoning,
            "checked_views": [list(v) for v in views],
            "raw_response": response_text,
            "source": "mllm",
        }

    def assess_pose(
        self,
        images: Sequence[np.ndarray],
        object_category: str,
        views: Sequence[Sequence[float]] = DEFAULT_POSE_CHECK_VIEWS,
    ) -> Dict[str, Any]:
        if self.client is None:
            return self._fallback_result(views, "Pose checker client unavailable. Fallback to normal pose assumption.")
        if not images:
            return self._fallback_result(views, "No pose-check images available. Fallback to normal pose assumption.")

        view_lines = [
            f"  视角{i + 1}: 仰角 {float(view[0]):.1f}°, 方位角 {float(view[1]):.1f}°"
            for i, view in enumerate(views)
        ]
        prompt = (
            f"你是 3D 点云姿态审核专家。请根据同一物体的多视角渲染图，判断该 {object_category} 当前摆放姿态是否符合常理。\n\n"
            f"【判定重点】\n"
            f"- 重点关注物体的上下朝向是否自然。\n"
            f"- 若支撑结构应朝下却明显朝上、承托面明显朝下、整体倒置、侧翻、严重倾倒或头脚颠倒，则判为姿态不合理。\n"
            f"- 不要纠结细小局部遮挡，优先判断整体摆放方向是否正常。\n\n"
            f"【输入视角】\n"
            + "\n".join(view_lines)
            + "\n\n"
            f"【输出格式】\n"
            f"只输出严格 JSON，不要输出任何其他文本：\n"
            f"{{\n"
            f"  \"is_pose_reasonable\": true,\n"
            f"  \"pose_type\": \"normal\",\n"
            f"  \"reasoning\": \"简要说明判断依据，重点说明上下姿态是否合理\"\n"
            f"}}"
        )

        content = [{"type": "text", "text": prompt}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._encode_image_to_base64(image)}"
                    },
                }
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=512,
                temperature=0.1,
                timeout=60.0,
            )
            response_text = response.choices[0].message.content or ""
            return self._parse_response(response_text, views)
        except Exception as e:
            return self._fallback_result(views, f"Pose check failed: {e}")
