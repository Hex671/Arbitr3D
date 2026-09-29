"""
Refinement Module: MLLM Feedback-based Segmentation Refinement

This module implements a feedback loop where MLLM evaluates segmentation quality
and triggers local re-segmentation for problematic masks.
"""

import re
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import json


def _normalize_fragment_id(fid) -> Optional[int]:
    """
    安全地将 fragment_id 转换为整数。
    支持格式: 0, "0", "f0", "f_new_1", "F0", "mask_1" 等
    """
    if fid is None:
        return None
    if isinstance(fid, int):
        return fid
    if isinstance(fid, float):
        return int(fid)
    if isinstance(fid, str):
        fid = fid.strip()
        # 移除常见前缀
        for prefix in ['f', 'F', 'fragment_', 'mask_']:
            if fid.lower().startswith(prefix):
                fid = fid[len(prefix):]
                break
        # 尝试直接转换为整数
        try:
            return int(fid)
        except ValueError:
            # 提取第一个数字序列
            match = re.search(r'\d+', fid)
            if match:
                return int(match.group())
    return None


class SegmentationRefiner:
    """MLLM feedback-driven segmentation refiner"""

    def __init__(self, sam_segmenter=None):
        """
        Args:
            sam_segmenter: SAMAutoSegmenter instance or SamPredictor
        """
        self.sam_segmenter = sam_segmenter
        self._predictor = None

    def _get_predictor(self, sam_predictor=None):
        """获取 SAM Predictor"""
        if sam_predictor is not None:
            return sam_predictor

        if self._predictor is not None:
            return self._predictor

        if self.sam_segmenter is not None:
            if hasattr(self.sam_segmenter, 'get_predictor'):
                self._predictor = self.sam_segmenter.get_predictor()
            elif hasattr(self.sam_segmenter, 'sam'):
                from segment_anything import SamPredictor
                self._predictor = SamPredictor(self.sam_segmenter.sam)
            else:
                self._predictor = self.sam_segmenter
        return self._predictor

    def expand_bbox(self, mask: np.ndarray, expand_ratio: float = 1.5) -> Tuple[int, int, int, int]:
        """
        计算掩码的边界框，并按比例扩大。

        Args:
            mask: 二值掩码 (H, W)
            expand_ratio: 扩大比例

        Returns:
            (x_min, y_min, x_max, y_max)
        """
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            return (0, 0, 0, 0)

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        h, w = mask.shape
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        bw, bh = x_max - x_min, y_max - y_min

        new_w, new_h = bw * expand_ratio, bh * expand_ratio

        new_x_min = max(0, int(cx - new_w / 2))
        new_y_min = max(0, int(cy - new_h / 2))
        new_x_max = min(w, int(cx + new_w / 2))
        new_y_max = min(h, int(cy + new_h / 2))

        return (new_x_min, new_y_min, new_x_max, new_y_max)

    def crop_roi(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Dict]:
        """
        裁剪图像 ROI，并记录坐标偏移。

        Args:
            image: (H, W, 3) RGB 图像
            bbox: (x_min, y_min, x_max, y_max)

        Returns:
            (roi_image, offset_info)
        """
        x_min, y_min, x_max, y_max = bbox
        roi = image[y_min:y_max, x_min:x_max].copy()
        offset = {"x_offset": x_min, "y_offset": y_min}
        return roi, offset

    def local_resegment(
        self,
        image: np.ndarray,
        old_mask: np.ndarray,
        part_name: str,
        sam_predictor=None
    ) -> Optional[np.ndarray]:
        """
        在原掩码区域局部重分割。

        Args:
            image: (H, W, 3) RGB 图像
            old_mask: 原掩码 (H, W)
            part_name: 部件名称（用于描述）
            sam_predictor: SAM predictor（如果未在 init 时提供）

        Returns:
            新的掩码 (H, W)，如果失败返回 None
        """
        predictor = self._get_predictor(sam_predictor)
        if predictor is None:
            print("[Refiner] No SAM predictor available, skipping refinement.")
            return None

        h, w = image.shape[:2]

        # 1. 获取扩大的 ROI
        bbox = self.expand_bbox(old_mask, expand_ratio=1.5)
        roi, offset = self.crop_roi(image, bbox)

        if roi.size == 0:
            return None

        # 2. 在 ROI 上运行 SAM 全图分割
        predictor.set_image(roi)

        # 使用多个点作为提示（中心点 + 边缘点）
        roi_mask = old_mask[offset["y_offset"]:offset["y_offset"]+roi.shape[0],
                           offset["x_offset"]:offset["x_offset"]+roi.shape[1]]

        # 计算掩码中心
        y_indices, x_indices = np.where(roi_mask > 0)
        if len(y_indices) == 0:
            # 如果原掩码在 ROI 中为空，使用 ROI 中心
            point_coords = np.array([[roi.shape[1]//2, roi.shape[0]//2]])
        else:
            # 使用掩码中心作为正提示
            cx, cy = int(np.mean(x_indices)), int(np.mean(y_indices))
            point_coords = np.array([[cx, cy]])

        point_labels = np.array([1])  # 1 = 前景

        # 分割
        try:
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )

            # 选择得分最高的掩码
            best_idx = np.argmax(scores)
            roi_new_mask = masks[best_idx]

            # 将 ROI 掩码映射回原图坐标
            full_mask = np.zeros((h, w), dtype=bool)
            roi_h, roi_w = roi_new_mask.shape
            x_off, y_off = offset["x_offset"], offset["y_offset"]

            # 确保不超出边界
            x_end = min(x_off + roi_w, w)
            y_end = min(y_off + roi_h, h)
            full_mask[y_off:y_end, x_off:x_end] = roi_new_mask[:y_end-y_off, :x_end-x_off]

            return full_mask

        except Exception as e:
            print(f"[Refiner] SAM prediction failed: {e}")
            return None

    def calculate_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """计算两个掩码的 IoU"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union

    def select_best_match(
        self,
        new_masks: List[np.ndarray],
        old_mask: np.ndarray,
        iou_threshold: float = 0.2
    ) -> Optional[Tuple[int, np.ndarray]]:
        """
        从多个新掩码中选择与原掩码最匹配的一个。

        Args:
            new_masks: 新掩码列表
            old_mask: 原掩码
            iou_threshold: IoU 阈值

        Returns:
            (最佳掩码索引, 最佳掩码) 或 None
        """
        if not new_masks:
            return None

        best_idx = None
        best_score = 0

        for i, m in enumerate(new_masks):
            iou = self.calculate_mask_iou(m, old_mask)

            # 面积比例
            old_area = old_mask.sum()
            new_area = m.sum()
            if old_area > 0:
                area_ratio = min(new_area, old_area) / max(new_area, old_area)
            else:
                area_ratio = 0

            # 综合得分：IoU 权重更高
            score = iou * 0.7 + area_ratio * 0.3

            if score > best_score and iou >= iou_threshold:
                best_score = score
                best_idx = i

        if best_idx is not None:
            return (best_idx, new_masks[best_idx])
        return None

    def refine_masks_with_feedback(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        semantics: List[str],
        feedback: Dict,
        max_refinements: int = 3,
        sam_predictor=None,
        display_id_to_mask_indices: Optional[Dict[int, List[int]]] = None,
    ) -> List[np.ndarray]:
        """
        根据 MLLM 反馈细化掩码。

        Args:
            image: (H, W, 3) RGB 图像
            masks: 当前掩码列表
            semantics: 掩码对应的语义标签
            feedback: MLLM 反馈 JSON
            max_refinements: 最大细化次数
            sam_predictor: 可选的 SAM predictor

        Returns:
            细化后的掩码列表
        """
        predictor = self._get_predictor(sam_predictor)
        if predictor is None:
            print("[Refiner] No SAM predictor, returning original masks.")
            return masks

        refined_masks = [m.copy() for m in masks]

        parts_needing_refine = feedback.get("parts_needing_refine", [])
        if not parts_needing_refine:
            return refined_masks

        refine_count = 0
        for item in parts_needing_refine:
            if refine_count >= max_refinements:
                break

            part_name = item.get("part_name", "")
            sid = item.get("semantic_display_id", None)
            indices: List[int] = []

            if sid is not None and display_id_to_mask_indices:
                indices = list(display_id_to_mask_indices.get(int(sid), []))
            if not indices:
                mid = item.get("mask_id", -1)
                if isinstance(mid, int) and 0 <= mid < len(refined_masks):
                    indices = [mid]

            for mask_id in indices:
                if refine_count >= max_refinements:
                    break
                if mask_id < 0 or mask_id >= len(refined_masks):
                    continue
                old_mask = refined_masks[mask_id]
                new_mask = self.local_resegment(image, old_mask, part_name, predictor)
                if new_mask is not None:
                    refined_masks[mask_id] = new_mask
                    refine_count += 1
                    print(f"[Refiner] Refined mask {mask_id} ({part_name})")

        return refined_masks

    # ============================================================
    # Layer 2: 碎片级迭代修正
    # ============================================================

    def refine_single_part_iterative(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        semantics: List[str],
        part_display_id: int,
        part_name: str,
        fragment_feedback: List[dict],
        grouping,
        sam_predictor=None,
        max_rounds: int = 2,
    ) -> List[np.ndarray]:
        """
        对单个部件执行 2 轮碎片级迭代修正。

        Args:
            image: (H, W, 3) RGB 图像
            masks: 当前掩码列表（会在内部被原地修改）
            semantics: 语义标签列表
            part_display_id: 部件的 display_id
            part_name: 部件名称
            fragment_feedback: 该部件的 fragment_errors 列表
            grouping: SemanticGroupingResult
            sam_predictor: SAM predictor
            max_rounds: 最大迭代轮次（默认 2）

        Returns:
            修正后的 masks 列表
        """
        predictor = self._get_predictor(sam_predictor)
        if predictor is None:
            print("[Refiner] No SAM predictor, skipping part refinement.")
            return masks

        current_masks = [m.copy() if isinstance(m, np.ndarray) else m for m in masks]
        frag_lookup = {
            fr["fragment_id"]: fr for fr in grouping.display_id_to_fragments.get(part_display_id, [])
        }

        img_h, img_w = image.shape[:2]

        for round_idx in range(max_rounds):
            has_changes = False

            pending_supplements: List[dict] = []
            for fb in fragment_feedback:
                action = fb.get("action", "")
                frag_id = fb.get("fragment_id")
                fid = _normalize_fragment_id(frag_id)

                if fid is None:
                    print(f"  [Round {round_idx+1}] Warning: Invalid fragment_id '{frag_id}' - skipping action")
                    continue

                if action == "delete":
                    if fid is not None and fid in frag_lookup:
                        frag = frag_lookup[fid]
                        for mid in frag["mask_indices"]:
                            if 0 <= mid < len(current_masks):
                                current_masks[mid] = np.zeros_like(current_masks[mid])
                                print(f"  [Round {round_idx+1}] Deleted fragment f{fid} (mask {mid})")
                                has_changes = True

                elif action == "supplement":
                    pending_supplements.append(fb)

            for fb in pending_supplements:
                frag_id = fb.get("fragment_id")
                fid = _normalize_fragment_id(frag_id)
                pos_pts = fb.get("positive_points", [])
                neg_pts = fb.get("negative_points", [])

                # 过滤无效点（坐标已由 mllm_classifier 映射到原图）
                pos_pts_valid = [p for p in pos_pts if isinstance(p, (list, tuple)) and len(p) == 2]
                neg_pts_valid = [p for p in neg_pts if isinstance(p, (list, tuple)) and len(p) == 2]

                if not pos_pts_valid:
                    print(f"  [Round {round_idx+1}] Skipping supplement for f{fid}: no valid positive points")
                    continue

                if fid is None:
                    print(f"  [Round {round_idx+1}] Skipping supplement: invalid fragment_id '{frag_id}'")
                    continue

                new_mask = self.local_resegment_multi_point(
                    image, pos_pts_valid, neg_pts_valid, predictor
                )
                if new_mask is not None and new_mask.any():
                    current_masks.append(new_mask)
                    print(f"  [Round {round_idx+1}] Added supplement fragment for f{fid}, mask_id={len(current_masks)-1}")
                    has_changes = True

            if not has_changes:
                print(f"  [Refiner] Part {part_name}: Round {round_idx+1} no changes, stopping.")
                break

        return current_masks

    def local_resegment_multi_point(
        self,
        image: np.ndarray,
        positive_points: List[List[int]],
        negative_points: List[List[int]],
        sam_predictor=None,
        expand_ratio: float = 2.0,
        use_full_image: bool = False,
    ) -> Optional[np.ndarray]:
        """
        使用多个正负点对在原图上执行 SAM 分割。

        Args:
            image: (H, W, 3) RGB 图像
            positive_points: 正点列表，每个点为 [x, y]
            negative_points: 负点列表，每个点为 [x, y]
            sam_predictor: SAM predictor
            expand_ratio: 基于所有点范围的 bbox 扩大比例（use_full_image=True 时忽略）
            use_full_image: 为 True 时 ROI 为整张图 (0,0)-(W,H)，避免跨视角时用
                提示点包络估计部件尺度导致 ROI 过小。

        Returns:
            新的掩码 (H, W)，如果失败返回 None
        """
        predictor = self._get_predictor(sam_predictor)
        if predictor is None:
            print("[Refiner] No SAM predictor available.")
            return None

        h, w = image.shape[:2]
        all_pts = positive_points + negative_points
        if not all_pts:
            print("[Refiner] No points provided for resegment.")
            return None

        if use_full_image:
            bx0, by0, bx1, by1 = 0, 0, w, h
            roi = image.copy()
        else:
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            x_min, x_max = max(0, min(xs)), min(w, max(xs))
            y_min, y_max = max(0, min(ys)), min(h, max(ys))

            bw, bh = x_max - x_min, y_max - y_min
            if bw <= 0 or bh <= 0:
                bw, bh = max(10, bw), max(10, bh)

            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            new_w, new_h = bw * expand_ratio, bh * expand_ratio

            bx0 = max(0, int(cx - new_w / 2))
            by0 = max(0, int(cy - new_h / 2))
            bx1 = min(w, int(cx + new_w / 2))
            by1 = min(h, int(cy + new_h / 2))

            roi = image[by0:by1, bx0:bx1].copy()
        if roi.size == 0:
            return None

        predictor.set_image(roi)

        roi_pts = []
        roi_labels = []
        for px, py in positive_points:
            rx, ry = px - bx0, py - by0
            if 0 <= rx < roi.shape[1] and 0 <= ry < roi.shape[0]:
                roi_pts.append([rx, ry])
                roi_labels.append(1)
        for px, py in negative_points:
            rx, ry = px - bx0, py - by0
            if 0 <= rx < roi.shape[1] and 0 <= ry < roi.shape[0]:
                roi_pts.append([rx, ry])
                roi_labels.append(0)

        if not roi_pts:
            return None

        point_coords = np.array(roi_pts, dtype=np.float32)
        point_labels = np.array(roi_labels, dtype=np.int32)

        try:
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            best_idx = np.argmax(scores)
            roi_mask = masks[best_idx]

            full_mask = np.zeros((h, w), dtype=bool)
            roi_h, roi_w = roi_mask.shape
            x_end = min(bx0 + roi_w, w)
            y_end = min(by0 + roi_h, h)
            full_mask[by0:y_end, bx0:x_end] = roi_mask[:y_end-by0, :x_end-bx0]
            return full_mask

        except Exception as e:
            print(f"[Refiner] SAM multi-point prediction failed: {e}")
            return None

    def local_resegment_multi_point_with_bbox(
        self,
        image: np.ndarray,
        positive_points: List[List[int]],
        negative_points: List[List[int]],
        bbox: Optional[Tuple[int, int, int, int]],
        sam_predictor=None,
    ) -> Optional[np.ndarray]:
        """
        使用正负点和显式 bbox 执行 SAM 分割。

        Args:
            image: (H, W, 3) RGB 图像
            positive_points: 正点列表，每个点为 [x, y]
            negative_points: 负点列表，每个点为 [x, y]
            bbox: 显式 bbox (x1, y1, x2, y2)，坐标系为 (x_right, y_down)
            sam_predictor: SAM predictor

        Returns:
            新的掩码 (H, W)，如果失败返回 None
        """
        predictor = self._get_predictor(sam_predictor)
        if predictor is None:
            print("[Refiner] No SAM predictor available.")
            return None

        h, w = image.shape[:2]

        if bbox is not None:
            bx0, by0, bx1, by1 = bbox
        else:
            if not (positive_points or negative_points):
                return None
            all_pts = positive_points + negative_points
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            bx0, by0 = max(0, int(min(xs))), max(0, int(min(ys)))
            bx1, by1 = min(w, int(max(xs) + 1)), min(h, int(max(ys) + 1))

        bx0 = max(0, min(bx0, w - 1))
        by0 = max(0, min(by0, h - 1))
        bx1 = max(bx0 + 1, min(bx1, w))
        by1 = max(by0 + 1, min(by1, h))

        roi = image[by0:by1, bx0:bx1].copy()
        if roi.size == 0:
            return None

        predictor.set_image(roi)

        roi_pts = []
        roi_labels = []
        for px, py in positive_points:
            rx, ry = px - bx0, py - by0
            if 0 <= rx < roi.shape[1] and 0 <= ry < roi.shape[0]:
                roi_pts.append([float(rx), float(ry)])
                roi_labels.append(1)
        for px, py in negative_points:
            rx, ry = px - bx0, py - by0
            if 0 <= rx < roi.shape[1] and 0 <= ry < roi.shape[0]:
                roi_pts.append([float(rx), float(ry)])
                roi_labels.append(0)

        if not roi_pts:
            return None

        point_coords = np.array(roi_pts, dtype=np.float32)
        point_labels = np.array(roi_labels, dtype=np.int32)

        try:
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            best_idx = np.argmax(scores)
            roi_mask = masks[best_idx]

            full_mask = np.zeros((h, w), dtype=bool)
            roi_h, roi_w = roi_mask.shape
            x_end = min(bx0 + roi_w, w)
            y_end = min(by0 + roi_h, h)
            full_mask[by0:y_end, bx0:x_end] = roi_mask[:y_end - by0, :x_end - bx0]
            return full_mask

        except Exception as e:
            print(f"[Refiner] SAM bbox+points prediction failed: {e}")
            return None


def build_feedback_prompt(
    image: np.ndarray,
    masks: List[np.ndarray],
    semantics: List[str],
    object_category: str
) -> str:
    """
    构建 MLLM 反馈 prompt。

    Args:
        image: (H, W, 3) RGB 图像
        masks: 掩码列表
        semantics: 语义标签
        object_category: 物体类别

    Returns:
        反馈 prompt 字符串
    """
    mask_descriptions = []
    for i, (mask, sem) in enumerate(zip(masks, semantics)):
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) > 0:
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            area = len(y_indices)
            desc = f"Mask {i}: {sem} (area={area}, bbox=[{x_min},{y_min},{x_max},{y_max}])"
        else:
            desc = f"Mask {i}: {sem} (empty)"
        mask_descriptions.append(desc)

    prompt = f"""请评估以下 {object_category} 的分割掩码质量。

当前分割结果：
{chr(10).join(mask_descriptions)}

请检查：
1. 每个预定义部件的掩码是否完整覆盖了该部件？
2. 是否有部件被漏分割（完全没有掩码）？
3. 是否有掩码包含了错误的区域？

请输出 JSON 格式的反馈：
{{
  "overall_quality": "acceptable" | "needs_refine",
  "parts_needing_refine": [
    {{
      "part_name": "部件名称",
      "mask_id": 0,
      "reason": "问题描述",
      "suggestion": "建议的改进方向"
    }}
  ],
  "missing_parts": ["应该存在但没有对应掩码的部件列表"]
}}

重要：
- 如果分割质量已经足够好，overall_quality 设为 "acceptable"
- 只指出真正需要改进的问题
- mask_id 必须与上方 Mask ID 对应
"""
    return prompt


def parse_feedback_response(response_text: str) -> Dict:
    """解析 MLLM 的反馈响应"""
    try:
        # 尝试提取 JSON
        json_match = None
        for pattern in ['```json', '```']:
            if pattern in response_text:
                parts = response_text.split(pattern)
                for part in parts:
                    part = part.strip()
                    if part.startswith('{') and part.endswith('}'):
                        json_match = part
                        break

        if json_match is None:
            # 尝试直接解析
            json_match = response_text.strip()

        feedback = json.loads(json_match)
        return feedback
    except json.JSONDecodeError as e:
        print(f"[Refiner] Failed to parse feedback JSON: {e}")
        return {
            "overall_quality": "acceptable",
            "parts_needing_refine": [],
            "missing_parts": []
        }
