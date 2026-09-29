"""
RES（指代分割）第一轮 MLLM 分类器。

继承自 MLLMClassifier 以**复用** _render_som_image / _encode_image_to_base64 /
_get_annotation_font 等可视化与编码工具方法，但**完全不修改父类**。父类的
predict_soft_probabilities / predict_soft_probabilities_all_views 行为保持原样，
PartNetE / PartObjaverse-Tiny 路径零影响。

本类新增 1 个公开方法：
    predict_res_all_views(...)
        与父类 predict_soft_probabilities_all_views 接口对应，但 prompt 替换为
        指代分割模式（target + distractors，未识别归 "unknown"）。

设计要点：
- 保留父类的"大掩码过滤 + 冲突图着色 + batch SOM 渲染"逻辑（自带一份精简实现）
- prompt 显式给出 target / distractor 列表，让 MLLM 区分指代目标与干扰部件
- label 集合：{target_label} ∪ {distractor_labels} ∪ {"unknown"}
- 不使用 pose_checker（PartVerse 网格已归一化）；可选保留 unified_knowledge 中的 _role 字段
"""

import json
import re
import time
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from datatypes import SoftMask2D
from config.feature_mapping import map_geometric_features_to_text
from config.res_prompts import build_res_task_intro, format_distractor_lines
from module_2d_fm.mllm_classifier import MLLMClassifier


class RESClassifier(MLLMClassifier):
    """指代分割第一轮 MLLM 分类器。继承复用父类工具方法，零修改父类。"""

    # ==================================================================
    # 顶层入口（多视角并行）
    # ==================================================================
    def predict_res_all_views(
        self,
        images_rgb: List[np.ndarray],
        depth_maps: List[np.ndarray],
        masks_across_views: List[List[np.ndarray]],
        view_angles_str: List[str],
        target_ku: Dict,
        distractors: List[Dict[str, str]],
        initial_instance_knowledge: Dict,
        max_workers: int = 5,
    ) -> Tuple[
        List[List[SoftMask2D]],          # soft_masks_across_views
        List[List[np.ndarray]],          # som_images_labeled (供可视化)
        List[Dict[int, Dict]],           # raw_predictions per view
        List[List[np.ndarray]],          # som_images_raw（发送给 MLLM 的图）
    ]:
        """
        Args:
            images_rgb / depth_maps / masks_across_views / view_angles_str: 与现有
                pipeline 中 SAM 之后产物完全一致
            target_ku: TargetKURReconstructor 的输出（含 short_label, spatial_location, ...）
            distractors: DistractorDiscoverer 的输出 List[{short_label, brief_features}]
            initial_instance_knowledge: 与 pipeline.py 阶段 1.6 产物 schema 一致
                （{view_<i>: {mask_<j>: <raw geometric features>}}）

        Returns:
            soft_masks_across_views, som_images_labeled, raw_predictions_across_views, som_images_raw
        """
        target_label = target_ku.get("short_label", "target")
        distractor_labels = [d["short_label"] for d in distractors]
        legal_classes = [target_label] + distractor_labels + ["unknown"]
        res_intro = build_res_task_intro(target_ku, distractors)

        num_views = len(images_rgb)
        soft_masks_across_views: List[List[SoftMask2D]] = [None] * num_views
        som_images_labeled: List[List[np.ndarray]] = [None] * num_views
        som_images_raw_across_views: List[List[np.ndarray]] = [None] * num_views
        raw_predictions_across_views: List[Dict[int, Dict]] = [{}] * num_views

        def _run_one_view(vid: int):
            return vid, self._predict_one_view(
                vid=vid,
                image=images_rgb[vid],
                depth_map=depth_maps[vid],
                masks=masks_across_views[vid],
                view_angle=view_angles_str[vid],
                target_label=target_label,
                distractor_labels=distractor_labels,
                legal_classes=legal_classes,
                res_intro=res_intro,
                view_initial_instance_knowledge=initial_instance_knowledge.get(f"view_{vid}", {}),
            )

        if num_views == 0:
            return [], [], [], []

        # 简单并行
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            futures = {ex.submit(_run_one_view, vid): vid for vid in range(num_views)}
            for fut in as_completed(futures):
                vid, (sm_list, som_lbl, som_raw, raw_pred) = fut.result()
                soft_masks_across_views[vid] = sm_list
                som_images_labeled[vid] = som_lbl
                som_images_raw_across_views[vid] = som_raw
                raw_predictions_across_views[vid] = raw_pred

        return (
            soft_masks_across_views,
            som_images_labeled,
            raw_predictions_across_views,
            som_images_raw_across_views,
        )

    # ==================================================================
    # 单视角预测（复用父类工具方法）
    # ==================================================================
    def _predict_one_view(
        self,
        vid: int,
        image: np.ndarray,
        depth_map: np.ndarray,
        masks: List[np.ndarray],
        view_angle: str,
        target_label: str,
        distractor_labels: List[str],
        legal_classes: List[str],
        res_intro: str,
        view_initial_instance_knowledge: Dict,
    ) -> Tuple[
        List[SoftMask2D],
        List[np.ndarray],
        List[np.ndarray],
        Dict[int, Dict],
    ]:
        """单视角内做：大掩码过滤 → 冲突图着色 batch → MLLM 调用 → 解析 → SoftMask2D 列表。"""
        if not masks:
            return [], [], [], {}

        # ==== 1) 大掩码过滤 + 冲突图着色（与父类逻辑一致的精简版） ====
        valid_indices, bboxes, mask_areas = self._filter_invalid_masks(
            masks, image.shape[:2], view_initial_instance_knowledge
        )
        if not valid_indices:
            return self._build_unknown_outputs(masks)

        batch_global_indices = self._graph_coloring_batch(
            valid_indices, masks, bboxes, mask_areas, max_per_batch=5
        )

        # ==== 2) per-batch MLLM ====
        all_predictions: Dict[str, str] = {}
        som_raw_list: List[np.ndarray] = []
        som_labeled_list: List[np.ndarray] = []

        # 准备 instance knowledge（per-mask 几何特征文本）
        text_instance_know: Dict[str, dict] = {}
        for m_key, raw_feat in view_initial_instance_knowledge.items():
            text_instance_know[m_key] = map_geometric_features_to_text(raw_feat)

        # 编码原图与深度图（per-view 一次）
        b64_orig = self._encode_image_to_base64(image)
        b64_depth = self._encode_depth_b64(depth_map)

        for batch_idx, global_indices in enumerate(batch_global_indices):
            batch_masks = [masks[g] for g in global_indices]
            som_image, _ = self._render_som_image(image, batch_masks, global_indices)
            som_raw_list.append(som_image.copy())
            b64_som = self._encode_image_to_base64(som_image)

            # 选当前 batch 的 instance knowledge
            batch_know = {
                f"mask_{g}": text_instance_know.get(f"mask_{g}", {})
                for g in global_indices
            }

            prompt = self._build_res_prompt(
                view_angle=view_angle,
                res_intro=res_intro,
                legal_classes=legal_classes,
                batch_indices=global_indices,
                batch_instance_knowledge=batch_know,
            )

            preds = self._call_mllm_with_retries(
                prompt=prompt,
                images_b64=[b64_orig, b64_som, b64_depth],
                required_ids=set(str(g) for g in global_indices),
            )
            for k, v in preds.items():
                all_predictions[k] = v

            # 渲染带语义标注的 SOM 图（可视化用）
            labels_for_batch = [
                self._normalize_pred(all_predictions.get(str(g), "unknown"))
                for g in global_indices
            ]
            som_lbl, _ = self._render_som_image(
                image, batch_masks, global_indices,
                mask_pred_labels=labels_for_batch,
            )
            som_labeled_list.append(som_lbl)

        # ==== 3) 整理为 SoftMask2D 列表 ====
        soft_masks: List[SoftMask2D] = []
        raw_predictions: Dict[int, Dict] = {}
        for i, m in enumerate(masks):
            pred_raw = all_predictions.get(str(i), "unknown")
            cls = self._normalize_pred(pred_raw)
            cls = self._match_legal_class(cls, legal_classes)
            soft_masks.append(SoftMask2D(
                mask=m,
                probabilities={cls: 1.0},
                reasoning="",
            ))
            raw_predictions[i] = {
                "mask_id": i,
                "semantic": pred_raw,
                "matched_class": cls,
            }
        return soft_masks, som_labeled_list, som_raw_list, raw_predictions

    # ==================================================================
    # Helpers
    # ==================================================================
    @staticmethod
    def _build_unknown_outputs(masks: List[np.ndarray]):
        soft = [SoftMask2D(mask=m, probabilities={"unknown": 1.0}, reasoning="") for m in masks]
        return soft, [], [], {i: {"mask_id": i, "semantic": "unknown"} for i in range(len(masks))}

    @staticmethod
    def _filter_invalid_masks(
        masks: List[np.ndarray],
        img_hw: Tuple[int, int],
        view_instance_know: Dict,
    ) -> Tuple[List[int], List[Dict[str, int]], List[int]]:
        """复刻父类大掩码 + 全 0 几何特征 + 包含关系过滤逻辑（精简版）。"""
        img_h, img_w = img_hw
        img_area = img_h * img_w

        bboxes: List[Dict[str, int]] = []
        mask_areas: List[int] = []
        for m in masks:
            area = int(np.sum(m))
            mask_areas.append(area)
            ys, xs = np.where(m)
            if len(ys) > 0:
                bboxes.append({
                    "ymin": int(np.min(ys)), "ymax": int(np.max(ys)),
                    "xmin": int(np.min(xs)), "xmax": int(np.max(xs)),
                })
            else:
                bboxes.append({"ymin": 0, "ymax": 0, "xmin": 0, "xmax": 0})

        largest_idx = int(np.argmax(mask_areas)) if mask_areas else -1
        valid_indices: List[int] = []
        for i in range(len(masks)):
            if i == largest_idx:
                continue
            # 全 0 几何特征 → 跳过
            mask_key = f"mask_{i}"
            feat = view_instance_know.get(mask_key, {})
            if (
                str(feat.get("principal_curvature", "")).startswith("0.0000") and
                str(feat.get("point_ratio", "")).startswith("0.0000") and
                (feat.get("normalized_z") is None or str(feat.get("normalized_z", "")).startswith("0.0000"))
            ):
                continue
            # 占比 >50% 且包含其他掩码 90%+ → 父级容器，跳过
            is_valid = True
            if mask_areas[i] > 0.5 * img_area:
                for j in range(len(masks)):
                    if j == i or j == largest_idx or mask_areas[j] <= 0:
                        continue
                    b1, b2 = bboxes[i], bboxes[j]
                    if not (b1["xmax"] < b2["xmin"] or b1["xmin"] > b2["xmax"] or
                            b1["ymax"] < b2["ymin"] or b1["ymin"] > b2["ymax"]):
                        inter = np.logical_and(masks[i], masks[j]).sum()
                        if inter / max(1, mask_areas[j]) > 0.9:
                            is_valid = False
                            break
            if is_valid:
                valid_indices.append(i)
        return valid_indices, bboxes, mask_areas

    @staticmethod
    def _graph_coloring_batch(
        valid_indices: List[int],
        masks: List[np.ndarray],
        bboxes: List[Dict[str, int]],
        mask_areas: List[int],
        max_per_batch: int = 5,
    ) -> List[List[int]]:
        """冲突图着色：把不重叠的 mask 放进同 batch（每张 SOM 图最多 5 个）。"""
        n_valid = len(valid_indices)
        # 冲突矩阵
        conflict = np.zeros((n_valid, n_valid), dtype=bool)
        for i in range(n_valid):
            gi = valid_indices[i]
            for j in range(i + 1, n_valid):
                gj = valid_indices[j]
                b1, b2 = bboxes[gi], bboxes[gj]
                if not (b1["xmax"] < b2["xmin"] or b1["xmin"] > b2["xmax"] or
                        b1["ymax"] < b2["ymin"] or b1["ymin"] > b2["ymax"]):
                    if np.logical_and(masks[gi], masks[gj]).any():
                        conflict[i, j] = True
                        conflict[j, i] = True
        # 按面积升序贪心着色
        order = np.argsort([mask_areas[valid_indices[i]] for i in range(n_valid)])
        batches_local: List[List[int]] = []
        for li in order:
            placed = False
            for batch in batches_local:
                if len(batch) >= max_per_batch:
                    continue
                if not any(conflict[li, j] for j in batch):
                    batch.append(int(li))
                    placed = True
                    break
            if not placed:
                batches_local.append([int(li)])
        # 转 global indices
        return [[valid_indices[li] for li in batch] for batch in batches_local]

    def _encode_depth_b64(self, depth_map: np.ndarray) -> str:
        """复用 utils.depth_visualization.enhance_depth_map 把深度图转高对比度 PNG。"""
        from utils.depth_visualization import enhance_depth_map
        depth_bgr = enhance_depth_map(depth_map)
        _, buf = cv2.imencode(".png", depth_bgr)
        return base64.b64encode(buf).decode("utf-8")

    # ------------------------------------------------------------------
    # Prompt 拼装
    # ------------------------------------------------------------------
    @staticmethod
    def _build_res_prompt(
        view_angle: str,
        res_intro: str,
        legal_classes: List[str],
        batch_indices: List[int],
        batch_instance_knowledge: Dict[str, dict],
    ) -> str:
        legal_str = ", ".join(f"\"{c}\"" for c in legal_classes)
        prompt = (
            f"你是 3D 零件指代分割专家。请判断图中标有编号的每个掩码区域属于哪个语义类别。\n\n"
            f"{res_intro}"
            f"【输入图片】（共三张，同一视角）：\n"
            f"  图1 = 无标注的原始渲染图\n"
            f"  图2 = SOM 标注图（彩色半透明掩码 + 数字 ID + 边界线）\n"
            f"  图3 = 深度图（仅物体区域有灰度，越亮越近）\n"
            f"注意：每个编号对应的是与该数字**颜色一致且像素连通的整片色域**，不要只看数字旁边的局部。\n\n"
            f"【任务上下文】\n"
            f"  当前视角：{view_angle}\n"
            f"  合法语义列表：[{legal_str}]\n"
            f"  本批次必须包含的掩码编号（缺一不可）：{[str(g) for g in batch_indices]}\n\n"
            f"【实例知识库】（当前批次每个掩码的 3D 几何特征）：\n"
            f"{json.dumps(batch_instance_knowledge, indent=2, ensure_ascii=False)}\n\n"
            f"【推理与思考步骤（Standard Operating Procedure）】\n"
            f"**第一步：观察 + 几何匹配**\n"
            f" - 观察【图1】和【图2】，判断该掩码在 2D 画面中的形状、位置；结合【图3】判断三维深度层级。\n"
            f" - 把【实例知识库】中该 mask 的 normalized_z / relative_dimensions / shape_description 与"
            f"【指代目标】描述中的 spatial_location / typical_3D_shape / z_height_range 逐项匹配。\n\n"
            f"**第二步：与干扰部件对比**\n"
            f" - 如果不像 target，依次和每个 distractor 的 brief_features 匹配，找最接近的。\n"
            f" - 不能因为'看起来像 target 的延伸/边框'就强行归为 target；窄边框 / 壳体外框等区域应判为 unknown。\n\n"
            f"**第三步：综合裁定**\n"
            f" - 经过前两步仍无法明确归属的 mask，果断输出 \"unknown\"。\n"
            f" - 实例知识库中所有特征都为 0 的 mask 必须输出 \"unknown\"。\n"
            f" - 输出语义必须严格属于上面【合法语义列表】，不要造新词。\n\n"
            f"【输出格式】\n"
            f"输出严格的 JSON 对象，每个键为掩码编号（字符串），值为语义名称（字符串）。\n"
            f"示例：\n"
            f"{{\n"
            f"  \"0\": \"{legal_classes[0]}\",\n"
            f"  \"1\": \"unknown\"\n"
            f"}}\n"
            f"只输出上述 JSON，不要包裹在 ```json 代码块中，不要输出任何其他文本。"
        )
        return prompt

    # ------------------------------------------------------------------
    # MLLM 调用 + 解析（带重试）
    # ------------------------------------------------------------------
    def _call_mllm_with_retries(
        self,
        prompt: str,
        images_b64: List[str],
        required_ids: set,
        max_retries: int = 3,
        timeout: float = 90.0,
    ) -> Dict[str, str]:
        """调用 MLLM，返回 {mask_id_str: semantic_label}。失败时给 required_ids 全部填 unknown。"""
        if self.client is None:
            return {mid: "unknown" for mid in required_ids}

        content_list = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        last_err = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content_list}],
                    max_tokens=2048,
                    temperature=0.1,
                    timeout=timeout,
                )
                response_text = (response.choices[0].message.content or "").strip()
                preds = self._parse_predictions(response_text)
                if preds:
                    # 补齐缺失
                    for mid in required_ids:
                        preds.setdefault(mid, "unknown")
                    return preds
            except Exception as e:
                last_err = e
                print(f"[RESClassifier] MLLM call attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt + 1 < max_retries:
                    time.sleep(3)
        if last_err is not None:
            print(f"[RESClassifier] All retries exhausted: {last_err}; padding unknown.")
        return {mid: "unknown" for mid in required_ids}

    @staticmethod
    def _parse_predictions(text: str) -> Dict[str, str]:
        """解析 {mask_id: label} JSON。容错：剥离代码块、regex 兜底。"""
        if not text:
            return {}
        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        out: Dict[str, str] = {}
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    for k, v in data.items():
                        digit_key = k
                        if not str(k).isdigit():
                            mm = re.match(r"mask[_\s]*(\d+)", str(k), re.IGNORECASE)
                            if mm:
                                digit_key = mm.group(1)
                            else:
                                continue
                        if isinstance(v, dict):
                            v = v.get("semantic", v.get("label", "unknown"))
                        out[str(digit_key)] = str(v)
            except json.JSONDecodeError:
                pass
        if not out:
            # regex 兜底
            for k, v in re.findall(r'"(?:mask[_\s]*)?(\d+)"\s*:\s*"([^"]+)"', text, re.IGNORECASE):
                out[k] = v
        return out

    # ------------------------------------------------------------------
    # Label 规范化
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_pred(raw: str) -> str:
        """规范化预测字符串：去引号、去前缀冠词、清空白。"""
        if not isinstance(raw, str):
            return "unknown"
        s = raw.strip().strip('"').strip("'")
        s = re.sub(r"^a\s+|^an\s+|^the\s+", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s)
        if not s:
            return "unknown"
        return s

    @staticmethod
    def _match_legal_class(pred: str, legal_classes: List[str]) -> str:
        """把 MLLM 预测匹配到合法 label；找不到则归 unknown。"""
        if pred in legal_classes:
            return pred
        pred_lc = pred.lower()
        # 直接小写匹配
        for c in legal_classes:
            if c.lower() == pred_lc:
                return c
        # 子串
        for c in legal_classes:
            if c.lower() in pred_lc or pred_lc in c.lower():
                return c
        # unknown 关键字
        if "unknown" in pred_lc or "background" in pred_lc or "unlabeled" in pred_lc:
            return "unknown"
        return "unknown"
