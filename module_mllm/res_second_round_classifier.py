"""
RES 第二轮复审 MLLM 分类器（target-only 优化版）。

继承自 SecondRoundMLLMClassifier 以**复用** OpenAI 客户端、SOM 渲染等基础能力，
但**完全不修改父类**。父类原有 predict_suspect_masks 行为保持不变，
PartNetE / PartObjaverse-Tiny 路径零影响。

本类新增 1 个公开方法：
    predict_res_suspect_masks(...)
        与父类 predict_suspect_masks 接口对应，但 prompt 替换为 RES 模式：
        - 合法语义为 [target_label, distractor_labels..., "unknown"]
        - 显式提示这是指代分割的复审
        - 仅复审 first_round 标记为 target 但被拓扑判定为可疑的 mask（target-only 由调用方控制）
"""

import base64
import json
import re
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from datatypes import SoftMask2D
from module_mllm.second_round_classifier import SecondRoundMLLMClassifier


class RESSecondRoundClassifier(SecondRoundMLLMClassifier):
    """RES 第二轮复审分类器。继承复用父类基础能力，零修改父类。"""

    # ==================================================================
    def predict_res_suspect_masks(
        self,
        vid: int,
        image: np.ndarray,
        depth_map: np.ndarray,
        suspect_masks: List[Dict],
        masks_in_view: List[np.ndarray],
        view_angle: str,
        target_ku: Dict,
        distractors: List[Dict[str, str]],
        text_view_instance_knowledge: Dict,
        global_topology_text: str = "",
    ) -> Tuple[List[SoftMask2D], List[np.ndarray], List[Dict[int, str]]]:
        """
        Args:
            suspect_masks: List[{
                "mask_idx": int,
                "first_round_semantic": str,        # 通常等于 target_label
                "violations": List[str],
                "mask_height": float,
                "neighbors": List[{part, distance, relation}],
                "topology_relations": List[{part, distance, relation, ...}],
                "reason": str,
            }]
            masks_in_view: 该视角所有 mask（按全局 mask_idx 索引）
            target_ku, distractors: 与第一轮一致

        Returns:
            (corrected_soft_masks, som_images_raw, batch_decisions)
        """
        if not suspect_masks:
            return [], [], []

        if self.client is None:
            # 无 API → 全部维持 unknown
            soft = [
                SoftMask2D(mask=masks_in_view[sm["mask_idx"]],
                           probabilities={"unknown": 1.0},
                           reasoning="MLLM unavailable")
                for sm in suspect_masks
            ]
            return soft, [], [{sm["mask_idx"]: "unknown" for sm in suspect_masks}]

        target_label = target_ku.get("short_label", "target")
        distractor_labels = [d["short_label"] for d in distractors]
        legal_classes = [target_label] + distractor_labels + ["unknown"]

        # 分批
        max_per_batch = 3
        batches = [suspect_masks[i:i + max_per_batch]
                   for i in range(0, len(suspect_masks), max_per_batch)]

        all_results: List[SoftMask2D] = []
        som_raw_list: List[np.ndarray] = []
        batch_decisions: List[Dict[int, str]] = []

        for b_idx, batch in enumerate(batches):
            som_image = self._render_suspect_som(image, masks_in_view, batch, b_idx)
            som_raw_list.append(som_image.copy())
            b64_orig = self._encode_image_to_base64(image)
            b64_som = self._encode_image_to_base64(som_image)
            b64_depth = self._encode_depth_b64(depth_map)

            # 构造批次实例知识
            batch_know = {}
            for sm in batch:
                m_key = f"mask_{sm['mask_idx']}"
                if m_key in text_view_instance_knowledge:
                    batch_know[m_key] = text_view_instance_knowledge[m_key]

            prompt = self._build_res_second_round_prompt(
                view_angle=view_angle,
                target_ku=target_ku,
                distractors=distractors,
                legal_classes=legal_classes,
                global_topology_text=global_topology_text,
                batch_instance_knowledge=batch_know,
                batch_suspects=batch,
            )

            preds = self._call_mllm_with_retries(
                prompt=prompt,
                images_b64=[b64_orig, b64_som, b64_depth],
                required_ids=set(str(sm["mask_idx"]) for sm in batch),
            )

            batch_sem_map: Dict[int, str] = {}
            for sm in batch:
                m_idx = sm["mask_idx"]
                pred_raw = preds.get(str(m_idx), "unknown")
                cls = self._normalize_pred_to_legal(pred_raw, legal_classes)
                batch_sem_map[m_idx] = cls
                all_results.append(SoftMask2D(
                    mask=masks_in_view[m_idx],
                    probabilities={cls: 1.0},
                    reasoning=sm.get("reason", ""),
                ))
            batch_decisions.append(batch_sem_map)

        return all_results, som_raw_list, batch_decisions

    # ==================================================================
    @staticmethod
    def _render_suspect_som(
        image: np.ndarray,
        masks_in_view: List[np.ndarray],
        batch_suspects: List[Dict],
        b_idx: int,
    ) -> np.ndarray:
        """渲染 SOM 图：彩色半透明 + 数字 + 边界（仅这一批 suspect mask）。"""
        som = image.copy()
        h, w = image.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        rng = np.random.RandomState(42 + b_idx)
        colors = rng.randint(60, 230, size=(len(masks_in_view), 3), dtype=np.uint8)

        for sm in batch_suspects:
            m_idx = sm["mask_idx"]
            m = masks_in_view[m_idx]
            color = colors[m_idx].tolist()
            overlay[m > 0] = color

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(som, contours, -1, color, 2)

            ys, xs = np.where(m > 0)
            if len(ys) > 0:
                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                text = str(m_idx)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fs = 0.45
                th = 1
                (tw, th2), _ = cv2.getTextSize(text, font, fs, th)
                pad = 2
                rx1, ry1 = cx - tw // 2 - pad, cy - th2 // 2 - pad
                cv2.rectangle(som, (rx1, ry1),
                              (rx1 + tw + 2 * pad, ry1 + th2 + 2 * pad),
                              (0, 0, 0), -1)
                cv2.putText(som, text, (rx1 + pad, ry1 + pad + th2),
                            font, fs, (255, 255, 255), th)

        cv2.addWeighted(overlay, 0.45, som, 1 - 0.45, 0, som)
        return som

    def _encode_depth_b64(self, depth_map: np.ndarray) -> str:
        from utils.depth_visualization import enhance_depth_map
        depth_bgr = enhance_depth_map(depth_map)
        _, buf = cv2.imencode(".png", depth_bgr)
        return base64.b64encode(buf).decode("utf-8")

    # ------------------------------------------------------------------
    @staticmethod
    def _build_res_second_round_prompt(
        view_angle: str,
        target_ku: Dict,
        distractors: List[Dict[str, str]],
        legal_classes: List[str],
        global_topology_text: str,
        batch_instance_knowledge: Dict,
        batch_suspects: List[Dict],
    ) -> str:
        target_label = target_ku.get("short_label", "target")
        legal_str = ", ".join(f"\"{c}\"" for c in legal_classes)

        # suspect 信息块
        suspect_lines: List[str] = []
        for sm in batch_suspects:
            m_idx = sm["mask_idx"]
            first_sem = sm.get("first_round_semantic", "?")
            mask_h = sm.get("mask_height", "N/A")
            line = f"- 掩码 {m_idx}：第一轮预测=\"{first_sem}\"，归一化高度={mask_h}"
            relations = sm.get("topology_relations", [])[:3]
            if relations:
                line += "\n  最近部件："
                for tr in relations:
                    line += (
                        f"\n    → {tr.get('part', '?')}"
                        f"(d={tr.get('distance', 0):.4f}, {tr.get('relation', '')})"
                    )
            violations = sm.get("violations", [])
            if violations:
                line += "\n  违规："
                for v in violations:
                    line += f"\n    • {v}"
            suspect_lines.append(line)

        distractor_lines = "\n".join(
            f"  - {d['short_label']}: {d.get('brief_features', '')}"
            for d in distractors
        ) or "  (无 distractor)"

        prompt = (
            f"你是 3D 零件指代分割专家。第二轮复审：以下掩码已被 3D 拓扑规则标记为可疑，请重新判定语义。\n\n"
            f"⚠️ 本任务是 3D **指代分割**，目标是「{target_label}」。\n"
            f"被复审的掩码大多在第一轮被识别为 target，但拓扑规则发现它们的 3D 几何/位置关系不合理，"
            f"它们可能实际属于某个干扰部件，或属于无法判断的 unknown。\n\n"
            f"【输入图片】（共三张，同一视角）\n"
            f"  图1 = 无标注原始渲染图。\n"
            f"  图2 = SOM 标注图，仅高亮当前批次的可疑掩码（彩色半透明 + 数字 ID + 边界）。\n"
            f"  图3 = 深度图，越亮越近。\n"
            f"注意：每个编号对应的是与该数字颜色一致且像素连通的整片色域。\n\n"
            f"【任务上下文】\n"
            f"  当前视角：{view_angle}\n"
            f"  合法语义列表：[{legal_str}]\n\n"
            f"【指代目标 K_u】\n"
            f"  short_label: {target_label}\n"
            f"  spatial_location: {target_ku.get('spatial_location', '未提及')}\n"
            f"  typical_3D_shape: {target_ku.get('typical_3D_shape', '未提及')}\n"
            f"  connection_context: {target_ku.get('connection_context', '未提及')}\n"
            f"  relative_size: {target_ku.get('relative_size', '未提及')}\n"
            f"  z_height_range: {target_ku.get('z_height_range', '[0.0, 1.0]')}\n\n"
            f"【其他可能存在的干扰部件】\n{distractor_lines}\n\n"
            f"【实例知识库（当前批次每个可疑掩码的 3D 几何特征）】\n"
            f"{json.dumps(batch_instance_knowledge, indent=2, ensure_ascii=False)}\n\n"
            f"【全局 3D 空间拓扑关系】\n{global_topology_text}\n\n"
            f"【嫌疑掩码清单（含拓扑违规详情）】\n" + "\n".join(suspect_lines) + "\n\n"
            f"【推理与思考步骤】\n"
            f"**第一步：审视拓扑证据与违规原因**\n"
            f" - 拓扑规则的违规判断由 3D 点云算法精确计算，是比 2D 视觉更可靠的物理证据。\n"
            f" - 高度越界 / 空间倒置 / 不合理邻接 / 孤立碎片各自代表不同问题，请逐项分析。\n\n"
            f"**第二步：与目标 K_u 逐字段比对**\n"
            f" - 把该掩码的 normalized_z / shape_description / relative_dimensions 与目标 K_u 的"
            f" spatial_location / typical_3D_shape / z_height_range 比对；不匹配就排除目标语义。\n\n"
            f"**第三步：与 distractor 比对**\n"
            f" - 如果排除了目标，依次匹配 distractor 的 brief_features，找最贴近的。\n\n"
            f"**第四步：综合裁定**\n"
            f" - 注意：嫌疑检测可能存在误报，若 3D 几何特征确实与 target 相符，可维持原判 \"{target_label}\"。\n"
            f" - 实例知识库所有特征值为 0 的 mask 必须输出 \"unknown\"。\n"
            f" - 输出语义必须严格属于上面【合法语义列表】。\n\n"
            f"【输出格式】\n"
            f"输出严格的 JSON 对象。每个键为掩码编号（字符串），值为语义名称（字符串）。\n"
            f"本批次必须包含的掩码编号（缺一不可）：{[str(sm['mask_idx']) for sm in batch_suspects]}\n"
            f"示例：\n"
            f"{{\n"
            f"  \"3\": \"{target_label}\",\n"
            f"  \"7\": \"unknown\"\n"
            f"}}\n"
            f"只输出 JSON，不要包裹在 ```json 代码块中，不要输出任何其他文本。"
        )
        return prompt

    # ------------------------------------------------------------------
    def _call_mllm_with_retries(
        self,
        prompt: str,
        images_b64: List[str],
        required_ids: set,
        max_retries: int = 3,
        timeout: float = 90.0,
    ) -> Dict[str, str]:
        if self.client is None:
            return {mid: "unknown" for mid in required_ids}

        content_list = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

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
                    for mid in required_ids:
                        preds.setdefault(mid, "unknown")
                    return preds
            except Exception as e:
                print(f"[RESSecondRoundClassifier] attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt + 1 < max_retries:
                    time.sleep(3)
        return {mid: "unknown" for mid in required_ids}

    @staticmethod
    def _parse_predictions(text: str) -> Dict[str, str]:
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
            for k, v in re.findall(r'"(?:mask[_\s]*)?(\d+)"\s*:\s*"([^"]+)"', text, re.IGNORECASE):
                out[k] = v
        return out

    @staticmethod
    def _normalize_pred_to_legal(raw: str, legal: List[str]) -> str:
        if not isinstance(raw, str):
            return "unknown"
        s = raw.strip().strip('"').strip("'")
        s = re.sub(r"^a\s+|^an\s+|^the\s+", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s)
        if not s:
            return "unknown"
        if s in legal:
            return s
        s_lc = s.lower()
        for c in legal:
            if c.lower() == s_lc:
                return c
        for c in legal:
            if c.lower() in s_lc or s_lc in c.lower():
                return c
        if "unknown" in s_lc or "background" in s_lc or "unlabeled" in s_lc:
            return "unknown"
        return "unknown"
