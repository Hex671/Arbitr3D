import json
import numpy as np
import base64
import cv2
import asyncio
from typing import List, Dict, Tuple
from datatypes import SoftMask2D
from module_mllm.pose_checker import (
    adapt_global_topology_text_for_pose,
    filter_text_instance_knowledge_for_no_height,
    filter_text_instance_knowledge_for_pose,
    filter_unified_knowledge_for_pose,
    pose_is_abnormal,
)

class SecondRoundMLLMClassifier:
    def __init__(self, api_key: str, base_url: str = None, model_name: str = "gpt-4o"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            print("OpenAI package not installed. MLLM Classifier will not work.")

    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', img_bgr)
        return base64.b64encode(buffer).decode('utf-8')

    def predict_suspect_masks(
        self,
        vid: int,
        image: np.ndarray,
        depth_map: np.ndarray,
        suspect_masks: List[dict],
        masks_in_view: List[np.ndarray],
        prompt_classes: List[str],
        object_category: str,
        view_angle: str,
        unified_knowledge: dict,
        text_view_instance_knowledge: dict,
        global_topology_text: str = "",
        pose_assessment: dict = None,
        disable_height_prior: bool = False,
    ) -> List[SoftMask2D]:
        """
        第二轮 MLLM 判别
        """
        if not suspect_masks:
            return []

        print(f"    [2nd Round MLLM] Re-evaluating {len(suspect_masks)} suspect masks in view {vid}...")

        view_id_str = f"View_{vid:02d}"
        filtered_unified_knowledge = filter_unified_knowledge_for_pose(
            unified_knowledge,
            object_category,
            view_id_str,
            pose_assessment,
        )
        pose_abnormal = pose_is_abnormal(pose_assessment)
        pose_reason = ""
        if pose_assessment:
            pose_reason = str(pose_assessment.get("reasoning", "")).strip()
        global_topology_text = adapt_global_topology_text_for_pose(global_topology_text, pose_assessment)

        # 分批处理嫌疑掩码，每批最多 3 个，保证 MLLM 精准推理拓扑关系
        max_masks_per_batch = 3
        batches = [suspect_masks[i:i + max_masks_per_batch] for i in range(0, len(suspect_masks), max_masks_per_batch)]

        all_results = []
        som_images_raw = []  # 每个 batch 发送给 MLLM 的原始 SOM 图
        batch_decisions = []  # 每个 batch 中 {mask_idx: semantic} 的映射
        for b_idx, batch_suspects in enumerate(batches):
            print(f"      -> Processing batch {b_idx+1}/{len(batches)} with {len(batch_suspects)} masks...")

            # 1. 渲染只包含这几个嫌疑掩码的 SOM 图
            img_h, img_w = image.shape[:2]
            som_image = image.copy()
            overlay = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            alpha = 0.5

            # 生成随机颜色
            np.random.seed(42 + b_idx)
            colors = np.random.randint(50, 255, size=(len(masks_in_view), 3), dtype=np.uint8)

            for sm in batch_suspects:
                m_idx = sm["mask_idx"]
                m_array = masks_in_view[m_idx]
                color = colors[m_idx]

                overlay[m_array > 0] = color

                # 画数字和边界
                contours, _ = cv2.findContours(m_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(som_image, contours, -1, color.tolist(), 2)

                y_indices, x_indices = np.where(m_array > 0)
                if len(y_indices) > 0:
                    cx, cy = int(np.mean(x_indices)), int(np.mean(y_indices))
                    text = str(m_idx)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.3
                    thickness = 1
                    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    pad = 1
                    rect_w = tw + 2 * pad
                    rect_h = th + 2 * pad
                    rx1 = cx - rect_w // 2
                    ry1 = cy - rect_h // 2
                    cv2.rectangle(som_image, (rx1, ry1), (rx1 + rect_w, ry1 + rect_h), (0, 0, 0), -1)
                    cv2.putText(som_image, text, (rx1 + pad, ry1 + pad + th), font, font_scale, (255, 255, 255), thickness)

            cv2.addWeighted(overlay, alpha, som_image, 1 - alpha, 0, som_image)
            som_images_raw.append(som_image.copy())

            base64_som = self._encode_image_to_base64(som_image)
            base64_img = self._encode_image_to_base64(image)

            from utils.depth_visualization import enhance_depth_map
            depth_bgr = enhance_depth_map(depth_map)
            _, buffer_depth = cv2.imencode('.png', depth_bgr)
            base64_depth = base64.b64encode(buffer_depth).decode('utf-8')

            batch_instance_knowledge = {}
            if text_view_instance_knowledge:
                for sm in batch_suspects:
                    m_idx = sm["mask_idx"]
                    mask_key = f"mask_{m_idx}"
                    if mask_key in text_view_instance_knowledge:
                        batch_instance_knowledge[mask_key] = text_view_instance_knowledge[mask_key]
            batch_instance_knowledge = filter_text_instance_knowledge_for_pose(batch_instance_knowledge, pose_assessment)
            batch_instance_knowledge = filter_text_instance_knowledge_for_no_height(
                batch_instance_knowledge, disable_height_prior
            )

            # 提取嫌疑掩码的结构化信息（包含第一轮预测和质疑原因）
            suspect_info_text = ""
            for sm in batch_suspects:
                m_idx = sm["mask_idx"]
                first_sem = sm.get("first_round_semantic", "unknown")
                reason = sm.get("reason", "")
                violations = sm.get("violations", [])
                mask_height = sm.get("mask_height", "N/A")
                neighbors = sm.get("neighbors", [])

                if pose_abnormal or disable_height_prior:
                    suspect_info_text += f"\n- 掩码 {m_idx}：第一轮预测=\"{first_sem}\""
                else:
                    suspect_info_text += f"\n- 掩码 {m_idx}：第一轮预测=\"{first_sem}\"，归一化高度={mask_height}"
                # 最近的 3 个确认部件的空间关系
                topology_relations = sm.get("topology_relations", [])[:3]
                if topology_relations:
                    suspect_info_text += f"\n  最近部件："
                    for tr in topology_relations:
                        suspect_info_text += (
                            f"\n    → {tr['part']}(d={tr['distance']:.4f}, {tr['relation']})"
                        )
                if violations:
                    suspect_info_text += f"\n  违规："
                    for v in violations:
                        suspect_info_text += f"\n    • {v}"

            # ===== 视角引导：非常规视角时提醒 MLLM =====
            import re as _re
            view_guidance = ""
            elev_match = _re.search(r'仰角\s*([\-\d.]+)', view_angle)
            if elev_match:
                elev_val = float(elev_match.group(1))
                if pose_abnormal:
                    if elev_val < -25:
                        view_guidance = (
                            f"  ⚠️ 仰视视角警告（仰角 {elev_val:.0f}°）：你正从下方观察物体，2D 外观与常规视角差异极大。\n"
                            f"  - 当前物体姿态又是奇异摆放，画面中的上下不能直接解释为常规顶部/底部\n"
                            f"  - 此视角下必须以局部几何、深度层级和邻接关系为主要判据\n"
                        )
                    elif elev_val < -5:
                        view_guidance = (
                            f"  注意：略仰视视角（仰角 {elev_val:.0f}°），当前物体姿态奇异，请结合深度图和局部几何判断\n"
                        )
                    elif elev_val > 50:
                        view_guidance = (
                            f"  注意：俯视视角（仰角 {elev_val:.0f}°），底部部件可能被遮挡；当前姿态奇异，请避免套用常规上下直觉\n"
                        )
                else:
                    if elev_val < -25:
                        view_guidance = (
                            f"  ⚠️ 仰视视角警告（仰角 {elev_val:.0f}°）：你正从下方观察物体，2D 外观与常规视角差异极大。\n"
                            f"  - 画面中「上方」≠物体「顶部」，请严格以 normalized_z 判断真实空间位置\n"
                            f"  - 此视角下必须以 3D 几何特征为主要判据，不要被 2D 外观误导\n"
                        )
                    elif elev_val < -5:
                        view_guidance = (
                            f"  注意：略仰视视角（仰角 {elev_val:.0f}°），请结合 normalized_z 辅助判断\n"
                        )
                    elif elev_val > 50:
                        view_guidance = (
                            f"  注意：俯视视角（仰角 {elev_val:.0f}°），底部部件可能被遮挡，请结合深度图判断\n"
                        )

            pose_guidance = ""
            if pose_abnormal:
                pose_guidance = (
                    f"  ⚠️ 姿态预检结论：该点云当前处于奇异/非常规摆放姿态。\n"
                    + (f"  - 判定依据：{pose_reason}\n" if pose_reason else "")
                    + f"  - 当前全局上下仅代表当前摆放下的相对高低，不代表物体常理状态中的上下。\n"
                    + f"  - 已禁用与常规姿态强相关的知识项：view_appearance、z_height_range、spatial_location、normalized_z、spatial_centroid、mask_height、mask_height_range。\n"
                    + f"  - 若出现依赖高度语义的拓扑违规，请将其视为弱证据，不要单独据此改判。\n"
                )

            first_step_line = (
                f" - 理解违规的具体含义：高度越界说明该掩码在 3D 空间中的位置与其标注语义不符；孤立碎片说明它远离同语义的其他区域；未确认语义说明该语义从未在多视角投票中被确认过，标注可信度极低；不合理邻接说明它与不该接触的部件相接触；空间倒置说明部件间上下关系违反物理常识。\n\n"
            )
            second_step_line = (
                f" - 将掩码的【实例知识库】中的物理量（normalized_z、relative_dimensions、shape_description、principal_curvature）与【泛化知识库】中各部件的典型特征进行逐项匹配。\n"
            )
            third_step_line_1 = (
                f" - 结合【全局 3D 空间拓扑关系】中已确认部件的高度范围，判断该掩码的 normalized_z 落入哪个部件的高度区间。\n"
            )
            third_step_line_2 = (
                f" - 如果掩码的高度、邻接和相对空间关系都与某个替代语义吻合，应果断纠正为该语义。\n\n"
            )
            if pose_abnormal:
                first_step_line = (
                    f" - 理解违规的具体含义：孤立碎片、未确认语义和不合理邻接仍然是重要证据；但任何依赖常规上下语义的高度越界或空间倒置信号，在当前奇异姿态下只能视为弱提示，不能单独作为定论。\n\n"
                )
                second_step_line = (
                    f" - 将掩码的【实例知识库】中的物理量（relative_dimensions、shape_description、principal_curvature）与【泛化知识库】中各部件的典型特征进行逐项匹配。由于当前姿态奇异，不要依赖常规上下含义相关字段。\n"
                )
                third_step_line_1 = (
                    f" - 结合【全局 3D 空间拓扑关系】与深度图，理解该掩码和已确认部件之间的相对分离、邻接和连接模式；其中当前相对高度只代表当前摆放下的高低，不等于常规姿态中的顶部/底部语义。\n"
                )
                third_step_line_2 = (
                    f" - 如果掩码的局部几何、邻接和相对空间关系都与某个替代语义吻合，应果断纠正为该语义。\n\n"
                )

            prompt = (
                f"你是 3D 零件语义分割专家。第二轮复审：以下掩码已被 3D 拓扑规则标记为可疑，请重新判定语义。\n\n"
                f"【输入图片】（共三张，同一视角）\n"
                f"  图1 = 无标注原始渲染图，用于观察纯净外观。\n"
                f"  图2 = SOM 标注图，彩色半透明掩码 + 数字 ID + 边界线。\n"
                f"  图3 = 深度图，仅物体区域有灰度；越亮越近，亮暗差异反映部件间相对空间位置。\n"
                f"注意：每个编号对应的是与该数字颜色一致且像素连通的整片色域，不要只看数字附近局部。\n\n"
                f"【任务上下文】\n"
                f"  物体类别：{object_category}\n"
                f"  当前视角：{view_angle}\n"
                + (view_guidance if view_guidance else "")
                + (pose_guidance if pose_guidance else "")
                + f"  合法部件列表：{prompt_classes + ['background', 'unlabeled']}\n\n"
                + f"【泛化知识库】：\n{json.dumps(filtered_unified_knowledge, indent=2, ensure_ascii=False)}\n\n"
                + f"【实例知识库】（当前掩码的 3D 几何特征）：\n{json.dumps(batch_instance_knowledge, indent=2, ensure_ascii=False)}\n\n"
                + f"【全局 3D 空间拓扑关系】：\n{global_topology_text}\n\n"
                + f"【嫌疑掩码清单】（第一轮预测 + 拓扑违规详情）：{suspect_info_text}\n\n"
                + f"【推理与思考步骤（Standard Operating Procedure）】\n"
                + f"这是第二轮复审。以下掩码已被 3D 拓扑规则标记为可疑，请严格按照以下步骤逐一分析每个嫌疑掩码：\n\n"
                + f"**第一步：审视拓扑证据与违规原因**\n"
                + f" - 仔细阅读每个嫌疑掩码的【拓扑违规】和【与各确认部件的相对空间关系】。这些是由 3D 点云算法精确计算出的物理证据，比 2D 视觉判断更可靠。\n"
                + first_step_line
                + f"**第二步：结合视觉与几何特征交叉验证**\n"
                + f" - 观察【图1 原始渲染图】和【图2 SOM 标注图】，判断该掩码在 2D 画面中的基本形状和位置。\n"
                + f" - 结合【图3 深度图】理解该掩码在三维空间中的纵深层级和相对远近关系。\n"
                + second_step_line
                + f" - 特别注意区分容易视觉混淆的相邻部件：必须依据 shape_description 区分\"直立延伸结构\"（如侧板/竖杆，y_height 占比显著）和\"水平延伸结构\"（如横杆/平板，y_height 占比极小）。\n\n"
                + f"**第三步：基于拓扑关系推断正确语义**\n"
                + f" - 查看该掩码与各确认部件的距离和相对空间关系。距离最近且相对位置模式合理的部件往往是该掩码真正所属的语义。\n"
                + third_step_line_1
                + third_step_line_2
                + f"**第四步：识别 unlabeled 与 background**\n"
                + f" - 如果掩码的【实例知识库】中所有特征值均为 0，则该掩码必定为 \"background\"。\n"
                + f" - 如果掩码确属物体自身结构，但其 3D 几何特征（形状、位置、尺寸）与【泛化知识库】中的任何核心部件都无法匹配，果断标记为 \"unlabeled\"。不要为了避免 unlabeled 而强行归入不匹配的部件。\n"
                + f" - 典型的 unlabeled 场景：该类物体定义之外的额外结构；过分割产生的微小碎片且几何特征无法归入任何部件；多部件交界处的模糊区域。\n"
                + f" - ⚠️ **窄边框 / 壳体外框规则（重要）**：紧贴某核心部件外缘、但并非该部件功能主体的窄条 / 环形带 / 框架 / 壳体面板区域，即使视觉上与相邻部件外观连续（同色、同材质、在同一平面），也**不属于该部件本身**。典型案例：微波炉门周围的金属/塑料边框（机身外壳，不是 door）；笔记本屏幕四周的边框（不是 screen）；按钮凹槽周围的面板条（不是 button）；抽屉面板周围的柜体板（不是 drawer）。这类\"框/壳\"区域必须判定为 \"unlabeled\"，绝不能因为\"它看起来像 X 的一部分\"就强行贴 X。只有部件的**功能主体表面**（门的玻璃/反光面、屏幕的显示区、按钮的凸起顶面、抽屉正面板本身）才应贴该部件的标签。\n\n"
                + f"**第五步：综合裁定**\n"
                + f" - 注意：嫌疑检测可能存在误报。如果经过以上步骤分析后，掩码的 3D 几何特征和拓扑关系确实与第一轮预测吻合，可以维持原判。\n"
                + f" - 结合前四步的分析（无需在输出中写出推理过程），从合法部件列表中选出最准确的标签。\n\n"
                + f"【输出格式】\n"
                + f"输出严格的 JSON 对象。每个键为掩码编号（字符串），值为部件名称（字符串）。\n"
                + f"本批次必须包含的掩码编号（缺一不可）：{[str(sm['mask_idx']) for sm in batch_suspects]}\n"
                + f"示例：\n"
                + f"{{\n"
                + f"  \"3\": \"leg\",\n"
                + f"  \"7\": \"background\"\n"
                + f"}}\n"
                + f"只输出 JSON，不要包裹在 ```json 代码块中，不要输出任何其他文本。"
            )

            import re
            max_retries = 3
            batch_success = False

            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_som}"}},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_depth}"}},
                                ],
                            }
                        ],
                        max_tokens=2048,
                        temperature=0.1,
                        timeout=90.0
                    )
                    text = (response.choices[0].message.content or "").strip()
                    if attempt == 0:
                        print(f"      [Debug] MLLM Output:\n{text[:500]}...") # 只打印前 500 个字符

                    # 尝试剥离代码块
                    if text.startswith("```json"):
                        text = text[7:]
                    elif text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()

                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                    else:
                        data = json.loads(text)

                    batch_results = []
                    batch_sem_map = {}  # 本批次的 mask_idx -> semantic
                    for sm in batch_suspects:
                        m_idx = sm["mask_idx"]
                        raw_val = data.get(str(m_idx), "background")
                        # 支持结构化格式 {"semantic": "leg"} 和简单格式 "leg"
                        if isinstance(raw_val, dict):
                            sem = raw_val.get("semantic", "background")
                        else:
                            sem = raw_val
                        batch_sem_map[m_idx] = sem
                        classes = prompt_classes + ["unlabeled", "background"]
                        matched_cls = None
                        if sem in classes:
                            matched_cls = sem
                        else:
                            for cls in classes:
                                if cls in sem or sem in cls:
                                    matched_cls = cls
                                    break
                            if matched_cls is None:
                                matched_cls = "background"
                        prob_dict = {matched_cls: 1.0}
                        batch_results.append(SoftMask2D(mask=masks_in_view[m_idx], probabilities=prob_dict, reasoning=sm.get("reason", "")))

                    all_results.extend(batch_results)
                    batch_decisions.append(batch_sem_map)
                    batch_success = True
                    break
                except Exception as e:
                    print(f"      2nd round MLLM Failed ({attempt+1}/{max_retries}): {e}")

            if not batch_success:
                print("      Batch completely failed, padding with original (background) predictions.")
                batch_sem_map = {}
                for sm in batch_suspects:
                    m_idx = sm["mask_idx"]
                    prob_dict = {"background": 1.0}
                    all_results.append(SoftMask2D(mask=masks_in_view[m_idx], probabilities=prob_dict, reasoning="MLLM failed to parse"))
                    batch_sem_map[m_idx] = "background"
                batch_decisions.append(batch_sem_map)

        return all_results, som_images_raw, batch_decisions
