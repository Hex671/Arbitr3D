import json
import os
import re
import base64
import cv2
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from datatypes import SoftMask2D, MaskPrediction, FeatureDescriptor, ViewInstanceKnowledge, PartFeature, InstanceKnowledge
from module_mllm.pose_checker import (
    filter_text_instance_knowledge_for_no_height,
    filter_text_instance_knowledge_for_pose,
    filter_unified_knowledge_for_pose,
    pose_is_abnormal,
)
from module_2d_fm.semantic_mask_grouping import (
    SemanticGroupingResult,
    build_semantic_grouping,
    render_semantic_grouped_overlay,
    format_legend_for_prompt,
    semantic_string_to_display_id,
    fragments_for_semantic_ordered,
)

class MLLMClassifier:
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
            print("OpenAI package not installed. MLLMClassifier will not work.")

    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        import io
        import cv2
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', img_bgr)
        return base64.b64encode(buffer).decode('utf-8')

    # ---> 修改：传入 Step1 的解剖学知识 和 Step2 的当前视角特征 <---
    def predict_soft_probabilities(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        masks: List[np.ndarray],
        prompt_classes: List[str],
        object_category: str,
        view_angle: str,
        unified_knowledge: dict,
        text_view_instance_knowledge: dict,
        pose_assessment: dict = None,
        skip_large_mask_filter: bool = False,
        pre_defined_semantics: List[str] = None
    ) -> tuple[List[SoftMask2D], List[np.ndarray], List[List[int]], dict, List[np.ndarray]]:

        # 处理合并后的单张掩码（Step 2.5 合并同语义掩码后变成单个 ndarray）
        if isinstance(masks, np.ndarray):
            # 如果是单个 ndarray，转为列表，并标记跳过过滤
            masks = [masks]
            skip_large_mask_filter = True
        elif not masks:
            return [], [], [], {}, []

        # Step 2.5 合并后的掩码：直接使用预定义语义，跳过 MLLM 调用
        if pre_defined_semantics is not None and len(pre_defined_semantics) == len(masks):
            print(f"[predict_soft_probabilities] Using pre-defined semantics for {len(masks)} merged masks (skipping MLLM).")
            soft_masks_list = []
            for i, mask in enumerate(masks):
                sem = pre_defined_semantics[i]
                classes = prompt_classes + ["unlabeled", "background"]
                if sem in classes:
                    matched_cls = sem
                elif "unlabeled" in sem.lower():
                    matched_cls = "unlabeled"
                elif "background" in sem.lower():
                    matched_cls = "background"
                else:
                    matched_cls = "background"
                prob_dict = {matched_cls: 1.0}
                soft_masks_list.append(SoftMask2D(
                    mask=mask,
                    probabilities=prob_dict,
                    reasoning=""
                ))
            return soft_masks_list, [], [], {}, []
        if self.client is None:
            print("Warning: OpenAI client not initialized. Returning empty probabilities.")
            return (
                [
                    SoftMask2D(
                        mask=m,
                        probabilities={cls: 1.0 / len(prompt_classes) for cls in prompt_classes},
                    )
                    for m in masks
                ],
                [],
                [],
                {},
                [],
            )

        classes = prompt_classes + ["unlabeled", "background"]
        all_predictions = {}

        # ==========================================
        # 核心算法：大掩码过滤 + 无冲突图着色分组 (Graph Coloring)
        # ==========================================
        num_masks = len(masks)
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        valid_indices = []
        mask_areas = []
        bboxes = []

        for i, m in enumerate(masks):
            area = np.sum(m)
            mask_areas.append(area)
            y_indices, x_indices = np.where(m)
            if len(y_indices) > 0:
                bboxes.append({
                    'ymin': np.min(y_indices), 'ymax': np.max(y_indices),
                    'xmin': np.min(x_indices), 'xmax': np.max(x_indices)
                })
            else:
                bboxes.append({'ymin': 0, 'ymax': 0, 'xmin': 0, 'xmax': 0})

        # === 新增逻辑：找出面积最大的掩码索引 ===
        largest_mask_idx = -1
        if mask_areas and not skip_large_mask_filter:
            largest_mask_idx = np.argmax(mask_areas)

        # 1. 过滤掉无意义的大掩码
        for i in range(num_masks):
            # === 新增逻辑：无条件过滤面积最大的那个完整物体掩码 ===
            if not skip_large_mask_filter and i == largest_mask_idx:
                continue

            is_valid = True

            # === 新增逻辑：过滤掉没有任何三维点映射的掩码（避免浪费 Token） ===
            mask_key = f"mask_{i}"
            if text_view_instance_knowledge and mask_key in text_view_instance_knowledge:
                feat = text_view_instance_knowledge[mask_key]
                pc_zero = str(feat.get("principal_curvature", "")).startswith("0.0000")
                point_zero = str(feat.get("point_ratio", "")).startswith("0.0000")
                norm_val = feat.get("normalized_z", None)
                norm_zero_or_missing = (norm_val is None) or str(norm_val).startswith("0.0000")
                if pc_zero and point_zero and norm_zero_or_missing:
                    is_valid = False

            # 如果面积超过 50%，检查它是否是一个"父级无用容器" (保留原有兜底逻辑)
            if is_valid and mask_areas[i] > 0.5 * img_area:
                for j in range(num_masks):
                    # 跳过最大掩码的比较，因为它已经被剔除了
                    if i != j and j != largest_mask_idx and mask_areas[j] > 0:
                        b1, b2 = bboxes[i], bboxes[j]
                        # 只有 bbox 发生包含关系时，才计算精确的 IoM
                        if not (b1['xmax'] < b2['xmin'] or b1['xmin'] > b2['xmax'] or
                                b1['ymax'] < b2['ymin'] or b1['ymin'] > b2['ymax']):
                            intersection = np.logical_and(masks[i], masks[j])
                            # 如果 j 的 90% 以上面积都在 i 内部，说明 i 是包容性大背景
                            if np.sum(intersection) / mask_areas[j] > 0.9:
                                is_valid = False
                                break

            if is_valid:
                valid_indices.append(i)

        # 2. 构建冲突矩阵 (仅针对有效掩码，跳过被过滤的废弃大掩码)
        num_valid = len(valid_indices)
        conflict_matrix = np.zeros((num_valid, num_valid), dtype=bool)

        for i in range(num_valid):
            idx_i = valid_indices[i]
            for j in range(i + 1, num_valid):
                idx_j = valid_indices[j]
                b1, b2 = bboxes[idx_i], bboxes[idx_j]
                # 快速 bbox 相交测试
                if not (b1['xmax'] < b2['xmin'] or b1['xmin'] > b2['xmax'] or
                        b1['ymax'] < b2['ymin'] or b1['ymin'] > b2['ymax']):
                    # 精确像素相交测试 (只要有1个像素重叠即视为有冲突，绝不放在同张图里)
                    intersection = np.logical_and(masks[idx_i], masks[idx_j])
                    if np.any(intersection):
                        conflict_matrix[i, j] = True
                        conflict_matrix[j, i] = True

        # 3. 贪心图着色算法进行分组
        # batches_valid 存储有效掩码在 valid_indices 中的局部索引
        batches_valid = []
        max_masks_per_batch = 5 # 每张图最多贴 6 个标签，防止输出截断

        # 优化技巧：按面积从小到大排序。优先把细小掩码（如把手）排进空位，大掩码往后放
        sorted_valid_order = np.argsort([mask_areas[valid_indices[i]] for i in range(num_valid)])

        for i in sorted_valid_order:
            placed = False
            for batch in batches_valid:
                if len(batch) >= max_masks_per_batch:
                    continue
                # 检查当前待放入的掩码 i 与该 batch 内所有已有掩码是否有冲突
                has_conflict = False
                for j in batch:
                    if conflict_matrix[i, j]:
                        has_conflict = True
                        break

                if not has_conflict:
                    batch.append(i)
                    placed = True
                    break

            # 如果跟现存的所有 Batch 都有冲突，或者大家都没位置了，开辟一张新图
            if not placed:
                batches_valid.append([i])

        num_batches = len(batches_valid)

        # 4. 转换回全局 indices 和 masks
        batch_global_indices = []
        batches = []
        for b_val in batches_valid:
            global_batch = [valid_indices[local_idx] for local_idx in b_val]
            batch_global_indices.append(global_batch)
            batches.append([masks[g_idx] for g_idx in global_batch])

        som_images_raw: List[np.ndarray] = []  # 收集发送给 MLLM 的原始 SOM 图
        for batch_idx in range(num_batches):
            batch_masks = batches[batch_idx]
            global_indices = batch_global_indices[batch_idx]
            current_batch_indices = global_indices  # 当前批次涉及的全局掩码索引，供兜底逻辑使用

            som_image, _ = self._render_som_image(image, batch_masks, global_indices)
            som_images_raw.append(som_image.copy())
            base64_image = self._encode_image_to_base64(som_image)

            # 过滤实例知识库，只保留当前批次的掩码，防止 prompt 污染
            batch_instance_knowledge = {}
            if text_view_instance_knowledge:
                for global_id in global_indices:
                    mask_key = f"mask_{global_id}"
                    if mask_key in text_view_instance_knowledge:
                        batch_instance_knowledge[mask_key] = text_view_instance_knowledge[mask_key]

            # 泛化知识库已由调用方按视角过滤，直接使用
            filtered_unified_knowledge = unified_knowledge if unified_knowledge else {}
            pose_abnormal = pose_is_abnormal(pose_assessment)
            pose_reason = ""
            if pose_assessment:
                pose_reason = str(pose_assessment.get("reasoning", "")).strip()

            # ===== 视角引导：非常规视角时提醒 MLLM =====
            view_guidance = ""
            elev_match = re.search(r'仰角\s*([\-\d.]+)', view_angle)
            if elev_match:
                elev_val = float(elev_match.group(1))
                if pose_abnormal:
                    if elev_val < -25:
                        view_guidance = (
                            f"  ⚠️ 仰视视角警告（仰角 {elev_val:.0f}°）：你正从下方观察物体，2D 外观与常规视角差异极大。\n"
                            f"  - 物体底部结构直接面向相机，顶部结构被遮挡或透视缩短\n"
                            f"  - 当前物体又处于奇异摆放姿态，不要把画面中的上下直接解释为常规顶部/底部\n"
                        )
                    elif elev_val < -5:
                        view_guidance = (
                            f"  注意：当前为略仰视视角（仰角 {elev_val:.0f}°），物体底部部分可见。\n"
                            f"  - 当前物体姿态奇异，请优先依赖深度差异、局部形状和连接关系\n"
                        )
                    elif elev_val > 50:
                        view_guidance = (
                            f"  注意：当前为俯视视角（仰角 {elev_val:.0f}°），物体顶部结构突出，底部可能被遮挡。\n"
                            f"  - 当前物体姿态奇异，请结合深度图和局部几何判断，不要套用常规朝向直觉\n"
                        )
                else:
                    if elev_val < -25:
                        view_guidance = (
                            f"  ⚠️ 仰视视角警告（仰角 {elev_val:.0f}°）：你正从下方观察物体，2D 外观与常规视角差异极大。\n"
                            f"  - 物体底部结构直接面向相机，顶部结构被遮挡或透视缩短\n"
                            f"  - 画面中「上方」的区域≠物体的「顶部」，请严格以 normalized_z 判断部件的真实空间位置\n"
                        )
                    elif elev_val < -5:
                        view_guidance = (
                            f"  注意：当前为略仰视视角（仰角 {elev_val:.0f}°），物体底部部分可见。\n"
                            f"  - 部件的 2D 外观与正视角有差异，请结合 normalized_z 辅助判断\n"
                        )
                    elif elev_val > 50:
                        view_guidance = (
                            f"  注意：当前为俯视视角（仰角 {elev_val:.0f}°），物体顶部结构突出，底部可能被遮挡。\n"
                            f"  - 请结合深度图区分被遮挡的底部部件\n"
                        )

            pose_guidance = ""
            if pose_abnormal:
                pose_guidance = (
                    f"  ⚠️ 姿态预检结论：该点云当前处于奇异/非常规摆放姿态。\n"
                    + (f"  - 判定依据：{pose_reason}\n" if pose_reason else "")
                    + f"  - 当前画面中的上下方向不等于物体常理摆放时的上下方向。\n"
                    + f"  - 已禁用与常规姿态强相关的知识项：view_appearance、z_height_range、spatial_location、normalized_z、spatial_centroid。\n"
                    + f"  - 请优先依据纯视觉外观、深度层级、relative_dimensions、shape_description、principal_curvature、局部邻接与连接关系判断语义。\n"
                )

            second_step_line_1 = (
                f" - 思考当前视角下通常能看到{object_category}的什么部件，思考时参考【泛化知识库】的知识。如果一个掩码的形状跟其中有的部件很像，请结合第一步的感知与分析思考该掩码周围的部件是否合理？该掩码在【实例知识库】的信息（如normalized_z、relative_dimensions）能否对的上？如果对的上就归入对应部件。\n"
            )
            second_step_line_2 = (
                f" - 引入【空间拓扑关系与物理常识推理】：绝不能仅凭形状相似就下结论，必须考虑部件之间的空间上下、支撑与连接关系（利用 normalized_z 验证相对高度）。例如：支撑类部件（如腿/底座）必须在最下方接触地面；扶手/侧边等辅助部件绝对不可能出现在主承托面（如坐垫/桌面）下方；靠背/显示器等立面通常位于主体后方或上方。\n"
            )
            second_step_line_3 = (
                f" - 如果对不上就从实例知识库(尤其是normalized_z 和 relative_dimensions)或者深度图中学习到各部件间的三维空间结构分析该掩码可能属于什么部件，同时结合二维视觉感知判断是否属于该部件。\n"
            )
            third_step_line_1 = (
                f" - 碎片融合：如果当前掩码只是物体的一个小碎片（过分割），请判断其空间位置和几何特征最贴合哪个核心大部件(要结合深度图中的三维空间关系，以及【实例知识库中 relative_dimensions, principal_curvature, normalized_z 等信息】)，只要它依附在该主体上且特征吻合，直接归入该核心部件。\n"
            )
            if pose_abnormal:
                second_step_line_1 = (
                    f" - 思考当前视角下通常能看到{object_category}的什么部件时，只能把【泛化知识库】当作弱语义先验。由于当前物体姿态奇异，view_appearance、z_height_range 和 spatial_location 已被禁用，不能再依赖常规摆放下的上下与视角外观经验。\n"
                )
                second_step_line_2 = (
                    f" - 引入【空间拓扑关系与物理常识推理】：绝不能仅凭形状相似就下结论，但在本对象姿态奇异时，不要把当前全局上下直接对应到常规语义上下；应更重视部件间连接、支撑接触、局部形状、深度层级与相邻部件关系。\n"
                )
                second_step_line_3 = (
                    f" - 如果对不上就从实例知识库(尤其是 relative_dimensions)或者深度图中学习到各部件间的三维空间结构，分析该掩码可能属于什么部件，同时结合二维视觉感知判断是否属于该部件。\n"
                )
                third_step_line_1 = (
                    f" - 碎片融合：如果当前掩码只是物体的一个小碎片（过分割），请判断其局部几何特征最贴合哪个核心大部件(要结合深度图中的三维空间关系，以及【实例知识库中 relative_dimensions、principal_curvature 等信息】)，只要它依附在该主体上且特征吻合，直接归入该核心部件。\n"
                )

            # ===== Prompt v2: 精简结构化 + 带推理输出 =====
            prompt = (
                f"你是 3D 零件语义分割专家。请判断图中标有编号的每个掩码区域属于哪个预定义部件。\n\n"
                f"【输入图片】（共三张，同一视角）：\n"
                f"  图1 = 无标注的原始渲染图（用于观察纯净外观）\n"
                f"  图2 = SOM 标注图（彩色半透明掩码 + 数字 ID + 边界线）\n"
                f"  图3 = 深度图（仅物体区域有灰度，背景纯黑；离相机越近越亮，部件间的亮暗差异反映部件在空间中的相对位置）\n"
                f"注意：每个编号对应的是与该数字**颜色一致且像素连通的整片色域**，不要只看数字旁边的局部。\n\n"
                f"【任务上下文】\n"
                f"  物体类别：{object_category}\n"
                f"  当前视角：{view_angle}\n"
                + (view_guidance if view_guidance else "")
                + (pose_guidance if pose_guidance else "")
                + f"  核心部件列表：{prompt_classes}\n\n"
                f"【泛化知识库】（从 {object_category} 类别提炼的通用先验）：\n"
                f"{json.dumps(filtered_unified_knowledge, indent=2, ensure_ascii=False)}\n\n"
                f"【实例知识库】（当前掩码基于 3D 点云计算的物理几何特征）：\n"
                f"{json.dumps(batch_instance_knowledge, indent=2, ensure_ascii=False)}\n\n"
                f"【推理与思考步骤（Standard Operating Procedure）】\n"
                f"在给出最终语义标签前，请严格按照以下步骤对每个掩码进行分析：\n"
                f"**第一步：初始视觉感知与深度解析**\n"
                f" - 观察【图1】和【图2】，判断该掩码在 2D 画面中的基本形状、位置以及大致结构。\n"
                f" - 务必结合【图3 深度图】，观察该掩码所在区域与周围其他部件的明暗差异来判断深度差异，通过这个深度差异来学习该物体的三维空间结构。这对于理解物体的整体三维结构和掩码间的物理邻接关系至关重要。\n\n"
                f"**第二步：基于当前视角特征和 3D 几何特征消除 2D 视觉偏见（核心步骤）**\n"
                + second_step_line_1
                + second_step_line_2
                + f" - 特别注意区分视觉上相邻且深度相近的部件（经常发生视觉混淆）。此时必须依据【实例知识库】中的 relative_dimensions (尺寸比例) 和 shape_description (形状描述)：\n"
                + f"   * 区分“直立结构”与“水平结构”：直立部件（如靠背的侧边、桌腿）的 shape_description 会显示为“细长直立杆”或“竖直侧板” (y_height 占比不小)；而水平部件（如扶手、桌面、横杠）会显示为“水平宽平板”或“水平长条” (y_height 占比极小)。\n"
                + f"   * 即使是长条形部件被切碎的局部掩码，其三维形状的延伸方向特征依然有效。必须严格信任 shape_description！\n"
                + second_step_line_3
                + f" - 处理的数据中存大量未标注的点，所以遇到怎么思考都觉得与任何一个部件都不太对的掩码时，不要犹豫，果断标记为 \"unlabeled\"。\n\n"
                + f"**第三步：处理过分割与尺度异常**\n"
                + third_step_line_1
                + f" - 掩码覆盖：若某掩码面积较大并且其内部大部分位置被其他明显独立的部件占据，优先将其判定为被包裹的“核心主体结构”或\"background\"。\n\n"
                + f"**第四步：综合裁定与标签规范**\n"
                + f" - 结合前三步的思考（无需在输出中写出推理过程），从【核心部件列表】中选出最贴切的精准词汇。\n"
                + f" - 若掩码明显属于图像边缘背景、环境或多部件严重融合导致的无效掩码，判定为 \"background\"。\n"
                + f" - 若确信属于物体自身结构，但完全不匹配任何给定核心类别，判定为 \"unlabeled\"。\n"
                + f" - ⚠️ **窄边框 / 壳体外框规则（重要）**：紧贴某核心部件外缘、但并非该部件功能主体的窄条 / 环形带 / 框架 / 壳体面板区域，即使视觉上与相邻部件外观连续（同色、同材质、在同一平面），也**不属于该部件本身**。典型案例：微波炉门周围的金属/塑料边框（机身外壳，不是 door）；笔记本屏幕四周的边框（不是 screen）；按钮凹槽周围的面板条（不是 button）；抽屉面板周围的柜体板（不是 drawer）。这类\"框/壳\"区域必须判定为 \"unlabeled\"，绝不能因为\"它看起来像 X 的一部分\"就强行贴 X。只有部件的**功能主体表面**（门的玻璃/反光面、屏幕的显示区、按钮的凸起顶面、抽屉正面板本身）才应贴该部件的标签。\n\n"
                + f"**额外规则**:若一个掩码的【实例知识库】中的所有特征的值都为0，那么该掩码必定为\"background\"。\n\n"
                + "【输出格式】\n"
                + "输出一个严格的 JSON 对象。每个键为掩码编号（字符串），值为部件名称（字符串）。\n"
                + f"本批次必须包含的掩码编号（缺一不可）：{[str(g) for g in global_indices]}\n"
                + "示例：\n"
                + "{\n"
                + "  \"0\": \"seat\",\n"
                + "  \"1\": \"back\",\n"
                + "  \"2\": \"background\"\n"
                + "}\n"
                + "只输出上述 JSON，不要包裹在 ```json 代码块中，不要输出任何其他文本。"
            )

            # 编码原始渲染图和深度图
            base64_image_original = self._encode_image_to_base64(image)

            # 将深度图转为高对比度可视化图像
            from utils.depth_visualization import enhance_depth_map
            depth_map_bgr = enhance_depth_map(depth_map)
            _, buffer_depth = cv2.imencode('.png', depth_map_bgr)
            base64_image_depth = base64.b64encode(buffer_depth).decode('utf-8')

            import time # 确保引入了 time 模块

            max_retries = 5 # 设定最大重试次数（首次失败后最多再试4次，总共5次）
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    # 增加 timeout=90 参数，超过 90 秒收不到任何字节直接抛出 Timeout 异常
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_image_original}"
                                        },
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_image}"
                                        },
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_image_depth}"
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=4096,
                        temperature=0.1,
                        timeout=90.0  # <--- 超时限制提升至 90 秒
                    )
                    response_text = response.choices[0].message.content or ""

                    required_ids = {str(g) for g in current_batch_indices}

                    # ===== 解析器 v2：优先 JSON 全量解析（支持结构化输出），regex 兜底 =====
                    parsed_ok = False
                    clean_text = response_text.strip()
                    # 剥离可能的 ```json ... ``` 代码块包裹
                    if clean_text.startswith("```"):
                        clean_text = re.sub(r'^```(?:json)?\s*', '', clean_text)
                        clean_text = re.sub(r'\s*```$', '', clean_text)

                    # 尝试 1：完整 JSON 解析（支持 {"0": {"semantic": "seat", "reason": "..."}} 格式）
                    json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
                    if json_match:
                        try:
                            data = json.loads(json_match.group(0))
                            for k, v in data.items():
                                # 提取纯数字 key: 支持 "4" 和 "mask_4" 两种格式
                                digit_key = k
                                if not k.isdigit():
                                    m = re.match(r'mask[_\s]*(\d+)', k, re.IGNORECASE)
                                    if m:
                                        digit_key = m.group(1)
                                    else:
                                        continue
                                if isinstance(v, dict):
                                    # 结构化格式: {"semantic": "seat", "reason": "..."}
                                    all_predictions[digit_key] = v
                                elif isinstance(v, str):
                                    # 简单格式兼容: {"0": "seat"}
                                    all_predictions[digit_key] = v
                            if any(k.isdigit() for k in all_predictions if k != '_raw_response_'):
                                parsed_ok = True
                                all_predictions['_raw_response_'] = response_text
                        except json.JSONDecodeError:
                            pass

                    # 尝试 2：regex 兜底（捕获 "0": "seat" 或 "mask_0": "seat" 键值对）
                    if not parsed_ok:
                        matches = re.findall(r'"(?:mask[_\s]*)?(\d+)"\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
                        if matches:
                            for k, v in matches:
                                all_predictions[k] = v
                            all_predictions['_raw_response_'] = response_text
                            parsed_ok = True

                    if not parsed_ok:
                        print(f"[Batch {batch_idx}] ⚠️ 未能在返回中找到有效 JSON 数据，准备重试...")
                        raise ValueError(f"JSON 解析失败，仅有原始文本: {response_text[:200]}")

                    # 检查缺失的掩码编号
                    missing_ids = sorted(
                        (i for i in required_ids if i not in all_predictions),
                        key=lambda x: int(x),
                    )
                    if missing_ids:
                        # 缺失超过一半 → 视为响应严重残缺，重试
                        if len(missing_ids) > len(required_ids) / 2:
                            for gid in current_batch_indices:
                                sk = str(gid)
                                if sk in all_predictions:
                                    del all_predictions[sk]
                            if "_raw_response_" in all_predictions:
                                del all_predictions["_raw_response_"]
                            raise ValueError(
                                f"Incomplete batch predictions (>{len(required_ids)//2} missing), missing mask ids: {missing_ids}"
                            )
                        # 少量缺失 → 补 background，保留已有的有效预测
                        print(f"[Batch {batch_idx}] ⚠️ 模型漏掉了 {len(missing_ids)} 个掩码 {missing_ids}，自动补为 background")
                        for mid in missing_ids:
                            all_predictions[mid] = "background"

                    success = True

                except Exception as e:
                    retry_count += 1
                    err_msg = str(e)
                    print(f"\n[Batch {batch_idx}] ❌ MLLM API 调用失败 (尝试 {retry_count}/{max_retries}): {err_msg}")

                    if retry_count < max_retries:
                        print(f"等待 5 秒后准备重新发起请求...")
                        time.sleep(5)
                    else:
                        # 重试耗尽后，兜底策略：该批次全部标记为 background，继续运行防止阻断全流程
                        print(f"[Batch {batch_idx}] ⚠️ API 重试耗尽，将该批次 {len(current_batch_indices)} 个掩码全部标记为 background")
                        for mask_id in current_batch_indices:
                            all_predictions[str(mask_id)] = "background"

        soft_masks_list = []
        raw_predictions = {}  # 用于存储带推理的预测结果，供后续阶段 B 使用
        for i in range(len(masks)):
            idx_str = str(i)
            pred_data = all_predictions.get(idx_str)

            # 处理结构化预测结果（包含 semantic, reason/reasoning）
            if isinstance(pred_data, dict):
                predicted_cls = pred_data.get("semantic", "background")
                reasoning = pred_data.get("reason", "") or pred_data.get("reasoning", "")
                reasoning_source = "unknown"
                confidence = 0.5
            else:
                # 兜底：处理简单字符串格式的预测
                predicted_cls = pred_data if pred_data else "background"
                reasoning = ""
                reasoning_source = "unknown"
                confidence = 0.5

            # 确定最终匹配的类别（精确匹配 → 子串匹配 → 兜底 background）
            matched_cls = None
            if predicted_cls in classes:
                matched_cls = predicted_cls
            elif "unlabeled" in predicted_cls.lower():
                matched_cls = "unlabeled"
            elif "background" in predicted_cls.lower():
                matched_cls = "background"
            else:
                for cls in classes:
                    if cls in predicted_cls or predicted_cls in cls:
                        matched_cls = cls
                        break
                if matched_cls is None:
                    matched_cls = "background"

            prob_dict = {matched_cls: 1.0}

            # 存储 SoftMask2D，同时更新概率分布
            soft_mask_obj = SoftMask2D(
                mask=masks[i],
                probabilities=prob_dict,
                reasoning=reasoning
            )
            soft_masks_list.append(soft_mask_obj)

            # 同时存储带推理的预测结果（用于后续阶段 B）
            raw_predictions[i] = {
                "mask_id": i,
                "semantic": predicted_cls,
                "reasoning": reasoning,
            }

        som_images_labeled: List[np.ndarray] = []
        for b_idx in range(num_batches):
            batch_masks = batches[b_idx]
            gidx = batch_global_indices[b_idx]
            labels = []
            for gi in gidx:
                pred = all_predictions.get(str(gi), "background")
                labels.append(pred.get("semantic", "background") if isinstance(pred, dict) else pred)
            som_labeled, _ = self._render_som_image(
                image, batch_masks, gidx, mask_pred_labels=labels
            )
            som_images_labeled.append(som_labeled)

        return soft_masks_list, som_images_labeled, batch_global_indices, raw_predictions, som_images_raw

    def refine_view_guidance(
        self,
        images_rgb: List[np.ndarray],
        view_angles_str: List[str],
        generic_view_guidance: Dict[str, str],
        prompt_classes: List[str],
        object_category: str,
        max_workers: int = 4,
    ) -> Dict[str, str]:
        """
        阶段 1.6：使用无掩码渲染图动态修正泛化视角知识库。

        将多视角渲染图两两配对并行发送给 MLLM，让 MLLM 基于实际观测
        判断哪些泛化视角知识是正确的、哪些需要修正。

        Args:
            images_rgb: 各视角的渲染 RGB 图
            view_angles_str: 各视角的物理角度描述
            generic_view_guidance: 从知识库加载的泛化视角知识 (key=角度描述, value=预期特征)
            prompt_classes: 预定义部件列表
            object_category: 物体类别
            max_workers: 并行调用数

        Returns:
            Dict[str, str]: 修正后的视角知识 (key=角度描述, value=修正后特征)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        num_views = len(images_rgb)
        # 两视角一组（与 pipeline 日志 "2 views per call" 一致）
        views_per_call = 2

        def _refine_one_pair(start_vid: int) -> List[Tuple[int, str]]:
            results = []
            # 构造该对的端点描述
            end_vid = min(start_vid + views_per_call, num_views)
            selected_views = list(range(start_vid, end_vid))

            content_list: List[dict] = []
            captions = []

            for vid in selected_views:
                img = images_rgb[vid]
                b64 = self._encode_image_to_base64(img)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                })
                angle_desc = view_angles_str[vid] if vid < len(view_angles_str) else f"视角 {vid}"
                generic_hint = generic_view_guidance.get(angle_desc, "（无泛化知识）")
                # 将复杂对象格式转换为纯文本描述
                if isinstance(generic_hint, dict):
                    highly_visible = generic_hint.get("highly_visible_parts", [])
                    occluded = generic_hint.get("occluded_parts", [])
                    shape_exp = generic_hint.get("2d_shape_expectations", "")
                    generic_hint = f"可见部件: {highly_visible}，遮挡部件: {occluded}，形状预期: {shape_exp}"
                captions.append(f"视角 {vid}（{angle_desc}）\n泛化知识参考：{generic_hint}")

            caption_block = "\n".join(captions)

            prompt_text = f"""你是一位 3D 部件分割专家。
当前物体类别: {object_category}
预定义部件列表: {prompt_classes}

【配图说明】
以下是 {len(selected_views)} 个不同视角下的无掩码渲染图（按顺序对应图1、图2…）：
{caption_block}

【任务】
请基于图中观测，评估并修正每个视角的预期特征描述。只描述图中最显著的特征即可。

请以严格的 JSON 格式返回结果（只需一个字符串描述）：
{{
  "view_0": "该视角下的主要部件和几何特征描述...",
  "view_1": "该视角下的主要部件和几何特征描述..."
}}
"""
            content_list.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content_list}]

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2048,
                )
                raw_text = response.choices[0].message.content or "{}"
                # 去掉 markdown 代码块
                raw_text = re.sub(r"```json\s*", "", raw_text)
                raw_text = re.sub(r"```\s*", "", raw_text).strip()
                parsed = json.loads(raw_text)
                for key, val in parsed.items():
                    # 统一返回字符串格式
                    if isinstance(val, dict):
                        # 如果返回了复杂对象，转换为纯文本描述
                        highly_visible = val.get("highly_visible_parts", [])
                        occluded = val.get("occluded_parts", [])
                        shape_exp = val.get("2d_shape_expectations", "")
                        desc = f"可见部件: {highly_visible}，遮挡部件: {occluded}，形状预期: {shape_exp}"
                    else:
                        desc = str(val)
                    results.append((int(key.split("_")[1]), desc))
            except Exception as exc:
                print(f"[refine_view_guidance] Pair {selected_views} failed: {exc}")
                # 回退：保留泛化知识
                for vid in selected_views:
                    angle = view_angles_str[vid] if vid < len(view_angles_str) else f"视角 {vid}"
                    hint = generic_view_guidance.get(angle, "")
                    if isinstance(hint, dict):
                        highly_visible = hint.get("highly_visible_parts", [])
                        occluded = hint.get("occluded_parts", [])
                        shape_exp = hint.get("2d_shape_expectations", "")
                        hint = f"可见部件: {highly_visible}，遮挡部件: {occluded}，形状预期: {shape_exp}"
                    results.append((vid, hint))

            return results

        # 并行执行所有视角对
        refined: Dict[str, str] = {}

        # ── 修正前先用泛化知识初始化兜底，确保所有视角都有知识 ───────────────
        # 这样即使某视角 MLLM 修正失败或未被覆盖，也保留原始泛化知识
        for angle in view_angles_str:
            if angle in generic_view_guidance:
                hint = generic_view_guidance[angle]
                if isinstance(hint, dict):
                    # 字典格式（cot_view_guidance.json）→ 转换为纯文本字符串
                    highly_visible = hint.get("highly_visible_parts", [])
                    occluded = hint.get("occluded_parts", [])
                    shape_exp = hint.get("2d_shape_expectations", "")
                    refined[angle] = f"可见部件: {highly_visible}，遮挡部件: {occluded}，形状预期: {shape_exp}"
                else:
                    # 已经是字符串格式
                    refined[angle] = str(hint)
            else:
                refined[angle] = "无特定预期信息，请结合解剖学常识和 2D 图像进行合理判断。"

        # ── 并行执行 MLLM 修正（只覆盖成功返回的视角）───────────────────────
        tasks = list(range(0, num_views, views_per_call))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_refine_one_pair, start): start for start in tasks}
            for future in as_completed(futures):
                for vid, desc in future.result():
                    angle = view_angles_str[vid] if vid < len(view_angles_str) else f"视角 {vid}"
                    refined[angle] = desc  # 用 MLLM 修正结果覆盖兜底

        return refined

    def extract_view_instance_knowledge(
        self,
        view_idx: int,
        image_rgb: np.ndarray,
        masks: List[np.ndarray],
        raw_predictions: Dict[int, Dict],
        prompt_classes: List[str],
        object_category: str,
        view_angle: str,
        category_anatomy_knowledge: str,
        current_view_guidance: str
    ) -> ViewInstanceKnowledge:
        """
        阶段 B：单视角实例知识提取。

        基于 Step 2.0 的原始预测结果，进一步调用 MLLM 提取该视角下的实例级知识：
        1. 确认/修正该视角的语义判定
        2. 提取该实例在该视角的具体特征（观测+推理共存）
        3. 记录泛化视角知识与本实例的差异
        4. 自评置信度 + 视角信息可信度评估

        Args:
            view_idx: 视角索引
            image_rgb: RGB 图像
            masks: 该视角的 SAM 掩码列表
            raw_predictions: Step 2.0 的原始预测结果 (mask_id -> {semantic, reasoning, reasoning_source, confidence})
            prompt_classes: 预定义部件列表
            object_category: 物体类别
            view_angle: 当前视角描述
            category_anatomy_knowledge: 类别解剖学知识
            current_view_guidance: 当前视角的视角知识

        Returns:
            ViewInstanceKnowledge 对象
        """
        if not masks or not raw_predictions:
            return ViewInstanceKnowledge(
                view_idx=view_idx,
                confirmed_semantics={},
                instance_view_features={},
                confidence=0.0
            )

        if self.client is None:
            print("Warning: OpenAI client not initialized. Returning empty instance knowledge.")
            return ViewInstanceKnowledge(
                view_idx=view_idx,
                confirmed_semantics={},
                instance_view_features={},
                confidence=0.0
            )

        # 1. 整理该视角的预测信息，构造 MLLM 调用内容
        predictions_summary = []
        for mask_id, pred in raw_predictions.items():
            if mask_id == '_raw_response_':
                continue
            # 处理两种格式：旧版字典格式和新的简单字符串格式
            if isinstance(pred, dict):
                pred_semantic = pred.get("semantic", "unknown")
                pred_reasoning = pred.get("reasoning", "")
            else:
                pred_semantic = pred if pred else "unknown"
                pred_reasoning = ""
            predictions_summary.append({
                "mask_id": mask_id,
                "semantic": pred_semantic,
                "reasoning": pred_reasoning,
            })

        if not predictions_summary:
            return ViewInstanceKnowledge(
                view_idx=view_idx,
                confirmed_semantics={},
                instance_view_features={},
                confidence=0.0
            )

        # 2. 构造 Prompt
        prompt_text = f"""你是一位专业的 3D 视觉分析专家，专注于从单视角图像中提取实例级知识。
当前处理：【{object_category}】的一个实例，视角为 {view_angle}。

【Step 2.0 初步预测结果】：
{json.dumps(predictions_summary, ensure_ascii=False, indent=2)}

【解剖学知识（通用）】：
{category_anatomy_knowledge}

【视角知识（通用）】：
{current_view_guidance}

【你的任务】：
请基于以上信息，对该视角的预测结果进行**实例知识提取**。

**重要约束**：只关注以下预定义部件类别：{json.dumps(prompt_classes, ensure_ascii=False)}。
超出这些类别的掩码区域，**不要写入主输出**，可忽略或简短标注为"other_non_part_area"。

1. **提取实例特征**：对每个预测为预定义类别（非 "background" 且在上述列表中）的掩码，提取该部件在本实例中的具体几何特征：
   - shape: 形状描述（如"比标准座面更宽"、"呈流线型"）
   - size: 相对大小（如"略小于平均水平"、"异常粗壮"）
   - spatial_relation: 空间关系（如"略偏左"、"紧贴主体"）
   - color: 颜色特征（如"深灰色"、"浅木纹色"）
2. **评估置信度**：评估该视角下知识的整体可靠性。

【输出格式】：
必须是一个严格的 JSON 对象，包含以下字段：
{{
  "confirmed_semantics": {{"掩码ID": "部件语义", ...}},
  "instance_features": {{
    "部件名": {{
      "shape": {{"description": "...", "confidence": 0.9}},
      "size": {{"description": "...", "confidence": 0.8}},
      "spatial_relation": {{"description": "...", "confidence": 0.7}},
      "color": {{"description": "...", "confidence": 0.6}}
    }},
    ...
  }},
  "view_confidence": 0.0到1.0的置信度
}}
不要输出任何解释性文字，只输出 JSON 代码块！"""

        # 3. 编码图像并调用 MLLM
        som_image, _ = self._render_som_image(image_rgb, masks, list(range(len(masks))))
        base64_image = self._encode_image_to_base64(som_image)

        max_retries = 5
        retry_count = 0
        success = False
        response_data = None

        while retry_count < max_retries and not success:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=4096,
                    temperature=0.1,
                    timeout=120.0
                )
                response_text = (response.choices[0].message.content or "").strip()
                success = True

                # 4. 解析 JSON 响应
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_data = json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON found in response")

            except Exception as e:
                retry_count += 1
                print(f"[View {view_idx}] ❌ Instance Knowledge Extraction failed (attempt {retry_count}/{max_retries}): {e}")
                import time
                time.sleep(5)
            # end while

        # 如果所有重试都失败，response_data 仍为 None，返回空知识
        if not success or response_data is None:
            print(f"[View {view_idx}] All {max_retries} attempts failed, returning empty knowledge.")
            return ViewInstanceKnowledge(
                view_idx=view_idx,
                confirmed_semantics={},
                instance_view_features={},
                confidence=0.0
            )

        # 5. 构建 ViewInstanceKnowledge 对象
        confirmed_semantics = response_data.get("confirmed_semantics", {})
        instance_features_raw = response_data.get("instance_features", {})
        view_confidence = response_data.get("view_confidence", 0.5)

        # 转换 instance_features 格式，同时将非预定义语义回拉到 prompt_classes
        instance_view_features = {}
        for part_name, features_dict in instance_features_raw.items():
            mapped_name = part_name
            if part_name not in prompt_classes:
                # 回拉：找包含关系最强的预定义类
                mapped_name = None
                for pc in prompt_classes:
                    if pc in part_name.lower() or part_name.lower() in pc:
                        mapped_name = pc
                        break
                if mapped_name is None:
                    # 既不包含也不被包含 → 丢弃（不写入视图知识，避免干扰）
                    continue

            if mapped_name not in instance_view_features:
                instance_view_features[mapped_name] = {}
            for feat_type, feat_data in features_dict.items():
                if isinstance(feat_data, dict):
                    instance_view_features[mapped_name][feat_type] = FeatureDescriptor(
                        feature_type=feat_type,
                        description=feat_data.get("description", ""),
                        source=f"view_{view_idx}",
                        confidence=feat_data.get("confidence", 0.5)
                    )
                else:
                    instance_view_features[mapped_name][feat_type] = FeatureDescriptor(
                        feature_type=feat_type,
                        description=str(feat_data),
                        source=f"view_{view_idx}",
                        confidence=0.5
                    )

        return ViewInstanceKnowledge(
            view_idx=view_idx,
            confirmed_semantics=confirmed_semantics,
            instance_view_features=instance_view_features,
            confidence=view_confidence
        )

    def fuse_instance_knowledge(
        self,
        view_instance_knowledge_list: List[ViewInstanceKnowledge],
        prompt_classes: List[str],
        object_category: str,
        category_anatomy_knowledge: str
    ) -> InstanceKnowledge:
        """
        阶段 C：跨视角实例知识融合。

        将所有视角的实例知识进行融合，生成全局的 InstanceKnowledge 对象。

        Args:
            view_instance_knowledge_list: 各视角的 ViewInstanceKnowledge 列表
            prompt_classes: 预定义部件列表
            object_category: 物体类别
            category_anatomy_knowledge: 类别解剖学知识

        Returns:
            InstanceKnowledge 对象
        """
        if not view_instance_knowledge_list:
            return InstanceKnowledge(
                instance_id="unknown",
                instance_part_features={},
                low_confidence_parts={}
            )

        if self.client is None:
            # 如果没有 MLLM，使用简单的投票融合
            return self._fuse_instance_knowledge_simple(
                view_instance_knowledge_list, prompt_classes, object_category
            )

        # 1. 构造跨视角融合的 Prompt
        views_data = []
        for vk in view_instance_knowledge_list:
            features_flat = {}
            for part, features in vk.instance_view_features.items():
                for k, v in features.items():
                    features_flat[k] = v.description

            views_data.append({
                "view_idx": vk.view_idx,
                "confirmed_semantics": vk.confirmed_semantics,
                "instance_features": features_flat,
                "confidence": vk.confidence
            })

        prompt_text = f"""你是一位专业的 3D 视觉知识融合专家。
当前处理：【{object_category}】的一个实例，需要将多个视角的观测知识融合为统一的实例级知识。

【多视角观测数据】：
{json.dumps(views_data, ensure_ascii=False, indent=2)}

【解剖学知识（通用）】：
{category_anatomy_knowledge}

【你的任务】：
请综合所有视角的信息，进行跨视角的知识融合。

**重要约束**：输出的每个部件名必须是以下预定义类别之一：{json.dumps(prompt_classes, ensure_ascii=False)}。
如果某部件描述与这些预定义类都不匹配但确实存在，**将其置信度记入 low_confidence_parts，而非 instance_part_features**。

1. **冲突仲裁**：当多个视角对同一特征描述不一致时，按照"多视角一致 > 高置信度 > 低置信度"的原则进行仲裁。
2. **特征融合**：将多视角一致的特征合并为统一的特征描述。
3. **低置信度标注**：对于只有单一视角观测到的特征或置信度较低的部件，标记到 low_confidence_parts 中。

【输出格式】：
必须是一个严格的 JSON 对象：
{{
  "instance_part_features": {{
    "部件名": {{
      "features": {{
        "shape": {{"description": "融合后的形状描述", "source": "consensus_2of4_views", "confidence": 0.85}},
        "size": {{...}},
        "spatial_relation": {{...}},
        "color": {{...}}
      }},
      "evidence_summary": "Evidence summary (e.g., observed in 3 out of 4 views)",
      "is_normal_variation": true或false,
      "multi_view_agreement": 0.0到1.0的多视角一致性
    }},
    ...
  }},
  "low_confidence_parts": {{"部件名": 0.0到1.0的置信度, ...}}
}}
不要输出任何解释性文字，只输出 JSON 代码块！"""

        max_retries = 5
        retry_count = 0
        success = False
        response_data = None

        while retry_count < max_retries and not success:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=4096,
                    temperature=0.1,
                    timeout=120.0
                )
                response_text = (response.choices[0].message.content or "").strip()

                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if not json_match:
                    raise ValueError(f"MLLM 返回内容中未找到 JSON 结构体: {response_text[:200]}")

                response_data = json.loads(json_match.group(0))
                success = True  # 只有 JSON 解析成功后才标记 success

            except Exception as e:
                retry_count += 1
                print(f"[Fusion] ❌ Instance Knowledge Fusion failed (attempt {retry_count}/{max_retries}): {e}")
                import time
                time.sleep(5)
            # end while

        # 如果所有重试都失败，response_data 仍为 None，使用简单融合
        if response_data is None:
            print(f"[Fusion] All {max_retries} attempts failed, using simple fusion fallback.")
            return self._fuse_instance_knowledge_simple(
                view_instance_knowledge_list, prompt_classes, object_category
            )

        # 构建 InstanceKnowledge 对象，对非预定义语义做回拉过滤
        part_features_raw = response_data.get("instance_part_features", {})
        instance_part_features = {}
        low_confidence_parts = dict(response_data.get("low_confidence_parts", {}))

        for part_name, part_data in part_features_raw.items():
            mapped_name = part_name
            if part_name not in prompt_classes:
                mapped_name = None
                for pc in prompt_classes:
                    if pc in part_name.lower() or part_name.lower() in pc:
                        mapped_name = pc
                        break
                if mapped_name is None:
                    # 既不匹配任何预定义类 → 降入 low_confidence_parts
                    low_confidence_parts[part_name] = low_confidence_parts.get(part_name, 0.5)
                    continue

            features_dict = {}
            for feat_type, feat_data in part_data.get("features", {}).items():
                if isinstance(feat_data, dict):
                    features_dict[feat_type] = FeatureDescriptor(
                        feature_type=feat_type,
                        description=feat_data.get("description", ""),
                        source=feat_data.get("source", "fusion"),
                        confidence=feat_data.get("confidence", 0.5)
                    )
                else:
                    features_dict[feat_type] = FeatureDescriptor(
                        feature_type=feat_type,
                        description=str(feat_data),
                        source="fusion",
                        confidence=0.5
                    )

            if mapped_name in instance_part_features:
                # 同名（回拉后合并）：取较高置信度的描述
                existing = instance_part_features[mapped_name]
                for ft, fd in features_dict.items():
                    if ft not in existing.features:
                        existing.features[ft] = fd
                    elif fd.confidence > existing.features[ft].confidence:
                        existing.features[ft] = fd
            else:
                instance_part_features[mapped_name] = PartFeature(
                    part_name=mapped_name,
                    features=features_dict,
                    evidence_summary=part_data.get("evidence_summary", ""),
                    is_normal_variation=part_data.get("is_normal_variation", True),
                    multi_view_agreement=part_data.get("multi_view_agreement", 0.5)
                )

        import time
        instance_id = f"{object_category}_{int(time.time())}"

        return InstanceKnowledge(
            instance_id=instance_id,
            instance_part_features=instance_part_features,
            low_confidence_parts=low_confidence_parts
        )

    def _fuse_instance_knowledge_simple(
        self,
        view_instance_knowledge_list: List[ViewInstanceKnowledge],
        prompt_classes: List[str],
        object_category: str
    ) -> InstanceKnowledge:
        """
        简单的实例知识融合（无 MLLM 时使用）。
        """
        import time
        instance_id = f"{object_category}_{int(time.time())}"

        instance_part_features = {}
        for vk in view_instance_knowledge_list:
            for part_name, features in vk.instance_view_features.items():
                if part_name not in prompt_classes:
                    # 非预定义语义 → 跳过（不写入实例知识，避免干扰下游）
                    continue
                if part_name not in instance_part_features:
                    instance_part_features[part_name] = PartFeature(
                        part_name=part_name,
                        features=dict(features),
                        evidence_summary=f"From view {vk.view_idx}",
                        is_normal_variation=True,
                        multi_view_agreement=1.0 / len(view_instance_knowledge_list)
                    )
                else:
                    existing = instance_part_features[part_name]
                    for feat_type, feat_desc in features.items():
                        if feat_type not in existing.features:
                            existing.features[feat_type] = feat_desc

        return InstanceKnowledge(
            instance_id=instance_id,
            instance_part_features=instance_part_features,
            low_confidence_parts={}
        )

    def _get_annotation_font(self, size: int = 13):
        """用于可视化 SoM 标注（支持中英文路径）。"""
        from PIL import ImageFont
        windir = os.environ.get("WINDIR", r"C:\Windows")
        candidates = [
            os.path.join(windir, "Fonts", "msyh.ttc"),
            os.path.join(windir, "Fonts", "msyhbd.ttc"),
            os.path.join(windir, "Fonts", "simhei.ttf"),
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for p in candidates:
            if p and os.path.isfile(p):
                try:
                    return ImageFont.truetype(p, size)
                except OSError:
                    continue
        return ImageFont.load_default()

    def _render_som_image(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        global_indices: List[int] = None,
        mask_pred_labels: Optional[List[str]] = None,
    ) -> tuple[np.ndarray, Dict[int, tuple]]:
        import cv2
        from PIL import Image, ImageDraw
        som_img = image.copy()
        mask_centers = {}

        if global_indices is None:
            global_indices = list(range(len(masks)))

        overlay = som_img.copy()

        for i, mask in enumerate(masks):
            global_idx = global_indices[i]
            color = np.random.randint(0, 255, (3,)).tolist()
            overlay[mask] = color

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cv2.drawContours(som_img, contours, -1, color, 2)

            dist_img = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            _, max_val, _, max_loc = cv2.minMaxLoc(dist_img)

            if max_val > 0:
                cX, cY = max_loc
            else:
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0:
                    cX = int(np.mean(x_indices))
                    cY = int(np.mean(y_indices))
                else:
                    continue

            mask_centers[i] = (cX, cY)

        alpha = 0.4
        cv2.addWeighted(overlay, alpha, som_img, 1 - alpha, 0, som_img)

        if mask_pred_labels is not None and len(mask_pred_labels) == len(masks):
            pil = Image.fromarray(som_img)
            draw = ImageDraw.Draw(pil)
            font = self._get_annotation_font(13)
            for i, (cX, cY) in mask_centers.items():
                global_idx = global_indices[i]
                pred_raw = mask_pred_labels[i] if i < len(mask_pred_labels) else ""
                # 处理字典（结构化预测）或字符串格式
                if isinstance(pred_raw, dict):
                    pred = pred_raw.get("semantic", "?")
                else:
                    pred = (pred_raw or "").strip() or "?"
                if len(pred) > 36:
                    pred = pred[:33] + "..."
                line = f"{global_idx}: {pred}"
                bbox = draw.textbbox((0, 0), line, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                pad_x, pad_y = 3, 2
                x0 = max(0, cX - tw // 2 - pad_x)
                y0 = max(0, cY - th - pad_y - 4)
                draw.rectangle(
                    [x0, y0, x0 + tw + 2 * pad_x, y0 + th + 2 * pad_y],
                    fill=(0, 0, 0),
                )
                draw.text((x0 + pad_x, y0 + pad_y), line, fill=(255, 255, 255), font=font)
            som_img = np.asarray(pil)
        else:
            for i, (cX, cY) in mask_centers.items():
                global_idx = global_indices[i]
                text = str(global_idx)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.3  # 同比例缩小数字大小 (原来是 0.4)
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                pad = 1  # 缩小黑框的边距

                # 确保黑框绝对居中于 (cX, cY)
                rect_w = text_width + 2 * pad
                rect_h = text_height + 2 * pad

                rect_x1 = cX - rect_w // 2
                rect_y1 = cY - rect_h // 2
                rect_x2 = rect_x1 + rect_w
                rect_y2 = rect_y1 + rect_h

                cv2.rectangle(som_img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)

                # OpenCV putText 的 y 坐标是文本的基线 (baseline)
                # 对于纯数字来说，其视觉高度即为 text_height
                text_x = rect_x1 + pad
                text_y = rect_y1 + pad + text_height

                cv2.putText(som_img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        return som_img, mask_centers

    # ------------------------------------------------------------------
    # 多图反馈模式辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _semantic_display_colors(n: int) -> List[Tuple[int, int, int]]:
        """BGR color wheel, stable."""
        base = [
            (255, 128, 0),
            (0, 255, 128),
            (128, 0, 255),
            (255, 255, 100),
            (100, 200, 255),
            (255, 100, 200),
            (180, 255, 180),
            (255, 180, 100),
            (100, 255, 255),
            (200, 100, 255),
        ]
        return [base[i % len(base)] for i in range(n)]

    # ============================================================
    # Layer 1: 单语义批量渲染图 + 精简反馈
    # ============================================================
    # 新版 Step 2.5: 语义级多视角评估（所有视图打包发送给 MLLM）
    # ============================================================

    @staticmethod
    def _angle_to_description(elev: float, azim: float) -> str:
        """将仰角+方位角转换为直观描述"""
        # 方位角描述
        azim_rad = np.radians(azim)
        cos_a = np.cos(azim_rad)
        sin_a = np.sin(azim_rad)
        if abs(cos_a) > abs(sin_a):
            if cos_a > 0:
                side = "正侧面偏前" if sin_a > 0.3 else "正侧面偏后" if sin_a < -0.3 else "正侧面"
            else:
                side = "背侧偏前" if sin_a > 0.3 else "背侧偏后" if sin_a < -0.3 else "正背面"
        else:
            if sin_a > 0:
                side = "左侧" if cos_a > 0 else "右侧"
            else:
                side = "前侧" if cos_a > 0 else "后侧"

        # 仰角描述
        elev_abs = abs(elev)
        if elev_abs < 5:
            height = "平视高度"
        elif elev_abs < 25:
            height = "俯视" if elev > 0 else "仰视"
        elif elev_abs < 50:
            height = "大角度俯视" if elev > 0 else "大角度仰视"
        else:
            height = "接近顶部的俯视" if elev > 0 else "接近底部的仰视"

        return f"从物体的{height}、{side}方向拍摄"

    # Step 2.5 输入图：每个预定义语义固定一种 RGB 半透明色（所有碎片同色）
    _STEP25_SEM_RGB_PALETTE: List[Tuple[int, int, int]] = [
        (90, 200, 255),
        (255, 150, 90),
        (140, 255, 170),
        (220, 130, 255),
        (255, 230, 100),
        (180, 180, 255),
        (255, 120, 200),
        (160, 240, 240),
    ]

    @staticmethod
    def _step25_view_label_prefix(view_idx: int) -> str:
        """
        每个视角唯一小写前缀（a, b, …, z, aa, ab, …），避免多图并列时 f0/f0 混淆。
        """
        s = ""
        v = view_idx
        while True:
            s = chr(ord("a") + (v % 26)) + s
            v = v // 26 - 1
            if v < 0:
                break
        return s

    def _render_all_views_for_semantic(
        self,
        images_rgb: List[np.ndarray],
        masks_across_views: List[List[np.ndarray]],
        semantics_across_views: List[List[str]],
        target_did: int,
        grouping_across_views: List["SemanticGroupingResult"],
        view_angles: List[Tuple[float, float]],
        prompt_classes: List[str],
        foreground_points_per_view: Optional[List[List[Tuple[int, int]]]] = None,
        pad: int = 20,
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        在「该视角完整原始 RGB 渲染图」上，仅叠加**一个**预定义语义。
        **每个碎片单独**半透明填充 + 独立外轮廓（不合并相邻碎片），标签为
        「视角前缀 + 局部下标」，如视角0 的 a0、a1，视角1 的 b0、b1。

        delete_fragments 在 JSON 的 per_view_errors 条目中应填**局部下标**整数
        （与图上标签数字部分一致），勿使用全局 fragment_id。

        Args:
            pad: 保留参数（兼容旧调用）。

        Returns:
            List of (vis_rgb, target_did, view_idx)，与 pipeline 保存可视化时的解包顺序一致。
        """
        import cv2
        if not prompt_classes or not (0 <= target_did < len(prompt_classes)):
            return []
        target_sem = prompt_classes[target_did]
        sem_color_rgb = self._STEP25_SEM_RGB_PALETTE[
            target_did % len(self._STEP25_SEM_RGB_PALETTE)
        ]
        sem_color_bgr = (
            int(sem_color_rgb[2]),
            int(sem_color_rgb[1]),
            int(sem_color_rgb[0]),
        )
        vis_results = []
        _ = pad
        for vid, (img, masks, semantics, g) in enumerate(
            zip(images_rgb, masks_across_views, semantics_across_views, grouping_across_views)
        ):
            h, w = img.shape[:2]
            fragments = fragments_for_semantic_ordered(g, target_sem)
            if not fragments:
                continue

            vpfx = self._step25_view_label_prefix(vid)
            base = img.astype(np.float32)
            out = base.copy()
            alpha = 0.36
            r0, g0, b0 = (
                float(sem_color_rgb[0]),
                float(sem_color_rgb[1]),
                float(sem_color_rgb[2]),
            )

            for local_i, frag in enumerate(fragments):
                fm = frag["merged_mask"].astype(bool)
                if not fm.any():
                    continue
                # 碎片间轮廓颜色略错开，便于贴邻时肉眼区分
                phase = (local_i % 5) * 18
                r = float(np.clip(r0 + phase - 36, 0, 255))
                gch = float(np.clip(g0 - phase // 2 + 18, 0, 255))
                b = float(np.clip(b0 + phase // 3, 0, 255))
                out[:, :, 0] = np.where(
                    fm, out[:, :, 0] * (1.0 - alpha) + r * alpha, out[:, :, 0],
                )
                out[:, :, 1] = np.where(
                    fm, out[:, :, 1] * (1.0 - alpha) + gch * alpha, out[:, :, 1],
                )
                out[:, :, 2] = np.where(
                    fm, out[:, :, 2] * (1.0 - alpha) + b * alpha, out[:, :, 2],
                )

            vis_u8 = np.clip(out, 0, 255).astype(np.uint8)
            vis_bgr = cv2.cvtColor(vis_u8, cv2.COLOR_RGB2BGR)

            for local_i, frag in enumerate(fragments):
                fm = frag["merged_mask"].astype(bool)
                if not fm.any():
                    continue
                phase = (local_i % 5) * 18
                br = int(np.clip(sem_color_bgr[0] + phase, 0, 255))
                bgc = int(np.clip(sem_color_bgr[1] - phase // 2, 0, 255))
                bb = int(np.clip(sem_color_bgr[2] + phase // 2, 0, 255))
                contours, _ = cv2.findContours(
                    fm.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(vis_bgr, contours, -1, (br, bgc, bb), 2, lineType=cv2.LINE_AA)

                cx, cy = frag.get("centroid", (0, 0))
                if not (0 <= int(cx) < w and 0 <= int(cy) < h):
                    continue
                label = f"{vpfx}{local_i}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                fs, th = 0.5, 1
                (tw, thh), bl = cv2.getTextSize(label, font, fs, th)
                x0, y0 = int(cx) - tw // 2 - 2, int(cy) + thh // 2
                cv2.rectangle(
                    vis_bgr,
                    (x0, y0 - thh - 2),
                    (x0 + tw + 4, y0 + bl + 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    vis_bgr, label, (x0 + 2, y0),
                    font, fs, (255, 255, 255), th, cv2.LINE_AA,
                )

            # 在渲染图上标记前景采样点（小星星）
            if foreground_points_per_view and vid < len(foreground_points_per_view):
                pts = foreground_points_per_view[vid]
                if pts:
                    self._draw_star_markers(vis_bgr, pts)

            vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            vis_results.append((vis_rgb.copy(), target_did, vid))

        return vis_results

    def _draw_star_markers(
        self,
        img_bgr: np.ndarray,
        points: List[Tuple[int, int]],
        color: Tuple[int, int, int] = (0, 255, 255),
        size: int = 8,
    ) -> None:
        """在图像上用黄色小星星标记前景采样点"""
        import cv2
        h, w = img_bgr.shape[:2]
        for px, py in points:
            px_i, py_i = int(round(px)), int(round(py))
            if not (0 <= px_i < w and 0 <= py_i < h):
                continue
            # 绘制星形（5 条线 + 中心点）
            half = size // 2
            inner_r = size // 4
            pts_star = []
            for i in range(10):
                angle = np.deg2rad(i * 36 - 90)
                r = inner_r if i % 2 == 1 else half
                pts_star.append((px_i + int(r * np.cos(angle)), py_i + int(r * np.sin(angle))))
            for i in range(5):
                cv2.line(img_bgr, pts_star[i], pts_star[(i + 2) % 10], color, 1, cv2.LINE_AA)
            cv2.circle(img_bgr, (px_i, py_i), 2, color, -1)

    def evaluate_all_views_per_semantic(
        self,
        images_rgb: List[np.ndarray],
        masks_across_views: List[List[np.ndarray]],
        semantics_across_views: List[List[str]],
        prompt_classes: List[str],
        object_category: str,
        view_angles: List[Tuple[float, float]],
        grouping_across_views: List["SemanticGroupingResult"],
        max_workers: int = 4,
        instance_knowledge: Optional["InstanceKnowledge"] = None,
        anatomy_knowledge: str = "",
        view_guidance_dict: Optional[Dict[str, str]] = None,
        foreground_points_per_view: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> Tuple[dict, List[Tuple[np.ndarray, int, int]]]:
        """
        将各预定义语义的所有视角图打包并发给 MLLM，评估：
        1. 每个视角中各语义部件的错误小碎片
        2. 该语义部件整体的完善程度

        所有语义并行调用 MLLM（ThreadPoolExecutor），大幅提速。

        Args:
            images_rgb: 各视角原图
            masks_across_views: 各视角的掩码列表
            semantics_across_views: 各视角的语义标签
            prompt_classes: 预定义部件列表
            object_category: 物体类别
            view_angles: 各视角的 (仰角, 方位角)
            grouping_across_views: 各视角的分组结果
            max_workers: 最大并发 MLLM 调用数
            instance_knowledge: 可选的实例级知识，用于指导评估
            anatomy_knowledge: 该物体类别的解剖学泛化知识
            view_guidance_dict: 各视角的定制化预期特征描述

        Returns:
            (feedback_dict, all_rendered_visuals)
            feedback_dict["evaluations"]: 每语义评估
            all_rendered_visuals: 每项为 (渲染图 RGB, display_id, view_idx)
        """
        if not prompt_classes:
            return {"evaluations": []}, []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _eval_one_semantic(did: int) -> Tuple[int, dict, List[Tuple[np.ndarray, int, int]]]:
            """
            在子线程中执行单个语义的所有视角评估。
            每个视角串行发送（每次一张图），避免单次内容过大；
            全部视角评估完后聚合为一条结果。
            """
            # 构造实例知识字符串（如果可用）
            instance_knowledge_str = ""
            if instance_knowledge is not None and hasattr(instance_knowledge, 'to_prompt_string'):
                inst_str = instance_knowledge.to_prompt_string()
                if inst_str and "当前尚未提取到有效的部件特征" not in inst_str:
                    instance_knowledge_str = inst_str

            # 构造泛化知识库字符串
            generic_knowledge_parts = []
            if anatomy_knowledge:
                generic_knowledge_parts.append(f"【{object_category} 各部件的解剖学特征】\n{anatomy_knowledge}")
            if view_guidance_dict:
                vg_lines = []
                for view_key, guidance in view_guidance_dict.items():
                    if guidance:
                        vg_lines.append(f"- {view_key}: {guidance}")
                if vg_lines:
                    generic_knowledge_parts.append("【各视角的预期特征引导】\n" + "\n".join(vg_lines))
            generic_knowledge_str = "\n\n".join(generic_knowledge_parts) if generic_knowledge_parts else ""

            visuals = self._render_all_views_for_semantic(
                images_rgb, masks_across_views, semantics_across_views,
                did, grouping_across_views, view_angles, prompt_classes,
                foreground_points_per_view=foreground_points_per_view,
            )
            if not visuals:
                dummy_result = {
                    "display_id": did,
                    "semantic": prompt_classes[did],
                    "overall_quality": "acceptable",
                    "overall_reasoning": "该语义部件在任何视角中均无碎片信息",
                    "per_view_errors": [],
                    "refine_action": "none",
                    "selected_views": [],
                    "missing_description": "",
                }
                return did, dummy_result, []

            # ── 自适应分批策略：视角数量 ≤4 则一次传完，否则尽量均匀分成两批 ─────────
            # 例：5视角→3+2，7视角→4+3，10视角→5+5，1~4视角→全部1次传完
            n_views = len(visuals)

            if n_views <= 4:
                # 单批：全部视角一次性发送
                batch1_views = visuals
                batch2_views = []
            else:
                # 两批：前半 ceil(n/2)，后半 floor(n/2)
                first_batch_size = (n_views + 1) // 2
                batch1_views = visuals[:first_batch_size]
                batch2_views = visuals[first_batch_size:]

            all_evaluations = []

            batch_views = batch1_views
            batch_offset = 0

            content_list: List[dict] = []
            captions = []
            for vis_rgb, _sem_did, vid in batch_views:
                b64 = self._encode_image_to_base64(vis_rgb)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                })
                elev, azim = view_angles[vid] if vid < len(view_angles) else (0, 0)
                desc = self._angle_to_description(elev, azim)
                vpfx = self._step25_view_label_prefix(vid)
                captions.append(
                    f"视角 {vid}（{desc}）；该图碎片标签前缀为「{vpfx}」，"
                    f"完整标签形如 {vpfx}0、{vpfx}1…（与其它视角前缀不同，勿混淆）"
                )

            caption_block = "\n".join(captions)

            # 组装知识库段落（先泛化、后实例，泛化作为默认参考，实例作为优先校正）
            knowledge_block_parts = []

            # 1. 泛化知识库（优先级低，作为默认参考）
            if generic_knowledge_str:
                knowledge_block_parts.append(
                    "【参考一：通用知识库（低优先级）】\n"
                    "这是该物体类别的通用解剖学知识，可作为评估的默认参考基准：\n"
                    + generic_knowledge_str
                )

            # 2. 实例知识库（优先级高，当实例知识与泛化知识冲突时，以实例知识为准）
            if instance_knowledge_str:
                knowledge_block_parts.append(
                    "【参考二：实例具体知识（高优先级，当与上述通用知识冲突时以此为准）】\n"
                    "这是从该实例多视角观测中提取的具体特征，**反映了这个具体样本的独特属性**，"
                    "比通用知识更能代表当前样本的真实情况。当实例知识与通用知识不一致时，"
                    "**应优先采纳实例知识中的描述**，因为该样本可能存在材质、设计或结构的个体差异：\n"
                    + instance_knowledge_str
                )
            else:
                knowledge_block_parts.append(
                    "【参考二：实例具体知识（高优先级）】\n"
                    "当前样本未提取到有效的实例级知识，请仅依赖上述通用知识库和图中观测进行评估。"
                )

            knowledge_block = "\n\n".join(knowledge_block_parts)

            prompt_text = f"""你是一位专业的 3D 部件分割质量评估员。
当前评估部件: {prompt_classes[did]}
预定义部件列表: {prompt_classes}
物体类别: {object_category}

【配图说明】
以下是 "{prompt_classes[did]}" 部件在 {len(captions)} 个不同视角下的分割渲染图（按顺序对应图1、图2、…、图{len(captions)}）：
{caption_block}
- **每个原始掩码碎片单独着色、单独描边**，相邻碎片**不会**被合并成一块；你可对某一碎片区单独建议删除。
- 碎片标签为「视角专属小写字母前缀 + 局部编号」，例如视角0 可能为 a0、a1，视角1 为 b0、b1…**不同视角前缀不同**，请勿把图1 的 a0 与图2 的 b0 混为一谈。
- 同一部件类别下多个碎片色调接近但轮廓独立；标签中的**数字部分**将在 JSON 中与 delete_fragments 对应。

{knowledge_block}

【评估任务】
请综合所有视角图，对该部件进行评估，重点是**识别并删除错误碎片**。

**【核心原则】**：你是质量把关者，**必须主动删除错误碎片**。正确碎片经过后续3D建模会保留，错误碎片会被直接用于3D重建，严重影响最终质量。

**判断碎片是否应删除的详细标准（满足任一即删除）：**

1. **【几何尺寸异常】**：
   - **过小碎片**：碎片面积 < 当前视角图像面积的 0.5%（通常为噪点/麻点）。
   - **极度细长碎片**：长宽比 > 10:1 且面积 < 1%，通常是误分割。
   - **极度扁平碎片**：高度 < 宽度的 5% 且面积 < 图像面积的 1%，通常是投影/阴影。

2. **【与其他预定义部件高度吻合】**：
   - 该碎片的空间位置和几何形状，与预定义部件列表 `{prompt_classes}` 中**除 {prompt_classes[did]} 以外的任意部件**高度吻合（如桌面形状出现在腿部区域）。
   - 该碎片在 `{object_category}` 的已知空间层级关系中，出现在【错误层级】的位置上。

3. **【多视角不一致】**：
   - 该碎片仅在 1~2 个视角中出现，但在其他 80% 以上视角中**完全不存在对应区域**。
   - 该碎片在不同视角中的投影位置、形状、大小存在**明显矛盾**，无法用3D遮挡解释。

4. **【空间位置明显错误】**：
   - 在 {object_category} 中，{prompt_classes[did]} 部件应位于空间某处，但该碎片出现在**完全不可能**出现的位置（如桌面部件出现在地面以下）。
   - 该碎片与图像中其他已确认正确的碎片之间**完全没有空间连贯性**。

5. **【形状与功能预期严重不符】**：
   - 该碎片的几何形状与 {prompt_classes[did]} 部件的任何可能形态（参考知识库）都**完全不匹配**。
   - 例如：{prompt_classes[did]} 应该是"腿"类细长结构，但碎片呈现为扁平大面积薄板。

6. **【与其他部件投影重叠】**：
   - 该碎片占据的区域，在同一视角下已被其他预定义部件（如 `{prompt_classes}` 中的其他部件）的掩码所覆盖。

**【不应删除的情况】**：
- 碎片形状怪异但符合物理可能性（如异形桌腿）。
- 碎片被遮挡导致形状不完整。
- 碎片仅在部分视角中缺失但在其他视角中存在且形状连贯。
- 碎片与其他碎片相邻但可分离。

**【强制规则】**：
- 你的首要任务是**删除错误碎片**，不要害怕删除。
- **宁可误删，不可漏删**：如果对某个碎片的正确性存疑，直接删除它。
- 如果碎片**明显**满足上述6条标准之一，**必须删除**。

【输出格式】
请输出严格合法的 JSON（勿在 JSON 内写说明文字）：
{{
  "per_view_errors": [
    {{
      "view_idx": 0,
      "view_description": "视角描述",
      "correct_fragments": [0, 1, 2],
      "delete_fragments": [3],
      "quality": "acceptable" | "needs_refine",
      "reasoning": "该视角分析"
    }}
  ],
  "overall_reasoning": "综合所有视角的整体分析…",
  "overall_quality": "acceptable" | "needs_refine",
  "refine_action": "none" | "needs_supplement",
  "missing_description": "缺失描述（无缺失时为空字符串）"
}}

（delete_fragments 示例：若该视角有碎片 b0（正常腿部）、b1（极小噪点，面积<0.5%）、b2（扁平薄片，面积<1%，长宽比>10:1），则 delete_fragments 填 [1, 2]。**仅填数字后缀**，勿填字母前缀。）
"""
            content_list.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content_list}]

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=16384,
                )
                response_text = response.choices[0].message.content or ""
                parsed = self._parse_single_semantic_json(response_text)

                # 收集本批次的评估结果
                evals = parsed.get("evaluations", [])
                aggregated = evals[0] if evals else None
                if not aggregated:
                    # 尝试直接从 parsed 里取（MLLM 可能返回扁平格式）
                    aggregated = {k: v for k, v in parsed.items()
                                  if k in ("per_view_errors", "overall_reasoning",
                                           "overall_quality", "refine_action",
                                           "missing_description")}
                if not aggregated:
                    aggregated = {
                        "per_view_errors": [], "overall_reasoning": "MLLM 返回格式解析失败",
                        "overall_quality": "acceptable", "refine_action": "none",
                        "missing_description": "",
                    }
                aggregated["display_id"] = did
                aggregated["semantic"] = prompt_classes[did]
                return did, aggregated, batch_views  # 返回本批次视角

            except Exception as exc:
                print(f"[MLLM] Semantic {prompt_classes[did]} batch 1 evaluation failed: {exc}")
                return did, {
                    "display_id": did,
                    "semantic": prompt_classes[did],
                    "overall_quality": "acceptable",
                    "overall_reasoning": f"API 调用失败: {exc}",
                    "per_view_errors": [],
                    "refine_action": "none",
                    "missing_description": "",
                }, batch_views

            # ── 第二批次（如有） ──────────────────────────────────────────────
            if not batch2_views:
                # 只有一批时，前面已返回，这里不会走到
                pass
            else:
                content_list2: List[dict] = []
                captions2 = []
                for vis_rgb, _sem_did2, vid2 in batch2_views:
                    b64 = self._encode_image_to_base64(vis_rgb)
                    content_list2.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                    })
                    elev, azim = view_angles[vid2] if vid2 < len(view_angles) else (0, 0)
                    desc = self._angle_to_description(elev, azim)
                    vpfx = self._step25_view_label_prefix(vid2)
                    captions2.append(
                        f"视角 {vid2}（{desc}）；该图碎片标签前缀为「{vpfx}」，"
                        f"完整标签形如 {vpfx}0、{vpfx}1…（与其它视角前缀不同，勿混淆）"
                    )

                caption_block2 = "\n".join(captions2)

                knowledge_block_parts2 = []
                if generic_knowledge_str:
                    knowledge_block_parts2.append(
                        "【参考一：通用知识库（低优先级）】\n"
                        "这是该物体类别的通用解剖学知识，可作为评估的默认参考基准：\n"
                        + generic_knowledge_str
                    )
                if instance_knowledge_str:
                    knowledge_block_parts2.append(
                        "【参考二：实例具体知识（高优先级，当与上述通用知识冲突时以此为准）】\n"
                        + instance_knowledge_str
                    )
                else:
                    knowledge_block_parts2.append(
                        "【参考二：实例具体知识（高优先级）】\n"
                        "当前样本未提取到有效的实例级知识，请仅依赖上述通用知识库和图中观测进行评估。"
                    )
                knowledge_block2 = "\n\n".join(knowledge_block_parts2)

                prompt_text2 = f"""你是一位专业的 3D 部件分割质量评估员。
当前评估部件: {prompt_classes[did]}
预定义部件列表: {prompt_classes}
物体类别: {object_category}

【配图说明】
以下是 "{prompt_classes[did]}" 部件在另外 {len(captions2)} 个视角下的分割渲染图（与上一批视角共同构成完整评估）：
{caption_block2}
- **每个原始掩码碎片单独着色、单独描边**，相邻碎片**不会**被合并成一块；你可对某一碎片区单独建议删除。
- 碎片标签为「视角专属小写字母前缀 + 局部编号」，不同视角前缀不同，勿混淆。
- 标签中的**数字部分**将在 JSON 中与 delete_fragments 对应。

{knowledge_block2}

【评估任务】
请综合这些视角图，对该部件进行评估，重点是**识别并删除错误碎片**。

**【核心原则】**：你是质量把关者，**必须主动删除错误碎片**。正确碎片经过后续3D建模会保留，错误碎片会被直接用于3D重建，严重影响最终质量。

**判断碎片是否应删除的详细标准（满足任一即删除）：**

1. **【几何尺寸异常】**：
   - **过小碎片**：碎片面积 < 当前视角图像面积的 0.5%（通常为噪点/麻点）。
   - **极度细长碎片**：长宽比 > 10:1 且面积 < 1%，通常是误分割。
   - **极度扁平碎片**：高度 < 宽度的 5% 且面积 < 图像面积的 1%，通常是投影/阴影。

2. **【与其他预定义部件高度吻合】**：
   - 该碎片的空间位置和几何形状，与预定义部件列表 `{prompt_classes}` 中**除 {prompt_classes[did]} 以外的任意部件**高度吻合。
   - 该碎片在 `{object_category}` 的已知空间层级关系中，出现在【错误层级】的位置上。

3. **【多视角不一致】**：
   - 该碎片仅在 1~2 个视角中出现，但在其他视角中不存在对应区域。

4. **【空间位置明显错误】**：
   - 该碎片出现在完全不可能出现的位置，或与已确认正确的碎片完全没有空间连贯性。

5. **【形状与功能预期严重不符】**：
   - 该碎片的几何形状与 {prompt_classes[did]} 部件的任何可能形态都不匹配。

6. **【与其他部件投影重叠】**：
   - 该碎片占据的区域在同一视角下已被其他预定义部件的掩码覆盖。

**【不应删除的情况】**：
- 碎片形状怪异但符合物理可能性。
- 碎片被遮挡导致形状不完整。
- 碎片与其他碎片相邻但可分离。

**【强制规则】**：
- 你的首要任务是**删除错误碎片**，不要害怕删除。
- **宁可误删，不可漏删**：如果对某个碎片的正确性存疑，直接删除它。

【输出格式】
请输出严格合法的 JSON：
{{
  "per_view_errors": [
    {{
      "view_idx": 0,
      "view_description": "视角描述",
      "correct_fragments": [0, 1, 2],
      "delete_fragments": [3],
      "quality": "acceptable" | "needs_refine",
      "reasoning": "该视角分析"
    }}
  ],
  "overall_reasoning": "综合分析…",
  "overall_quality": "acceptable" | "needs_refine",
  "refine_action": "none" | "needs_supplement",
  "missing_description": "缺失描述（无缺失时为空字符串）"
}}

（delete_fragments 示例：若某碎片是极小噪点或明显错误，则填对应编号；没有极度确信的错误碎片时输出 []。**仅填数字后缀**，勿填字母前缀。）
"""
                content_list2.append({"type": "text", "text": prompt_text2})
                messages2 = [{"role": "user", "content": content_list2}]

                try:
                    response2 = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages2,
                        temperature=0.1,
                        max_tokens=16384,
                    )
                    response_text2 = response2.choices[0].message.content or ""
                    parsed2 = self._parse_single_semantic_json(response_text2)

                    evals2 = parsed2.get("evaluations", [])
                    aggregated2 = evals2[0] if evals2 else None
                    if not aggregated2:
                        aggregated2 = {k: v for k, v in parsed2.items()
                                      if k in ("per_view_errors", "overall_reasoning",
                                               "overall_quality", "refine_action",
                                               "missing_description")}
                    if not aggregated2:
                        aggregated2 = {
                            "per_view_errors": [], "overall_reasoning": "第二批 MLLM 返回格式解析失败",
                            "overall_quality": "acceptable", "refine_action": "none",
                            "missing_description": "",
                        }

                    # 合并两批次的 per_view_errors
                    evals1 = parsed.get("evaluations", [])
                    evals2 = aggregated2.get("per_view_errors", [])
                    merged_evals = evals1 + evals2

                    # 合并整体信息（取第一批次结果，仅将 per_view_errors 合并）
                    aggregated["per_view_errors"] = merged_evals
                    aggregated["display_id"] = did
                    aggregated["semantic"] = prompt_classes[did]

                    return did, aggregated, batch_views + batch2_views

                except Exception as exc2:
                    print(f"[MLLM] Semantic {prompt_classes[did]} batch 2 evaluation failed: {exc2}")
                    # 第二批失败时返回第一批结果（已在之前返回过，这里不会走到）
                    return did, aggregated, batch_views

        # 并行执行所有语义（每个语义内各视角串行）
        all_evaluations: List[dict] = []
        all_rendered_visuals: List[Tuple[np.ndarray, int, int]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_eval_one_semantic, did): did for did in range(len(prompt_classes))}
            for future in as_completed(futures):
                did, result, visuals = future.result()
                all_evaluations.append(result)
                all_rendered_visuals.extend(visuals)

        all_evaluations.sort(key=lambda x: x.get("display_id", -1))
        return {"evaluations": all_evaluations}, all_rendered_visuals

    def select_views_for_refine(
        self,
        images_rgb: List[np.ndarray],
        masks_across_views: List[List[np.ndarray]],
        semantics_across_views: List[List[str]],
        grouping_across_views: List["SemanticGroupingResult"],
        view_angles: List[Tuple[float, float]],
        prompt_classes: List[str],
        object_category: str,
        instance_knowledge: Optional["InstanceKnowledge"] = None,
        anatomy_knowledge: Optional[str] = None,
        view_guidance_dict: Optional[dict] = None,
    ) -> dict:
        """
        单独调用 MLLM，基于已删除错误碎片的渲染图，选择需要修补的视角。

        Args:
            images_rgb: 原图列表
            masks_across_views: 每视角的掩码列表
            semantics_across_views: 每视角的语义列表
            grouping_across_views: 每视角的分组结果
            view_angles: 各视角的 (仰角, 方位角)
            prompt_classes: 预定义部件列表
            object_category: 物体类别
            instance_knowledge: 实例知识库（可选）
            anatomy_knowledge: 泛化解剖学知识（可选）
            view_guidance_dict: 视角引导字典（可选）

        Returns:
            dict with "selections": [{"display_id": int, "selected_views": [int], "reasoning": str}, ...]
        """
        if not prompt_classes:
            return {"selections": []}

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _select_one_semantic(did: int) -> Tuple[int, dict]:
            target_sem = prompt_classes[did]

            # 构造实例知识字符串
            instance_knowledge_str = ""
            if instance_knowledge is not None and hasattr(instance_knowledge, 'to_prompt_string'):
                inst_str = instance_knowledge.to_prompt_string()
                if inst_str and "当前尚未提取到有效的部件特征" not in inst_str:
                    instance_knowledge_str = inst_str

            # 构造泛化知识库字符串
            generic_knowledge_parts = []
            if anatomy_knowledge:
                generic_knowledge_parts.append(f"【{object_category} 各部件的解剖学特征】\n{anatomy_knowledge}")
            if view_guidance_dict:
                vg_lines = []
                for view_key, guidance in view_guidance_dict.items():
                    if guidance:
                        vg_lines.append(f"- {view_key}: {guidance}")
                if vg_lines:
                    generic_knowledge_parts.append("【各视角的预期特征引导】\n" + "\n".join(vg_lines))
            generic_knowledge_str = "\n\n".join(generic_knowledge_parts) if generic_knowledge_parts else ""

            # 为该语义渲染所有视角（去除 delete_fragments 后的掩码）
            visuals = self._render_all_views_for_semantic(
                images_rgb, masks_across_views, semantics_across_views,
                did, grouping_across_views, view_angles, prompt_classes,
            )
            if not visuals:
                return did, {
                    "display_id": did,
                    "selected_views": [],
                    "reasoning": "该语义部件在任何视角中均无碎片信息",
                }

            # 打包所有视角图
            content_list: List[dict] = []
            captions = []
            for vis_rgb, _sem_did, vid in visuals:
                b64 = self._encode_image_to_base64(vis_rgb)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                })
                elev, azim = view_angles[vid] if vid < len(view_angles) else (0, 0)
                desc = self._angle_to_description(elev, azim)
                vpfx = self._step25_view_label_prefix(vid)
                captions.append(
                    f"视角 {vid}（{desc}）；碎片标签前缀为「{vpfx}」"
                )

            caption_block = "\n".join(captions)

            # 知识库段落
            knowledge_block_parts = []
            if generic_knowledge_str:
                knowledge_block_parts.append(
                    "【参考一：通用知识库（低优先级）】\n"
                    + generic_knowledge_str
                )
            if instance_knowledge_str:
                knowledge_block_parts.append(
                    "【参考二：实例具体知识（高优先级，当与上述通用知识冲突时以此为准）】\n"
                    + instance_knowledge_str
                )
            else:
                knowledge_block_parts.append(
                    "【参考二：实例具体知识（高优先级）】\n"
                    "当前样本未提取到有效的实例级知识，请仅依赖上述通用知识库和图中观测进行评估。"
                )
            knowledge_block = "\n\n".join(knowledge_block_parts)

            n_frags_list = []
            for _sem_did, vid in [(v[1], v[2]) for v in visuals]:
                g = grouping_across_views[vid]
                frags = fragments_for_semantic_ordered(g, target_sem)
                n_frags_list.append(len(frags))

            prompt_text = f"""你是一位专业的 3D 部件分割视角选择专家。
当前评估部件: {target_sem}
物体类别: {object_category}
预定义部件列表: {prompt_classes}

【配图说明】
以下是 "{target_sem}" 部件在 {len(captions)} 个不同视角下的分割渲染图（已删除 MLLM 确认的错误碎片）：
{caption_block}
- 每个碎片单独着色并标注标签（视角前缀+编号）。
- 各视角碎片数量: {n_frags_list}

{knowledge_block}

【任务】
综合所有视角图，判断该部件在哪些视角存在**明显缺失区域**（分割不完整、漏掉了该部件的某部分），
需要通过 SAM 补充采样来填补。

注意：
- 被前景遮挡的区域缺失不计入。
- 如果某视角的分割���经足够完整，不需要补充，则不应选入。
- 优先选择**缺失区域最明显、且该视角下碎片形状仍有参考价值**的视角。
- 选择 0~4 个视角，不必全部选择。

【输出格式】
请输出严格合法的 JSON（勿在 JSON 内写说明文字）：
{{
  "selected_views": [视角编号列表，如 [0, 3, 5]]，无缺失时为空数组 [],
  "reasoning": "简要说明为什么选择这些视角，或为什么无需选择。"
}}

（示例：若仅视角 2 和视角 7 存在明显缺失，则 selected_views 填 [2, 7]。）
"""
            content_list.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content_list}]

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=8192,
                )
                response_text = response.choices[0].message.content or ""
                parsed = self._parse_single_semantic_json(response_text)
                selected_views = parsed.get("selected_views", [])
                reasoning = parsed.get("reasoning", "")
                return did, {
                    "display_id": did,
                    "selected_views": selected_views if isinstance(selected_views, list) else [],
                    "reasoning": reasoning,
                }
            except Exception as exc:
                print(f"[MLLM] View selection for semantic {target_sem} failed: {exc}")
                return did, {
                    "display_id": did,
                    "selected_views": [],
                    "reasoning": f"API 调用失败: {exc}",
                }

        # 并行执行所有语义的视角选择
        selections = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_select_one_semantic, did): did for did in range(len(prompt_classes))}
            for future in as_completed(futures):
                did, result = future.result()
                selections.append(result)

        selections.sort(key=lambda x: x.get("display_id", -1))
        return {"selections": selections}

    def evaluate_new_masks(
        self,
        images_rgb: List[np.ndarray],
        new_masks_per_view: List[List[np.ndarray]],
        prompt_classes: List[str],
        object_category: str,
        view_angles: List[Tuple[float, float]],
        anatomy_knowledge: str = "",
        instance_knowledge: Optional["InstanceKnowledge"] = None,
        view_guidance_dict: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        将新分割出的掩码（仅含新掩码）渲染打包发给 MLLM，判断新掩码是否正确。

        Args:
            images_rgb: 原图列表
            new_masks_per_view: 每视角的新掩码列表（仅本次 SAM 新分割出的）
            prompt_classes: 预定义部件列表
            object_category: 物体类别
            view_angles: 各视角的 (仰角, 方位角)

        Returns:
            feedback_dict with evaluations for each view
        """
        import cv2

        all_evaluations = []
        for vid, (img, new_masks) in enumerate(zip(images_rgb, new_masks_per_view)):
            if not new_masks:
                continue

            h, w = img.shape[:2]
            vis_rgb = img.copy()
            overlay = vis_rgb.copy()
            colors = []
            for mid, mask in enumerate(new_masks):
                color = (
                    np.random.randint(100, 255),
                    np.random.randint(100, 255),
                    np.random.randint(100, 255),
                )
                colors.append(color)
                overlay[mask.astype(bool)] = color
            cv2.addWeighted(overlay, 0.55, vis_rgb, 0.45, 0, vis_rgb)

            # 标注掩码编号
            for mid, mask in enumerate(new_masks):
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    continue
                cy, cx = int(np.median(ys)), int(np.median(xs))
                text = f"m{mid}"
                (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                cv2.rectangle(vis_rgb, (cx - tw // 2 - 2, cy - th - 2),
                               (cx + tw // 2 + 2, cy + bl + 2), (0, 0, 0), -1)
                cv2.putText(vis_rgb, text, (cx - tw // 2, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

            elev, azim = view_angles[vid] if vid < len(view_angles) else (0, 0)
            desc = self._angle_to_description(elev, azim)
            bar_h = 28
            bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
            cv2.putText(bar, f"视角: {desc} | SAM 新分割掩码 ({len(new_masks)} 个)",
                        (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            out = np.vstack([bar, vis_rgb])

            b64 = self._encode_image_to_base64(out)
            # 拼装三个知识库
            anatomy_str = anatomy_knowledge if anatomy_knowledge else "（无可用解剖学知识）"

            instance_str = ""
            if instance_knowledge is not None and hasattr(instance_knowledge, 'to_prompt_string'):
                inst_raw = instance_knowledge.to_prompt_string()
                if inst_raw:
                    instance_str = inst_raw

            current_view_desc = self._angle_to_description(
                view_angles[vid][0], view_angles[vid][1]
            ) if vid < len(view_angles) else "未知视角"
            view_guidance_str = "（无视角定制知识）"
            if view_guidance_dict:
                view_key = current_view_desc
                vg = view_guidance_dict.get(view_key, view_guidance_dict.get(str(vid), ""))
                if vg:
                    if isinstance(vg, dict):
                        parts = []
                        if vg.get("expected_parts"):
                            parts.append(f"预期可见部件：{', '.join(vg['expected_parts'])}")
                        if vg.get("common_features"):
                            parts.append(f"常见特征：{vg['common_features']}")
                        if vg.get("spatial_hints"):
                            parts.append(f"空间布局：{vg['spatial_hints']}")
                        view_guidance_str = "；".join(parts) if parts else "（无详细信息）"
                    else:
                        view_guidance_str = str(vg)

            prompt_text = f"""你是一位严谨的 3D 视觉分析专家，专注于掩码质量评估。

【待检对象】
物体类别: {object_category}
本次目标语义: {prompt_classes}

【该视角的定制化预期特征】
{view_guidance_str}

【该物体各部件的 3D 解剖学特征】
{anatomy_str}

【该物体实例级部件知识（来自多视角融合）】
{instance_str}

【配图说明】
- 以下是 SAM 从 3D 点云投影到视角图后，对目标语义进行新分割出的掩码（图中彩色区域）。
- **同一编号（如 m0, m1）对应的是与该数字同属一块、颜色一致且像素连通的整片区域**，数字只是 ID 标注位置，不代表掩码仅覆盖数字附近。请依据**整块连通色域**判断类别，勿仅凭数字旁的局部下结论。
- 图中其他区域为原图背景。

【评估任务】
判断每个新掩码是否真的属于预定义部件之一，即是否为**有效的目标语义掩码**。
1. 如果掩码属于预定义部件列表中的某个语义，标记为正确。
2. 如果掩码是背景、噪点、属于列表之外的部件，或严重过/欠分割，标记为错误并给出 ID。

【极其重要的掩码语义判别法则】（违反将导致灾难性错误，请严格逐条执行）：
1. 空间与结构推理：判断掩码语义前，必须首先观察该掩码的【空间位置】和【几何形状】，并结合 {object_category} 的解剖学结构常识、当前视角的预期特征以及实例级知识进行推断。即使形状怪异，只要符合功能与物理位置，也应判定为正确。
2. 视点与遮挡感知：在观察掩码时，必须建立 3D 深度认知。判断该掩码所在部位是否正受到其他部件的前景遮挡。不要因为掩码被遮挡物"切断"或"挖空"就否定它的语义。同时，严格区分视觉上相邻但在 3D 深度上处于不同层级的部件，严禁将存在遮挡关系的两个不同部件混为一谈。
3. 过分割处理（碎片接受）：SAM 极易产生过分割。掩码可能是一个部件的【完美整体】，也可能只是局部【碎片】。只要该碎片从属于{prompt_classes}，就判定为正确。
4. 掩码过大处理：若掩码过大且覆盖了其他合法部件，应将其判定为错误。
5. 未知结构处理（真正的额外部件）：当你非常确信该掩码是一个独立的、且【不可能属于{prompt_classes}】的结构时，判定为错误。
6. 混合掩码处理：如果一个掩码内既包含大量额外部件，也包含部分合法部件，优先判定为错误，防止污染合法部件的投票。
7. 兜底与无效掩码处理：如果对该掩码的语义完全无法确定，或者它严重跨界融合，请立刻判定为错误。
9. [!!!致命红线 - 3D空间支撑层级禁止!!!]：所有物体都遵循【下层支撑上层】的物理规律。
   - 3D空间中位于【下方】的部件，其几何中心在图像中的垂直位置绝不会高于【位于其上方的部件】。
   - 如果观察到部件A的几何中心在图像中【明显低于】部件B，那么部件A在3D空间中绝不可能位于部件B的上方。
   - 违反层级位置关系的掩码必须判定为错误！
   【几何形状辅助判断】：
   - 【水平扁平区域】（宽>高，明显扁平）：通常是工作面/承载面/平台部件。
   - 【垂直细长区域】（高>宽，长宽比明显>1）：通常是支撑柱/立柱/连接件。
   - 如果掩码的几何形状与其候选语义存在物理矛盾，必须判定为错误！
   【视觉欺骗识别】：
   - 将投影阴影、凹陷区域、被遮挡后残留的碎片误认为独立部件是高频错误。
   - 如果掩码的几何形状极其不规则、破碎，或与其所在空间位置的功能预期不符，应判定为错误。
10. [负例禁止规则 - 宁可漏分不可错分!!!]：如果某个编号的掩码既不像 A 也不像 B，更不像 C，但"好像有点接近某个部件"——这种情况【绝对不要强行归类】。正确做法：判定为错误。宁可遗漏一个真实部件，也不要将其误分类到错误的部件上。

【输出格式】
请输出严格合法的 JSON：
{{
  "evaluations": [
    {{
      "view_idx": {vid},
      "view_description": "{current_view_desc}",
      "new_masks_review": [
        {{
          "mask_id": 0,
          "appears_correct": true | false,
          "likely_semantic": "seat" | "back" | ... | "unknown",
          "reasoning": "判断理由（重点描述直接观测到的特征和结合知识库的推理）"
        }}
      ],
      "quality": "acceptable" | "needs_refine",
      "incorrect_mask_ids": [1, 3]
    }}
  ]
}}

【重要规则】
- 如果所有新掩码都正确，incorrect_mask_ids 填写空数组 []。
- 一定要分析清楚各个预定义语义部件的真实边界，有些掩码可能只是单纯离这些语义部件很近但并不属于预定义语义部件，不要将这些误归类为预定义语义部件。
"""
            content_list = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                {"type": "text", "text": prompt_text},
            ]
            messages = [{"role": "user", "content": content_list}]
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                )
                response_text = response.choices[0].message.content or ""
                parsed = self._parse_single_semantic_json(response_text)
                evals = parsed.get("evaluations", [])
                all_evaluations.extend(evals)
            except Exception as e:
                print(f"[MLLM] New mask evaluation view {vid} failed: {e}")

        return {"evaluations": all_evaluations}

    # ============================================================
    # 旧版单视图方法（保留兼容）
    # ============================================================

    def _render_single_semantic_image(
        self,
        image_rgb: np.ndarray,
        masks: List[np.ndarray],
        grouping: "SemanticGroupingResult",
        target_did: int,
        pad: int = 20,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        渲染单个预定义语义的视角图。
        整张图仅高亮该语义的所有碎片（同一颜色），背景淡化显示。
        每个碎片中心标注其 fragment_id (f0, f1, ...)。

        Args:
            image_rgb: 原始 RGB 图像
            masks: 所有碎片掩码列表
            grouping: 语义分组结果
            target_did: 目标语义的 display_id
            pad: 裁剪时在掩码周围的 padding

        Returns:
            (vis_rgb, crop_origin)
            crop_origin = (x0, y0, crop_w, crop_h) 用于坐标映射
        """
        import cv2
        h, w = image_rgb.shape[:2]
        fragments = grouping.display_id_to_fragments.get(target_did, [])
        if not fragments:
            return np.zeros((28 + h, w, 3), dtype=np.uint8), (0, 0, w, h)

        combined_mask = np.zeros((h, w), dtype=bool)
        for frag in fragments:
            combined_mask |= frag["merged_mask"]

        if combined_mask.any():
            kernel = np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8)
            dilated = cv2.dilate(combined_mask.astype(np.uint8), kernel)
        else:
            dilated = combined_mask.astype(np.uint8)

        ys, xs = np.where(dilated)
        if len(ys) == 0:
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
            x1, y1 = min(w, int(xs.max()) + pad), min(h, int(ys.max()) + pad)

        crop = image_rgb[y0:y1, x0:x1].copy()
        crop_h, crop_w = crop.shape[:2]

        max_dim = max(crop_h, crop_w)
        square_canvas = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        paste_y = (max_dim - crop_h) // 2
        paste_x = (max_dim - crop_w) // 2
        square_canvas[paste_y:paste_y + crop_h, paste_x:paste_x + crop_w] = crop

        vis = square_canvas.astype(np.float32) * 0.75

        sem = grouping.id_to_semantic.get(target_did, "")
        color = self._semantic_display_colors(1)[0]
        color_f = np.array(color, dtype=np.float32)

        for frag in fragments:
            frag_local = frag["merged_mask"][y0:y1, x0:x1].astype(np.float32)
            local_h, local_w = frag_local.shape
            local_canvas = np.zeros((max_dim, max_dim), dtype=np.float32)
            local_canvas[paste_y:paste_y + local_h, paste_x:paste_x + local_w] = frag_local
            vis += local_canvas[:, :, None] * color_f * 0.40

        vis_bgr = np.clip(vis, 0, 255).astype(np.uint8)
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

        combined_local = combined_mask[y0:y1, x0:x1].astype(np.uint8)
        local_canvas2 = np.zeros((max_dim, max_dim), dtype=np.uint8)
        local_canvas2[paste_y:paste_y + crop_h, paste_x:paste_x + crop_w] = combined_local
        contours, _ = cv2.findContours(local_canvas2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_rgb, contours, -1, (255, 255, 255), 1)

        for frag in fragments:
            frag_fid = frag["fragment_id"]
            cfx, cfy = frag["centroid"]
            lx = cfx - x0 + paste_x
            ly = cfy - y0 + paste_y

            text = f"f{frag_fid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs, thk = 0.50, 1
            (tw, th), bl = cv2.getTextSize(text, font, fs, thk)
            cv2.rectangle(vis_rgb, (lx - tw // 2 - 2, ly - th - 2),
                           (lx + tw // 2 + 2, ly + bl + 2), (0, 0, 0), -1)
            cv2.putText(vis_rgb, text, (lx - tw // 2, ly), font, fs, (255, 255, 255), thk)

        fid_list = ", ".join(f"f{fr['fragment_id']}" for fr in fragments)
        title_h = 28
        title_bar = np.zeros((title_h, max_dim, 3), dtype=np.uint8)
        did_label = f"{sem} (碎片: [{fid_list}])"
        cv2.putText(title_bar, did_label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (210, 210, 210), 1)

        out = np.vstack([title_bar, vis_rgb])

        # --- 叠加半透明坐标网格（仅用于 MLLM 评估图）---
        out = self._overlay_grid(out, title_h)

        crop_origin = (x0, y0, crop_w, crop_h)
        return out, crop_origin

    def _overlay_grid(self, vis_rgb: np.ndarray, title_h: int = 28, alpha: float = 0.07) -> np.ndarray:
        """
        在图像上叠加极淡的像素坐标网格，帮助 MLLM 估算点位置。

        Args:
            vis_rgb: 输入图像
            title_h: 标题栏高度，网格不覆盖标题栏
            alpha: 网格线透明度（越小越淡，默认 0.07，几乎不可见）

        Returns:
            叠加网格后的图像
        """
        import cv2
        grid_img = vis_rgb.copy()
        h, w = grid_img.shape[:2]

        # 主网格间隔（大格，40px）
        major_step = 40
        # 次网格间隔（小格，10px），更淡
        minor_step = 10
        minor_alpha = alpha * 0.4

        # 主网格线：中等灰度
        color_major = (180, 180, 180)
        for y in range(title_h, h, major_step):
            cv2.line(grid_img, (0, y), (w - 1, y), color_major, 1, cv2.LINE_AA)
        for x in range(0, w, major_step):
            cv2.line(grid_img, (x, title_h), (x, h - 1), color_major, 1, cv2.LINE_AA)

        # 次网格线：极淡灰度
        color_minor = (150, 150, 150)
        for y in range(title_h, h, minor_step):
            if y % major_step == 0:
                continue
            cv2.line(grid_img, (0, y), (w - 1, y), color_minor, 1, cv2.LINE_AA)
        for x in range(0, w, minor_step):
            if x % major_step == 0:
                continue
            cv2.line(grid_img, (x, title_h), (x, h - 1), color_minor, 1, cv2.LINE_AA)

        # 刻度数字（仅在主网格交点标注坐标值，极小字体）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.22
        thickness = 1
        for y in range(title_h, h, major_step):
            for x in range(0, w, major_step):
                label = f"{x},{y}"
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                tx = min(x + 1, w - tw - 1)
                ty = min(y + th + 1, h - 1)
                cv2.putText(grid_img, label, (tx, ty), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)

        # 混合：vis = vis * (1-alpha) + grid * alpha（仅叠加网格线，不影响原有内容）
        vis_f = vis_rgb.astype(np.float32)
        grid_f = grid_img.astype(np.float32)
        blended = vis_f * (1 - alpha) + grid_f * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _build_single_semantic_content(
        self,
        images_batch: List[np.ndarray],
        grouping: "SemanticGroupingResult",
        prompt_classes: List[str],
        batch_metas: List[dict],
        batch_start: int = 0,
    ) -> Tuple[List[dict], dict]:
        """
        构建单批单语义视角图消息。

        Args:
            images_batch: 本批次的 N 张单语义渲染图
            grouping: 语义分组结果
            prompt_classes: 预定义部件列表
            batch_metas: 本批次每个图的元信息
            batch_start: 全局批次起始索引

        Returns:
            (content_list, prompt_dict)
        """
        content_list: List[dict] = []
        captions = []

        for idx, (vis_img, meta) in enumerate(zip(images_batch, batch_metas)):
            b64 = self._encode_image_to_base64(vis_img)
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            })
            sem = meta.get("part_name", "")
            frag_ids = meta.get("fragment_ids", [])
            global_idx = batch_start + idx + 1
            caption = f"图{global_idx}: 部件={sem}, 碎片ID=[{', '.join(f'f{i}' for i in frag_ids)}]"
            content_list.append({"type": "text", "text": caption})
            captions.append(caption)

        caption_block = "\n".join(captions)

        prompt_text = f"""你是一位专业的 3D 部件分割质量评估员。
预定义部件列表: {prompt_classes}

【配图说明】
- 每张图仅高亮显示一种预定义部件的所有碎片（图中仅该部件着色，其他区域为淡化背景）。
- 碎片中心标注的文本（如 f0, f1）是该碎片的唯一标识。
- 同一颜色的多个碎片属于同一部件的不同连通区域。
- **图中叠加有极淡坐标网格**，网格交点处标有像素坐标 (x,y)，左上角为原点 (0,0)，x 向右为正，y 向下为正。请利用网格精确读取和估算坐标。

【输入图像】
{caption_block}

【评估任务】
对每张图单独分析其对应部件的碎片是否完整正确，判断标准：
1. 几何语义一致性：该部件碎片的形状是否与部件语义相符（如 seat 应为扁平水平，back 应为垂直靠背）。
2. 空间位置合理性：空间位置是否符合该物体的物理结构。
3. 完整性分析：是否存在明显应有但未被高亮覆盖的区域（需考虑 3D 视角的合理遮挡，若被前景遮挡则不算缺失）。

【输出格式要求】
请输出严格合法的 JSON，格式如下：
{{
  "evaluations": [
    {{
      "image_idx": 1,
      "display_id": 0,
      "reasoning": "简要分析该图的高亮碎片形状、位置以及是否有漏缺区域...",
      "quality": "acceptable" | "needs_refine",
      "delete_fragments": [0, 2],
      "supplement_points": [[120, 250], [300, 410]]
    }}
  ]
}}

【重要规则（严格遵守）】
- reasoning: 必须先进行简要推理，再决定质量和操作。
- delete_fragments: 仅提取带有错误碎片标识的纯数字！例如图中有错误碎片 f0 和 f2，此处必须输出整数数组 [0, 2]；无错误输出 []。
- supplement_points: 若存在缺失，请仔细观察缺失区域在图中的大致像素位置，给出 1~3 个 SAM 正向提示点坐标 [x, y]；无缺失输出 []。
- 坐标说明：图像原点(0,0)在左上角，x 向右为正，y 向下为正。图中叠加有极淡坐标网格（主格 40px，小格 10px），网格交点处标注有坐标值。请利用网格精确读取坐标，给出的坐标必须落在缺失的部件实体上，且为整数。
- quality: 仅当 delete_fragments 或 supplement_points 不为空时，才设为 "needs_refine"，否则为 "acceptable"。
- 确保每张输入的图片都有对应的评估结果字典。
"""
        return content_list, {"type": "text", "text": prompt_text}

    def _parse_single_semantic_json(self, response_text: str) -> dict:
        """解析单语义批量评估的 JSON 响应，支持自动修复常见语法错误"""
        import re, json

        def _extract_json(text: str) -> str | None:
            """从文本中提取 JSON 字符串"""
            for pattern in ['```json', '```']:
                if pattern in text:
                    parts = text.split(pattern)
                    for part in parts:
                        part = part.strip()
                        if part.startswith('{') and part.endswith('}'):
                            return part
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                return text[start:end]
            return None

        def _fix_json(text: str) -> str:
            """自动修复常见 JSON 语法错误"""

            # 0. 转义双引号字符串内的原始控制字符（换行/制表等）和多余引号
            #    LLM 经常返回 "reasoning": "多行\n文本" 写成多行未转义形式
            #    也可能写出 "looks "good"" 这样字符串内多余的双引号
            escaped = []
            i = 0
            in_str = False
            while i < len(text):
                c = text[i]
                if not in_str:
                    if c == '"':
                        in_str = True
                    escaped.append(c)
                else:
                    if c == '"':
                        j = i + 1
                        while j < len(text) and text[j] in ' \t':
                            j += 1
                        next_ch = text[j] if j < len(text) else ''
                        if next_ch in (',', '}', ']', ':'):
                            in_str = False
                        else:
                            escaped.append('\\')
                        escaped.append(c)
                    elif c == '\n':
                        escaped.append('\\n')
                    elif c == '\r':
                        pass
                    elif c == '\t':
                        escaped.append('\\t')
                    else:
                        escaped.append(c)
                i += 1
            text = ''.join(escaped)

            # 去除单引号替换为双引号（处理键和字符串值）
            # 1. 将所有单引号包裹的字符串替换（防止破坏已有双引号）
            result = []
            i = 0
            in_str = False
            str_char = None
            while i < len(text):
                c = text[i]
                if not in_str:
                    if c in ('"', "'"):
                        in_str = True
                        str_char = c
                        result.append(c)
                    else:
                        result.append(c)
                else:
                    if c == str_char and (i == 0 or text[i - 1] != '\\'):
                        in_str = False
                        str_char = None
                        result.append(c)
                    elif c == "'" and str_char == '"':
                        # 单引号在双引号字符串内，保持原样
                        result.append(c)
                    else:
                        result.append(c)
                i += 1
            text = ''.join(result)

            # 2. 如果仍有单引号，尝试替换单引号为双引号
            if "'" in text:
                # 找出所有字符串（双引号内）的位置，保留原样；其他单引号替换
                fixed = []
                i = 0
                while i < len(text):
                    if text[i] == "'":
                        fixed.append('"')
                    else:
                        fixed.append(text[i])
                    i += 1
                text = ''.join(fixed)

            # 3. 去除尾部逗号（逗号后紧跟 ] 或 }）
            text = re.sub(r',(\s*[}\]])', r'\1', text)

            # 4. 去除单行注释 // ...
            text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)

            # 5. 去除多行注释 /* ... */
            text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

            # 6. 去除多余的逗号（如 [a, b, , c]）
            text = re.sub(r',\s*,', ',', text)

            # 7. 清理多余空白（可选，有助于解析）
            # text = re.sub(r'\s+', ' ', text)

            return text.strip()

        raw = _extract_json(response_text)
        if raw is None:
            print("[MLLM] No JSON found in response")
            return {"evaluations": []}

        # 尝试直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            e = None

        # 尝试自动修复后解析
        fixed = _fix_json(raw)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e2:
            e = e2

        # ── 截断修复：MLLM 返回被截断的 JSON（常见于 token 耗尽或输出过长）──
        # 截断通常发生在字符串值内部（未关闭的双引号），此时解析器报
        # "Unterminated string starting at" 或 "Expecting ',' delimiter"。
        def _try_truncated_fix(txt: str) -> dict | None:
            """从截断位置向前找到未关闭字符串的开头，闭合后尝试解析"""
            import json as _json
            # line/col info is unreliable; scan backwards for a plausible cut point.
            # Find positions of unescaped " that could be string starts before the cut.
            candidates: list[tuple[int, int]] = []  # (score, cut_pos)
            i = 0
            while i < len(txt):
                if txt[i] == '"' and (i == 0 or txt[i - 1] != '\\'):
                    # potential string start
                    depth = 1
                    j = i + 1
                    while j < len(txt):
                        if txt[j] == '"' and txt[j - 1] != '\\':
                            depth -= 1
                            if depth == 0:
                                break
                            j += 1
                        elif txt[j] == '"' and txt[j - 1] == '\\':
                            j += 1
                        else:
                            if txt[j] in '{[':
                                depth += 1
                            elif txt[j] in '}]':
                                depth -= 1
                            j += 1
                    if depth > 0:
                        # string not closed → possible truncation
                        score = 0
                        k = j if depth == 0 else len(txt)
                        for ch in txt[i:k]:
                            if ch in ('{', '[', ':', ','):
                                score += 1
                        candidates.append((score, j if depth == 0 else len(txt)))
                i += 1

            if not candidates:
                return None
            # Prefer cut points that look like they were inside a value (score > 0)
            best = max(candidates, key=lambda x: x[0])
            cut = best[1]
            if cut < len(txt):
                fixed_txt = txt[:cut]
                # Close any unclosed string at the end
                in_str = False
                result = []
                for ch in reversed(fixed_txt):
                    if not in_str:
                        if ch == '"':
                            in_str = True
                        result.append(ch)
                    else:
                        if ch == '"':
                            in_str = False
                        result.append(ch)
                result = list(reversed(result))
                result_str = ''.join(result)
                # Add a trailing "]}"} to try to close the structure
                for suffix in ['"]}', ']}', '}', '']:
                    trial = result_str + suffix
                    try:
                        obj = _json.loads(trial)
                        return obj
                    except Exception:
                        pass
            return None

        # 尝试从截断位置修复
        truncated = _try_truncated_fix(fixed)
        if truncated is not None:
            return truncated

        # ── 逐条正则提取（兜底）──
        # 先按 "display_id" 提取
        evals = []
        for m in re.finditer(
            r'\{\s*"display_id"\s*:\s*\d+.*?\}(?=\s*,\s*\{|\s*\])',
            fixed, re.DOTALL
        ):
            try:
                evals.append(json.loads(_fix_json(m.group())))
            except Exception:
                pass
        if evals:
            return {"evaluations": evals}

        # 按 "image_idx" 提取（兼容性兜底）
        for m in re.finditer(r'\{[^{}]*"image_idx"[^{}]*\}', raw):
            try:
                evals.append(json.loads(_fix_json(m.group())))
            except Exception:
                pass
        if evals:
            return {"evaluations": evals}

        err_msg = f"{e}" if e else "all strategies failed"
        print(f"[MLLM] Failed to parse JSON (tried raw + fix + 2 regex strategies): {err_msg}")
        print(f"  Raw snippet: {raw[:200]}")
        return {"evaluations": []}

    def get_single_semantic_batch_feedback(
        self,
        image_rgb: np.ndarray,
        masks: List[np.ndarray],
        prompt_classes: List[str],
        object_category: str,
        mask_semantics: List[str],
        batch_size: int = 3,
    ) -> Tuple[dict, "SemanticGroupingResult", List[np.ndarray]]:
        """
        单语义的批量碎片评估（分批调用 API）。

        Args:
            image_rgb: 原始 RGB 图像
            masks: 碎片掩码列表
            prompt_classes: 预定义部件列表
            object_category: 物体类别
            mask_semantics: 每个掩码的预测语义
            batch_size: 每批最多发送的图片数量（默认 3）

        Returns:
            (feedback, grouping, rendered_images)
            feedback["evaluations"]: 所有部件图的评估结果
            grouping: 碎片分组信息
            rendered_images: 每张发送给 MLLM 的渲染图（用于可视化）
        """
        empty_fb = {"evaluations": []}

        if self.client is None:
            print("Warning: OpenAI client not initialized. Returning empty feedback.")
            return empty_fb.copy(), None

        grouping = build_semantic_grouping(
            masks, mask_semantics,
            ignore_semantics={"background", "unlabeled"},
        )

        sorted_dids = sorted(grouping.id_to_semantic.keys())
        if not sorted_dids:
            return empty_fb.copy(), grouping

        # 预先渲染所有部件图，并保存裁剪坐标用于坐标映射
        rendered_images = []
        crop_origins = {}  # did -> (x0, y0, crop_w, crop_h)
        for did in sorted_dids:
            vis_img, crop_origin = self._render_single_semantic_image(
                image_rgb, masks, grouping, did
            )
            rendered_images.append(vis_img)
            crop_origins[did] = crop_origin

        # 分批调用 API
        all_evaluations = []
        for batch_start in range(0, len(rendered_images), batch_size):
            batch_end = min(batch_start + batch_size, len(rendered_images))
            batch_images = rendered_images[batch_start:batch_end]
            batch_dids = sorted_dids[batch_start:batch_end]

            # 构建本批的 image_metas（image_idx 相对于当前批次，从 1 开始）
            batch_metas = []
            for local_idx, did in enumerate(batch_dids):
                batch_metas.append({
                    "batch_offset": batch_start,
                    "local_image_idx": local_idx + 1,
                    "display_id": did,
                    "part_name": grouping.id_to_semantic.get(did, ""),
                    "fragment_ids": [fr["fragment_id"] for fr in grouping.display_id_to_fragments.get(did, [])],
                    "crop_origin": crop_origins.get(did, (0, 0, image_rgb.shape[1], image_rgb.shape[0])),
                    "_rendered_image": rendered_images[batch_start + local_idx],
                })

            content_list, prompt_dict = self._build_single_semantic_content(
                batch_images, grouping, prompt_classes, batch_metas, batch_start
            )
            content_list.append(prompt_dict)

            messages = [{"role": "user", "content": content_list}]
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                )
                response_text = response.choices[0].message.content or ""
                raw_feedback = self._parse_single_semantic_json(response_text)

                # 将 local image_idx 转换为全局 image_idx，并映射坐标
                mapped_batch = self._map_batch_feedback_coords(
                    raw_feedback, batch_images, batch_metas, batch_start, image_rgb.shape[:2]
                )
                all_evaluations.extend(mapped_batch.get("evaluations", []))
            except Exception as e:
                print(f"[MLLM] Batch {batch_start // batch_size + 1} failed: {e}")

        # 合并所有批次结果
        merged_feedback = {"evaluations": all_evaluations}
        return merged_feedback, grouping, rendered_images

    def _map_batch_feedback_coords(
        self,
        feedback: dict,
        raw_batch_images: List[np.ndarray],
        batch_metas: List[dict],
        batch_start: int,
        orig_shape: tuple,
    ) -> dict:
        """将批量评估中的坐标映射回原图（分批模式）"""
        import re
        h, w = orig_shape[:2]
        title_h = 28

        # meta_by_local_idx: local_image_idx -> meta
        meta_by_local = {m["local_image_idx"]: m for m in batch_metas}

        def _safe_int(v):
            try:
                return int(float(v))
            except (ValueError, TypeError):
                match = re.search(r'\d+', str(v))
                return int(match.group()) if match else 0

        def _map_point(pt, local_image_idx: int) -> List[int]:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                return [0, 0]
            px = _safe_int(pt[0])
            py = _safe_int(pt[1])

            if local_image_idx not in meta_by_local:
                return [px, py]

            meta = meta_by_local[local_image_idx]
            x0, y0 = meta["crop_origin"][0], meta["crop_origin"][1]
            crop_w, crop_h = meta["crop_origin"][2], meta["crop_origin"][3]
            rendered_img = meta.get("_rendered_image")
            if rendered_img is None:
                return [px, py]

            max_dim = rendered_img.shape[1]  # width = max_dim
            paste_y = (max_dim - crop_h) // 2
            paste_x = (max_dim - crop_w) // 2

            if py < title_h:
                return [px, py]

            local_y = py - title_h
            orig_x = px - paste_x + x0
            orig_y = local_y - paste_y + y0
            orig_x = max(0, min(w - 1, orig_x))
            orig_y = max(0, min(h - 1, orig_y))
            return [orig_x, orig_y]

        mapped_evals = []
        for ev in feedback.get("evaluations", []):
            mapped_ev = dict(ev)
            local_idx = _safe_int(ev.get("image_idx", 1))
            global_idx = batch_start + local_idx
            mapped_ev["image_idx"] = global_idx

            if local_idx in meta_by_local:
                meta = meta_by_local[local_idx]
                mapped_ev["display_id"] = meta["display_id"]
                mapped_ev["part_name"] = meta["part_name"]
                mapped_ev["fragment_ids"] = meta["fragment_ids"]

            pts = ev.get("supplement_points", [])
            mapped_ev["supplement_points"] = [_map_point(p, local_idx) for p in pts if isinstance(p, (list, tuple))]
            mapped_evals.append(mapped_ev)

        return {"evaluations": mapped_evals}

    def predict_soft_probabilities_all_views(
        self,
        images_rgb: List[np.ndarray],
        depth_maps: List[np.ndarray],
        masks_across_views: List[List[np.ndarray]],
        prompt_classes: List[str],
        object_category: str,
        view_angles_str: List[str],
        unified_knowledge: dict,
        initial_instance_knowledge: dict,
        pose_assessment: dict = None,
        max_workers: int = 5,
        pre_defined_semantics_across_views: List[List[str]] = None,
        merged_masks_semantics: List[List[str]] = None,
        disable_height_prior: bool = False,
    ) -> Tuple[List[List[SoftMask2D]], List[List[np.ndarray]], List[List[Dict[int, Dict]]]]:
        """
        并行调用 MLLM，对所有视角执行 soft probability 预测。

        各视角之间并行调用 MLLM，收集完所有视角结果后再返回。
        返回值可直接替换 pipeline 中 step 2 的逐视角串行循环。

        优化策略：
        1. 增加并行数到 5（max_workers=5）
        2. 在并行调用前统计每个视角的 batch 数量，按 batch 数量从多到少排序视角 ID
        3. 优先将 batch 数量多的视角加入并行队列，让最慢的视角最先开始执行

        Args:
            images_rgb: 各视角原图
            depth_maps: 各视角深度图
            masks_across_views: 各视角掩码列表
            prompt_classes: 预定义部件
            object_category: 物体类别
            view_angles_str: 各视角描述字符串
            unified_knowledge: 统一泛化知识库字典
            initial_instance_knowledge: 初级实例知识库字典
            max_workers: 最大并发视角数

        Returns:
            (soft_masks_across_views, som_images_across_views, raw_predictions_across_views)
            soft_masks_across_views: 每视角的 SoftMask2D 列表
            som_images_across_views: 每视角的 SOM 渲染图列表
            raw_predictions_across_views: 每视角的原始预测（含 reasoning, confidence）
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _count_batches_for_view(vid: int) -> Tuple[int, int]:
            """
            统计单个视角的 batch 数量（图着色分组），不调用 MLLM。
            返回 (vid, batch_count)。
            """
            masks = masks_across_views[vid]
            if not masks:
                return (vid, 0)

            num_masks = len(masks)
            valid_indices = []
            mask_areas = []
            bboxes = []

            for i, m in enumerate(masks):
                area = np.sum(m)
                mask_areas.append(area)
                y_indices, x_indices = np.where(m)
                if len(y_indices) > 0:
                    bboxes.append({
                        'ymin': np.min(y_indices), 'ymax': np.max(y_indices),
                        'xmin': np.min(x_indices), 'xmax': np.max(x_indices)
                    })
                else:
                    bboxes.append({'ymin': 0, 'ymax': 0, 'xmin': 0, 'xmax': 0})

            largest_mask_idx = np.argmax(mask_areas) if mask_areas else -1

            # 过滤无效掩码
            img_h, img_w = images_rgb[vid].shape[:2]
            img_area = img_h * img_w

            view_key = f"view_{vid}"
            view_instance_know = initial_instance_knowledge.get(view_key, {})

            for i in range(num_masks):
                if i == largest_mask_idx:
                    continue
                is_valid = True

                mask_key = f"mask_{i}"
                if mask_key in view_instance_know:
                    raw_feat = view_instance_know[mask_key]
                    if (raw_feat.get("point_ratio", 1.0) == 0.0 and
                        raw_feat.get("principal_curvature", 1.0) == 0.0 and
                        raw_feat.get("normalized_z", 1.0) == 0.0):
                        is_valid = False

                if is_valid and mask_areas[i] > 0.5 * img_area:
                    for j in range(num_masks):
                        if i != j and j != largest_mask_idx and mask_areas[j] > 0:
                            b1, b2 = bboxes[i], bboxes[j]
                            if not (b1['xmax'] < b2['xmin'] or b1['xmin'] > b2['xmax'] or
                                    b1['ymax'] < b2['ymin'] or b1['ymin'] > b2['ymax']):
                                intersection = np.logical_and(masks[i], masks[j])
                                if np.sum(intersection) / mask_areas[j] > 0.9:
                                    is_valid = False
                                    break
                if is_valid:
                    valid_indices.append(i)

            if not valid_indices:
                return (vid, 0)

            # 构建冲突矩阵
            num_valid = len(valid_indices)
            conflict_matrix = np.zeros((num_valid, num_valid), dtype=bool)
            for i in range(num_valid):
                idx_i = valid_indices[i]
                for j in range(i + 1, num_valid):
                    idx_j = valid_indices[j]
                    b1, b2 = bboxes[idx_i], bboxes[idx_j]
                    if not (b1['xmax'] < b2['xmin'] or b1['xmin'] > b2['xmax'] or
                            b1['ymax'] < b2['ymin'] or b1['ymin'] > b2['ymax']):
                        intersection = np.logical_and(masks[idx_i], masks[idx_j])
                        if np.any(intersection):
                            conflict_matrix[i, j] = True
                            conflict_matrix[j, i] = True

            # 图着色分组
            batches_valid = []
            max_masks_per_batch = 6
            sorted_valid_order = np.argsort([mask_areas[valid_indices[i]] for i in range(num_valid)])

            for i in sorted_valid_order:
                placed = False
                for batch in batches_valid:
                    if len(batch) >= max_masks_per_batch:
                        continue
                    has_conflict = False
                    for j in batch:
                        if conflict_matrix[i, j]:
                            has_conflict = True
                            break
                    if not has_conflict:
                        batch.append(i)
                        placed = True
                        break
                if not placed:
                    batches_valid.append([i])

            return (vid, len(batches_valid))

        def _predict_one_view(vid: int) -> Tuple[int, List[SoftMask2D], List[np.ndarray], Dict[int, Dict]]:
            img = images_rgb[vid]
            depth_map = depth_maps[vid]
            masks = masks_across_views[vid]
            current_angle_str = view_angles_str[vid]

            # 提取该视图相关的实例知识并映射为文本
            view_key = f"view_{vid}"
            view_instance_know = initial_instance_knowledge.get(view_key, {})
            text_view_instance_know = {}
            for m_key, raw_feat in view_instance_know.items():
                from config.feature_mapping import map_geometric_features_to_text
                text_view_instance_know[m_key] = map_geometric_features_to_text(raw_feat)
            text_view_instance_know = filter_text_instance_knowledge_for_pose(text_view_instance_know, pose_assessment)
            text_view_instance_know = filter_text_instance_knowledge_for_no_height(
                text_view_instance_know, disable_height_prior
            )

            view_semantics = None
            if merged_masks_semantics is not None and vid < len(merged_masks_semantics):
                view_semantics = merged_masks_semantics[vid]

            view_id_str = f"View_{vid:02d}"
            filtered_unified_knowledge = filter_unified_knowledge_for_pose(
                unified_knowledge,
                object_category,
                view_id_str,
                pose_assessment,
            )

            if view_semantics is not None:
                soft_masks, som_images, _, raw_predictions, som_images_raw = self.predict_soft_probabilities(
                    image=img,
                    depth_map=depth_map,
                    masks=masks,
                    prompt_classes=prompt_classes,
                    object_category=object_category,
                    view_angle=current_angle_str,
                    unified_knowledge=filtered_unified_knowledge,
                    text_view_instance_knowledge=text_view_instance_know,
                    pose_assessment=pose_assessment,
                    pre_defined_semantics=view_semantics,
                )
            else:
                soft_masks, som_images, _, raw_predictions, som_images_raw = self.predict_soft_probabilities(
                    image=img,
                    depth_map=depth_map,
                    masks=masks,
                    prompt_classes=prompt_classes,
                    object_category=object_category,
                    view_angle=current_angle_str,
                    unified_knowledge=filtered_unified_knowledge,
                    text_view_instance_knowledge=text_view_instance_know,
                    pose_assessment=pose_assessment,
                )
            return vid, soft_masks, som_images, raw_predictions, som_images_raw

        num_views = len(images_rgb)
        soft_masks_across_views: List[List[SoftMask2D]] = [None] * num_views
        som_images_across_views: List[List[np.ndarray]] = [None] * num_views
        som_images_raw_across_views: List[List[np.ndarray]] = [None] * num_views
        raw_predictions_across_views: List[List[Dict[int, Dict]]] = [None] * num_views

        # ===== 优化1: 统计每个视角的 batch 数量，按 batch 数量从多到少排序 =====
        print("[Step 2.0] 统计每个视角的 batch 数量...")
        view_batch_counts = []
        for vid in range(num_views):
            _, batch_count = _count_batches_for_view(vid)
            view_batch_counts.append((vid, batch_count))

        # 按 batch 数量从多到少排序（优先处理最慢的视角）
        sorted_views = sorted(view_batch_counts, key=lambda x: x[1], reverse=True)
        sorted_view_ids = [vid for vid, _ in sorted_views]

        print(f"[Step 2.0] 视角 batch 数量: {view_batch_counts}")
        print(f"[Step 2.0] 按 batch 数量排序后的视角顺序: {sorted_view_ids}")

        # ===== 优化2: 按排序后的顺序提交任务 =====
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 优先提交 batch 数量多的视角
            futures = {executor.submit(_predict_one_view, vid): vid for vid in sorted_view_ids}
            for future in as_completed(futures):
                vid, soft_masks, som_images, raw_predictions, som_images_raw = future.result()
                soft_masks_across_views[vid] = soft_masks
                som_images_across_views[vid] = som_images
                som_images_raw_across_views[vid] = som_images_raw
                raw_predictions_across_views[vid] = raw_predictions

        return soft_masks_across_views, som_images_across_views, raw_predictions_across_views, som_images_raw_across_views
