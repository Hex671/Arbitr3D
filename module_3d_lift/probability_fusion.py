import numpy as np
import json
from typing import List, Dict
from datatypes import PointProbabilityField

class ProbabilityFusion:
    def __init__(self, sp_reg_strength: float = 0.03,
                 sp_k_nn_adj: int = 10,
                 sp_num_points_per_sp: int = 20,
                 sp_spatial_weight: float = 5.0,
                 coverage_ratio_thresh: float = 0.03):
        self.sp_reg_strength = sp_reg_strength
        self.sp_k_nn_adj = sp_k_nn_adj
        self.sp_num_points_per_sp = sp_num_points_per_sp
        self.sp_spatial_weight = sp_spatial_weight
        self.coverage_ratio_thresh = coverage_ratio_thresh

    def fuse_probabilities(
        self,
        projected_data: List[Dict],
        num_points: int,
        prompt_classes: List[str],
        point_cloud: object,
        anatomy_knowledge_json_str: str = "{}"
    ) -> PointProbabilityField:
        """
        基于超点 (Superpoints) 的 3D 软概率场融合
        包含了基于大模型动态生成的 Z 轴空间先验过滤约束
        """
        # 0. 解析物理知识字典
        anatomy_dict = {}
        try:
            knowledge_data = json.loads(anatomy_knowledge_json_str)

            # 兼容新版的统一知识库结构 { "arm": { "z_height_range": "[0.4, 0.8]" }, ... }
            # 以及旧版的 anatomy_knowledge 结构
            if "anatomy_knowledge" in knowledge_data:
                anatomy_dict = knowledge_data["anatomy_knowledge"].get("predefined_parts", {})
            elif "predefined_parts" in knowledge_data:
                anatomy_dict = knowledge_data.get("predefined_parts", {})
            else:
                # 已经是新版结构，直接用
                anatomy_dict = knowledge_data

            # 处理大模型生成的 "[0.4, 0.8]" 字符串格式
            for part, info in anatomy_dict.items():
                if isinstance(info, dict) and "z_height_range" in info:
                    z_range = info["z_height_range"]
                    if isinstance(z_range, str):
                        try:
                            # 尝试安全地评估字符串为列表
                            import ast
                            parsed_range = ast.literal_eval(z_range)
                            if isinstance(parsed_range, list) and len(parsed_range) == 2:
                                info["z_height_range"] = parsed_range
                            else:
                                info["z_height_range"] = [0.0, 1.0]
                        except Exception:
                            info["z_height_range"] = [0.0, 1.0]
        except Exception as e:
            print(f"    [Warning] Failed to parse anatomy knowledge for physical constraints: {e}")
        # 1. 计算 3D 点云的归一化高度 (0.0 ~ 1.0)
        points = np.asarray(point_cloud.points)

        # 【关键修复1：高度轴索引】
        # 改回 2 (Z轴)。如果你的数据集是 Z 轴向上，之前的 1(Y轴) 会把深度当成高度，导致腿部被物理约束误杀！
        # 若实际为 Y 轴向上，请改回 1
        up_axis_index = 1

        height_coords = points[:, up_axis_index]
        h_min, h_max = np.min(height_coords), np.max(height_coords)
        z_normalized = (height_coords - h_min) / (h_max - h_min + 1e-6)

        # 2. 生成超点（参数从外部传入或使用默认值）
        from module_3d_lift.superpoint_generator import generate_superpoints
        print("    Generating superpoints for robust 3D fusion...")
        sp_labels = generate_superpoints(
            point_cloud,
            num_points_per_superpoint=self.sp_num_points_per_sp,
            reg_strength=self.sp_reg_strength,
            k_nn_adj=self.sp_k_nn_adj,
            spatial_weight=self.sp_spatial_weight,
        )
        num_superpoints = np.max(sp_labels) + 1
        print(f"    Generated {num_superpoints} superpoints.")

        # 3. 计算每个超点的平均归一化高度（向量化）
        sp_height_sum = np.bincount(sp_labels, weights=z_normalized,
                                     minlength=num_superpoints)
        sp_count = np.bincount(sp_labels, minlength=num_superpoints).astype(np.float32)
        sp_count = np.maximum(sp_count, 1.0)
        sp_heights = (sp_height_sum / sp_count).astype(np.float32)

        # 增加 unlabeled 作为合法列
        classes = prompt_classes + ["unlabeled", "background"]
        num_classes = len(classes)
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        # =====================================================================
        # 阶段 1: 超点级投票 (与旧逻辑一致，保留超点多数投票的边界稀释能力)
        # =====================================================================
        sp_prob_matrix = np.zeros((num_superpoints, num_classes), dtype=np.float32)
        sp_weight_matrix = np.zeros((num_superpoints, 1), dtype=np.float32)

        # 预计算每个超点的大小，避免循环内重复计算
        sp_sizes = np.bincount(sp_labels, minlength=num_superpoints)

        for data in projected_data:
            indices = data['point_indices']
            probs = data['probabilities']

            if not probs:
                continue
            best_cls = max(probs, key=probs.get)
            if best_cls in {"background", "unlabeled"}:
                continue

            hit_superpoints = sp_labels[indices]
            unique_sps, counts = np.unique(hit_superpoints, return_counts=True)

            for sp_id, count in zip(unique_sps, counts):
                sp_total_points = sp_sizes[sp_id]
                coverage_ratio = count / sp_total_points

                # 闸门 2：比例 OR 绝对计数双通道。
                # 旧逻辑 `coverage_ratio > thresh` 在 SP 总点数大、单 mask 命中点数少时
                # （例：腿部 SP=200 点，掩码命中 5 点 → 2.5%）会把整票丢掉，
                # 进而该 SP 在所有视角都拿不到票而被强行 unlabeled。
                # 现在保留比例阈值的同时，若绝对命中数 ≥5 也直接通过；
                # 同时把比例分支的下限稍微抬到 max(2, ...)，避免 1 点也算通过。
                min_count_by_ratio = max(2, int(self.coverage_ratio_thresh * sp_total_points))
                if count >= min_count_by_ratio or count >= 5:
                    current_sp_height = sp_heights[sp_id]

                    for cls_name, cls_prob in probs.items():
                        if cls_name not in class_to_idx or cls_name in {"background", "unlabeled"}:
                            continue
                        cls_idx = class_to_idx[cls_name]

                        is_valid_location = True
                        if cls_name in anatomy_dict:
                            z_range = anatomy_dict[cls_name].get("z_height_range", [0.0, 1.0])
                            min_allowed = z_range[0] - 0.15
                            max_allowed = z_range[1] + 0.15

                            if current_sp_height < min_allowed or current_sp_height > max_allowed:
                                is_valid_location = False

                        if is_valid_location:
                            weighted_vote = cls_prob
                            sp_prob_matrix[sp_id, cls_idx] += weighted_vote
                            sp_weight_matrix[sp_id, 0] += weighted_vote

        mask = sp_weight_matrix[:, 0] > 0
        sp_prob_matrix[mask] /= sp_weight_matrix[mask]

        uncovered_sp_before = int(np.sum(~mask))
        if uncovered_sp_before > 0:
            sp_prob_matrix[~mask, class_to_idx["unlabeled"]] = 1.0

        # =====================================================================
        # 阶段 1.5: 零票超点 → 3D 最近已标注超点 的概率向量传播
        # 处理两类典型零票来源：
        #   (a) 内部表面点单独成 SP（_orient_normals_outward 把法向量翻反），
        #       在所有 10 视角都被外表面点遮挡，0 次出现在 index_map → 0 票。
        #   (b) 闸门 2 滤掉所有命中（细长 SP + 小掩码场景的极端 case）。
        # 邻居距离超过 max_dist（按物体尺度自适应）时保留 unlabeled，
        # 避免远端孤立点被错误吸附到对面零件。
        # =====================================================================
        propagated_sp = 0
        kept_unlabeled_sp = uncovered_sp_before
        if uncovered_sp_before > 0 and np.any(mask):
            from scipy.spatial import cKDTree

            # 每个 SP 的几何质心（向量化）
            sp_xyz_sum = np.zeros((num_superpoints, 3), dtype=np.float64)
            np.add.at(sp_xyz_sum, sp_labels, points)
            sp_centroids = sp_xyz_sum / sp_count[:, None]  # sp_count 已 max(1,·)

            covered_sp_ids = np.where(mask)[0]
            uncovered_sp_ids = np.where(~mask)[0]

            # 距离上限按整体 bbox 对角线缩放。原值 0.25 对单位球点云 ≈ 0.5（半个对象），
            # 对 Laptop/Microwave 这种壳体类太宽松，会把"机身"SP 全吸到最近部件上、
            # 把 GT 中本应 unlabeled 的机身点强行染色，导致 IoU 暴跌。
            # 改为 0.02 后只兜底真正紧贴外壳的"薄壳孪生 SP"
            # （灯泡内壁 / 灯罩内壁等，几何距离 0.005~0.015），
            # 离任何已标注 SP 都较远的孤立机身 SP 保留 unlabeled。
            bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
            max_dist = 0.02 * bbox_diag

            tree = cKDTree(sp_centroids[covered_sp_ids])
            dists, nn_idx = tree.query(sp_centroids[uncovered_sp_ids], k=1)

            within = dists <= max_dist
            propagate_targets = uncovered_sp_ids[within]
            propagate_sources = covered_sp_ids[nn_idx[within]]

            if propagate_targets.size > 0:
                # 复制邻居的归一化概率向量，并把 unlabeled 列清零，
                # 防止 argmax 在两类概率势均力敌时仍落到 unlabeled。
                sp_prob_matrix[propagate_targets] = sp_prob_matrix[propagate_sources]
                sp_prob_matrix[propagate_targets, class_to_idx["unlabeled"]] = 0.0

            propagated_sp = int(propagate_targets.size)
            kept_unlabeled_sp = uncovered_sp_before - propagated_sp

        # =====================================================================
        # 阶段 2: 反投影回点云
        # =====================================================================
        point_prob_matrix = sp_prob_matrix[sp_labels]

        covered_pts_before = int(np.sum(mask[sp_labels]))
        uncovered_pts_before = num_points - covered_pts_before
        print(
            f"    [Fusion] 投票后零票 SP = {uncovered_sp_before}/{num_superpoints}, "
            f"投票后 unlabeled 点 = {uncovered_pts_before}/{num_points}"
        )
        print(
            f"    [Fusion] 传播后零票 SP = {kept_unlabeled_sp}/{num_superpoints} "
            f"(被传播 {propagated_sp}, 因超距 max_dist=0.02×bbox_diag 保留 {kept_unlabeled_sp})"
        )

        return PointProbabilityField(point_cloud=point_cloud, prob_matrix=point_prob_matrix)
