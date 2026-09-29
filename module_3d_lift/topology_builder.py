"""
全局 3D 空间拓扑图构建器

基于 Step 2.0 的所有掩码投票结果，在 3D 点云上构建全局拓扑关系：
1. 将所有掩码映射回 3D 点云，逐点投票聚合，>= threshold 确认语义
2. 按语义分组 → 计算每个部件区域的 centroid / bbox / height_range
3. 基于正面视角（X=左右, Y=高度）计算部件间空间拓扑关系
4. 对每个掩码检测拓扑规则违规
5. 输出全局拓扑文本（供 MLLM 使用）
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from datatypes import SoftMask2D


class TopologyGraphBuilder:
    """从 Step 2.0 结果构建全局 3D 空间拓扑图"""

    def __init__(
        self,
        pc_xyz: np.ndarray,
        index_maps: List[np.ndarray],
        unified_knowledge: Optional[dict] = None,
        object_category: str = "",
        spatial_height_order: Optional[dict] = None,
    ):
        """
        Args:
            pc_xyz: (N, 3) 点云坐标
            index_maps: 每个视角的像素→3D点索引映射
            unified_knowledge: 离线泛化知识库 (含 z_height_range 等)
            object_category: 物体类别名 (e.g., "Chair")
            spatial_height_order: 部件空间高度顺序规则 (e.g., {"back": ["seat", "leg"], "seat": ["leg"]})
        """
        self.pc_xyz = pc_xyz
        self.index_maps = index_maps
        self.object_category = object_category
        self.spatial_height_order = spatial_height_order or {}

        # 解析 unified_knowledge 中每个部件的高度范围
        self.part_height_ranges = {}  # {semantic: (y_min_norm, y_max_norm)}
        self._parse_unified_knowledge(unified_knowledge)

        # 全局归一化高度参考（Y 轴）
        self._y_min = float(pc_xyz[:, 1].min())
        self._y_max = float(pc_xyz[:, 1].max())
        self._y_span = self._y_max - self._y_min + 1e-8

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _parse_unified_knowledge(self, unified_knowledge: Optional[dict]):
        """从泛化知识库解析每个部件的高度范围 (兼容 z_height_range / y_height_range)"""
        if not unified_knowledge:
            return
        target = unified_knowledge
        if self.object_category and self.object_category in unified_knowledge:
            target = unified_knowledge[self.object_category]
        for part_name, part_info in target.items():
            if not isinstance(part_info, dict):
                continue
            z_range = part_info.get("z_height_range", None) or part_info.get("y_height_range", None)
            if z_range is None:
                continue
            if isinstance(z_range, str):
                import ast
                try:
                    z_range = ast.literal_eval(z_range)
                except Exception:
                    continue
            if isinstance(z_range, (list, tuple)) and len(z_range) == 2:
                self.part_height_ranges[part_name] = (float(z_range[0]), float(z_range[1]))

    def _normalize_y(self, y_val: float) -> float:
        """将 Y 坐标归一化到 [0, 1]"""
        return (y_val - self._y_min) / self._y_span

    def _extract_point_indices(self, mask: np.ndarray, view_idx: int) -> np.ndarray:
        """从 2D 掩码提取对应的 3D 点索引（去重）"""
        idx_map = self.index_maps[view_idx]
        if idx_map.ndim == 3:
            idx_map = idx_map[:, :, 0]
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return np.array([], dtype=np.int64)
        pt_indices = idx_map[ys, xs]
        valid = pt_indices >= 0
        return np.unique(pt_indices[valid]).astype(np.int64)

    # ------------------------------------------------------------------
    # Step A: 3D 语义点云
    # ------------------------------------------------------------------
    def build_semantic_point_cloud(
        self,
        soft_masks_across_views: List[List[SoftMask2D]],
        confidence_threshold: float = 0.70,
        min_vote_count: int = 3,
    ) -> Dict[int, str]:
        """
        将所有掩码映射回 3D 点云，逐点投票聚合。

        对每个 3D 点，累加所有覆盖它的掩码的概率向量，
        归一化后，同时满足以下两个条件才确认该点语义：
          1. 最高票语义的占比 >= confidence_threshold
          2. 最高票语义的投票次数 >= min_vote_count

        Returns:
            point_semantics: {point_idx: confirmed_semantic}
        """
        print("    [Topology] Building 3D semantic point cloud ...")
        # point_idx -> {semantic: accumulated_probability}
        point_votes: Dict[int, Dict[str, float]] = {}
        # point_idx -> {semantic: vote_count}  记录每个语义被多少个掩码投票
        point_vote_counts: Dict[int, Dict[str, int]] = {}

        for v_idx, soft_masks in enumerate(soft_masks_across_views):
            idx_map = self.index_maps[v_idx]
            if idx_map.ndim == 3:
                idx_map = idx_map[:, :, 0]

            for sm in soft_masks:
                ys, xs = np.where(sm.mask > 0)
                if len(ys) == 0:
                    continue
                pt_indices = idx_map[ys, xs]
                valid_mask = pt_indices >= 0
                valid_indices = pt_indices[valid_mask]

                # 对每个掩码的概率字典投票
                for pidx in np.unique(valid_indices):
                    pidx_int = int(pidx)
                    if pidx_int not in point_votes:
                        point_votes[pidx_int] = {}
                        point_vote_counts[pidx_int] = {}
                    for sem, prob in sm.probabilities.items():
                        if sem in ("background", "unlabeled"):
                            continue
                        point_votes[pidx_int][sem] = point_votes[pidx_int].get(sem, 0.0) + prob
                        point_vote_counts[pidx_int][sem] = point_vote_counts[pidx_int].get(sem, 0) + 1

        # 归一化 + 阈值确认
        point_semantics: Dict[int, str] = {}
        for pidx, votes in point_votes.items():
            total = sum(votes.values())
            if total <= 0:
                continue
            best_sem = max(votes, key=votes.get)
            best_count = point_vote_counts.get(pidx, {}).get(best_sem, 0)
            if votes[best_sem] / total >= confidence_threshold and best_count >= min_vote_count:
                point_semantics[pidx] = best_sem

        print(f"    [Topology] Confirmed {len(point_semantics)}/{len(point_votes)} points (threshold={confidence_threshold}, min_votes={min_vote_count})")
        return point_semantics

    # ------------------------------------------------------------------
    # Step B: 部件区域聚合（含连通分量拆分）
    # ------------------------------------------------------------------
    def build_part_regions(
        self,
        point_semantics: Dict[int, str],
        cluster_eps: float = 0.025,
        cluster_min_samples: int = 10,
        min_cluster_points: int = 50,
    ) -> Dict[str, dict]:
        """
        将确认语义的点按部件名分组，并通过 DBSCAN 聚类拆分空间不连通的
        同语义实例（例如左 arm 和右 arm）。

        - 若某语义只有 1 个连通分量，key 保持原名（如 "arm"）
        - 若有多个连通分量，key 改为 "arm_0", "arm_1", ...
        - 每个区域额外存储 "base_semantic" 字段记录原始语义名

        Args:
            cluster_eps: DBSCAN 的邻域半径
            cluster_min_samples: DBSCAN 核心点最少邻居数
            min_cluster_points: 丢弃点数少于此值的小簇（噪声）

        Returns:
            {
              "arm_0": {
                "centroid": np.ndarray(3,),
                "bbox_min": np.ndarray(3,),
                "bbox_max": np.ndarray(3,),
                "height_range": (norm_min, norm_max),
                "point_count": int,
                "point_indices": List[int],
                "base_semantic": "arm",
              }, ...
            }
        """
        from sklearn.cluster import DBSCAN

        # 按语义名分组点索引
        semantic_to_indices: Dict[str, List[int]] = {}
        for pidx, sem in point_semantics.items():
            if sem not in semantic_to_indices:
                semantic_to_indices[sem] = []
            semantic_to_indices[sem].append(pidx)

        part_regions: Dict[str, dict] = {}
        for sem, indices in semantic_to_indices.items():
            pts = self.pc_xyz[indices]  # (M, 3)

            # DBSCAN 聚类拆分空间不连通的实例
            if len(pts) >= cluster_min_samples:
                db = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples)
                labels = db.fit_predict(pts)
                unique_labels = set(labels)
                unique_labels.discard(-1)  # 去掉噪声标签

                # 过滤掉太小的簇
                valid_clusters = []
                for lbl in sorted(unique_labels):
                    cluster_mask = (labels == lbl)
                    cluster_count = int(cluster_mask.sum())
                    if cluster_count >= min_cluster_points:
                        valid_clusters.append(lbl)

                if len(valid_clusters) == 0:
                    # 所有点被判为噪声或过小，回退到整体
                    valid_clusters = [None]
            else:
                valid_clusters = [None]  # 点太少，不做聚类

            # 是否需要实例化编号
            use_instance_id = (len(valid_clusters) > 1)

            for ci, lbl in enumerate(valid_clusters):
                if lbl is None:
                    cluster_indices = indices
                else:
                    cluster_mask = (labels == lbl)
                    cluster_indices = [indices[k] for k in range(len(indices)) if cluster_mask[k]]

                cluster_pts = self.pc_xyz[cluster_indices]
                centroid = cluster_pts.mean(axis=0)
                bbox_min = cluster_pts.min(axis=0)
                bbox_max = cluster_pts.max(axis=0)
                h_min = self._normalize_y(float(bbox_min[1]))
                h_max = self._normalize_y(float(bbox_max[1]))

                region_key = f"{sem}_{ci}" if use_instance_id else sem
                part_regions[region_key] = {
                    "centroid": centroid,
                    "bbox_min": bbox_min,
                    "bbox_max": bbox_max,
                    "height_range": (round(h_min, 3), round(h_max, 3)),
                    "point_count": len(cluster_indices),
                    "point_indices": cluster_indices,
                    "base_semantic": sem,
                }

        print(f"    [Topology] Built {len(part_regions)} part regions: {list(part_regions.keys())}")
        return part_regions

    @staticmethod
    def base_semantic(region_key: str, part_regions: Dict[str, dict]) -> str:
        """从 part_regions 中提取某个 key 的原始语义名。"""
        info = part_regions.get(region_key)
        if info and "base_semantic" in info:
            return info["base_semantic"]
        return region_key

    # ------------------------------------------------------------------
    # Step C: 全局拓扑图（邻接 + 正面视角空间关系）
    # ------------------------------------------------------------------
    def build_global_topology(
        self,
        part_regions: Dict[str, dict],
        adjacency_threshold: float = 0.05,
    ) -> dict:
        """
        计算部件间的空间拓扑关系（正面视角：X=左右, Y=上下）。

        Returns:
            {
              "nodes": {
                "seat": {"centroid_normalized": [x, y, z], "height_range": (h_min, h_max), "point_count": N},
                ...
              },
              "edges": [
                {"from": "back", "to": "seat", "relations": ["上方", "接触"], "distance": 0.02},
                ...
              ]
            }
        """
        import open3d as o3d

        nodes = {}
        for sem, info in part_regions.items():
            c = info["centroid"]
            nodes[sem] = {
                "base_semantic": info.get("base_semantic", sem),
                "x": round(float(c[0]), 3),
                "y_height": round(self._normalize_y(float(c[1])), 3),
                "z_depth": round(float(c[2]), 3),
                "height_range": info["height_range"],
                "point_count": info["point_count"],
            }

        edges = []
        sem_list = list(part_regions.keys())

        for i in range(len(sem_list)):
            for j in range(i + 1, len(sem_list)):
                sem_a, sem_b = sem_list[i], sem_list[j]
                base_a = part_regions[sem_a].get("base_semantic", sem_a)
                base_b = part_regions[sem_b].get("base_semantic", sem_b)

                # 跳过同 base_semantic 的实例对（如 wheel_0 ↔ wheel_1），避免冗余
                if base_a == base_b:
                    continue

                pts_a = self.pc_xyz[part_regions[sem_a]["point_indices"]]
                pts_b = self.pc_xyz[part_regions[sem_b]["point_indices"]]

                # 计算最短 3D 距离（降采样加速）
                dist = self._compute_min_distance(pts_a, pts_b)

                # 正面视角空间关系 (X=左右, Y=高度)
                ca = part_regions[sem_a]["centroid"]
                cb = part_regions[sem_b]["centroid"]
                relations = self._compute_front_view_relations(ca, cb, dist, adjacency_threshold)

                edges.append({
                    "from": sem_a,
                    "to": sem_b,
                    "relations": relations,
                    "distance": round(dist, 4),
                })

        topology = {"nodes": nodes, "edges": edges}
        print(f"    [Topology] Global topology: {len(nodes)} nodes, {len(edges)} edges")
        return topology

    def _compute_min_distance(self, pts_a: np.ndarray, pts_b: np.ndarray, max_sample: int = 2000) -> float:
        """计算两组点云之间的最短距离"""
        import open3d as o3d

        if len(pts_a) == 0 or len(pts_b) == 0:
            return float("inf")
        if len(pts_a) > max_sample:
            pts_a = pts_a[np.random.choice(len(pts_a), max_sample, replace=False)]
        if len(pts_b) > max_sample:
            pts_b = pts_b[np.random.choice(len(pts_b), max_sample, replace=False)]

        pcd_b = o3d.geometry.PointCloud()
        pcd_b.points = o3d.utility.Vector3dVector(pts_b)
        kdtree = o3d.geometry.KDTreeFlann(pcd_b)

        min_dist_sq = float("inf")
        for pt in pts_a:
            _, _, dist_sq = kdtree.search_knn_vector_3d(pt, 1)
            if dist_sq[0] < min_dist_sq:
                min_dist_sq = dist_sq[0]

        return float(np.sqrt(min_dist_sq))

    def _compute_front_view_relations(
        self,
        centroid_a: np.ndarray,
        centroid_b: np.ndarray,
        dist_3d: float,
        adj_thresh: float,
    ) -> List[str]:
        """
        基于正面视角计算 A 相对于 B 的空间关系。
        坐标系: X=左右, Y=高度(上), Z=深度(前后)
        返回的关系描述：A相对于B
        """
        dy = float(centroid_a[1] - centroid_b[1])
        dx = float(centroid_a[0] - centroid_b[0])
        dz = float(centroid_a[2] - centroid_b[2])

        relations = []

        # 接触关系
        if dist_3d < adj_thresh:
            relations.append("接触")
        elif dist_3d < adj_thresh * 3:
            relations.append("近邻")

        # 主导方向 (正面视角 X-Y 平面)
        abs_dy = abs(dy)
        abs_dx = abs(dx)

        # 高度关系 (Y 轴)
        if abs_dy > 0.02:
            if dy > 0:
                relations.append("上方")
            else:
                relations.append("下方")

        # 左右关系 (X 轴)
        if abs_dx > 0.02:
            relations.append("侧向错位")

        # 前后关系 (Z 轴) - 辅助信息
        if abs(dz) > 0.05:
            relations.append("纵深错位")

        if not relations:
            relations.append("重叠")

        return relations

    # ------------------------------------------------------------------
    # Step D: 逐掩码拓扑规则检测
    # ------------------------------------------------------------------
    def validate_masks_topology(
        self,
        soft_masks_across_views: List[List[SoftMask2D]],
        semantics_across_views: List[List[str]],
        part_regions: Dict[str, dict],
        global_topology: dict,
        reasonable_adjacencies: Optional[dict] = None,
    ) -> List[dict]:
        """
        对 Step 2.0 初判后的每个掩码进行 3D 拓扑规则裁判。

        检测规则:
        1. 高度越界：掩码 normalized_height 超出该语义的 z_height_range
        2. 不合理邻接：掩码与不该相邻的部件发生 3D 接触
        3. 空间倒置：某掩码出现在不合理的相对位置
        4. 孤立碎片：掩码远离同语义区域

        Returns:
            violations: [{
                "view_idx": int,
                "mask_idx": int,
                "global_id": str,
                "current_semantic": str,
                "violations": [str, ...],  # 违规描述列表
                "mask_centroid_3d": [x, y, z],
                "mask_height": float,  # normalized
                "neighbors": [{"part": str, "distance": float, "relation": str}, ...]
            }]
        """
        print("    [Topology] Validating masks against topology rules ...")
        violations_list = []

        # 预计算每个部件区域的 KDTree 用于距离查询
        part_kdtrees = self._build_part_kdtrees(part_regions)

        # 构建 base_semantic → [region_key, ...] 的反向索引
        base_sem_to_keys: Dict[str, List[str]] = {}
        for rk, info in part_regions.items():
            bs = info.get("base_semantic", rk)
            if bs not in base_sem_to_keys:
                base_sem_to_keys[bs] = []
            base_sem_to_keys[bs].append(rk)

        for v_idx, (soft_masks, sems) in enumerate(zip(soft_masks_across_views, semantics_across_views)):
            for m_idx, (sm, sem) in enumerate(zip(soft_masks, sems)):
                if sem in ("background", "unlabeled"):
                    continue

                # 提取该掩码的 3D 点
                pt_indices = self._extract_point_indices(sm.mask, v_idx)
                if len(pt_indices) < 5:
                    continue

                mask_pts = self.pc_xyz[pt_indices]
                mask_centroid = mask_pts.mean(axis=0)
                mask_h = self._normalize_y(float(mask_centroid[1]))
                mask_h_min = self._normalize_y(float(mask_pts[:, 1].min()))
                mask_h_max = self._normalize_y(float(mask_pts[:, 1].max()))

                violations = []
                neighbors = []

                # 规则1: 高度越界（用 base_semantic 查找高度范围）
                if sem in self.part_height_ranges:
                    expected_min, expected_max = self.part_height_ranges[sem]
                    margin = 0.05  # 允许 5% 的容差
                    if mask_h < expected_min - margin:
                        violations.append(
                            f"高度过低: 该掩码高度={mask_h:.2f}, {sem}允许范围=[{expected_min:.2f}, {expected_max:.2f}]"
                        )
                    elif mask_h > expected_max + margin:
                        violations.append(
                            f"高度过高: 该掩码高度={mask_h:.2f}, {sem}允许范围=[{expected_min:.2f}, {expected_max:.2f}]"
                        )

                # 规则2: 与确认区域的距离和邻接关系
                for part_key, kdtree_info in part_kdtrees.items():
                    part_base = part_regions[part_key].get("base_semantic", part_key)
                    if part_base == sem:
                        continue  # 同语义（含实例）不检查邻接

                    # 计算该掩码到各部件区域的最短距离
                    min_dist = self._query_min_distance(mask_pts, kdtree_info)

                    if min_dist < 0.03:  # 接触
                        # 计算相对方位
                        part_centroid = part_regions[part_key]["centroid"]
                        rel = self._describe_relative_position(mask_centroid, part_centroid)
                        neighbors.append({
                            "part": part_key,
                            "distance": round(min_dist, 4),
                            "relation": rel,
                        })

                        # 检查是否为不合理邻接（用 base_semantic 匹配规则）
                        if reasonable_adjacencies:
                            allowed = reasonable_adjacencies.get(sem, [])
                            if part_base not in allowed:
                                violations.append(
                                    f"不合理邻接: {sem}不应与{part_base}接触(实例={part_key}, 距离={min_dist:.4f}), 允许邻接={allowed}"
                                )

                # 规则3a: 孤立碎片 - 掩码质心远离同语义确认区域
                same_sem_keys = base_sem_to_keys.get(sem, [])
                if same_sem_keys:
                    # 找到最近的同语义实例
                    best_dist = float("inf")
                    for rk in same_sem_keys:
                        d = self._query_centroid_distance(mask_centroid, part_kdtrees.get(rk))
                        if d is not None and d < best_dist:
                            best_dist = d
                    if best_dist < float("inf") and best_dist > 0.08:
                        violations.append(
                            f"孤立碎片: 距离同语义({sem})最近确认区域={best_dist:.3f}, 远超正常范围"
                        )
                else:
                    # 规则3b: 未确认语义 - 该语义从未在3D投票中被确认过
                    violations.append(
                        f"未确认语义: {sem}从未在3D投票中被确认(不存在于part_regions), 该标注可信度极低"
                    )

                # 规则4: 空间倒置（基于常识高度排序）
                height_order_violations = self._check_height_order(sem, mask_h, neighbors, part_regions)
                violations.extend(height_order_violations)

                if violations:
                    # 计算该掩码与所有确认部件的完整空间关系（含距离和方位）
                    topology_relations = []
                    for part_key, kdtree_info in part_kdtrees.items():
                        part_base = part_regions[part_key].get("base_semantic", part_key)
                        min_dist = self._query_min_distance(mask_pts, kdtree_info)
                        part_centroid = part_regions[part_key]["centroid"]
                        rel = self._describe_relative_position(mask_centroid, part_centroid)
                        part_h_range = part_regions[part_key].get("height_range", (0, 0))
                        topology_relations.append({
                            "part": part_key,
                            "distance": round(min_dist, 4),
                            "relation": rel,
                            "part_height_range": [round(part_h_range[0], 2), round(part_h_range[1], 2)],
                            "is_same_semantic": (part_base == sem),
                        })
                    # 按距离排序，只保留最近的 3 个（减少 prompt token）
                    topology_relations.sort(key=lambda x: x["distance"])
                    topology_relations = topology_relations[:3]

                    violations_list.append({
                        "view_idx": v_idx,
                        "mask_idx": m_idx,
                        "global_id": f"view_{v_idx}_mask_{m_idx}",
                        "current_semantic": sem,
                        "violations": violations,
                        "mask_centroid_3d": [round(float(mask_centroid[i]), 4) for i in range(3)],
                        "mask_height": round(mask_h, 3),
                        "mask_height_range": (round(mask_h_min, 3), round(mask_h_max, 3)),
                        "neighbors": neighbors,
                        "topology_relations": topology_relations,
                    })

        print(f"    [Topology] Found {len(violations_list)} masks with topology violations")
        return violations_list

    def _build_part_kdtrees(self, part_regions: Dict[str, dict]) -> dict:
        """为每个部件区域构建 KDTree"""
        import open3d as o3d
        kdtrees = {}
        for sem, info in part_regions.items():
            pts = self.pc_xyz[info["point_indices"]]
            if len(pts) > 3000:
                pts = pts[np.random.choice(len(pts), 3000, replace=False)]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            kdtree = o3d.geometry.KDTreeFlann(pcd)
            kdtrees[sem] = {"kdtree": kdtree, "pts": pts}
        return kdtrees

    def _query_centroid_distance(self, centroid: np.ndarray, kdtree_info: Optional[dict]) -> Optional[float]:
        """查询质心到某 KDTree 的最短距离"""
        if kdtree_info is None:
            return None
        kdtree = kdtree_info["kdtree"]
        _, _, dist_sq = kdtree.search_knn_vector_3d(centroid.astype(np.float64), 1)
        return float(np.sqrt(dist_sq[0]))

    def _query_min_distance(self, query_pts: np.ndarray, kdtree_info: Optional[dict]) -> Optional[float]:
        """查询一组点到某 KDTree 的最短距离"""
        if kdtree_info is None:
            return None
        kdtree = kdtree_info["kdtree"]
        # 对 query 降采样
        if len(query_pts) > 500:
            query_pts = query_pts[np.random.choice(len(query_pts), 500, replace=False)]
        min_dist_sq = float("inf")
        for pt in query_pts:
            _, _, dist_sq = kdtree.search_knn_vector_3d(pt, 1)
            if dist_sq[0] < min_dist_sq:
                min_dist_sq = dist_sq[0]
        return float(np.sqrt(min_dist_sq))

    def _describe_relative_position(self, centroid_a: np.ndarray, centroid_b: np.ndarray) -> str:
        """描述 A 相对于 B 的位置（Y=高度, X=左右, Z=前后深度）"""
        dy = float(centroid_a[1] - centroid_b[1])
        dx = float(centroid_a[0] - centroid_b[0])
        dz = float(centroid_a[2] - centroid_b[2])
        parts = []
        if abs(dy) > 0.02:
            parts.append("上方" if dy > 0 else "下方")
        if abs(dx) > 0.02:
            parts.append("侧向错位")
        if abs(dz) > 0.05:
            parts.append("纵深错位")
        return "+".join(parts) if parts else "重叠"

    def _check_height_order(
        self, sem: str, mask_h: float, neighbors: List[dict],
        part_regions: Optional[Dict[str, dict]] = None,
    ) -> List[str]:
        """
        基于常识检查高度倒置。
        例如: 如果 mask 被标记为 back 但出现在 leg 下方，这违反空间常识。
        """
        violations = []
        expected_above = self.spatial_height_order

        for nbr in neighbors:
            nbr_key = nbr["part"]  # 可能是实例 key（如 arm_0）
            # 提取 base_semantic 用于规则匹配
            if part_regions:
                nbr_base = self.base_semantic(nbr_key, part_regions)
            else:
                nbr_base = nbr_key
            rel = nbr["relation"]

            # 检查: 如果当前 sem 应在 nbr_base 上方，但实际在下方
            if sem in expected_above and nbr_base in expected_above[sem]:
                if "下方" in rel:
                    violations.append(
                        f"空间倒置: {sem}应在{nbr_base}上方, 但该掩码在其下方(h={mask_h:.2f})"
                    )

            # 反向检查: 如果 nbr_base 应在当前 sem 上方
            if nbr_base in expected_above and sem in expected_above[nbr_base]:
                if "上方" in rel:
                    violations.append(
                        f"空间倒置: {sem}应在{nbr_base}下方, 但该掩码在其上方(h={mask_h:.2f})"
                    )

        return violations

    # ------------------------------------------------------------------
    # Step E: 输出全局拓扑文本
    # ------------------------------------------------------------------
    def to_prompt_text(
        self,
        global_topology: dict,
        violations_list: Optional[List[dict]] = None,
        near_threshold: float = 0.15,
    ) -> str:
        """
        生成层次化的全局拓扑关系文本（传给 MLLM prompt）。

        按 base_semantic 聚合同语义实例，自动检测空间分布模式，
        只展示接触/近邻关系，按高度从顶到底排列。
        """
        nodes = global_topology.get("nodes", {})
        edges = global_topology.get("edges", [])

        # ── 1. 按 base_semantic 聚合实例信息 ──
        from collections import defaultdict
        group_info: Dict[str, dict] = {}  # base_sem -> aggregated info
        group_instances: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)

        for key, info in nodes.items():
            base = info.get("base_semantic", key)
            group_instances[base].append((key, info))

        for base, instances in group_instances.items():
            count = len(instances)
            # 聚合高度范围（所有实例的并集）
            h_mins = [inst[1]["height_range"][0] for inst in instances]
            h_maxs = [inst[1]["height_range"][1] for inst in instances]
            agg_h_min = min(h_mins)
            agg_h_max = max(h_maxs)
            # 聚合中心高度（用于排序）
            avg_y = sum(inst[1]["y_height"] for inst in instances) / count
            # 聚合点数
            total_pts = sum(inst[1]["point_count"] for inst in instances)

            # 检测空间分布模式
            distribution = ""
            if count >= 2:
                xs = [inst[1]["x"] for inst in instances]
                zs = [inst[1]["z_depth"] for inst in instances]
                x_spread = max(xs) - min(xs)
                z_spread = max(zs) - min(zs)

                if count == 2 and x_spread > 0.08:
                    distribution = "成对分离分布"
                elif count >= 3:
                    # 检测放射状分布：多个实例围绕中心分散
                    avg_x = sum(xs) / count
                    avg_z = sum(zs) / count
                    radii = [((x - avg_x)**2 + (z - avg_z)**2)**0.5 for x, z in zip(xs, zs)]
                    avg_r = sum(radii) / count
                    if avg_r > 0.05:
                        distribution = "底部放射状分布"
                    elif x_spread > 0.08:
                        distribution = "水平分散分布"
                    else:
                        distribution = "聚集分布"

            group_info[base] = {
                "count": count,
                "h_range": (agg_h_min, agg_h_max),
                "avg_y": avg_y,
                "total_pts": total_pts,
                "distribution": distribution,
                "instances": instances,
            }

        # ── 2. 按 base_semantic 聚合边（只保留接触/近邻） ──
        # adjacency: (base_a, base_b) -> min_distance
        adjacency: Dict[Tuple[str, str], float] = {}
        for edge in edges:
            base_from = nodes[edge["from"]].get("base_semantic", edge["from"])
            base_to = nodes[edge["to"]].get("base_semantic", edge["to"])
            dist = edge["distance"]
            if dist >= near_threshold:
                continue  # 跳过远距离边
            pair = tuple(sorted([base_from, base_to]))
            if pair not in adjacency or dist < adjacency[pair]:
                adjacency[pair] = dist

        # 为每个 base_sem 收集接触的其他 base_sem
        contact_map: Dict[str, List[str]] = defaultdict(list)
        for (a, b), dist in adjacency.items():
            contact_map[a].append(b)
            contact_map[b].append(a)

        # ── 3. 生成文本 ──
        lines = []
        lines.append("【全局空间拓扑图】 Y=高度(0=底,1=顶)；其余空间信息仅表示部件间相对偏移与分离，不表示固定全局朝向")
        lines.append("")
        lines.append("=== 部件空间层次（从顶到底） ===")

        sorted_groups = sorted(group_info.items(), key=lambda x: x[1]["avg_y"], reverse=True)
        for base, ginfo in sorted_groups:
            h = ginfo["h_range"]
            count = ginfo["count"]
            dist_tag = f", {ginfo['distribution']}" if ginfo["distribution"] else ""

            lines.append(
                f"  {base} (×{count}, h=[{h[0]:.2f}~{h[1]:.2f}], "
                f"点数={ginfo['total_pts']}{dist_tag})"
            )

            # 接触关系
            contacts = contact_map.get(base, [])
            if contacts:
                # 按高度排序
                contacts_sorted = sorted(contacts, key=lambda c: group_info.get(c, {}).get("avg_y", 0), reverse=True)
                lines.append(f"    ↕ 接触: {', '.join(contacts_sorted)}")

        # 违规掩码信息已在 suspect_info_text 中提供，此处不再重复

        return "\n".join(lines)

    def generate_mask_topology_context(
        self,
        mask_info: dict,
        global_topology: dict,
        part_regions: Dict[str, dict],
    ) -> str:
        """
        为单个掩码生成它的拓扑上下文文本（用于第二轮 MLLM prompt）。

        Args:
            mask_info: 包含 view_idx, mask_idx, mask_centroid_3d, neighbors, violations 等
        """
        lines = []
        gid = mask_info.get("global_id", "unknown")
        sem = mask_info.get("current_semantic", "unknown")
        h = mask_info.get("mask_height", 0)
        h_range = mask_info.get("mask_height_range", (0, 0))

        lines.append(f"  掩码 {gid} (当前标签={sem}):")
        lines.append(f"    高度: {h:.2f} (范围 {h_range[0]:.2f}~{h_range[1]:.2f})")

        if mask_info.get("neighbors"):
            nbr_strs = []
            for n in mask_info["neighbors"]:
                nbr_strs.append(f"{n['part']}({n['relation']}, 距离={n['distance']:.3f})")
            lines.append(f"    相邻部件: {', '.join(nbr_strs)}")

        if mask_info.get("violations"):
            for v_desc in mask_info["violations"]:
                lines.append(f"    ⚠️ {v_desc}")

        return "\n".join(lines)
