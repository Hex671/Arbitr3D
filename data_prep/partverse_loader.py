"""
PartVerse 数据集加载器（仅供 RES Pipeline 使用，不影响现有 PartNetE / PartObjaverse-Tiny 路径）。

PartVerse 数据布局（与 normalized_glbs.tar.gz / anno_infos.tar.gz / text_captions.json 对齐）：

    <partverse_data_root>/
        text_captions.json                 # {uid: {part_id: [brief, detailed]}}
        normalized_glbs/<uid>.glb          # 整个物体的归一化 mesh
        anno_infos/<uid>/<uid>_face2label.json   # face_id -> part_label

核心函数：

    load_instance(uid)
        → PartVerseInstance(uid, mesh_path, captions_dict, face2label_dict)

    load_instance_pointcloud(instance, num_points)
        → (pcd_open3d, pc_xyz, pc_colors, point_face_idx, point_part_label)
        其中 point_face_idx[i] 给出第 i 个采样点对应的原 mesh face_id；
             point_part_label[i] 是该点的 GT part_label（int）。

    iter_queries(instance, caption_types=("brief","detailed"), max_queries=None)
        → 生成 PartVerseQuery 对象（uid, part_id, caption_type, brief, detailed, gt_part_label）

设计要点：
- 完全独立的模块，不依赖也不影响 data_prep/dataloader.py 的 PointCloudDataset 类。
- mesh→点云走 trimesh.sample 并保留 face_id（关键：用于把 face-level GT 反查到点）。
- 同一物体的多 query 共享同一组采样点，避免重复采样。
"""

import json
import os
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator

import numpy as np


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------
@dataclass
class PartVerseInstance:
    """PartVerse 单 instance 元数据。"""
    uid: str
    mesh_path: str                                  # normalized_glbs/<uid>.glb 绝对路径
    captions: Dict[str, List[str]]                  # {part_id_str: [brief, detailed]}
    face2label: Optional[Dict[int, int]] = None     # face_id -> part_label，可能 None（仅推理时不需要 GT）
    num_parts: int = 0                              # captions 里 part 的数量

    @property
    def part_ids(self) -> List[str]:
        return sorted(self.captions.keys(), key=lambda x: int(x) if x.isdigit() else x)


@dataclass
class PartVerseQuery:
    """单条 RES query：1 个 instance 的 1 个 part 的 1 种 caption 类型。"""
    uid: str
    part_id: str
    caption_type: str       # "brief" | "detailed" | "both"
    brief: str
    detailed: str
    gt_part_label: int = -1     # face2label 中该 part 对应的 label 值；-1 表示 GT 不可用

    @property
    def query_hash(self) -> str:
        """query 的稳定哈希值（用于 cache 子目录命名）。"""
        text = f"{self.uid}|{self.part_id}|{self.caption_type}"
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

    def text_for_mllm(self) -> Tuple[str, str]:
        """根据 caption_type 返回最终送给 MLLM 的 (brief, detailed)."""
        if self.caption_type == "brief":
            return self.brief, ""
        if self.caption_type == "detailed":
            return "", self.detailed
        # both
        return self.brief, self.detailed


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------
class PartVerseLoader:
    """PartVerse 数据集加载器。仅在 RES pipeline 中实例化。"""

    def __init__(
        self,
        data_root: str,
        text_captions_json: str = "text_captions.json",
        glbs_subdir: str = "normalized_glbs",
        anno_infos_subdir: str = "anno_infos",
    ):
        self.data_root = data_root
        self.glbs_dir = os.path.join(data_root, glbs_subdir)
        self.anno_dir = os.path.join(data_root, anno_infos_subdir)
        captions_path = os.path.join(data_root, text_captions_json)

        if not os.path.exists(captions_path):
            raise FileNotFoundError(
                f"text_captions.json not found at {captions_path}; "
                f"set res.partverse_data_root in config.yaml correctly."
            )
        with open(captions_path, "r", encoding="utf-8") as f:
            self._all_captions: Dict[str, Dict[str, List[str]]] = json.load(f)

        print(f"  [PartVerseLoader] Loaded {len(self._all_captions)} instances from {captions_path}")

    # ------------------------------------------------------------------
    # Instance metadata
    # ------------------------------------------------------------------
    def list_uids(self) -> List[str]:
        return list(self._all_captions.keys())

    def has_instance(self, uid: str) -> bool:
        return uid in self._all_captions

    def load_instance(self, uid: str) -> PartVerseInstance:
        """加载 1 个 instance 的元数据（不渲染、不采样）。"""
        if uid not in self._all_captions:
            raise KeyError(f"uid '{uid}' not in PartVerse captions")

        captions = self._all_captions[uid]
        mesh_path = os.path.join(self.glbs_dir, f"{uid}.glb")
        if not os.path.exists(mesh_path):
            # fallback：有些下载格式可能展平到 normalized_glbs/<uid>/<uid>.glb
            alt = os.path.join(self.glbs_dir, uid, f"{uid}.glb")
            if os.path.exists(alt):
                mesh_path = alt

        face2label = self._load_face2label(uid)
        return PartVerseInstance(
            uid=uid,
            mesh_path=mesh_path,
            captions=captions,
            face2label=face2label,
            num_parts=len(captions),
        )

    def _load_face2label(self, uid: str) -> Optional[Dict[int, int]]:
        """加载 face_id -> part_label 映射（PartVerse anno_infos 格式）。"""
        candidates = [
            os.path.join(self.anno_dir, uid, f"{uid}_face2label.json"),
            os.path.join(self.anno_dir, f"{uid}_face2label.json"),
            os.path.join(self.anno_dir, uid, "face2label.json"),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # 兼容 dict / list 两种格式
                if isinstance(raw, dict):
                    return {int(k): int(v) for k, v in raw.items()}
                if isinstance(raw, list):
                    return {i: int(v) for i, v in enumerate(raw)}
            except Exception as e:
                print(f"  [PartVerseLoader] Warning: failed to parse {path}: {e}")
                continue
        return None

    # ------------------------------------------------------------------
    # Mesh -> point cloud（保留 face_id 用于 GT 反查）
    # ------------------------------------------------------------------
    def load_pointcloud_with_faces(
        self,
        instance: PartVerseInstance,
        num_points: int = 10000,
    ) -> Tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        从 mesh 采样得到点云，同时记录每个点的 face_id 与对应 GT part_label。

        Returns:
            pcd_o3d:       open3d.geometry.PointCloud（已带颜色，未归一化）
            pc_xyz:        (N, 3) 原始坐标
            pc_colors:     (N, 3) RGB ∈ [0, 1]
            point_face_id: (N,)   int64，每点对应的 mesh face_id
            point_label:   (N,)   int64，每点的 GT part_label（face2label 缺失时全 -1）
        """
        import trimesh
        import open3d as o3d

        if not os.path.exists(instance.mesh_path):
            raise FileNotFoundError(f"mesh not found: {instance.mesh_path}")

        tm_raw = trimesh.load(instance.mesh_path, process=False)
        if isinstance(tm_raw, trimesh.Scene):
            geom_list = [
                g for g in tm_raw.geometry.values()
                if isinstance(g, trimesh.Trimesh) and g.faces is not None and len(g.faces) > 0
            ]
            if not geom_list:
                raise RuntimeError(f"No valid Trimesh geometry in {instance.mesh_path}")
            # ⚠️ 关键：anno_infos 里的 face_id 通常对应 dump(concatenate=True) 后的全局 face 索引
            tm = tm_raw.dump(concatenate=True)
        elif isinstance(tm_raw, trimesh.Trimesh):
            tm = tm_raw
        else:
            raise RuntimeError(f"Unsupported mesh type {type(tm_raw)} for {instance.mesh_path}")

        if tm.faces is None or len(tm.faces) == 0:
            raise RuntimeError(f"mesh has no faces: {instance.mesh_path}")

        # 按面积权重采样 + 返回每点对应的 face_id
        n_target = max(num_points, 1)
        sample_pts, face_idx = tm.sample(n_target, return_index=True)
        sample_pts = np.asarray(sample_pts, dtype=np.float64)
        face_idx = np.asarray(face_idx, dtype=np.int64)

        # 颜色采样：复用 PointCloudDataset._get_face_colors 的策略，但简化为整体 mesh 一次性
        sample_colors = self._sample_colors_for_faces(tm, face_idx)

        # GT label
        if instance.face2label is not None:
            n_faces = len(tm.faces)
            label_arr = np.full(n_faces, -1, dtype=np.int64)
            for fid, lbl in instance.face2label.items():
                if 0 <= fid < n_faces:
                    label_arr[fid] = lbl
            point_label = label_arr[face_idx]
        else:
            point_label = np.full(len(sample_pts), -1, dtype=np.int64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sample_pts)
        pcd.colors = o3d.utility.Vector3dVector(sample_colors)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        return pcd, sample_pts.astype(np.float32), sample_colors.astype(np.float32), face_idx, point_label

    @staticmethod
    def _sample_colors_for_faces(tm, face_idx: np.ndarray) -> np.ndarray:
        """从 trimesh.Trimesh 采样面颜色。失败时返回灰色。"""
        import trimesh as _tm
        try:
            n_faces = len(tm.faces)
            # 与 data_prep/dataloader.py:_get_face_colors 一致的策略链
            if hasattr(tm.visual, "mesh"):
                tm.visual.mesh = tm
            if isinstance(tm.visual, _tm.visual.TextureVisuals):
                try:
                    vc = tm.visual.to_color().vertex_colors
                    vc = np.asarray(vc)[:, :3].astype(np.float64) / 255.0
                    fc = vc[tm.faces].mean(axis=1)
                    return fc[face_idx]
                except Exception:
                    pass
                try:
                    mat = tm.visual.material
                    base = None
                    if hasattr(mat, "baseColorFactor"):
                        base = np.asarray(mat.baseColorFactor, dtype=np.float64)
                    elif hasattr(mat, "main_color"):
                        base = np.asarray(mat.main_color, dtype=np.float64)
                    if base is not None:
                        if base.max() > 1.0:
                            base = base / 255.0
                        rgb = base[:3]
                        return np.tile(rgb, (len(face_idx), 1))
                except Exception:
                    pass
            if isinstance(tm.visual, _tm.visual.ColorVisuals):
                try:
                    cv = tm.visual
                    cv.mesh = tm
                    fc = np.asarray(cv.face_colors)[:, :3].astype(np.float64) / 255.0
                    return fc[face_idx]
                except Exception:
                    pass
            try:
                cv = tm.visual.to_color()
                cv.mesh = tm
                fc = np.asarray(cv.face_colors)[:, :3].astype(np.float64) / 255.0
                return fc[face_idx]
            except Exception:
                pass
        except Exception:
            pass
        # 兜底灰色
        return np.full((len(face_idx), 3), 0.5, dtype=np.float64)

    # ------------------------------------------------------------------
    # Query 迭代
    # ------------------------------------------------------------------
    def iter_queries(
        self,
        instance: PartVerseInstance,
        caption_types: Tuple[str, ...] = ("brief", "detailed"),
        max_queries: Optional[int] = None,
    ) -> Iterator[PartVerseQuery]:
        """
        遍历 1 个 instance 的所有 (part_id, caption_type) 组合作为 RES query。
        """
        count = 0
        for part_id in instance.part_ids:
            cap_pair = instance.captions[part_id]
            if not isinstance(cap_pair, list) or len(cap_pair) < 2:
                continue
            brief, detailed = cap_pair[0], cap_pair[1]
            gt_label = -1
            if instance.face2label is not None:
                # 把 part_id 字符串转 int，用作 part_label
                try:
                    gt_label = int(part_id)
                except ValueError:
                    gt_label = -1
            for ctype in caption_types:
                yield PartVerseQuery(
                    uid=instance.uid,
                    part_id=part_id,
                    caption_type=ctype,
                    brief=brief,
                    detailed=detailed,
                    gt_part_label=gt_label,
                )
                count += 1
                if max_queries is not None and count >= max_queries:
                    return
