"""
RES (Referring Expression Segmentation) Pipeline —— PartVerse 评测专用。

⚠️ 重要：本文件**完全独立**，**不修改也不继承** pipeline.py 中的 Arbitr3DPipeline。
        现有的 PartNetE / PartObjaverse-Tiny 测试入口（test.py / batch_test.py）
        路径不会触碰到这个文件，零影响。

运行流程（单 instance × 多 query）：

    Step 0: 数据 / 渲染 / SAM / 几何特征 K_0      （一次，所有 query 共享）
    ----------- 以下 per-query 重新执行 -----------
    Step 1.A: 目标 K_u 重构（caption → K_u 字段 + short_label）
    Step 1.B: 干扰部件发现（4 视角 collage → distractor list）
    Step 2:   第一轮 RES 分类（MLLM per-view）
    Step 3:   命中语义集合 → 在线拓扑推断
    Step 4:   3D 拓扑构建（复用 TopologyGraphBuilder）+ 拓扑违规检测
    Step 5:   target-only 第二轮复审
    Step 6:   3D 软投影 + 概率融合 → 每点最终 label
    Step 7:   导出 target mask + IoU 计算
"""

import os
import sys
import json
import time
import gc
import yaml
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d


# 把项目根目录加入 sys.path，避免子模块 import 失败
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


from datatypes import RenderOutput, SoftMask2D, PointProbabilityField
from data_prep.partverse_loader import PartVerseLoader, PartVerseInstance, PartVerseQuery
from module_render.camera_poses import generate_cameras_from_views, generate_sphere_cameras
from module_render.renderer import MultiViewRenderer
from module_2d_fm.sam_auto_segmenter import SAMAutoSegmenter
from module_2d_fm.image_enhancer import create_edge_enhancer
from module_2d_fm.res_classifier import RESClassifier
from module_3d_lift.geometric_features import GeometricFeatureExtractor
from module_3d_lift.soft_projector import SoftProjector
from module_3d_lift.probability_fusion import ProbabilityFusion
from module_3d_lift.topology_builder import TopologyGraphBuilder
from module_mllm.k_u_target_reconstructor import TargetKURReconstructor
from module_mllm.k_u_distractor_discoverer import DistractorDiscoverer
from module_mllm.topology_inferrer import TopologyInferrer
from module_mllm.res_second_round_classifier import RESSecondRoundClassifier
from evaluation.res_metrics import (
    RESPerQueryResult,
    aggregate_results,
    compute_query_iou,
    format_aggregate_summary,
    to_serializable,
)


# 与 pipeline.py 中复用的 GPU 自动选卡逻辑（轻量内嵌，避免反向 import）
def _set_torch_device(gpu_index: int = -1) -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        if gpu_index >= 0:
            torch.cuda.set_device(gpu_index)
            return f"cuda:{gpu_index}"
        # 简单选 0 卡（PartVerse RES 不强求自动选卡复杂度）
        torch.cuda.set_device(0)
        return "cuda:0"
    except Exception:
        return "cpu"


class RESPipeline:
    """PartVerse 指代分割完整流水线。"""

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def __init__(self, config_path: str, gpu_index: int = -1):
        torch_device = _set_torch_device(gpu_index)
        print(f"  [RESPipeline] PyTorch device: {torch_device}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        from utils.runtime_config import resolve_api_key, resolve_base_url, resolve_model_name
        api_key = resolve_api_key(self.config["mllm"])
        base_url = resolve_base_url(self.config["mllm"])
        model_name = resolve_model_name(self.config["mllm"])
        self.config_path = config_path

        if "res" not in self.config:
            raise KeyError(
                "config.yaml 缺少 res: 段落。请按照本项目最新模板补全 res: 字段。"
            )
        self.res_cfg = self.config["res"]

        # 1) 数据加载器
        self.loader = PartVerseLoader(
            data_root=self.res_cfg["partverse_data_root"],
            text_captions_json=self.res_cfg.get("text_captions_json", "text_captions.json"),
            glbs_subdir=self.res_cfg.get("glbs_subdir", "normalized_glbs"),
            anno_infos_subdir=self.res_cfg.get("anno_infos_subdir", "anno_infos"),
        )

        # 2) 渲染
        self.renderer = MultiViewRenderer(
            resolution=tuple(self.config["render"]["resolution"]),
            point_radius=self.config["render"].get("point_radius", 0.015),
            albedo_min_value=self.config["render"].get("albedo_min_value", 0.55),
        )

        # 3) SAM + 边缘增强（复用现有模块）
        self.sam_auto = SAMAutoSegmenter(
            self.config["model"]["sam_weight_path"],
            self.config["model"]["sam_model_type"],
        )
        self.edge_enhancer = create_edge_enhancer(self.config)

        # 4) MLLM 客户端配置

        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

        # 5) MLLM 模块
        self.target_reconstructor = TargetKURReconstructor(
            api_key=api_key, base_url=base_url, model_name=model_name
        )
        self.distractor_discoverer = DistractorDiscoverer(
            api_key=api_key, base_url=base_url, model_name=model_name
        )
        self.res_classifier = RESClassifier(
            api_key=api_key, base_url=base_url, model_name=model_name
        )
        self.topology_inferrer = TopologyInferrer(
            api_key=api_key, base_url=base_url, model_name=model_name
        )
        self.res_second_round = RESSecondRoundClassifier(
            api_key=api_key, base_url=base_url, model_name=model_name
        )

        # 6) 3D 后处理
        self.soft_projector = SoftProjector(
            depth_threshold=self.config["algorithm"].get("depth_threshold", 0.05),
            erode_pixels=self.config["algorithm"].get("mask_erode_pixels", 3),
        )
        sp_cfg = self.config.get("superpoint", {})
        self.prob_fusion = ProbabilityFusion(
            sp_reg_strength=sp_cfg.get("reg_strength", 0.03),
            sp_k_nn_adj=sp_cfg.get("k_nn_adj", 10),
            sp_num_points_per_sp=sp_cfg.get("num_points_per_sp", 20),
            sp_spatial_weight=sp_cfg.get("spatial_weight", 5.0),
            coverage_ratio_thresh=sp_cfg.get("coverage_ratio_thresh", 0.03),
        )

        # 7) 输出目录
        self.results_root = self.res_cfg.get("results_dir", "results_res")
        os.makedirs(self.results_root, exist_ok=True)

    # ==================================================================
    # Public entry: per-instance evaluation
    # ==================================================================
    def evaluate_instance(
        self,
        uid: str,
        caption_types: Optional[List[str]] = None,
        max_queries: Optional[int] = None,
        out_dir: Optional[str] = None,
    ) -> List[RESPerQueryResult]:
        """
        对单个 PartVerse instance 跑所有 (part_id × caption_type) query。

        Returns:
            per-query 评测结果列表
        """
        instance = self.loader.load_instance(uid)
        if not os.path.exists(instance.mesh_path):
            print(f"  [RESPipeline] mesh missing for {uid}; skip.")
            return []

        caption_types = caption_types or self.res_cfg.get("caption_types", ["brief", "detailed"])
        if max_queries is None:
            max_queries = self.res_cfg.get("max_queries_per_instance", None)

        out_dir = out_dir or os.path.join(self.results_root, uid)
        os.makedirs(out_dir, exist_ok=True)

        # ===================== Stage 0: 共享准备 =====================
        ctx = self._prepare_shared_context(instance, out_dir)
        if ctx is None:
            return []

        # ===================== Stage 1+: per query =====================
        queries = list(self.loader.iter_queries(instance, tuple(caption_types), max_queries))
        print(f"  [RESPipeline] {uid}: {len(queries)} queries to evaluate.")

        per_query_results: List[RESPerQueryResult] = []
        for q in queries:
            try:
                result = self._run_single_query(ctx, q, out_dir)
                if result is not None:
                    per_query_results.append(result)
            except Exception as e:
                print(f"  [RESPipeline] {uid} part {q.part_id} ({q.caption_type}) failed: {e}")
                import traceback
                traceback.print_exc()

        # 释放共享 context（释放显存）
        self._release_shared_context(ctx)

        # 保存 per-instance 汇总
        if per_query_results:
            inst_agg = aggregate_results(per_query_results)
            with open(os.path.join(out_dir, "instance_results.json"), "w", encoding="utf-8") as f:
                json.dump(to_serializable(inst_agg), f, indent=2, ensure_ascii=False)
            with open(os.path.join(out_dir, "instance_summary.txt"), "w", encoding="utf-8") as f:
                f.write(format_aggregate_summary(inst_agg))

        return per_query_results

    # ==================================================================
    # Stage 0: shared context（per-instance）
    # ==================================================================
    def _prepare_shared_context(self, instance: PartVerseInstance, out_dir: str) -> Optional[Dict]:
        """采样点云 + 多视角渲染 + SAM + K_0 + 4 视角 collage 渲染。所有 query 共享。"""
        print(f"\n  [RESPipeline] Preparing shared context for {instance.uid} ...")

        # ---- 1) mesh -> 点云 + face_id + GT label ----
        num_points = int(self.res_cfg.get("num_points", 10000))
        try:
            pcd_raw, pc_xyz_raw, pc_colors_raw, point_face_id, point_gt_label = \
                self.loader.load_pointcloud_with_faces(instance, num_points=num_points)
        except Exception as e:
            print(f"  [RESPipeline] load_pointcloud_with_faces failed for {instance.uid}: {e}")
            return None

        # 归一化（与 dataloader.normalize_pc 等价）
        centroid = pc_xyz_raw.mean(axis=0)
        centered = pc_xyz_raw - centroid
        max_dist = float(np.max(np.linalg.norm(centered, axis=1)))
        scale = 1.0 / max_dist if max_dist > 1e-9 else 1.0
        pc_xyz = (centered * scale).astype(np.float32)
        pc_colors = pc_colors_raw.astype(np.float32)

        norm_pcd = o3d.geometry.PointCloud()
        norm_pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        norm_pcd.colors = o3d.utility.Vector3dVector(pc_colors)
        norm_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        # ---- 2) 多视角渲染 ----
        cam_positions, cam_rotations = generate_sphere_cameras(
            self.config["render"]["num_cameras"],
            radius=self.config["render"]["camera_distance"],
        )
        view_angles_str = []
        for pos in cam_positions:
            r = float(np.linalg.norm(pos))
            elev = float(np.degrees(np.arcsin(pos[1] / r)))
            azim = float(np.degrees(np.arctan2(pos[0], pos[2])))
            view_angles_str.append(f"仰角 {elev:.1f}度, 方位角 {azim:.1f}度")

        render_output = self.renderer.render(pc_xyz, pc_colors, cam_positions, cam_rotations)

        # 边缘增强（与 pipeline.py 一致）
        if self.edge_enhancer.enable_enhancement:
            enhanced = []
            for img in render_output.images:
                enhanced.append(self.edge_enhancer.enhance_pc_render(img))
            render_output = RenderOutput(
                images=enhanced,
                depth_maps=render_output.depth_maps,
                index_maps=render_output.index_maps,
            )

        # ---- 3) SAM 分割 ----
        masks_across_views: List[List[np.ndarray]] = []
        for img in render_output.images:
            masks = self.sam_auto.segment(img)
            masks_across_views.append(masks)

        # ---- 4) K_0 几何特征 ----
        feat_extractor = GeometricFeatureExtractor(pc_xyz)
        initial_instance_knowledge: Dict[str, Dict[str, dict]] = {}
        for vid in range(len(render_output.images)):
            view_key = f"view_{vid}"
            initial_instance_knowledge[view_key] = {}
            idx_map = render_output.index_maps[vid]
            if idx_map.ndim == 3:
                idx_map = idx_map[:, :, 0]
            for m_idx, mask in enumerate(masks_across_views[vid]):
                ys, xs = np.where(mask > 0)
                pc_indices = idx_map[ys, xs]
                valid = pc_indices[pc_indices >= 0]
                if len(valid) > 0:
                    feats = feat_extractor.compute_features(pc_xyz[valid])
                else:
                    feats = feat_extractor._empty_features()
                initial_instance_knowledge[view_key][f"mask_{m_idx}"] = feats

        # ---- 5) 4 视角 collage（Distractor / Topology 共用） ----
        azs = self.res_cfg.get("distractor_view_azimuths", [0, 90, 180, 270])
        elev = float(self.res_cfg.get("distractor_view_elevation", 20))
        collage_views = [(elev, az) for az in azs]
        collage_pos, collage_rot = generate_cameras_from_views(
            collage_views, radius=self.config["render"]["camera_distance"]
        )
        collage_render = self.renderer.render(pc_xyz, pc_colors, collage_pos, collage_rot)
        collage_imgs = list(collage_render.images)

        # ---- 6) 保存共享可视化 ----
        shared_vis_dir = os.path.join(out_dir, "shared_context")
        os.makedirs(shared_vis_dir, exist_ok=True)
        for vid, img in enumerate(render_output.images):
            cv2.imwrite(
                os.path.join(shared_vis_dir, f"view_{vid:02d}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
        for vid, img in enumerate(collage_imgs):
            cv2.imwrite(
                os.path.join(shared_vis_dir, f"collage_{vid:02d}_az{azs[vid]:03d}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
        with open(os.path.join(shared_vis_dir, "view_angles.json"), "w", encoding="utf-8") as f:
            json.dump(view_angles_str, f, indent=2, ensure_ascii=False)

        return {
            "instance": instance,
            "pc_xyz": pc_xyz,
            "pc_colors": pc_colors,
            "norm_pcd": norm_pcd,
            "point_face_id": point_face_id,
            "point_gt_label": point_gt_label,
            "render_output": render_output,
            "view_angles_str": view_angles_str,
            "masks_across_views": masks_across_views,
            "initial_instance_knowledge": initial_instance_knowledge,
            "collage_imgs": collage_imgs,
            "collage_elevation": elev,
            "shared_vis_dir": shared_vis_dir,
        }

    @staticmethod
    def _release_shared_context(ctx: Dict):
        """释放 GPU/内存（render_output 中的 tensor 可能很大）。"""
        for k in list(ctx.keys()):
            ctx[k] = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    # ==================================================================
    # Stage 1+: per-query
    # ==================================================================
    def _run_single_query(
        self,
        ctx: Dict,
        query: PartVerseQuery,
        out_dir: str,
    ) -> Optional[RESPerQueryResult]:
        instance: PartVerseInstance = ctx["instance"]
        query_dir = os.path.join(
            out_dir, "queries", f"part{query.part_id}_{query.caption_type}_{query.query_hash}"
        )
        os.makedirs(query_dir, exist_ok=True)

        meta = {
            "uid": instance.uid,
            "part_id": query.part_id,
            "caption_type": query.caption_type,
            "brief": query.brief,
            "detailed": query.detailed,
            "gt_part_label": query.gt_part_label,
        }
        print(f"\n  [{instance.uid}] part={query.part_id} type={query.caption_type}: \"{query.brief[:60]}\"")

        # ---------- 1.A) Target K_u 重构 ----------
        if self.res_cfg.get("use_target_K_u_reconstruction", True):
            target_ku = self.target_reconstructor.reconstruct(
                brief=query.brief, detailed=query.detailed
            )
        else:
            target_ku = {
                "short_label": "target",
                "spatial_location": "未提及",
                "typical_3D_shape": "未提及",
                "connection_context": "未提及",
                "relative_size": "未提及",
                "z_height_range": [0.0, 1.0],
            }
        with open(os.path.join(query_dir, "target_ku.json"), "w", encoding="utf-8") as f:
            json.dump(target_ku, f, indent=2, ensure_ascii=False)

        # ---------- 1.B) Distractor 发现 ----------
        if self.res_cfg.get("use_distractor_discovery", True):
            distractors = self.distractor_discoverer.discover(
                view_images_rgb=ctx["collage_imgs"],
                target_ku=target_ku,
                elevation=ctx["collage_elevation"],
                max_distractors=int(self.res_cfg.get("max_distractors", 8)),
            )
        else:
            distractors = []
        with open(os.path.join(query_dir, "distractors.json"), "w", encoding="utf-8") as f:
            json.dump(distractors, f, indent=2, ensure_ascii=False)

        # ---------- 2) 第一轮 RES 分类 ----------
        soft_masks_across_views, som_labeled, raw_preds, som_raw = \
            self.res_classifier.predict_res_all_views(
                images_rgb=list(ctx["render_output"].images),
                depth_maps=list(ctx["render_output"].depth_maps),
                masks_across_views=ctx["masks_across_views"],
                view_angles_str=ctx["view_angles_str"],
                target_ku=target_ku,
                distractors=distractors,
                initial_instance_knowledge=ctx["initial_instance_knowledge"],
            )
        # 保存 som 可视化
        som_dir = os.path.join(query_dir, "step2_first_round_som")
        os.makedirs(som_dir, exist_ok=True)
        for vid, batch_list in enumerate(som_labeled or []):
            for b_idx, img in enumerate(batch_list):
                cv2.imwrite(
                    os.path.join(som_dir, f"view_{vid:02d}_batch_{b_idx:02d}_labeled.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )
        with open(os.path.join(query_dir, "step2_first_round_predictions.json"), "w", encoding="utf-8") as f:
            json.dump(
                {f"view_{i:02d}": v for i, v in enumerate(raw_preds)},
                f, indent=2, ensure_ascii=False,
            )

        # 命中语义集合
        target_label = target_ku.get("short_label", "target")
        hit_semantics_set = set()
        for view_sm in soft_masks_across_views:
            for sm in view_sm:
                if not sm.probabilities:
                    continue
                best = max(sm.probabilities, key=sm.probabilities.get)
                if best != "unknown":
                    hit_semantics_set.add(best)
        hit_semantics = sorted(hit_semantics_set)

        semantics_across_views: List[List[str]] = []
        for view_sm in soft_masks_across_views:
            view_sems = []
            for sm in view_sm:
                if sm.probabilities:
                    view_sems.append(max(sm.probabilities, key=sm.probabilities.get))
                else:
                    view_sems.append("unknown")
            semantics_across_views.append(view_sems)

        # ---------- 3) 在线拓扑推断 + 4) 拓扑构建 + 违规检测 + 5) 二轮复审 ----------
        topology_text = ""
        topology_violations: List[Dict] = []
        if self.res_cfg.get("use_online_topology", True) and len(hit_semantics) >= 2:
            topo_inf = self.topology_inferrer.infer(
                view_images_rgb=ctx["collage_imgs"],
                hit_semantics=hit_semantics,
                elevation=ctx["collage_elevation"],
            )
            with open(os.path.join(query_dir, "topology_inference.json"), "w", encoding="utf-8") as f:
                json.dump(topo_inf, f, indent=2, ensure_ascii=False)

            # 复用 TopologyGraphBuilder
            unified_knowledge_lite = {target_label: {"z_height_range": target_ku.get("z_height_range", [0.0, 1.0])}}
            for d in distractors:
                unified_knowledge_lite[d["short_label"]] = {"z_height_range": [0.0, 1.0]}

            topo_builder = TopologyGraphBuilder(
                ctx["pc_xyz"], ctx["render_output"].index_maps,
                unified_knowledge=unified_knowledge_lite,
                object_category="partverse_object",
                spatial_height_order=topo_inf.get("spatial_height_order", {}),
            )
            point_semantics = topo_builder.build_semantic_point_cloud(
                soft_masks_across_views, confidence_threshold=0.70
            )
            part_regions = topo_builder.build_part_regions(point_semantics)
            global_topology = topo_builder.build_global_topology(part_regions, adjacency_threshold=0.05)
            reasonable_adjacencies = topo_inf.get("reasonable_adjacencies", {})
            topology_violations = topo_builder.validate_masks_topology(
                soft_masks_across_views, semantics_across_views,
                part_regions, global_topology,
                reasonable_adjacencies=reasonable_adjacencies,
            )
            topology_text = topo_builder.to_prompt_text(global_topology, topology_violations)
            with open(os.path.join(query_dir, "global_topology.txt"), "w", encoding="utf-8") as f:
                f.write(topology_text)

            # ---------- 5) target-only 第二轮复审 ----------
            if self.res_cfg.get("enable_second_round", True) and topology_violations:
                target_only = bool(self.res_cfg.get("second_round_target_only", True))
                # 按视角聚合 suspect
                from module_3d_lift.geometric_features import GeometricFeatureExtractor as _GFE
                _ = _GFE  # 仅提示依赖，不实例化
                from config.feature_mapping import map_geometric_features_to_text

                suspects_by_view: Dict[int, List[Dict]] = {}
                for v_info in topology_violations:
                    if target_only and v_info["current_semantic"] != target_label:
                        continue
                    vid = v_info["view_idx"]
                    suspects_by_view.setdefault(vid, []).append({
                        "mask_idx": v_info["mask_idx"],
                        "global_id": v_info.get("global_id", f"v{vid}m{v_info['mask_idx']}"),
                        "first_round_semantic": v_info["current_semantic"],
                        "violations": v_info["violations"],
                        "mask_height": v_info["mask_height"],
                        "neighbors": v_info.get("neighbors", []),
                        "topology_relations": v_info.get("topology_relations", []),
                        "reason": "; ".join(v_info["violations"]),
                    })

                # 并行复审
                second_round_dir = os.path.join(query_dir, "step5_second_round_som")
                os.makedirs(second_round_dir, exist_ok=True)

                def _review_view(vid: int):
                    view_suspects = suspects_by_view[vid]
                    view_inst_know_text: Dict[str, dict] = {}
                    for m_key, raw_feat in ctx["initial_instance_knowledge"].get(f"view_{vid}", {}).items():
                        view_inst_know_text[m_key] = map_geometric_features_to_text(raw_feat)
                    corrected, view_som_raw, batch_decs = \
                        self.res_second_round.predict_res_suspect_masks(
                            vid=vid,
                            image=ctx["render_output"].images[vid],
                            depth_map=ctx["render_output"].depth_maps[vid],
                            suspect_masks=view_suspects,
                            masks_in_view=ctx["masks_across_views"][vid],
                            view_angle=ctx["view_angles_str"][vid],
                            target_ku=target_ku,
                            distractors=distractors,
                            text_view_instance_knowledge=view_inst_know_text,
                            global_topology_text=topology_text,
                        )
                    return vid, view_suspects, corrected, view_som_raw, batch_decs

                with ThreadPoolExecutor(max_workers=4) as ex:
                    futures = [ex.submit(_review_view, vid) for vid in suspects_by_view.keys()]
                    for fut in as_completed(futures):
                        vid, view_suspects, corrected, raw_imgs, batch_decs = fut.result()
                        # 把 corrected mask 回写到 soft_masks_across_views
                        for sm, c in zip(view_suspects, corrected):
                            soft_masks_across_views[vid][sm["mask_idx"]] = c
                        for b_idx, img in enumerate(raw_imgs):
                            cv2.imwrite(
                                os.path.join(second_round_dir, f"view_{vid:02d}_batch_{b_idx:02d}_raw.png"),
                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            )
                with open(os.path.join(query_dir, "step5_second_round_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {f"view_{vid}": [s["mask_idx"] for s in suspects_by_view.get(vid, [])]
                         for vid in suspects_by_view},
                        f, indent=2, ensure_ascii=False,
                    )

        # ---------- 6) 3D 软投影 + 概率融合 ----------
        # 仅保留 target_label / distractor / unknown 三类做融合（避免 prob_fusion 过滤掉 unknown 之外的语义）
        prompt_classes = [target_label] + [d["short_label"] for d in distractors] + ["unknown"]

        projected_data = self.soft_projector.project_with_depth_check(
            soft_masks_across_views,
            list(ctx["render_output"].depth_maps),
            list(ctx["render_output"].index_maps),
            ctx["pc_xyz"],
        )
        # 构造 prob_fusion 期望的 unified_knowledge JSON
        anatomy_dict = {target_label: {"z_height_range": target_ku.get("z_height_range", [0.0, 1.0])}}
        for d in distractors:
            anatomy_dict[d["short_label"]] = {"z_height_range": [0.0, 1.0]}
        anatomy_dict["unknown"] = {"z_height_range": [0.0, 1.0]}
        anatomy_json = json.dumps(anatomy_dict, ensure_ascii=False)

        try:
            prob_field = self.prob_fusion.fuse_probabilities(
                projected_data,
                len(ctx["pc_xyz"]),
                prompt_classes,
                ctx["norm_pcd"],
                anatomy_knowledge_json_str=anatomy_json,
            )
            final_label_idx = np.argmax(prob_field.probabilities, axis=1)
            target_idx = prompt_classes.index(target_label)
            pred_target_mask = (final_label_idx == target_idx)
        except Exception as e:
            print(f"  [RESPipeline] prob_fusion failed; fallback to simple voting: {e}")
            # Fallback: per-point 投票（仅 target / non-target 二分）
            pred_target_mask = self._fallback_target_mask(
                projected_data, len(ctx["pc_xyz"]), target_label
            )

        # ---------- 7) GT 比对 + IoU ----------
        gt_mask = (ctx["point_gt_label"] == query.gt_part_label)
        inter, union, pred_count, gt_count = compute_query_iou(pred_target_mask, gt_mask)
        result = RESPerQueryResult(
            uid=instance.uid,
            part_id=query.part_id,
            caption_type=query.caption_type,
            intersection=inter,
            pred_count=pred_count,
            gt_count=gt_count,
            union=union,
        )

        # 写每条 query 的结果
        meta_out = dict(meta)
        meta_out.update({
            "target_label": target_label,
            "num_distractors": len(distractors),
            "hit_semantics": hit_semantics,
            "num_topology_violations": len(topology_violations),
            "intersection": inter,
            "union": union,
            "pred_count": pred_count,
            "gt_count": gt_count,
            "iou": result.iou,
        })
        with open(os.path.join(query_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False)

        # 保存预测点云（target 部分高亮红色）
        try:
            self._save_pred_pointcloud(
                pc_xyz=ctx["pc_xyz"],
                pc_colors=ctx["pc_colors"],
                pred_mask=pred_target_mask,
                gt_mask=gt_mask,
                save_path=os.path.join(query_dir, "pred_target.ply"),
            )
        except Exception as e:
            print(f"    [save ply] {e}")

        print(
            f"    iou={result.iou:.4f}  pred={pred_count}  gt={gt_count}  inter={inter}"
        )
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_target_mask(
        projected_data: List[Dict],
        n_points: int,
        target_label: str,
    ) -> np.ndarray:
        """fuse_probabilities 失败时的兜底：直接累加每点 target 概率与其他概率，谁大归谁。"""
        target_acc = np.zeros(n_points, dtype=np.float64)
        other_acc = np.zeros(n_points, dtype=np.float64)
        for entry in projected_data:
            pts = entry.get("point_indices") or []
            probs = entry.get("probabilities") or {}
            t = float(probs.get(target_label, 0.0))
            o = sum(float(v) for k, v in probs.items() if k != target_label)
            if not pts:
                continue
            arr = np.asarray(pts, dtype=np.int64)
            target_acc[arr] += t
            other_acc[arr] += o
        return target_acc > other_acc

    @staticmethod
    def _save_pred_pointcloud(
        pc_xyz: np.ndarray,
        pc_colors: np.ndarray,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        save_path: str,
    ):
        """保存预测可视化点云：TP=绿，FP=红，FN=蓝，TN=灰。"""
        colors = np.tile(np.array([0.6, 0.6, 0.6], dtype=np.float32), (len(pc_xyz), 1))
        tp = np.logical_and(pred_mask, gt_mask)
        fp = np.logical_and(pred_mask, np.logical_not(gt_mask))
        fn = np.logical_and(np.logical_not(pred_mask), gt_mask)
        colors[tp] = (0.10, 0.85, 0.10)
        colors[fp] = (0.95, 0.20, 0.20)
        colors[fn] = (0.20, 0.40, 0.95)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(save_path, pcd, write_ascii=False)


# ----------------------------------------------------------------------
# Multi-instance evaluation helper
# ----------------------------------------------------------------------
def evaluate_uids(
    config_path: str,
    uids: List[str],
    gpu_index: int = -1,
    caption_types: Optional[List[str]] = None,
    max_queries: Optional[int] = None,
    out_root: Optional[str] = None,
):
    """评测多个 instance，返回聚合结果。"""
    pipeline = RESPipeline(config_path=config_path, gpu_index=gpu_index)
    out_root = out_root or pipeline.results_root
    os.makedirs(out_root, exist_ok=True)

    all_results: List[RESPerQueryResult] = []
    for i, uid in enumerate(uids):
        print(f"\n========== [{i+1}/{len(uids)}] {uid} ==========")
        try:
            results = pipeline.evaluate_instance(
                uid=uid,
                caption_types=caption_types,
                max_queries=max_queries,
                out_dir=os.path.join(out_root, uid),
            )
            all_results.extend(results)
        except Exception as e:
            print(f"[evaluate_uids] {uid} failed: {e}")
            import traceback
            traceback.print_exc()

    # 聚合
    agg = aggregate_results(all_results)
    summary_text = format_aggregate_summary(agg)
    print("\n" + summary_text)

    with open(os.path.join(out_root, "global_results.json"), "w", encoding="utf-8") as f:
        json.dump(to_serializable(agg), f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_root, "global_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)
    return agg
