import os
import sys
import yaml
import numpy as np
import open3d as o3d
import cv2
import json
import gc
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional


# ============================================================
# GPU 自动选择工具（自动寻找显存最充裕的卡）
# ============================================================
def _auto_select_best_gpu(config_path: str = "config/config.yaml") -> int:
    """
    遍历所有可见 GPU，选择「显存剩余量 >= config 中阈值」且最大的那张。
    如果所有卡都不够阈值，直接抛出异常终止程序。
    无 torch 或无 GPU 时返回 -1。

    Args:
        config_path: config/config.yaml 路径，从中读取 gpu.min_free_gb。
    """
    free_gb_threshold = 4.0
    try:
        import yaml
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            free_gb_threshold = float(
                cfg.get("gpu", {}).get("min_free_gb", 4.0)
            )
    except Exception:
        pass

    def _query_nvidia_smi() -> dict:
        """调用 nvidia-smi 获取系统级显存占用（所有进程），返回 {gpu_idx: free_gb}。"""
        try:
            import subprocess
            result = subprocess.run(
                [
                    "nvidia-smi", "--query-gpu=index,memory.free,memory.total",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return {}
            free_gb_map = {}
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                idx = int(parts[0])
                free_mb = float(parts[1])
                free_gb_map[idx] = free_mb / 1024.0
            return free_gb_map
        except Exception:
            return {}

    try:
        import torch
        if not torch.cuda.is_available():
            return -1

        free_gb_map = _query_nvidia_smi()

        best_idx = None
        best_free = -1.0
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024 ** 3)

            if i in free_gb_map:
                free_gb = free_gb_map[i]
            else:
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                free_gb = total_gb - allocated

            print(f"    [GPU] cuda:{i}  总显存={total_gb:.1f} GB，剩余={free_gb:.1f} GB")

            if free_gb >= free_gb_threshold and free_gb > best_free:
                best_free = free_gb
                best_idx = i

        # 所有卡均低于阈值 → 直接终止
        if best_idx is None:
            print(f"    [GPU] ❌ 所有 GPU 剩余显存均低于阈值 {free_gb_threshold:.1f} GB，程序终止。")
            print(f"    [GPU]    请等待其他任务释放 GPU，或调低 config/config.yaml 中的 gpu.min_free_gb。")
            raise RuntimeError(f"No GPU with >= {free_gb_threshold:.1f} GB free memory. Aborting.")

        torch.cuda.set_device(best_idx)
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        print(f"    [GPU] 选用 cuda:{best_idx}（剩余显存 ~{best_free:.1f} GB，阈值 >= {free_gb_threshold:.1f} GB）")
        return best_idx
    except Exception as e:
        print(f"    [GPU] 自动选择失败: {e}，回退到 cuda:0")
        return 0


def _set_torch_device(gpu_index: int = -1, config_path: str = "config/config.yaml") -> str:
    """
    设置 PyTorch 使用的 CUDA 设备。
    gpu_index >= 0：使用指定逻辑索引；-1：自动寻找最空闲卡。

    Args:
        gpu_index: GPU 逻辑索引。-1 = 自动选择，>= 0 = 固定卡号。
        config_path: config/config.yaml 路径，从中读取显存阈值。
    """
    try:
        import torch
        if gpu_index >= 0:
            torch.cuda.set_device(gpu_index)
            return f"cuda:{gpu_index}"
        selected = _auto_select_best_gpu(config_path)
        if selected < 0:
            return "cpu"
        return f"cuda:{selected}"
    except Exception:
        return "cpu"
# ============================================================

# ============================================================

# 将根目录加入环境变量，防止模块导入失败
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datatypes import RenderOutput, SoftMask2D, PointProbabilityField, ViewInstanceKnowledge, InstanceKnowledge
from data_prep.dataloader import PointCloudDataset
from module_render.camera_poses import generate_cameras_from_views, generate_sphere_cameras
from module_render.renderer import MultiViewRenderer
from module_2d_fm.sam_auto_segmenter import SAMAutoSegmenter
from module_2d_fm.mllm_classifier import MLLMClassifier
from module_2d_fm.image_enhancer import PointCloudEdgeEnhancer, create_edge_enhancer
from module_2d_fm.refine_module import SegmentationRefiner, _normalize_fragment_id
from module_2d_fm.semantic_mask_grouping import (
    SemanticGroupingResult,
    build_semantic_grouping,
    render_semantic_grouped_overlay,
    semantic_string_to_display_id,
    fragments_for_semantic_ordered,
)
from scripts.process_masks import process_masks
# ---> 新增导入：知识库管理器 <---
from module_mllm.knowledge_manager import CategoryKnowledgeManager
from module_mllm.pose_checker import (
    DEFAULT_POSE_CHECK_VIEWS,
    PosePlausibilityChecker,
    adapt_global_topology_text_for_pose,
    filter_text_instance_knowledge_for_no_height,
    filter_unified_knowledge_for_no_height,
    strip_height_from_global_topology_text,
)
from module_3d_lift.soft_projector import SoftProjector
from module_3d_lift.probability_fusion import ProbabilityFusion
from evaluation.visualizer import Visualizer3D

from module_3d_lift.geometric_features import GeometricFeatureExtractor
from config.feature_mapping import map_geometric_features_to_text

def _load_gt_semantic_seg(config: dict, point_cloud_filename: str,
                          norm_pcd) -> Optional[np.ndarray]:
    """
    统一加载 GT 语义标签（per-point），兼容 PartNetE 和 PartObjaverse-Tiny。

    返回:
        (N,) int64 数组（与 norm_pcd 点数一致），加载失败返回 None。
    """
    base_path = config.get('data', {}).get('base_path', '')
    full_pc_path = os.path.join(base_path, point_cloud_filename)

    # ---------- 路径 1: PartNetE（pc.ply → label.npy） ----------
    label_path = full_pc_path.replace("pc.ply", "label.npy")
    if label_path != full_pc_path and label_path.endswith(".npy"):
        if not os.path.exists(label_path):
            alt = point_cloud_filename.replace("pc.ply", "label.npy")
            if os.path.exists(alt):
                label_path = alt
        if os.path.exists(label_path):
            try:
                gt_data = np.load(label_path, allow_pickle=True).item()
                seg = gt_data.get('semantic_seg', None)
                if seg is not None:
                    return np.asarray(seg, dtype=np.int64)
            except Exception:
                pass

    # ---------- 路径 2: PartObjaverse-Tiny（per-face GT, trimesh 密集采样） ----------
    pot_cfg = config.get('partobjaverse_tiny', {})
    ext = os.path.splitext(point_cloud_filename)[1].lower()
    if ext in ('.glb', '.gltf', '.obj', '.stl', '.fbx') and pot_cfg:
        pot_base = pot_cfg.get('base_path', base_path)
        gt_subdir = pot_cfg.get('gt_subdir', 'PartObjaverse-Tiny_semantic_gt')
        uid = os.path.splitext(os.path.basename(point_cloud_filename))[0]
        gt_path = os.path.join(pot_base, gt_subdir, f"{uid}.npy")
        mesh_path = full_pc_path

        if os.path.exists(gt_path) and os.path.exists(mesh_path):
            try:
                import trimesh
                import open3d as o3d

                gt_labels = np.load(gt_path, allow_pickle=True)
                if isinstance(gt_labels, np.ndarray) and gt_labels.ndim == 0:
                    gt_labels = gt_labels.item()
                if isinstance(gt_labels, dict):
                    gt_labels = gt_labels.get('labels', gt_labels.get('label', None))
                    if gt_labels is None:
                        return None
                gt_labels = np.asarray(gt_labels, dtype=np.int64).ravel()

                tm = trimesh.load(mesh_path)
                if isinstance(tm, trimesh.Scene):
                    tm = tm.dump(concatenate=True)
                n_verts = len(tm.vertices)
                n_faces = len(tm.faces)

                sampled_pts = np.asarray(norm_pcd.points)
                n_pred = len(sampled_pts)

                if len(gt_labels) == n_faces:
                    n_sample = max(n_pred * 3, 500000)
                    ref_pts, face_idx = tm.sample(n_sample, return_index=True)
                    ref_pts = np.asarray(ref_pts, dtype=np.float64)
                    ref_labels = gt_labels[face_idx]
                elif len(gt_labels) == n_verts:
                    ref_pts = np.asarray(tm.vertices, dtype=np.float64)
                    ref_labels = gt_labels
                else:
                    return None

                centroid = ref_pts.mean(axis=0)
                centered = ref_pts - centroid
                max_dist = np.max(np.sqrt(np.sum(centered ** 2, axis=1)))
                pts_norm = centered / max_dist if max_dist > 0 else centered

                ref_pcd = o3d.geometry.PointCloud()
                ref_pcd.points = o3d.utility.Vector3dVector(pts_norm)
                kdtree = o3d.geometry.KDTreeFlann(ref_pcd)

                per_point_gt = np.zeros(n_pred, dtype=np.int64)
                for i, pt in enumerate(sampled_pts):
                    _, idx, _ = kdtree.search_knn_vector_3d(pt, 1)
                    per_point_gt[i] = ref_labels[idx[0]]
                return per_point_gt
            except Exception as e:
                print(f"  [GT Load] PartObjaverse-Tiny GT 加载失败: {e}")
                import traceback
                traceback.print_exc()
                return None

    return None


def _majority_gt_semantic_for_projected_points(
    mask_gt_labels: np.ndarray,
    part_class_names: List[str],
) -> Tuple[str, Dict[str, int]]:
    """
    对掩码内投影点的 GT 语义索引做多数表决。
    所有 < 0 的索引视为「无部件标注 / unlabeled」，与 prompt 中的 \"unlabeled\" 对齐，
    不再被过滤掉后误用剩余点的部件类别代表整掩码。
    """
    if mask_gt_labels.size == 0:
        return "unlabeled", {}

    neg_count = int(np.sum(mask_gt_labels < 0))
    pos = mask_gt_labels[mask_gt_labels >= 0]
    buckets: List[Tuple[str, int]] = []
    if neg_count > 0:
        buckets.append(("unlabeled", neg_count))
    if pos.size > 0:
        ul, cnts = np.unique(pos, return_counts=True)
        for lab, c in zip(ul, cnts):
            li = int(lab)
            name = part_class_names[li] if 0 <= li < len(part_class_names) else f"unknown_{li}"
            buckets.append((name, int(c)))

    if not buckets:
        return "unlabeled", {}

    majority_name, _ = max(buckets, key=lambda x: x[1])

    dist: Dict[str, int] = {}
    if neg_count > 0:
        dist["unlabeled"] = neg_count
    if pos.size > 0:
        ul, cnts = np.unique(pos, return_counts=True)
        for lab, c in zip(ul, cnts):
            li = int(lab)
            key = part_class_names[li] if 0 <= li < len(part_class_names) else f"unknown_{li}"
            dist[key] = int(c)

    return majority_name, dist


class Arbitr3DPipeline:
    def __init__(self, config_path: str, gpu_index: int = -1):
        """
        Args:
            config_path: config/config.yaml 路径
            gpu_index: GPU 逻辑索引。-1 = 自动寻找显存 >= 4GB 的最空闲卡；
                       >= 0 = 使用 CUDA_VISIBLE_DEVICES 映射后的指定卡。
        """
        # 在任何 torch 调用前先选卡，确保 os.environ["CUDA_VISIBLE_DEVICES"] 影响一致
        torch_device = _set_torch_device(gpu_index, config_path)
        print(f"    [Pipeline] PyTorch device: {torch_device}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        from utils.runtime_config import resolve_api_key, resolve_base_url, resolve_model_name
        api_key = resolve_api_key(self.config["mllm"])
        base_url = resolve_base_url(self.config["mllm"])
        model_name = resolve_model_name(self.config["mllm"])

        # 1. 数据加载
        self.dataloader = PointCloudDataset(self.config['data']['base_path'])

        # 2. 渲染模块
        self.renderer = MultiViewRenderer(
            resolution=tuple(self.config['render']['resolution']),
            point_radius=self.config['render'].get('point_radius', 0.015),
            albedo_min_value=self.config['render'].get('albedo_min_value', 0.55),
        )

        # 3. 2D 基础模型模块
        self.sam_auto = SAMAutoSegmenter(
            self.config['model']['sam_weight_path'],
            self.config['model']['sam_model_type']
        )

        # ---> 新增：图像边缘增强模块 <---
        self.edge_enhancer = create_edge_enhancer(self.config)



        self.mllm_classifier = MLLMClassifier(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name
        )
        self.pose_checker = PosePlausibilityChecker(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name
        )

        # ---> 新增：知识库管理器初始化 <---
        self.knowledge_manager = CategoryKnowledgeManager(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name
        )

        # 4. 3D 映射与融合模块
        self.soft_projector = SoftProjector(
            depth_threshold=self.config['algorithm'].get('depth_threshold', 0.05),
            erode_pixels=self.config['algorithm'].get('mask_erode_pixels', 3),
        )
        sp_cfg = self.config.get('superpoint', {})
        self.prob_fusion = ProbabilityFusion(
            sp_reg_strength=sp_cfg.get('reg_strength', 0.03),
            sp_k_nn_adj=sp_cfg.get('k_nn_adj', 10),
            sp_num_points_per_sp=sp_cfg.get('num_points_per_sp', 20),
            sp_spatial_weight=sp_cfg.get('spatial_weight', 5.0),
            coverage_ratio_thresh=sp_cfg.get('coverage_ratio_thresh', 0.03),
        )

        # ---> 新增：细化模块初始化 <---
        self.refiner = SegmentationRefiner(sam_segmenter=self.sam_auto)

        self.visualizer = Visualizer3D()

    def run(self, point_cloud_filename: str, prompt_classes_str: str, vis_dir: str = None,
            object_category: str = None, num_points: int = 0):
        """
        Args:
            point_cloud_filename: 相对于 data.base_path 的文件路径（.ply/.glb 等）
            prompt_classes_str: 逗号分隔的部件类别名
            vis_dir: 可视化输出目录
            object_category: 物体类别名；None 时从路径自动推断
            num_points: 仅网格文件有效，采样点数（0=使用原始顶点）
        """
        print(f"--- Processing {point_cloud_filename} ---")
        prompt_classes = [c.strip() for c in prompt_classes_str.split(',')]

        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if object_category is None:
            path_parts = point_cloud_filename.replace('\\', '/').split('/')
            object_category = "Object"
            for part in path_parts:
                if part and part[0].isupper() and not part.endswith('.ply'):
                    object_category = part
                    break
        print(f"Detected object category from path: {object_category}")

        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
            knowledge_system_dir = os.path.join(vis_dir, "knowledge_system")
            pose_check_dir = os.path.join(vis_dir, "step0_pose_check")
            step1_dir = os.path.join(vis_dir, "step1_sam")
            step2_dir = os.path.join(vis_dir, "step2_mllm_2d")
            step2_5_dir = os.path.join(vis_dir, "step2_5_mllm_refine")
            step3_dir = os.path.join(vis_dir, "step3_3d_prob")
            step4_dir = os.path.join(vis_dir, "step4_final")
            for d in [knowledge_system_dir, pose_check_dir, step1_dir, step2_dir, step2_5_dir, step3_dir, step4_dir]:
                os.makedirs(d, exist_ok=True)

        pcd = self.dataloader.load_point_cloud(point_cloud_filename, num_points=num_points)
        norm_pcd = self.dataloader.normalize_pc(pcd)
        pc_xyz = np.asarray(norm_pcd.points)
        pc_colors = self.dataloader.extract_color(norm_pcd)

        print("Step 0.5: Pose plausibility pre-check...")
        pose_check_views = self.config.get('render', {}).get('pose_check_views', list(DEFAULT_POSE_CHECK_VIEWS))
        pose_cam_positions, pose_cam_rotations = generate_cameras_from_views(
            pose_check_views,
            radius=self.config['render']['camera_distance']
        )
        pose_render_output = self.renderer.render(pc_xyz, pc_colors, pose_cam_positions, pose_cam_rotations)
        pose_assessment = self.pose_checker.assess_pose(
            images=pose_render_output.images,
            object_category=object_category,
            views=pose_check_views,
        )
        pose_state = "normal" if pose_assessment.get("is_pose_reasonable", True) else "abnormal"
        print(f"  [Pose Check] Result: {pose_state}")
        if pose_assessment.get("reasoning"):
            print(f"  [Pose Check] Reason: {pose_assessment['reasoning']}")
        if vis_dir:
            for i, (img, view) in enumerate(zip(pose_render_output.images, pose_check_views)):
                out_name = f"view_{i:02d}_e{float(view[0]):.0f}_a{float(view[1]):.0f}.png"
                cv2.imwrite(os.path.join(pose_check_dir, out_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            with open(os.path.join(knowledge_system_dir, "pose_assessment.json"), "w", encoding="utf-8") as f:
                json.dump(pose_assessment, f, indent=4, ensure_ascii=False)

        print("Step 1: Rendering multi-views and SAM auto-segmentation...")

        cam_positions, cam_rotations = generate_sphere_cameras(
            self.config['render']['num_cameras'],
            radius=self.config['render']['camera_distance']
        )

        view_angles_str = []
        for pos in cam_positions:
            r = np.linalg.norm(pos)
            elev = np.degrees(np.arcsin(pos[1] / r)) # 仰角
            azim = np.degrees(np.arctan2(pos[0], pos[2])) # 方位角
            view_angles_str.append(f"仰角 {elev:.1f}度, 方位角 {azim:.1f}度")

        render_output = self.renderer.render(pc_xyz, pc_colors, cam_positions, cam_rotations)

        # ---> 渲染后进行 RGB 边缘增强，提升SAM分割效果 <---
        if self.edge_enhancer.enable_enhancement:
            print("Enhancing rendered images for better SAM segmentation...")
            render_output.images = self.edge_enhancer.enhance_batch(
                render_output.images
            )

        masks_across_views = self.sam_auto.generate_masks_automatically(render_output.images)

        # 【阶段 1.2】掩码去交集处理 - 消除掩码之间的重叠和包含关系
        enable_mask_processing = self.config['algorithm'].get('enable_mask_processing', True)
        if enable_mask_processing:
            print("\nStep 1.2: Processing masks to remove intersections and containment relations...")
            mask_min_area = self.config['algorithm'].get('mask_process_min_area', 200)
            mask_containment_thresh = self.config['algorithm'].get('mask_process_containment_thresh', 0.95)
            processed_masks_across_views = []
            for i, masks in enumerate(masks_across_views):
                if masks:
                    processed_masks = process_masks(masks, containment_threshold=mask_containment_thresh, min_area=mask_min_area)
                    # 额外过滤过小的掩码
                    processed_masks = [m for m in processed_masks if np.sum(m) >= mask_min_area]
                    print(f"  View {i}: {len(masks)} masks -> {len(processed_masks)} non-overlapping masks")
                    processed_masks_across_views.append(processed_masks)
                else:
                    processed_masks_across_views.append([])

            masks_across_views = processed_masks_across_views
        else:
            print("\nStep 1.2: Skipped (mask processing disabled in config)")

        # 【阶段 1.3】掩码面积占比过滤 - 剔除占物体渲染像素比例过大的掩码
        mask_max_object_ratio = self.config['algorithm'].get('mask_max_object_ratio', 0.0)
        if mask_max_object_ratio > 0:
            print(f"\nStep 1.3: Filtering masks exceeding {mask_max_object_ratio:.0%} of object pixels...")
            filtered_masks_across_views = []
            for i, (masks, idx_map) in enumerate(
                    zip(masks_across_views, render_output.index_maps)):
                if not masks:
                    filtered_masks_across_views.append([])
                    continue
                imap = idx_map[:, :, 0] if idx_map.ndim == 3 else idx_map
                object_pixels = int(np.sum(imap != -1))
                if object_pixels == 0:
                    filtered_masks_across_views.append(masks)
                    continue
                object_mask = (imap != -1)
                kept = []
                dropped = 0
                for m in masks:
                    mask_on_object = int(np.sum(m & object_mask))
                    ratio = mask_on_object / object_pixels
                    if ratio <= mask_max_object_ratio:
                        kept.append(m)
                    else:
                        dropped += 1
                if dropped > 0:
                    print(f"  View {i}: 剔除 {dropped} 个过大掩码 "
                          f"(物体像素 {object_pixels}，阈值 {mask_max_object_ratio:.0%})")
                filtered_masks_across_views.append(kept)
            masks_across_views = filtered_masks_across_views

        if vis_dir:
            print(f"Saving visualizations to {step1_dir}...")

            raw_render_dir = os.path.join(step1_dir, "raw_render")
            depth_render_dir = os.path.join(step1_dir, "depth_render")
            sam_render_dir = os.path.join(step1_dir, "sam_render")

            os.makedirs(raw_render_dir, exist_ok=True)
            os.makedirs(depth_render_dir, exist_ok=True)
            os.makedirs(sam_render_dir, exist_ok=True)

            for i, (img, masks, depth) in enumerate(zip(render_output.images, masks_across_views, render_output.depth_maps)):
                # 1. 保存原生渲染图
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(raw_render_dir, f"view_{i:02d}_raw.png"), img_bgr)

                # 2. 保存深度图 (高对比度增强，仅在物体有效区域内归一化)
                from utils.depth_visualization import enhance_depth_map
                depth_vis = enhance_depth_map(depth)
                cv2.imwrite(os.path.join(depth_render_dir, f"view_{i:02d}_depth.png"), depth_vis)

                # 3. 保存 SAM 分割图
                vis_img = img_bgr.copy()
                overlay = vis_img.copy()

                # 过滤背景掩码：接触图像边界的掩码通常为背景
                def is_background_mask(mask: np.ndarray) -> bool:
                    """检查掩码是否可能是背景（接触图像边界）"""
                    h, w = mask.shape
                    # 如果掩码覆盖了四条边的任意一条的大部分（>50%），认为是背景
                    top_covered = np.sum(mask[0, :]) > w * 0.5
                    bottom_covered = np.sum(mask[-1, :]) > w * 0.5
                    left_covered = np.sum(mask[:, 0]) > h * 0.5
                    right_covered = np.sum(mask[:, -1]) > h * 0.5
                    return top_covered or bottom_covered or left_covered or right_covered

                # 为每个掩码生成并记录颜色
                mask_colors = []
                for mask in masks:
                    if is_background_mask(mask):
                        mask_colors.append(None)
                        continue
                    color = np.random.randint(50, 255, (3,)).tolist()
                    mask_colors.append(color)
                    overlay[mask.astype(bool)] = color

                # 先做半透明混合
                cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)

                # 再用同一颜色画边界线（不透明）
                for mask, color in zip(masks, mask_colors):
                    if color is None:
                        continue
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_img, contours, -1, color, 2)

                cv2.imwrite(os.path.join(step1_dir, f"view_{i:02d}_sam.png"), vis_img)

                # ===== 【修改阶段 1.5】 优先加载离线知识库，兜底使用在线 CoT 推理 =====
        print("Step 1.5: Loading Offline Knowledge or Building CoT...")

        # 尝试读取本地的 unified_knowledge.json（基于项目根目录定位）
        project_root = os.path.abspath(os.path.dirname(__file__))

        # 读取消融实验开关
        ablation_cfg = self.config.get('ablation', {})
        use_unified_knowledge = ablation_cfg.get('use_unified_knowledge', True)
        use_initial_instance_knowledge = ablation_cfg.get('use_initial_instance_knowledge', True)
        use_updated_instance_knowledge = ablation_cfg.get('use_updated_instance_knowledge', True)
        disable_height_prior = bool(ablation_cfg.get('disable_height_prior', False))
        if not use_unified_knowledge:
            print("  [Ablation] 泛化知识库已禁用")
        if not use_initial_instance_knowledge:
            print("  [Ablation] 初级实例知识库已禁用")
        if not use_updated_instance_knowledge:
            print("  [Ablation] 更新实例知识库已禁用（将跳过全局拓扑构建和第二轮 MLLM 复审）")
        if disable_height_prior:
            print(
                "  [Ablation] 高度先验已禁用（z_height_range 硬过滤 + 拓扑规则 1/4 + "
                "MLLM prompt 中 z_height_range/normalized_z/mask_height/h=[..] 全部剥除）"
            )
        unified_knowledge_path = os.path.join(project_root, "config", "knowledge", f"{object_category}_unified_knowledge.json")

        unified_knowledge = {}
        if os.path.exists(unified_knowledge_path):
            try:
                with open(unified_knowledge_path, 'r', encoding='utf-8') as f:
                    unified_knowledge = json.load(f)
                print(f"  [Info] 成功加载 unified_knowledge.json")
            except Exception as e:
                print(f"  [Error] 无法解析 unified_knowledge.json: {e}")
        else:
            print(f"  [Warning] 未找到 unified_knowledge.json: {unified_knowledge_path}")

        # 按当前实例的部件列表过滤知识库（PartObjaverse-Tiny 等部件不固定的数据集）
        if unified_knowledge and object_category in unified_knowledge:
            full_cat_knowledge = unified_knowledge[object_category]
            prompt_classes_set = set(prompt_classes)
            filtered_cat_knowledge = {
                part: info for part, info in full_cat_knowledge.items()
                if part in prompt_classes_set
            }
            n_total = len(full_cat_knowledge)
            n_matched = len(filtered_cat_knowledge)
            if n_matched < n_total:
                print(f"  [Info] 按实例部件过滤知识库: {n_matched}/{n_total} 部件匹配")
            unified_knowledge = {object_category: filtered_cat_knowledge}

        if not use_unified_knowledge:
            unified_knowledge = {}
            print("  [Ablation] unified_knowledge 已置空")

        # disable_height_prior 消融：从 unified_knowledge 中剥除 z_height_range / y_height_range
        # 该剥除会同时影响：
        #   1) 第一轮 / 第二轮 MLLM prompt 中的 unified_knowledge 文本（自动失效）
        #   2) ProbabilityFusion 的 Z 轴硬过滤（anatomy_dict 不再含 z_height_range）
        #   3) TopologyGraphBuilder.part_height_ranges（→ 拓扑规则1 高度越界自动失效）
        if disable_height_prior and unified_knowledge:
            unified_knowledge = filter_unified_knowledge_for_no_height(
                unified_knowledge, object_category, disable_height_prior
            )

        if vis_dir:
            # 将加载的统一知识库存放到 knowledge_system_dir
            with open(os.path.join(knowledge_system_dir, "unified_knowledge.json"), "w", encoding="utf-8") as f:
                json.dump(unified_knowledge, f, indent=4, ensure_ascii=False)

        # 【阶段 1.6】构建初级实例知识库 (Initial Instance Knowledge Base)
        initial_instance_knowledge = {}
        if use_initial_instance_knowledge:
            print("\nStep 1.6: Building Initial Instance Knowledge Base...")
            feature_extractor = GeometricFeatureExtractor(pc_xyz)

            for vid in range(len(render_output.images)):
                view_key = f"view_{vid}"
                initial_instance_knowledge[view_key] = {}
                masks_in_view = masks_across_views[vid]
                index_map_in_view = render_output.index_maps[vid]

                for m_idx, mask in enumerate(masks_in_view):
                    mask_key = f"mask_{m_idx}"
                    # 将 2D 掩码映射到 3D 点云索引
                    mask_pixels = np.where(mask > 0)
                    pc_indices = index_map_in_view[mask_pixels[0], mask_pixels[1]]
                    valid_pc_indices = pc_indices[pc_indices >= 0]

                    if len(valid_pc_indices) > 0:
                        mask_pts_3d = pc_xyz[valid_pc_indices]
                        raw_features = feature_extractor.compute_features(mask_pts_3d)
                    else:
                        raw_features = feature_extractor._empty_features()

                    initial_instance_knowledge[view_key][mask_key] = raw_features

            if vis_dir:
                with open(os.path.join(knowledge_system_dir, "initial_instance_knowledge.json"), "w", encoding="utf-8") as f:
                    json.dump(initial_instance_knowledge, f, indent=4, ensure_ascii=False)

                # 同时保存一份映射为文本的初级知识库供调试查看
                text_mapped_initial_knowledge = {}
                for v_key, v_dict in initial_instance_knowledge.items():
                    text_mapped_initial_knowledge[v_key] = {}
                    for m_key, raw_feat in v_dict.items():
                        text_mapped_initial_knowledge[v_key][m_key] = map_geometric_features_to_text(raw_feat)
                with open(os.path.join(knowledge_system_dir, "initial_instance_knowledge_text.json"), "w", encoding="utf-8") as f:
                    json.dump(text_mapped_initial_knowledge, f, indent=4, ensure_ascii=False)
        else:
            print("\nStep 1.6: Skipped (use_initial_instance_knowledge=false in config)")

        # 【阶段 2】轻量级 2D 语义初筛（使用统一泛化知识库和初级实例知识库）
        print("Step 2: MLLM soft probability assignment (with CoT Knowledge) [parallel]...")
        mllm_2d_predictions = {}

        # 消融：传给 MLLM 的初级实例知识库
        instance_knowledge_for_mllm = initial_instance_knowledge if use_initial_instance_knowledge else {}
        if not use_initial_instance_knowledge:
            print("  [Ablation] 传给 MLLM 的 initial_instance_knowledge 已置空")

        soft_masks_across_views, som_images_across_views, raw_predictions_across_views, som_images_raw_across_views = \
            self.mllm_classifier.predict_soft_probabilities_all_views(
                images_rgb=render_output.images,
                depth_maps=render_output.depth_maps,  # 传入深度图
                masks_across_views=masks_across_views,
                prompt_classes=prompt_classes,
                object_category=object_category,
                view_angles_str=view_angles_str,
                unified_knowledge=unified_knowledge,
                initial_instance_knowledge=instance_knowledge_for_mllm,
                pose_assessment=pose_assessment,
                disable_height_prior=disable_height_prior,
            )

        if vis_dir:
            for i, (masks, som_images) in enumerate(zip(masks_across_views, som_images_across_views)):
                view_preds = {}
                soft_masks = soft_masks_across_views[i]
                raw_view = raw_predictions_across_views[i] if i < len(raw_predictions_across_views) else {}
                mask_id_to_pred_cls = {}
                for j, sm in enumerate(soft_masks):
                    best_cls = max(sm.probabilities, key=sm.probabilities.get) if sm.probabilities else "background"
                    rp = raw_view.get(j) if isinstance(raw_view, dict) else None
                    if not isinstance(rp, dict):
                        rp = {}
                    view_preds[f"mask_{j}"] = {
                        "predicted_class": best_cls,
                        "probabilities": sm.probabilities,
                        "reasoning": rp.get("reasoning") or (sm.reasoning or ""),
                    }
                    mask_id_to_pred_cls[j] = best_cls

                mllm_2d_predictions[f"view_{i:02d}"] = view_preds

                # 保存原始 SOM 图（与发送给 MLLM 的输入完全一致）
                raw_som_list = som_images_raw_across_views[i] if som_images_raw_across_views[i] else []
                for batch_idx, raw_som in enumerate(raw_som_list):
                    raw_bgr = cv2.cvtColor(raw_som, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(step2_dir, f"view_{i:02d}_batch_{batch_idx:02d}_som.png"), raw_bgr)

                # 保存带 MLLM 语义标注的 SOM 图
                for batch_idx, som_img in enumerate(som_images):
                    som_img_bgr = cv2.cvtColor(som_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(step2_dir, f"view_{i:02d}_batch_{batch_idx:02d}_som_labeled.png"), som_img_bgr)

        # 预计算每个视角的语义标签列表（供 Step 2.5 使用）
        semantics_across_views: List[List[str]] = []
        for soft_masks in soft_masks_across_views:
            view_semantics = []
            for sm in soft_masks:
                best_cls = max(sm.probabilities, key=sm.probabilities.get) if sm.probabilities else "background"
                view_semantics.append(best_cls)
            semantics_across_views.append(view_semantics)

        # ============================================================
        # 【阶段 2.5】全局 3D 拓扑图构建 + 拓扑规则裁判 + 第二轮 MLLM 语义复审
        # ============================================================
        if use_updated_instance_knowledge:
            print("\nStep 2.5: Building Global 3D Topology & 2nd Round MLLM Refinement...")

            # ----- 2.5.1 构建全局 3D 空间拓扑图（= 更新实例知识库） -----
            from module_3d_lift.topology_builder import TopologyGraphBuilder
            spatial_height_order_path = os.path.join(project_root, "config", "knowledge", f"{object_category}_spatial_height_order.json")
            spatial_height_order = {}
            if os.path.exists(spatial_height_order_path):
                try:
                    with open(spatial_height_order_path, 'r', encoding='utf-8') as f:
                        spatial_height_order = json.load(f)
                    print(f"  [Info] Loaded spatial_height_order ({len(spatial_height_order)} rules)")
                except Exception as e:
                    print(f"  [Error] Failed to parse spatial_height_order: {e}")
            else:
                print(f"  [Warning] spatial_height_order not found: {spatial_height_order_path}")
            # 按实例部件过滤
            if spatial_height_order:
                prompt_classes_set_h = set(prompt_classes)
                spatial_height_order = {
                    k: [p for p in v if p in prompt_classes_set_h]
                    for k, v in spatial_height_order.items()
                    if k in prompt_classes_set_h
                }
            # disable_height_prior：清空 spatial_height_order → 拓扑规则4 自动失效
            if disable_height_prior:
                spatial_height_order = {}
            topo_builder = TopologyGraphBuilder(
                pc_xyz, render_output.index_maps,
                unified_knowledge=unified_knowledge,
                object_category=object_category,
                spatial_height_order=spatial_height_order,
            )
            point_semantics = topo_builder.build_semantic_point_cloud(
                soft_masks_across_views, confidence_threshold=0.70
            )
            part_regions = topo_builder.build_part_regions(point_semantics)
            global_topology = topo_builder.build_global_topology(part_regions, adjacency_threshold=0.05)

            # ----- 2.5.3 加载离线合理邻接关系 -----
            reasonable_adj_path = os.path.join(project_root, "config", "knowledge", f"{object_category}_reasonable_adjacencies.json")
            reasonable_adjacencies = {}
            if os.path.exists(reasonable_adj_path):
                try:
                    with open(reasonable_adj_path, 'r', encoding='utf-8') as f:
                        reasonable_adjacencies = json.load(f)
                    print(f"  [Info] Loaded reasonable_adjacencies")
                except Exception as e:
                    print(f"  [Error] Failed to parse reasonable_adjacencies: {e}")
            else:
                print(f"  [Warning] reasonable_adjacencies not found: {reasonable_adj_path}")
            # 按实例部件过滤
            if reasonable_adjacencies:
                prompt_classes_set_a = set(prompt_classes)
                reasonable_adjacencies = {
                    k: [p for p in v if p in prompt_classes_set_a]
                    for k, v in reasonable_adjacencies.items()
                    if k in prompt_classes_set_a
                }
            if vis_dir:
                with open(os.path.join(knowledge_system_dir, "reasonable_adjacencies.json"), "w", encoding="utf-8") as f:
                    json.dump(reasonable_adjacencies, f, indent=4, ensure_ascii=False)

            # ----- 2.5.4 逐掩码拓扑规则裁判 -----
            topology_violations = topo_builder.validate_masks_topology(
                soft_masks_across_views, semantics_across_views,
                part_regions, global_topology,
                reasonable_adjacencies=reasonable_adjacencies,
            )
            global_topology_text = topo_builder.to_prompt_text(global_topology, topology_violations)
            global_topology_text = adapt_global_topology_text_for_pose(global_topology_text, pose_assessment)
            global_topology_text = strip_height_from_global_topology_text(
                global_topology_text, disable_height_prior
            )
            if vis_dir:
                with open(os.path.join(knowledge_system_dir, "global_topology.txt"), "w", encoding="utf-8") as f:
                    f.write(global_topology_text)
                # 构建更新实例知识库 = 拓扑空间关系（合并 part_regions + edges + violations）
                part_regions_serializable = {}
                for sem, info in part_regions.items():
                    part_regions_serializable[sem] = {
                        "base_semantic": info.get("base_semantic", sem),
                        "centroid": info["centroid"].tolist() if hasattr(info["centroid"], "tolist") else list(info["centroid"]),
                        "bbox_min": info["bbox_min"].tolist() if hasattr(info["bbox_min"], "tolist") else list(info["bbox_min"]),
                        "bbox_max": info["bbox_max"].tolist() if hasattr(info["bbox_max"], "tolist") else list(info["bbox_max"]),
                        "height_range": info.get("height_range", (0, 0)),
                        "point_count": info.get("point_count", 0),
                    }
                topo_edges_serializable = []
                for edge in global_topology.get("edges", []):
                    topo_edges_serializable.append({
                        "from": edge["from"],
                        "to": edge["to"],
                        "distance": float(edge["distance"]),
                        "relations": edge["relations"],
                    })
                serializable_violations = [{k: v_ for k, v_ in v.items()} for v in topology_violations]
                updated_instance_knowledge = {
                    "part_regions": part_regions_serializable,
                    "topology_edges": topo_edges_serializable,
                    "topology_violations": serializable_violations,
                }
                with open(os.path.join(knowledge_system_dir, "updated_instance_knowledge.json"), "w", encoding="utf-8") as f:
                    json.dump(updated_instance_knowledge, f, indent=4, ensure_ascii=False)
                print(f"    [Topology] Saved global_topology.txt and updated_instance_knowledge.json")

                # -- 可视化：拓扑违规掩码高亮图 --
                topo_vis_dir = os.path.join(step2_5_dir, "topology_violations_vis")
                os.makedirs(topo_vis_dir, exist_ok=True)
                violations_by_view = {}
                for v_info in topology_violations:
                    vid = v_info["view_idx"]
                    if vid not in violations_by_view:
                        violations_by_view[vid] = []
                    violations_by_view[vid].append(v_info)
                for vid in range(len(render_output.images)):
                    img = render_output.images[vid].copy()
                    view_violations = violations_by_view.get(vid, [])
                    if not view_violations:
                        continue
                    overlay_red = np.zeros_like(img)
                    for v_info in view_violations:
                        m_idx = v_info["mask_idx"]
                        m_array = masks_across_views[vid][m_idx]
                        overlay_red[m_array.astype(bool)] = (255, 60, 60)
                        contours, _ = cv2.findContours(m_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
                    cv2.addWeighted(overlay_red, 0.40, img, 1.0, 0, img)
                    from PIL import Image as PILImage, ImageDraw, ImageFont
                    pil_img = PILImage.fromarray(img)
                    draw = ImageDraw.Draw(pil_img)
                    pil_font = self.mllm_classifier._get_annotation_font(12)
                    for v_info in view_violations:
                        m_idx = v_info["mask_idx"]
                        m_array = masks_across_views[vid][m_idx]
                        yx = np.where(m_array > 0)
                        if yx[0].size > 0:
                            cy, cx = int(np.median(yx[0])), int(np.median(yx[1]))
                            label_text = f"m{m_idx}:{v_info['current_semantic']}"
                            bbox = draw.textbbox((0, 0), label_text, font=pil_font)
                            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                            x0 = cx - tw // 2 - 2
                            y0 = cy - th - 4
                            draw.rectangle([x0, y0, x0 + tw + 4, y0 + th + 6], fill=(0, 0, 0))
                            draw.text((x0 + 2, y0 + 2), label_text, fill=(255, 255, 255), font=pil_font)
                    img = np.asarray(pil_img)
                    cv2.imwrite(os.path.join(topo_vis_dir, f"view_{vid:02d}_violations.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"    [Topology Vis] Saved violation highlight images")

                # -- 可视化：部件区域投影图（每个视角标注所有 part_regions 的 2D 投影） --
                topo_regions_dir = os.path.join(step2_5_dir, "topology_regions_vis")
                os.makedirs(topo_regions_dir, exist_ok=True)
                preset_colors = [
                    (0, 255, 0), (0, 200, 255), (255, 255, 0), (255, 0, 255),
                    (128, 255, 128), (255, 128, 0), (0, 128, 255), (200, 200, 0),
                ]
                # 按 base_semantic 分配颜色（同语义实例共用颜色）
                base_sem_set = list(dict.fromkeys(
                    info.get('base_semantic', k) for k, info in part_regions.items()
                ))
                base_color_map = {bs: preset_colors[i % len(preset_colors)] for i, bs in enumerate(base_sem_set)}
                part_colors_map = {}
                for sem_name, info in part_regions.items():
                    part_colors_map[sem_name] = base_color_map[info.get('base_semantic', sem_name)]

                # 预构建 point_index → semantic 的查找数组，加速像素级查询
                non_empty = [info["point_indices"] for info in part_regions.values() if info["point_indices"]]
                max_pt_idx = (max(max(pi) for pi in non_empty) + 1) if non_empty else 0
                pt_to_sem_id = np.full(max_pt_idx, -1, dtype=np.int32)
                sem_id_list = list(part_regions.keys())
                for sid, sem_name in enumerate(sem_id_list):
                    for pidx in part_regions[sem_name]["point_indices"]:
                        if pidx < max_pt_idx:
                            pt_to_sem_id[pidx] = sid

                for vid in range(len(render_output.images)):
                    img = render_output.images[vid].copy()
                    idx_map = render_output.index_maps[vid]
                    if idx_map.ndim == 3:
                        idx_map = idx_map[:, :, 0]

                    overlay = np.zeros_like(img)
                    # 逐像素查找所属部件
                    valid_mask = (idx_map >= 0) & (idx_map < max_pt_idx)
                    sem_ids_img = np.full(idx_map.shape, -1, dtype=np.int32)
                    sem_ids_img[valid_mask] = pt_to_sem_id[idx_map[valid_mask]]

                    for sid, sem_name in enumerate(sem_id_list):
                        color = part_colors_map[sem_name]
                        region_mask = (sem_ids_img == sid)
                        if not np.any(region_mask):
                            continue
                        overlay[region_mask] = color

                    cv2.addWeighted(overlay, 0.30, img, 1.0, 0, img)

                    # 标注每个部件的 2D centroid + 标签
                    for sid, sem_name in enumerate(sem_id_list):
                        color = part_colors_map[sem_name]
                        ys, xs = np.where(sem_ids_img == sid)
                        if len(ys) == 0:
                            continue
                        cy, cx = int(np.median(ys)), int(np.median(xs))
                        h_range = part_regions[sem_name].get("height_range", (0, 0))
                        label = f"{sem_name} h=[{h_range[0]:.2f}~{h_range[1]:.2f}]"
                        # 绘制十字标记
                        cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)
                        # 绘制标签背景 + 文字
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.35
                        thickness = 1
                        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        cv2.rectangle(img, (cx + 10, cy - th - 4), (cx + 10 + tw + 4, cy + 4), (0, 0, 0), -1)
                        cv2.putText(img, label, (cx + 12, cy - 2), font, font_scale, color, thickness)

                    cv2.imwrite(os.path.join(topo_regions_dir, f"view_{vid:02d}_regions.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"    [Topology Vis] Saved part region projection images")

            # ----- 2.5.5 第二轮 MLLM 语义复审（仅针对拓扑违规掩码） -----
            if len(topology_violations) > 0:
                from module_mllm.second_round_classifier import SecondRoundMLLMClassifier
                second_round_classifier = SecondRoundMLLMClassifier(
                    api_key=self.mllm_classifier.api_key,
                    base_url=self.mllm_classifier.base_url,
                    model_name=self.mllm_classifier.model_name
                )
                suspects_by_view = {}
                for v_info in topology_violations:
                    vid = v_info["view_idx"]
                    if vid not in suspects_by_view:
                        suspects_by_view[vid] = []
                    # disable_height_prior 时把高度数值替换成 N/A，使下游 prompt 自动跳过显式高度行
                    mask_height_val = v_info["mask_height"]
                    topo_relations_val = v_info.get("topology_relations", [])
                    if disable_height_prior:
                        mask_height_val = "N/A"
                        topo_relations_val = [
                            {**tr, "part_height_range": "N/A"}
                            for tr in topo_relations_val
                        ]
                    suspects_by_view[vid].append({
                        "mask_idx": v_info["mask_idx"],
                        "global_id": v_info["global_id"],
                        "first_round_semantic": v_info["current_semantic"],
                        "violations": v_info["violations"],
                        "mask_height": mask_height_val,
                        "neighbors": v_info.get("neighbors", []),
                        "topology_relations": topo_relations_val,
                        "reason": "; ".join(v_info["violations"]),
                    })
                max_masks_per_batch_2nd = 3
                view_batch_info = []
                for vid, view_suspects in suspects_by_view.items():
                    batch_count = math.ceil(len(view_suspects) / max_masks_per_batch_2nd)
                    view_batch_info.append((vid, batch_count))
                view_batch_info.sort(key=lambda x: x[1], reverse=True)
                sorted_view_ids_2nd = [vid for vid, _ in view_batch_info]
                print(f"    [2nd Round MLLM] view batch counts: {view_batch_info}")

                def _predict_one_view_2nd(vid):
                    view_suspects_local = suspects_by_view[vid]
                    view_key = f"view_{vid}"
                    view_instance_know = initial_instance_knowledge.get(view_key, {})
                    text_view_instance_know = {}
                    for m_key, raw_feat in view_instance_know.items():
                        text_view_instance_know[m_key] = map_geometric_features_to_text(raw_feat)
                    # disable_height_prior：剥除 normalized_z / spatial_centroid.y_height
                    text_view_instance_know = filter_text_instance_knowledge_for_no_height(
                        text_view_instance_know, disable_height_prior
                    )
                    corrected_soft_masks, view_som_raw, view_batch_decs = second_round_classifier.predict_suspect_masks(
                        vid=vid,
                        image=render_output.images[vid],
                        depth_map=render_output.depth_maps[vid],
                        suspect_masks=view_suspects_local,
                        masks_in_view=masks_across_views[vid],
                        prompt_classes=prompt_classes,
                        object_category=object_category,
                        view_angle=view_angles_str[vid],
                        unified_knowledge=unified_knowledge,
                        text_view_instance_knowledge=text_view_instance_know,
                        global_topology_text=global_topology_text,
                        pose_assessment=pose_assessment,
                        disable_height_prior=disable_height_prior,
                    )
                    return vid, view_suspects_local, corrected_soft_masks, view_som_raw, view_batch_decs

                step25_som_data = {}
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(_predict_one_view_2nd, vid): vid for vid in sorted_view_ids_2nd}
                    for future in as_completed(futures):
                        vid, view_suspects_local, corrected_soft_masks, view_som_raw, view_batch_decs = future.result()
                        for sm, corrected_sm in zip(view_suspects_local, corrected_soft_masks):
                            m_idx = sm["mask_idx"]
                            soft_masks_across_views[vid][m_idx] = corrected_sm
                        step25_som_data[vid] = (view_som_raw, view_batch_decs)
                if vis_dir:
                    for vid, (raw_soms, batch_decs_list) in step25_som_data.items():
                        for b_idx, (raw_som, batch_decs) in enumerate(zip(raw_soms, batch_decs_list)):
                            raw_bgr = cv2.cvtColor(raw_som, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(os.path.join(step2_5_dir, f"view_{vid:02d}_batch_{b_idx:02d}_som.png"), raw_bgr)
                            labeled_som = raw_som.copy()
                            for m_idx, sem_label in batch_decs.items():
                                m_array = masks_across_views[vid][m_idx]
                                y_indices, x_indices = np.where(m_array > 0)
                                if len(y_indices) > 0:
                                    cx, cy = int(np.mean(x_indices)), int(np.mean(y_indices))
                                    label_text = f"{m_idx}:{sem_label}"
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.35
                                    thickness = 1
                                    (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
                                    pad = 2
                                    rx1 = cx - (tw + 2 * pad) // 2
                                    ry1 = cy - (th + 2 * pad) // 2
                                    cv2.rectangle(labeled_som, (rx1, ry1), (rx1 + tw + 2 * pad, ry1 + th + 2 * pad), (0, 0, 0), -1)
                                    cv2.putText(labeled_som, label_text, (rx1 + pad, ry1 + pad + th), font, font_scale, (255, 255, 255), thickness)
                            labeled_bgr = cv2.cvtColor(labeled_som, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(os.path.join(step2_5_dir, f"view_{vid:02d}_batch_{b_idx:02d}_som_labeled.png"), labeled_bgr)
                semantics_across_views = []
                for soft_masks in soft_masks_across_views:
                    view_semantics = [max(sm.probabilities, key=sm.probabilities.get) if sm.probabilities else "background" for sm in soft_masks]
                    semantics_across_views.append(view_semantics)
            else:
                print("    [2nd Round MLLM] No topology violations found. Skipping 2nd round.")

        else:
            print("\nStep 2.5: Skipped (use_updated_instance_knowledge=false in config)")

        masks_for_step25_eval = masks_across_views
        sems_for_step25_eval = semantics_across_views

        # 【Step 2.5 后处理】可视化 + GT 评估
        if vis_dir:
            step2_5_eval_dir = os.path.join(step2_5_dir, "step2_5_refine_eval")
            os.makedirs(step2_5_eval_dir, exist_ok=True)

            all_views_dir = os.path.join(step2_5_eval_dir, "all_views_all_parts")
            os.makedirs(all_views_dir, exist_ok=True)

            incorrect_dir = os.path.join(step2_5_eval_dir, "incorrect_masks_by_view")
            os.makedirs(incorrect_dir, exist_ok=True)

            eval_masks_per_view = masks_for_step25_eval
            eval_sems_per_view = sems_for_step25_eval

            # 加载 GT（兼容 PartNetE 和 PartObjaverse-Tiny）
            gt_semantic_seg = _load_gt_semantic_seg(self.config, point_cloud_filename, norm_pcd)
            if gt_semantic_seg is not None:
                print(f"  [Step 2.5 Eval] GT 已加载，共 {len(gt_semantic_seg)} 点")
            incorrect_eval_summary = {}

            for vid in range(len(render_output.images)):
                img = render_output.images[vid]
                masks = eval_masks_per_view[vid]
                sems = eval_sems_per_view[vid]
                index_map = render_output.index_maps[vid]
                if index_map.ndim == 3:
                    index_map = index_map[:, :, 0]

                # ---- 子目录 1：总览图 ----
                # 总览图：保留 unlabeled，仅忽略 background；图内标注「编号 + 语义名」
                grouping_all_parts = None
                if isinstance(masks, list) and len(masks) > 0:
                    grouping_all_parts = build_semantic_grouping(
                        masks,
                        sems,
                        ignore_semantics={"background"},
                        merge_adjacent_same_semantic=False,
                    )
                if grouping_all_parts is not None:
                    overlay = render_semantic_grouped_overlay(
                        img,
                        masks,
                        sems,
                        grouping_all_parts,
                        alpha=0.35,
                        contour_thickness=2,
                        label_style="both",
                    )
                    cv2.imwrite(
                        os.path.join(all_views_dir, f"view{vid:02d}_all_parts_overlay.png"),
                        overlay,
                    )
                    meta = {
                        "view_index": vid,
                        "masks": [
                            {"mask_index": i, "semantic": str(sems[i]) if i < len(sems) else "unknown"}
                            for i in range(len(masks))
                        ],
                        "grouping_legend": grouping_all_parts.legend_lines,
                    }
                    with open(
                        os.path.join(all_views_dir, f"view{vid:02d}_mask_semantics.json"),
                        "w",
                        encoding="utf-8",
                    ) as jf:
                        json.dump(meta, jf, indent=2, ensure_ascii=False)

                # ---- 子目录 2：错判掩码图（基于 GT）----
                incorrect_info = []
                if isinstance(masks, list):
                    for mid, mask in enumerate(masks):
                        pixel_indices = np.where(mask)
                        point_indices = index_map[pixel_indices]
                        valid_mask = point_indices != -1
                        valid_pts = point_indices[valid_mask]

                        if len(valid_pts) == 0:
                            continue

                        pred_sem = sems[mid] if mid < len(sems) else "unknown"
                        majority_gt, gt_dist = _majority_gt_semantic_for_projected_points(
                            gt_semantic_seg[valid_pts] if gt_semantic_seg is not None else valid_pts,
                            prompt_classes,
                        )
                        is_correct = (pred_sem == majority_gt)

                        if not is_correct:
                            incorrect_info.append({
                                'mask_id': mid,
                                'pred': pred_sem,
                                'gt': majority_gt,
                                'mask': mask,
                            })

                if incorrect_info:
                    eval_img = img.copy()
                    for info in incorrect_info:
                        m = info['mask']
                        mid = info['mask_id']
                        color = (255, 0, 0)  # RGB 原图上的红色标注
                        overlay = np.zeros_like(eval_img)
                        overlay[m.astype(bool)] = color
                        cv2.addWeighted(overlay, 0.50, eval_img, 1.0, 0, eval_img)
                        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(eval_img, contours, -1, color, 2)
                        yx = np.where(m)
                        if len(yx) >= 2 and yx[0].size > 0:
                            cy, cx = int(np.median(yx[0])), int(np.median(yx[1]))
                            label_text = f"m{mid}:P={info['pred']} GT={info['gt']}"
                            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                            cv2.rectangle(eval_img, (cx - tw // 2 - 2, cy - th - 4), (cx + tw // 2 + 2, cy + 4), (0, 0, 0), -1)
                            cv2.putText(eval_img, label_text, (cx - tw // 2, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    cv2.imwrite(
                        os.path.join(incorrect_dir, f"view{vid:02d}_incorrect_masks.png"),
                        cv2.cvtColor(eval_img, cv2.COLOR_RGB2BGR),
                    )
                    incorrect_eval_summary[f"view_{vid:02d}"] = {
                        "total_masks": len(masks),
                        "incorrect_count": len(incorrect_info),
                        "incorrect_ids": [info['mask_id'] for info in incorrect_info],
                    }

            with open(os.path.join(step2_5_eval_dir, "step2_5_incorrect_eval_summary.json"), "w", encoding="utf-8") as f:
                json.dump(incorrect_eval_summary, f, indent=4, ensure_ascii=False)
            print(f"  [Step 2.5 Eval] Saved to {step2_5_eval_dir}")

        if vis_dir:
            # 加载 GT（兼容 PartNetE 和 PartObjaverse-Tiny）
            gt_semantic_seg_step2 = _load_gt_semantic_seg(self.config, point_cloud_filename, norm_pcd)

            if gt_semantic_seg_step2 is not None:
                print(f"Found ground truth, evaluating MLLM 2D predictions... ({len(gt_semantic_seg_step2)} points)")
                try:
                    gt_semantic_seg = gt_semantic_seg_step2
                    gt_class_names = prompt_classes
                    evaluation_results = {}
                    total_masks = 0
                    correct_masks = 0
                    skipped_pred_background = 0
                    skipped_pred_unlabeled = 0

                    for i, (soft_masks, index_map, img) in enumerate(zip(soft_masks_across_views, render_output.index_maps, render_output.images)):
                        view_eval = {}
                        incorrect_masks_info = []
                        view_sem = semantics_across_views[i] if i < len(semantics_across_views) else []

                        if index_map.ndim == 3:
                            index_map = index_map[:, :, 0]

                        for j, sm in enumerate(soft_masks):
                            mask = sm.mask
                            pixel_indices = np.where(mask)
                            point_indices_in_mask = index_map[pixel_indices]

                            valid_mask = point_indices_in_mask != -1
                            valid_point_indices = point_indices_in_mask[valid_mask]

                            if len(valid_point_indices) == 0:
                                continue

                            mask_gt_labels = gt_semantic_seg[valid_point_indices]
                            majority_gt_class, gt_dist = _majority_gt_semantic_for_projected_points(
                                mask_gt_labels, gt_class_names
                            )

                            if j < len(view_sem):
                                best_pred_cls = view_sem[j]
                            else:
                                best_pred_cls = max(sm.probabilities, key=sm.probabilities.get) if sm.probabilities else "background"

                            pred_lower = str(best_pred_cls).strip().lower()
                            if pred_lower == "background":
                                skipped_pred_background += 1
                                view_eval[f"mask_{j}"] = {
                                    "predicted_class": best_pred_cls,
                                    "gt_class": majority_gt_class,
                                    "is_correct": None,
                                    "excluded_from_accuracy": True,
                                    "reason_excluded": "predicted_class_is_background",
                                    "gt_label_distribution": gt_dist,
                                }
                                continue
                            if pred_lower == "unlabeled":
                                skipped_pred_unlabeled += 1
                                view_eval[f"mask_{j}"] = {
                                    "predicted_class": best_pred_cls,
                                    "gt_class": majority_gt_class,
                                    "is_correct": None,
                                    "excluded_from_accuracy": True,
                                    "reason_excluded": "predicted_class_is_unlabeled",
                                    "gt_label_distribution": gt_dist,
                                }
                                continue

                            is_correct = (best_pred_cls == majority_gt_class)

                            view_eval[f"mask_{j}"] = {
                                "predicted_class": best_pred_cls,
                                "gt_class": majority_gt_class,
                                "is_correct": is_correct,
                                "excluded_from_accuracy": False,
                                "gt_label_distribution": gt_dist,
                            }

                            total_masks += 1
                            if is_correct:
                                correct_masks += 1
                            else:
                                incorrect_masks_info.append({
                                    'mask': mask,
                                    'global_idx': j,
                                    'pred': best_pred_cls,
                                    'gt': majority_gt_class
                                })

                        evaluation_results[f"view_{i:02d}"] = view_eval

                        if incorrect_masks_info:
                            batch_size = 10
                            num_batches = (len(incorrect_masks_info) + batch_size - 1) // batch_size

                            mask_centers = []
                            for info in incorrect_masks_info:
                                y_indices, x_indices = np.where(info['mask'])
                                if len(y_indices) > 0:
                                    mask_centers.append((np.mean(x_indices), np.mean(y_indices)))
                                else:
                                    mask_centers.append((0, 0))

                            sorted_indices = np.argsort([c[0] for c in mask_centers])
                            batches = [[] for _ in range(num_batches)]
                            for k, idx in enumerate(sorted_indices):
                                batch_idx = k % num_batches
                                batches[batch_idx].append(incorrect_masks_info[idx])

                            for batch_idx, batch_info in enumerate(batches):
                                if not batch_info:
                                    continue

                                batch_masks = [info['mask'] for info in batch_info]
                                batch_global_indices_incorrect = [info['global_idx'] for info in batch_info]

                                som_img, mask_centers_dict = self.mllm_classifier._render_som_image(img, batch_masks, batch_global_indices_incorrect)
                                som_img_bgr = cv2.cvtColor(som_img, cv2.COLOR_RGB2BGR)

                                for k, info in enumerate(batch_info):
                                    cX, cY = mask_centers_dict[k]
                                    text = f"P:{info['pred']} | GT:{info['gt']}"
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.4
                                    thickness = 1
                                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                                    text_y = cY + 15
                                    cv2.rectangle(som_img_bgr, (cX - text_width//2 - 1, text_y - text_height - 1),
                                                  (cX + text_width//2 + 1, text_y + baseline), (0, 0, 255), -1)
                                    cv2.putText(som_img_bgr, text, (cX - text_width//2, text_y), font, font_scale, (255, 255, 255), thickness)

                                cv2.imwrite(os.path.join(step2_dir, f"view_{i:02d}_incorrect_batch_{batch_idx:02d}.png"), som_img_bgr)

                    accuracy = correct_masks / total_masks if total_masks > 0 else 0
                    evaluation_results["summary"] = {
                        "total_evaluated_masks": total_masks,
                        "masks_skipped_predicted_background": skipped_pred_background,
                        "masks_skipped_predicted_unlabeled": skipped_pred_unlabeled,
                        "correct_masks": correct_masks,
                        "accuracy": accuracy,
                        "note": "accuracy 仅在预测为预定义部件的掩码上统计（排除 background 和 unlabeled）；GT 含 <0 索引时点计入 unlabeled 再多数表决；predicted_class 为 Step2.5 后语义（若启用）。",
                    }

                    print(
                        f"MLLM 2D Mask Accuracy (excl. pred=background/unlabeled): {accuracy:.2%} "
                        f"({correct_masks}/{total_masks}), skipped_background={skipped_pred_background}, skipped_unlabeled={skipped_pred_unlabeled}"
                    )

                    with open(os.path.join(step2_dir, "mllm_2d_evaluation.json"), "w", encoding="utf-8") as f:
                        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)

                except Exception as e:
                    print(f"Failed to evaluate MLLM predictions against ground truth: {e}")

            # 写入磁盘前补充 Step 2.5 后的最终语义，便于与评估口径一致
            if mllm_2d_predictions and semantics_across_views:
                for vi, sem_list in enumerate(semantics_across_views):
                    vk = f"view_{vi:02d}"
                    if vk not in mllm_2d_predictions:
                        continue
                    for j, sem in enumerate(sem_list):
                        mk = f"mask_{j}"
                        if mk in mllm_2d_predictions[vk]:
                            mllm_2d_predictions[vk][mk]["predicted_class_after_step25"] = sem

            with open(os.path.join(step2_dir, "mllm_2d_predictions.json"), "w", encoding="utf-8") as f:
                json.dump(mllm_2d_predictions, f, indent=4, ensure_ascii=False)

        # 【阶段 3】基于深度的 2D-to-3D 软映射与融合
        print("Step 3: 3D Probability Field Fusion...")
        projected_data = self.soft_projector.project_with_depth_check(
            soft_masks_across_views,
            render_output.depth_maps,
            render_output.index_maps,
            pc_xyz
        )

        prob_field = self.prob_fusion.fuse_probabilities(
            projected_data,
            len(pc_xyz),
            prompt_classes,
            norm_pcd,
            anatomy_knowledge_json_str=json.dumps(unified_knowledge)
        )

        if vis_dir:
            initial_labels = np.argmax(prob_field.prob_matrix, axis=1)
            self.visualizer.visualize_3d_result(norm_pcd, initial_labels, save_path=os.path.join(step3_dir, "initial_labels.ply"))

        print("Step 4: Final Labeling...")
        final_labels = np.argmax(prob_field.prob_matrix, axis=1)

        classes = prompt_classes + ["unlabeled", "background"]
        class_to_id = {cls: i for i, cls in enumerate(classes)}


        if vis_dir:
            self.visualizer.visualize_3d_result(norm_pcd, final_labels, save_path=os.path.join(step4_dir, "final_result.ply"), class_to_id=class_to_id)

            # 额外输出：每个部件单独高亮的点云
            print("Step 6.1: Saving individual part point clouds...")
            self.visualizer.save_parts_separately(
                norm_pcd, final_labels, class_to_id,
                save_dir=os.path.join(step4_dir, "individual_parts")
            )

        print("Processing finished.")
        return norm_pcd, final_labels, class_to_id

    # ============================================================
    # Step 2.5 可视化辅助方法
    # ============================================================

    @staticmethod
    def _normalize_feedback_display_ids(
        feedback: dict, grouping_meta: Optional[SemanticGroupingResult]
    ) -> None:
        """若模型只返回 mask_id，则尽量映射为 semantic_display_id。"""
        if grouping_meta is None:
            return
        for item in feedback.get("parts_needing_refine", []):
            if item.get("semantic_display_id") is not None:
                continue
            mid = item.get("mask_id")
            if not isinstance(mid, int):
                continue
            if mid < 0 or mid >= len(grouping_meta.original_index_to_display_id):
                continue
            did = grouping_meta.original_index_to_display_id[mid]
            if did >= 0:
                item["semantic_display_id"] = did

    @staticmethod
    def _problem_original_mask_indices(
        parts_needing_refine: List[dict],
        grouping_meta: Optional[SemanticGroupingResult],
        num_masks: int,
    ) -> set:
        out: set = set()
        for item in parts_needing_refine:
            sid = item.get("semantic_display_id")
            if sid is not None and grouping_meta is not None:
                group = grouping_meta.display_id_to_mask_indices.get(int(sid), [])
                if group:
                    for mid in group:
                        out.add(mid)
                    continue
            mid = item.get("mask_id", -1)
            if isinstance(mid, int) and 0 <= mid < num_masks:
                out.add(mid)
        return out

    def _visualize_feedback_input(
        self,
        img: np.ndarray,
        masks: List[np.ndarray],
        semantics: List[str],
        save_path: str,
        grouping_meta: SemanticGroupingResult,
    ):
        """
        可视化 2.5.1: 与 MLLM 一致的语义分组图（相邻同语义合并，同色同编号）。
        """
        vis_bgr = render_semantic_grouped_overlay(img, masks, semantics, grouping_meta)
        cv2.imwrite(save_path, vis_bgr)

    def _visualize_feedback_info(
        self,
        feedback: dict,
        semantics: List[str],
        save_path: str,
        grouping_meta: Optional[SemanticGroupingResult] = None,
    ):
        """
        可视化 2.5.2: 保存 MLLM 反馈信息为 JSON
        """
        legend = grouping_meta.legend_lines if grouping_meta else []
        id_to_sem = grouping_meta.id_to_semantic if grouping_meta else {}
        dmap = (
            {str(k): v for k, v in grouping_meta.display_id_to_mask_indices.items()}
            if grouping_meta
            else {}
        )

        feedback_with_semantics = {
            "overall_quality": feedback.get("overall_quality", "unknown"),
            "parts_needing_refine": [],
            "missing_parts": feedback.get("missing_parts", []),
            "mask_semantics": semantics,
            "semantic_id_legend": legend,
            "id_to_semantic": {str(k): v for k, v in id_to_sem.items()},
            "display_id_to_mask_indices": dmap,
        }

        for item in feedback.get("parts_needing_refine", []):
            mask_id = item.get("mask_id", -1)
            sid = item.get("semantic_display_id")
            if sid is not None and grouping_meta is not None and (
                not isinstance(mask_id, int) or mask_id < 0 or mask_id >= len(semantics)
            ):
                msem = grouping_meta.id_to_semantic.get(int(sid), "unknown")
            else:
                msem = (
                    semantics[mask_id]
                    if isinstance(mask_id, int) and 0 <= mask_id < len(semantics)
                    else "unknown"
                )
            row = {
                "semantic_display_id": sid,
                "mask_id": mask_id,
                "mask_semantic": msem,
                "part_name": item.get("part_name", ""),
                "reason": item.get("reason", ""),
                "suggestion": item.get("suggestion", ""),
            }
            if sid is not None and grouping_meta is not None:
                row["masks_in_group"] = grouping_meta.display_id_to_mask_indices.get(
                    int(sid), []
                )
            feedback_with_semantics["parts_needing_refine"].append(row)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(feedback_with_semantics, f, indent=4, ensure_ascii=False)

    def _visualize_refined_masks(
        self,
        img: np.ndarray,
        masks: List[np.ndarray],
        semantics: List[str],
        save_path: str,
    ):
        """
        可视化 2.5.3: 细化后的掩码（语义分组着色与编号）
        """
        # 补齐 semantics 长度以匹配 masks（新增碎片的语义为 unknown）
        padded_semantics = list(semantics)
        while len(padded_semantics) < len(masks):
            padded_semantics.append("unknown")
        g = build_semantic_grouping(
            masks, padded_semantics, ignore_semantics={"background", "unlabeled"}
        )
        vis_bgr = render_semantic_grouped_overlay(
            img, masks, padded_semantics, g, alpha=0.4, contour_thickness=2
        )
        cv2.imwrite(save_path, vis_bgr)

    def _visualize_round_comparison(
        self,
        img: np.ndarray,
        masks_before: List[np.ndarray],
        masks_after: List[np.ndarray],
        semantics: List[str],
        save_path: str,
    ):
        """
        可视化 Step 2.5 轮次对比图。

        左侧：BEFORE — 修正前的分割掩码语义分组图
        右侧：AFTER — 修正后的分割掩码语义分组图
        中间：DIFF  — 绿色为新增掩码，红色为被删除掩码

        用于直观展示每轮 MLLM 反馈 + SAM 修正的效果。
        """
        h, w = img.shape[:2]

        def _semantic_vis(im, msks, sems):
            padded = list(sems)
            while len(padded) < len(msks):
                padded.append("unknown")
            g = build_semantic_grouping(msks, padded, ignore_semantics={"background", "unlabeled"})
            return render_semantic_grouped_overlay(im, msks, padded, g, alpha=0.40, contour_thickness=2)

        vis_before = _semantic_vis(img, masks_before, semantics)
        vis_after = _semantic_vis(img, masks_after, semantics)

        # --- 构建 DIFF 图 ---
        diff_bgr = img.copy()

        # 被删除的掩码（before 有 after 没有）
        deleted_color = (0, 0, 255)    # BGR 红色
        added_color = (0, 255, 0)      # BGR 绿色

        # 收集 after 中存在的像素
        after_union = np.zeros((h, w), dtype=bool)
        for m in masks_after:
            after_union |= m.astype(bool)

        # 收集 before 中存在的像素
        before_union = np.zeros((h, w), dtype=bool)
        for m in masks_before:
            before_union |= m.astype(bool)

        # 纯新增（在 after 中有但 before 中没有）
        added_mask = after_union & ~before_union
        # 纯删除（在 before 中有但 after 中没有）
        deleted_mask = before_union & ~after_union

        overlay = diff_bgr.copy()
        overlay[added_mask] = np.clip(overlay[added_mask].astype(int) + np.array([0, 80, 0]), 0, 255).astype(np.uint8)
        overlay[deleted_mask] = np.clip(overlay[deleted_mask].astype(int) + np.array([0, 0, 80]), 0, 255).astype(np.uint8)
        cv2.addWeighted(overlay, 0.45, diff_bgr, 0.55, 0, diff_bgr)

        if added_mask.any():
            contours, _ = cv2.findContours(added_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(diff_bgr, contours, -1, added_color, 2)
        if deleted_mask.any():
            contours, _ = cv2.findContours(deleted_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.RETR_EXTERNAL)
            cv2.drawContours(diff_bgr, contours, -1, deleted_color, 2)

        # 图例
        for txt, clr, y in [("Green=Added", added_color, 20), ("Red=Deleted", deleted_color, 42)]:
            cv2.rectangle(diff_bgr, (4, y - 14), (130, y + 4), (0, 0, 0), -1)
            cv2.putText(diff_bgr, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1)

        # --- 拼接三列 ---
        panel_before = self._add_title_bar(vis_before, "BEFORE", (255, 255, 255))
        panel_after = self._add_title_bar(vis_after, "AFTER", (255, 255, 255))
        panel_diff = self._add_title_bar(diff_bgr, "DIFF", (200, 200, 200))

        combined = np.hstack([panel_before, panel_diff, panel_after])
        cv2.imwrite(save_path, combined)

    def _add_title_bar(self, img_bgr: np.ndarray, text: str, color: Tuple[int, int, int]) -> np.ndarray:
        """在图像顶部添加一行标题栏"""
        bar_h = 28
        bar = np.full((bar_h, img_bgr.shape[1], 3), color, dtype=np.uint8)
        cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        return np.vstack([bar, img_bgr])

    def _visualize_mllm_feedback(
        self,
        feedback: dict,
        semantics: List[str],
        save_path: str,
    ):
        """
        可视化 MLLM 反馈信息：将评估结果渲染为图像。

        每张卡片代表一个语义部件的评估结果，包含：
        - 部件名（优先 MLLM 返回的 semantic，与 prompt_classes 一致）
        - 质量评级（绿色=acceptable / 橙色=needs_refine）
        - 删除碎片 / 补充点 / selected_views 等摘要
        """
        evaluations = feedback.get("evaluations", [])
        if not evaluations:
            return

        card_w, card_h = 420, 100
        gap = 12
        cols = 3
        rows = (len(evaluations) + cols - 1) // cols
        panel_w = cols * (card_w + gap) + gap
        panel_h = rows * (card_h + gap) + gap + 36

        panel = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 30
        cv2.putText(
            panel, "MLLM Evaluation Feedback",
            (gap, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1
        )

        for idx, ev in enumerate(evaluations):
            row = idx // cols
            col = idx % cols
            x = gap + col * (card_w + gap)
            y = gap + 36 + row * (card_h + gap)

            did = ev.get("display_id", -1)
            quality = ev.get("overall_quality") or ev.get("quality", "unknown")
            reasoning = str(ev.get("overall_reasoning") or ev.get("reasoning", ""))[:80]
            deletes = ev.get("delete_fragments", [])
            if not deletes:
                for pv in ev.get("per_view_errors", []) or []:
                    deletes = deletes + list(pv.get("delete_fragments", []) or [])
            supplements = ev.get("supplement_points", [])
            refine_action = ev.get("refine_action", "")

            # 背景色：绿=acceptable，橙=needs_refine
            if quality == "acceptable":
                bg = (18, 50, 18)
                border = (60, 180, 60)
            elif quality == "needs_refine":
                bg = (18, 40, 50)
                border = (200, 130, 40)
            else:
                bg = (25, 25, 25)
                border = (120, 120, 120)

            card = np.full((card_h, card_w, 3), bg, dtype=np.uint8)
            cv2.rectangle(card, (0, 0), (card_w - 1, card_h - 1), border, 2)

            # 部件名：优先 JSON 中的 semantic（与 prompt_classes 一致）
            sem = str(ev.get("semantic") or "").strip()
            if not sem and isinstance(did, int) and 0 <= did < len(semantics):
                sem = str(semantics[did])
            if not sem:
                sem = f"id_{did}"

            cv2.putText(card, f"#{idx+1} did={did} {sem}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)

            # 质量
            q_color = (100, 255, 100) if quality == "acceptable" else (200, 200, 80)
            cv2.putText(card, f"quality: {quality}", (8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, q_color, 1)

            # 操作
            actions = []
            if deletes:
                actions.append(f"del_frag {deletes}")
            if supplements:
                n = len(supplements) if isinstance(supplements, list) else 0
                actions.append(f"supp +{n}pts")
            if refine_action and refine_action != "none":
                actions.append(str(refine_action))
            if actions:
                line = " | ".join(actions)[:72]
                cv2.putText(card, line, (8, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1)
            else:
                cv2.putText(card, "No fragment/point actions", (8, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130, 130, 130), 1)

            # reasoning 摘要
            if reasoning:
                cv2.putText(card, reasoning[:68], (8, 76),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.26, (150, 150, 150), 1)

            panel[y:y + card_h, x:x + card_w] = card

        cv2.imwrite(save_path, panel)

    def _visualize_before_after_comparison(
        self,
        img: np.ndarray,
        masks_before: List[np.ndarray],
        masks_after: List[np.ndarray],
        semantics: List[str],
        parts_needing_refine: List[dict],
        save_path: str,
        grouping_meta: Optional[SemanticGroupingResult] = None,
    ):
        """
        可视化 2.5.4: 细化前后对比图（左右拼接）
        """
        vis_before = self._create_comparison_panel(
            img, masks_before, semantics, "BEFORE", grouping_meta
        )
        vis_after = self._create_comparison_panel(img, masks_after, semantics, "AFTER", None)
        vis_combined = np.hstack([vis_before, vis_after])
        cv2.imwrite(save_path, vis_combined)

    def _create_comparison_panel(
        self,
        img: np.ndarray,
        masks: List[np.ndarray],
        semantics: List[str],
        title: str,
        grouping_meta: Optional[SemanticGroupingResult] = None,
    ) -> np.ndarray:
        """创建单个对比面板（语义分组着色；grouping_meta 为 None 时在内部重建）"""
        g = grouping_meta or build_semantic_grouping(
            masks, semantics, ignore_semantics={"background", "unlabeled"}
        )
        vis_img_bgr = render_semantic_grouped_overlay(
            img, masks, semantics, g, alpha=0.35, contour_thickness=2
        )
        cv2.putText(
            vis_img_bgr,
            title,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return vis_img_bgr

    def _visualize_problem_regions(
        self,
        img: np.ndarray,
        masks: List[np.ndarray],
        parts_needing_refine: List[dict],
        semantics: List[str],
        save_path: str,
        grouping_meta: Optional[SemanticGroupingResult] = None,
    ):
        """
        可视化 2.5.5: 语义分组底图 + 需细化原始掩码的红色粗轮廓
        """
        g = grouping_meta or build_semantic_grouping(
            masks, semantics, ignore_semantics={"background", "unlabeled"}
        )
        vis_img_bgr = render_semantic_grouped_overlay(
            img, masks, semantics, g, alpha=0.32, contour_thickness=2
        )

        problem_idx = self._problem_original_mask_indices(
            parts_needing_refine, grouping_meta, len(masks)
        )
        for idx in problem_idx:
            if idx < 0 or idx >= len(masks):
                continue
            mask = masks[idx]
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis_img_bgr, contours, -1, (0, 0, 255), 3)

        cv2.rectangle(vis_img_bgr, (5, 35), (200, 60), (0, 0, 255), -1)
        cv2.putText(
            vis_img_bgr,
            "Red outline = refine target",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        cv2.imwrite(save_path, vis_img_bgr)

    # ============================================================
    # Step 2.5 辅助方法
    # ============================================================

    @staticmethod
    def _derive_semantics_from_masks(
        current_masks: List[np.ndarray],
        original_soft_masks: List[SoftMask2D],
    ) -> List[str]:
        """
        基于原始 soft_masks 的概率，为当前 masks 推断语义标签。
        新增的掩码（超出 original_soft_masks 范围的）标记为 "unknown"。
        """
        derived = []
        n_original = len(original_soft_masks)
        for i, mask in enumerate(current_masks):
            if i < n_original:
                sm = original_soft_masks[i]
                if sm.probabilities:
                    derived.append(max(sm.probabilities, key=sm.probabilities.get))
                else:
                    derived.append("unknown")
            else:
                derived.append("unknown")
        return derived

    # ──────────────────────────────────────────────────────────────
    # Step 2.5 辅助方法：3D 点采样 & 投影
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _farthest_point_sample_2d(
        points_2d: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """
        Farthest Point Sampling (FPS) 在 2D 像素空间均匀采样点。

        Args:
            points_2d: (N, 2) 像素坐标 [x, y]
            n_samples: 需要采样的点数
        Returns:
            (n_samples, 2) 采样后的 2D 坐标
        """
        if len(points_2d) <= n_samples:
            return points_2d

        chosen = [0]
        distances = np.full(len(points_2d), np.inf)
        for _ in range(1, n_samples):
            last = np.array(points_2d[chosen[-1]], dtype=np.float64)
            dists = np.linalg.norm(points_2d.astype(np.float64) - last, axis=1)
            np.minimum(distances, dists, out=distances)
            farthest = int(np.argmax(distances))
            chosen.append(farthest)

        return points_2d[chosen]

    @staticmethod
    def _mask_boundary_distance(
        fg_mask: np.ndarray, all_pts_2d: np.ndarray
    ) -> np.ndarray:
        """
        计算 all_pts_2d 中每个点到前景掩码边界的距离（像素数）。
        距离定义为到最近零值像素的曼哈顿/欧氏距离。
        落在前景内部的点距离 > 0。
        """
        from scipy.ndimage import distance_transform_edt

        dist = distance_transform_edt(fg_mask.astype(bool))
        h, w = fg_mask.shape
        pts_int = np.clip(all_pts_2d.astype(int), 0, [[w - 1, h - 1]])
        return dist[pts_int[:, 1], pts_int[:, 0]]

    def _sample_near_boundary_points(
        self,
        fg_mask: np.ndarray,
        index_map: np.ndarray,
        n_bg: int,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        在前景掩码的紧邻边界外侧均匀采样背景点。

        步骤：
        1. 取前景掩码的边界（dilate 后异或原掩码）
        2. 在边界像素上做 FPS 均匀采样
        3. 若边界像素不足 n_bg，再从紧邻内部补采

        Returns:
            (bg_2d, valid): bg_2d 为 (n, 2) [x, y] 像素坐标，valid 表示是否采样成功
        """
        if not fg_mask.any():
            return None, False

        from scipy.ndimage import binary_dilation

        struct = np.ones((5, 5), dtype=bool)
        dilated = binary_dilation(fg_mask, structure=struct)
        boundary = dilated ^ fg_mask  # 布尔异或，等价于 XOR
        boundary_pixels = np.array(np.where(boundary)).T[:, ::-1]  # (N, 2) = [x, y]

        if len(boundary_pixels) >= n_bg:
            bg_2d = self._farthest_point_sample_2d(boundary_pixels, n_bg)
            return bg_2d, True

        # 边界不够，先用全部边界，再用紧邻内部的点凑够
        bg_2d_list = [boundary_pixels]
        n_remaining = n_bg - len(boundary_pixels)

        inner_shell = dilated & ~binary_dilation(fg_mask, structure=np.ones((3, 3), dtype=bool))
        inner_pixels = np.array(np.where(inner_shell)).T[:, ::-1]
        if len(inner_pixels) > 0:
            extra = self._farthest_point_sample_2d(inner_pixels, min(n_remaining, len(inner_pixels)))
            bg_2d_list.append(extra)

        combined = np.vstack(bg_2d_list) if len(bg_2d_list) > 1 else bg_2d_list[0]
        return self._farthest_point_sample_2d(combined, n_bg), True

    def _masks_to_3d_points(
        self,
        masks_per_view: List[np.ndarray],
        index_maps: List[np.ndarray],
        depth_maps: List[np.ndarray],
        pc_xyz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将各视角的掩码合并后，映射回 3D 点云，提取 3D 点坐标。

        Args:
            masks_per_view: 每视角的布尔掩码列表（List of HxW bool）
            index_maps: List[HxW] — 渲染时每个像素对应的点云索引
            depth_maps: List[HxW] — 深度图
            pc_xyz: (N, 3) 点云坐标

        Returns:
            (points_3d, areas): (K, 4) 每行 [x, y, z, pc_index], (K,) 每点面积权重
        """
        if not masks_per_view:
            return np.empty((0, 4), dtype=np.float32), np.array([], dtype=np.float32)

        seen = {}  # pc_index -> list_index
        coords = []   # [pc_index] -> [x, y, z]
        areas = []    # [pc_index] -> 面积计数

        for vid, (mask, index_map) in enumerate(zip(masks_per_view, index_maps)):
            if index_map is None:
                continue
            if index_map.ndim == 3:
                index_map = index_map[:, :, 0]
            ys, xs = np.where(mask.astype(bool))
            for y, x in zip(ys, xs):
                pidx = int(index_map[y, x])
                if pidx < 0 or pidx >= len(pc_xyz):
                    continue
                if pidx in seen:
                    areas[seen[pidx]] += 1
                else:
                    idx = len(coords)
                    seen[pidx] = idx
                    pt = pc_xyz[pidx]
                    coords.append([pt[0], pt[1], pt[2], float(pidx)])
                    areas.append(1)

        if not coords:
            return np.empty((0, 4), dtype=np.float32), np.array([], dtype=np.float32)
        return np.array(coords, dtype=np.float32), np.array(areas, dtype=np.float32)

    def _sample_background_points(
        self,
        pc_xyz: np.ndarray,
        masks_across_views: List[List[np.ndarray]],
        semantics_across_views: List[List[str]],
        target_semantic: str,
        index_maps: List[np.ndarray],
        n_bg: int,
    ) -> np.ndarray:
        """
        从非目标语义的区域采样背景点（3D 坐标）。

        Args:
            pc_xyz: (N, 3) 点云坐标
            masks_across_views: 各视角掩码列表
            semantics_across_views: 各视角的语义标签列表
            target_semantic: 目标语义名（要排除的语义）
            index_maps: List[HxW]
            n_bg: 需要采样的背景点数量

        Returns:
            (n_bg, 3) 背景点 3D 坐标
        """
        h, w = (0, 0)
        for v in masks_across_views:
            if v:
                h, w = v[0].shape[:2]
                break

        all_valid_pts: set[int] = set()
        target_pt_indices: set[int] = set()

        for vid, (vmasks, vsems) in enumerate(zip(masks_across_views, semantics_across_views)):
            if vid >= len(index_maps) or index_maps[vid] is None:
                continue
            im = index_maps[vid]
            if im.ndim == 3:
                im = im[:, :, 0]

            valid_mask = im >= 0
            pts = im[valid_mask].flatten().tolist()
            all_valid_pts.update(p for p in pts if 0 <= p < len(pc_xyz))

            tg_mask = np.zeros((h, w), dtype=bool)
            for m, s in zip(vmasks, vsems):
                if s == target_semantic and m is not None:
                    tg_mask |= m.astype(bool)

            ys, xs = np.where(tg_mask)
            for y, x in zip(ys, xs):
                pidx = int(im[y, x])
                if 0 <= pidx < len(pc_xyz):
                    target_pt_indices.add(pidx)

        bg_indices = list(all_valid_pts - target_pt_indices)
        if len(bg_indices) == 0:
            return np.empty((0, 3), dtype=np.float32)
        n = min(n_bg, len(bg_indices))
        chosen = np.random.choice(bg_indices, size=n, replace=False)
        return pc_xyz[chosen].astype(np.float32)

    @staticmethod
    def _pack_xyz_with_pc_index(
        fg_2d_filtered: np.ndarray,
        fg_pixel_to_pc: Dict[Tuple[int, int], int],
        pc_xyz: np.ndarray,
    ) -> np.ndarray:
        """
        将前景点对应的 3D 坐标与点云索引打包为 (N, 4)，
        供 _project_3d_to_2d_simple 用 index_map 反查各视角像素。
        """
        rows: List[List[float]] = []
        for p in fg_2d_filtered:
            k = (int(p[0]), int(p[1]))
            if k not in fg_pixel_to_pc:
                continue
            pidx = int(fg_pixel_to_pc[k])
            if not (0 <= pidx < len(pc_xyz)):
                continue
            x, y, z = pc_xyz[pidx]
            rows.append([float(x), float(y), float(z), float(pidx)])
        if not rows:
            return np.empty((0, 4), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    def _project_3d_to_2d_simple(
        self,
        points_3d: np.ndarray,
        target_view_idx: int,
        index_maps: List[np.ndarray],
        pc_xyz: np.ndarray,
    ) -> np.ndarray:
        """
        将 3D 点（通过点云索引）投影到 2D 视角图。

        Args:
            points_3d: (N, 4) 每行 [x, y, z, point_cloud_index]（推荐；勿传纯 (N,3)，
                否则会用行号当点索引导致跨视角投影错误）
            target_view_idx: 目标视角索引
            index_maps: List[HxW] 每个像素对应的点云索引
            pc_xyz: (M, 3) 完整点云

        Returns:
            (N, 2) 每行 [x, y] 2D 像素坐标（浮点）
        """
        if len(points_3d) == 0 or target_view_idx >= len(index_maps):
            return np.empty((0, 2), dtype=np.float32)

        index_map = index_maps[target_view_idx]
        if index_map.ndim == 3:
            index_map = index_map[:, :, 0]

        H, W = index_map.shape[:2]

        # 建立 point_idx → (x, y) 的反向索引
        pt_to_xy: Dict[int, List[Tuple[int, int]]] = {}
        for y in range(H):
            for x in range(W):
                pidx = int(index_map[y, x])
                if pidx >= 0:
                    if pidx not in pt_to_xy:
                        pt_to_xy[pidx] = []
                    pt_to_xy[pidx].append((x, y))

        result = []
        for i in range(len(points_3d)):
            if points_3d.shape[1] > 3:
                pidx = int(points_3d[i, 3])
            else:
                # 历史误用：仅 xyz 时无法用 index_map 反查，行号 ≠ 点云索引
                pidx = i
            if pidx in pt_to_xy and pt_to_xy[pidx]:
                x, y = pt_to_xy[pidx][0]
                result.append([float(x), float(y)])

        return np.array(result, dtype=np.float32) if result else np.empty((0, 2), dtype=np.float32)

    def _points_to_bbox(
        self,
        points_2d: np.ndarray,
        H: int,
        W: int,
        expand: float = 1.2,
    ) -> Optional[Tuple[int, int, int, int]]:
        """从 2D 点集生成 SAM bbox，x1>y1"""
        if len(points_2d) == 0:
            return None
        xs = points_2d[:, 0]
        ys = points_2d[:, 1]
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
        new_w, new_h = int(w * expand), int(h * expand)
        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(W - 1, x1 + new_w)
        y2 = min(H - 1, y1 + new_h)
        return (x1, y1, x2, y2)

    def _merge_same_semantic_masks(
        self,
        masks_across_views: List[List[np.ndarray]],
        semantics_across_views: List[List[str]],
        prompt_classes: List[str],
    ) -> Tuple[List[List[np.ndarray]], List[List[str]]]:
        """
        每个视角中，将同语义的多个碎片掩码合并为一个掩码。
        保留不同的语义类别（包括 unlabeled），每个语义类别对应一个合并后的掩码。

        Returns:
            merged_masks: 每视角合并后的掩码列表
            merged_semantics: 与上式逐元素对应的语义字符串（与 prompt_classes 顺序无关的错位来源已消除）
        """
        merged_masks_out: List[List[np.ndarray]] = []
        merged_sems_out: List[List[str]] = []
        prompt_set = set(prompt_classes)
        for vid, (masks, semantics) in enumerate(zip(masks_across_views, semantics_across_views)):
            if not masks or len(masks) != len(semantics):
                merged_masks_out.append([np.asarray(m, dtype=np.uint8) for m in masks] if masks else [])
                merged_sems_out.append(list(semantics) if semantics else [])
                continue

            sem_to_mask: Dict[str, np.ndarray] = {}
            for m, s in zip(masks, semantics):
                s_key = str(s).strip() if s is not None else "unknown"
                if s_key not in sem_to_mask:
                    sem_to_mask[s_key] = np.zeros_like(m, dtype=bool)
                sem_to_mask[s_key] = sem_to_mask[s_key] | m.astype(bool)

            keys_ordered: List[str] = []
            for c in prompt_classes:
                if c in sem_to_mask:
                    keys_ordered.append(c)
            extras = sorted(
                (k for k in sem_to_mask.keys() if k not in prompt_set),
                key=lambda x: (str(x).lower(), str(x)),
            )
            keys_ordered.extend(extras)

            view_merged: List[np.ndarray] = []
            view_sems: List[str] = []
            for sem in keys_ordered:
                view_merged.append(sem_to_mask[sem].astype(np.uint8))
                view_sems.append(sem)
            merged_masks_out.append(view_merged)
            merged_sems_out.append(view_sems)
        return merged_masks_out, merged_sems_out

    def _visualize_new_masks_overlay(
        self,
        image: np.ndarray,
        new_masks: List[np.ndarray],
        save_path: str,
        fg_prompt_xy: Optional[np.ndarray] = None,
    ):
        """
        可视化新掩码叠加在原图上。
        image 为 RGB；保存前转为 BGR，避免 OpenCV imwrite 色道错误。
        小面积掩码用高对比填充 + 粗轮廓，并可选绘制 SAM 正提示点。
        """
        h, w = image.shape[:2]
        vis = image.astype(np.float32).copy()
        contour_thick = max(3, int(round(600 / max(h, w, 1))))
        colors_rgb = [
            (255, 64, 64),
            (64, 255, 128),
            (64, 128, 255),
            (255, 200, 64),
        ]
        for mi, mask in enumerate(new_masks):
            m_bool = mask.astype(bool)
            if not m_bool.any():
                continue
            c = np.array(colors_rgb[mi % len(colors_rgb)], dtype=np.float32)
            vis[m_bool] = vis[m_bool] * 0.22 + c * 0.78
        vis_u8 = np.clip(vis, 0, 255).astype(np.uint8)
        for mi, mask in enumerate(new_masks):
            u8 = (mask.astype(bool).astype(np.uint8) * 255)
            contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # drawContours 在 RGB 图上用 RGB 颜色
            cc = colors_rgb[mi % len(colors_rgb)]
            cv2.drawContours(vis_u8, contours, -1, cc, contour_thick)
        if fg_prompt_xy is not None and len(fg_prompt_xy) > 0:
            for px, py in fg_prompt_xy:
                ix, iy = int(round(float(px))), int(round(float(py)))
                if 0 <= ix < w and 0 <= iy < h:
                    cv2.circle(vis_u8, (ix, iy), max(4, contour_thick + 2), (0, 220, 0), 2)
                    cv2.circle(vis_u8, (ix, iy), 2, (0, 255, 0), -1)
        cv2.imwrite(save_path, cv2.cvtColor(vis_u8, cv2.COLOR_RGB2BGR))

    def _apply_single_semantic_feedback(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        semantics: List[str],
        feedback: dict,
        grouping,
        sam_predictor,
        max_rounds: int = 2,
    ) -> List[np.ndarray]:
        """
        根据新版单语义批量反馈执行修正。

        feedback["evaluations"] 中每个条目包含：
        - display_id: 部件 ID
        - delete_fragments: 需要删除的 fragment_id 列表
        - supplement_points: 补充用的正提示点列表 [x, y]

        对于 delete：直接置零对应碎片掩码
        对于 supplement：用 SAM 在 supplement_points 重分割，结果追加为 "unknown" 语义
        """
        evaluations = feedback.get("evaluations", [])
        if not evaluations:
            return masks

        current_masks = [m.copy() for m in masks]
        sam = self.refiner
        predictor = sam._get_predictor(sam_predictor)
        if predictor is None:
            print("[Pipeline] No SAM predictor, skipping feedback application.")
            return current_masks

        for ev in evaluations:
            did = int(ev.get("display_id", -1))
            sem_fb = str(ev.get("semantic", "") or "").strip()
            delete_fids = ev.get("delete_fragments", [])
            supplement_pts = ev.get("supplement_points", [])

            # === DELETE: 置零指定碎片 ===
            if delete_fids:
                local_gid = (
                    semantic_string_to_display_id(grouping, sem_fb)
                    if sem_fb
                    else None
                )
                if local_gid is None and did in grouping.display_id_to_fragments:
                    local_gid = did
                if local_gid is None:
                    continue
                if sem_fb:
                    ordered = fragments_for_semantic_ordered(grouping, sem_fb)
                    frag_lookup = {i: fr for i, fr in enumerate(ordered)}
                else:
                    frag_lookup = {
                        fr["fragment_id"]: fr
                        for fr in grouping.display_id_to_fragments.get(local_gid, [])
                    }
                for raw_fid in delete_fids:
                    fid = _normalize_fragment_id(raw_fid)
                    if fid is None or fid not in frag_lookup:
                        continue
                    frag = frag_lookup[fid]
                    for mid in frag.get("mask_indices", []):
                        if 0 <= mid < len(current_masks):
                            current_masks[mid] = np.zeros_like(current_masks[mid])

            # === SUPPLEMENT: SAM 重分割补充 ===
            if supplement_pts and isinstance(supplement_pts, list) and len(supplement_pts) > 0:
                valid_pts = [p for p in supplement_pts if isinstance(p, (list, tuple)) and len(p) == 2]
                if valid_pts:
                    new_mask = sam.local_resegment_multi_point(
                        image, valid_pts, [], predictor
                    )
                    if new_mask is not None and new_mask.any():
                        current_masks.append(new_mask)
                        print(f"  [Feedback] Added supplement mask for display_id={did}, total masks now={len(current_masks)}")

        return current_masks


if __name__ == "__main__":
    pipeline = Arbitr3DPipeline("config/config.yaml")
    print("Pipeline initialized successfully.")
