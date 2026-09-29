"""
PartObjaverse-Tiny 单实例测试脚本

用法:
    # 指定类别和实例 ID
    python test_objaverse.py --category Food --object-id 00aee5c2fef743d69421bb642d446a5b

    # 指定 GPU
    python test_objaverse.py --category Animals --object-id 29e480b958df44a2ac3a4ed12b7ee1a0 --gpu 2

    # 指定采样点数（默认 30000）
    python test_objaverse.py --category Food --object-id 00aee5c2fef743d69421bb642d446a5b --num-points 50000

    # 不指定 object-id 时，列出该类别下所有可用实例
    python test_objaverse.py --category Food --list
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
import json
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# GPU 选择
# ============================================================
def _get_physical_gpu_map() -> dict:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {}
        mapping = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            idx = int(parts[0])
            free_gb = float(parts[1]) / 1024.0
            total_gb = float(parts[2]) / 1024.0
            mapping[idx] = (free_gb, total_gb)
        return mapping
    except Exception:
        return {}


def _auto_best_physical(physical_map: dict, threshold: float = 4.0) -> int:
    if not physical_map:
        raise RuntimeError("No GPU available.")
    candidates = {idx: free for idx, (free, _) in physical_map.items() if free >= threshold}
    if not candidates:
        raise RuntimeError(f"No GPU with >= {threshold:.1f} GB free memory.")
    return max(candidates, key=lambda i: candidates[i])


def setup_gpu(gpu_id: int = -1) -> int:
    import yaml
    threshold = 4.0
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        threshold = float(cfg.get("gpu", {}).get("min_free_gb", 4.0))
    except Exception:
        pass

    physical_map = _get_physical_gpu_map()
    print(f"\n    物理 GPU 状态:")
    for idx in sorted(physical_map.keys()):
        free_gb, total_gb = physical_map[idx]
        bar_len = int((total_gb - free_gb) / total_gb * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        used_pct = (total_gb - free_gb) / total_gb * 100
        print(f"    GPU {idx}: [{bar}] {used_pct:.0f}% 已用，剩余 {free_gb:.1f} / {total_gb:.1f} GB")
    print()

    raw_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw_cvd and raw_cvd.lower() != "auto":
        print(f"[GPU] CUDA_VISIBLE_DEVICES={raw_cvd} → 逻辑设备 0")
        return 0

    if gpu_id >= 0:
        print(f"[GPU] 使用命令行指定 GPU {gpu_id}")
        return gpu_id

    selected = _auto_best_physical(physical_map, threshold)
    print(f"[GPU] 自动选择 GPU {selected}（剩余 ~{physical_map[selected][0]:.1f} GB）\n")
    return selected


# ============================================================
# Meta 加载
# ============================================================
def load_meta(meta_path: str) -> dict:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta JSON not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_instances(meta: dict, category: str):
    if category not in meta:
        print(f"类别 '{category}' 不存在。可用类别: {list(meta.keys())}")
        return
    instances = meta[category]
    print(f"\n类别 '{category}' 下共 {len(instances)} 个实例:\n")
    for i, (obj_id, parts) in enumerate(sorted(instances.items())):
        print(f"  [{i+1:3d}] {obj_id}  ({len(parts)} 部件: {', '.join(parts[:5])}{'...' if len(parts) > 5 else ''})")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="PartObjaverse-Tiny 单实例测试")
    parser.add_argument("--category", "-c", type=str, required=True,
                        help="物体类别 (如 Food, Animals, Human-Shape, Daily-Used, "
                             "Buildings&&Outdoor, Transportations, Plants, Electronics)")
    parser.add_argument("--object-id", "-o", type=str, default=None,
                        help="实例 ID (如 00aee5c2fef743d69421bb642d446a5b)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="仅列出该类别下所有可用实例，不运行测试")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU 编号，-1=自动")
    parser.add_argument("--num-points", type=int, default=None,
                        help="网格采样点数（覆盖 config，默认 30000）")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--exp-suffix", default=None, help="实验后缀（影响输出目录名）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载 meta（优先项目根目录，兜底数据集目录）
    import yaml as _yaml
    with open(args.config, "r", encoding="utf-8") as _f:
        _cfg = _yaml.safe_load(_f) or {}
    _pot = _cfg.get("partobjaverse_tiny", {})
    _meta_rel = _pot.get("meta_json", "PartObjaverse-Tiny_semantic.json")
    meta_path = os.path.join(PROJECT_ROOT, _meta_rel)
    if not os.path.exists(meta_path):
        meta_path = os.path.join(_pot.get("base_path", ""), _meta_rel)
    meta = load_meta(meta_path)

    # 仅列出实例
    if args.list:
        list_instances(meta, args.category)
        return

    if args.object_id is None:
        print("请指定 --object-id，或使用 --list 查看可用实例。")
        list_instances(meta, args.category)
        return

    # 验证类别和实例
    if args.category not in meta:
        print(f"类别 '{args.category}' 不存在。可用类别: {list(meta.keys())}")
        return

    instances = meta[args.category]
    if args.object_id not in instances:
        print(f"实例 '{args.object_id}' 不在类别 '{args.category}' 中。")
        print(f"使用 --list 查看可用实例。")
        return

    part_names = instances[args.object_id]
    prompt_classes_str = ", ".join(part_names)

    print(f"{'='*60}")
    print(f"  类别:   {args.category}")
    print(f"  实例:   {args.object_id}")
    print(f"  部件:   {prompt_classes_str}")
    print(f"{'='*60}\n")

    # GPU 设置
    logical_gpu = setup_gpu(args.gpu)
    import torch
    torch.cuda.set_device(logical_gpu)

    import yaml
    from pipeline import Arbitr3DPipeline
    from evaluation.metrics import IoUEvaluator

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pot_cfg = config.get("partobjaverse_tiny", {})
    BASE_PATH = pot_cfg.get("base_path",
        "datasets/PartObjaverse-Tiny/PartObjaverse-Tiny")
    MESH_SUBDIR = pot_cfg.get("mesh_subdir", "PartObjaverse-Tiny_mesh")
    GT_SUBDIR = pot_cfg.get("gt_subdir", "PartObjaverse-Tiny_semantic_gt")
    NUM_POINTS = args.num_points if args.num_points is not None else int(pot_cfg.get("num_points", 30000))

    # 验证文件存在
    mesh_rel_path = os.path.join(MESH_SUBDIR, f"{args.object_id}.glb")
    mesh_full_path = os.path.join(BASE_PATH, mesh_rel_path)
    gt_full_path = os.path.join(BASE_PATH, GT_SUBDIR, f"{args.object_id}.npy")

    if not os.path.exists(mesh_full_path):
        print(f"网格文件不存在: {mesh_full_path}")
        return
    print(f"网格文件: {mesh_full_path}")
    print(f"GT 标签:  {gt_full_path} ({'存在' if os.path.exists(gt_full_path) else '不存在'})")
    print(f"采样点数: {NUM_POINTS} (0=使用原始顶点)\n")

    # 创建临时配置（base_path 指向 PartObjaverse-Tiny）
    config["data"]["base_path"] = BASE_PATH
    config["data"]["dataset_name"] = "PartObjaverse-Tiny"
    tmp_config_path = os.path.join(PROJECT_ROOT, "config", "_tmp_objaverse_test_config.yaml")
    with open(tmp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    # 初始化 Pipeline
    pipeline = Arbitr3DPipeline(tmp_config_path, gpu_index=logical_gpu)

    # 构建输出目录
    exp_suffix = args.exp_suffix or f"{args.category}"
    vis_dir = f"visual_res/{args.object_id}_vis_{exp_suffix}"

    # 运行 Pipeline
    print(f"Starting Arbitr3D Pipeline for {args.category}/{args.object_id}...")
    pcd, final_labels, class_mapping = pipeline.run(
        point_cloud_filename=mesh_rel_path,
        prompt_classes_str=prompt_classes_str,
        vis_dir=vis_dir,
        object_category=args.category,
        num_points=NUM_POINTS,
    )

    print(f"类别映射: {class_mapping}")

    # 保存预测点云
    output_ply_path = os.path.join(vis_dir, f"pred_{args.object_id}.ply")
    pipeline.visualizer.visualize_3d_result(
        pcd, final_labels, save_path=output_ply_path, class_to_id=class_mapping
    )
    print(f"预测结果已保存至: {output_ply_path}")

    # 评估
    if os.path.exists(gt_full_path):
        evaluator = IoUEvaluator()
        gt_labels = IoUEvaluator.load_gt_labels(gt_full_path)

        if gt_labels is not None:
            n_pred = len(final_labels)
            n_gt = len(gt_labels)

            if n_pred != n_gt and NUM_POINTS > 0:
                print(f"\n预测点数({n_pred}) != GT点数({n_gt})，通过最近邻重映射 GT...")
                gt_labels = _remap_gt(gt_full_path, mesh_full_path, pcd, gt_labels)

            if gt_labels is not None and len(gt_labels) == len(final_labels):
                # 保存 GT 可视化
                gt_ply_path = os.path.join(vis_dir, f"gt_{args.object_id}.ply")
                pipeline.visualizer.visualize_3d_result(
                    pcd, gt_labels, save_path=gt_ply_path, class_to_id=class_mapping
                )
                print(f"GT 可视化已保存至: {gt_ply_path}")

                # 同时在 pipeline 的 step4_final 目录下保存一份 gt_result.ply，
                # 与 final_result.ply 并列，方便直接对照预测与真值。
                step4_gt_path = os.path.join(vis_dir, "step4_final", "gt_result.ply")
                pipeline.visualizer.visualize_3d_result(
                    pcd, gt_labels, save_path=step4_gt_path, class_to_id=class_mapping
                )
                print(f"GT 副本已保存至: {step4_gt_path}")

                ious, miou = evaluator.evaluate_with_gt_array(
                    pred_labels=final_labels,
                    gt_labels=gt_labels,
                    class_mapping=class_mapping,
                )
            else:
                print(f"GT 标签点数({len(gt_labels) if gt_labels is not None else 'None'}) "
                      f"与预测点数({n_pred})不一致，无法评估。")
        else:
            print("GT 标签加载失败。")
    else:
        print(f"\nGT 标签文件不存在，跳过定量评估。")
        print(f"预期路径: {gt_full_path}")

    # 清理临时配置
    if os.path.exists(tmp_config_path):
        os.remove(tmp_config_path)

    print(f"\n可视化输出目录: {vis_dir}")
    print("Done.")


def _remap_gt(gt_path, mesh_path, pred_pcd, gt_labels):
    """
    用 trimesh 从网格面上密集采样并直接通过面索引赋 GT 标签，
    再与 pipeline 归一化点云做 KDTree 最近邻匹配。
    避免 per-face→per-vertex 投票造成的边界精度丢失。
    加载方式与 PartObjaverse-Tiny 官方一致。
    """
    import open3d as o3d
    import trimesh
    try:
        # 官方标准加载方式（不加 force/process 参数）
        tm = trimesh.load(mesh_path)
        if isinstance(tm, trimesh.Scene):
            tm = tm.dump(concatenate=True)

        num_verts = len(tm.vertices)
        num_faces = len(tm.faces)
        n_gt = len(gt_labels)

        unique_gt, gt_counts = np.unique(gt_labels, return_counts=True)
        print(f"  [trimesh] 网格: {num_verts} 顶点, {num_faces} 面")
        print(f"  原始 GT 标签分布: {dict(zip(unique_gt.tolist(), gt_counts.tolist()))}")

        pred_pts = np.asarray(pred_pcd.points)
        n_pred = len(pred_pts)

        if n_gt == num_faces:
            # per-face GT：从网格面上密集采样，直接用面索引赋标签
            n_sample = max(n_pred * 3, 500000)
            print(f"  GT 为逐面标注 ({n_gt} faces)，采样 {n_sample} 点...")
            sample_pts, face_idx = tm.sample(n_sample, return_index=True)
            sample_pts = np.asarray(sample_pts, dtype=np.float64)
            sample_labels = gt_labels[face_idx]
        elif n_gt == num_verts:
            print(f"  GT 为逐顶点标注 ({n_gt} verts)")
            sample_pts = np.asarray(tm.vertices, dtype=np.float64)
            sample_labels = gt_labels
        else:
            print(f"  GT({n_gt}) != faces({num_faces}) 且 != verts({num_verts})，无法重映射。")
            return None

        # 独立归一化到单位球（与 pipeline 的 normalize_pc 逻辑一致）
        centroid = sample_pts.mean(axis=0)
        centered = sample_pts - centroid
        max_dist = np.max(np.sqrt(np.sum(centered ** 2, axis=1)))
        if max_dist > 0:
            pts_norm = centered / max_dist
        else:
            pts_norm = centered

        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(pts_norm)
        kdtree = o3d.geometry.KDTreeFlann(ref_pcd)

        remapped = np.zeros(n_pred, dtype=np.int64)
        dists = np.zeros(n_pred, dtype=np.float64)
        for i, pt in enumerate(pred_pts):
            _, idx, dist2 = kdtree.search_knn_vector_3d(pt, 1)
            remapped[i] = sample_labels[idx[0]]
            dists[i] = np.sqrt(dist2[0])

        unique_r, counts_r = np.unique(remapped, return_counts=True)
        print(f"  重映射完成: {len(sample_labels)} 参考点 -> {n_pred} 采样点")
        print(f"  重映射后标签分布: {dict(zip(unique_r.tolist(), counts_r.tolist()))}")
        print(f"  匹配距离: mean={dists.mean():.6f}, max={dists.max():.6f}, "
              f">0.01 的点数={int((dists > 0.01).sum())} ({(dists > 0.01).mean()*100:.1f}%)")
        return remapped
    except Exception as e:
        import traceback
        print(f"  重映射失败: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
