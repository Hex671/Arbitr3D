"""
PartObjaverse-Tiny 数据集批量评测脚本

用法:
    # 评测所有类别
    python batch_test_objaverse.py

    # 仅评测指定类别
    python batch_test_objaverse.py --categories "Food" "Animals"

    # 指定 GPU 和采样点数
    python batch_test_objaverse.py --gpu 0 --num-points 30000

    # 仅评测前 5 个实例（跨所有目标类别累计）后停止
    python batch_test_objaverse.py --max-instances 5

    # 环境变量控制（兼容 run_ablation 风格）
    TARGET_CATEGORY=Food python batch_test_objaverse.py
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
import json
import argparse
import datetime
import numpy as np
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# GPU 选择逻辑（复用 batch_test.py 的策略）
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
        visible_ids = [int(x.strip()) for x in raw_cvd.split(",") if x.strip().isdigit()]
        print(f"[GPU] CUDA_VISIBLE_DEVICES={raw_cvd} → 逻辑设备 0")
        return 0

    if gpu_id >= 0:
        print(f"[GPU] 使用命令行指定 GPU {gpu_id}")
        return gpu_id

    selected = _auto_best_physical(physical_map, threshold)
    print(f"[GPU] 自动选择 GPU {selected}（剩余 ~{physical_map[selected][0]:.1f} GB）\n")
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="PartObjaverse-Tiny 批量评测")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="要评测的类别名（默认全部）；也可通过 TARGET_CATEGORY 环境变量指定单个类别")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU 编号，-1=自动")
    parser.add_argument("--num-points", type=int, default=None,
                        help="网格采样点数（覆盖 config 中的 partobjaverse_tiny.num_points）")
    parser.add_argument("--exp-suffix", default=None, help="实验后缀")
    parser.add_argument("--start-index", type=int, default=0, help="起始实例索引")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="分片大小（每类别），0=全部。仅限制单类别内取多少实例，不跨类别累计。")
    parser.add_argument("--max-instances", type=int, default=0,
                        help="全局实例上限：累计评测到第 N 个实例后停止（跨所有类别）。"
                             "0=不限制（默认）。与 --chunk-size 同时生效时取更严格者。"
                             "计数包含 SKIP 的实例（与控制台 [{n}/{total}] 一致），"
                             "用于快速冒烟测试或调试。")
    return parser.parse_args()


def main():
    args = parse_args()

    # GPU 设置
    logical_gpu = setup_gpu(args.gpu)
    import torch
    torch.cuda.set_device(logical_gpu)

    import yaml
    from pipeline import Arbitr3DPipeline
    from evaluation.metrics import IoUEvaluator
    from data_prep.dataloader import PointCloudDataset

    # ============================================================
    # 1. 加载配置
    # ============================================================
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pot_cfg = config.get("partobjaverse_tiny", {})
    BASE_PATH = pot_cfg.get("base_path",
        "datasets/PartObjaverse-Tiny/PartObjaverse-Tiny")
    MESH_SUBDIR = pot_cfg.get("mesh_subdir", "PartObjaverse-Tiny_mesh")
    GT_SUBDIR = pot_cfg.get("gt_subdir", "PartObjaverse-Tiny_semantic_gt")
    META_JSON_REL = pot_cfg.get("meta_json", "PartObjaverse-Tiny_semantic.json")
    NUM_POINTS = args.num_points if args.num_points is not None else int(pot_cfg.get("num_points", 30000))

    EXP_SUFFIX = args.exp_suffix or os.environ.get("EXP_SUFFIX", "objaverse_eval")
    START_INDEX = args.start_index or int(os.environ.get("WORKER_START_INDEX", "0"))
    CHUNK_SIZE = args.chunk_size or int(os.environ.get("WORKER_CHUNK_SIZE", "0"))
    MAX_INSTANCES = max(0, int(args.max_instances))  # 0 = unlimited
    LOG_DIR = "logs"

    # ============================================================
    # 2. 加载 Meta JSON（每实例独立部件列表）
    # ============================================================
    project_root = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(project_root, META_JSON_REL)
    if not os.path.exists(meta_path):
        meta_path = os.path.join(BASE_PATH, META_JSON_REL)
    if not os.path.exists(meta_path):
        print(f"找不到 Meta JSON: {meta_path}")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # 确定评测类别
    env_cat = os.environ.get("TARGET_CATEGORY", "").strip()
    if args.categories:
        target_categories = args.categories
    elif env_cat:
        target_categories = [env_cat]
    else:
        target_categories = list(meta_data.keys())

    print(f"评测类别: {target_categories}")
    print(f"数据集根目录: {BASE_PATH}")
    print(f"采样点数: {NUM_POINTS} (0=使用原始顶点)")

    # ============================================================
    # 3. 初始化 Pipeline（base_path 指向 PartObjaverse-Tiny 根目录）
    # ============================================================
    config["data"]["base_path"] = BASE_PATH
    config["data"]["dataset_name"] = "PartObjaverse-Tiny"

    tmp_config_path = os.path.join(project_root, "config", "_tmp_objaverse_config.yaml")
    with open(tmp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    pipeline = Arbitr3DPipeline(tmp_config_path, gpu_index=logical_gpu)
    evaluator = IoUEvaluator()

    # ============================================================
    # 4. 准备日志
    # ============================================================
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"batch_objaverse_{EXP_SUFFIX}_{timestamp}.txt")

    with open(log_filename, "w", encoding="utf-8") as lf:
        lf.write(f"{'='*60}\n")
        lf.write(f"PartObjaverse-Tiny 批量评测\n")
        lf.write(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"类别: {target_categories}\n")
        lf.write(f"数据路径: {BASE_PATH}\n")
        lf.write(f"采样点数: {NUM_POINTS}\n")
        lf.write(f"{'='*60}\n\n")

    # ============================================================
    # 5. 评测主循环
    # ============================================================
    # 全局统计
    global_all_mious = []
    global_class_iou_sum = defaultdict(float)
    global_class_valid_count = defaultdict(int)

    # 按类别统计
    category_results = {}

    total_instances = sum(len(meta_data.get(cat, {})) for cat in target_categories)
    # 显示进度时使用 "虚拟总数" -- 若有 --max-instances 限制，分母用上限值，更直观。
    display_total = (
        min(MAX_INSTANCES, total_instances) if MAX_INSTANCES > 0 else total_instances
    )
    processed_count = 0
    stop_eval = False  # --max-instances 触发后置 True，跳出外层 category 循环

    for category in target_categories:
        if stop_eval:
            break
        if category not in meta_data:
            print(f"\n类别 '{category}' 在 Meta JSON 中不存在，跳过。")
            continue

        instances = meta_data[category]
        obj_ids = sorted(instances.keys())

        print(f"\n{'='*60}")
        print(f"类别: {category} — 共 {len(obj_ids)} 个实例")
        print(f"{'='*60}")

        cat_mious = []
        cat_class_iou_sum = defaultdict(float)
        cat_class_valid_count = defaultdict(int)
        cat_skipped = 0

        end_idx = START_INDEX + CHUNK_SIZE if CHUNK_SIZE > 0 else len(obj_ids)

        for idx, obj_id in enumerate(obj_ids):
            if idx < START_INDEX:
                continue
            if idx >= end_idx:
                break

            # 全局上限检查（在递增 processed_count 之前，确保最多处理 N 个实例）
            if MAX_INSTANCES > 0 and processed_count >= MAX_INSTANCES:
                stop_msg = (f"[stop] 已达 --max-instances={MAX_INSTANCES}，"
                            f"提前结束评测。")
                print(f"\n{stop_msg}")
                with open(log_filename, "a", encoding="utf-8") as lf:
                    lf.write(f"\n{stop_msg}\n")
                stop_eval = True
                break

            processed_count += 1
            print(f"\n--- [{processed_count}/{display_total}] {category}/{obj_id} ---")

            # 路径构建
            mesh_path = os.path.join(MESH_SUBDIR, f"{obj_id}.glb")
            mesh_full_path = os.path.join(BASE_PATH, mesh_path)
            gt_full_path = os.path.join(BASE_PATH, GT_SUBDIR, f"{obj_id}.npy")

            if not os.path.exists(mesh_full_path):
                msg = f"跳过: 网格文件不存在 {mesh_full_path}"
                print(f"  {msg}")
                with open(log_filename, "a", encoding="utf-8") as lf:
                    lf.write(f"[{category}/{obj_id}] {msg}\n")
                continue

            if not os.path.exists(gt_full_path):
                msg = f"跳过: GT 标签不存在 {gt_full_path}"
                print(f"  {msg}")
                with open(log_filename, "a", encoding="utf-8") as lf:
                    lf.write(f"[{category}/{obj_id}] {msg}\n")
                continue

            # 获取该实例的部件列表
            part_names = instances[obj_id]
            prompt_classes_str = ", ".join(part_names)
            print(f"  部件: {prompt_classes_str}")

            vis_dir = f"visual_res/batch_objaverse_{EXP_SUFFIX}/{category}/{obj_id}"

            try:
                # 运行 Pipeline
                pcd, final_labels, class_mapping = pipeline.run(
                    point_cloud_filename=mesh_path,
                    prompt_classes_str=prompt_classes_str,
                    vis_dir=vis_dir,
                    object_category=category,
                    num_points=NUM_POINTS,
                )

                # 加载 GT 标签
                gt_labels = IoUEvaluator.load_gt_labels(gt_full_path)
                if gt_labels is None:
                    msg = f"GT 标签加载失败: {gt_full_path}"
                    print(f"  {msg}")
                    with open(log_filename, "a", encoding="utf-8") as lf:
                        lf.write(f"[{category}/{obj_id}] {msg}\n")
                    continue

                # 点数对齐检查
                n_pred = len(final_labels)
                n_gt = len(gt_labels)
                if n_pred != n_gt:
                    print(f"  [Warning] 预测点数({n_pred}) != GT点数({n_gt})")
                    if NUM_POINTS > 0:
                        print(f"  [Info] 使用采样模式(num_points={NUM_POINTS})，尝试通过最近邻重映射GT标签...")
                        gt_labels = _remap_gt_to_pred(
                            gt_full_path, mesh_full_path, pcd, gt_labels
                        )
                        if gt_labels is None or len(gt_labels) != n_pred:
                            msg = f"GT 标签重映射失败，跳过"
                            print(f"  {msg}")
                            with open(log_filename, "a", encoding="utf-8") as lf:
                                lf.write(f"[{category}/{obj_id}] {msg}\n")
                            continue
                    else:
                        msg = f"点数不一致且无法对齐，跳过"
                        print(f"  {msg}")
                        with open(log_filename, "a", encoding="utf-8") as lf:
                            lf.write(f"[{category}/{obj_id}] {msg}\n")
                        continue

                # 评估
                ious, miou = evaluator.evaluate_with_gt_array(
                    pred_labels=final_labels,
                    gt_labels=gt_labels,
                    class_mapping=class_mapping,
                )

                # 记录日志
                with open(log_filename, "a", encoding="utf-8") as lf:
                    lf.write(f"--- {category}/{obj_id} ---\n")
                    lf.write(f"  部件: {prompt_classes_str}\n")
                    for cls_name, iou_val in ious.items():
                        if np.isnan(iou_val):
                            lf.write(f"  {cls_name.ljust(20)}: N/A\n")
                        else:
                            lf.write(f"  {cls_name.ljust(20)}: {iou_val:.4f}\n")
                    if np.isnan(miou):
                        lf.write(f"  >> mIoU: N/A\n\n")
                    else:
                        lf.write(f"  >> mIoU: {miou:.4f}\n\n")

                if np.isnan(miou):
                    cat_skipped += 1
                    continue

                cat_mious.append(miou)
                global_all_mious.append(miou)
                for cls_name, iou_val in ious.items():
                    if not np.isnan(iou_val):
                        cat_class_iou_sum[cls_name] += iou_val
                        cat_class_valid_count[cls_name] += 1
                        global_class_iou_sum[cls_name] += iou_val
                        global_class_valid_count[cls_name] += 1

            except Exception as e:
                import traceback
                err_msg = f"处理失败: {e}"
                print(f"  {err_msg}")
                traceback.print_exc()
                with open(log_filename, "a", encoding="utf-8") as lf:
                    lf.write(f"[{category}/{obj_id}] {err_msg}\n")
                continue

        # 类别汇总
        cat_avg_miou = np.mean(cat_mious) if cat_mious else float("nan")
        category_results[category] = {
            "num_instances": len(obj_ids),
            "num_evaluated": len(cat_mious),
            "num_skipped": cat_skipped,
            "avg_miou": cat_avg_miou,
            "per_part_avg_iou": {
                cls: cat_class_iou_sum[cls] / cat_class_valid_count[cls]
                for cls in cat_class_iou_sum
                if cat_class_valid_count[cls] > 0
            },
        }

        cat_summary = f"\n  [{category}] 平均 mIoU: {cat_avg_miou:.4f} ({len(cat_mious)}/{len(obj_ids)} 实例)"
        print(cat_summary)
        with open(log_filename, "a", encoding="utf-8") as lf:
            lf.write(f"\n{'='*40}\n")
            lf.write(f"类别汇总: {category}\n")
            lf.write(cat_summary + "\n")
            for cls, avg in category_results[category]["per_part_avg_iou"].items():
                lf.write(f"  {cls.ljust(20)}: {avg:.4f}\n")
            lf.write(f"{'='*40}\n\n")

    # ============================================================
    # 6. 全局汇总
    # ============================================================
    summary_lines = []
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append(f"PartObjaverse-Tiny 全局评测汇总")
    summary_lines.append(f"{'='*60}")

    summary_lines.append(f"\n各类别 mIoU:")
    for cat, res in category_results.items():
        summary_lines.append(f"  {cat.ljust(25)}: {res['avg_miou']:.4f}  "
                             f"({res['num_evaluated']}/{res['num_instances']} 实例)")

    if category_results:
        valid_cat_mious = [r["avg_miou"] for r in category_results.values() if not np.isnan(r["avg_miou"])]
        if valid_cat_mious:
            overall_cat_avg = np.mean(valid_cat_mious)
            summary_lines.append(f"\n  类别平均 mIoU (Category-level mean): {overall_cat_avg:.4f}  "
                                 f"({len(valid_cat_mious)} 个有效类别)")

    if global_all_mious:
        instance_avg = np.mean(global_all_mious)
        summary_lines.append(f"  实例平均 mIoU (Instance-level mean):  {instance_avg:.4f}  "
                             f"({len(global_all_mious)} 个实例)")
    else:
        summary_lines.append(f"\n  无有效实例参与统计。")

    summary_lines.append(f"\n{'='*60}\n")
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(log_filename, "a", encoding="utf-8") as lf:
        lf.write("\n================ FINAL SUMMARY ================\n")
        lf.write(summary_text)

    # 保存结构化结果 JSON
    result_json_path = log_filename.replace(".txt", "_results.json")
    result_data = {
        "timestamp": timestamp,
        "config": {
            "base_path": BASE_PATH,
            "num_points": NUM_POINTS,
            "exp_suffix": EXP_SUFFIX,
        },
        "categories": {},
        "global": {
            "instance_count": len(global_all_mious),
            "instance_avg_miou": float(np.mean(global_all_mious)) if global_all_mious else None,
        },
    }
    for cat, res in category_results.items():
        result_data["categories"][cat] = {
            "num_instances": res["num_instances"],
            "num_evaluated": res["num_evaluated"],
            "avg_miou": float(res["avg_miou"]) if not np.isnan(res["avg_miou"]) else None,
            "per_part_avg_iou": {k: float(v) for k, v in res["per_part_avg_iou"].items()},
        }
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)

    print(f"日志: {log_filename}")
    print(f"结果 JSON: {result_json_path}")

    # 清理临时配置
    if os.path.exists(tmp_config_path):
        os.remove(tmp_config_path)


def _remap_gt_to_pred(gt_path, mesh_path, pred_pcd, gt_labels):
    """
    用 trimesh 从网格面上密集采样并直接通过面索引赋 GT 标签，
    再与 pipeline 归一化点云做 KDTree 最近邻匹配。
    加载方式与 PartObjaverse-Tiny 官方一致。
    """
    import open3d as o3d
    import trimesh
    try:
        tm = trimesh.load(mesh_path)
        if isinstance(tm, trimesh.Scene):
            tm = tm.dump(concatenate=True)

        num_verts = len(tm.vertices)
        num_faces = len(tm.faces)
        n_gt = len(gt_labels)

        pred_pts = np.asarray(pred_pcd.points)
        n_pred = len(pred_pts)

        if n_gt == num_faces:
            n_sample = max(n_pred * 3, 500000)
            print(f"  [remap] GT 为逐面标注 ({n_gt} faces)，采样 {n_sample} 点...")
            sample_pts, face_idx = tm.sample(n_sample, return_index=True)
            sample_pts = np.asarray(sample_pts, dtype=np.float64)
            sample_labels = gt_labels[face_idx]
        elif n_gt == num_verts:
            print(f"  [remap] GT 为逐顶点标注 ({n_gt} verts)")
            sample_pts = np.asarray(tm.vertices, dtype=np.float64)
            sample_labels = gt_labels
        else:
            print(f"  [remap] GT({n_gt}) != faces({num_faces}) 且 != verts({num_verts})，无法重映射")
            return None

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
        for i, pt in enumerate(pred_pts):
            _, idx, _ = kdtree.search_knn_vector_3d(pt, 1)
            remapped[i] = sample_labels[idx[0]]
        print(f"  [remap] 完成: {len(sample_labels)} 参考点 -> {n_pred} 采样点")
        return remapped
    except Exception as e:
        import traceback
        print(f"  [remap] 失败: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
