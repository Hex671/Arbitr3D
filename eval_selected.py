"""
对指定类别的 test 集前 N 个样本进行测评。

用法:
    python eval_selected.py --category Table -n 5
    python eval_selected.py --category Chair -n 10 --exp_suffix my_ablation
    python eval_selected.py --category Display -n 3 --gpu 0
"""
import os

# 确保 CUDA 设备编号与 nvidia-smi 一致（必须在任何 CUDA 调用之前设置）
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
import json
import argparse
import numpy as np
import datetime
from collections import defaultdict

# 确保能找到 Arbitr3D 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# GPU 自动选择（必须在 pipeline import 前执行）
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


def _auto_best_physical(physical_map: dict) -> int:
    import yaml
    threshold = 4.0
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        threshold = float(cfg.get("gpu", {}).get("min_free_gb", 4.0))
    except Exception:
        pass

    if not physical_map:
        print("[eval_selected] 未检测到任何 GPU，程序终止。")
        raise RuntimeError("No GPU available.")

    candidates = {idx: free for idx, (free, _) in physical_map.items() if free >= threshold}
    if not candidates:
        print(f"[eval_selected] 所有 GPU 剩余显存均低于阈值 {threshold:.1f} GB，程序终止。")
        raise RuntimeError(f"No GPU with >= {threshold:.1f} GB free memory.")

    return max(candidates, key=lambda i: candidates[i])


def parse_args():
    parser = argparse.ArgumentParser(description="对指定类别的若干样例运行 Arbitr3D Pipeline 并评估")
    parser.add_argument("--category", type=str, required=True,
                        help="测试类别名，如 Table, Chair, Display 等")
    parser.add_argument("-n", "--num_samples", type=int, required=True,
                        help="要测试的样本数量（test 集中排序后的前 N 个）")
    parser.add_argument("--exp_suffix", type=str, default="selected",
                        help="实验后缀，用于区分输出文件夹 (默认: selected)")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="指定物理 GPU 编号，-1 表示自动选择 (默认: -1)")
    return parser.parse_args()


def load_prompt_classes_from_meta(meta_path: str, category: str) -> str:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
    if category not in meta_data:
        raise ValueError(f"Category '{category}' not found in {meta_path}")
    parts = meta_data[category]
    return ", ".join(parts)


def main():
    args = parse_args()
    CATEGORY = args.category
    NUM_SAMPLES = args.num_samples
    EXP_SUFFIX = args.exp_suffix
    if NUM_SAMPLES <= 0:
        print("num_samples 必须 > 0")
        return

    # --- GPU 选择 ---
    physical_map = _get_physical_gpu_map()
    if physical_map:
        print(f"\n    物理 GPU 状态：")
        for idx in sorted(physical_map.keys()):
            free_gb, total_gb = physical_map[idx]
            bar_len = int((total_gb - free_gb) / total_gb * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            used_pct = (total_gb - free_gb) / total_gb * 100
            print(f"    GPU {idx}: [{bar}] {used_pct:.0f}% 已用，剩余 {free_gb:.1f} / {total_gb:.1f} GB")
        print()

    if args.gpu >= 0:
        selected_physical = args.gpu
        print(f"[eval_selected] 使用指定 GPU {selected_physical}")
    else:
        selected_physical = _auto_best_physical(physical_map)
        print(f"[eval_selected] 自动选择 GPU {selected_physical}，剩余 ~{physical_map[selected_physical][0]:.1f} GB")

    import torch
    torch.cuda.set_device(selected_physical)
    logical_gpu = selected_physical

    from pipeline import Arbitr3DPipeline
    from evaluation.metrics import IoUEvaluator

    # --- 路径配置 ---
    import yaml
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    DATASET_BASE_PATH = cfg.get("data", {}).get("base_path", "datasets/PartNetE/test")

    category_dir = os.path.join(DATASET_BASE_PATH, CATEGORY)
    if not os.path.exists(category_dir):
        print(f"找不到类别目录: {category_dir}")
        return

    # 列出 test 集下所有样例并排序，取前 N 个
    all_ids = sorted([d for d in os.listdir(category_dir)
                      if os.path.isdir(os.path.join(category_dir, d))])
    if not all_ids:
        print(f"类别 {CATEGORY} 下没有任何样例。")
        return
    if NUM_SAMPLES > len(all_ids):
        print(f"[Warning] 请求 {NUM_SAMPLES} 个样本，但该类别仅有 {len(all_ids)} 个，已截断")
    valid_ids = all_ids[:NUM_SAMPLES]

    # --- 日志 ---
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"eval_{CATEGORY}_{EXP_SUFFIX}_{timestamp}.txt")

    print(f"\n测试类别: {CATEGORY}")
    print(f"测试样例 ({len(valid_ids)}): {valid_ids}")
    print(f"实验后缀: {EXP_SUFFIX}")
    print(f"日志文件: {log_filename}\n")

    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write(f"==========================================\n")
        log_file.write(f"测评开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"测试类别: {CATEGORY}\n")
        log_file.write(f"测试样例: {valid_ids}\n")
        log_file.write(f"数据集路径: {DATASET_BASE_PATH}\n")
        log_file.write(f"==========================================\n\n")

    # --- 初始化 Pipeline ---
    config_path = "config/config.yaml"
    pipeline = Arbitr3DPipeline(config_path, gpu_index=logical_gpu)
    evaluator = IoUEvaluator()

    meta_file_path = os.path.join(os.path.dirname(__file__), "PartNetE_meta.json")
    prompt_classes_str = load_prompt_classes_from_meta(meta_file_path, CATEGORY)
    print(f"Prompt 类别: {prompt_classes_str}\n")

    # --- 结果统计 ---
    all_mious = []
    class_iou_sum = defaultdict(float)
    class_valid_count = defaultdict(int)
    skipped_no_valid_gt = 0

    # --- 逐样例测评 ---
    for idx, obj_id in enumerate(valid_ids):
        print("\n" + "=" * 60)
        print(f"正在处理: {obj_id} ({idx + 1}/{len(valid_ids)})")
        print("=" * 60)

        pc_rel_path = f"{CATEGORY}/{obj_id}/pc.ply"
        gt_label_path = os.path.join(DATASET_BASE_PATH, CATEGORY, obj_id, "label.npy")
        vis_dir = f"visual_res/{CATEGORY}_{EXP_SUFFIX}/{obj_id}"

        if not os.path.exists(gt_label_path):
            skip_msg = f"跳过 {obj_id}: 找不到 label.npy"
            print(skip_msg)
            with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write(f"[{obj_id}] {skip_msg}\n")
            continue

        try:
            pcd, final_labels, class_mapping = pipeline.run(
                point_cloud_filename=pc_rel_path,
                prompt_classes_str=prompt_classes_str,
                vis_dir=vis_dir,
            )

            ious, miou = evaluator.evaluate(
                pred_labels=final_labels,
                gt_path=gt_label_path,
                class_mapping=class_mapping,
            )

            with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write(f"--- 实例: {obj_id} ({idx + 1}/{len(valid_ids)}) ---\n")
                for cls_name, iou_val in ious.items():
                    if np.isnan(iou_val):
                        log_file.write(f"  {cls_name.ljust(15)}: N/A (GT中不存在)\n")
                    else:
                        log_file.write(f"  {cls_name.ljust(15)}: {iou_val:.4f}\n")
                if np.isnan(miou):
                    log_file.write("  >> mIoU: N/A (GT中无任何有效标注，不参与类别整体mIoU统计)\n\n")
                else:
                    log_file.write(f"  >> mIoU: {miou:.4f}\n\n")

            if np.isnan(miou):
                print("  mIoU = N/A (GT中无任何有效标注，不参与类别整体mIoU统计)")
                skipped_no_valid_gt += 1
                continue

            print(f"  mIoU = {miou:.4f}")

            all_mious.append(miou)
            for cls_name, iou_val in ious.items():
                if not np.isnan(iou_val):
                    class_iou_sum[cls_name] += iou_val
                    class_valid_count[cls_name] += 1

        except Exception as e:
            err_msg = f"处理 {obj_id} 时发生错误: {e}"
            print(err_msg)
            with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write(f"[{obj_id}] {err_msg}\n")
            continue

    # --- 汇总 ---
    summary_lines = []
    summary_lines.append("\n" + "=" * 60)
    summary_lines.append(f"{CATEGORY} 选定样例测评完成，共纳入总体 mIoU 统计 {len(all_mious)}/{len(valid_ids)} 个实例，排除 GT 无有效标注实例 {skipped_no_valid_gt} 个")
    summary_lines.append("=" * 60)

    if all_mious:
        summary_lines.append("\n各部件平均 IoU:")
        for cls_name in sorted(class_iou_sum.keys()):
            if class_valid_count[cls_name] > 0:
                avg_iou = class_iou_sum[cls_name] / class_valid_count[cls_name]
                summary_lines.append(f"  {cls_name.ljust(15)}: {avg_iou:.4f} ({class_valid_count[cls_name]} 个有效实例)")

        final_avg_miou = np.mean(all_mious)
        summary_lines.append("-" * 60)
        summary_lines.append(f"总体平均 mIoU: {final_avg_miou:.4f}")
    else:
        summary_lines.append("没有任何实例被纳入总体 mIoU 统计。")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(log_filename, 'a', encoding='utf-8') as log_file:
        log_file.write("\n================ FINAL SUMMARY ================\n")
        log_file.write(summary_text + "\n")

    print(f"\n日志已保存至: {log_filename}")


if __name__ == "__main__":
    main()
