import os

# 确保 CUDA 设备编号与 nvidia-smi 一致（必须在任何 CUDA 调用之前设置）
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
import json
import numpy as np
import datetime
from collections import defaultdict

# 确保能找到 Arbitr3D 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# GPU 自动选择（必须在 pipeline import 前执行）
# ============================================================
def _get_physical_gpu_map() -> dict:
    """调用 nvidia-smi，获取 {物理索引: (free_gb, total_gb)} 的映射。"""
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
    """从 physical_map 中，按 config.yaml 阈值选出显存最充裕的物理 GPU。"""
    import yaml
    threshold = 4.0
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        threshold = float(cfg.get("gpu", {}).get("min_free_gb", 4.0))
    except Exception:
        pass

    if not physical_map:
        print("[batch_test] ❌ 未检测到任何 GPU（nvidia-smi 不可用或无 CUDA 设备），程序终止。")
        raise RuntimeError("No GPU available. Cannot proceed.")

    candidates = {idx: free for idx, (free, _) in physical_map.items() if free >= threshold}
    if not candidates:
            print(f"[batch_test] ❌ 所有 GPU 剩余显存均低于阈值 {threshold:.1f} GB，程序终止。")
            print("[batch_test]    请等待其他任务释放 GPU，或调低 config/config.yaml 中的 gpu.min_free_gb。")
            raise RuntimeError(f"No GPU with >= {threshold:.1f} GB free memory. Aborting.")

    return max(candidates, key=lambda i: candidates[i])


# 解析 CUDA_VISIBLE_DEVICES，支持 auto/空/物理编号
raw_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "auto").strip()
externally_set = raw_cvd.lower() not in ("", "auto")

physical_map = _get_physical_gpu_map()
print(f"\n    物理 GPU 状态（nvidia-smi 编号 → 显存剩余 / 总显存）：")

if externally_set:
    # 外部（如 run_ablation.py）已通过 CUDA_VISIBLE_DEVICES 指定了 GPU
    # PyTorch 只能看到这些卡，逻辑编号从 0 开始
    visible_ids = [int(x.strip()) for x in raw_cvd.split(",") if x.strip().isdigit()]
    print(f"[batch_test] CUDA_VISIBLE_DEVICES={raw_cvd}，物理 GPU {visible_ids} → 逻辑设备 0")
    logical_gpu = 0
else:
    print("[batch_test] 未设置 CUDA_VISIBLE_DEVICES，将自动寻找显存最充裕的 GPU")

for idx in sorted(physical_map.keys()):
    free_gb, total_gb = physical_map[idx]
    bar_len = int((total_gb - free_gb) / total_gb * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    used_pct = (total_gb - free_gb) / total_gb * 100
    if not externally_set:
        marker = " ◄── 将自动选用" if idx == _auto_best_physical(physical_map) else ""
    else:
        marker = " ◄── 已指定" if idx in visible_ids else ""
    print(f"    GPU {idx}: [{bar}] {used_pct:.0f}% 已用，剩余 {free_gb:.1f} / {total_gb:.1f} GB{marker}")
print()

if not externally_set:
    selected_physical = _auto_best_physical(physical_map)
    print(f"[batch_test] 自动选择物理 GPU {selected_physical}，显存剩余 ~{physical_map[selected_physical][0]:.1f} GB\n")
    # auto 模式：未设 CUDA_VISIBLE_DEVICES，物理编号 = 逻辑编号
    logical_gpu = selected_physical

# ============================================================
import torch
torch.cuda.set_device(logical_gpu)
# ============================================================

from pipeline import Arbitr3DPipeline
from evaluation.metrics import IoUEvaluator

def load_prompt_classes_from_meta(meta_path: str, category: str) -> str:
    """
    从 PartNetE_meta.json 中读取指定类别的部件列表，并转换为逗号分隔的字符串
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    if category not in meta_data:
        raise ValueError(f"Category '{category}' not found in {meta_path}")

    parts = meta_data[category]
    return ", ".join(parts)

def main():
    # ==========================================
    # 1. 核心配置区
    # ==========================================
    # 测试集的基础路径
    import yaml
    config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as config_file:
        batch_config = yaml.safe_load(config_file)
    DATASET_BASE_PATH = batch_config["data"]["base_path"]

    # 当前要批量测试的类别
    CATEGORY = os.environ.get("TARGET_CATEGORY", "Display")  # 可以通过环境变量指定，默认为 "Keyboard"

    # 实验后缀，用于区分输出文件夹
    EXP_SUFFIX = os.environ.get("EXP_SUFFIX", "eval_v5")

    # 日志文件保存目录
    LOG_DIR = "logs"

    # 如果代码在处理第 15 个实例（索引为 14）时中断，你想从它开始重新跑，这里填 14
    START_INDEX = int(os.environ.get("WORKER_START_INDEX", "0"))
    # 分片大小（多 worker 并行时每个 worker 处理的实例数），0 = 全部
    CHUNK_SIZE = int(os.environ.get("WORKER_CHUNK_SIZE", "0"))
    # ==========================================

    category_dir = os.path.join(DATASET_BASE_PATH, CATEGORY)
    if not os.path.exists(category_dir):
        print(f"❌ 找不到类别目录: {category_dir}")
        return

    # 准备日志文件
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"batch_test_{CATEGORY}_{EXP_SUFFIX}_{timestamp}.txt")

    # 获取该类别下所有的实例 ID (即 number 文件夹)
    object_ids = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
    # 排序以保证执行顺序一致
    object_ids.sort()

    start_msg = f"🔍 找到 {len(object_ids)} 个 {CATEGORY} 实例准备测试。\n"
    print(start_msg)

    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write(f"==========================================\n")
        log_file.write(f"批量测试开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"测试类别: {CATEGORY}\n")
        log_file.write(f"数据集路径: {DATASET_BASE_PATH}\n")
        log_file.write(f"实例总数: {len(object_ids)}\n")
        log_file.write(f"==========================================\n\n")

    # 2. 初始化 Pipeline 和评估器
    # 注意：传入 logical_gpu（已补偿偏移），避免 Pipeline 内部重复选卡
    config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    pipeline = Arbitr3DPipeline(config_path, gpu_index=logical_gpu)
    evaluator = IoUEvaluator()

    meta_file_path = os.path.join(os.path.dirname(__file__), "PartNetE_meta.json")
    prompt_classes_str = load_prompt_classes_from_meta(meta_file_path, CATEGORY)
    print(f"📖 加载 {CATEGORY} 的 Prompt 类别: {prompt_classes_str}")

    # 3. 结果统计容器
    all_mious = []
    class_iou_sum = defaultdict(float)
    class_valid_count = defaultdict(int)
    skipped_no_valid_gt = 0

    # 4. 开始批量遍历
    END_INDEX = START_INDEX + CHUNK_SIZE if CHUNK_SIZE > 0 else len(object_ids)
    print(f"📋 本 Worker 处理范围: [{START_INDEX}, {END_INDEX}) / {len(object_ids)} 个实例")
    for idx, obj_id in enumerate(object_ids):
        if idx < START_INDEX:
            continue
        if idx >= END_INDEX:
            break

        print("\n" + "="*60)
        print(f"🚀 正在处理实例: {obj_id} ({idx + 1}/{len(object_ids)})")
        print("="*60)

        # 构建相对路径传给 pipeline
        pc_rel_path = f"{CATEGORY}/{obj_id}/pc.ply"
        gt_label_path = os.path.join(DATASET_BASE_PATH, CATEGORY, obj_id, "label.npy")

        # 为每个实例创建独立的可视化输出目录
        vis_dir = f"visual_res/batch_{CATEGORY}_{EXP_SUFFIX}/{obj_id}"

        if not os.path.exists(gt_label_path):
            skip_msg = f"⚠️ 跳过 {obj_id}: 找不到真实的 label.npy 文件\n"
            print(skip_msg.strip())
            with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write(f"[{obj_id}] {skip_msg}")
            continue

        try:
            # 运行 Pipeline
            pcd, final_labels, class_mapping = pipeline.run(
                point_cloud_filename=pc_rel_path,
                prompt_classes_str=prompt_classes_str,
                vis_dir=vis_dir
            )

            # 运行评估
            ious, miou = evaluator.evaluate(
                pred_labels=final_labels,
                gt_path=gt_label_path,
                class_mapping=class_mapping
            )

            # --- 记录单个实例的日志 ---
            with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write(f"--- 实例: {obj_id} ({idx + 1}/{len(object_ids)}) ---\n")
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
                print("  >> mIoU: N/A (GT中无任何有效标注，不参与类别整体mIoU统计)")
                skipped_no_valid_gt += 1
                continue

            all_mious.append(miou)
            for cls_name, iou_val in ious.items():
                if not np.isnan(iou_val):
                    class_iou_sum[cls_name] += iou_val
                    class_valid_count[cls_name] += 1

        except Exception as e:
            err_msg = f"❌ 处理实例 {obj_id} 时发生错误: {e}\n"
            print(err_msg.strip())
            with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write(f"[{obj_id}] {err_msg}")
            continue

    # 5. 打印并记录最终汇总统计结果
    summary_lines = []
    summary_lines.append("\n" + "🌟"*30)
    summary_lines.append(f"🏆 {CATEGORY} 类别批量测试完成！共纳入总体 mIoU 统计 {len(all_mious)} 个实例，排除 GT 无有效标注实例 {skipped_no_valid_gt} 个。")
    summary_lines.append("🌟"*30)

    if len(all_mious) > 0:
        summary_lines.append("\n📊 各部件平均 IoU (仅统计包含该部件的实例):")
        for cls_name in class_iou_sum.keys():
            if class_valid_count[cls_name] > 0:
                avg_iou = class_iou_sum[cls_name] / class_valid_count[cls_name]
                summary_lines.append(f"  - {cls_name.ljust(15)}: {avg_iou:.4f} (有效实例数: {class_valid_count[cls_name]})")

        final_avg_miou = np.mean(all_mious)
        summary_lines.append("-" * 60)
        summary_lines.append(f"🎯 总体平均 mIoU: {final_avg_miou:.4f}")
    else:
        summary_lines.append("⚠️ 没有任何实例被纳入总体 mIoU 统计。")
    summary_lines.append("🌟"*30 + "\n")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(log_filename, 'a', encoding='utf-8') as log_file:
        log_file.write("\n================ FINAL SUMMARY ================\n")
        log_file.write(summary_text)

    print(f"📄 测试日志已保存至: {log_filename}")

if __name__ == "__main__":
    main()
