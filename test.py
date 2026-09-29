import os

# 确保 CUDA 设备编号与 nvidia-smi 一致（必须在任何 CUDA 调用之前设置）
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
import json
import numpy as np

# 确保能找到 Arbitr3D 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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


# ============================================================
# GPU 自动选择（必须在 pipeline import 前执行）
# ============================================================
# 用法：
#   CUDA_VISIBLE_DEVICES=auto  → 自动选显存最充裕的卡
#   CUDA_VISIBLE_DEVICES=4     → 只能用第 4 张卡
#   不设 CUDA_VISIBLE_DEVICES  → 自动选
def _resolve_gpu_env() -> int:
    """
    解析 GPU 参数，返回物理 GPU 索引（与 nvidia-smi 序号对应）。

    用法：
      CUDA_VISIBLE_DEVICES=2      → 用 nvidia-smi 里编号为 2 的卡
      CUDA_VISIBLE_DEVICES=auto  → 自动寻找显存最充裕的卡
      不设 CUDA_VISIBLE_DEVICES   → 自动寻找
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "auto").strip()
    if raw.lower() in ("", "auto"):
        return -1
    try:
        return int(raw)
    except ValueError:
        return -1


def _auto_best_physical(physical_map: dict) -> int:
    """
    根据 physical_map，从 config 读取阈值，自动选出显存最充裕的物理 GPU 编号。
    """
    import yaml
    threshold = 4.0
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        threshold = float(cfg.get("gpu", {}).get("min_free_gb", 4.0))
    except Exception:
        pass

    if not physical_map:
        print("[test] ❌ 未检测到任何 GPU（nvidia-smi 不可用或无 CUDA 设备），程序终止。")
        raise RuntimeError("No GPU available. Cannot proceed.")

    candidates = {idx: free for idx, (free, _) in physical_map.items() if free >= threshold}
    if not candidates:
            print(f"[test] ❌ 所有 GPU 剩余显存均低于阈值 {threshold:.1f} GB，程序终止。")
            print("[test]    请等待其他任务释放 GPU，或调低 config/config.yaml 中的 gpu.min_free_gb。")
            raise RuntimeError(f"No GPU with >= {threshold:.1f} GB free memory. Aborting.")

    return max(candidates, key=lambda i: candidates[i])


def _get_physical_gpu_map() -> dict:
    """
    调用 nvidia-smi，获取 {物理索引: (free_gb, total_gb)} 的映射。
    物理索引即 nvidia-smi 第一列显示的编号，与 CUDA 逻辑编号独立。
    """
    try:
        import subprocess
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
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


def main():
    # ============================================================
    # GPU 解析：支持物理编号（nvidia-smi 显示的）和 auto
    # ============================================================
    raw_physical = os.environ.get("CUDA_VISIBLE_DEVICES", "auto").strip()
    if raw_physical.lower() in ("", "auto"):
        selected_physical = -1  # auto
        print("[test] 未设置 CUDA_VISIBLE_DEVICES，将自动寻找显存最充裕的 GPU")
    else:
        try:
            selected_physical = int(raw_physical)
        except ValueError:
            selected_physical = -1
        print(f"[test] CUDA_VISIBLE_DEVICES={selected_physical}，将使用物理 GPU {selected_physical}")

    # 打印所有物理 GPU 状态（nvidia-smi 编号）
    physical_map = _get_physical_gpu_map()
    print(f"\n    物理 GPU 状态（nvidia-smi 编号 → 显存剩余 / 总显存）：")
    for idx in sorted(physical_map.keys()):
        free_gb, total_gb = physical_map[idx]
        bar_len = int((total_gb - free_gb) / total_gb * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        used_pct = (total_gb - free_gb) / total_gb * 100
        marker = " ◄── 自动选用" if selected_physical < 0 and idx == _auto_best_physical(physical_map) else ""
        print(f"    GPU {idx}: [{bar}] {used_pct:.0f}% 已用，剩余 {free_gb:.1f} / {total_gb:.1f} GB{marker}")
    print()

    if selected_physical < 0:
        selected_physical = _auto_best_physical(physical_map)
        print(f"[test] 自动选择物理 GPU {selected_physical}，显存剩余 ~{physical_map[selected_physical][0]:.1f} GB\n")

    # 通过 CUDA_DEVICE_ORDER=PCI_BUS_ID（文件顶部已设置），
    # CUDA runtime 编号与 nvidia-smi 物理编号一致，可直接使用
    import torch
    torch.cuda.set_device(selected_physical)
    logical_gpu = selected_physical

    # ==========================================
    # 1. 核心配置区 (留出的接口)
    # ==========================================
    OBJECT_CATEGORY = os.environ.get("TARGET_CATEGORY", "Chair")

    OBJECT_ID = os.environ.get("OBJECT_ID", "179")

    EXP_SUFFIX = os.environ.get("EXP_SUFFIX", "demo")
    # ==========================================

    # 2. 初始化 Pipeline（传入逻辑 GPU 索引，固定为 0）
    config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    pipeline = Arbitr3DPipeline(config_path, gpu_index=logical_gpu)

    # 3. 自动构建路径和 Prompt
    point_cloud_filename = f"{OBJECT_CATEGORY}/{OBJECT_ID}/pc.ply"

    meta_file_path = os.path.join(os.path.dirname(__file__), "PartNetE_meta.json")
    prompt_classes_str = load_prompt_classes_from_meta(meta_file_path, OBJECT_CATEGORY)
    print(f"Loaded prompt classes for {OBJECT_CATEGORY}: {prompt_classes_str}")

    vis_dir = f"visual_res/{OBJECT_ID}_vis_{EXP_SUFFIX}"

    # 4. 运行 Pipeline
    print(f"🚀 Starting Arbitr3D Pipeline for {OBJECT_CATEGORY} - {OBJECT_ID}...")
    pcd, final_labels, class_mapping = pipeline.run(
        point_cloud_filename=point_cloud_filename,
        prompt_classes_str=prompt_classes_str,
        vis_dir=vis_dir
    )

    print(f"✅ 检测到的类别映射 (Class to ID): {class_mapping}")

    # 5. 保存预测的 3D 点云结果
    output_ply_path = os.path.join(vis_dir, f"pred_{OBJECT_CATEGORY.lower()}_{OBJECT_ID}.ply")
    pipeline.visualizer.visualize_3d_result(pcd, final_labels, save_path=output_ply_path, class_to_id=class_mapping)
    print(f"💾 预测结果已保存至: {output_ply_path}")

    # 5b. 保存预测标签数组为 npz（供 scripts/render_qualitative_4x6.py 重新染色使用）
    pred_npz_path = os.path.join(vis_dir, f"pred_{OBJECT_CATEGORY.lower()}_{OBJECT_ID}_sem.npz")
    np.savez(
        pred_npz_path,
        sem_label=np.asarray(final_labels, dtype=np.int64),
        class_mapping=np.array(list(class_mapping.items()), dtype=object),
    )
    print(f"💾 预测标签 npz 已保存至: {pred_npz_path}")

    # 6. 定量评估 (计算 IoU) 并生成 GT 可视化
    base_path = pipeline.config['data']['base_path']
    gt_label_path = os.path.join(base_path, f"{OBJECT_CATEGORY}/{OBJECT_ID}/label.npy")

    # --- 新增：保存真实的 GT 点云 ---
    if os.path.exists(gt_label_path):
        try:
            gt_data = np.load(gt_label_path, allow_pickle=True).item()
            gt_labels = gt_data.get('semantic_seg', None)
            if gt_labels is not None:
                gt_ply_path = os.path.join(vis_dir, f"gt_{OBJECT_CATEGORY.lower()}_{OBJECT_ID}.ply")
                # 使用同样的 visualizer 渲染真实标签，确保颜色一致
                pipeline.visualizer.visualize_3d_result(pcd, gt_labels, save_path=gt_ply_path, class_to_id=class_mapping)
                print(f"💾 真实标签(GT) 可视化已保存至: {gt_ply_path}")

                # 同时在 pipeline 的 step4_final 目录下保存一份 gt_result.ply，
                # 与 final_result.ply 并列，方便直接对照预测与真值。
                step4_gt_path = os.path.join(vis_dir, "step4_final", "gt_result.ply")
                pipeline.visualizer.visualize_3d_result(pcd, gt_labels, save_path=step4_gt_path, class_to_id=class_mapping)
                print(f"💾 真实标签(GT) 副本已保存至: {step4_gt_path}")

                # 保存 GT 标签为 npz（供 4x6 图渲染脚本使用）
                gt_npz_path = os.path.join(vis_dir, f"gt_{OBJECT_CATEGORY.lower()}_{OBJECT_ID}_sem.npz")
                np.savez(
                    gt_npz_path,
                    sem_label=np.asarray(gt_labels, dtype=np.int64),
                    class_mapping=np.array(list(class_mapping.items()), dtype=object),
                )
                print(f"💾 GT 标签 npz 已保存至: {gt_npz_path}")
            else:
                print("⚠️ label.npy 中缺少 'semantic_seg' 键，跳过 GT 可视化")
        except Exception as e:
            print(f"无法生成 GT 可视化: {e}")
    else:
        print(f"⚠️ 未找到 GT 文件: {gt_label_path}（跳过 GT 保存与 IoU 评估）")
    # ----------------------------------

    evaluator = IoUEvaluator()
    ious, miou = evaluator.evaluate(
        pred_labels=final_labels,
        gt_path=gt_label_path,
        class_mapping=class_mapping
    )

if __name__ == "__main__":
    main()
